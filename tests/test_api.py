from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.agent import KnowledgeAgent
from app.rag_engine import KnowledgeBase

SEED_DOC = (
    "# 检索服务架构\n\n"
    "## RAG 证据链\n\n"
    "RAG 从仓库文档和运行手册检索相关证据，再让模型基于证据回答。"
    "系统通过稠密向量召回、重排、证据验证和引用追踪降低幻觉。\n"
)


@pytest.fixture()
def api(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("AI_AGENT_VECTOR_BACKEND", "memory")
    monkeypatch.setenv("AI_AGENT_EMBEDDING_PROVIDER", "test")
    monkeypatch.setenv("AI_AGENT_ALLOW_TEST_EMBEDDINGS", "1")
    monkeypatch.setenv("AI_AGENT_RERANKER", "heuristic")

    (tmp_path / "00-seed.md").write_text(SEED_DOC, encoding="utf-8")
    kb = KnowledgeBase(tmp_path)
    agent = KnowledgeAgent(kb)

    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main_mod, "kb", kb)
    monkeypatch.setattr(main_mod, "agent", agent)
    # TestClient without the context manager skips the lifespan background
    # thread; kb/agent are injected directly above for determinism.
    return TestClient(main_mod.app)


def test_health_reports_ready(api: TestClient) -> None:
    response = api.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["kb"]["documents"] == 1
    assert payload["kb"]["chunks"] >= 1


def test_health_initializing_when_kb_missing(api: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "kb", None)
    payload = api.get("/api/health").json()
    assert payload["status"] == "initializing"
    assert payload["kb"] is None


def test_kb_stats_documents_workflow(api: TestClient) -> None:
    assert api.get("/api/kb/stats").status_code == 200
    docs = api.get("/api/kb/documents").json()
    assert docs and docs[0]["title"] == "检索服务架构"
    workflow = api.get("/api/workflow").json()
    assert workflow["framework"] == "langgraph"
    assert workflow["checkpointing"] == "langgraph_in_memory"
    assert workflow["entrypoint"] == "route"
    assert "verify_evidence" in workflow["nodes"]
    assert "recall_memory" in workflow["nodes"]
    assert "reflect" in workflow["nodes"]
    assert api.get("/api/skills").json()
    assert api.get("/api/tools").json()


def test_training_dashboard_endpoint(api: TestClient, tmp_path: Path, monkeypatch) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "evaluation-results.json").write_text(
        json.dumps({"summary": {"cases": 1, "passed": 1, "pass_rate": 1.0}}),
        encoding="utf-8",
    )
    (reports / "posttraining-trajectories.jsonl").write_text(
        json.dumps({"instance_id": "run-1", "trajectory": {}, "reward": {"total": 1.0}}) + "\n",
        encoding="utf-8",
    )
    (reports / "posttraining-sft.jsonl").write_text("{}\n", encoding="utf-8")
    (reports / "posttraining-preferences.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(main_mod, "REPORTS_DIR", reports)

    response = api.get("/api/training/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "data_ready"
    assert payload["boundary"]["actual_training_run"] is False


def test_ask_returns_grounded_response(api: TestClient) -> None:
    response = api.post("/api/ask", json={"query": "什么是 RAG 检索增强生成？", "top_k": 6})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["intent"]
    assert payload["metrics"]["retrieval_attempts"] >= 1
    assert payload["metrics"]["graph_path"]


def test_ask_stream_emits_live_graph_events_and_final_response(api: TestClient) -> None:
    response = api.post(
        "/api/ask/stream",
        json={"query": "什么是 RAG 检索增强生成？", "top_k": 6},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: graph_event" in response.text
    assert '"node": "route"' in response.text
    assert '"node": "reflect"' in response.text
    assert "event: final" in response.text


def test_ask_validates_query_length(api: TestClient) -> None:
    assert api.post("/api/ask", json={"query": "", "top_k": 6}).status_code == 422


def test_upload_markdown_is_idempotent(api: TestClient) -> None:
    content = base64.b64encode(
        "# 变更记录\n\n## 影响\n\n这是用于上传测试的文档内容，包含足够长的正文段落。\n".encode()
    ).decode()
    payload = {"filename": "change-log.md", "content_base64": content}

    first = api.post("/api/kb/upload", json=payload)
    assert first.status_code == 200
    assert first.json()["created"] is True

    second = api.post("/api/kb/upload", json=payload)
    assert second.status_code == 200
    assert second.json()["idempotent"] is True
    assert api.get("/api/kb/stats").json()["documents"] == 2


def test_upload_rejects_oversized_payload(api: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "MAX_UPLOAD_BYTES", 100)
    content = base64.b64encode(b"x" * 200).decode()
    response = api.post(
        "/api/kb/upload", json={"filename": "big.md", "content_base64": content}
    )
    assert response.status_code == 400
    assert "限制" in response.json()["detail"]


def test_upload_rejects_unsupported_extension(api: TestClient) -> None:
    content = base64.b64encode(b"not an image").decode()
    response = api.post(
        "/api/kb/upload", json={"filename": "photo.png", "content_base64": content}
    )
    assert response.status_code == 400
    assert "暂只支持" in response.json()["detail"]


def test_upload_rejects_corrupt_pdf(api: TestClient) -> None:
    content = base64.b64encode(b"%PDF-1.4 this is not a real pdf").decode()
    response = api.post(
        "/api/kb/upload", json={"filename": "broken.pdf", "content_base64": content}
    )
    assert response.status_code == 400
    assert "PDF 解析失败" in response.json()["detail"]


def test_upload_parses_docx(api: TestClient) -> None:
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>初始化失败排查手册</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>模型下载超时时应检查网络、缓存与代理配置，并保留原始错误信息用于诊断。</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    content = base64.b64encode(buffer.getvalue()).decode()

    response = api.post(
        "/api/kb/upload",
        json={"filename": "runbook.docx", "content_base64": content},
    )
    assert response.status_code == 200
    assert response.json()["created"] is True
    assert response.json()["document"].startswith("runbook-")


def test_add_and_delete_document(api: TestClient) -> None:
    created = api.post(
        "/api/kb/documents",
        json={"title": "新文档", "content": "这是一段用于直接添加接口测试的文档内容，长度满足最小要求。"},
    )
    assert created.status_code == 200
    doc_id = created.json()["document"].removesuffix(".md")
    assert created.json()["created"] is True

    duplicated = api.post(
        "/api/kb/documents",
        json={"title": "新文档", "content": "这是一段用于直接添加接口测试的文档内容，长度满足最小要求。"},
    )
    assert duplicated.json()["idempotent"] is True

    updated = api.put(
        f"/api/kb/documents/{doc_id}",
        json={
            "title": "新文档（已更新）",
            "content": "这是原位更新后的文档内容，稳定 doc_id 不变，只重建发生变化的 chunk。",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["doc_id"] == doc_id
    assert any(item["doc_id"] == doc_id for item in api.get("/api/kb/documents").json())

    deleted = api.delete(f"/api/kb/documents/{doc_id}")
    assert deleted.status_code == 200
    assert deleted.json()["soft_deleted"]
    assert deleted.json()["recoverable"] is True
    assert api.get("/api/kb/deleted-documents").json()

    restored = api.post("/api/kb/restore", json={"source": deleted.json()["soft_deleted"]})
    assert restored.status_code == 200
    assert restored.json()["restored"] == deleted.json()["soft_deleted"]

    deleted_again = api.delete(f"/api/kb/documents/{doc_id}")
    assert deleted_again.status_code == 200


def test_memory_import_endpoint_is_idempotent(api: TestClient) -> None:
    payload = {
        "source": "hebb-mind",
        "records": [
            {
                "id": "incident-1",
                "partition": "episodic",
                "content": "Embedding 下载超时后检查网络代理和模型缓存。",
                "task_type": "incident_diagnosis",
            }
        ],
    }

    first = api.post("/api/memory/import", json=payload)
    second = api.post("/api/memory/import", json=payload)

    assert first.status_code == 200
    assert first.json()["created"] == 1
    assert second.json()["duplicates"] == 1
    assert api.get("/api/memory/stats").json()["episodic"] >= 1


def test_debug_run_resume_cancel_flow(api: TestClient) -> None:
    run = api.post(
        "/api/debug/run",
        json={"query": "什么是 RAG？", "top_k": 6, "breakpoints": ["verify_evidence"]},
    )
    assert run.status_code == 200
    paused = run.json()
    assert paused["run_status"] == "paused"
    assert paused["next_node"] == "verify_evidence"
    run_id = paused["run_id"]

    resumed = api.post(
        "/api/debug/resume", json={"run_id": run_id, "breakpoints": []}
    )
    assert resumed.status_code == 200
    assert resumed.json()["run_status"] == "completed"
    assert resumed.json()["answer"]

    already_done = api.post(
        "/api/debug/resume", json={"run_id": run_id, "breakpoints": []}
    )
    assert already_done.status_code == 404


def test_debug_cancel_then_resume_is_404(api: TestClient) -> None:
    run = api.post(
        "/api/debug/run",
        json={"query": "什么是 RAG？", "top_k": 6, "breakpoints": ["synthesize"]},
    ).json()
    assert run["run_status"] == "paused"

    cancelled = api.delete(f"/api/debug/runs/{run['run_id']}")
    assert cancelled.status_code == 200

    resumed = api.post(
        "/api/debug/resume", json={"run_id": run["run_id"], "breakpoints": []}
    )
    assert resumed.status_code == 404


def test_unready_agent_returns_503(api: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "agent", None)
    response = api.post("/api/ask", json={"query": "测试", "top_k": 6})
    assert response.status_code == 503
    assert "初始化" in response.json()["detail"]
