from __future__ import annotations

from pathlib import Path

from app.agent import KnowledgeAgent
from app.rag_engine import KnowledgeBase


def build_agent(tmp_path: Path, monkeypatch) -> KnowledgeAgent:
    monkeypatch.setenv("AI_AGENT_VECTOR_BACKEND", "memory")
    monkeypatch.setenv("AI_AGENT_EMBEDDING_PROVIDER", "test")
    monkeypatch.setenv("AI_AGENT_ALLOW_TEST_EMBEDDINGS", "1")
    monkeypatch.setenv("AI_AGENT_RERANKER", "heuristic")
    (tmp_path / "repository.md").write_text(
        "# 检索服务架构\n\n## RAG 证据链\n\n"
        "RAG 从仓库文档和运行手册检索相关证据，再让模型基于证据回答。"
        "系统通过稠密向量召回、重排、证据验证和引用追踪降低幻觉。\n\n"
        "## Cross-Encoder 变更\n\nCross-Encoder 延迟加载，失败时回退到 heuristic，"
        "同时保留原始错误并用固定评测集验证来源命中和延迟。\n",
        encoding="utf-8",
    )
    return KnowledgeAgent(KnowledgeBase(tmp_path))


def test_strong_evidence_takes_single_graph_pass(tmp_path: Path, monkeypatch) -> None:
    agent = build_agent(tmp_path, monkeypatch)

    response = agent.ask("什么是 RAG 检索增强生成？")

    assert response.intent == "repository_qa"
    assert response.metrics["retrieval_attempts"] == 1
    assert response.metrics["stop_reason"] == "evidence_sufficient"
    assert "rewrite_query" not in response.metrics["graph_path"]
    assert "evidence_verifier" in [item.name for item in response.trace]


def test_low_evidence_retries_then_stops(tmp_path: Path, monkeypatch) -> None:
    agent = build_agent(tmp_path, monkeypatch)

    response = agent.ask("量子色动力学重整化群方程的三圈修正是多少？")

    assert response.metrics["retrieval_attempts"] == 2
    assert response.metrics["stop_reason"] == "retrieval_retry_limit"
    assert response.metrics["evidence_quality"] == "low"
    assert "rewrite_query" in response.metrics["graph_path"]
    assert [item.name for item in response.trace].count("evidence_verifier") == 2


def test_low_evidence_never_calls_external_llm(tmp_path: Path, monkeypatch) -> None:
    class GuardLLM:
        enabled = True

        def chat(self, query: str, context: str) -> str:
            raise AssertionError("low-evidence query must not reach the external LLM")

    agent = build_agent(tmp_path, monkeypatch)
    agent.llm = GuardLLM()

    response = agent.ask("量子色动力学重整化群方程的三圈修正是多少？")

    assert response.metrics["stop_reason"] == "retrieval_retry_limit"
    assert "直接证据不足" in response.answer


def test_invalid_retry_budget_uses_safe_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AI_AGENT_MAX_RETRIEVAL_ATTEMPTS", "not-a-number")

    agent = build_agent(tmp_path, monkeypatch)

    assert agent.max_retrieval_attempts == 2


def test_debug_run_pauses_edits_and_restarts(tmp_path: Path, monkeypatch) -> None:
    agent = build_agent(tmp_path, monkeypatch)

    paused = agent.debug_start("什么是 RAG？", breakpoints=["verify_evidence"])

    assert paused.status == "paused"
    assert paused.next_node == "verify_evidence"
    assert paused.response.metrics["graph_path"] == ["route", "retrieve"]

    restarted = agent.debug_resume(
        paused.run_id,
        query="解释 RAG 检索增强生成的证据链",
        breakpoints=[],
        restart=True,
    )

    assert restarted.status == "completed"
    assert restarted.response.answer
    assert "debug_query_edit" in [item.name for item in restarted.response.trace]


def test_follow_up_uses_session_memory(tmp_path: Path, monkeypatch) -> None:
    agent = build_agent(tmp_path, monkeypatch)
    first = agent.ask("什么是 RAG 检索增强生成？")

    second = agent.ask("刚才那个再讲讲作用", session_id=first.session_id)

    assert second.session_id == first.session_id
    assert second.metrics["retrieved_chunks"] > 0
