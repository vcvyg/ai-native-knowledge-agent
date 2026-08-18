from __future__ import annotations

from pathlib import Path

from app.agent import KnowledgeAgent
from app.evaluation import EvaluationCase, evaluate_agent
from app.rag_engine import KnowledgeBase


def test_evaluation_reports_intent_source_tool_and_evidence_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("AI_AGENT_VECTOR_BACKEND", "memory")
    monkeypatch.setenv("AI_AGENT_EMBEDDING_PROVIDER", "test")
    monkeypatch.setenv("AI_AGENT_ALLOW_TEST_EMBEDDINGS", "1")
    monkeypatch.setenv("AI_AGENT_RERANKER", "heuristic")
    (tmp_path / "rag.md").write_text(
        "# RAG\n\n## 定义\n\nRAG 使用检索证据增强模型回答，并输出引用。\n",
        encoding="utf-8",
    )
    agent = KnowledgeAgent(KnowledgeBase(tmp_path))
    cases = [
        EvaluationCase(
            id="rag",
            query="什么是 RAG？",
            expected_intent="repository_qa",
            expected_sources=("rag.md",),
            expected_tools=("evidence_verifier", "explain_component"),
        )
    ]

    result = evaluate_agent(agent, cases)

    assert result["summary"]["cases"] == 1
    assert result["summary"]["pass_rate"] == 1.0
    assert result["cases"][0]["passed"] is True
