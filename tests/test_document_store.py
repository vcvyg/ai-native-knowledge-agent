from __future__ import annotations

from app.document_store import resolve_document_target
from app.rag_engine import KnowledgeBase


def test_repeated_content_resolves_to_existing_document(tmp_path) -> None:
    first = resolve_document_target(tmp_path, "Architecture", "same\r\ncontent")
    first.path.write_text("same\ncontent", encoding="utf-8")

    repeated = resolve_document_target(tmp_path, "Renamed upload", "same\ncontent")

    assert repeated.already_exists is True
    assert repeated.path == first.path
    assert repeated.digest == first.digest


def test_changed_content_gets_a_new_content_address(tmp_path) -> None:
    first = resolve_document_target(tmp_path, "Runbook", "version one")
    first.path.write_text("version one", encoding="utf-8")

    changed = resolve_document_target(tmp_path, "Runbook", "version two")

    assert changed.already_exists is False
    assert changed.path != first.path
    assert changed.digest != first.digest


def test_knowledge_document_soft_delete_and_restore(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AI_AGENT_VECTOR_BACKEND", "memory")
    monkeypatch.setenv("AI_AGENT_EMBEDDING_PROVIDER", "test")
    monkeypatch.setenv("AI_AGENT_ALLOW_TEST_EMBEDDINGS", "1")
    monkeypatch.setenv("AI_AGENT_RERANKER", "heuristic")
    path = tmp_path / "runbook.md"
    path.write_text(
        "# Runbook\n\n## Recovery\n\nRollback is approved, executed, and verified before closure.",
        encoding="utf-8",
    )
    kb = KnowledgeBase(tmp_path)
    doc_id = kb.documents()[0]["doc_id"]

    tombstone = kb.soft_delete_document(doc_id)

    assert tombstone is not None
    assert kb.documents() == []
    assert kb.deleted_documents()[0]["source"] == "runbook.md"

    restored = kb.restore_document("runbook.md")

    assert restored == path
    assert kb.documents()[0]["doc_id"] == doc_id
