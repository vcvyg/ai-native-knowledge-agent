from __future__ import annotations

from pathlib import Path

from app.context_engine import ContextPacker, EvidenceGraph
from app.rag_engine import Chunk, SearchHit, chunk_document


def hit(chunk: Chunk, score: float = 0.8) -> SearchHit:
    return SearchHit(
        chunk=chunk,
        score=score,
        vector_score=score,
        keyword_score=0.2,
        title_score=0.1,
    )


def test_chunk_identity_survives_unrelated_section_insert(tmp_path: Path) -> None:
    path = tmp_path / "architecture.md"
    original = chunk_document(
        path,
        "# Architecture\n\n## Retrieval\n\nDense retrieval keeps stable evidence identifiers.\n",
    )
    changed = chunk_document(
        path,
        "# Architecture\n\n## New\n\nA newly inserted section has enough text to become a chunk.\n\n"
        "## Retrieval\n\nDense retrieval keeps stable evidence identifiers.\n",
    )

    original_chunk = next(item for item in original if item.section == "Retrieval")
    changed_chunk = next(item for item in changed if item.section == "Retrieval")

    assert original_chunk.id == changed_chunk.id


def test_wikilink_graph_expands_and_context_packer_enforces_budget() -> None:
    source = Chunk("a-1", "a", "Architecture", "Links", "a.md", "See [[Runbook]] for recovery.")
    target = Chunk(
        "b-1",
        "b",
        "Runbook",
        "Recovery",
        "runbook.md",
        "Rollback restores the previous version after approval and validation.",
    )
    graph = EvidenceGraph([source, target])

    expanded, relations = graph.expand([hit(source)], limit=1)
    bundle = ContextPacker(token_budget=128, max_chunks_per_source=1).pack(expanded, relations)

    assert [item.chunk.doc_id for item in expanded] == ["a", "b"]
    assert relations[0].relation == "wikilink"
    assert bundle.estimated_tokens <= bundle.token_budget
    assert bundle.source_count == 2
