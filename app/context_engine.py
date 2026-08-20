from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .rag_engine import Chunk, SearchHit


WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


@dataclass(frozen=True)
class EvidenceRelation:
    source_doc_id: str
    target_doc_id: str
    relation: str
    label: str


@dataclass(frozen=True)
class ContextBundle:
    hits: list[SearchHit]
    relations: list[EvidenceRelation]
    token_budget: int
    estimated_tokens: int
    dropped_chunks: int
    source_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_budget": self.token_budget,
            "estimated_tokens": self.estimated_tokens,
            "dropped_chunks": self.dropped_chunks,
            "source_count": self.source_count,
            "relations": [asdict(item) for item in self.relations],
        }


class EvidenceGraph:
    """Bounded document relationship graph built from links and document order."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._doc_chunks: dict[str, list[Chunk]] = {}
        self._title_to_doc: dict[str, str] = {}
        self._relations: list[EvidenceRelation] = []
        for chunk in chunks:
            self._doc_chunks.setdefault(chunk.doc_id, []).append(chunk)
            self._title_to_doc.setdefault(_normal_key(chunk.title), chunk.doc_id)
            self._title_to_doc.setdefault(_normal_key(chunk.doc_id), chunk.doc_id)
        self._build_relations()

    @property
    def relations(self) -> list[EvidenceRelation]:
        return list(self._relations)

    def expand(self, hits: list[SearchHit], limit: int = 2) -> tuple[list[SearchHit], list[EvidenceRelation]]:
        """Add at most ``limit`` linked-document candidates with a score decay."""

        if not hits or limit <= 0:
            return list(hits), []
        selected_doc_ids = {hit.chunk.doc_id for hit in hits}
        by_source: dict[str, list[EvidenceRelation]] = {}
        for relation in self._relations:
            by_source.setdefault(relation.source_doc_id, []).append(relation)

        expanded = list(hits)
        used_relations: list[EvidenceRelation] = []
        for hit in hits:
            for relation in by_source.get(hit.chunk.doc_id, []):
                if relation.target_doc_id in selected_doc_ids:
                    used_relations.append(relation)
                    continue
                target_chunks = self._doc_chunks.get(relation.target_doc_id, [])
                if not target_chunks:
                    continue
                expanded.append(
                    replace(
                        hit,
                        chunk=target_chunks[0],
                        score=max(0.0, hit.score * 0.72),
                        vector_score=max(0.0, hit.vector_score * 0.72),
                        keyword_score=0.0,
                        title_score=0.0,
                    )
                )
                selected_doc_ids.add(relation.target_doc_id)
                used_relations.append(relation)
                if len(expanded) >= len(hits) + limit:
                    return expanded, used_relations
        return expanded, used_relations

    def _build_relations(self) -> None:
        seen: set[tuple[str, str, str]] = set()
        for source_doc_id, chunks in self._doc_chunks.items():
            for chunk in chunks:
                for raw_target in WIKILINK_RE.findall(chunk.text):
                    target_doc_id = self._title_to_doc.get(_normal_key(raw_target))
                    if not target_doc_id or target_doc_id == source_doc_id:
                        continue
                    key = (source_doc_id, target_doc_id, "wikilink")
                    if key in seen:
                        continue
                    seen.add(key)
                    self._relations.append(
                        EvidenceRelation(
                            source_doc_id=source_doc_id,
                            target_doc_id=target_doc_id,
                            relation="wikilink",
                            label=raw_target.strip(),
                        )
                    )


class ContextPacker:
    """Pack evidence under a token budget with deduplication and source diversity."""

    def __init__(self, token_budget: int = 1800, max_chunks_per_source: int = 2) -> None:
        self.token_budget = max(128, token_budget)
        self.max_chunks_per_source = max(1, max_chunks_per_source)

    def pack(
        self,
        hits: list[SearchHit],
        relations: list[EvidenceRelation] | None = None,
    ) -> ContextBundle:
        unique: dict[str, SearchHit] = {}
        for hit in hits:
            current = unique.get(hit.chunk.id)
            if current is None or hit.score > current.score:
                unique[hit.chunk.id] = hit

        ranked = sorted(
            unique.values(),
            key=lambda item: (
                _source_priority(item.chunk.source),
                item.score,
                item.keyword_score,
            ),
            reverse=True,
        )
        selected: list[SearchHit] = []
        per_source: dict[str, int] = {}
        estimated_tokens = 0
        for hit in ranked:
            source = hit.chunk.source
            if per_source.get(source, 0) >= self.max_chunks_per_source:
                continue
            cost = estimate_tokens(f"{hit.chunk.title}\n{hit.chunk.section}\n{hit.chunk.text}")
            if selected and estimated_tokens + cost > self.token_budget:
                continue
            if not selected and cost > self.token_budget:
                # Keep one grounded chunk even when it needs truncation later.
                cost = self.token_budget
            selected.append(hit)
            per_source[source] = per_source.get(source, 0) + 1
            estimated_tokens += cost
            if estimated_tokens >= self.token_budget:
                break

        selected_docs = {hit.chunk.doc_id for hit in selected}
        relevant_relations = [
            relation
            for relation in relations or []
            if relation.source_doc_id in selected_docs or relation.target_doc_id in selected_docs
        ]
        return ContextBundle(
            hits=selected,
            relations=relevant_relations,
            token_budget=self.token_budget,
            estimated_tokens=min(estimated_tokens, self.token_budget),
            dropped_chunks=max(0, len(unique) - len(selected)),
            source_count=len({hit.chunk.source for hit in selected}),
        )


def estimate_tokens(text: str) -> int:
    """Cheap mixed Chinese/ASCII estimate for deterministic context budgeting."""

    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    non_chinese = max(0, len(text) - chinese)
    return max(1, math.ceil(chinese / 1.5 + non_chinese / 4))


def _normal_key(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value, flags=re.UNICODE).lower()


def _source_priority(source: str) -> int:
    lowered = source.lower()
    if "runbook" in lowered or "standard" in lowered:
        return 3
    if "architecture" in lowered or "playbook" in lowered:
        return 2
    return 1
