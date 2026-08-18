from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .rag_engine import SearchHit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list["SearchHit"]) -> list["SearchHit"]: ...


class HeuristicReranker:
    name = "heuristic"

    def rerank(self, query: str, hits: list["SearchHit"]) -> list["SearchHit"]:
        # Import here to keep the protocol independent from the retrieval module.
        from .rag_engine import expand_query, extract_terms

        query_terms = extract_terms(expand_query(query))
        for hit in hits:
            dense_bonus = min(0.08, len(query_terms & extract_terms(hit.chunk.text)) * 0.015)
            section = hit.chunk.section.lower()
            structure_bonus = 0.04 if any(
                keyword in section for keyword in ["agent", "rag", "评估", "架构"]
            ) else 0.0
            hit.score = float(hit.score + dense_bonus + structure_bonus)
        return sorted(hits, key=lambda item: item.score, reverse=True)


class CrossEncoderReranker:
    """Optional lazy-loaded Sentence Transformers Cross-Encoder reranker."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.name = f"cross_encoder:{model_name}"
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list["SearchHit"]) -> list["SearchHit"]:
        if not hits:
            return hits
        model = self._load()
        pairs = [(query, hit.chunk.text) for hit in hits]
        predictions = model.predict(pairs, show_progress_bar=False)
        for hit, prediction in zip(hits, predictions):
            cross_score = float(prediction)
            hit.score = 0.35 * hit.score + 0.65 * cross_score
        return sorted(hits, key=lambda item: item.score, reverse=True)


class RerankerPipeline:
    """Run a primary reranker with circuit-breaking fallback semantics."""

    def __init__(self, primary: Reranker, fallback: Reranker | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.requested_name = primary.name
        self.active_name = primary.name
        self.last_error: str | None = None
        self.degraded = False
        self.fallback_count = 0

    @property
    def name(self) -> str:
        return self.active_name

    def rerank(self, query: str, hits: list["SearchHit"]) -> list["SearchHit"]:
        if self.degraded and self.fallback is not None:
            self.active_name = self.fallback.name
            self.fallback_count += 1
            return self.fallback.rerank(query, hits)
        try:
            result = self.primary.rerank(query, hits)
            self.active_name = self.primary.name
            self.last_error = None
            return result
        except Exception as exc:
            if self.fallback is None:
                raise
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.degraded = True
            self.fallback_count += 1
            self.active_name = self.fallback.name
            return self.fallback.rerank(query, hits)


def build_reranker(preference: str | None = None) -> RerankerPipeline:
    if preference:
        selected = preference
    else:
        selected = os.getenv("AI_AGENT_RERANKER", "heuristic")
    selected = selected.strip().lower()
    heuristic = HeuristicReranker()
    if selected in {"cross_encoder", "cross-encoder", "crossencoder"}:
        model_name = os.getenv(
            "AI_AGENT_RERANKER_MODEL",
            "BAAI/bge-reranker-base",
        )
        return RerankerPipeline(CrossEncoderReranker(model_name), fallback=heuristic)
    if selected in {"heuristic", "local"}:
        return RerankerPipeline(heuristic)
    raise ValueError(
        "AI_AGENT_RERANKER must be heuristic or cross_encoder"
    )
