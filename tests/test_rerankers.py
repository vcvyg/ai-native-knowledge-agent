from __future__ import annotations

import pytest

from app.rerankers import RerankerPipeline, build_reranker


class FailingReranker:
    name = "cross_encoder:test"

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, hits):
        self.calls += 1
        raise RuntimeError("model unavailable")


class RecordingFallback:
    name = "heuristic"

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, hits):
        self.calls += 1
        return hits


def test_reranker_failure_opens_circuit_and_reuses_fallback() -> None:
    primary = FailingReranker()
    fallback = RecordingFallback()
    pipeline = RerankerPipeline(primary, fallback)

    pipeline.rerank("first", [])
    pipeline.rerank("second", [])

    assert primary.calls == 1
    assert fallback.calls == 2
    assert pipeline.degraded is True
    assert pipeline.name == "heuristic"
    assert pipeline.fallback_count == 2
    assert pipeline.last_error == "RuntimeError: model unavailable"


def test_invalid_reranker_configuration_fails_fast() -> None:
    with pytest.raises(ValueError, match="heuristic or cross_encoder"):
        build_reranker("silent-unknown-strategy")
