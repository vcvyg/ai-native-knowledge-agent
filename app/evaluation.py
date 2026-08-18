from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .agent import KnowledgeAgent


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    query: str
    expected_intent: str
    expected_sources: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    expect_low_evidence: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationCase":
        return cls(
            id=str(payload["id"]),
            query=str(payload["query"]),
            expected_intent=str(payload["expected_intent"]),
            expected_sources=tuple(payload.get("expected_sources", [])),
            expected_tools=tuple(payload.get("expected_tools", [])),
            expect_low_evidence=bool(payload.get("expect_low_evidence", False)),
        )


def load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            cases.append(EvaluationCase.from_dict(json.loads(line)))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation case at {path}:{line_no}: {exc}") from exc
    return cases


def evaluate_agent(agent: KnowledgeAgent, cases: Iterable[EvaluationCase]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for case in cases:
        response = agent.ask(case.query)
        trace_names = [call.name for call in response.trace]
        citation_sources = {item["source"] for item in response.citations}
        evidence_is_low = response.metrics["evidence_quality"] == "low"

        intent_ok = response.intent == case.expected_intent
        source_ok = not case.expected_sources or bool(
            citation_sources.intersection(case.expected_sources)
        )
        tools_ok = all(tool in trace_names for tool in case.expected_tools)
        evidence_ok = evidence_is_low == case.expect_low_evidence
        details.append(
            {
                "id": case.id,
                "query": case.query,
                "intent": response.intent,
                "intent_ok": intent_ok,
                "source_ok": source_ok,
                "tools_ok": tools_ok,
                "evidence_ok": evidence_ok,
                "latency_ms": response.metrics["latency_ms"],
                "retrieval_attempts": response.metrics["retrieval_attempts"],
                "stop_reason": response.metrics["stop_reason"],
                "trace": trace_names,
                "citations": sorted(citation_sources),
                "passed": intent_ok and source_ok and tools_ok and evidence_ok,
            }
        )

    latencies = [item["latency_ms"] for item in details]
    total = len(details)
    return {
        "summary": {
            "cases": total,
            "passed": sum(1 for item in details if item["passed"]),
            "pass_rate": ratio(sum(1 for item in details if item["passed"]), total),
            "intent_accuracy": ratio(sum(1 for item in details if item["intent_ok"]), total),
            "source_hit_rate": ratio(sum(1 for item in details if item["source_ok"]), total),
            "tool_accuracy": ratio(sum(1 for item in details if item["tools_ok"]), total),
            "evidence_gate_accuracy": ratio(
                sum(1 for item in details if item["evidence_ok"]), total
            ),
            "average_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": percentile(latencies, 0.95),
            "average_retrieval_attempts": round(
                statistics.mean(item["retrieval_attempts"] for item in details), 2
            )
            if details
            else 0.0,
        },
        "cases": details,
    }


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def case_to_dict(case: EvaluationCase) -> dict[str, Any]:
    return asdict(case)
