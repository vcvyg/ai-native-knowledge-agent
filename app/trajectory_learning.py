from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from .agent import AgentResponse


@dataclass(frozen=True)
class TrajectoryOracle:
    expected_intent: str
    expected_sources: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    expect_low_evidence: bool = False


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: dict[str, float]
    penalties: dict[str, float]


def score_response(response: AgentResponse, oracle: TrajectoryOracle) -> RewardBreakdown:
    trace_names = [call.name for call in response.trace]
    citation_sources = {str(item["source"]) for item in response.citations}
    evidence_is_low = response.metrics.get("evidence_quality") == "low"
    stop_reason = str(response.metrics.get("stop_reason", ""))
    graph_path = list(response.metrics.get("graph_path", []))

    components = {
        "intent": 0.20 if response.intent == oracle.expected_intent else 0.0,
        "source": 0.20
        if not oracle.expected_sources or citation_sources.intersection(oracle.expected_sources)
        else 0.0,
        "tool_selection": 0.20
        if all(tool in trace_names for tool in oracle.expected_tools)
        else 0.0,
        "evidence_gate": 0.25 if evidence_is_low == oracle.expect_low_evidence else 0.0,
        "bounded_stop": 0.15
        if stop_reason
        in {"evidence_sufficient", "retrieval_retry_limit", "tool_only_workflow"}
        and len(graph_path) <= 16
        else 0.0,
    }
    penalties: dict[str, float] = {}
    if oracle.expect_low_evidence and stop_reason != "retrieval_retry_limit":
        penalties["unsupported_completion"] = -0.50
    if not oracle.expect_low_evidence and evidence_is_low:
        penalties["false_rejection"] = -0.25
    if response.metrics.get("reranker_degraded") and not response.metrics.get("reranker_error"):
        penalties["silent_degradation"] = -0.25
    if len(graph_path) > 16 or stop_reason == "graph_transition_limit":
        penalties["transition_overflow"] = -0.50
    if int(response.metrics.get("retrieval_attempts", 0)) > 3:
        penalties["retrieval_budget_violation"] = -0.40
    if int(response.metrics.get("react_steps", 0)) > 5:
        penalties["react_budget_violation"] = -0.40
    if stop_reason == "human_approval_required" and response.metrics.get(
        "reflection", {}
    ).get("decision") != "stop":
        penalties["approval_bypass"] = -0.75

    total = sum(components.values()) + sum(penalties.values())
    return RewardBreakdown(
        total=round(max(-1.0, min(1.0, total)), 4),
        components=components,
        penalties=penalties,
    )


def build_training_record(
    query: str,
    response: AgentResponse,
    oracle: TrajectoryOracle,
) -> dict[str, Any]:
    reward = score_response(response, oracle)
    record_id = sha256(query.encode("utf-8")).hexdigest()[:16]
    return {
        "schema_version": 2,
        "instance_id": f"engineering-agent-{record_id}",
        "prompt": query,
        "oracle": asdict(oracle),
        "trajectory": {
            "intent": response.intent,
            "assistant_answer": response.answer,
            "tool_calls": [
                {
                    "name": call.name,
                    "input": call.input,
                    "output": call.output,
                    "latency_ms": call.latency_ms,
                }
                for call in response.trace
            ],
            "graph_events": response.metrics.get("graph_events", []),
            "citations": [
                {
                    "source": item["source"],
                    "section": item["section"],
                    "score": item["score"],
                }
                for item in response.citations
            ],
            "stop_reason": response.metrics.get("stop_reason"),
            "selected_skill": response.metrics.get("selected_skill"),
            "plan": response.metrics.get("plan", {}),
            "reflection": response.metrics.get("reflection", {}),
            "context": response.metrics.get("context", {}),
            "memory_recall": response.metrics.get("memory_recall", []),
            "policy_steps": build_policy_steps(response),
        },
        "reward": asdict(reward),
        "loss_mask_policy": {
            "train": ["assistant_answer", "assistant_tool_calls"],
            "mask": ["tool_results", "retrieved_evidence", "system_context"],
        },
    }


def build_policy_steps(response: AgentResponse) -> list[dict[str, Any]]:
    """Convert a run into next-action supervision for an Agent policy model."""

    steps: list[dict[str, Any]] = []
    previous_observation: dict[str, Any] | None = None
    for index, call in enumerate(response.trace):
        if call.name in {"intent_router", "memory_recall", "planner", "hybrid_retrieval", "rerank"}:
            previous_observation = call.output
            continue
        steps.append(
            {
                "step": index,
                "state": {
                    "intent": response.intent,
                    "evidence_quality": response.metrics.get("evidence_quality"),
                    "previous_observation": previous_observation,
                },
                "action": call.name,
                "action_input": call.input,
                "observation": call.output,
            }
        )
        previous_observation = call.output
    return steps


def build_sft_example(record: dict[str, Any]) -> dict[str, Any]:
    """Export tool-selection supervision while masking retrieved/tool content."""

    trajectory = record["trajectory"]
    policy = [
        {
            "state": step["state"],
            "action": step["action"],
            "action_input": step["action_input"],
        }
        for step in trajectory.get("policy_steps", [])
    ]
    return {
        "id": record["instance_id"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "Select the next bounded engineering-agent action. "
                    "Use evidence, obey retry/approval budgets, and stop on unsupported claims."
                ),
            },
            {"role": "user", "content": record["prompt"]},
            {
                "role": "assistant",
                "content": trajectory["assistant_answer"],
                "tool_policy": policy,
            },
        ],
        "reward": record["reward"]["total"],
        "loss_mask": record["loss_mask_policy"],
    }


def build_preference_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create preference pairs only when the same prompt has distinct rewards."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["prompt"]), []).append(record)
    pairs: list[dict[str, Any]] = []
    for prompt, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda item: float(item["reward"]["total"]), reverse=True)
        if len(ordered) < 2 or ordered[0]["reward"]["total"] == ordered[-1]["reward"]["total"]:
            continue
        pairs.append(
            {
                "prompt": prompt,
                "chosen": ordered[0]["trajectory"],
                "rejected": ordered[-1]["trajectory"],
                "chosen_reward": ordered[0]["reward"]["total"],
                "rejected_reward": ordered[-1]["reward"]["total"],
            }
        )
    return pairs
