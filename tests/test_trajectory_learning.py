from __future__ import annotations

from app.agent import AgentResponse, ToolCall
from app.trajectory_learning import (
    TrajectoryOracle,
    build_preference_pairs,
    build_sft_example,
    build_training_record,
    score_response,
)


def response(*, low: bool = False, stop_reason: str = "evidence_sufficient") -> AgentResponse:
    return AgentResponse(
        session_id="session",
        answer="grounded answer",
        intent="change_impact",
        citations=[{"source": "change.md", "section": "impact", "score": 0.8}],
        trace=[
            ToolCall("evidence_verifier", {}, {}, 1),
            ToolCall("impact_analysis", {}, {}, 1),
        ],
        suggestions=[],
        metrics={
            "evidence_quality": "low" if low else "ok",
            "stop_reason": stop_reason,
            "graph_path": ["route", "retrieve", "verify_evidence", "execute_tools", "synthesize"],
            "graph_events": [],
            "reranker_degraded": False,
            "reranker_error": None,
            "retrieval_attempts": 1,
            "react_steps": 1,
            "reflection": {"decision": "accept"},
        },
    )


def test_fully_verified_trajectory_receives_full_reward() -> None:
    oracle = TrajectoryOracle(
        expected_intent="change_impact",
        expected_sources=("change.md",),
        expected_tools=("evidence_verifier", "impact_analysis"),
    )

    reward = score_response(response(), oracle)

    assert reward.total == 1.0
    assert reward.penalties == {}


def test_unsupported_completion_is_penalized() -> None:
    oracle = TrajectoryOracle(
        expected_intent="change_impact",
        expected_sources=("missing.md",),
        expected_tools=("query_rewrite",),
        expect_low_evidence=True,
    )

    reward = score_response(response(low=False), oracle)

    assert reward.total < 0.5
    assert reward.penalties["unsupported_completion"] == -0.5


def test_training_record_keeps_tools_and_masks_observations() -> None:
    oracle = TrajectoryOracle(expected_intent="change_impact")

    record = build_training_record("analyze change", response(), oracle)

    assert record["trajectory"]["tool_calls"][1]["name"] == "impact_analysis"
    assert "tool_results" in record["loss_mask_policy"]["mask"]
    assert record["reward"]["total"] == 1.0

    sft = build_sft_example(record)
    assert sft["messages"][-1]["tool_policy"]
    assert sft["loss_mask"]["mask"] == record["loss_mask_policy"]["mask"]


def test_preference_pairs_require_same_prompt_and_distinct_reward() -> None:
    oracle = TrajectoryOracle(expected_intent="change_impact")
    chosen = build_training_record("same task", response(), oracle)
    rejected = build_training_record(
        "same task",
        response(low=True, stop_reason="retrieval_retry_limit"),
        oracle,
    )

    pairs = build_preference_pairs([rejected, chosen])

    assert len(pairs) == 1
    assert pairs[0]["chosen_reward"] > pairs[0]["rejected_reward"]
