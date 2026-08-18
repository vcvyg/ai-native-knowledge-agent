from __future__ import annotations

from app.agent import AgentResponse, ToolCall
from app.trajectory_learning import TrajectoryOracle, build_training_record, score_response


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
