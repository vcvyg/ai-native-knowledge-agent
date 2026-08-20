from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARTIFACTS = (
    ("evaluation", "evaluation-results.json", "Evaluation report"),
    ("trajectories", "posttraining-trajectories.jsonl", "Agent trajectories"),
    ("sft", "posttraining-sft.jsonl", "SFT examples"),
    ("preferences", "posttraining-preferences.jsonl", "Preference pairs"),
)


def load_training_dashboard(report_dir: Path) -> dict[str, Any]:
    """Build a read-only post-training dashboard from generated artifacts.

    This intentionally does not manufacture model-training metrics. Loss,
    accelerator utilisation and checkpoints stay unavailable until an actual
    SFT or preference-training job writes those artifacts.
    """

    issues: list[str] = []
    evaluation = _read_json(report_dir / "evaluation-results.json", issues)
    trajectory_rows = _read_jsonl(report_dir / "posttraining-trajectories.jsonl", issues)
    sft_rows = _read_jsonl(report_dir / "posttraining-sft.jsonl", issues)
    preference_rows = _read_jsonl(report_dir / "posttraining-preferences.jsonl", issues)

    evaluation_summary = evaluation.get("summary", {}) if isinstance(evaluation, dict) else {}
    trajectories = [_trajectory_summary(row) for row in trajectory_rows]
    rewards = [item["reward"] for item in trajectories]
    mean_reward = round(sum(rewards) / len(rewards), 4) if rewards else 0.0

    summary = {
        "evaluation_cases": _as_int(evaluation_summary.get("cases")),
        "evaluation_pass_rate": _as_float(evaluation_summary.get("pass_rate")),
        "trajectory_records": len(trajectory_rows),
        "sft_examples": len(sft_rows),
        "preference_pairs": len(preference_rows),
        "mean_reward": mean_reward,
    }

    stages = [
        _stage(
            "trajectory",
            "Trajectory collection",
            "complete" if trajectory_rows else "empty",
            f"{len(trajectory_rows)} records",
        ),
        _stage(
            "sft_data",
            "SFT dataset",
            "ready" if sft_rows else "blocked",
            f"{len(sft_rows)} examples",
        ),
        _stage(
            "preference_data",
            "Preference dataset",
            "ready" if preference_rows else "blocked",
            f"{len(preference_rows)} same-prompt pairs",
        ),
        _stage("sft_train", "SFT training", "not_started", "No model run recorded"),
        _stage(
            "rl_train",
            "GRPO / GSPO",
            "blocked" if not preference_rows else "not_started",
            "Requires preference data and an accelerator-backed run",
        ),
    ]

    return {
        "status": "data_ready" if trajectory_rows and sft_rows else "incomplete",
        "boundary": {
            "actual_training_run": False,
            "message": (
                "No SFT or GRPO/GSPO model training run has been executed. "
                "Loss, GPU utilisation and checkpoint metrics are therefore unavailable."
            ),
        },
        "summary": summary,
        "evaluation": evaluation_summary,
        "reward_distribution": _reward_distribution(rewards),
        "trajectories": trajectories,
        "stages": stages,
        "artifacts": _artifact_statuses(report_dir, evaluation, trajectory_rows, sft_rows, preference_rows),
        "issues": issues,
    }


def _read_json(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"Missing artifact: {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"Could not read {path.name}: {type(exc).__name__}")
        return {}
    if not isinstance(value, dict):
        issues.append(f"Invalid artifact shape: {path.name}")
        return {}
    return value


def _read_jsonl(path: Path, issues: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        issues.append(f"Missing artifact: {path.name}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        issues.append(f"Could not read {path.name}: {type(exc).__name__}")
        return []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            issues.append(f"Invalid JSON in {path.name}:{line_number}")
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            issues.append(f"Invalid row shape in {path.name}:{line_number}")
    return rows


def _trajectory_summary(row: dict[str, Any]) -> dict[str, Any]:
    trajectory = row.get("trajectory", {})
    trajectory = trajectory if isinstance(trajectory, dict) else {}
    reward = row.get("reward", {})
    reward = reward if isinstance(reward, dict) else {}
    context = trajectory.get("context", {})
    context = context if isinstance(context, dict) else {}
    plan = trajectory.get("plan", {})
    plan = plan if isinstance(plan, dict) else {}
    actions = plan.get("actions", [])
    actions = actions if isinstance(actions, list) else []
    graph_events = trajectory.get("graph_events", [])
    graph_events = graph_events if isinstance(graph_events, list) else []
    tool_calls = trajectory.get("tool_calls", [])
    tool_calls = tool_calls if isinstance(tool_calls, list) else []
    evidence_quality = "unknown"
    for call in tool_calls:
        if not isinstance(call, dict) or call.get("name") != "evidence_verifier":
            continue
        output = call.get("output", {})
        if isinstance(output, dict):
            evidence_quality = str(output.get("quality", "unknown"))

    return {
        "id": str(row.get("instance_id", "unknown")),
        "prompt": str(row.get("prompt", "")),
        "intent": str(trajectory.get("intent", "unknown")),
        "selected_skill": str(trajectory.get("selected_skill", "none")),
        "reward": _as_float(reward.get("total")),
        "reward_components": reward.get("components", {}),
        "stop_reason": str(trajectory.get("stop_reason", "unknown")),
        "tool_calls": len(tool_calls),
        "react_steps": sum(
            1 for event in graph_events if isinstance(event, dict) and event.get("node") == "react"
        )
        or len(actions),
        "source_count": _as_int(context.get("source_count")),
        "evidence_quality": evidence_quality,
    }


def _reward_distribution(rewards: list[float]) -> list[dict[str, Any]]:
    buckets = [
        ("< 0", lambda reward: reward < 0),
        ("0–0.49", lambda reward: 0 <= reward < 0.5),
        ("0.5–0.79", lambda reward: 0.5 <= reward < 0.8),
        ("0.8–0.99", lambda reward: 0.8 <= reward < 1),
        ("1.0", lambda reward: reward >= 1),
    ]
    return [
        {"bucket": label, "count": sum(1 for reward in rewards if predicate(reward))}
        for label, predicate in buckets
    ]


def _artifact_statuses(
    report_dir: Path,
    evaluation: dict[str, Any],
    trajectories: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    preference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = {
        "evaluation": _as_int(evaluation.get("summary", {}).get("cases")) if evaluation else 0,
        "trajectories": len(trajectories),
        "sft": len(sft_rows),
        "preferences": len(preference_rows),
    }
    statuses: list[dict[str, Any]] = []
    for artifact_id, filename, label in ARTIFACTS:
        path = report_dir / filename
        statuses.append(
            {
                "id": artifact_id,
                "label": label,
                "filename": filename,
                "exists": path.exists(),
                "records": records[artifact_id],
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return statuses


def _stage(stage_id: str, label: str, status: str, detail: str) -> dict[str, str]:
    return {"id": stage_id, "label": label, "status": status, "detail": detail}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
