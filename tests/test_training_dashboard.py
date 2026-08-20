from __future__ import annotations

import json
from pathlib import Path

from app.training_dashboard import load_training_dashboard


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def test_dashboard_reports_data_readiness_without_claiming_training(tmp_path: Path) -> None:
    (tmp_path / "evaluation-results.json").write_text(
        json.dumps(
            {
                "summary": {
                    "cases": 2,
                    "passed": 2,
                    "pass_rate": 1.0,
                    "intent_accuracy": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        tmp_path / "posttraining-trajectories.jsonl",
        [
            {
                "instance_id": "run-1",
                "prompt": "diagnose the embedding timeout",
                "trajectory": {
                    "intent": "incident_diagnosis",
                    "selected_skill": "incident_diagnosis",
                    "stop_reason": "evidence_sufficient",
                    "context": {"source_count": 2},
                    "plan": {"actions": ["incident_diagnosis"]},
                    "tool_calls": [
                        {"name": "evidence_verifier", "output": {"quality": "ok"}},
                        {"name": "incident_diagnosis", "output": {"status": "completed"}},
                    ],
                },
                "reward": {"total": 0.8, "components": {"intent": 0.2}},
            }
        ],
    )
    write_jsonl(tmp_path / "posttraining-sft.jsonl", [{"messages": []}])
    write_jsonl(tmp_path / "posttraining-preferences.jsonl", [])

    dashboard = load_training_dashboard(tmp_path)

    assert dashboard["status"] == "data_ready"
    assert dashboard["boundary"]["actual_training_run"] is False
    assert dashboard["summary"]["trajectory_records"] == 1
    assert dashboard["summary"]["sft_examples"] == 1
    assert dashboard["summary"]["preference_pairs"] == 0
    assert dashboard["summary"]["mean_reward"] == 0.8
    assert dashboard["trajectories"][0]["evidence_quality"] == "ok"
    assert dashboard["trajectories"][0]["source_count"] == 2
    assert dashboard["stages"][3]["status"] == "not_started"
    assert dashboard["stages"][4]["status"] == "blocked"


def test_dashboard_surfaces_missing_and_invalid_artifacts(tmp_path: Path) -> None:
    (tmp_path / "evaluation-results.json").write_text("not-json", encoding="utf-8")
    (tmp_path / "posttraining-trajectories.jsonl").write_text("{broken\n", encoding="utf-8")
    (tmp_path / "posttraining-sft.jsonl").write_text("", encoding="utf-8")

    dashboard = load_training_dashboard(tmp_path)

    assert dashboard["status"] == "incomplete"
    assert dashboard["summary"]["trajectory_records"] == 0
    assert any("evaluation-results.json" in issue for issue in dashboard["issues"])
    assert any("posttraining-trajectories.jsonl:1" in issue for issue in dashboard["issues"])
    assert any("posttraining-preferences.jsonl" in issue for issue in dashboard["issues"])
