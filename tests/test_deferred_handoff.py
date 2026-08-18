from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from scripts.deferred_handoff import continuation_prompt, ready_handoffs, save_handoff


def test_handoff_is_delayed_and_becomes_ready(tmp_path) -> None:
    now = datetime(2026, 8, 18, 8, 0, tzinfo=timezone.utc)
    target = save_handoff(
        tmp_path / "handoffs",
        objective="finish trajectory exporter",
        reason="quota unavailable",
        resume_at=now + timedelta(minutes=30),
        completed=["upstream inspected"],
        remaining=["implement exporter"],
        next_action="run pytest",
        workdir=tmp_path,
        now=now,
    )

    assert ready_handoffs(tmp_path / "handoffs", now + timedelta(minutes=20)) == []
    ready = ready_handoffs(tmp_path / "handoffs", now + timedelta(minutes=31))
    assert ready[0]["id"] == json.loads(target.read_text(encoding="utf-8"))["id"]
    assert "implement exporter" in continuation_prompt(ready[0])


def test_handoff_redacts_secrets(tmp_path) -> None:
    now = datetime(2026, 8, 18, 8, 0, tzinfo=timezone.utc)
    target = save_handoff(
        tmp_path / "handoffs",
        objective="continue task token=super-secret",
        reason="Bearer abc.def.ghi",
        resume_at=now,
        workdir=tmp_path,
        now=now,
    )

    text = target.read_text(encoding="utf-8")
    assert "super-secret" not in text
    assert "abc.def.ghi" not in text
    assert "[REDACTED]" in text
