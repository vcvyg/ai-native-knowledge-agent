from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / ".task_handoffs"
SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret)(\s*[:=]\s*)\S+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def redact(value: str) -> str:
    cleaned = value
    for pattern in SECRET_PATTERNS:
        if "bearer" in pattern.pattern.lower():
            cleaned = pattern.sub("Bearer [REDACTED]", cleaned)
        else:
            cleaned = pattern.sub(r"\1\2[REDACTED]", cleaned)
    return cleaned


def parse_resume_at(value: str | None, delay_minutes: int, now: datetime) -> datetime:
    if value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return now + timedelta(minutes=max(0, delay_minutes))


def git_snapshot(workdir: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=workdir,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status": run("status", "--short").splitlines(),
    }


def save_handoff(
    store: Path,
    *,
    objective: str,
    reason: str,
    resume_at: datetime,
    completed: list[str] | None = None,
    remaining: list[str] | None = None,
    evidence: list[str] | None = None,
    next_action: str = "",
    relevant_paths: list[str] | None = None,
    workdir: Path = ROOT,
    now: datetime | None = None,
) -> Path:
    created_at = now or utc_now()
    safe_objective = redact(objective.strip())
    handoff_id = (
        created_at.strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + sha256(safe_objective.encode("utf-8")).hexdigest()[:8]
    )
    payload = {
        "schema_version": 1,
        "id": handoff_id,
        "status": "deferred",
        "created_at": created_at.isoformat(),
        "resume_after": resume_at.astimezone(timezone.utc).isoformat(),
        "reason": redact(reason.strip()),
        "objective": safe_objective,
        "completed": [redact(item) for item in completed or []],
        "remaining": [redact(item) for item in remaining or []],
        "evidence": [redact(item) for item in evidence or []],
        "next_action": redact(next_action),
        "relevant_paths": [redact(item) for item in relevant_paths or []],
        "workspace": str(workdir.resolve()),
        "git": git_snapshot(workdir),
    }
    store.mkdir(parents=True, exist_ok=True)
    target = store / f"{handoff_id}.json"
    temporary = store / f".{handoff_id}.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_handoffs(store: Path) -> list[dict[str, Any]]:
    if not store.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(store.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record["_path"] = str(path)
        records.append(record)
    return records


def ready_handoffs(store: Path, now: datetime | None = None) -> list[dict[str, Any]]:
    current = now or utc_now()
    ready = []
    for record in load_handoffs(store):
        if record.get("status") != "deferred":
            continue
        resume_after = datetime.fromisoformat(str(record["resume_after"]).replace("Z", "+00:00"))
        if resume_after <= current:
            ready.append(record)
    return ready


def complete_handoff(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    record["status"] = "completed"
    record["completed_at"] = utc_now().isoformat()
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def continuation_prompt(record: dict[str, Any]) -> str:
    lines = [
        f"继续任务：{record['objective']}",
        f"交接原因：{record.get('reason', '')}",
        "已完成：",
        *[f"- {item}" for item in record.get("completed", [])],
        "剩余工作：",
        *[f"- {item}" for item in record.get("remaining", [])],
        f"下一步：{record.get('next_action', '')}",
        "先核对工作区和证据，不要重做已经完成的部分。",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persist a resumable task handoff instead of waiting in a blocking process."
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    save = subparsers.add_parser("save")
    save.add_argument("--objective", required=True)
    save.add_argument("--reason", default="quota unavailable")
    save.add_argument("--delay-minutes", type=int, default=60)
    save.add_argument("--resume-at")
    save.add_argument("--completed", action="append", default=[])
    save.add_argument("--remaining", action="append", default=[])
    save.add_argument("--evidence", action="append", default=[])
    save.add_argument("--next-action", default="")
    save.add_argument("--path", action="append", default=[])
    save.add_argument("--workdir", type=Path, default=ROOT)

    subparsers.add_parser("list")
    subparsers.add_parser("ready")
    show = subparsers.add_parser("show")
    show.add_argument("handoff", type=Path)
    prompt = subparsers.add_parser("prompt")
    prompt.add_argument("handoff", type=Path)
    complete = subparsers.add_parser("complete")
    complete.add_argument("handoff", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "save":
        now = utc_now()
        target = save_handoff(
            args.store,
            objective=args.objective,
            reason=args.reason,
            resume_at=parse_resume_at(args.resume_at, args.delay_minutes, now),
            completed=args.completed,
            remaining=args.remaining,
            evidence=args.evidence,
            next_action=args.next_action,
            relevant_paths=args.path,
            workdir=args.workdir,
            now=now,
        )
        print(target)
        return 0
    if args.command == "list":
        print(json.dumps(load_handoffs(args.store), ensure_ascii=False, indent=2))
        return 0
    if args.command == "ready":
        records = ready_handoffs(args.store)
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0 if records else 3
    if args.command == "show":
        print(args.handoff.read_text(encoding="utf-8"), end="")
        return 0
    if args.command == "prompt":
        record = json.loads(args.handoff.read_text(encoding="utf-8"))
        print(continuation_prompt(record))
        return 0
    if args.command == "complete":
        print(json.dumps(complete_handoff(args.handoff), ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
