from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Literal

from .rag_engine import compact_text, extract_terms

MemoryPartition = Literal["working", "episodic", "semantic", "procedural"]


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    partition: MemoryPartition
    content: str
    source: str
    task_type: str = ""
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryHit:
    record: MemoryRecord
    score: float
    relevance: float
    recency: float
    task_match: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self.record),
            "score": round(self.score, 4),
            "relevance": round(self.relevance, 4),
            "recency": round(self.recency, 4),
            "task_match": round(self.task_match, 4),
        }


class AgentMemory:
    """Partitioned Agent memory with idempotent imports and explainable recall.

    Working memory remains session-scoped. Episodic, semantic and procedural
    records are shared long-term memory and are ranked by relevance, recency,
    task match and importance.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._working: dict[str, dict[str, Any]] = {}
        self._records: dict[str, MemoryRecord] = {}
        self._lock = threading.RLock()

    def ensure(self, session_id: str | None) -> str:
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            self._sessions.setdefault(sid, [])
            self._working.setdefault(sid, {})
        return sid

    def add(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": role, "content": content})
            del history[:-8]

    def recent(self, session_id: str, limit: int = 4) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, [])[-limit:])

    def last_user_topic(self, session_id: str) -> str:
        with self._lock:
            for item in reversed(self._sessions.get(session_id, [])):
                if item["role"] == "user":
                    return item["content"]
        return ""

    def update_working(self, session_id: str, **state: Any) -> None:
        with self._lock:
            self._working.setdefault(session_id, {}).update(state)

    def working(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._working.get(session_id, {}))

    def remember(
        self,
        partition: MemoryPartition,
        content: str,
        *,
        source: str,
        task_type: str = "",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        external_id: str | None = None,
    ) -> tuple[MemoryRecord, bool]:
        if partition == "working":
            raise ValueError("working memory must be updated with update_working")
        normalized = " ".join(content.split())
        stable_key = external_id or normalized
        record_id = sha256(f"{source}\n{partition}\n{stable_key}".encode()).hexdigest()[:20]
        with self._lock:
            existing = self._records.get(record_id)
            if existing is not None:
                return existing, False
            record = MemoryRecord(
                id=record_id,
                partition=partition,
                content=normalized,
                source=source,
                task_type=task_type,
                importance=max(0.0, min(1.0, importance)),
                metadata=dict(metadata or {}),
            )
            self._records[record_id] = record
            return record, True

    def import_records(
        self,
        records: list[dict[str, Any]],
        *,
        source: str,
    ) -> dict[str, int]:
        """Import a Hebb-style corpus with stable source ids and partition routing."""

        created = 0
        duplicates = 0
        for raw in records:
            partition = str(raw.get("partition", "episodic")).lower()
            if partition not in {"episodic", "semantic", "procedural"}:
                partition = "episodic"
            _, was_created = self.remember(
                partition,  # type: ignore[arg-type]
                str(raw.get("content", "")),
                source=source,
                task_type=str(raw.get("task_type", "")),
                importance=float(raw.get("importance", 0.5)),
                metadata=dict(raw.get("metadata", {})),
                external_id=str(raw.get("id", "")) or None,
            )
            created += int(was_created)
            duplicates += int(not was_created)
        return {"created": created, "duplicates": duplicates, "total": len(records)}

    def remember_episode(
        self,
        *,
        query: str,
        answer: str,
        intent: str,
        stop_reason: str,
        sources: list[str],
        success: bool,
    ) -> MemoryRecord:
        content = (
            f"任务：{compact_text(query, 180)}；结果：{compact_text(answer, 260)}；"
            f"停止原因：{stop_reason}；证据：{', '.join(sources[:4]) or 'none'}"
        )
        record, _ = self.remember(
            "episodic",
            content,
            source="agent_trajectory",
            task_type=intent,
            importance=0.8 if success else 0.65,
            metadata={"success": success, "stop_reason": stop_reason, "sources": sources[:4]},
            external_id=sha256(f"{query}\n{answer}".encode()).hexdigest(),
        )
        return record

    def recall(
        self,
        query: str,
        *,
        task_type: str = "",
        limit: int = 4,
        now: float | None = None,
    ) -> list[MemoryHit]:
        query_terms = extract_terms(query)
        current_time = now or time.time()
        with self._lock:
            records = list(self._records.values())
        hits: list[MemoryHit] = []
        for record in records:
            record_terms = extract_terms(record.content)
            relevance = (
                len(query_terms.intersection(record_terms)) / max(len(query_terms), 1)
                if query_terms
                else 0.0
            )
            age_days = max(0.0, (current_time - record.created_at) / 86400)
            recency = math.exp(-age_days / 30)
            task_match = 1.0 if task_type and record.task_type == task_type else 0.0
            partition_bonus = 0.08 if record.partition == "procedural" and task_match else 0.0
            score = (
                0.50 * relevance
                + 0.18 * recency
                + 0.17 * task_match
                + 0.15 * record.importance
                + partition_bonus
            )
            if relevance <= 0 and task_match <= 0:
                continue
            hits.append(
                MemoryHit(
                    record=record,
                    score=score,
                    relevance=relevance,
                    recency=recency,
                    task_match=task_match,
                )
            )
        hits.sort(key=lambda item: (item.score, item.record.created_at), reverse=True)
        return hits[: max(0, limit)]

    def stats(self) -> dict[str, int]:
        with self._lock:
            counts = {name: 0 for name in ("episodic", "semantic", "procedural")}
            for record in self._records.values():
                counts[record.partition] += 1
            return {
                **counts,
                "working_sessions": len(self._working),
                "conversation_sessions": len(self._sessions),
            }
