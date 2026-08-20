from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .rag_engine import SearchHit

END = "__end__"


@dataclass
class AgentState:
    """Mutable state shared by every node in the agent workflow graph."""

    query: str
    normalized_query: str
    top_k: int
    trace: list[Any]
    session_id: str = ""
    intent: str = ""
    tools: list[str] = field(default_factory=list)
    selected_skill: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    memory_hits: list[dict[str, Any]] = field(default_factory=list)
    context_metrics: dict[str, Any] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)
    reflection: dict[str, Any] = field(default_factory=dict)
    react_steps: int = 0
    router_confidence: float = 0.0
    retrieval_query: str = ""
    retrieval_attempts: int = 0
    hits: list[SearchHit] = field(default_factory=list)
    best_hits: list[SearchHit] = field(default_factory=list)
    answer: str = ""
    evidence_quality: str = "unknown"
    stop_reason: str = ""


@dataclass(frozen=True)
class GraphEvent:
    sequence: int
    node: str
    status: str
    latency_ms: int
    next_node: str | None
    detail: dict[str, Any]


@dataclass(frozen=True)
class GraphRun:
    path: list[str]
    transitions: int
    status: str
    next_node: str | None
    events: list[GraphEvent]


Node = Callable[[AgentState], None]
Router = Callable[[AgentState], str]
EventSink = Callable[[GraphEvent], None]


def _constant_router(next_node: str) -> Router:
    """Build a router that always returns the same target node."""

    def _route(_state: AgentState) -> str:
        return next_node

    return _route


class StateGraph:
    """Small dependency-free state graph with explicit, bounded transitions.

    A node mutates shared state. Its router chooses the next node, which makes
    retries and stop conditions visible instead of hiding them in nested code.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._routers: dict[str, Router] = {}
        self._entrypoint = ""

    def add_node(self, name: str, node: Node) -> None:
        if name == END:
            raise ValueError(f"{END} is reserved")
        self._nodes[name] = node

    def add_edge(self, source: str, target: str | Router) -> None:
        if callable(target):
            self._routers[source] = target
        else:
            self._routers[source] = _constant_router(target)

    def set_entrypoint(self, name: str) -> None:
        self._entrypoint = name

    def run(
        self,
        state: AgentState,
        max_transitions: int = 16,
        *,
        start_at: str | None = None,
        prior_path: list[str] | None = None,
        prior_events: list[GraphEvent] | None = None,
        breakpoints: set[str] | None = None,
        skip_breakpoint_once: str | None = None,
        event_sink: EventSink | None = None,
    ) -> GraphRun:
        if not self._entrypoint:
            raise RuntimeError("workflow entrypoint is not configured")

        current = start_at or self._entrypoint
        path = list(prior_path or [])
        events = list(prior_events or [])
        active_breakpoints = breakpoints or set()
        remaining = max_transitions - len(path)
        if remaining <= 0:
            state.stop_reason = "graph_transition_limit"
            raise RuntimeError(f"workflow exceeded {max_transitions} transitions")

        for _ in range(remaining):
            if current in active_breakpoints and current != skip_breakpoint_once:
                event = GraphEvent(
                    sequence=len(events) + 1,
                    node=current,
                    status="paused",
                    latency_ms=0,
                    next_node=current,
                    detail={"reason": "breakpoint"},
                )
                events.append(event)
                if event_sink:
                    event_sink(event)
                return GraphRun(
                    path=path,
                    transitions=len(path),
                    status="paused",
                    next_node=current,
                    events=events,
                )
            skip_breakpoint_once = None
            node = self._nodes.get(current)
            if node is None:
                raise RuntimeError(f"workflow node is not configured: {current}")
            path.append(current)
            started = time.perf_counter()
            try:
                node(state)
            except Exception as exc:
                event = GraphEvent(
                    sequence=len(events) + 1,
                    node=current,
                    status="failed",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    next_node=None,
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
                events.append(event)
                if event_sink:
                    event_sink(event)
                raise

            router = self._routers.get(current)
            if router is None:
                raise RuntimeError(f"workflow edge is not configured: {current}")
            next_node = router(state)
            status = "completed"
            if current == "rewrite_query":
                status = "retry"
            elif current == "verify_evidence" and state.evidence_quality == "low":
                status = "rejected"
            elif current == "reflect" and state.reflection.get("decision") == "revise":
                status = "retry"
            event = GraphEvent(
                sequence=len(events) + 1,
                node=current,
                status=status,
                latency_ms=int((time.perf_counter() - started) * 1000),
                next_node=None if next_node == END else next_node,
                detail={
                    "attempt": state.retrieval_attempts,
                    "evidence_quality": state.evidence_quality,
                    "stop_reason": state.stop_reason,
                    "selected_skill": state.selected_skill,
                    "react_steps": state.react_steps,
                },
            )
            events.append(event)
            if event_sink:
                event_sink(event)
            current = next_node
            if current == END:
                return GraphRun(
                    path=path,
                    transitions=len(path),
                    status="completed",
                    next_node=None,
                    events=events,
                )

        state.stop_reason = "graph_transition_limit"
        raise RuntimeError(f"workflow exceeded {max_transitions} transitions")
