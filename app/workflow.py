from __future__ import annotations

import copy
import operator
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START
from langgraph.graph import StateGraph as LangGraphStateGraph
from langgraph.types import Command, interrupt

from .rag_engine import SearchHit


@dataclass
class AgentState:
    """Business state shared by RepoPilot's LangGraph nodes."""

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
    thread_id: str


class WorkflowState(TypedDict):
    agent_state: AgentState
    path: Annotated[list[str], operator.add]
    events: Annotated[list[GraphEvent], operator.add]
    breakpoints: list[str]


Node = Callable[[AgentState], None]
Router = Callable[[AgentState], str]
EventSink = Callable[[GraphEvent], None]


_event_sink: ContextVar[EventSink | None] = ContextVar("repopilot_event_sink", default=None)


def _constant_router(next_node: str) -> Router:
    def _route(_state: AgentState) -> str:
        return next_node

    return _route


class StateGraph:
    """RepoPilot facade over the official LangGraph runtime.

    The facade keeps RepoPilot's existing mutating business nodes small while
    LangGraph owns conditional execution, recursion limits, checkpointing and
    interrupt/resume behavior. A fresh state copy is written after each node so
    every checkpoint represents a stable workflow boundary.
    """

    framework = "langgraph"

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._routers: dict[str, Router] = {}
        self._entrypoint = ""
        self._compiled: Any = None
        self._compile_lock = threading.Lock()
        self._checkpointer = InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    ("app.workflow", "AgentState"),
                    ("app.workflow", "GraphEvent"),
                    ("app.agent", "ToolCall"),
                    ("app.rag_engine", "SearchHit"),
                    ("app.rag_engine", "Chunk"),
                ]
            )
        )

    def add_node(self, name: str, node: Node) -> None:
        if name == END:
            raise ValueError(f"{END} is reserved")
        self._nodes[name] = node
        self._compiled = None

    def add_edge(self, source: str, target: str | Router) -> None:
        self._routers[source] = target if callable(target) else _constant_router(target)
        self._compiled = None

    def set_entrypoint(self, name: str) -> None:
        self._entrypoint = name
        self._compiled = None

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
        thread_id: str | None = None,
    ) -> GraphRun:
        """Invoke or resume the compiled graph.

        The legacy cursor arguments remain for API compatibility. A resumed run
        uses LangGraph's saved cursor rather than manually jumping to a node.
        """

        if not self._entrypoint:
            raise RuntimeError("workflow entrypoint is not configured")
        completed_transitions = len(prior_path or []) if start_at is not None else 0
        remaining_transitions = max_transitions - completed_transitions
        if remaining_transitions <= 0:
            state.stop_reason = "graph_transition_limit"
            raise RuntimeError(f"workflow exceeded {max_transitions} transitions")

        compiled = self._compile()
        resolved_thread_id = thread_id or str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": resolved_thread_id},
            # LangGraph counts the terminal superstep in addition to executed
            # nodes. Add one so max_transitions keeps its public meaning.
            "recursion_limit": remaining_transitions + 1,
        }
        active_breakpoints = sorted(breakpoints or set())
        if start_at is None:
            payload: WorkflowState | Command = {
                "agent_state": copy.deepcopy(state),
                "path": list(prior_path or []),
                "events": list(prior_events or []),
                "breakpoints": active_breakpoints,
            }
        else:
            if not thread_id:
                raise ValueError("resuming a LangGraph run requires thread_id")
            payload = Command(
                resume={"action": "continue", "node": skip_breakpoint_once or start_at},
                update={"breakpoints": active_breakpoints},
            )

        token = _event_sink.set(event_sink)
        try:
            for update in compiled.stream(payload, config, stream_mode="updates"):
                if "__interrupt__" in update:
                    continue
                for node_update in update.values():
                    if not isinstance(node_update, dict):
                        continue
                    for event in node_update.get("events", []):
                        if event_sink:
                            event_sink(event)
        except GraphRecursionError as exc:
            state.stop_reason = "graph_transition_limit"
            raise RuntimeError(f"workflow exceeded {max_transitions} transitions") from exc
        finally:
            _event_sink.reset(token)

        snapshot = compiled.get_state(config)
        values = snapshot.values
        current_state = values.get("agent_state", state)
        self._copy_state(current_state, state)
        path = list(values.get("path", []))
        events = list(values.get("events", []))
        next_node = snapshot.next[0] if snapshot.next else None
        status = "paused" if next_node else "completed"
        return GraphRun(
            path=path,
            transitions=len(path),
            status=status,
            next_node=next_node,
            events=events,
            thread_id=resolved_thread_id,
        )

    def cancel(self, thread_id: str) -> None:
        """Delete a paused thread and all of its in-memory checkpoints."""

        self._checkpointer.delete_thread(thread_id)

    def _compile(self):
        if self._compiled is not None:
            return self._compiled
        with self._compile_lock:
            if self._compiled is not None:
                return self._compiled
            if self._entrypoint not in self._nodes:
                raise RuntimeError(f"workflow node is not configured: {self._entrypoint}")

            builder = LangGraphStateGraph(WorkflowState)
            for name, node in self._nodes.items():
                builder.add_node(name, self._wrap_node(name, node))
            builder.add_edge(START, self._entrypoint)
            for source, router in self._routers.items():
                builder.add_conditional_edges(
                    source,
                    lambda workflow_state, route=router: route(
                        workflow_state["agent_state"]
                    ),
                )
            self._compiled = builder.compile(checkpointer=self._checkpointer)
        return self._compiled

    def _wrap_node(self, name: str, node: Node):
        router = self._routers.get(name)

        def _wrapped(workflow_state: WorkflowState) -> dict[str, Any]:
            if name in workflow_state.get("breakpoints", []):
                interrupt({"reason": "breakpoint", "node": name})

            current = copy.deepcopy(workflow_state["agent_state"])
            started = time.perf_counter()
            try:
                node(current)
            except Exception as exc:
                event = GraphEvent(
                    sequence=len(workflow_state.get("events", [])) + 1,
                    node=name,
                    status="failed",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    next_node=None,
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
                sink = _event_sink.get()
                if sink:
                    sink(event)
                raise

            if router is None:
                raise RuntimeError(f"workflow edge is not configured: {name}")
            target = router(current)
            status = "completed"
            if name == "rewrite_query":
                status = "retry"
            elif name == "verify_evidence" and current.evidence_quality == "low":
                status = "rejected"
            elif name == "reflect" and current.reflection.get("decision") == "continue":
                status = "retry"
            event = GraphEvent(
                sequence=len(workflow_state.get("events", [])) + 1,
                node=name,
                status=status,
                latency_ms=int((time.perf_counter() - started) * 1000),
                next_node=None if target == END else target,
                detail={
                    "attempt": current.retrieval_attempts,
                    "evidence_quality": current.evidence_quality,
                    "stop_reason": current.stop_reason,
                    "selected_skill": current.selected_skill,
                    "react_steps": current.react_steps,
                },
            )
            return {"agent_state": current, "path": [name], "events": [event]}

        _wrapped.__name__ = f"repopilot_{name}"
        return _wrapped

    @staticmethod
    def _copy_state(source: AgentState, target: AgentState) -> None:
        for item in fields(AgentState):
            setattr(target, item.name, copy.deepcopy(getattr(source, item.name)))
