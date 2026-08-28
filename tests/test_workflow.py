from __future__ import annotations

from app.workflow import END, AgentState, StateGraph


def test_state_graph_records_conditional_path() -> None:
    state = AgentState(query="q", normalized_query="q", retrieval_query="q", top_k=3, trace=[])
    graph = StateGraph()
    graph.add_node("route", lambda current: setattr(current, "intent", "rag_answer"))
    graph.add_node("finish", lambda current: setattr(current, "answer", "done"))
    graph.set_entrypoint("route")
    graph.add_edge("route", lambda current: "finish" if current.intent else END)
    graph.add_edge("finish", END)

    run = graph.run(state)

    assert run.path == ["route", "finish"]
    assert state.answer == "done"


def test_state_graph_enforces_transition_limit() -> None:
    state = AgentState(query="q", normalized_query="q", retrieval_query="q", top_k=3, trace=[])
    graph = StateGraph()
    graph.add_node("loop", lambda _state: None)
    graph.set_entrypoint("loop")
    graph.add_edge("loop", "loop")

    try:
        graph.run(state, max_transitions=2)
    except RuntimeError as exc:
        assert "exceeded" in str(exc)
    else:
        raise AssertionError("expected the transition guard to stop the loop")
    assert state.stop_reason == "graph_transition_limit"


def test_state_graph_pauses_and_resumes_at_breakpoint() -> None:
    state = AgentState(query="q", normalized_query="q", retrieval_query="q", top_k=3, trace=[])
    graph = StateGraph()
    graph.add_node("first", lambda current: setattr(current, "intent", "routed"))
    graph.add_node("second", lambda current: setattr(current, "answer", "done"))
    graph.set_entrypoint("first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    paused = graph.run(state, breakpoints={"second"})
    resumed = graph.run(
        state,
        start_at=paused.next_node,
        prior_path=paused.path,
        prior_events=paused.events,
        breakpoints={"second"},
        skip_breakpoint_once=paused.next_node,
        thread_id=paused.thread_id,
    )

    assert paused.status == "paused"
    assert paused.next_node == "second"
    assert resumed.status == "completed"
    assert resumed.path == ["first", "second"]
    assert state.answer == "done"


def test_state_graph_is_backed_by_langgraph() -> None:
    graph = StateGraph()

    assert graph.framework == "langgraph"
