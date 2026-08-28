"""LangGraph runtime adapter for RepoPilot.

The existing workflow implementation remains the source of node logic. This
module only provides a LangGraph execution layer so RepoPilot can expose a
standard Agent workflow with explicit state transitions.
"""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


class RepoPilotGraphState(TypedDict, total=False):
    query: str
    plan: dict[str, Any]
    evidence: list[Any]
    observations: list[Any]
    tool_results: list[Any]
    reflection: dict[str, Any]
    answer: str
    error: str


def _verify_route(state: RepoPilotGraphState) -> str:
    """Route retrieval failures back to retrieval instead of hallucinating."""

    return "tool" if state.get("evidence") else "retrieve"


def _reflection_route(state: RepoPilotGraphState) -> str:
    """Continue tool investigation until reflection marks the task complete."""

    reflection = state.get("reflection", {})
    return "synthesize" if reflection.get("done") else "tool"


def build_langgraph_workflow(
    planner,
    retriever,
    verifier,
    tool_executor,
    reflector,
    synthesizer,
):
    """Build RepoPilot's Planner-Retrieve-Tool-Reflection graph."""

    graph = StateGraph(RepoPilotGraphState)

    graph.add_node("planner", planner)
    graph.add_node("retrieve", retriever)
    graph.add_node("verify", verifier)
    graph.add_node("tool", tool_executor)
    graph.add_node("reflect", reflector)
    graph.add_node("synthesize", synthesizer)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "verify")

    graph.add_conditional_edges(
        "verify",
        _verify_route,
        {
            "tool": "tool",
            "retrieve": "retrieve",
        },
    )

    graph.add_edge("tool", "reflect")

    graph.add_conditional_edges(
        "reflect",
        _reflection_route,
        {
            "synthesize": "synthesize",
            "tool": "tool",
        },
    )

    graph.add_edge("synthesize", END)

    return graph.compile()


def run_langgraph_agent(graph, query: str, **kwargs: Any) -> dict[str, Any]:
    """Run RepoPilot through LangGraph with extensible runtime options."""

    initial_state: RepoPilotGraphState = {
        "query": query,
        "evidence": [],
        "observations": [],
        "tool_results": [],
    }
    initial_state.update(kwargs)

    result = graph.invoke(initial_state)
    return dict(result)
