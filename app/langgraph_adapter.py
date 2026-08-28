"""LangGraph adapter for the RepoPilot workflow.

Keeps the existing workflow engine intact while exposing the same
Planner -> Retrieve -> Verify -> Tool -> Reflect style graph through the
LangGraph runtime.
"""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


class RepoPilotGraphState(TypedDict, total=False):
    query: str
    plan: dict[str, Any]
    evidence: list[Any]
    observations: list[Any]
    reflection: dict[str, Any]
    answer: str


def build_langgraph_workflow(
    planner,
    retriever,
    verifier,
    tool_executor,
    reflector,
    synthesizer,
):
    """Build a LangGraph version of the engineering diagnosis flow."""

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
        lambda state: "tool" if state.get("evidence") else "retrieve",
        {"tool": "tool", "retrieve": "retrieve"},
    )
    graph.add_edge("tool", "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda state: "synthesize" if state.get("reflection", {}).get("done") else "tool",
        {"synthesize": "synthesize", "tool": "tool"},
    )
    graph.add_edge("synthesize", END)

    return graph.compile()
