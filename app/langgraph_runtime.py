"""Runtime bridge between RepoPilot workflow nodes and LangGraph.

The existing workflow keeps its business logic. This module only adapts
node callables into LangGraph compatible state transitions.
"""

from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph


class RepoPilotState(TypedDict, total=False):
    query: str
    plan: dict[str, Any]
    context: list[Any]
    evidence: list[Any]
    observations: list[Any]
    reflection: dict[str, Any]
    answer: str
    error: str


Node = Callable[[RepoPilotState], RepoPilotState]


def build_runtime(
    planner: Node,
    retriever: Node,
    verifier: Node,
    executor: Node,
    reflector: Node,
    responder: Node,
):
    graph = StateGraph(RepoPilotState)

    graph.add_node("planner", planner)
    graph.add_node("retriever", retriever)
    graph.add_node("verifier", verifier)
    graph.add_node("executor", executor)
    graph.add_node("reflector", reflector)
    graph.add_node("responder", responder)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "verifier")

    graph.add_conditional_edges(
        "verifier",
        lambda state: "executor" if state.get("evidence") else "retriever",
        {
            "executor": "executor",
            "retriever": "retriever",
        },
    )

    graph.add_edge("executor", "reflector")
    graph.add_conditional_edges(
        "reflector",
        lambda state: "responder"
        if state.get("reflection", {}).get("finished")
        else "executor",
        {
            "responder": "responder",
            "executor": "executor",
        },
    )

    graph.add_edge("responder", END)
    return graph.compile()


def invoke_runtime(graph, query: str, **kwargs: Any) -> dict[str, Any]:
    state: RepoPilotState = {"query": query, **kwargs}
    return dict(graph.invoke(state))
