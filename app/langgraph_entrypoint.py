"""LangGraph execution entrypoint for RepoPilot.

Keeps existing agent business logic intact while exposing a LangGraph based
runtime boundary.
"""

from typing import Any, Callable

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


class RepoPilotRuntimeState(TypedDict, total=False):
    query: str
    evidence: list[Any]
    observations: list[Any]
    answer: str
    reflection: dict[str, Any]


Node = Callable[[RepoPilotRuntimeState], RepoPilotRuntimeState]


def build_repopilot_graph(
    planner: Node,
    retrieve: Node,
    verify: Node,
    execute: Node,
    reflect: Node,
    answer: Node,
):
    graph = StateGraph(RepoPilotRuntimeState)

    graph.add_node("planner", planner)
    graph.add_node("retrieve", retrieve)
    graph.add_node("verify", verify)
    graph.add_node("execute", execute)
    graph.add_node("reflect", reflect)
    graph.add_node("answer", answer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "verify")

    graph.add_conditional_edges(
        "verify",
        lambda state: "execute" if state.get("evidence") else "retrieve",
    )

    graph.add_edge("execute", "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda state: "answer"
        if state.get("reflection", {}).get("finished")
        else "execute",
    )

    graph.add_edge("answer", END)
    return graph.compile()


def run(graph, query: str) -> dict[str, Any]:
    return dict(graph.invoke({"query": query}))
