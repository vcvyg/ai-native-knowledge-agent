"""LangGraph execution entrypoint for RepoPilot.

Provides the LangGraph orchestration boundary while keeping existing business
nodes injectable. The existing agent modules can migrate incrementally by
passing their node functions here.
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
    trace: list[dict[str, Any]]


Node = Callable[[RepoPilotRuntimeState], RepoPilotRuntimeState]


def build_repopilot_graph(
    planner: Node,
    retrieve: Node,
    verify: Node,
    execute: Node,
    reflect: Node,
    answer: Node,
):
    """Build RepoPilot's LangGraph workflow.

    Nodes remain injected so existing retrieval/tools/memory implementations
    can be reused without rewriting them into LangChain abstractions.
    """
    graph = StateGraph(RepoPilotRuntimeState)

    for name, node in {
        "planner": planner,
        "retrieve": retrieve,
        "verify": verify,
        "execute": execute,
        "reflect": reflect,
        "answer": answer,
    }.items():
        graph.add_node(name, node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "verify")

    graph.add_conditional_edges(
        "verify",
        lambda state: "execute" if state.get("evidence") else "retrieve",
        {"execute": "execute", "retrieve": "retrieve"},
    )

    graph.add_edge("execute", "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda state: "answer"
        if state.get("reflection", {}).get("finished")
        else "execute",
        {"answer": "answer", "execute": "execute"},
    )

    graph.add_edge("answer", END)
    return graph.compile()


def run(graph, query: str, **kwargs: Any) -> dict[str, Any]:
    """Execute a compiled RepoPilot LangGraph workflow."""
    payload: RepoPilotRuntimeState = {"query": query, **kwargs}
    return dict(graph.invoke(payload))
