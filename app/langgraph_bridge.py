"""LangGraph execution bridge for RepoPilot.

Keeps the existing KnowledgeAgent nodes reusable while exposing a LangGraph
workflow for Agent applications.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


class RepoPilotGraphState(TypedDict, total=False):
    query: str
    context: list[Any]
    observations: list[Any]
    answer: str


def build_langgraph_bridge(agent: Any):
    """Build a LangGraph workflow backed by existing RepoPilot capabilities."""

    graph = StateGraph(RepoPilotGraphState)

    def retrieve(state: RepoPilotGraphState):
        result = agent.kb.build_context(state["query"], top_k=5)
        return {"context": result.hits}

    def synthesize(state: RepoPilotGraphState):
        answer = agent._synthesize_answer(state["query"], state.get("context", []))
        return {"answer": answer}

    graph.add_node("retrieve", retrieve)
    graph.add_node("synthesize", synthesize)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
