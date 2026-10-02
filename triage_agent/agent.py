"""Code for constructing the triage graph: ingest -> analyze -> retrieve -> decide."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .utils.nodes import analyze_node, decide_node, ingest_node, retrieve_node
from .utils.state import TicketState, TriageResult


def build_graph():  # type: ignore[no-untyped-def]
    """Linear START -> ingest -> analyze -> retrieve -> decide -> END."""
    graph = StateGraph(TicketState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decide", decide_node)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_edge("decide", END)
    return graph.compile()


app = build_graph()


def run_ticket(ticket: dict) -> TriageResult:
    """End-to-end via the compiled graph: ingest -> analyze -> retrieve -> decide."""
    final = app.invoke({"ticket": ticket})
    result = final["result"]
    return result if isinstance(result, TriageResult) else TriageResult(**dict(result))
