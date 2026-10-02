"""Code for constructing the triage ReAct graph: ingest -> agent <-> tools -> decide -> END."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .utils.nodes import agent_node, decide_node, ingest_node, route_after_agent
from .utils.state import TicketState, TriageResult
from .utils.tools import as_langchain_tools


def build_graph():  # type: ignore[no-untyped-def]
    """ReAct StateGraph: ingest -> agent <-> tools -> decide -> END."""
    graph = StateGraph(TicketState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(as_langchain_tools()))
    graph.add_node("decide", decide_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "decide": "decide",
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("decide", END)
    return graph.compile()


app = build_graph()


def run_ticket(ticket: dict, recursion_limit: int = 15) -> TriageResult:
    """End-to-end via the compiled graph: ingest -> agent <-> tools -> decide."""
    final = app.invoke({"ticket": ticket}, config={"recursion_limit": recursion_limit})
    result = final["result"]
    return result if isinstance(result, TriageResult) else TriageResult(**dict(result))
