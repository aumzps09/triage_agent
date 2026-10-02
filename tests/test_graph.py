"""Unit tests for triage graph compilation and end-to-end execution."""

from __future__ import annotations

from unittest.mock import patch

from triage_agent.agent import build_graph, run_ticket
from triage_agent.sample_tickets import SAMPLE_TICKETS
from triage_agent.utils.state import TriageResult


def test_build_graph_returns_runnable() -> None:
    app = build_graph()
    assert hasattr(app, "invoke")


def test_run_ticket_fallback_mode() -> None:
    """Without LLM, all tickets fail-closed to escalate to human."""
    with patch("triage_agent.utils.nodes.get_llm", return_value=None):
        for ticket in SAMPLE_TICKETS:
            result = run_ticket(ticket)
            assert isinstance(result, TriageResult)
            assert result.next_action == "escalate to human"
            assert result.confidence == 0.5


def test_react_routing_and_nodes() -> None:
    """Verify route_after_agent and node execution."""
    from langchain_core.messages import AIMessage, ToolCall
    from triage_agent.utils.nodes import agent_node, ingest_node, route_after_agent

    # Test ingest_node
    state = {"ticket": SAMPLE_TICKETS[0]}
    ingest_out = ingest_node(state)
    assert "messages" in ingest_out
    assert len(ingest_out["messages"]) == 2

    # Test route_after_agent with tool call
    call = ToolCall(name="check_billing", args={"customer_id": "c1"}, id="call_1")
    msg_with_tool = AIMessage(content="", tool_calls=[call])
    assert route_after_agent({"messages": [msg_with_tool]}) == "tools"

    # Test route_after_agent without tool call
    msg_no_tool = AIMessage(content="Final assessment")
    assert route_after_agent({"messages": [msg_no_tool]}) == "decide"

    # Test fallback routing
    assert route_after_agent({"fallback": True, "messages": [msg_with_tool]}) == "decide"


def test_main_cli_execution(capsys) -> None:  # type: ignore[no-untyped-def]
    from triage_agent.__main__ import main

    with patch("triage_agent.utils.nodes.get_llm", return_value=None):
        main()
    captured = capsys.readouterr()
    assert "ticket-1-billing" in captured.out
    assert "ticket-2-outage" in captured.out
    assert "ticket-3-darkmode" in captured.out


