"""Unit tests for triage graph nodes and decision policy."""

from __future__ import annotations

from triage_agent.utils.nodes import (
    decide,
    decide_node,
    ingest,
)


def test_ingest_formats_messages() -> None:
    ticket = {
        "id": "t1",
        "messages": [
            ("1h ago", "First message"),
            ("just now", "Second message"),
        ],
    }
    text = ingest(ticket)
    assert "[1h ago] First message" in text
    assert "[just now] Second message" in text


def test_decide_node_fallback_profile_only() -> None:
    """Fallback fetches the customer profile only — no billing/status/KB calls."""
    state = {
        "fallback": True,
        "ticket": {"customer_id": "cust_pro_123", "messages": [("now", "test")]},
        "ticket_text": "[now] just a friendly hello",
        "messages": [],
    }
    out = decide_node(state)
    tools_out = out["tool_outputs"]
    assert tools_out["kb_ids"] == []
    assert "profile only" in tools_out["billing"]
    assert "profile only" in tools_out["status"]
    assert out["result"].next_action == "escalate to human"
    assert out["result"].confidence == 0.5
    # Profile is quoted into reasoning for the human handoff.
    assert "plan=Pro" in out["result"].reasoning


def test_decide_node_fallback_unknown_customer() -> None:
    """Fallback with an unknown id still escalates with a readable profile note."""
    state = {
        "fallback": True,
        "ticket": {"customer_id": "ghost", "messages": [("now", "test")]},
        "ticket_text": "[now] test",
        "messages": [],
    }
    out = decide_node(state)
    assert out["result"].next_action == "escalate to human"
    assert "Customer not found" in out["result"].reasoning


def test_decide_degraded_fallback() -> None:
    analysis = {"fallback": True, "reasoning": "LLM failed"}
    tool_outputs = {"kb_ids": ["kb-1"], "billing": "3 charges", "status": "ok"}
    result = decide(analysis, tool_outputs)
    assert result.next_action == "escalate to human"
    assert result.confidence == 0.5
    assert result.kb_refs == ["kb-1"]


def test_decide_trusts_llm_on_outage_signals() -> None:
    """LLM decides: tool outage signals alone no longer force escalation."""
    analysis = {
        "urgency": "critical",
        "issue_types": ["outage"],
        "reasoning": "System is unreachable",
        "next_action": "route to specialist",
    }
    tool_outputs = {
        "kb_ids": ["kb-outage-500"],
        "status": "Region asia: degraded. HTTP 500 on app servers",
        "billing": "skipped",
    }
    result = decide(analysis, tool_outputs, ticket_text="error 500 everywhere")
    assert result.next_action == "route to specialist"


def test_decide_trusts_llm_on_billing_deadline() -> None:
    """LLM decides: billing+deadline hints alone no longer force escalation."""
    analysis = {
        "urgency": "critical",
        "issue_types": ["billing"],
        "reasoning": "Customer needs presentation export",
        "next_action": "route to specialist",
    }
    tool_outputs = {
        "kb_ids": ["kb-billing-duplicate"],
        "status": "skipped",
        "billing": "Billing for cust_1: 3 charge(s), $89.97 pending/unrefunded",
    }
    ticket_text = "I have a presentation in 2 hours and 3 charges pending"
    result = decide(analysis, tool_outputs, ticket_text=ticket_text)
    assert result.next_action == "route to specialist"


def test_decide_trusts_llm_on_dispute_threat() -> None:
    """LLM decides: dispute wording alone no longer forces escalation."""
    analysis = {
        "urgency": "high",
        "issue_types": ["billing"],
        "reasoning": "Customer is threatening dispute",
        "next_action": "route to specialist",
    }
    tool_outputs = {"kb_ids": [], "status": "skipped", "billing": "skipped"}
    ticket_text = "If not refunded, I am disputing all charges with my bank."
    result = decide(analysis, tool_outputs, ticket_text=ticket_text)
    assert result.next_action == "route to specialist"


def test_decide_ignores_negated_dispute_reasoning() -> None:
    """Verifies fix for false-positive escalation when reasoning says 'no dispute'."""
    analysis = {
        "urgency": "low",
        "issue_types": ["feature-request"],
        "reasoning": "There are no urgent issues or disputes involved. Friendly inquiry.",
        "next_action": "auto-respond",
    }
    tool_outputs = {"kb_ids": ["kb-dark-mode"], "status": "skipped", "billing": "skipped"}
    ticket_text = "Can you add dark mode schedule? No rush!"
    result = decide(analysis, tool_outputs, ticket_text=ticket_text)
    assert result.next_action == "auto-respond"
    assert result.confidence == 0.9


def test_decide_routes_to_specialist() -> None:
    analysis = {
        "urgency": "medium",
        "issue_types": ["integration"],
        "reasoning": "Customer asking for API webhook configuration",
        "next_action": "route to specialist",
    }
    tool_outputs = {"kb_ids": [], "status": "skipped", "billing": "skipped"}
    ticket_text = "How do I configure webhooks for our staging environment?"
    result = decide(analysis, tool_outputs, ticket_text=ticket_text)
    assert result.next_action == "route to specialist"
    assert result.confidence == 0.8


def test_decide_routes_bugs_to_specialist() -> None:
    analysis = {
        "urgency": "medium",
        "issue_types": ["bug", "feature-request"],
        "reasoning": "Customer reports display bug where app does not follow macOS dark mode",
        "next_action": "route to specialist",
    }
    tool_outputs = {"kb_ids": ["kb-dark-mode"], "status": "skipped", "billing": "skipped"}
    ticket_text = "Is this a bug or am I missing something?"
    result = decide(analysis, tool_outputs, ticket_text=ticket_text)
    assert result.next_action == "route to specialist"
    assert result.confidence == 0.8



def test_sentiment_normalization() -> None:
    from triage_agent.utils.state import TriageResult

    assert TriageResult(urgency="low", sentiment="pos").sentiment == "positive"
    assert TriageResult(urgency="low", sentiment="positive").sentiment == "positive"
    assert TriageResult(urgency="low", sentiment="neg").sentiment == "negative"
    assert TriageResult(urgency="low", sentiment="negative").sentiment == "negative"
    assert TriageResult(urgency="low", sentiment="neu").sentiment == "neutral"
    assert TriageResult(urgency="low", sentiment="nue").sentiment == "neutral"
    # Clearly valenced adjectives map; ambiguous ones stay neutral.
    assert TriageResult(urgency="low", sentiment="furious").sentiment == "negative"
    assert TriageResult(urgency="low", sentiment="friendly").sentiment == "neutral"


def test_decide_node_fallback() -> None:
    from triage_agent.utils.nodes import decide_node

    state = {
        "fallback": True,
        "ticket": {"customer_id": "cust_free_001", "messages": [("now", "test")]},
        "ticket_text": "[now] test",
        "messages": [],
    }
    out = decide_node(state)
    assert out["result"].next_action == "escalate to human"
    assert out["result"].confidence == 0.5


def test_decide_node_with_tool_messages() -> None:
    from langchain_core.messages import AIMessage, ToolMessage
    from unittest.mock import patch
    from triage_agent.utils.nodes import decide_node

    msgs = [
        ToolMessage(content="Billing for 'c1': 3 charge(s), $89.97 pending", name="check_billing", tool_call_id="call_1"),
        ToolMessage(content="Region asia: degraded. HTTP 500", name="check_system_status", tool_call_id="call_2"),
        ToolMessage(content="KB matches: kb-1: Title 1 | kb-2: Title 2", name="search_knowledge_base", tool_call_id="call_3"),
        AIMessage(content="Analysis complete"),
    ]
    state = {
        "ticket": {"customer_id": "c1", "messages": []},
        "ticket_text": "Need presentation in 2 hours error 500",
        "messages": msgs,
    }
    with patch("triage_agent.utils.nodes.get_llm", return_value=None):
        out = decide_node(state)
        assert out["result"].next_action == "escalate to human"
        assert "kb-1" in out["result"].kb_refs


def test_agent_node_fallback() -> None:
    from unittest.mock import patch
    from triage_agent.utils.nodes import agent_node

    with patch("triage_agent.utils.nodes.get_llm", return_value=None):
        out = agent_node({"messages": []})
        assert out["fallback"] is True


