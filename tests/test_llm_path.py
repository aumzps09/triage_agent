"""LLM-path tests with a fake ChatModel + adversarial Thai paraphrases.

The fake implements only the surface `nodes.py` touches:
`bind_tools(...).invoke(...)` for the ReAct loop and
`with_structured_output(...).invoke(...)` for the finalizer.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from triage_agent.agent import run_ticket
from triage_agent.utils.nodes import decide, decide_node
from triage_agent.utils.state import TriageResult


class _FakeLLM:
    """Minimal ChatModel double for agent_node + decide_node."""

    def __init__(self, final: TriageResult):
        self._final = final

    def bind_tools(self, _tools):  # type: ignore[no-untyped-def]
        return RunnableLambda(lambda _msgs: AIMessage(content="assessment done"))

    def with_structured_output(self, _schema, method=None):  # type: ignore[no-untyped-def]
        return RunnableLambda(lambda _x: self._final)


def _run_with_fake(ticket: dict, final: TriageResult) -> TriageResult:
    with patch(
        "triage_agent.utils.nodes.get_llm", return_value=_FakeLLM(final)
    ):
        return run_ticket(ticket)


def _billing_ticket() -> dict:
    return {
        "id": "ticket-1-billing",
        "customer_id": "cust_free_001",
        "messages": [
            ("1h ago", "THREE charges of $29.99, none refunded, no Pro access."),
            ("just now", "Presentation in 2 hours, disputing today."),
        ],
    }


# ---------- LLM path end-to-end ----------


def test_llm_path_billing_escalates() -> None:
    final = TriageResult(
        urgency="critical",
        product="billing",
        issue_types=["billing", "access"],
        sentiment="negative",
        kb_refs=[],
        next_action="escalate to human",
        reasoning="Triple charge with deadline and dispute threat.",
        confidence=0.95,
    )
    result = _run_with_fake(_billing_ticket(), final)
    assert result.next_action == "escalate to human"
    assert result.urgency == "critical"
    assert result.confidence == 0.95
    # KB grounding still applied on the live path.
    assert "kb-billing-duplicate" in result.kb_refs


def test_llm_path_bug_routes_to_specialist() -> None:
    final = TriageResult(
        urgency="medium",
        product="appearance",
        issue_types=["bug", "feature-request"],
        sentiment="neutral",
        kb_refs=[],
        next_action="route to specialist",
        reasoning="macOS System Default display bug plus schedule request.",
        confidence=0.8,
    )
    ticket = {
        "id": "ticket-3-darkmode",
        "customer_id": "cust_pro_123",
        "messages": [("today", "System Default stays light on dark macOS.")],
    }
    result = _run_with_fake(ticket, final)
    assert result.next_action == "route to specialist"
    assert result.confidence == 0.8
    assert "kb-dark-mode" in result.kb_refs


def test_llm_structured_failure_falls_back() -> None:
    """Schema violation in the finalizer must fail closed, not crash."""

    class _BrokenLLM(_FakeLLM):
        def with_structured_output(self, _schema, method=None):  # type: ignore[no-untyped-def]
            return RunnableLambda(lambda _x: 1 / 0)

    with patch("triage_agent.utils.nodes.get_llm", return_value=_BrokenLLM(None)):  # type: ignore[arg-type]
        result = run_ticket(_billing_ticket())
    assert result.next_action == "escalate to human"
    assert result.confidence == 0.5


# ---------- Adversarial Thai paraphrases (policy level, no LLM) ----------


def test_thai_outage_paraphrase_trusts_llm() -> None:
    """LLM decides: outage wording alone no longer forces escalation."""
    analysis = {
        "urgency": "critical",
        "issue_types": ["outage"],
        "reasoning": "ภูมิภาค Asia ใช้งานไม่ได้",
        "next_action": "route to specialist",
    }
    tool_outputs = {
        "kb_ids": ["kb-outage-500"],
        "status": "Region asia: degraded. HTTP 500 on app servers.",
        "billing": "skipped",
    }
    ticket_text = "เซิร์ฟเวอร์ล่มครับ ขึ้น error 500 มี demo บ่ายนี้"
    result = decide(analysis, tool_outputs, ticket_text=ticket_text)
    assert result.next_action == "route to specialist"


def test_thai_billing_paraphrase_trusts_llm() -> None:
    """LLM decides: billing+deadline hints alone no longer force escalation."""
    analysis = {
        "urgency": "high",
        "issue_types": ["billing"],
        "reasoning": "ลูกค้าถูกตัดเงินซ้ำ",
        "next_action": "route to specialist",
    }
    tool_outputs = {
        "kb_ids": ["kb-billing-duplicate"],
        "status": "skipped",
        "billing": "Billing for 'c': 3 charge(s), $89.97 pending/unrefunded",
    }
    ticket_text = "ตัดเงินไป 3 รอบแล้วยังไม่ได้คืนเงินเลย มีพรีเซนต์บ่ายนี้ครับ"
    result = decide(analysis, tool_outputs, ticket_text=ticket_text)
    assert result.next_action == "route to specialist"


def test_order_number_not_confused_with_500() -> None:
    """Word-boundary: '#5001' must not trigger the outage rule."""
    analysis = {
        "urgency": "low",
        "issue_types": ["question"],
        "reasoning": "ถามสถานะคำสั่งซื้อ",
        "next_action": "auto-respond",
    }
    tool_outputs = {
        "kb_ids": [],
        "status": "Region us: operational.",
        "billing": "skipped",
    }
    result = decide(
        analysis, tool_outputs, ticket_text="Where is order #5001?"
    )
    assert result.next_action == "auto-respond"


def test_chargeback_trusts_llm() -> None:
    """LLM decides: chargeback wording alone no longer forces escalation."""
    analysis = {
        "urgency": "high",
        "issue_types": ["billing"],
        "reasoning": "Customer unhappy",
        "next_action": "route to specialist",
    }
    tool_outputs = {"kb_ids": [], "status": "skipped", "billing": "skipped"}
    result = decide(
        analysis,
        tool_outputs,
        ticket_text="I will chargeback this payment tomorrow.",
    )
    assert result.next_action == "route to specialist"


def test_thai_urgent_deadline_trusts_llm() -> None:
    """LLM decides: urgent deadline wording alone no longer forces escalation."""
    analysis = {
        "urgency": "high",
        "issue_types": ["billing"],
        "reasoning": "ลูกค้าถูกตัดเงินซ้ำ",
        "next_action": "route to specialist",
    }
    tool_outputs = {
        "kb_ids": ["kb-billing-duplicate"],
        "status": "skipped",
        "billing": "Billing for 'c': 2 charge(s), $59.98 pending/unrefunded",
    }
    result = decide(
        analysis, tool_outputs, ticket_text="ตัดเงินซ้ำครับ ด่วนมาก ต้องใช้วันนี้"
    )
    assert result.next_action == "route to specialist"


def test_thai_friendly_no_false_escalation() -> None:
    """Polite Thai must not escalate: no dispute/outage/deadline signals."""
    analysis = {
        "urgency": "low",
        "issue_types": ["feature-request"],
        "reasoning": "คำถามทั่วไป ไม่มีความขัดข้อง",
        "next_action": "auto-respond",
    }
    tool_outputs = {"kb_ids": ["kb-feature-request"], "status": "skipped", "billing": "skipped"}
    result = decide(
        analysis,
        tool_outputs,
        ticket_text="ขอบคุณมากครับ อยากทราบว่ามีวิธีตั้งเวลาสลับธีมไหม ไม่รีบครับ 😊",
    )
    assert result.next_action == "auto-respond"


def test_thai_outage_ticket_fallback_profile() -> None:
    """Fallback on the Thai outage thread cites the Enterprise profile only."""
    from triage_agent.sample_tickets import SAMPLE_TICKETS

    ticket = next(t for t in SAMPLE_TICKETS if t["id"] == "ticket-2-outage")
    text = "\n".join(f"[{t}] {m}" for t, m in ticket["messages"])
    state = {
        "fallback": True,
        "ticket": ticket,
        "ticket_text": text,
        "messages": [],
    }
    out = decide_node(state)
    assert out["result"].next_action == "escalate to human"
    assert out["result"].kb_refs == []
    assert "plan=Enterprise" in out["result"].reasoning
