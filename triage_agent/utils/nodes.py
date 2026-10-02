"""Node functions for the triage graph: ingest -> analyze -> retrieve -> decide.

`analyze` tries the LangChain structured-output path first
(`triage_prompt | llm.with_structured_output`); if the LLM is unavailable the
degraded fallback marks the analysis as such and `decide` escalates every
ticket to a human (no pretend classification).
"""

from __future__ import annotations

import os
import re
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..prompts import REACT_AGENT_SYSTEM_PROMPT, finalize_prompt, triage_prompt
from . import tools
from .mock_data import CUSTOMERS, SYSTEM_STATUS
from .state import TicketState, TriageResult

_BILLING_HINTS = ("charge", "billing", "refund", "dispute", "pending", "$", "upgrade")
_OUTAGE_HINTS = ("500", "outage", "error", "region", "status", "demo", "เข้าไม่ได้", "โวย")


def _mentions(low: str, hints: tuple[str, ...]) -> bool:
    """Keyword check for tool routing in `retrieve`."""
    return any(h in low for h in hints)


def ingest(ticket: dict) -> str:
    """Join the whole thread (timestamps kept) into one text for analysis."""
    return "\n".join(f"[{t}] {m}" for t, m in ticket["messages"])


def get_llm():  # type: ignore[no-untyped-def]
    """Return the configured ChatModel instance, or None if keys are absent."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    try:
        if provider == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                return None
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"), temperature=0
            )
        else:
            if not os.getenv("OPENAI_API_KEY"):
                return None
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    except Exception:
        return None


def _llm_analyze(ticket_text: str) -> TriageResult | None:
    """Try the LangChain structured-output path; return None on any failure."""
    llm = get_llm()
    if llm is None:
        return None
    try:
        chain = triage_prompt | _with_structured_output_no_afc(llm)
        out = chain.invoke({"ticket_text": ticket_text})
        return out if isinstance(out, TriageResult) else TriageResult(**dict(out))
    except Exception:
        return None


def _with_structured_output_no_afc(llm):  # type: ignore[no-untyped-def]
    """Structured output with SDK automatic function calling disabled."""
    structured = llm.with_structured_output(TriageResult, method="json_schema")
    if llm.__class__.__name__ != "ChatGoogleGenerativeAI":
        return structured
    try:
        from google.genai.types import AutomaticFunctionCallingConfig
    except Exception:
        return structured
    try:
        return structured.bind(
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True)
        )
    except Exception:
        return structured


def _degraded_analyze() -> dict:
    """Degraded fallback (LLM unavailable): no classification, escalate all."""
    return {
        "urgency": "high",
        "product": "unknown",
        "issue_types": ["unknown"],
        "sentiment": "unknown",
        "reasoning": "LLM analysis unavailable; routing to human without classification.",
        "fallback": True,
    }


def analyze(ticket_text: str) -> dict:
    """Classify + extract via LLM; degraded escalate-all if the LLM is unavailable."""
    llm_result = _llm_analyze(ticket_text)
    if llm_result is not None:
        return llm_result.to_dict()
    return _degraded_analyze()


def _extract_region(ticket_text: str, customer_id: str) -> str:
    """Prefer the customer profile region; fall back to thread keywords."""
    profile = CUSTOMERS.get(customer_id.strip(), {})
    region_field = str(profile.get("region", "")).lower()
    for known in sorted(SYSTEM_STATUS["regions"]):
        if known and known in region_field:
            return known
    if "thailand" in region_field or "thai" in ticket_text.lower():
        return "asia"
    low = ticket_text.lower()
    for known in sorted(SYSTEM_STATUS["regions"]):
        if known in low:
            return known
    return "asia"


def retrieve(ticket_text: str, customer_id: str) -> dict:
    """Call tools directly: profile+KB always, others by keyword."""
    low = ticket_text.lower()
    out = {
        "profile": tools.get_customer_profile.invoke({"customer_id": customer_id}),
        "kb": tools.search_knowledge_base.invoke({"query": ticket_text}),
    }
    out["billing"] = (
        tools.check_billing.invoke({"customer_id": customer_id})
        if _mentions(low, _BILLING_HINTS)
        else "skipped (no billing signals)."
    )
    if _mentions(low, _OUTAGE_HINTS):
        out["status"] = tools.check_system_status.invoke({"region": _extract_region(ticket_text, customer_id)})
    else:
        out["status"] = "skipped (no outage signals)."
    out["kb_ids"] = re.findall(r"kb-[\w-]+", out["kb"])[:2]
    return out


def decide(analysis: dict, tool_outputs: dict, ticket_text: str = "") -> TriageResult:
    """Apply the locked OR escalation policy over analysis + tool evidence."""
    if analysis.get("fallback"):
        return TriageResult(
            urgency=analysis.get("urgency", "high"),
            product=analysis.get("product", "unknown"),
            issue_types=analysis.get("issue_types", ["unknown"]),
            sentiment=analysis.get("sentiment", "unknown"),
            kb_refs=tool_outputs.get("kb_ids", []),
            next_action="escalate to human",
            reasoning=analysis.get("reasoning", "") + f" | evidence: {tool_outputs.get('billing', '')} {tool_outputs.get('status', '')}".strip(),
            confidence=0.5,
        )

    status_text = tool_outputs.get("status", "").lower()
    billing_text = tool_outputs.get("billing", "").lower()
    ticket_lower = ticket_text.lower()
    reasoning_lower = analysis.get("reasoning", "").lower()

    outage_escalation = "degraded" in status_text and ("500" in status_text or "500" in ticket_lower)

    deadline_hints = ("presentation", "deadline", "end of day", "2 hours", "2h", "บ่ายนี้")
    has_billing_pending = "pending" in billing_text or "unrefunded" in billing_text
    billing_deadline = has_billing_pending and any(
        kw in ticket_lower or kw in reasoning_lower for kw in deadline_hints
    )

    is_negated_dispute = any(
        neg in reasoning_lower
        for neg in ("no dispute", "no urgent issues or disputes", "without dispute", "not disput")
    )
    dispute_threat = "disput" in ticket_lower or ("disput" in reasoning_lower and not is_negated_dispute)

    llm_escalate = (
        analysis.get("next_action") == "escalate to human"
        or analysis.get("urgency") == "critical"
    )

    escalate = outage_escalation or billing_deadline or dispute_threat or llm_escalate

    if escalate:
        action, confidence = "escalate to human", 0.95
    elif analysis.get("next_action") == "route to specialist" or any("bug" in str(t).lower() for t in analysis.get("issue_types", [])):
        action, confidence = "route to specialist", 0.8
    elif analysis.get("urgency") == "low" or analysis.get("next_action") == "auto-respond":
        action, confidence = "auto-respond", 0.9
    else:
        action, confidence = "route to specialist", 0.7

    return TriageResult(
        urgency=analysis.get("urgency", "medium"),
        product=analysis.get("product", "general"),
        issue_types=analysis.get("issue_types", []),
        sentiment=analysis.get("sentiment", "neutral"),
        kb_refs=tool_outputs.get("kb_ids", []),
        next_action=action,
        reasoning=analysis.get("reasoning", "") + f" | evidence: {tool_outputs.get('billing', '')} {tool_outputs.get('status', '')}".strip(),
        confidence=confidence,
    )


# --- ReAct Graph Nodes ---


def ingest_node(state: TicketState) -> dict:
    """Format thread and initialize ReAct agent messages."""
    ticket = state["ticket"]
    ticket_text = ingest(ticket)
    cust_id = ticket.get("customer_id", "unknown")
    user_prompt = f"Customer ID: {cust_id}\n\nTicket Messages:\n{ticket_text}"
    return {
        "ticket_text": ticket_text,
        "messages": [
            SystemMessage(content=REACT_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ],
    }


def agent_node(state: TicketState) -> dict:
    """Invoke LLM with bound tools in the ReAct loop."""
    llm = get_llm()
    if llm is None:
        return {
            "fallback": True,
            "messages": [AIMessage(content="LLM unavailable; routing to human fallback.")],
        }
    try:
        tools_list = tools.as_langchain_tools()
        llm_with_tools = llm.bind_tools(tools_list)
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {
            "fallback": True,
            "messages": [AIMessage(content=f"LLM execution error: {e}")],
        }


def route_after_agent(state: TicketState) -> Literal["tools", "decide"]:
    """Conditional edge: route to tools if tool_calls exist, else finalize via decide."""
    if state.get("fallback"):
        return "decide"
    msgs = state.get("messages", [])
    if msgs and getattr(msgs[-1], "tool_calls", None):
        return "tools"
    return "decide"


def decide_node(state: TicketState) -> dict:
    """Synthesize full ReAct message history into validated TriageResult."""
    ticket_text = state.get("ticket_text", "")
    msgs = state.get("messages", [])

    # Extract all observations returned by ToolNode
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    billing_out = next((m.content for m in tool_msgs if m.name == "check_billing"), "skipped (no billing signals).")
    status_out = next((m.content for m in tool_msgs if m.name == "check_system_status"), "skipped (no outage signals).")

    kb_msgs = [m.content for m in tool_msgs if m.name == "search_knowledge_base"]
    kb_ids: list[str] = []
    for k in kb_msgs:
        for match in re.findall(r"kb-[\w-]+", k):
            if match not in kb_ids:
                kb_ids.append(match)

    if not kb_ids and ticket_text:
        direct_kb = tools.search_knowledge_base.invoke({"query": ticket_text})
        kb_ids = re.findall(r"kb-[\w-]+", direct_kb)[:2]

    tool_outputs = {
        "billing": billing_out,
        "status": status_out,
        "kb_ids": kb_ids[:2],
    }

    if state.get("fallback") or not msgs:
        cust_id = state.get("ticket", {}).get("customer_id", "")
        fallback_tools = retrieve(ticket_text, cust_id) if cust_id else tool_outputs
        analysis = _degraded_analyze()
        res = decide(analysis, fallback_tools, ticket_text)
        return {"result": res, "tool_outputs": fallback_tools, "analysis": analysis}

    llm = get_llm()
    if llm is None:
        analysis = _degraded_analyze()
    else:
        try:
            chain = finalize_prompt | _with_structured_output_no_afc(llm)
            structured = chain.invoke({"messages": msgs})
            analysis = structured.to_dict() if isinstance(structured, TriageResult) else dict(structured)
        except Exception:
            analysis = _degraded_analyze()

    res = decide(analysis, tool_outputs, ticket_text)
    return {"result": res, "tool_outputs": tool_outputs, "analysis": analysis}


# Backward-compatible exports
def analyze_node(state: TicketState) -> dict:
    return {"analysis": analyze(state["ticket_text"])}


def retrieve_node(state: TicketState) -> dict:
    return {"tool_outputs": retrieve(state["ticket_text"], state["ticket"]["customer_id"])}
