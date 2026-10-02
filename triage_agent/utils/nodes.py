"""Node functions for the triage graph: ingest -> agent <-> tools -> decide.

`decide_node` synthesizes the ReAct message history via structured output
(`finalize_prompt | llm.with_structured_output`); if the LLM is unavailable the
degraded fallback marks the analysis as such and `decide` escalates every
ticket to a human (no pretend classification).
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..prompts import REACT_AGENT_SYSTEM_PROMPT, finalize_prompt
from . import tools
from .state import TicketState, TriageResult




def ingest(ticket: dict) -> str:
    """Join the whole thread (timestamps kept) into one text for analysis.

    Accepts both internal tuple pairs ``(timestamp, text)`` and API-style
    dicts ``{"timestamp": ..., "text": ...}`` so direct ``run_ticket`` calls
    can't crash on shape mismatch.
    """
    lines = []
    for m in ticket.get("messages", []):
        if isinstance(m, dict):
            t, text = m.get("timestamp", "just now"), m.get("text", "")
        else:
            try:
                t, text = m
            except Exception:
                continue
        lines.append(f"[{t}] {text}")
    return "\n".join(lines)


def _build_llm(provider: str, model: str, api_key: str):  # type: ignore[no-untyped-def]
    """Construct a fresh ChatModel per call (no global cache).

    Previously cached with lru_cache keyed on the raw api_key — that pinned
    secrets in memory and shared one client across asyncio.to_thread workers.
    Construction is cheap (no network); invoke() does the I/O, so building
    per node call is safer and barely slower.
    """
    if not api_key:
        return None
    try:
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model,
                temperature=0,
                timeout=60,
                max_retries=2,
                google_api_key=api_key,
            )
        else:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                temperature=0,
                timeout=60,
                max_retries=2,
                openai_api_key=api_key,
            )
    except Exception:
        return None


def get_llm():  # type: ignore[no-untyped-def]
    """Return a ChatModel instance, or None if keys are absent.

    Unknown LLM_PROVIDER values fall back to openai with a warning (typos
    previously failed silently to openai with no signal).
    """
    import logging

    raw = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if raw == "gemini":
        provider = "gemini"
        model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        api_key = os.getenv("GOOGLE_API_KEY", "")
    else:
        if raw not in ("", "openai"):
            logging.getLogger("triage_agent.nodes").warning(
                "unknown LLM_PROVIDER=%r, falling back to openai", raw
            )
        provider = "openai"
        model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
        api_key = os.getenv("OPENAI_API_KEY", "")
    return _build_llm(provider, model, api_key)


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





def _extract_kb_ids(text: str) -> list[str]:
    """Extract kb-* IDs from tool output without regex."""
    ids: list[str] = []
    for token in str(text).replace("|", " ").replace(":", " ").split():
        cand = token.strip().strip(".,;()[]\"'")
        cand = "".join(ch for ch in cand if ch.isalnum() or ch in ("-", "_"))
        if cand.startswith("kb-") and len(cand) > 3 and cand not in ids:
            ids.append(cand)
    return ids


def decide(analysis: dict, tool_outputs: dict, ticket_text: str = "") -> TriageResult:
    """Trust the LLM analysis; apply only confidence floors, no regex policy."""
    _ = ticket_text  # kept for caller compat; ignored — LLM decides.
    if analysis.get("fallback"):
        # No evidence suffix: fallback collects the customer profile only,
        # and decide_node quotes it into reasoning directly.
        return TriageResult(
            urgency=analysis.get("urgency", "high"),
            product=analysis.get("product", "unknown"),
            issue_types=analysis.get("issue_types", ["unknown"]),
            sentiment=analysis.get("sentiment", "unknown"),
            kb_refs=tool_outputs.get("kb_ids", []),
            next_action="escalate to human",
            reasoning=analysis.get("reasoning", ""),
            confidence=0.5,
        )

    # LLM decides: next_action alone determines escalation.
    # urgency="critical" no longer forces escalate; the model expresses
    # escalation via next_action="escalate to human".
    llm_escalate = analysis.get("next_action") == "escalate to human"

    base_urgency = analysis.get("urgency", "medium")
    # Preserve LLM calibration: policy sets a floor, never downgrades a higher
    # LLM confidence. E.g. LLM 0.99 route-to-specialist stays 0.99, not 0.8.
    try:
        llm_conf = float(analysis.get("confidence", 0.0))
    except (TypeError, ValueError):
        llm_conf = 0.0
    if llm_escalate:
        # LLM chose escalate: keep its urgency as-is, floor confidence.
        action, confidence = "escalate to human", max(0.95, min(1.0, llm_conf) if llm_conf else 0.95)
        urgency = base_urgency
    elif analysis.get("next_action") == "route to specialist" or any("bug" in str(t).lower() for t in analysis.get("issue_types", [])):
        action, confidence = "route to specialist", max(0.8, min(1.0, llm_conf) if llm_conf else 0.8)
        urgency = base_urgency
    elif analysis.get("urgency") == "low" or analysis.get("next_action") == "auto-respond":
        action, confidence = "auto-respond", max(0.9, min(1.0, llm_conf) if llm_conf else 0.9)
        urgency = base_urgency
    else:
        action, confidence = "route to specialist", max(0.7, min(1.0, llm_conf) if llm_conf else 0.7)
        urgency = base_urgency

    evidence_bits = []
    for label, text in (("billing", tool_outputs.get("billing", "")), ("status", tool_outputs.get("status", ""))):
        t = str(text).strip()
        if not t:
            continue
        low = t.lower()
        if low.startswith("skipped"):
            continue
        # Bound reasoning size / PII surface: evidence is a short excerpt, not a dump.
        if len(t) > 300:
            t = t[:300] + "…"
        evidence_bits.append(f"{label}: {t}")
    evidence_suffix = f" | evidence: {' ; '.join(evidence_bits)}" if evidence_bits else ""

    return TriageResult(
        urgency=urgency,
        product=analysis.get("product", "general"),
        issue_types=analysis.get("issue_types", []),
        sentiment=analysis.get("sentiment", "neutral"),
        kb_refs=tool_outputs.get("kb_ids", []),
        next_action=action,
        reasoning=analysis.get("reasoning", "") + evidence_suffix,
        confidence=confidence,
    )


# --- ReAct Graph Nodes ---


def ingest_node(state: TicketState) -> dict:
    """Format thread and initialize ReAct agent messages."""
    ticket = state["ticket"]
    ticket_text = ingest(ticket)
    cust_id = ticket.get("customer_id", "unknown")
    user_prompt = f"Customer ID: {cust_id}\n\nTicket Messages:\n<ticket>\n{ticket_text}\n</ticket>"
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
        for match in _extract_kb_ids(k):
            if match not in kb_ids:
                kb_ids.append(match)

    if not kb_ids and ticket_text:
        # Cap the fallback query: full threads can be 200k chars and only
        # dilute keyword scoring. First 2000 chars carry the signal.
        direct_kb = tools.search_knowledge_base.invoke({"query": ticket_text[:2000]})
        kb_ids = _extract_kb_ids(direct_kb)[:2]

    tool_outputs = {
        "billing": billing_out,
        "status": status_out,
        "kb_ids": kb_ids[:2],
    }

    if state.get("fallback") or not msgs:
        # No-LLM path: fetch the customer profile only — the outcome is
        # always `escalate to human`, so billing/status/KB lookups add no
        # decision value. The profile is quoted into reasoning for handoff.
        cust_id = state.get("ticket", {}).get("customer_id", "").strip()
        profile_ev = (
            tools.get_customer_profile.invoke({"customer_id": cust_id})
            if cust_id
            else "unknown customer."
        )
        fallback_tools = {
            "billing": "skipped (fallback collects customer profile only).",
            "status": "skipped (fallback collects customer profile only).",
            "kb_ids": [],
        }
        analysis = _degraded_analyze()
        res = decide(analysis, fallback_tools, ticket_text)
        res.reasoning = f"{res.reasoning} | profile: {profile_ev}"
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



