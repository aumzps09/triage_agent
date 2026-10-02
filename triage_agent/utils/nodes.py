"""Node functions for the triage graph: ingest -> analyze -> retrieve -> decide.

`analyze` tries the LangChain structured-output path first
(`triage_prompt | llm.with_structured_output`); if the LLM is unavailable the
degraded fallback marks the analysis as such and `decide` escalates every
ticket to a human (no pretend classification).
"""

from __future__ import annotations

import os
import re

from ..prompts import triage_prompt
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


def _llm_analyze(ticket_text: str) -> TriageResult | None:
    """Try the LangChain structured-output path; return None on any failure.

    Provider selected via `LLM_PROVIDER` (`openai` default, or `gemini`).
    OpenAI uses `OPENAI_API_KEY`/`OPENAI_MODEL` (default `gpt-6-luna`);
    Gemini uses `GOOGLE_API_KEY`/`GEMINI_MODEL` (default `gemini-3.1-flash-lite`).
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    try:
        if provider == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                return None
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"), temperature=0
            )
        else:
            if not os.getenv("OPENAI_API_KEY"):
                return None
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-6-luna"))
        chain = triage_prompt | _with_structured_output_no_afc(llm)
        out = chain.invoke({"ticket_text": ticket_text})
        return out if isinstance(out, TriageResult) else TriageResult(**dict(out))
    except Exception:
        return None


def _with_structured_output_no_afc(llm):  # type: ignore[no-untyped-def]
    """Structured output with SDK automatic function calling disabled.

    Works around https://github.com/langchain-ai/langchain-google/pull/1980:
    `ChatGoogleGenerativeAI` never sets `automatic_function_calling`, so the
    underlying `google-genai` SDK takes its AFC path in
    `Models.generate_content` and logs a "not recommended" warning on every
    request — even with no tools bound. LangChain runs its own tool loop and
    only sends tool declarations, so SDK-side AFC should stay off.
    """
    structured = llm.with_structured_output(TriageResult, method="json_schema")
    # Only the Gemini chat model supports this kwarg; OpenAI takes another path.
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
    """Degraded fallback (LLM unavailable): no classification, escalate all.

    Deliberately generic — `decide` sees `"fallback": True` and routes every
    ticket to a human with low confidence.
    """
    return {
        "urgency": "high",
        "product": "unknown",
        "issue_types": ["unknown"],
        "sentiment": "unknown",
        "sentiment_trajectory": "unknown (LLM unavailable)",
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
    """Call tools directly (no ToolNode loop): profile+KB always, others by keyword."""
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
    out["kb_ids"] = re.findall(r"kb-[\w-]+", out["kb"])[:2]  # top-2 expected refs per ticket
    return out


def decide(analysis: dict, tool_outputs: dict) -> TriageResult:
    """Apply the locked OR escalation policy over analysis + tool evidence.

    Degraded fallback (`analysis["fallback"]`, LLM unavailable) escalates every
    ticket to a human with low confidence — fail-closed, never auto-respond.
    """
    if analysis.get("fallback"):
        return TriageResult(
            urgency=analysis.get("urgency", "high"),
            product=analysis.get("product", "unknown"),
            issue_types=analysis.get("issue_types", ["unknown"]),
            sentiment=analysis.get("sentiment", "unknown"),
            sentiment_trajectory=analysis.get("sentiment_trajectory", ""),
            kb_refs=tool_outputs.get("kb_ids", []),
            next_action="escalate to human",
            reasoning=analysis.get("reasoning", "") + f" | evidence: {tool_outputs.get('billing', '')} {tool_outputs.get('status', '')}".strip(),
            confidence=0.5,
        )
    text = " ".join([analysis.get("reasoning", ""), tool_outputs.get("billing", ""), tool_outputs.get("status", "")]).lower()
    escalate = (
        "disput" in text
        or ("pending" in text and ("presentation" in text or "deadline" in text or "end of day" in text))
        or ("degraded" in text and "500" in text)
    )
    if escalate:
        action, confidence = "escalate to human", 0.95
    elif analysis.get("urgency") == "low":
        action, confidence = "auto-respond", 0.9
    else:
        action, confidence = "route to specialist", 0.7
    return TriageResult(
        urgency=analysis.get("urgency", "medium"),
        product=analysis.get("product", "general"),
        issue_types=analysis.get("issue_types", []),
        sentiment=analysis.get("sentiment", "neutral"),
        sentiment_trajectory=analysis.get("sentiment_trajectory", ""),
        kb_refs=tool_outputs.get("kb_ids", []),
        next_action=action,
        reasoning=analysis.get("reasoning", "") + f" | evidence: {tool_outputs.get('billing', '')} {tool_outputs.get('status', '')}".strip(),
        confidence=confidence,
    )


# --- Graph nodes (one per topology step) ---


def ingest_node(state: TicketState) -> dict:
    return {"ticket_text": ingest(state["ticket"])}


def analyze_node(state: TicketState) -> dict:
    return {"analysis": analyze(state["ticket_text"])}


def retrieve_node(state: TicketState) -> dict:
    return {"tool_outputs": retrieve(state["ticket_text"], state["ticket"]["customer_id"])}


def decide_node(state: TicketState) -> dict:
    return {"result": decide(state["analysis"], state["tool_outputs"])}
