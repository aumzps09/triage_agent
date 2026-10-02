"""Mock tool contracts for the triage agent.

Design: the LangGraph `agent` node binds these tools via a `ToolNode` loop —
no network, no API key. Each tool is defined with
`langchain_core.tools.tool` at the definition site so the same object works
both as a plain callable (via `.invoke`) and when bound to an LLM.

Failure policy: never raise on bad input; return a model-readable string so the
`decide` node can fall back to `escalate to human`.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool

from .mock_data import BILLING_CHARGES, CUSTOMERS, KB_ENTRIES, SYSTEM_STATUS


@tool
def get_customer_profile(customer_id: str) -> str:
    """Look up plan, tenure, seats, and ticket history. Mock returns static string."""
    cid = str(customer_id or "").strip()
    customer = CUSTOMERS.get(cid)
    if customer is None:
        known = ", ".join(sorted(CUSTOMERS))
        return f"Customer not found: {cid!r}. Known ids: {known}."
    parts = [f"{k}={v}" for k, v in customer.items()]
    return "Customer profile: " + "; ".join(parts) + "."


@tool
def search_knowledge_base(query: str) -> str:
    """Search FAQ/docs. Mock keyword-matches static entries, returns ids + titles."""
    low_query = str(query or "").lower()[:2000]
    tokens = [t for t in low_query.split() if len(t) > 2]
    if not tokens:
        return "No matching articles. Hint: query with keywords like 'charge', '500', 'dark mode'."
    scored = []
    for entry in KB_ENTRIES:
        hay = " ".join([entry["title"], entry["text"], " ".join(entry["keywords"])]).lower()
        score = 0
        for t in tokens:
            if re.fullmatch(r"[a-z0-9]+", t):
                if re.search(rf"\b{re.escape(t)}\b", hay):
                    score += 1
            elif t in hay:
                score += 1
        for kw in entry["keywords"]:
            kl = kw.lower()
            if re.fullmatch(r"[a-z0-9][a-z0-9 \-]*", kl):
                if re.search(rf"\b{re.escape(kl)}\b", low_query):
                    score += 2
            elif kl in low_query:
                score += 2
        if score:
            scored.append((score, entry))
    if not scored:
        return "No matching articles. Hint: query with keywords like 'charge', '500', 'dark mode'."
    scored.sort(key=lambda s: -s[0])
    # Top-2 matches the kb_refs cap enforced in decide_node.
    lines = [f"{e['id']}: {e['title']}" for _, e in scored[:2]]
    return "KB matches: " + " | ".join(lines) + "."


@tool
def check_system_status(region: str) -> str:
    """Check per-region status. Global status page can be stale; region is authoritative."""
    raw = str(region or "")
    key = raw.strip().lower()
    regions = SYSTEM_STATUS["regions"]
    if key not in regions:
        return (
            f"Unknown region {region!r}. Known regions: {', '.join(sorted(regions))}. "
            f"Global page says: {SYSTEM_STATUS['global_page']}."
        )
    info = regions[key]
    return (
        f"Region {key}: {info['status']}. {info['detail']} "
        f"(Global page says: {SYSTEM_STATUS['global_page']}.)"
    ).strip()


@tool
def check_billing(customer_id: str) -> str:
    """List charges for a customer: amounts, pending vs refunded."""
    cid = str(customer_id or "").strip()
    charges = BILLING_CHARGES.get(cid)
    if not charges:
        return f"No charges on file for {cid!r}."
    pending = sum(c["amount"] for c in charges if not c["refunded"])
    return (
        f"Billing for {cid!r}: {len(charges)} charge(s), "
        f"${pending:.2f} pending/unrefunded. "
        "Account still shows Free plan until settlement."
    )


_ALL_TOOLS = (get_customer_profile, search_knowledge_base, check_system_status, check_billing)

# Back-compat: name -> {name, description, args} derived from the tool objects
# (no hand-maintained duplicate schema).
TOOL_SCHEMAS = {
    t.name: {"name": t.name, "description": t.description, "args": t.args}
    for t in _ALL_TOOLS
}


def as_langchain_tools():  # type: ignore[no-untyped-def]
    """Return the mock tools as LangChain tools (already @tool objects)."""
    return list(_ALL_TOOLS)
