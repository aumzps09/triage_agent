"""Mock tool contracts for the triage agent.

Design (per topology ticket): the LangGraph `retrieve` node calls these tools —
no ToolNode loop, no network, no API key. Each tool is defined with
`langchain_core.tools.tool` at the definition site so the same object works
both as a plain callable (via `.invoke`) and when bound to an LLM.

Failure policy: never raise on bad input; return a model-readable string so the
`decide` node can fall back to `escalate to human`.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .mock_data import BILLING_CHARGES, CUSTOMERS, KB_ENTRIES, SYSTEM_STATUS


@tool
def get_customer_profile(customer_id: str) -> str:
    """Look up plan, tenure, seats, and ticket history. Mock returns static string."""
    customer = CUSTOMERS.get(customer_id.strip())
    if customer is None:
        known = ", ".join(sorted(CUSTOMERS))
        return f"Customer not found: {customer_id!r}. Known ids: {known}."
    parts = [f"{k}={v}" for k, v in customer.items()]
    return "Customer profile: " + "; ".join(parts) + "."


@tool
def search_knowledge_base(query: str) -> str:
    """Search FAQ/docs. Mock keyword-matches static entries, returns ids + titles."""
    tokens = [t.lower() for t in query.split() if len(t) > 2]
    if not tokens:
        return "No matching articles. Hint: query with keywords like 'charge', '500', 'dark mode'."
    scored = []
    for entry in KB_ENTRIES:
        hay = " ".join([entry["title"], entry["text"], " ".join(entry["keywords"])]).lower()
        score = sum(1 for t in tokens if t in hay)
        if score:
            scored.append((score, entry))
    if not scored:
        return "No matching articles. Hint: query with keywords like 'charge', '500', 'dark mode'."
    scored.sort(key=lambda s: -s[0])
    lines = [f"{e['id']}: {e['title']}" for _, e in scored[:3]]
    return "KB matches: " + " | ".join(lines) + "."


@tool
def check_system_status(region: str) -> str:
    """Check per-region status. Global status page can be stale; region is authoritative."""
    key = region.strip().lower()
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
    charges = BILLING_CHARGES.get(customer_id.strip())
    if not charges:
        return f"No charges on file for {customer_id!r}."
    pending = sum(c["amount"] for c in charges if not c["refunded"])
    return (
        f"Billing for {customer_id!r}: {len(charges)} charge(s), "
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
