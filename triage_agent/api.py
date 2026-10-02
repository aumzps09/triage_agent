"""FastAPI service for the Support Ticket Triage Agent.

Run live:
    uvicorn triage_agent.api:app --reload --port 8000
    # docs: http://localhost:8000/docs

Without an LLM key every ticket fail-closes to `escalate to human`
(confidence 0.5) — same degraded fallback as the console runner.
Set LLM_PROVIDER + provider key in .env to enable the real LLM path.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import run_ticket
from .sample_tickets import SAMPLE_TICKETS
from .utils import tools as mock_tools

app = FastAPI(
    title="Support Ticket Triage Agent",
    description=(
        "ReAct triage agent: classify urgency, extract product/issue/sentiment, "
        "search KB, decide next action (auto-respond / route to specialist / escalate to human)."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------


class TicketMessage(BaseModel):
    timestamp: str = Field(default="just now", examples=["just now"])
    text: str = Field(examples=["My payment failed, I see 3 pending charges of $29.99"])


class TriageRequest(BaseModel):
    id: str = Field(default="ticket-custom-1", examples=["ticket-custom-1"])
    customer_id: str = Field(examples=["cust_free_001"])
    messages: list[TicketMessage] = Field(
        examples=[
            [
                {"timestamp": "1h ago", "text": "My payment failed when I tried to upgrade to Pro."},
                {"timestamp": "just now", "text": "Now I see THREE charges of $29.99 and still no Pro access!"},
            ]
        ]
    )


class BatchTriageRequest(BaseModel):
    tickets: list[TriageRequest]


def _to_internal_ticket(req: TriageRequest) -> dict:
    if not req.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")
    return {
        "id": req.id,
        "customer_id": req.customer_id,
        "messages": [(m.timestamp, m.text) for m in req.messages],
    }


def _sample_to_api(ticket: dict) -> dict:
    return {
        "id": ticket["id"],
        "customer_id": ticket["customer_id"],
        "messages": [{"timestamp": t, "text": m} for t, m in ticket["messages"]],
    }


def _llm_status() -> dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        configured = bool(os.getenv("GOOGLE_API_KEY"))
    else:
        configured = bool(os.getenv("OPENAI_API_KEY"))
    return {"llm_provider": provider, "llm_configured": configured}


# ---------- Routes ----------


@app.get("/")
def root() -> dict:
    return {
        "service": "support-ticket-triage-agent",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "GET /health",
            "GET /tickets/samples",
            "POST /triage",
            "POST /triage/batch",
            "GET /tools/profile?customer_id=cust_free_001",
            "GET /tools/billing?customer_id=cust_free_001",
            "GET /tools/status?region=asia",
            "GET /tools/kb?query=duplicate+charge",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **_llm_status()}


@app.get("/tickets/samples")
def list_samples() -> dict:
    return {"count": len(SAMPLE_TICKETS), "tickets": [_sample_to_api(t) for t in SAMPLE_TICKETS]}


@app.post("/triage")
async def triage(req: TriageRequest) -> dict:
    ticket = _to_internal_ticket(req)
    try:
        result = await asyncio.to_thread(run_ticket, ticket)
    except Exception as e:  # fail-closed, never 500 on triage logic
        raise HTTPException(status_code=502, detail=f"triage failed: {e}") from e
    return {"ticket_id": ticket["id"], **_llm_status(), "result": result.to_dict()}


@app.post("/triage/batch")
async def triage_batch(req: BatchTriageRequest) -> dict:
    if not req.tickets:
        raise HTTPException(status_code=422, detail="tickets must not be empty")
    out = []
    for t in req.tickets:
        ticket = _to_internal_ticket(t)
        try:
            result = await asyncio.to_thread(run_ticket, ticket)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"triage failed for {ticket['id']}: {e}") from e
        out.append({"ticket_id": ticket["id"], "result": result.to_dict()})
    return {"count": len(out), **_llm_status(), "results": out}


# ---------- Direct mock-tool endpoints (no LLM key needed) ----------


@app.get("/tools/profile")
def tool_profile(customer_id: str = Query(examples=["cust_free_001"])) -> dict:
    return {
        "tool": "get_customer_profile",
        "customer_id": customer_id,
        "output": mock_tools.get_customer_profile.invoke({"customer_id": customer_id}),
    }


@app.get("/tools/billing")
def tool_billing(customer_id: str = Query(examples=["cust_free_001"])) -> dict:
    return {
        "tool": "check_billing",
        "customer_id": customer_id,
        "output": mock_tools.check_billing.invoke({"customer_id": customer_id}),
    }


@app.get("/tools/status")
def tool_status(region: str = Query(examples=["asia"])) -> dict:
    return {
        "tool": "check_system_status",
        "region": region,
        "output": mock_tools.check_system_status.invoke({"region": region}),
    }


@app.get("/tools/kb")
def tool_kb(query: str = Query(examples=["duplicate charge refund"])) -> dict:
    return {
        "tool": "search_knowledge_base",
        "query": query,
        "output": mock_tools.search_knowledge_base.invoke({"query": query}),
    }


def main() -> None:
    import uvicorn

    uvicorn.run("triage_agent.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
