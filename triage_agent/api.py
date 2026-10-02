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
import logging
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from .agent import run_ticket
from .sample_tickets import SAMPLE_TICKETS
from .utils import tools as mock_tools

logger = logging.getLogger("triage_agent.api")

# Max total chars per batch (20 tickets x 50 msgs x 4000 chars would be 4M
# chars to the LLM — cap the batch to keep latency/cost bounded).
_MAX_BATCH_CHARS = 60000
# Same guard for a single ticket (50 msgs x 4000 chars = 200k would go
# straight to the LLM without the batch cap).
_MAX_SINGLE_CHARS = 20000
# Cap for direct /tools/kb queries (mirrors the 2000-char cap used for the
# fallback KB lookup in decide_node so long threads don't dilute scoring).
_MAX_KB_QUERY_CHARS = 2000


def _triage_timeout_s() -> float:
    """Read per-call (not import-time) so env changes apply without restart."""
    try:
        v = float(os.getenv("TRIAGE_TIMEOUT_S", "120"))
    except ValueError:
        return 120.0
    return v if v > 0 else 120.0


def _batch_concurrency() -> int:
    """Read per-request (not import-time) so env changes apply without restart."""
    try:
        return max(1, min(8, int(os.getenv("BATCH_CONCURRENCY", "4"))))
    except ValueError:
        return 4


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach X-Request-ID to every response + log method/path/status/latency."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s (%.1fms) rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response


_RATE_LIMIT_WINDOW_S = 60.0
_rate_hits: dict[str, list[float]] = {}
# Bound memory: per-process fixed-window store must not grow with unique IPs.
_RATE_MAX_IPS = 1000

# asyncio lock — threading.Lock would block the event loop inside async dispatch.
_rate_lock = asyncio.Lock()


def _rate_limit_per_min() -> int:
    """Parse RATE_LIMIT_PER_MIN defensively: bad values fall back to 120, never 500."""
    try:
        return max(0, int(os.getenv("RATE_LIMIT_PER_MIN", "120")))
    except ValueError:
        return 120


def _cors_origins() -> list[str]:
    """Parse ALLOWED_ORIGINS (comma-separated, blanks dropped). Default open for local dev.

    Prod: set to explicit origins, e.g. ALLOWED_ORIGINS=https://app.example.com.
    """
    raw = os.getenv("ALLOWED_ORIGINS", "*")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


def _expected_api_key() -> str:
    return os.getenv("TRIAGE_API_KEY", "").strip() or os.getenv("API_KEY", "").strip()


def _check_api_key(request: Request) -> None:
    """Optional shared-secret auth: open when no key configured (local dev/tests).

    Set TRIAGE_API_KEY (or API_KEY) in prod; callers send `x-api-key: <secret>`.
    """
    expected = _expected_api_key()
    if not expected:
        return
    got = request.headers.get("x-api-key", "")
    if got != expected:
        raise HTTPException(status_code=401, detail="invalid or missing api key")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-IP limit on POST /triage* (stdlib only).

    `RATE_LIMIT_PER_MIN=0` disables. Excess returns 429, never 500.
    429s carry X-Request-ID so they stay traceable even though this
    middleware sits outside RequestIdMiddleware.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.method == "POST" and request.url.path.startswith(("/triage", "/v1/triage")):
            limit = _rate_limit_per_min()
            if limit > 0:
                now = time.monotonic()
                ip = request.client.host if request.client else "unknown"
                request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
                async with _rate_lock:
                    if ip not in _rate_hits and len(_rate_hits) >= _RATE_MAX_IPS:
                        # Evict oldest IP bucket instead of growing unbounded.
                        oldest = min(_rate_hits, key=lambda k: _rate_hits[k][-1] if _rate_hits[k] else 0)
                        _rate_hits.pop(oldest, None)
                    hits = _rate_hits.setdefault(ip, [])
                    hits[:] = [h for h in hits if now - h < _RATE_LIMIT_WINDOW_S]
                    if len(hits) >= limit:
                        return JSONResponse(
                            status_code=429,
                            content={"detail": "rate limit exceeded"},
                            headers={"X-Request-ID": request_id, "Retry-After": "60"},
                        )
                    hits.append(now)
        return await call_next(request)

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
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(RateLimitMiddleware)


# ---------- Schemas ----------


class TicketMessage(BaseModel):
    timestamp: str = Field(default="just now", max_length=64, examples=["just now"])
    text: str = Field(
        min_length=1,
        max_length=4000,
        examples=["My payment failed, I see 3 pending charges of $29.99"],
    )


class TriageRequest(BaseModel):
    id: str = Field(default="ticket-custom-1", min_length=1, max_length=128, examples=["ticket-custom-1"])
    customer_id: str = Field(min_length=1, max_length=128, examples=["cust_free_001"])
    messages: list[TicketMessage] = Field(
        min_length=1,
        max_length=50,
        examples=[
            [
                {"timestamp": "1h ago", "text": "My payment failed when I tried to upgrade to Pro."},
                {"timestamp": "just now", "text": "Now I see THREE charges of $29.99 and still no Pro access!"},
            ]
        ],
    )


class BatchTriageRequest(BaseModel):
    tickets: list[TriageRequest] = Field(min_length=1, max_length=20)


def _to_internal_ticket(req: TriageRequest) -> dict:
    if not req.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")
    if not req.id.strip():
        raise HTTPException(status_code=422, detail="id must not be blank")
    cleaned = [(m.timestamp.strip() or "just now", m.text.strip()) for m in req.messages]
    if any(not text for _, text in cleaned):
        raise HTTPException(status_code=422, detail="message text must not be blank")
    if not req.customer_id.strip():
        raise HTTPException(status_code=422, detail="customer_id must not be blank")
    return {
        "id": req.id.strip(),
        "customer_id": req.customer_id.strip(),
        "messages": cleaned,
    }


async def _run_ticket_guarded(ticket: dict):  # type: ignore[no-untyped-def]
    """Run the sync graph off the event loop with an overall timeout."""
    return await asyncio.wait_for(
        asyncio.to_thread(run_ticket, ticket), timeout=_triage_timeout_s()
    )


def _sample_to_api(ticket: dict) -> dict:
    return {
        "id": ticket["id"],
        "customer_id": ticket["customer_id"],
        "messages": [{"timestamp": t, "text": m} for t, m in ticket["messages"]],
    }


def _llm_status() -> dict[str, Any]:
    raw = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if raw == "gemini":
        provider, configured = "gemini", bool(os.getenv("GOOGLE_API_KEY"))
    else:
        if raw not in ("", "openai"):
            logger.warning("unknown LLM_PROVIDER=%r, falling back to openai", raw)
        provider, configured = "openai", bool(os.getenv("OPENAI_API_KEY"))
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
            "GET /v1/health",
            "POST /v1/triage",
            "POST /v1/triage/batch",
        ],
    }


@app.get("/health")
@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok", **_llm_status()}


@app.get("/tickets/samples")
@app.get("/v1/tickets/samples")
def list_samples() -> dict:
    return {"count": len(SAMPLE_TICKETS), "tickets": [_sample_to_api(t) for t in SAMPLE_TICKETS]}


@app.post("/triage")
@app.post("/v1/triage")
async def triage(req: TriageRequest, request: Request) -> dict:
    _check_api_key(request)
    ticket = _to_internal_ticket(req)
    total_chars = sum(len(text) for _, text in ticket["messages"])
    if total_chars > _MAX_SINGLE_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"ticket too large: {total_chars} chars > {_MAX_SINGLE_CHARS} max",
        )
    try:
        result = await _run_ticket_guarded(ticket)
    except asyncio.TimeoutError as e:
        raise HTTPException(status_code=502, detail="triage timed out") from e
    except Exception as e:  # fail-closed, never 500 on triage logic
        raise HTTPException(status_code=502, detail=f"triage failed: {e}") from e
    return {"ticket_id": ticket["id"], **_llm_status(), "result": result.to_dict()}


async def _triage_one(ticket: dict, sem: asyncio.Semaphore) -> dict:
    """Run one ticket; per-item errors are captured, never abort the batch."""
    async with sem:
        try:
            result = await _run_ticket_guarded(ticket)
        except asyncio.TimeoutError:
            logger.exception("triage timed out for %s", ticket["id"])
            return {"ticket_id": ticket["id"], "error": "triage timed out"}
        except Exception as e:
            logger.exception("triage failed for %s", ticket["id"])
            return {"ticket_id": ticket["id"], "error": f"triage failed: {e}"}
        return {"ticket_id": ticket["id"], "result": result.to_dict()}


@app.post("/triage/batch")
@app.post("/v1/triage/batch")
async def triage_batch(req: BatchTriageRequest, request: Request) -> dict:
    _check_api_key(request)
    if not req.tickets:
        raise HTTPException(status_code=422, detail="tickets must not be empty")
    tickets = [_to_internal_ticket(t) for t in req.tickets]
    total_chars = sum(len(text) for t in tickets for _, text in t["messages"])
    if total_chars > _MAX_BATCH_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"batch too large: {total_chars} chars > {_MAX_BATCH_CHARS} max",
        )
    sem = asyncio.Semaphore(_batch_concurrency())
    # Parallel fan-out (order preserved); one ticket failing no longer
    # aborts the whole batch — its entry carries "error" instead.
    out = await asyncio.gather(*(_triage_one(t, sem) for t in tickets))
    errors = sum(1 for x in out if "error" in x)
    return {
        "count": len(out),
        "ok": len(out) - errors,
        "errors": errors,
        **_llm_status(),
        "results": list(out),
    }


# ---------- Direct mock-tool endpoints (no LLM key needed) ----------


@app.get("/tools/profile")
def tool_profile(customer_id: str = Query(max_length=128, examples=["cust_free_001"])) -> dict:
    return {
        "tool": "get_customer_profile",
        "customer_id": customer_id,
        "output": mock_tools.get_customer_profile.invoke({"customer_id": customer_id}),
    }


@app.get("/tools/billing")
def tool_billing(customer_id: str = Query(max_length=128, examples=["cust_free_001"])) -> dict:
    return {
        "tool": "check_billing",
        "customer_id": customer_id,
        "output": mock_tools.check_billing.invoke({"customer_id": customer_id}),
    }


@app.get("/tools/status")
def tool_status(region: str = Query(max_length=64, examples=["asia"])) -> dict:
    return {
        "tool": "check_system_status",
        "region": region,
        "output": mock_tools.check_system_status.invoke({"region": region}),
    }


@app.get("/tools/kb")
def tool_kb(query: str = Query(max_length=_MAX_KB_QUERY_CHARS, examples=["duplicate charge refund"])) -> dict:
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
