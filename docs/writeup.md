# Write-up: Support Ticket Triage Agent

## 1. Architecture decisions (and why)

ReAct StateGraph `ingest → agent <-> tools (ToolNode) → decide → END`. `ingest` joins
the full thread (timestamps preserved, Thai read fluently) into initial agent messages.
The `agent` node binds 4 mock tools (`get_customer_profile`, `search_knowledge_base`,
`check_billing`, `check_system_status`) to the LLM (OpenAI `gpt-6-luna` or Gemini
`gemini-3.1-flash-lite`, temp 0). The agent autonomously decides which tools to call
based on ticket signals, loops through `ToolNode`, and reasons over observations.
When tool execution finishes (`route_after_agent`), `decide` synthesizes the entire
interaction history via `finalize_prompt` with structured output (`TriageResult`),
enforcing an OR escalation safety guard (dispute threat | pending money + deadline |
org/region outage | critical analysis) with negative-lookahead protection.
Why ReAct: allows dynamic multi-hop factual retrieval (e.g., discovering region from
profile then verifying regional status) while `decide` ensures deterministic safety,
schema conformance, and fail-closed human handoff.

## 2. What can break + mitigations

LLM failure (missing key, timeout, model error, schema violation) → degraded fallback:
no classification, every ticket routes to `escalate to human` at confidence 0.5
(fail-closed, never auto-responds).
Tool misses (unknown customer/region, KB no-match) return model-readable strings, never
raise exceptions, and are quoted in reasoning. Stale global status page → per-region
check is authoritative (`asia: degraded`). Long-thread KB noise → `kb_refs` capped at
top-2 exact hits. Thai input → English-only output rule ensures consistent downstream
triage. False-positive reasoning keywords → negated phrasing checks in `decide` prevent
mis-escalating benign inquiries.

## 3. Evaluating in production

Golden set: the 3 threads plus adversarial variants (tier-vs-urgency flips, Thai-only
paraphrases, single- vs multi-intent); assert urgency/next_action/kb_refs per case in CI.
Live: sample + human-label agreement rate, override rate per action, escalation precision
(% escalates human agents uphold), tool invocation accuracy (% unnecessary tool calls),
latency/token cost per ticket, and weekly KB-hit coverage for newly seen intents before
adding new tools.

## 4. API exposure (FastAPI — added after the console agent)

Thin stateless wrapper (`triage_agent/api.py`) over the same `run_ticket()` the console
runner uses — no graph fork, so console and HTTP can never drift. Root `app.py` is the
direct entrypoint: it re-exports `app` (so both `python app.py` and `uvicorn app:app`
work), loads `.env` via `python-dotenv`, and adds `--host/--port/--reload` flags
(defaulting from `HOST`/`PORT` env). `POST /triage` (one
ticket) and `POST /triage/batch` (sequential, fail-fast per ticket) validate via Pydantic
(`TriageRequest` / `BatchTriageRequest`), convert to the internal ticket dict, and run the
sync LangGraph inside `asyncio.to_thread` to keep the event loop free. `GET /tickets/samples`
returns the 3 golden threads in request shape for copy-paste testing; `GET /tools/*`
exposes the 4 mock tools directly (no LLM key needed) for debugging grounding.
Fail-closed maps cleanly onto HTTP: no key → still `200` with `escalate to human` at 0.5
(same as console); bad payload → `422` (`messages`/`tickets` empty); unexpected crash →
`502`, never a silent wrong action. Production adds API-level signals on top of §3:
p95 latency per endpoint, `4xx`/`5xx` rate, batch size distribution, and per-ticket
`llm_provider`/`llm_configured` echo from `/health` and triage responses to detect
key-misconfig incidents.
