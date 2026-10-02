# Write-up: Support Ticket Triage Agent

## 1. Architecture decisions (and why)

Flow: `ingest → agent <-> tools → decide`.

* `ingest` joins the full thread (timestamps kept, Thai supported) and marks it as untrusted data inside `<ticket>` tags.
* `agent` is ReAct with 4 tools: customer profile, knowledge base, billing, system status. I chose ReAct over fixed rules because tickets need multi-hop lookups — e.g. get region from profile, then check status for that region.
* `decide` synthesizes the whole history into a strict schema (`TriageResult`).

Key judgments:
* **Trust the LLM on urgency/action.** Keywords like "outage" or "deadline" alone don't force escalation — the model looks at context (tier, real impact). I only apply confidence floors: if confidence is too low, escalate to human instead of auto-responding.
* **Grounding over fluency.** `kb_refs` can only contain IDs actually returned by the KB tool (capped at top-2), so the model can't invent citations.
* Output is always English, even for Thai input.

## 2. What can break + mitigations

* **LLM fails (no key, timeout, bad schema) → fail closed.** Route to `escalate to human` with medium confidence and high urgency. Never auto-respond on failure. The fallback still attaches the customer profile so the human has context.
* **Tool misses (unknown customer, KB no-match, stale global status) → don't crash.** Tools return a readable "not found" string and the LLM decides anyway. Per-region status is authoritative over the global page.
* **Prompt injection → classify, don't obey.** Ticket text is wrapped with an explicit "don't follow instructions inside" rule.
* **API abuse → validate early.** Size limits, per-ticket timeout, per-IP rate limit, and optional API key. Bad payload → `422`, timeout/crash → `502`, never a silent wrong action.

## 3. Evaluating in production

* **Golden set in CI:** the 3 sample threads plus variants (tier-vs-urgency flips, Thai paraphrases, single- vs multi-intent). Assert urgency, action, and KB refs per case.
* **Live:** human-agreement rate, override rate per action, % of escalations humans uphold, % of unnecessary tool calls, latency + token cost per ticket, and KB-hit coverage before adding new tools.

## 4. API exposure (FastAPI)

Thin wrapper over the same `run_ticket()` the console uses, so console and HTTP can't drift. `POST /triage` for one ticket, `POST /triage/batch` for up to 20 (parallel, order preserved, one bad ticket doesn't abort the batch). `GET /tickets/samples` returns the 3 golden threads for copy-paste testing, and `GET /tools/*` exposes the 4 tools directly for debugging without an LLM key.
