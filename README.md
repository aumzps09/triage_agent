# Support Ticket Triage Agent

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green?style=flat-square)

ReAct triage agent for customer-support tickets: classify urgency, extract product / issue / sentiment, search a knowledge base, and decide the next action (`auto-respond` / `route to specialist` / `escalate to human`). Terminal console only — no chat UI.

Supports **OpenAI (default)** and **Gemini** via `LLM_PROVIDER`. No API key is bundled — bring your own.

[Overview](#overview) • [Features](#features) • [How it works](#how-it-works) • [Getting started](#getting-started) • [Run](#run) • [Testing](#testing) • [Project structure](#project-structure) • [Troubleshooting](#troubleshooting)

## Overview

This project implements the "Support Ticket Triage Agent" homework: an AI agent that processes multi-message ticket threads (including Thai input), gathers facts with tools, and returns a structured `TriageResult` with reasoning and confidence.

Sample tickets and expected LLM-path results:

| Ticket | Thread | Expected |
|---|---|---|
| `ticket-1-billing` (Free, 3× $29.99 pending, dispute threat, presentation in 2h) | billing + access | `critical` / `escalate to human` |
| `ticket-2-outage` (Enterprise 45 seats, Thai, error 500, Asia region) | outage + status-check | `critical` / `escalate to human` |
| `ticket-3-darkmode` (Pro, friendly, macOS display bug + feature request) | bug + feature-request | `medium` / `route to specialist` |

See `docs/writeup.md` for architecture decisions, failure modes, and production eval; `docs/assignment-summary.md` for the assignment source summary.

## Features

- **Urgency classification** — `critical` / `high` / `medium` / `low`, weighing business impact and deadline over plan tier.
- **Multi-intent extraction** — product, `issue_types[]`, sentiment across the full thread (English and Thai input, English-only output).
- **Knowledge-base grounding** — keyword search over 6 FAQ entries, cited as `kb_refs[]` (capped at top-2).
- **Deterministic safety policy** — OR escalation (dispute threat, unrefunded money + deadline, org/region outage) applied over LLM analysis + tool evidence.
- **4 mock tools** — `get_customer_profile`, `search_knowledge_base`, `check_system_status`, `check_billing` (no network, no key needed).
- **Fail-closed fallback** — no key or LLM error → every ticket `escalate to human` at confidence 0.5, never auto-responds.

## How it works

ReAct `StateGraph`: `ingest → agent <-> tools (ToolNode) → decide → END`.

```mermaid
flowchart TD
    START([START]) --> ingest["ingest<br/>format full thread & init messages"]
    ingest --> agent["agent<br/>LLM with tools bound<br/>(profile, KB, billing, status)"]
    agent --> cond{"has tool_calls?"}
    cond -- yes --> tools["tools (ToolNode)<br/>execute tool calls"]
    tools --> agent
    cond -- no --> decide["decide<br/>structured synthesis + safety policy -> TriageResult"]
    decide --> END([END])
```

- `ingest` joins the full thread (timestamps preserved) into initial agent messages.
- `agent` binds the 4 mock tools to `ChatOpenAI` / `ChatGoogleGenerativeAI` and loops through `ToolNode` until no more `tool_calls`.
- `decide` synthesizes the full message history via `finalize_prompt` with structured output (`TriageResult`), enforces the escalation guard, and caps `kb_refs` at top-2.

> [!NOTE]
> The global status page can be stale (`all systems operational`). The per-region check is authoritative (`asia: degraded`).

## Getting started

Prerequisites: Python 3.12+, `pip`, a terminal. Optional: Docker, LangGraph CLI.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .   # LLM path needs deps; mocks + degraded fallback run on stdlib
cp .env.example .env  # then fill in your own key; never commit .env
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | `gpt-6-luna` | OpenAI path |
| `GOOGLE_API_KEY` / `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Gemini path |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | unset | Optional tracing |

> [!IMPORTANT]
> Export the file before running — the code reads `os.getenv` directly:
> `set -a; source .env; set +a`

> [!WARNING]
> No key → degraded fallback: no real classification, every ticket `escalate to human` at confidence 0.5 with `product` / `issue_types` / `sentiment` as `unknown`. Set a provider key to enable the real LLM path.

## Run

Console runner (triages the 3 sample tickets, prints each `TriageResult` as JSON):

```bash
set -a; source .env; set +a
.venv/bin/python -m triage_agent
```

Docker (console only, no ports):

```bash
docker build -t triage-agent .
docker run --rm --env-file .env triage-agent
```

Or with Compose (same console behavior, no ports):

```bash
docker compose run --rm triage-agent
```

LangGraph Studio / deploy config (`langgraph.json` exposes `triage_agent` graph from `./triage_agent/agent.py:app`, env from `./.env`):

```bash
langgraph dev  # or: langgraph up
```

Programmatic use:

```python
from triage_agent.agent import run_ticket
result = run_ticket({"id": "...", "customer_id": "cust_free_001", "messages": [...]})
print(result.to_dict())
```

Example output (truncated):

```json
{
  "urgency": "critical",
  "product": "billing",
  "issue_types": ["billing", "access"],
  "sentiment": "negative",
  "kb_refs": ["kb-billing-duplicate", "kb-upgrade-pro-access"],
  "next_action": "escalate to human",
  "confidence": 0.95
}
```

> [!TIP]
> Gemini path disables SDK-side automatic function calling (`AutomaticFunctionCallingConfig(disable=True)`) after `with_structured_output` to silence the `google-genai` AFC warning. LangChain runs its own tool loop, so SDK-side AFC stays off — 0 warnings on stderr.

## Testing

```bash
.venv/bin/python -m pytest -q
```

Covers graph compilation, fail-closed fallback (all tickets escalate at 0.5 with no key), ReAct routing (`tools` vs `decide`), all 4 mock tools, and CLI output for the 3 sample tickets.

## Project structure

```text
triage_agent/
  agent.py            # build_graph, compiled app, run_ticket
  prompts.py          # REACT_AGENT_SYSTEM_PROMPT + finalize_prompt (English-only output, 3 few-shots)
  sample_tickets.py   # 3 sample threads (timestamps kept, Thai kept for ticket 2)
  __main__.py         # console runner (prints TriageResult JSON per ticket)
  utils/
    state.py          # TicketState (TypedDict) + TriageResult (Pydantic output schema)
    nodes.py          # ingest / agent / decide nodes + degraded fallback + AFC workaround
    tools.py          # 4 mock tools as @tool objects (never raise, return model-readable strings)
    mock_data.py      # 3 customers, billing charges, region status, 6 KB entries
docs/
  writeup.md          # 1-page: architecture, failure modes, production eval
  assignment-summary.md # assignment source summary (Thai)
tests/
  test_graph.py test_nodes.py test_tools.py
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| All tickets `escalate to human`, confidence 0.5, fields `unknown` | Degraded fallback — no/invalid key. Check `LLM_PROVIDER` matches the key you set (`OPENAI_API_KEY` vs `GOOGLE_API_KEY`) and that `.env` is exported. |
| `automatic function calling` warnings on stderr (Gemini) | Outdated workaround — `nodes.py:_with_structured_output_no_afc` should bind `AutomaticFunctionCallingConfig(disable=True)`. |
| `Customer not found` / `Unknown region` / `No matching articles` in reasoning | Mock-data miss — tools return strings, never raise; `decide` fail-closes to `escalate to human`. Check `mock_data.py` ids/regions. |
| `ModuleNotFoundError: langchain_*` | Venv not installed — run `pip install -e .` inside `.venv`. The fallback path alone runs on stdlib, the LLM path needs deps. |
```

