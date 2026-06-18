# AI Refund Support Agent

An AI customer-support agent that **processes or denies e-commerce refunds** end to
end. A customer describes their problem in plain language; the agent looks up the
order, runs it through a deterministic refund-policy engine, executes the refund when
it is allowed, and explains the outcome — while streaming its full reasoning trace so
an operator can watch every step.

The entire backend is testable **from the terminal** and runs **with zero setup**: it
ships with a dependency-free deterministic planner, and optionally drives a **local,
open-source LLM via [Ollama](https://ollama.com)**. No external/paid APIs are called
at any point.

---

## Table of contents

1. [Key design principles](#key-design-principles)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [Using the CLI](#using-the-cli)
5. [The refund policy](#the-refund-policy)
6. [Mock data and decision scenarios](#mock-data-and-decision-scenarios)
7. [Optional: local LLM with Ollama](#optional-local-llm-with-ollama)
8. [Optional: HTTP API (admin dashboard)](#optional-http-api-admin-dashboard)
9. [Optional: voice pipeline](#optional-voice-pipeline)
10. [Configuration](#configuration)
11. [Testing](#testing)
12. [Project structure](#project-structure)
13. [Scalability and fault tolerance](#scalability-and-fault-tolerance)

---

## Key design principles

- **The LLM never decides refunds.** A deterministic `PolicyEngine` is the single
  source of truth. The LLM only *interprets* the customer's request and *explains*
  the result. Even if the model "decides" to approve something, no money moves unless
  the policy engine agrees — the refund service re-runs the policy before executing
  (defense in depth).
- **Pluggable LLM provider.** The agent is written against a small `LLMProvider`
  interface with two implementations: a local open-source **Ollama** provider and a
  **deterministic heuristic** provider that needs no model, network, or dependencies.
  The default `auto` mode uses Ollama when it is reachable and silently falls back to
  the heuristic provider otherwise.
- **Money is `Decimal` everywhere**, quantized to two places — never floats.
- **Every decision is fully auditable.** Each evaluation returns a per-rule trail, and
  every agent action is captured in a structured, streamable reasoning log.
- **Clean layering and dependency injection** make every component swappable and
  unit-testable in isolation (a `Clock` is injected so time-based rules are
  deterministic in tests).

## Architecture

```
                ┌──────────────────────────────────────────────┐
   customer ──► │                  CLI / HTTP API               │
                └───────────────────────┬──────────────────────┘
                                        │ user message
                                        ▼
                ┌──────────────────────────────────────────────┐
                │          RefundAgent (function-calling)       │
                │  loop: LLM turn → tool calls → results → …    │
                └──────┬───────────────────────────┬───────────┘
                       │ generate()                │ execute(tool)
                       ▼                            ▼
            ┌────────────────────┐        ┌────────────────────────┐
            │   LLMProvider      │        │      ToolRegistry       │
            │ Ollama | Heuristic │        │  lookup_order,          │
            └────────────────────┘        │  check_refund_eligibility,
                                          │  process_refund, …      │
                                          └──────────┬─────────────┘
                                                     ▼
                              ┌──────────────────────────────────────┐
                              │           RefundService              │
                              │  (orchestration + idempotency)       │
                              └───────┬───────────────────┬──────────┘
                                      ▼                   ▼
                         ┌────────────────────┐   ┌────────────────────┐
                         │    PolicyEngine    │   │   CrmRepository    │
                         │ (deterministic,    │   │  (15 customers /   │
                         │  authoritative)    │   │   orders, indexed) │
                         └────────────────────┘   └────────────────────┘

         Every step is emitted to the ReasoningLog (observable / streamable).
```

A single composition root (`app.py`) wires the object graph; the CLI, the API, and the
tests all build the system the same way.

## Quick start

Requires **Python 3.10+**. From the project root:

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. install the core runtime
pip install -r requirements.txt

# 3. run a scripted walkthrough of every decision path (no LLM or network needed)
python run_cli.py --demo
```

That's it — the `--demo` run exercises all 15 scenarios using the built-in
deterministic provider. To chat interactively:

```bash
python run_cli.py
```

> You can also install the package (`pip install -e .`) and use the `refund-agent`
> command, or run `python -m refund_agent.cli` with `src` on your `PYTHONPATH`.

## Using the CLI

Inside the interactive REPL you talk to the agent as if you were a customer. Mention
an order id (for example `ORD-10001`) and your reason:

```
you> My headphones ORD-10001 arrived defective, I'd like a refund.
   · [TOOL_CALL] Calling tool 'lookup_order' …
   · [TOOL_CALL] Calling tool 'check_refund_eligibility' …
   · [DECISION] Policy decision: APPROVED
   · [TOOL_CALL] Calling tool 'process_refund' …
agent> Good news — your refund of USD 129.99 has been approved and processed …
```

The lines beginning with `·` are the **live reasoning trace**. Slash-commands let you
inspect the system:

| Command | Description |
| --- | --- |
| `/help` | Show available commands |
| `/customers` | List all 15 mock customers |
| `/orders <CUST-ID>` | List a customer's orders |
| `/logs` | Show the full reasoning trace for the current session |
| `/policy [topic]` | Print the refund policy (optionally a single section) |
| `/new` | Start a fresh conversation |
| `/quit` | Exit |

Useful flags:

```bash
python run_cli.py --message "Refund ORD-10004, no longer needed"   # one-shot
python run_cli.py --demo                                           # scripted walkthrough
python run_cli.py --provider heuristic                             # force a provider
python run_cli.py --verbose                                        # full event detail
```

## The refund policy

The policy lives in two files under `data/`: a machine-readable
`refund_policy_rules.json` (thresholds the engine enforces) and a human-readable
`refund_policy.md` (what the agent quotes to customers). In summary:

- **Return window:** 30 days normally; **90 days** when the fault is the seller's
  (defective, damaged, wrong item, not as described, arrived late).
- **Order status:** only `DELIVERED` / `PARTIALLY_REFUNDED` orders are refundable;
  undelivered orders should be cancelled, and already-refunded orders are rejected.
- **Non-refundable:** digital goods and gift cards — always.
- **Hygiene items** (beauty, grocery): non-refundable once delivered unless faulty.
- **Final-sale / clearance:** non-refundable unless faulty.
- **Restocking fee:** opened electronics returned for a non-fault reason incur a
  **15%** fee (a partial approval).
- **High value:** refunds over **$500** require manual review (escalation).
- **Account standing:** suspended accounts and fraud-flagged accounts are escalated;
  closed accounts are denied.

Outcomes map to four decision types: **APPROVED**, **PARTIALLY_APPROVED** (fee
applied), **DENIED**, and **ESCALATED** (routed to a human). Rule severities aggregate
by precedence `BLOCK > REVIEW > ADJUST > PASS`.

## Mock data and decision scenarios

The CRM ships with 15 customers / orders, each chosen to exercise a distinct policy
path. The dates are stored as **relative offsets** (e.g. "delivered 8 days ago") and
resolved to absolute timestamps at load time, so the demo behaves identically no
matter when you run it.

| Order | Item | Reason | Expected outcome |
| --- | --- | --- | --- |
| ORD-10001 | Headphones ($129.99) | Defective | **APPROVED** — full refund |
| ORD-10002 | Shoes ($89) | Changed mind | **DENIED** — outside 30-day window |
| ORD-10003 | Smartwatch ($499) | Changed mind | **PARTIALLY_APPROVED** — 15% fee → $424.15 |
| ORD-10004 | Game code ($59.99) | No longer needed | **DENIED** — digital, non-refundable |
| ORD-10005 | Bag ($850) | Defective | **ESCALATED** — fraud flag + high value |
| ORD-10006 | Coffee maker ($75) | Defective | **APPROVED** — within 90-day fault window |
| ORD-10007 | Clearance tee ($15) | Changed mind | **DENIED** — final sale |
| ORD-10008 | Gift card ($100) | Changed mind | **DENIED** — gift card, non-refundable |
| ORD-10009 | Laptop ($1500) | Defective | **ESCALATED** — high value |
| ORD-10010 | Blender ($45) | Changed mind | **DENIED** — not delivered (shipped) |
| ORD-10011 | Earbuds ($60) | Defective | **DENIED** — already refunded |
| ORD-10012 | Jacket ($120) | Wrong item | **APPROVED** — full refund |
| ORD-10013 | Perfume ($40) | Changed mind | **DENIED** — hygiene restriction |
| ORD-10014 | Monitor ($200) | Defective | **ESCALATED** — suspended account |
| ORD-10015 | Cookware ($300, $50 already refunded) | Defective | **APPROVED** — remaining $250 |

Run `python run_cli.py --demo` to see all of these decided live.

## Optional: local LLM with Ollama

To drive the agent with a real open-source model instead of the heuristic planner:

```bash
# install Ollama from https://ollama.com, then:
ollama serve            # start the local server
ollama pull llama3.1    # pull an open-source model with tool-calling support

pip install -r requirements.txt          # httpx (already included) talks to Ollama
python run_cli.py --provider ollama
```

In the default `auto` mode the agent uses Ollama automatically when it is reachable
and falls back to the heuristic provider when it is not — so nothing breaks if Ollama
is offline. No API keys are ever used; all inference is local.

## Optional: HTTP API (admin dashboard)

A FastAPI server exposes the agent and the reasoning log over HTTP, which is what a
web chat UI and the real-time admin dashboard would build on:

```bash
pip install -r requirements-api.txt
uvicorn refund_agent.api.server:create_app --factory --reload
```

| Method & path | Purpose |
| --- | --- |
| `POST /chat` | Send a message, receive the agent's reply + metadata |
| `GET /admin/customers` | List mock customers |
| `GET /admin/sessions` | List active reasoning sessions |
| `GET /admin/sessions/{id}/logs` | Full reasoning trace for a session |
| `GET /healthz` | Liveness/readiness probe |

## Optional: voice pipeline

Fully open-source, fully local speech in/out (no external API):

```bash
pip install -r requirements-voice.txt
```

- **Speech-to-text:** `WhisperSpeechToText` (faster-whisper / Whisper).
- **Text-to-speech:** `Pyttsx3TextToSpeech` (offline, native engine).

Both sit behind the `SpeechToText` / `TextToSpeech` interfaces in
`refund_agent.voice`, so a voice front-end transcribes the caller, passes the text to
the same `RefundAgent`, and speaks the reply back.

## Configuration

All settings are environment variables prefixed with `REFUND_AGENT_` (and can live in
a `.env` file — copy `.env.example`). The most useful:

| Variable | Default | Description |
| --- | --- | --- |
| `REFUND_AGENT_LLM_PROVIDER` | `auto` | `auto` \| `ollama` \| `heuristic` |
| `REFUND_AGENT_OLLAMA_HOST` | `http://localhost:11434` | Local Ollama server URL |
| `REFUND_AGENT_OLLAMA_MODEL` | `llama3.1` | Model tag to use |
| `REFUND_AGENT_LOG_LEVEL` | `INFO` | Logging verbosity |
| `REFUND_AGENT_AGENT_MAX_ITERATIONS` | `8` | Tool-call loop guard |

## Testing

```bash
pip install -r requirements-dev.txt
pytest                  # or:  PYTHONPATH=src python -m pytest
```

The suite covers all 15 policy scenarios, restocking-fee math, amount clamping, CRM
loading and error handling, refund execution and idempotency, and the full agent loop
(driven by the deterministic provider, so the tests need no network or model).

## Project structure

```
ai-refund-agent/
├── data/
│   ├── crm_database.json          # 15 mock customers / orders
│   ├── refund_policy_rules.json   # machine-readable policy thresholds
│   └── refund_policy.md           # human-readable policy
├── src/refund_agent/
│   ├── app.py                     # composition root (DI container)
│   ├── cli.py                     # terminal interface (REPL + --demo)
│   ├── config.py                  # settings (env-driven)
│   ├── exceptions.py              # typed exception hierarchy
│   ├── logging_config.py          # structured logging setup
│   ├── models/                    # enums + pydantic domain models
│   ├── repositories/              # CRM + policy data access
│   ├── services/                  # PolicyEngine + RefundService
│   ├── llm/                       # provider interface, Ollama, heuristic, factory
│   ├── agent/                     # tools, registry, prompts, agent loop
│   ├── observability/             # streamable reasoning log
│   ├── api/                       # optional FastAPI server
│   └── voice/                     # optional STT/TTS adapters
├── tests/                         # pytest suite
├── requirements*.txt              # core / api / voice / dev
├── pyproject.toml
├── run_cli.py                     # zero-install launcher
└── README.md
```

## Scalability and fault tolerance

- **Fault tolerance.** The `auto` provider degrades gracefully from Ollama to the
  deterministic planner, so the agent always responds. Tool failures are captured and
  fed back to the planner as structured errors rather than crashing a turn; a hard
  iteration cap prevents runaway tool-calling; and the agent returns a safe
  human-handoff message on any unrecoverable error. All money-moving logic is
  idempotent per order.
- **Scalability.** The agent and services are stateless aside from clearly isolated
  stores. The in-memory `SessionStore` and `ReasoningLog` sit behind small interfaces
  and can be swapped for Redis/Kafka/a database to run many stateless agent workers
  behind a load balancer. The CRM repository is the only component that knows how data
  is stored, so replacing the JSON file with a real database is a localized change. The
  policy engine is pure and deterministic, making evaluations trivially parallelizable
  and cacheable.

---

*Built as a backend reference implementation. Replace the mock CRM and the
"execute refund" step with a real datastore and payment gateway to take it to
production.*
