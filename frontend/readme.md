# Refund Desk — Frontend

A decoupled, dependency-free web UI for the AI Refund Support Agent. Plain HTML, CSS,
and ES-module JavaScript — no build step, no framework. It talks to the backend over
HTTP and streams the agent's reasoning live via Server-Sent Events.

It has two halves:

- **Customer chat** (left) — a calm conversational panel with an optional microphone
  button (browser-native speech recognition, no external service).
- **Live reasoning console** (right) — the "glass box": every tool call, policy
  decision, and guardrail intervention streams in real time, color-coded by type,
  with decisions and guardrail moments emphasized.

## Architecture

The code is split by responsibility so each piece is independently understandable:

```
js/
├── config.js   # where the backend lives (the only place that knows the API URL)
├── api.js      # transport layer — all HTTP / SSE, no DOM
├── store.js    # single source of truth for UI state + pub/sub
├── views.js    # pure rendering — state in, DOM out, no side effects
├── voice.js    # microphone (Web Speech API), isolated and optional
└── app.js      # controller — binds events, wires the pieces together
```

Data flows one way: `user action → controller → api → store → views`. Views never
call the API and never mutate state, which keeps everything decoupled and predictable.

## Running it

The frontend and backend run as **separate** processes (they're decoupled). From the
project root:

```bash
# 1. Start the backend API (terminal 1)
pip install -r requirements-api.txt
PYTHONPATH=src python -m uvicorn refund_agent.api.server:create_app --factory --port 8000

# 2. Serve the frontend (terminal 2)
python -m http.server 5500 -d frontend
```

Then open <http://localhost:5500>.

If your backend runs on a different host/port, set it before the app loads — either
edit `js/config.js`, or add this above the module script in `index.html`:

```html
<script>window.REFUND_AGENT_API_BASE = "http://localhost:8000";</script>
```

## Notes

- **Real-time feed:** the console subscribes to `/admin/sessions/{id}/stream` (SSE).
  If that connection can't be established it automatically falls back to polling
  `/admin/sessions/{id}/logs`, so the dashboard keeps working either way.
- **Voice** uses the browser's built-in speech recognition (best support in Chrome).
  Where it isn't available the mic button is disabled and the rest of the UI is
  unaffected.
- **Scaling note:** the live feed is an in-process observer today; for many concurrent
  users the same SSE contract can be backed by a shared broker (e.g. Redis pub/sub)
  with no change to this frontend.