/**
 * Controller / entry point.
 *
 * Wires the decoupled pieces together: it owns the API client, the store, and the
 * voice input, binds DOM events, and subscribes views to the store. It is the only
 * module that both mutates state and touches the DOM directly for event binding —
 * everything else stays single-purpose.
 *
 * Data flow:
 *   user action ─▶ controller ─▶ ApiClient ─▶ Store (mutate) ─▶ views (render)
 *   backend SSE ─▶ ApiClient stream ─▶ Store.addEvent ─▶ trace view (render)
 */

import { CONFIG } from "./config.js";
import { ApiClient } from "./api.js";
import { Store } from "./store.js";
import { VoiceInput } from "./voice.js";
import {
  renderChat,
  renderTrace,
  renderStatus,
  renderConnection,
  renderChips,
} from "./views.js";

/* ----------------------------- DOM handles ----------------------------- */
const dom = {
  chatLog: document.getElementById("chatLog"),
  chatEmpty: document.getElementById("chatEmpty"),
  chips: document.getElementById("chips"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  sendBtn: document.getElementById("sendBtn"),
  micBtn: document.getElementById("micBtn"),
  trace: document.getElementById("trace"),
  traceEmpty: document.getElementById("traceEmpty"),
  providerLabel: document.getElementById("providerLabel"),
  sessionLabel: document.getElementById("sessionLabel"),
  conn: document.getElementById("conn"),
  connLabel: document.getElementById("connLabel"),
};

/* ----------------------------- Wiring ----------------------------- */
const api = new ApiClient();
const store = new Store();
const voice = new VoiceInput();

/** Live reasoning feed handles, tracked so a reset can tear them down cleanly. */
let reasoningStream = null;
let pollTimer = null;

/** Tools whose presence in a turn means a refund request reached a verdict. */
const CONCLUDING_TOOLS = ["check_refund_eligibility", "process_refund"];

/** Generate a reasonably-unique client session id. */
function newSessionId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID().replace(/-/g, "").slice(0, 16);
  return Math.random().toString(16).slice(2, 18);
}

/* Render on every state change. */
store.subscribe((state) => {
  renderChat(dom.chatLog, state);
  renderTrace(dom.trace, dom.traceEmpty, state);
  renderStatus({ provider: dom.providerLabel, session: dom.sessionLabel }, state);
  renderConnection(dom.conn, dom.connLabel, state);
});

/* ----------------------------- Actions ----------------------------- */

/** Send the current input as a customer message and record the agent's reply. */
async function send(text) {
  const message = (text ?? dom.input.value).trim();
  if (!message || store.state.sending) return;

  // If the previous refund request already concluded, the next message silently
  // begins a brand-new conversation — a clean slate, exactly like the first one.
  if (store.state.completed) {
    startFreshConversation();
  }

  store.addMessage("customer", message);
  dom.input.value = "";
  autoGrow();
  store.setSending(true);
  dom.sendBtn.disabled = true;

  try {
    const result = await api.sendMessage(store.state.sessionId, message);
    store.addMessage("agent", result.reply);

    // A turn that ran eligibility or processing has reached a verdict: mark the
    // request complete and show a clear divider. The next message starts fresh.
    const concluded = (result.tool_calls || []).some((t) => CONCLUDING_TOOLS.includes(t));
    if (concluded) {
      store.addMessage("system", "Request complete · send a message to start a new one");
      store.setCompleted(true);
    }
  } catch (err) {
    store.addMessage(
      "agent",
      "I couldn't reach the refund service. Please make sure the backend is running, then try again."
    );
    console.error(err);
  } finally {
    store.setSending(false);
    dom.sendBtn.disabled = false;
    dom.input.focus();
  }
}

/** Tear down the current conversation and start a fresh session + reasoning feed. */
function startFreshConversation() {
  stopReasoningFeed();
  dom.chatLog.innerHTML = "";
  dom.trace.innerHTML = "";
  if (dom.traceEmpty) {
    dom.trace.appendChild(dom.traceEmpty);
    dom.traceEmpty.style.display = "";
  }
  const sessionId = newSessionId();
  store.resetConversation(sessionId);
  startReasoningFeed(sessionId);
}

/** Stop any active SSE stream and polling timer. */
function stopReasoningFeed() {
  if (reasoningStream) {
    reasoningStream.close();
    reasoningStream = null;
  }
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

/**
 * Subscribe to the live reasoning feed, falling back to polling if SSE drops.
 * @param {string} sessionId
 */
function startReasoningFeed(sessionId) {
  const beginPolling = () => {
    if (pollTimer) return;
    store.setConnection("polling");
    pollTimer = setInterval(async () => {
      const events = await api.fetchReasoning(sessionId);
      events.forEach((e) => store.addEvent(e));
    }, CONFIG.POLL_INTERVAL_MS);
  };

  try {
    reasoningStream = api.openReasoningStream(
      sessionId,
      (event) => store.addEvent(event),
      (status) => {
        if (status === "live") {
          store.setConnection("live");
          if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
          }
        } else {
          beginPolling();
        }
      }
    );
  } catch {
    beginPolling();
  }
}

/* ----------------------------- Voice ----------------------------- */
function setupVoice() {
  if (!voice.supported) {
    dom.micBtn.classList.add("composer__mic--off");
    dom.micBtn.title = "Voice input isn’t supported in this browser";
    dom.micBtn.disabled = true;
    return;
  }
  dom.micBtn.addEventListener("click", () => {
    if (voice.listening) {
      voice.stop();
      return;
    }
    dom.micBtn.classList.add("composer__mic--on");
    voice.start({
      onResult: (text) => {
        dom.input.value = dom.input.value ? `${dom.input.value} ${text}` : text;
        autoGrow();
      },
      onEnd: () => dom.micBtn.classList.remove("composer__mic--on"),
      onError: () => dom.micBtn.classList.remove("composer__mic--on"),
    });
  });
}

/* ----------------------------- Input affordances ----------------------------- */
function autoGrow() {
  dom.input.style.height = "auto";
  dom.input.style.height = `${Math.min(dom.input.scrollHeight, 120)}px`;
}

function setupComposer() {
  dom.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });
  dom.input.addEventListener("input", autoGrow);
  dom.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
}

/* ----------------------------- Bootstrap ----------------------------- */
async function init() {
  const sessionId = newSessionId();
  store.setSession(sessionId);

  setupComposer();
  setupVoice();
  startReasoningFeed(sessionId);

  // Health + provider label (best-effort; UI still works if it fails).
  try {
    const health = await api.health();
    store.setProvider(health.llm_provider || "unknown");
  } catch {
    store.setProvider("offline");
  }

  // Sample order chips.
  try {
    const customers = await api.customers();
    store.setCustomers(customers);
    renderChips(dom.chips, customers, (orderId) => {
      dom.input.value = `I'd like a refund for ${orderId}`;
      dom.input.focus();
      autoGrow();
    });
  } catch {
    /* chips are optional */
  }

  dom.input.focus();
}

init();