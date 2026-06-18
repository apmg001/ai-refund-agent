/**
 * Views.
 *
 * Pure rendering: each function takes a DOM element and a slice of state and paints
 * it. Views never call the API and never mutate the store — they are a one-way
 * projection of state onto the DOM, which keeps rendering predictable and testable.
 */

/** Map an event_type to its CSS modifier and short display tag. */
const EVENT_STYLE = {
  USER_MESSAGE: { cls: "event--user", tag: "USER" },
  LLM_REQUEST: { cls: "event--llm-req", tag: "LLM →" },
  LLM_RESPONSE: { cls: "event--llm-res", tag: "LLM ←" },
  TOOL_CALL: { cls: "event--tool", tag: "TOOL" },
  TOOL_RESULT: { cls: "event--result", tag: "RESULT" },
  TOOL_ERROR: { cls: "event--error", tag: "TOOL!" },
  DECISION: { cls: "event--decision", tag: "DECISION" },
  GUARDRAIL: { cls: "event--guardrail", tag: "GUARDRAIL" },
  AGENT_RESPONSE: { cls: "event--agent", tag: "REPLY" },
  ERROR: { cls: "event--error", tag: "ERROR" },
};

/** Escape user/model text before inserting as HTML. */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

/** Render a one-line, human-readable summary of an event's detail. */
function summariseDetail(event) {
  const d = event.detail || {};
  switch (event.event_type) {
    case "TOOL_CALL":
      return `${d.tool || ""}(${formatArgs(d.arguments)})`;
    case "TOOL_RESULT":
      return d.result ? compact(d.result) : "";
    case "DECISION":
      return d.decision ? `→ ${d.decision}` : "";
    case "GUARDRAIL":
      return d.order_id ? `order ${d.order_id}` : "";
    case "USER_MESSAGE":
      return d.message ? `“${d.message}”` : "";
    case "AGENT_RESPONSE":
      return d.reply ? compact(d.reply) : "";
    default:
      return "";
  }
}

function formatArgs(args) {
  if (!args || typeof args !== "object") return "";
  return Object.entries(args)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
}

function compact(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 160 ? `${text.slice(0, 160)}…` : text;
}

/**
 * Render the chat log.
 * @param {HTMLElement} el Container element.
 * @param {object} state Application state.
 */
export function renderChat(el, state) {
  if (state.messages.length === 0 && !state.sending) return; // keep empty state
  el.innerHTML = "";
  for (const msg of state.messages) {
    if (msg.role === "system") {
      const divider = document.createElement("div");
      divider.className = "divider";
      divider.innerHTML = `<span>${escapeHtml(msg.text)}</span>`;
      el.appendChild(divider);
      continue;
    }
    const wrap = document.createElement("div");
    wrap.className = `msg msg--${msg.role}`;
    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    bubble.innerHTML = escapeHtml(msg.text);
    wrap.appendChild(bubble);
    el.appendChild(wrap);
  }
  if (state.sending) {
    const wrap = document.createElement("div");
    wrap.className = "msg msg--agent";
    wrap.innerHTML =
      '<div class="msg__bubble"><span class="typing"><span></span><span></span><span></span></span></div>';
    el.appendChild(wrap);
  }
  el.scrollTop = el.scrollHeight;
}

/**
 * Render the live reasoning trace.
 * @param {HTMLElement} el The <ol> trace container.
 * @param {HTMLElement} emptyEl The empty-state element to toggle.
 * @param {object} state Application state.
 */
export function renderTrace(el, emptyEl, state) {
  if (state.events.length === 0) {
    if (emptyEl) emptyEl.style.display = "";
    return;
  }
  if (emptyEl) emptyEl.style.display = "none";

  el.innerHTML = "";
  for (const ev of state.events) {
    const style = EVENT_STYLE[ev.event_type] || { cls: "", tag: ev.event_type };
    const li = document.createElement("li");
    li.className = `event ${style.cls}`;

    const detail = summariseDetail(ev);
    li.innerHTML = `
      <div class="event__row">
        <span class="event__seq">${ev.sequence}</span>
        <span class="event__tag">${escapeHtml(style.tag)}</span>
        <span class="event__title">${escapeHtml(ev.title)}</span>
      </div>
      ${detail ? `<div class="event__detail">${escapeHtml(detail)}</div>` : ""}
    `;
    el.appendChild(li);
  }
  el.scrollTop = el.scrollHeight;
}

/**
 * Render the top-bar status pills.
 * @param {{provider:HTMLElement, session:HTMLElement}} els
 * @param {object} state
 */
export function renderStatus(els, state) {
  els.provider.textContent = state.provider;
  els.session.textContent = state.sessionId ? state.sessionId.slice(0, 8) : "—";
}

/**
 * Render the live-feed connection indicator.
 * @param {HTMLElement} connEl
 * @param {HTMLElement} labelEl
 * @param {object} state
 */
export function renderConnection(connEl, labelEl, state) {
  const map = {
    connecting: ["", "connecting"],
    live: ["conn--live", "live"],
    polling: ["conn--polling", "polling"],
    offline: ["conn--offline", "offline"],
  };
  const [cls, label] = map[state.connection] || ["", state.connection];
  connEl.className = `conn ${cls}`;
  labelEl.textContent = label;
}

/**
 * Render sample-order chips from the customer list.
 * @param {HTMLElement} el
 * @param {Array} customers
 * @param {(orderId:string) => void} onPick
 */
export function renderChips(el, customers, onPick) {
  el.innerHTML = "";
  const samples = customers.flatMap((c) => c.order_ids || []).slice(0, 6);
  for (const orderId of samples) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.innerHTML = `<b>${escapeHtml(orderId)}</b>`;
    chip.addEventListener("click", () => onPick(orderId));
    el.appendChild(chip);
  }
}