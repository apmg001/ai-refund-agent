/**
 * API client.
 *
 * The transport layer: every call to the refund-agent backend goes through here,
 * and nothing in this module touches the DOM. This keeps the network contract in
 * one decoupled place so views and controllers depend on a small, stable interface
 * rather than on `fetch` details.
 */

import { CONFIG } from "./config.js";

export class ApiClient {
  /**
   * @param {string} [baseUrl] Base URL of the backend API.
   */
  constructor(baseUrl = CONFIG.API_BASE) {
    this._base = baseUrl.replace(/\/$/, "");
  }

  /**
   * Fetch service health (status + active LLM provider).
   * @returns {Promise<{status:string, app_name:string, llm_provider:string}>}
   */
  async health() {
    return this._getJson("/healthz");
  }

  /**
   * List the mock customers (used to surface sample order IDs).
   * @returns {Promise<Array<object>>}
   */
  async customers() {
    return this._getJson("/admin/customers");
  }

  /**
   * Send a customer message and return the agent's reply.
   * @param {string} sessionId Conversation id.
   * @param {string} message The customer's message.
   * @returns {Promise<{session_id:string, reply:string, iterations:number, tool_calls:string[]}>}
   */
  async sendMessage(sessionId, message) {
    const res = await fetch(`${this._base}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    if (!res.ok) {
      throw new Error(`Chat request failed (${res.status})`);
    }
    return res.json();
  }

  /**
   * Open a live Server-Sent Events stream of reasoning events for a session.
   *
   * @param {string} sessionId Conversation id.
   * @param {(event:object) => void} onEvent Called for each reasoning event.
   * @param {(status:'live'|'offline') => void} [onStatus] Connection-status callback.
   * @returns {EventSource} The live stream (call `.close()` to stop).
   */
  openReasoningStream(sessionId, onEvent, onStatus) {
    const url = `${this._base}/admin/sessions/${encodeURIComponent(sessionId)}/stream`;
    const source = new EventSource(url);
    source.onopen = () => onStatus && onStatus("live");
    source.onmessage = (msg) => {
      try {
        onEvent(JSON.parse(msg.data));
      } catch {
        /* ignore keep-alive / malformed frames */
      }
    };
    source.onerror = () => onStatus && onStatus("offline");
    return source;
  }

  /**
   * Fetch the full reasoning trace for a session (polling fallback for SSE).
   * @param {string} sessionId Conversation id.
   * @returns {Promise<Array<object>>} The recorded events (empty if none yet).
   */
  async fetchReasoning(sessionId) {
    try {
      const data = await this._getJson(
        `/admin/sessions/${encodeURIComponent(sessionId)}/logs`
      );
      return data.events || [];
    } catch {
      return [];
    }
  }

  /** @private */
  async _getJson(path) {
    const res = await fetch(`${this._base}${path}`);
    if (!res.ok) {
      throw new Error(`GET ${path} failed (${res.status})`);
    }
    return res.json();
  }
}