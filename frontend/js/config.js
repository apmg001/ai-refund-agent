/**
 * Application configuration.
 *
 * The single place that knows where the backend lives. Kept separate so the API
 * base URL can be changed for a different deployment without touching any logic.
 * Override at runtime by setting `window.REFUND_AGENT_API_BASE` before this loads.
 */

export const CONFIG = Object.freeze({
  /** Base URL of the refund-agent HTTP API. */
  API_BASE:
    (typeof window !== "undefined" && window.REFUND_AGENT_API_BASE) ||
    "http://localhost:8000",

  /** Polling interval (ms) used only if the live SSE stream is unavailable. */
  POLL_INTERVAL_MS: 1000,
});