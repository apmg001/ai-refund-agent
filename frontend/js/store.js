/**
 * Application store.
 *
 * The single source of truth for UI state, with a minimal observer (pub/sub) so
 * views re-render when state changes. Holding state here — rather than in the DOM —
 * keeps the controller, the API client, and the views decoupled: the controller
 * mutates state, the store notifies, and views read state. No view talks to another.
 */

export class Store {
  constructor() {
    /** @type {{sessionId:string, messages:Array, events:Array, eventSeqs:Set<number>, connection:string, provider:string, sending:boolean, customers:Array}} */
    this._state = {
      sessionId: "",
      messages: [],
      events: [],
      eventSeqs: new Set(),
      connection: "connecting",
      provider: "connecting…",
      sending: false,
      customers: [],
      completed: false,
    };
    /** @type {Array<(state:object) => void>} */
    this._listeners = [];
  }

  /** @returns {object} A shallow snapshot of current state. */
  get state() {
    return this._state;
  }

  /**
   * Subscribe to state changes.
   * @param {(state:object) => void} listener Called after every change.
   * @returns {() => void} Unsubscribe function.
   */
  subscribe(listener) {
    this._listeners.push(listener);
    return () => {
      this._listeners = this._listeners.filter((l) => l !== listener);
    };
  }

  /** @param {string} sessionId */
  setSession(sessionId) {
    this._state.sessionId = sessionId;
    this._emit();
  }

  /** @param {string} provider */
  setProvider(provider) {
    this._state.provider = provider;
    this._emit();
  }

  /** @param {string} connection One of 'connecting' | 'live' | 'polling' | 'offline'. */
  setConnection(connection) {
    this._state.connection = connection;
    this._emit();
  }

  /** @param {Array} customers */
  setCustomers(customers) {
    this._state.customers = customers;
    this._emit();
  }

  /**
   * Append a chat message.
   * @param {'customer'|'agent'} role
   * @param {string} text
   */
  addMessage(role, text) {
    this._state.messages.push({ role, text, at: Date.now() });
    this._emit();
  }

  /** @param {boolean} sending Whether a reply is in flight (drives the typing indicator). */
  setSending(sending) {
    this._state.sending = sending;
    this._emit();
  }

  /**
   * Mark the current refund request as concluded (a policy decision was reached).
   * @param {boolean} completed
   */
  setCompleted(completed) {
    this._state.completed = completed;
    this._emit();
  }

  /**
   * Start a brand-new conversation: a fresh session id and an empty transcript and
   * reasoning trace, as if the page had just loaded.
   * @param {string} newSessionId
   */
  resetConversation(newSessionId) {
    this._state.sessionId = newSessionId;
    this._state.messages = [];
    this._state.events = [];
    this._state.eventSeqs = new Set();
    this._state.completed = false;
    this._state.connection = "connecting";
    this._emit();
  }

  /**
   * Add a reasoning event, de-duplicated by sequence number.
   *
   * De-duplication matters because the live stream replays existing events on
   * connect, which can overlap with events already received.
   * @param {object} event A reasoning event with a numeric `sequence`.
   */
  addEvent(event) {
    const seq = event.sequence;
    if (this._state.eventSeqs.has(seq)) return;
    this._state.eventSeqs.add(seq);
    this._state.events.push(event);
    this._state.events.sort((a, b) => a.sequence - b.sequence);
    this._emit();
  }

  /** @private */
  _emit() {
    for (const listener of this._listeners) {
      listener(this._state);
    }
  }
}