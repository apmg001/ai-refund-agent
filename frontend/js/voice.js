/**
 * Voice input.
 *
 * A thin wrapper around the browser's built-in SpeechRecognition (Web Speech API),
 * so the microphone feature is fully self-contained and degrades gracefully where
 * the API is unavailable. No external service or key is used — recognition is the
 * browser's own. Kept isolated so the rest of the app depends only on a small
 * start/stop/onResult surface.
 */

export class VoiceInput {
  constructor() {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    /** @type {boolean} Whether the browser supports speech recognition. */
    this.supported = Boolean(SpeechRecognition);
    this._recognition = null;
    this._listening = false;

    if (this.supported) {
      const recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      this._recognition = recognition;
    }
  }

  /** @returns {boolean} Whether recognition is currently active. */
  get listening() {
    return this._listening;
  }

  /**
   * Start listening. Callbacks fire for the final transcript, end, and errors.
   * @param {{onResult:(text:string)=>void, onEnd?:()=>void, onError?:(err:any)=>void}} handlers
   */
  start({ onResult, onEnd, onError }) {
    if (!this.supported || this._listening) return;
    const recognition = this._recognition;

    recognition.onresult = (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript || "";
      if (transcript) onResult(transcript);
    };
    recognition.onerror = (event) => onError && onError(event.error);
    recognition.onend = () => {
      this._listening = false;
      onEnd && onEnd();
    };

    try {
      recognition.start();
      this._listening = true;
    } catch (err) {
      this._listening = false;
      onError && onError(err);
    }
  }

  /** Stop listening (the current transcript, if any, is still delivered). */
  stop() {
    if (this.supported && this._listening) {
      this._recognition.stop();
    }
  }
}