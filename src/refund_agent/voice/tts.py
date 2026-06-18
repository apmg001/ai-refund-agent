"""Text-to-speech implementation.

Uses the open-source, offline ``pyttsx3`` engine (which wraps the platform's native
TTS) so speech synthesis requires no external API. The engine is created lazily and
reused across calls.
"""

from __future__ import annotations

from ..exceptions import VoiceError
from ..logging_config import get_logger
from .base import TextToSpeech

__all__ = ["Pyttsx3TextToSpeech"]

_logger = get_logger(__name__)


class Pyttsx3TextToSpeech(TextToSpeech):
    """Local, offline TTS backed by pyttsx3."""

    def __init__(self, rate: int = 175, volume: float = 1.0) -> None:
        """Initialize the synthesizer.

        Args:
            rate: Speech rate in words per minute.
            volume: Output volume in the range ``[0.0, 1.0]``.
        """
        self._rate = rate
        self._volume = volume
        self._engine = None  # lazily created

    def _ensure_engine(self):
        """Create the pyttsx3 engine on first use.

        Returns:
            The initialized engine.

        Raises:
            VoiceError: If pyttsx3 is not installed or the engine fails to start.
        """
        if self._engine is not None:
            return self._engine
        try:
            import pyttsx3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise VoiceError(
                "pyttsx3 is required for text-to-speech. Install voice extras with "
                "`pip install -r requirements-voice.txt`."
            ) from exc
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
        except Exception as exc:  # noqa: BLE001 - surface as domain error
            raise VoiceError(f"Failed to initialize TTS engine: {exc}") from exc
        self._engine = engine
        return engine

    def speak(self, text: str) -> None:
        """Speak ``text`` aloud through the default audio device.

        Args:
            text: The text to speak.

        Raises:
            VoiceError: If synthesis or playback fails.
        """
        engine = self._ensure_engine()
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:  # noqa: BLE001 - surface as domain error
            raise VoiceError(f"TTS playback failed: {exc}") from exc

    def synthesize_to_file(self, text: str, output_path: str) -> str:
        """Synthesize ``text`` to an audio file.

        Args:
            text: The text to synthesize.
            output_path: Destination file path.

        Returns:
            The written file path.

        Raises:
            VoiceError: If synthesis fails.
        """
        engine = self._ensure_engine()
        try:
            engine.save_to_file(text, output_path)
            engine.runAndWait()
        except Exception as exc:  # noqa: BLE001 - surface as domain error
            raise VoiceError(f"TTS file synthesis failed: {exc}") from exc
        return output_path
