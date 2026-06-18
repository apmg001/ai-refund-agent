"""Speech-to-text implementation.

Uses the open-source ``faster-whisper`` model (a CTranslate2 reimplementation of
OpenAI Whisper) so transcription runs fully locally with no external API. The model
is loaded lazily on first use and cached, since loading is comparatively expensive.
"""

from __future__ import annotations

from ..exceptions import VoiceError
from ..logging_config import get_logger
from .base import SpeechToText

__all__ = ["WhisperSpeechToText"]

_logger = get_logger(__name__)


class WhisperSpeechToText(SpeechToText):
    """Local STT backed by faster-whisper."""

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8") -> None:
        """Initialize the transcriber.

        Args:
            model_size: Whisper model size (e.g. ``tiny``, ``base``, ``small``).
            device: Inference device (``cpu`` or ``cuda``).
            compute_type: CTranslate2 compute type (e.g. ``int8`` for CPU).
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None  # lazily loaded

    def _ensure_model(self) -> None:
        """Load the Whisper model on first use.

        Raises:
            VoiceError: If faster-whisper is not installed or the model fails to load.
        """
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise VoiceError(
                "faster-whisper is required for speech-to-text. Install voice extras "
                "with `pip install -r requirements-voice.txt`."
            ) from exc
        try:
            _logger.info("Loading Whisper model '%s' on %s", self._model_size, self._device)
            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        except Exception as exc:  # noqa: BLE001 - surface as domain error
            raise VoiceError(f"Failed to load Whisper model: {exc}") from exc

    def transcribe(self, audio_path: str) -> str:
        """Transcribe ``audio_path`` to text.

        Args:
            audio_path: Path to an audio file.

        Returns:
            The transcribed text (segments concatenated).

        Raises:
            VoiceError: If transcription fails or the dependency is missing.
        """
        self._ensure_model()
        assert self._model is not None  # for type-checkers
        try:
            segments, _info = self._model.transcribe(audio_path)
            text = " ".join(segment.text.strip() for segment in segments)
        except Exception as exc:  # noqa: BLE001 - surface as domain error
            raise VoiceError(f"Transcription failed for '{audio_path}': {exc}") from exc
        return text.strip()
