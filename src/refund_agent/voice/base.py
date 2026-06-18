"""Voice interfaces.

Abstract base classes for speech-to-text and text-to-speech so the agent's voice
front-end depends on stable interfaces rather than concrete engines. Swapping
faster-whisper for another STT engine, or pyttsx3 for a neural TTS, is a localized
change behind these contracts.
"""

from __future__ import annotations

import abc

__all__ = ["SpeechToText", "TextToSpeech"]


class SpeechToText(abc.ABC):
    """Convert spoken audio into text."""

    @abc.abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to a readable audio file (e.g. WAV).

        Returns:
            The transcribed text.

        Raises:
            VoiceError: If transcription fails or dependencies are missing.
        """
        raise NotImplementedError


class TextToSpeech(abc.ABC):
    """Convert text into spoken audio."""

    @abc.abstractmethod
    def speak(self, text: str) -> None:
        """Synthesize and play ``text`` aloud.

        Args:
            text: The text to speak.

        Raises:
            VoiceError: If synthesis fails or dependencies are missing.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def synthesize_to_file(self, text: str, output_path: str) -> str:
        """Synthesize ``text`` to an audio file.

        Args:
            text: The text to synthesize.
            output_path: Where to write the audio file.

        Returns:
            The path to the written audio file.

        Raises:
            VoiceError: If synthesis fails or dependencies are missing.
        """
        raise NotImplementedError
