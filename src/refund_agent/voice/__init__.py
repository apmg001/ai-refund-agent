"""Voice pipeline package (optional bonus).

Provides speech-to-text and text-to-speech adapters behind small interfaces so a
voice front-end can be layered on top of the same agent. All heavy, optional
dependencies (faster-whisper, sounddevice, pyttsx3) are imported lazily, so importing
this package never fails on a minimal install. Install the extras with
``pip install -r requirements-voice.txt`` to enable them.
"""

from __future__ import annotations

from .base import SpeechToText, TextToSpeech

__all__ = ["SpeechToText", "TextToSpeech"]
