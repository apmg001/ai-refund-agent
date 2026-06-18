"""LLM provider factory.

Builds the configured :class:`LLMProvider`, encapsulating the fault-tolerant
"auto" behaviour: prefer the local open-source Ollama model when it is reachable, but
fall back to the dependency-free deterministic heuristic provider otherwise. This is
what lets the system run end-to-end from the terminal with zero setup while still
supporting a real local LLM when one is available.
"""

from __future__ import annotations

from ..config import Settings
from ..exceptions import LLMProviderError
from ..logging_config import get_logger
from .base import LLMProvider
from .heuristic_provider import HeuristicProvider

__all__ = ["build_llm_provider"]

_logger = get_logger(__name__)


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Construct the LLM provider selected by ``settings.llm_provider``.

    Behaviour by setting:
        * ``heuristic`` -> always the deterministic :class:`HeuristicProvider`.
        * ``ollama``    -> the :class:`OllamaProvider`; raises if it is unreachable.
        * ``auto``      -> Ollama if its health check passes, else the heuristic
          provider (with a warning). Never raises for transport problems.

    Args:
        settings: The application settings.

    Returns:
        A ready-to-use :class:`LLMProvider`.

    Raises:
        LLMProviderError: If ``llm_provider == "ollama"`` and Ollama cannot be reached
            or its optional dependency is missing.
    """
    provider = settings.llm_provider

    if provider == "heuristic":
        _logger.info("Using deterministic heuristic LLM provider")
        return HeuristicProvider()

    if provider == "ollama":
        return _build_ollama_or_raise(settings)

    # provider == "auto"
    try:
        ollama = _build_ollama(settings)
        if ollama.health_check():
            _logger.info("Ollama reachable; using model '%s'", settings.ollama_model)
            return ollama
        _logger.warning(
            "Ollama not reachable at %s; falling back to heuristic provider",
            settings.ollama_host,
        )
    except LLMProviderError as exc:
        _logger.warning("Ollama unavailable (%s); falling back to heuristic provider", exc)

    return HeuristicProvider()


def _build_ollama(settings: Settings) -> LLMProvider:
    """Instantiate the Ollama provider (imported lazily to keep httpx optional)."""
    from .ollama_provider import OllamaProvider

    return OllamaProvider(
        host=settings.ollama_host,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )


def _build_ollama_or_raise(settings: Settings) -> LLMProvider:
    """Build the Ollama provider and verify reachability, raising on failure."""
    ollama = _build_ollama(settings)
    if not ollama.health_check():
        raise LLMProviderError(
            f"Ollama provider was requested but the server at '{settings.ollama_host}' "
            "is not reachable. Start Ollama (`ollama serve`) and pull the model, or set "
            "REFUND_AGENT_LLM_PROVIDER=heuristic."
        )
    _logger.info("Ollama reachable; using model '%s'", settings.ollama_model)
    return ollama
