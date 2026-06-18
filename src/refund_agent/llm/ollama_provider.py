"""Ollama LLM provider.

Talks to a locally-running `Ollama <https://ollama.com>`_ server (default
``http://localhost:11434``) to drive an open-source model such as ``llama3.1`` or
``qwen2.5``. Ollama runs entirely on the user's machine — there is no third-party
API key and no data leaves the host — which satisfies the "open-source, no external
API" requirement while still exercising real LLM-driven tool calling.

The provider is fault-tolerant: connection and timeout failures raise typed errors
that the factory uses to fall back to the heuristic provider.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import LLMProviderError, LLMTimeoutError
from ..logging_config import get_logger
from .base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec

__all__ = ["OllamaProvider"]

_logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        """Initialize the provider.

        Args:
            host: Base URL of the Ollama server.
            model: Model tag to use (must be pulled in Ollama beforehand).
            timeout_seconds: Per-request timeout.
            max_retries: Number of retry attempts on transient transport errors.

        Raises:
            LLMProviderError: If the optional ``httpx`` dependency is unavailable.
        """
        try:
            import httpx  # local import keeps httpx optional for heuristic-only use
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMProviderError(
                "The 'httpx' package is required for the Ollama provider. "
                "Install it or set REFUND_AGENT_LLM_PROVIDER=heuristic."
            ) from exc

        self._httpx = httpx
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    def health_check(self) -> bool:
        """Return ``True`` if the Ollama server responds to a tags request."""
        try:
            resp = self._httpx.get(f"{self._host}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001 - health check must never raise
            _logger.warning("Ollama health check failed: %s", exc)
            return False

    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        """Call the Ollama ``/api/chat`` endpoint and parse the result.

        Args:
            messages: The conversation so far.
            tools: Tools available to the model.

        Returns:
            The parsed :class:`LLMResponse`.

        Raises:
            LLMTimeoutError: If the request times out.
            LLMProviderError: For transport or protocol failures.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_provider_dict() for m in messages],
            "tools": [t.to_provider_dict() for t in tools],
            "stream": False,
            "options": {"temperature": 0.1},
        }

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._httpx.post(
                    f"{self._host}/api/chat",
                    json=payload,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                return self._parse_response(resp.json())
            except self._httpx.TimeoutException as exc:
                raise LLMTimeoutError(
                    f"Ollama request timed out after {self._timeout}s."
                ) from exc
            except self._httpx.HTTPStatusError as exc:
                # 4xx are not retryable; 5xx might be transient.
                if exc.response.status_code < 500:
                    raise LLMProviderError(
                        f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
                    ) from exc
                last_error = exc
                _logger.warning("Ollama 5xx on attempt %d/%d", attempt, self._max_retries)
            except self._httpx.HTTPError as exc:
                last_error = exc
                _logger.warning("Ollama transport error on attempt %d/%d: %s", attempt, self._max_retries, exc)

        raise LLMProviderError(f"Ollama request failed after {self._max_retries} attempts: {last_error}")

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        """Translate an Ollama chat response into an :class:`LLMResponse`."""
        message = data.get("message", {})
        content = message.get("content", "") or ""

        tool_calls: list[ToolCall] = []
        for raw in message.get("tool_calls", []) or []:
            function = raw.get("function", {})
            name = function.get("name")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                # Some models emit a JSON string; tolerate it.
                import json

                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if name:
                tool_calls.append(ToolCall(name=name, arguments=arguments or {}))

        return LLMResponse(content=content, tool_calls=tuple(tool_calls))
