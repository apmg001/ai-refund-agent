"""LLM provider interface and shared message types.

The agent loop is written against this provider-agnostic interface, so swapping the
underlying model (a local open-source Ollama model, the deterministic heuristic
planner, or — in future — any other backend) requires no changes to the agent.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Message",
    "ToolSpec",
    "ToolCall",
    "LLMResponse",
    "LLMProvider",
]


@dataclass
class Message:
    """A single chat message.

    Attributes:
        role: One of ``system``, ``user``, ``assistant``, or ``tool``.
        content: Text content of the message.
        name: Tool name (only for ``tool`` messages).
        tool_calls: Tool calls requested by the assistant (only for ``assistant``).
    """

    role: str
    content: str = ""
    name: str | None = None
    tool_calls: list["ToolCall"] = field(default_factory=list)

    def to_provider_dict(self) -> dict[str, Any]:
        """Return a dict shaped for chat-completion style APIs."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return payload


@dataclass(frozen=True)
class ToolSpec:
    """Declarative description of a tool exposed to the LLM.

    Attributes:
        name: Unique tool name.
        description: What the tool does (used by the model to decide when to call it).
        parameters: JSON-schema describing the tool's arguments.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_provider_dict(self) -> dict[str, Any]:
        """Return the tool spec in OpenAI/Ollama function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to invoke a tool.

    Attributes:
        name: The tool to invoke.
        arguments: Parsed keyword arguments for the tool.
        call_id: Optional provider-supplied identifier.
    """

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """A single turn produced by a provider.

    Exactly one of ``content`` (a final textual answer) or ``tool_calls`` (one or
    more tool invocations) is expected to be meaningful on any given turn.

    Attributes:
        content: Assistant text, if any.
        tool_calls: Tool invocations requested this turn, if any.
    """

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def wants_tools(self) -> bool:
        """Return ``True`` when the model requested at least one tool call."""
        return len(self.tool_calls) > 0


class LLMProvider(abc.ABC):
    """Abstract base class for all LLM providers."""

    name: str = "abstract"

    @abc.abstractmethod
    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        """Produce the next assistant turn.

        Args:
            messages: The full conversation so far.
            tools: The tools available to the model.

        Returns:
            An :class:`LLMResponse` containing either text or tool calls.

        Raises:
            LLMProviderError: If the provider fails to produce a response.
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """Return ``True`` if the provider is ready to serve requests.

        The default implementation always reports healthy; network-backed providers
        override this.
        """
        return True
