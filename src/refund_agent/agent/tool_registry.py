"""Tool registry.

A small indirection layer between the agent loop and the concrete tools. The agent
asks the registry for the tool specs to advertise to the LLM and dispatches tool
calls by name. Keeping this separate means new tools can be added (or swapped for a
remote tool service) without touching the agent loop.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import ToolNotFoundError
from ..llm.base import ToolSpec
from ..logging_config import get_logger
from .tools import Tool

__all__ = ["ToolRegistry"]

_logger = get_logger(__name__)


class ToolRegistry:
    """Holds the set of tools available to the agent and dispatches calls."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        """Initialize the registry.

        Args:
            tools: Optional initial list of tools to register.
        """
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool, replacing any existing tool with the same name.

        Args:
            tool: The tool to register.
        """
        if tool.name in self._tools:
            _logger.warning("Tool '%s' is already registered; overwriting", tool.name)
        self._tools[tool.name] = tool
        _logger.debug("Registered tool '%s'", tool.name)

    def get_specs(self) -> list[ToolSpec]:
        """Return the specs of all registered tools (for the LLM).

        Returns:
            A list of :class:`ToolSpec` objects.
        """
        return [tool.spec() for tool in self._tools.values()]

    def names(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a registered tool by name.

        Args:
            name: The tool name requested by the LLM.
            arguments: The raw arguments for the tool.

        Returns:
            The tool's JSON-serializable result.

        Raises:
            ToolNotFoundError: If no tool with ``name`` is registered.
            InvalidToolArgumentsError: If the arguments fail validation.
            ToolExecutionError: If the tool fails unexpectedly.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                f"Unknown tool '{name}'. Available tools: {sorted(self._tools)}."
            )
        _logger.debug("Executing tool '%s' with args=%s", name, arguments)
        return tool.execute(arguments)
