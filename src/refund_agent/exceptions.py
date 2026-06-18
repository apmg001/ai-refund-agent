"""Custom exception hierarchy for the refund agent.

A single, well-defined exception tree lets every layer raise a specific, catchable
error instead of leaking generic ``Exception``/``ValueError`` types. Callers can
catch the broad :class:`RefundAgentError` to handle any domain failure, or a narrow
subclass when they need to react to a specific condition.
"""

from __future__ import annotations

__all__ = [
    "RefundAgentError",
    "ConfigurationError",
    "DataLoadError",
    "CustomerNotFoundError",
    "OrderNotFoundError",
    "PolicyEvaluationError",
    "LLMProviderError",
    "LLMTimeoutError",
    "ToolError",
    "ToolNotFoundError",
    "InvalidToolArgumentsError",
    "ToolExecutionError",
    "AgentError",
    "VoiceError",
]


class RefundAgentError(Exception):
    """Base class for every error raised by this package."""


# --------------------------------------------------------------------------- #
# Configuration & data loading
# --------------------------------------------------------------------------- #
class ConfigurationError(RefundAgentError):
    """Raised when configuration is missing, malformed, or inconsistent."""


class DataLoadError(RefundAgentError):
    """Raised when a backing data file cannot be read or parsed."""


# --------------------------------------------------------------------------- #
# Repository lookups
# --------------------------------------------------------------------------- #
class CustomerNotFoundError(RefundAgentError):
    """Raised when a customer cannot be located in the CRM."""


class OrderNotFoundError(RefundAgentError):
    """Raised when an order cannot be located in the CRM."""


# --------------------------------------------------------------------------- #
# Policy engine
# --------------------------------------------------------------------------- #
class PolicyEvaluationError(RefundAgentError):
    """Raised when a refund request cannot be evaluated against the policy."""


# --------------------------------------------------------------------------- #
# LLM providers
# --------------------------------------------------------------------------- #
class LLMProviderError(RefundAgentError):
    """Raised when the LLM provider fails to produce a response."""


class LLMTimeoutError(LLMProviderError):
    """Raised when the LLM provider does not respond within the timeout."""


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class ToolError(RefundAgentError):
    """Base class for tool-related failures."""


class ToolNotFoundError(ToolError):
    """Raised when the agent requests a tool that is not registered."""


class InvalidToolArgumentsError(ToolError):
    """Raised when tool arguments fail schema validation."""


class ToolExecutionError(ToolError):
    """Raised when a registered tool fails during execution."""


# --------------------------------------------------------------------------- #
# Agent & voice
# --------------------------------------------------------------------------- #
class AgentError(RefundAgentError):
    """Raised for failures in the agent control loop."""


class VoiceError(RefundAgentError):
    """Raised for failures in the (optional) voice pipeline."""
