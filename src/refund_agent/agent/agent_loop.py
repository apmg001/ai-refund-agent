"""Agent loop.

Implements a raw function-calling agent (no framework) over the provider-agnostic
:class:`LLMProvider` interface. The loop drives the standard tool-use cycle:

    user message
        -> LLM turn
            -> if tool calls: execute them, feed results back, repeat
            -> else: return the final assistant message

Every step is recorded in the :class:`ReasoningLog` so the admin dashboard / terminal
can stream the agent's reasoning in real time. Per-session conversation history is held
in a small, swappable :class:`SessionStore`; for a horizontally-scaled deployment this
could be backed by Redis without changing the loop.

The loop is intentionally defensive: tool failures are captured and fed back to the
model as structured errors (so it can recover or escalate) rather than crashing the
turn, and a hard iteration cap prevents runaway tool-calling.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from ..exceptions import (
    LLMProviderError,
    ToolError,
)
from ..llm.base import LLMProvider, Message, ToolCall
from ..logging_config import get_logger
from ..observability.reasoning_log import EventType, ReasoningLog
from ..utils.serialization import dumps
from .prompts import SYSTEM_PROMPT
from .responses import (
    asserts_completion,
    compose_customer_reply,
    current_turn_messages,
    derive_financial_state,
    extract_tool_results,
    latest_call_args,
    UNSUPPORTED_CLAIM_REPLY,
)
from .tool_registry import ToolRegistry

__all__ = ["RefundAgent", "AgentResult", "SessionStore", "InMemorySessionStore"]

_logger = get_logger(__name__)


# ====================================================================== #
# Result type
# ====================================================================== #
@dataclass(frozen=True)
class AgentResult:
    """The outcome of a single :meth:`RefundAgent.run` call.

    Attributes:
        session_id: The session the turn belongs to.
        reply: The final assistant message shown to the customer.
        iterations: How many LLM turns were consumed.
        tool_calls: The names of tools invoked during the turn, in order.
    """

    session_id: str
    reply: str
    iterations: int
    tool_calls: tuple[str, ...]


# ====================================================================== #
# Session store
# ====================================================================== #
class SessionStore:
    """Abstract per-session conversation store.

    The default implementation is in-memory; subclassing allows a distributed backend
    (e.g. Redis) without modifying the agent.
    """

    def history(self, session_id: str) -> list[Message]:  # pragma: no cover - interface
        """Return the message history for ``session_id`` (creating it if needed)."""
        raise NotImplementedError

    def reset(self, session_id: str) -> None:  # pragma: no cover - interface
        """Clear the history for ``session_id``."""
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    """Thread-safe in-memory session store seeded with the system prompt."""

    def __init__(self, system_prompt: str) -> None:
        """Initialize the store.

        Args:
            system_prompt: The system prompt that seeds every new session.
        """
        self._system_prompt = system_prompt
        self._sessions: dict[str, list[Message]] = {}
        self._lock = threading.Lock()

    def history(self, session_id: str) -> list[Message]:
        """Return (and lazily create) the history for ``session_id``."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = [Message(role="system", content=self._system_prompt)]
            return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        """Clear the history for ``session_id``."""
        with self._lock:
            self._sessions.pop(session_id, None)


# ====================================================================== #
# Agent
# ====================================================================== #
class RefundAgent:
    """A function-calling agent that resolves customer refund requests."""

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        reasoning_log: ReasoningLog,
        *,
        max_iterations: int = 8,
        system_prompt: str = SYSTEM_PROMPT,
        session_store: SessionStore | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            llm: The LLM provider that produces turns and tool calls.
            tools: The registry of tools the agent may call.
            reasoning_log: The observability sink for the reasoning trace.
            max_iterations: Hard cap on LLM turns per user message (loop guard).
            system_prompt: The system prompt seeding each session.
            session_store: Conversation store. Defaults to an in-memory store.
        """
        self._llm = llm
        self._tools = tools
        self._log = reasoning_log
        self._max_iterations = max_iterations
        self._sessions = session_store or InMemorySessionStore(system_prompt)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def new_session(self) -> str:
        """Create and return a fresh session id."""
        return uuid.uuid4().hex[:12]

    def run(self, user_message: str, session_id: str | None = None) -> AgentResult:
        """Process a single user message and return the agent's reply.

        Args:
            user_message: The customer's free-text message.
            session_id: Existing session to continue. A new one is created if omitted.

        Returns:
            An :class:`AgentResult` with the reply and turn metadata.
        """
        session_id = session_id or self.new_session()
        history = self._sessions.history(session_id)
        history.append(Message(role="user", content=user_message))
        self._log.record(
            session_id,
            EventType.USER_MESSAGE,
            "Customer message received",
            {"message": user_message},
        )

        specs = self._tools.get_specs()
        invoked: list[str] = []

        for iteration in range(1, self._max_iterations + 1):
            self._log.record(
                session_id,
                EventType.LLM_REQUEST,
                f"Requesting model turn (iteration {iteration})",
                {"provider": getattr(self._llm, "name", "unknown"), "iteration": iteration},
            )

            try:
                response = self._llm.generate(history, specs)
            except LLMProviderError as exc:
                _logger.error("LLM provider error: %s", exc)
                return self._fail(session_id, history, iteration, invoked, str(exc))

            # --- Final answer ------------------------------------------------ #
            if not response.wants_tools:
                return self._finalize(
                    session_id, history, response.content, iteration, invoked
                )

            # --- Tool calls -------------------------------------------------- #
            self._log.record(
                session_id,
                EventType.LLM_RESPONSE,
                "Model requested tool calls",
                {"tools": [tc.name for tc in response.tool_calls]},
            )
            history.append(Message(role="assistant", content="", tool_calls=list(response.tool_calls)))

            for call in response.tool_calls:
                invoked.append(call.name)
                self._execute_tool_call(session_id, history, call)

        # Iteration budget exhausted -> reconcile any reached decision, then escalate.
        self._log.record(
            session_id,
            EventType.ERROR,
            "Reached maximum reasoning steps",
            {"max_iterations": self._max_iterations},
        )
        return self._finalize(session_id, history, "", self._max_iterations, invoked)

    def reset(self, session_id: str) -> None:
        """Clear conversation history and reasoning log for ``session_id``."""
        self._sessions.reset(session_id)
        self._log.clear(session_id)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _finalize(
        self,
        session_id: str,
        history: list[Message],
        model_text: str,
        iterations: int,
        invoked: list[str],
    ) -> AgentResult:
        """Reconcile the model's proposed reply against the real financial outcome.

        This is the safety boundary that prevents the agent from telling a customer a
        refund happened when it did not. It:

        1. Derives the authoritative outcome from the recorded tool results.
        2. If policy approved a refund the model never executed, executes it now
           (idempotent, re-checks policy) so the claim becomes true.
        3. For any reached decision, authors the customer-facing message from ground
           truth rather than trusting the model's text.
        4. If the model claims a completed refund with no decision to back it, replaces
           the claim with a safe message.

        Args:
            session_id: The active session.
            history: The conversation history (tool results are read from here).
            model_text: The model's proposed final reply (may be empty).
            iterations: Iteration count to report on the result.
            invoked: Names of tools invoked so far (mutated if a refund is enforced).

        Returns:
            The reconciled :class:`AgentResult`.
        """
        # Reconcile only against THIS turn's tool activity. A refund concluded in an
        # earlier turn is already done and must not be re-applied or re-reported here.
        turn = current_turn_messages(history)
        results = extract_tool_results(turn)
        state = derive_financial_state(results)

        # (2) Enforce execution of an approved-but-skipped refund.
        if state.is_approval and not state.executed:
            self._log.record(
                session_id,
                EventType.GUARDRAIL,
                "Refund approved but not executed by the model; enforcing execution",
                {"order_id": state.order_id, "decision": state.decision},
            )
            args = latest_call_args(turn, "check_refund_eligibility")
            if not args.get("order_id") and state.order_id:
                args["order_id"] = state.order_id
            if args.get("order_id"):
                self._execute_tool_call(session_id, history, ToolCall("process_refund", args))
                if "process_refund" not in invoked:
                    invoked.append("process_refund")
                turn = current_turn_messages(history)
                results = extract_tool_results(turn)
                state = derive_financial_state(results)

        # (3) Author the financial message from ground truth.
        if state.has_decision:
            reply = compose_customer_reply(state)
            if reply != (model_text or "").strip():
                self._log.record(
                    session_id,
                    EventType.GUARDRAIL,
                    "Authored customer reply from the verified financial outcome",
                    {"decision": state.decision, "executed": state.executed},
                )
        else:
            # (4) No decision reached: trust the model unless it fabricates success.
            text = (model_text or "").strip()
            if text and asserts_completion(text):
                self._log.record(
                    session_id,
                    EventType.GUARDRAIL,
                    "Suppressed an unsupported refund claim (no policy decision)",
                    {"model_text": text},
                )
                reply = UNSUPPORTED_CLAIM_REPLY
            else:
                reply = text or _FALLBACK_REPLY

        history.append(Message(role="assistant", content=reply))
        self._log.record(
            session_id,
            EventType.AGENT_RESPONSE,
            "Agent produced final reply",
            {"reply": reply},
        )
        return AgentResult(
            session_id=session_id,
            reply=reply,
            iterations=iterations,
            tool_calls=tuple(invoked),
        )

    def _execute_tool_call(
        self,
        session_id: str,
        history: list[Message],
        call: ToolCall,
    ) -> None:
        """Execute a single tool call and append its result to the history.

        Tool errors are captured and appended as a structured ``{"error": ...}`` tool
        message so the model (or the heuristic planner) can react instead of crashing.
        """
        self._log.record(
            session_id,
            EventType.TOOL_CALL,
            f"Calling tool '{call.name}'",
            {"tool": call.name, "arguments": call.arguments},
        )
        try:
            result = self._tools.execute(call.name, call.arguments)
        except ToolError as exc:
            _logger.warning("Tool '%s' error: %s", call.name, exc)
            result = {"error": str(exc)}
            self._log.record(
                session_id,
                EventType.TOOL_ERROR,
                f"Tool '{call.name}' failed",
                {"tool": call.name, "error": str(exc)},
            )
        else:
            self._log.record(
                session_id,
                EventType.TOOL_RESULT,
                f"Tool '{call.name}' returned",
                {"tool": call.name, "result": result},
            )
            if "decision" in result:
                self._log.record(
                    session_id,
                    EventType.DECISION,
                    f"Policy decision: {result.get('decision')}",
                    {"tool": call.name, "decision": result.get("decision")},
                )

        history.append(Message(role="tool", name=call.name, content=dumps(result)))

    def _fail(
        self,
        session_id: str,
        history: list[Message],
        iterations: int,
        invoked: list[str],
        reason: str,
    ) -> AgentResult:
        """Record an error and return a safe, customer-facing fallback reply."""
        self._log.record(
            session_id,
            EventType.ERROR,
            "Agent could not complete the request",
            {"reason": reason},
        )
        history.append(Message(role="assistant", content=_FALLBACK_REPLY))
        return AgentResult(
            session_id=session_id,
            reply=_FALLBACK_REPLY,
            iterations=iterations,
            tool_calls=tuple(invoked),
        )


_FALLBACK_REPLY = (
    "I'm sorry, but I'm having trouble completing your request right now. "
    "I'm connecting you with a human support agent who can help."
)