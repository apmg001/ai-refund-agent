"""Heuristic (deterministic) LLM provider.

This provider implements a small, rule-based planner that mimics LLM function
calling **without requiring any model, network, or external dependency**. It lets
the entire agent be exercised from the terminal out of the box, and it doubles as a
fully-deterministic fallback when the Ollama server is unavailable (fault tolerance).

It follows a fixed plan that mirrors how a competent agent would behave:

    1. Extract the order id and refund reason from the customer's message.
    2. Call ``lookup_order`` to fetch the order and customer context.
    3. Call ``check_refund_eligibility`` to run the policy engine.
    4. If eligible, call ``process_refund``; otherwise explain the outcome.
    5. Summarize the result for the customer.

State is reconstructed each turn from the message history (which tool results are
already present), so the planner is stateless and safe to reuse across sessions.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..logging_config import get_logger
from ..models.enums import RefundReason
from ..models.reason_classifier import classify_reason
from .base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec

__all__ = ["HeuristicProvider"]

_logger = get_logger(__name__)

_ORDER_ID_PATTERN = re.compile(r"\bORD[\s_-]*(\d{3,})\b", re.IGNORECASE)


class HeuristicProvider(LLMProvider):
    """Deterministic planner that requires no model or network."""

    name = "heuristic"

    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        """Decide the next action from the conversation state.

        Args:
            messages: The conversation so far.
            tools: Available tools (used only to confirm capability availability).

        Returns:
            An :class:`LLMResponse` requesting a tool call or returning a final answer.
        """
        available = {t.name for t in tools}
        user_text = self._latest_user_text(messages)
        order_id = self._extract_order_id(user_text)
        reason = self._classify_reason(user_text)
        results = self._collect_tool_results(messages)

        _logger.debug("Heuristic plan: order=%s reason=%s done=%s", order_id, reason, list(results))

        if not order_id:
            return LLMResponse(
                content=(
                    "I can help with that. Could you share your order number "
                    "(for example, ORD-10001) so I can look into your refund?"
                )
            )

        # Step 1: look up the order.
        if "lookup_order" not in results and "lookup_order" in available:
            return LLMResponse(tool_calls=(ToolCall("lookup_order", {"order_id": order_id}),))

        if self._has_error(results.get("lookup_order")):
            return LLMResponse(
                content=(
                    f"I couldn't find an order matching '{order_id}'. Please double-check "
                    "the order number and try again."
                )
            )

        # Step 2: evaluate eligibility.
        if "check_refund_eligibility" not in results and "check_refund_eligibility" in available:
            return LLMResponse(
                tool_calls=(
                    ToolCall(
                        "check_refund_eligibility",
                        {"order_id": order_id, "reason": reason.value},
                    ),
                )
            )

        eligibility = results.get("check_refund_eligibility") or {}
        decision = str(eligibility.get("decision", "")).upper()

        # Step 3: act on the decision.
        if decision in {"APPROVED", "PARTIALLY_APPROVED"}:
            if "process_refund" not in results and "process_refund" in available:
                return LLMResponse(
                    tool_calls=(
                        ToolCall(
                            "process_refund",
                            {"order_id": order_id, "reason": reason.value},
                        ),
                    )
                )
            return LLMResponse(content=self._summarize(eligibility, results.get("process_refund")))

        # DENIED / ESCALATED / unknown -> explain without processing.
        return LLMResponse(content=self._summarize(eligibility, None))

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _latest_user_text(messages: list[Message]) -> str:
        """Return the most recent user message text."""
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        """Extract a normalized order id (``ORD-#####``) from free text."""
        match = _ORDER_ID_PATTERN.search(text or "")
        if not match:
            return None
        return f"ORD-{match.group(1)}"

    @staticmethod
    def _classify_reason(text: str) -> RefundReason:
        """Map free-text to a :class:`RefundReason`, defaulting to ``OTHER``.

        Delegates to the shared deterministic classifier so the heuristic provider and
        the agent tools always agree on how a phrase maps to a reason.
        """
        return classify_reason(text) or RefundReason.OTHER

    @staticmethod
    def _collect_tool_results(messages: list[Message]) -> dict[str, dict[str, Any]]:
        """Return the latest parsed result for each tool in the *current turn*.

        Results are scoped to the messages produced after the most recent user message
        so that tool outputs from an earlier, already-concluded request in the same
        session do not leak into the planning for a new request.
        """
        start = 0
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                start = index + 1
                break

        results: dict[str, dict[str, Any]] = {}
        for message in messages[start:]:
            if message.role == "tool" and message.name:
                try:
                    results[message.name] = json.loads(message.content)
                except (json.JSONDecodeError, TypeError):
                    results[message.name] = {"raw": message.content}
        return results

    @staticmethod
    def _has_error(result: dict[str, Any] | None) -> bool:
        """Return ``True`` if a tool result represents an error."""
        return bool(result) and "error" in result

    # ------------------------------------------------------------------ #
    # Response generation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _summarize(eligibility: dict[str, Any], receipt: dict[str, Any] | None) -> str:
        """Compose a customer-facing summary from the decision and optional receipt."""
        decision = str(eligibility.get("decision", "UNKNOWN")).upper()
        rationale = eligibility.get("rationale", "")
        currency = eligibility.get("currency", "USD")
        amount = eligibility.get("approved_amount", "0.00")
        fee = eligibility.get("restocking_fee", "0.00")

        if decision == "APPROVED" and receipt and "transaction_id" in receipt:
            return (
                f"Good news — your refund of {currency} {receipt.get('refunded_amount', amount)} "
                f"has been approved and processed. Your confirmation number is "
                f"{receipt['transaction_id']}. It should appear on your original payment "
                "method within 5–7 business days."
            )
        if decision == "PARTIALLY_APPROVED" and receipt and "transaction_id" in receipt:
            return (
                f"Your refund has been approved with a restocking fee of {currency} {fee}. "
                f"A net amount of {currency} {receipt.get('refunded_amount', amount)} has been "
                f"refunded (confirmation {receipt['transaction_id']}). {rationale}"
            )
        if decision == "ESCALATED":
            return (
                "Your request needs a quick manual review by one of our specialists before a "
                f"refund can be issued. Reason: {rationale} A human agent will follow up shortly."
            )
        if decision == "DENIED":
            return (
                "I'm sorry, but I'm unable to approve this refund under our policy. "
                f"Reason: {rationale} If you believe this is a mistake, I can connect you "
                "with a human agent."
            )
        return (
            "I've reviewed your request but couldn't reach a clear decision automatically. "
            "Let me connect you with a human agent to help further."
        )