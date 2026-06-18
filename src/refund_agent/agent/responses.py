"""Deterministic customer-reply composition and financial reconciliation.

Financial facts — whether a refund was approved, how much, whether money actually
moved, and the transaction id — must never depend on the LLM's free-text output. A
model can be persuaded (or can simply hallucinate) into telling a customer "your
refund has been processed" when no refund tool ever ran.

This module derives the **authoritative outcome** of a turn from the tool results
recorded in the conversation and composes the exact customer-facing message from that
ground truth. The agent loop uses it as a guardrail so the agent can never assert a
financial action that did not happen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from ..llm.base import Message

__all__ = [
    "FinancialState",
    "current_turn_messages",
    "extract_tool_results",
    "derive_financial_state",
    "compose_customer_reply",
    "asserts_completion",
    "latest_call_args",
    "UNSUPPORTED_CLAIM_REPLY",
]

# Decision constants (kept as plain strings so this module stays dependency-light).
_APPROVED = "APPROVED"
_PARTIAL = "PARTIALLY_APPROVED"
_DENIED = "DENIED"
_ESCALATED = "ESCALATED"

# Phrases that indicate the model is *claiming a completed financial action*. Used
# only as a last-resort safety net when there is no policy decision to back a claim.
_COMPLETION_MARKERS = (
    "has been approved",
    "has been processed",
    "has been refunded",
    "been refunded",
    "your refund of",
    "refund of $",
    "credited back",
    "credited to",
    "confirmation number",
    "transaction id",
    "refund has been",
    "processed a refund",
    "processed your refund",
)

# Safe reply used when the model claims success with nothing to back it up.
UNSUPPORTED_CLAIM_REPLY = (
    "I'm not able to confirm a refund just yet — I still need to look up your order and "
    "check it against our policy before anything can be processed. Could you share your "
    "order id (for example, ORD-10001) and the reason for the refund?"
)


@dataclass(frozen=True)
class FinancialState:
    """The authoritative financial outcome derived from a turn's tool results.

    Attributes:
        decision: The policy decision (``APPROVED``/``PARTIALLY_APPROVED``/``DENIED``/
            ``ESCALATED``) or ``None`` if no decision was reached this turn.
        order_id: The order the decision relates to, if known.
        approved_amount: Amount approved by policy (string for exact decimals).
        restocking_fee: Restocking fee applied, if any.
        currency: ISO currency code.
        rationale: Policy rationale for the decision.
        executed: Whether a refund was actually executed (a receipt exists).
        transaction_id: The real transaction id when executed, else ``None``.
        refunded_amount: The amount actually refunded when executed, else ``None``.
    """

    decision: Optional[str]
    order_id: Optional[str] = None
    approved_amount: str = "0.00"
    restocking_fee: str = "0.00"
    currency: str = "USD"
    rationale: str = ""
    executed: bool = False
    transaction_id: Optional[str] = None
    refunded_amount: Optional[str] = None

    @property
    def has_decision(self) -> bool:
        """Return ``True`` when a policy decision was reached this turn."""
        return self.decision is not None

    @property
    def is_approval(self) -> bool:
        """Return ``True`` for decisions that authorize money to move."""
        return self.decision in (_APPROVED, _PARTIAL)


def current_turn_messages(history: list[Message]) -> list[Message]:
    """Return only the messages belonging to the current turn.

    The current turn is everything after the most recent ``user`` message. Scoping
    reconciliation to the current turn is essential in a multi-turn session: a refund
    concluded in an earlier turn must never be re-applied to (or re-reported on) a
    later, unrelated request.

    Args:
        history: The full conversation history.

    Returns:
        The slice of messages produced since the last user message (inclusive of all
        assistant/tool messages that followed it). Falls back to the whole history if
        no user message is present.
    """
    for index in range(len(history) - 1, -1, -1):
        if history[index].role == "user":
            return history[index + 1:]
    return list(history)


def extract_tool_results(history: list[Message]) -> dict[str, dict[str, Any]]:
    """Return the latest parsed result for each tool from ``tool`` messages.

    Args:
        history: The conversation history.

    Returns:
        A mapping of tool name to its most recent parsed result dictionary.
    """
    results: dict[str, dict[str, Any]] = {}
    for message in history:
        if message.role == "tool" and message.name:
            try:
                results[message.name] = json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                results[message.name] = {"raw": message.content}
    return results


def derive_financial_state(results: dict[str, dict[str, Any]]) -> FinancialState:
    """Derive the authoritative :class:`FinancialState` from tool results.

    The ``process_refund`` result is preferred as the source of truth (it knows whether
    money actually moved); otherwise the ``check_refund_eligibility`` result is used.
    Error results (which carry no ``decision``) are ignored so a failed execution does
    not mask a valid eligibility decision.

    Args:
        results: Mapping of tool name to its latest result (see
            :func:`extract_tool_results`).

    Returns:
        The derived :class:`FinancialState`. ``decision`` is ``None`` when no policy
        decision has been reached yet.
    """
    proc = results.get("process_refund")
    elig = results.get("check_refund_eligibility")

    source: Optional[dict[str, Any]] = None
    if proc and "decision" in proc:
        source = proc
    elif elig and "decision" in elig:
        source = elig

    if source is None:
        fallback_order = (proc or elig or {}).get("order_id")
        return FinancialState(decision=None, order_id=fallback_order)

    executed = bool(proc and proc.get("executed") and proc.get("transaction_id"))
    refunded = proc.get("refunded_amount") if proc else None

    return FinancialState(
        decision=str(source.get("decision", "")).upper() or None,
        order_id=source.get("order_id"),
        approved_amount=str(source.get("approved_amount", "0.00")),
        restocking_fee=str(source.get("restocking_fee", "0.00")),
        currency=source.get("currency", "USD"),
        rationale=source.get("rationale", ""),
        executed=executed,
        transaction_id=proc.get("transaction_id") if proc else None,
        refunded_amount=str(refunded) if refunded is not None else None,
    )


def compose_customer_reply(state: FinancialState) -> str:
    """Compose the exact customer-facing message for a financial outcome.

    The wording is fully determined by ``state`` (ground truth), never by the model.
    A "processed" confirmation with a transaction id is only ever produced when a
    refund was genuinely executed.

    Args:
        state: The authoritative financial state for the turn.

    Returns:
        The customer-facing reply text.
    """
    decision = state.decision

    if decision == _APPROVED and state.executed and state.transaction_id:
        amount = state.refunded_amount or state.approved_amount
        return (
            f"Good news — your refund of {state.currency} {amount} has been approved "
            f"and processed. Your confirmation number is {state.transaction_id}. It "
            "should appear on your original payment method within 5–7 business days."
        )

    if decision == _PARTIAL and state.executed and state.transaction_id:
        amount = state.refunded_amount or state.approved_amount
        return (
            f"Your refund has been approved with a restocking fee of {state.currency} "
            f"{state.restocking_fee}. A net amount of {state.currency} {amount} has been "
            f"refunded (confirmation {state.transaction_id}). {state.rationale}"
        )

    if decision in (_APPROVED, _PARTIAL):
        # Approved by policy but not actually executed (e.g. an execution error). Never
        # claim a transaction that does not exist; route to a human to finalize.
        return (
            f"Your refund of {state.currency} {state.approved_amount} has been approved. "
            "We hit a brief snag finalizing the payment automatically, so one of our "
            "specialists will complete it and send your confirmation shortly."
        )

    if decision == _ESCALATED:
        return (
            "Your request needs a quick manual review by one of our specialists before a "
            f"refund can be issued. Reason: {state.rationale} A human agent will follow "
            "up shortly."
        )

    if decision == _DENIED:
        return (
            "I'm sorry, but I'm unable to approve this refund under our policy. "
            f"Reason: {state.rationale} If you believe this is a mistake, I can connect "
            "you with a human agent."
        )

    return (
        "I've reviewed your request but couldn't reach a clear decision automatically. "
        "Let me connect you with a human agent to help further."
    )


def asserts_completion(text: str) -> bool:
    """Return ``True`` if ``text`` claims a *completed* refund action.

    Used only when no policy decision exists, as a safety net against a model that
    fabricates a success message without ever calling the refund tools.

    Args:
        text: The model's proposed final reply.

    Returns:
        Whether the text asserts a completed financial action.
    """
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _COMPLETION_MARKERS)


def latest_call_args(history: list[Message], tool_name: str) -> dict[str, Any]:
    """Return the arguments of the most recent call to ``tool_name``.

    Args:
        history: The conversation history.
        tool_name: The tool whose latest call arguments are wanted.

    Returns:
        The arguments dict of the most recent matching tool call, or ``{}`` if none.
    """
    for message in reversed(history):
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                if call.name == tool_name:
                    return dict(call.arguments)
    return {}