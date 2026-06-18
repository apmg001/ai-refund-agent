"""Refund service.

Orchestrates the end-to-end refund workflow: resolve the order and customer, run the
policy engine, and — only when the policy authorizes it — execute the refund. The
service enforces the decision in code (defense in depth): even if an upstream LLM
"decides" to approve a refund, no money moves unless :class:`PolicyEngine` agrees.

In this mock implementation "executing" a refund updates the in-memory order state
and returns a receipt. Replacing this with a real payment gateway call would be a
localized change.
"""

from __future__ import annotations

import threading
import uuid
from decimal import Decimal

from ..exceptions import PolicyEvaluationError
from ..logging_config import get_logger
from ..models.domain import (
    Order,
    PolicyDecision,
    RefundReceipt,
    RefundRequest,
)
from ..models.enums import DecisionType, OrderStatus
from ..repositories.crm_repository import CrmRepository
from ..utils.clock import Clock, SystemClock
from .policy_engine import PolicyEngine

__all__ = ["RefundService", "RefundOutcome"]

_logger = get_logger(__name__)


class RefundOutcome:
    """Bundles a policy decision with an optional execution receipt.

    Attributes:
        decision: The policy decision that was reached.
        receipt: The refund receipt when money was actually moved, else ``None``.
    """

    __slots__ = ("decision", "receipt")

    def __init__(self, decision: PolicyDecision, receipt: RefundReceipt | None) -> None:
        self.decision = decision
        self.receipt = receipt

    @property
    def executed(self) -> bool:
        """Return ``True`` when a refund was actually processed."""
        return self.receipt is not None


class RefundService:
    """Coordinates policy evaluation and refund execution."""

    def __init__(
        self,
        crm: CrmRepository,
        policy_engine: PolicyEngine,
        clock: Clock | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            crm: The CRM repository for lookups and order updates.
            policy_engine: The engine that decides refunds.
            clock: Clock for timestamping receipts. Defaults to :class:`SystemClock`.
        """
        self._crm = crm
        self._policy = policy_engine
        self._clock: Clock = clock or SystemClock()
        # Serialize refund execution so the read-modify-write of an order's refunded
        # balance is atomic. Over-refund protection comes from ``remaining_refundable``
        # (the cumulative refund can never exceed the order total), which also allows
        # legitimate partial / top-up refunds — so no separate per-order cache is used.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, request: RefundRequest) -> PolicyDecision:
        """Evaluate a refund request without executing it.

        Args:
            request: The normalized refund request.

        Returns:
            The :class:`PolicyDecision`.

        Raises:
            OrderNotFoundError: If the order does not exist (propagated from CRM).
            PolicyEvaluationError: If evaluation fails unexpectedly.
        """
        order = self._crm.get_order(request.order_id)
        customer = self._crm.get_customer_for_order(request.order_id)
        try:
            return self._policy.evaluate(order, customer, request)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise PolicyEvaluationError(
                f"Failed to evaluate refund for {request.order_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def process(self, request: RefundRequest) -> RefundOutcome:
        """Evaluate and, if authorized, execute the refund.

        The refund is executed only for ``APPROVED`` or ``PARTIALLY_APPROVED``
        decisions; ``DENIED`` and ``ESCALATED`` decisions return without moving money.

        Repeat-safety and over-refund protection are provided by the order's remaining
        refundable balance rather than a cache: each execution draws the balance down,
        and once an order is fully refunded the policy engine denies further requests
        (``This order has already been fully refunded``). This also supports legitimate
        partial and top-up refunds (e.g. a later seller-fault claim recovering a
        previously withheld restocking fee).

        Args:
            request: The normalized refund request.

        Returns:
            A :class:`RefundOutcome` containing the decision and any receipt.
        """
        with self._lock:
            decision = self.evaluate(request)
            if decision.decision not in (DecisionType.APPROVED, DecisionType.PARTIALLY_APPROVED):
                _logger.info(
                    "Refund not executed for %s (decision=%s)",
                    request.order_id,
                    decision.decision.value,
                )
                return RefundOutcome(decision=decision, receipt=None)

            receipt = self._execute(decision)
            return RefundOutcome(decision=decision, receipt=receipt)

    def _execute(self, decision: PolicyDecision) -> RefundReceipt:
        """Apply the refund to the in-memory order state and build a receipt."""
        order = self._crm.get_order(decision.order_id)
        new_total_refunded = (order.amount_already_refunded + decision.approved_amount)
        fully_refunded = new_total_refunded >= order.amount

        updated_order = order.model_copy(
            update={
                "amount_already_refunded": new_total_refunded,
                "status": OrderStatus.REFUNDED if fully_refunded else OrderStatus.PARTIALLY_REFUNDED,
            }
        )
        self._crm.replace_order(updated_order)

        receipt = RefundReceipt(
            transaction_id=f"RFND-{uuid.uuid4().hex[:12].upper()}",
            order_id=decision.order_id,
            refunded_amount=decision.approved_amount,
            currency=decision.currency,
            processed_at=self._clock.now(),
            decision=decision.decision,
        )
        _logger.info(
            "Refund executed order=%s txn=%s amount=%s",
            receipt.order_id,
            receipt.transaction_id,
            receipt.refunded_amount,
        )
        return receipt