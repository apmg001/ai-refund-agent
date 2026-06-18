"""Validated domain entities.

These pydantic models are the canonical in-memory representation of the business
domain. Monetary values use :class:`~decimal.Decimal` to avoid binary
floating-point rounding errors, which is mandatory for anything that touches money.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import (
    AccountStatus,
    CustomerTier,
    DecisionType,
    OrderStatus,
    ProductCategory,
    RefundReason,
    RuleSeverity,
)

__all__ = [
    "Order",
    "Customer",
    "RefundRequest",
    "RuleResult",
    "PolicyDecision",
    "RefundReceipt",
]

# Two-decimal money quantum used across the domain.
_MONEY_QUANTUM = Decimal("0.01")


def _as_money(value: Decimal | float | int | str) -> Decimal:
    """Coerce a value into a 2-decimal-place :class:`Decimal`."""
    return Decimal(str(value)).quantize(_MONEY_QUANTUM)


class _Base(BaseModel):
    """Shared base model with strict, immutable configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Order(_Base):
    """A single purchased order line.

    Attributes:
        order_id: Unique order identifier (e.g. ``ORD-10001``).
        product_name: Human-readable product name.
        category: Product category driving policy rules.
        amount: Total order amount.
        status: Current order lifecycle status.
        order_date: When the order was placed (UTC).
        delivery_date: When the order was delivered (UTC), or ``None`` if not yet
            delivered.
        is_final_sale: Whether the item was sold as final-sale / clearance.
        amount_already_refunded: Amount previously refunded against this order.
    """

    order_id: str
    product_name: str
    category: ProductCategory
    amount: Decimal
    status: OrderStatus
    order_date: datetime
    delivery_date: Optional[datetime] = None
    is_final_sale: bool = False
    amount_already_refunded: Decimal = Decimal("0.00")

    @field_validator("amount", "amount_already_refunded", mode="before")
    @classmethod
    def _coerce_money(cls, value: Decimal | float | int | str) -> Decimal:
        return _as_money(value)

    @property
    def remaining_refundable(self) -> Decimal:
        """Return the amount still eligible to be refunded (never negative)."""
        remaining = self.amount - self.amount_already_refunded
        return remaining if remaining > 0 else Decimal("0.00")


class Customer(_Base):
    """A CRM customer profile.

    Attributes:
        customer_id: Unique customer identifier.
        name: Full name.
        email: Contact email.
        tier: Loyalty tier.
        account_status: Account standing.
        fraud_flag: Whether the account is flagged for potential fraud/abuse.
        lifetime_refunds: Number of refunds historically issued to this customer.
        orders: The customer's orders.
    """

    customer_id: str
    name: str
    email: str
    tier: CustomerTier = CustomerTier.STANDARD
    account_status: AccountStatus = AccountStatus.ACTIVE
    fraud_flag: bool = False
    lifetime_refunds: int = 0
    orders: tuple[Order, ...] = Field(default_factory=tuple)

    def find_order(self, order_id: str) -> Optional[Order]:
        """Return the order with ``order_id`` if it belongs to this customer."""
        return next((o for o in self.orders if o.order_id == order_id), None)


class RefundRequest(_Base):
    """A normalized refund request ready for policy evaluation.

    Attributes:
        order_id: The order the refund is requested against.
        reason: The categorized refund reason.
        requested_amount: Optional explicit amount; defaults to the full remaining
            refundable balance when omitted.
        raw_text: The original free-text customer message, retained for audit.
    """

    order_id: str
    reason: RefundReason
    requested_amount: Optional[Decimal] = None
    raw_text: Optional[str] = None

    @field_validator("requested_amount", mode="before")
    @classmethod
    def _coerce_amount(cls, value: object) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            amount = _as_money(value)  # type: ignore[arg-type]
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(f"requested_amount is not a valid number: {value!r}") from exc
        # A zero or negative requested amount is meaningless for a refund. Treat it as
        # "no specific amount" so the policy engine falls back to the full remaining
        # refundable balance instead of approving a zero or negative refund.
        if amount <= Decimal("0.00"):
            return None
        return amount


class RuleResult(_Base):
    """Outcome of a single policy rule.

    Attributes:
        rule_id: Stable identifier of the rule (e.g. ``return_window``).
        severity: Severity of the outcome.
        message: Human-readable explanation suitable for the customer or auditor.
    """

    rule_id: str
    severity: RuleSeverity
    message: str

    @property
    def passed(self) -> bool:
        """Return ``True`` when the rule did not block or require review."""
        return self.severity in (RuleSeverity.PASS, RuleSeverity.ADJUST)


class PolicyDecision(_Base):
    """The aggregated result of evaluating a refund request.

    Attributes:
        order_id: The evaluated order.
        decision: Final decision type.
        approved_amount: Amount approved for refund (``0`` when denied; the pending
            amount when escalated).
        restocking_fee: Restocking fee deducted, if any.
        currency: ISO currency code.
        rule_results: The full set of per-rule outcomes (audit trail).
        rationale: A concise, customer-facing summary of the decision.
    """

    order_id: str
    decision: DecisionType
    approved_amount: Decimal = Decimal("0.00")
    restocking_fee: Decimal = Decimal("0.00")
    currency: str = "USD"
    rule_results: tuple[RuleResult, ...] = Field(default_factory=tuple)
    rationale: str = ""

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` when no human action is required (approved or denied)."""
        return self.decision in (
            DecisionType.APPROVED,
            DecisionType.PARTIALLY_APPROVED,
            DecisionType.DENIED,
        )


class RefundReceipt(_Base):
    """Confirmation that a refund was actually executed.

    Attributes:
        transaction_id: Idempotent transaction identifier.
        order_id: The refunded order.
        refunded_amount: Amount actually refunded.
        currency: ISO currency code.
        processed_at: When the refund was processed (UTC).
        decision: The decision type that authorized the refund.
    """

    transaction_id: str
    order_id: str
    refunded_amount: Decimal
    currency: str
    processed_at: datetime
    decision: DecisionType