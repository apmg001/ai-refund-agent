"""Deterministic refund policy engine.

This is the authoritative decision-maker. The LLM is only ever used to *interpret*
the customer's request and *explain* the outcome; it is never trusted to make the
refund decision. Every refund is decided here by evaluating a pipeline of
independent rules, each returning a :class:`RuleResult`. The results are aggregated
using a fixed severity precedence:

    BLOCK  -> DENIED
    REVIEW -> ESCALATED
    ADJUST -> PARTIALLY_APPROVED (restocking fee applied)
    (all PASS) -> APPROVED

Determinism and a full per-rule audit trail make every decision explainable and
reproducible, which is essential for a financial workflow.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Callable

from ..logging_config import get_logger
from ..models.domain import Customer, Order, PolicyDecision, RefundRequest, RuleResult
from ..models.enums import (
    AccountStatus,
    DecisionType,
    OrderStatus,
    RuleSeverity,
)
from ..repositories.policy_repository import PolicyRuleSet
from ..utils.clock import Clock, SystemClock

__all__ = ["PolicyEngine"]

_logger = get_logger(__name__)

_MONEY_QUANTUM = Decimal("0.01")

# Type alias for a rule: (order, customer, request, remaining) -> RuleResult
_Rule = Callable[[Order, Customer, RefundRequest, Decimal], RuleResult]


class PolicyEngine:
    """Evaluates refund requests against the loaded policy rule set."""

    def __init__(self, rules: PolicyRuleSet, clock: Clock | None = None) -> None:
        """Initialize the engine.

        Args:
            rules: The policy thresholds to enforce.
            clock: Clock used to compute the elapsed return window. Defaults to
                :class:`SystemClock`.
        """
        self._rules = rules
        self._clock: Clock = clock or SystemClock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        order: Order,
        customer: Customer,
        request: RefundRequest,
    ) -> PolicyDecision:
        """Evaluate a refund request and return a fully-explained decision.

        Args:
            order: The order being refunded.
            customer: The order's owner.
            request: The normalized refund request.

        Returns:
            A :class:`PolicyDecision` with the outcome, approved amount, any
            restocking fee, and the complete per-rule audit trail.
        """
        remaining = order.remaining_refundable

        rules: list[_Rule] = [
            self._rule_order_status,
            self._rule_account_standing,
            self._rule_fraud_flag,
            self._rule_non_refundable_category,
            self._rule_hygiene_category,
            self._rule_final_sale,
            self._rule_return_window,
            self._rule_high_value,
            self._rule_restocking_fee,
        ]

        results = tuple(rule(order, customer, request, remaining) for rule in rules)
        decision = self._aggregate(order, request, remaining, results)
        _logger.info(
            "Policy evaluated order=%s decision=%s amount=%s",
            order.order_id,
            decision.decision.value,
            decision.approved_amount,
        )
        return decision

    # ------------------------------------------------------------------ #
    # Individual rules
    # ------------------------------------------------------------------ #
    def _rule_order_status(
        self, order: Order, customer: Customer, request: RefundRequest, remaining: Decimal
    ) -> RuleResult:
        """Order must be in a refundable status and not already fully refunded."""
        if order.status == OrderStatus.REFUNDED or remaining <= 0:
            return RuleResult(
                rule_id="order_status",
                severity=RuleSeverity.BLOCK,
                message="This order has already been fully refunded.",
            )
        if order.status not in self._rules.refundable_statuses:
            if order.status in (OrderStatus.PENDING, OrderStatus.PROCESSING, OrderStatus.SHIPPED):
                return RuleResult(
                    rule_id="order_status",
                    severity=RuleSeverity.BLOCK,
                    message=(
                        f"Order is '{order.status.value}' and has not been delivered yet. "
                        "Please request an order cancellation instead of a refund."
                    ),
                )
            return RuleResult(
                rule_id="order_status",
                severity=RuleSeverity.BLOCK,
                message=f"Order status '{order.status.value}' is not eligible for a refund.",
            )
        return RuleResult(
            rule_id="order_status",
            severity=RuleSeverity.PASS,
            message=f"Order status '{order.status.value}' is eligible.",
        )

    def _rule_account_standing(
        self, order: Order, customer: Customer, request: RefundRequest, remaining: Decimal
    ) -> RuleResult:
        """Account must be in good standing."""
        if customer.account_status == AccountStatus.CLOSED:
            return RuleResult(
                rule_id="account_standing",
                severity=RuleSeverity.BLOCK,
                message="The customer account is closed and not eligible for refunds.",
            )
        if customer.account_status == AccountStatus.SUSPENDED:
            return RuleResult(
                rule_id="account_standing",
                severity=RuleSeverity.REVIEW,
                message="The customer account is suspended; manual review is required.",
            )
        return RuleResult(
            rule_id="account_standing",
            severity=RuleSeverity.PASS,
            message="Customer account is active and in good standing.",
        )

    def _rule_fraud_flag(
        self, order: Order, customer: Customer, request: RefundRequest, remaining: Decimal
    ) -> RuleResult:
        """Fraud-flagged accounts are escalated for manual review."""
        if customer.fraud_flag:
            return RuleResult(
                rule_id="fraud_flag",
                severity=RuleSeverity.REVIEW,
                message="The account is flagged for potential fraud/abuse; manual review required.",
            )
        return RuleResult(
            rule_id="fraud_flag",
            severity=RuleSeverity.PASS,
            message="No fraud flag on the account.",
        )

    def _rule_non_refundable_category(
        self, order: Order, customer: Customer, request: RefundRequest, remaining: Decimal
    ) -> RuleResult:
        """Digital goods and gift cards are never refundable."""
        if order.category in self._rules.non_refundable_categories:
            return RuleResult(
                rule_id="non_refundable_category",
                severity=RuleSeverity.BLOCK,
                message=f"{order.category.value} items are non-refundable under all circumstances.",
            )
        return RuleResult(
            rule_id="non_refundable_category",
            severity=RuleSeverity.PASS,
            message=f"{order.category.value} is a refundable category.",
        )

    def _rule_hygiene_category(
        self, order: Order, customer: Customer, request: RefundRequest, remaining: Decimal
    ) -> RuleResult:
        """Hygiene items are non-refundable unless the reason is seller-fault."""
        is_seller_fault = request.reason in self._rules.seller_fault_reasons
        if order.category in self._rules.hygiene_categories and not is_seller_fault:
            return RuleResult(
                rule_id="hygiene_category",
                severity=RuleSeverity.BLOCK,
                message=(
                    f"{order.category.value} items cannot be returned once delivered unless "
                    "the item is defective, damaged, or incorrect."
                ),
            )
        return RuleResult(
            rule_id="hygiene_category",
            severity=RuleSeverity.PASS,
            message="No hygiene restriction applies.",
        )

    def _rule_final_sale(
        self, order: Order, customer: Customer, request: RefundRequest, remaining: Decimal
    ) -> RuleResult:
        """Final-sale items are non-refundable unless the reason is seller-fault."""
        is_seller_fault = request.reason in self._rules.seller_fault_reasons
        if order.is_final_sale and not is_seller_fault:
            return RuleResult(
                rule_id="final_sale",
                severity=RuleSeverity.BLOCK,
                message="Final-sale / clearance items are non-refundable unless they are faulty.",
            )
        return RuleResult(
            rule_id="final_sale",
            severity=RuleSeverity.PASS,
            message="Item is not restricted by a final-sale rule.",
        )

    def _rule_return_window(
        self, order: Order, customer: Customer, request: RefundRequest, remaining: Decimal
    ) -> RuleResult:
        """Request must fall within the applicable return window."""
        if order.delivery_date is None:
            return RuleResult(
                rule_id="return_window",
                severity=RuleSeverity.BLOCK,
                message="Order has no delivery date; the return window cannot start.",
            )

        now = self._clock.now()
        if order.delivery_date > now:
            return RuleResult(
                rule_id="return_window",
                severity=RuleSeverity.REVIEW,
                message=(
                    "The recorded delivery date is in the future, which indicates "
                    "inconsistent order data; routing to manual review rather than "
                    "auto-deciding."
                ),
            )

        is_seller_fault = request.reason in self._rules.seller_fault_reasons
        window_days = (
            self._rules.extended_window_days if is_seller_fault else self._rules.standard_window_days
        )
        days_elapsed = (now - order.delivery_date).days

        if days_elapsed > window_days:
            return RuleResult(
                rule_id="return_window",
                severity=RuleSeverity.BLOCK,
                message=(
                    f"The {window_days}-day return window has passed "
                    f"({days_elapsed} days since delivery)."
                ),
            )
        return RuleResult(
            rule_id="return_window",
            severity=RuleSeverity.PASS,
            message=f"Within the {window_days}-day return window ({days_elapsed} days since delivery).",
        )

    def _rule_high_value(
        self, order: Order, customer: Customer, request: RefundRequest, remaining: Decimal
    ) -> RuleResult:
        """High-value refunds require manual approval."""
        target = self._target_amount(request, remaining)
        if target > self._rules.high_value_threshold:
            return RuleResult(
                rule_id="high_value",
                severity=RuleSeverity.REVIEW,
                message=(
                    f"Refund amount {target} exceeds the manual-review threshold "
                    f"{self._rules.high_value_threshold}."
                ),
            )
        return RuleResult(
            rule_id="high_value",
            severity=RuleSeverity.PASS,
            message="Refund amount is below the manual-review threshold.",
        )

    def _rule_restocking_fee(
        self, order: Order, customer: Customer, request: RefundRequest, remaining: Decimal
    ) -> RuleResult:
        """Opened electronics returned for a non-fault reason incur a restocking fee."""
        is_seller_fault = request.reason in self._rules.seller_fault_reasons
        if order.category in self._rules.restocking_fee_categories and not is_seller_fault:
            pct = (self._rules.restocking_fee_rate * 100).quantize(Decimal("1"))
            return RuleResult(
                rule_id="restocking_fee",
                severity=RuleSeverity.ADJUST,
                message=f"A {pct}% restocking fee applies to this {order.category.value} return.",
            )
        return RuleResult(
            rule_id="restocking_fee",
            severity=RuleSeverity.PASS,
            message="No restocking fee applies.",
        )

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #
    def _aggregate(
        self,
        order: Order,
        request: RefundRequest,
        remaining: Decimal,
        results: tuple[RuleResult, ...],
    ) -> PolicyDecision:
        """Combine per-rule results into a final decision following severity precedence."""
        blocks = [r for r in results if r.severity == RuleSeverity.BLOCK]
        reviews = [r for r in results if r.severity == RuleSeverity.REVIEW]
        adjusts = [r for r in results if r.severity == RuleSeverity.ADJUST]

        target = self._target_amount(request, remaining)

        if blocks:
            return PolicyDecision(
                order_id=order.order_id,
                decision=DecisionType.DENIED,
                approved_amount=Decimal("0.00"),
                currency=self._rules.currency,
                rule_results=results,
                rationale="; ".join(r.message for r in blocks),
            )

        if reviews:
            return PolicyDecision(
                order_id=order.order_id,
                decision=DecisionType.ESCALATED,
                approved_amount=target,  # pending human approval
                currency=self._rules.currency,
                rule_results=results,
                rationale="; ".join(r.message for r in reviews),
            )

        if adjusts:
            fee = (target * self._rules.restocking_fee_rate).quantize(_MONEY_QUANTUM)
            approved = (target - fee).quantize(_MONEY_QUANTUM)
            return PolicyDecision(
                order_id=order.order_id,
                decision=DecisionType.PARTIALLY_APPROVED,
                approved_amount=approved,
                restocking_fee=fee,
                currency=self._rules.currency,
                rule_results=results,
                rationale="; ".join(r.message for r in adjusts),
            )

        return PolicyDecision(
            order_id=order.order_id,
            decision=DecisionType.APPROVED,
            approved_amount=target,
            currency=self._rules.currency,
            rule_results=results,
            rationale="All policy checks passed; full refund approved.",
        )

    @staticmethod
    def _target_amount(request: RefundRequest, remaining: Decimal) -> Decimal:
        """Return the amount under consideration, clamped to ``[0, remaining]``.

        A missing or non-positive requested amount falls back to the full remaining
        balance; any positive request is capped at the remaining balance so a customer
        can never be refunded more than they are owed.
        """
        if request.requested_amount is None or request.requested_amount <= 0:
            return remaining
        return min(request.requested_amount, remaining)