"""Policy engine tests.

Exercises every decision path of the deterministic :class:`PolicyEngine` against the
15 curated mock orders, plus a few targeted edge cases (amount clamping, restocking
fee math). These tests are the safety net for the single source of truth that decides
whether money moves.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from refund_agent.models.domain import RefundRequest
from refund_agent.models.enums import DecisionType, RefundReason
from refund_agent.repositories.crm_repository import CrmRepository
from refund_agent.services.policy_engine import PolicyEngine

# (order_id, reason, expected decision, expected approved amount or None)
_SCENARIOS = [
    ("ORD-10001", RefundReason.DEFECTIVE, DecisionType.APPROVED, Decimal("129.99")),
    ("ORD-10002", RefundReason.CHANGED_MIND, DecisionType.DENIED, Decimal("0.00")),
    ("ORD-10003", RefundReason.CHANGED_MIND, DecisionType.PARTIALLY_APPROVED, Decimal("424.15")),
    ("ORD-10004", RefundReason.NO_LONGER_NEEDED, DecisionType.DENIED, Decimal("0.00")),
    ("ORD-10005", RefundReason.DEFECTIVE, DecisionType.ESCALATED, None),
    ("ORD-10006", RefundReason.DEFECTIVE, DecisionType.APPROVED, Decimal("75.00")),
    ("ORD-10007", RefundReason.CHANGED_MIND, DecisionType.DENIED, Decimal("0.00")),
    ("ORD-10008", RefundReason.CHANGED_MIND, DecisionType.DENIED, Decimal("0.00")),
    ("ORD-10009", RefundReason.DEFECTIVE, DecisionType.ESCALATED, None),
    ("ORD-10010", RefundReason.CHANGED_MIND, DecisionType.DENIED, Decimal("0.00")),
    ("ORD-10011", RefundReason.DEFECTIVE, DecisionType.DENIED, Decimal("0.00")),
    ("ORD-10012", RefundReason.WRONG_ITEM, DecisionType.APPROVED, Decimal("120.00")),
    ("ORD-10013", RefundReason.CHANGED_MIND, DecisionType.DENIED, Decimal("0.00")),
    ("ORD-10014", RefundReason.DEFECTIVE, DecisionType.ESCALATED, None),
    ("ORD-10015", RefundReason.DEFECTIVE, DecisionType.APPROVED, Decimal("250.00")),
]


@pytest.mark.parametrize("order_id,reason,expected_decision,expected_amount", _SCENARIOS)
def test_decision_paths(
    crm: CrmRepository,
    policy_engine: PolicyEngine,
    order_id: str,
    reason: RefundReason,
    expected_decision: DecisionType,
    expected_amount: Decimal | None,
) -> None:
    """Each curated order produces the expected decision and amount."""
    order = crm.get_order(order_id)
    customer = crm.get_customer_for_order(order_id)
    request = RefundRequest(order_id=order_id, reason=reason)

    decision = policy_engine.evaluate(order, customer, request)

    assert decision.decision is expected_decision, (
        f"{order_id}: expected {expected_decision} got {decision.decision} "
        f"({decision.rationale})"
    )
    if expected_amount is not None:
        assert decision.approved_amount == expected_amount


def test_restocking_fee_math(crm: CrmRepository, policy_engine: PolicyEngine) -> None:
    """A 15% restocking fee is computed and deducted correctly."""
    order = crm.get_order("ORD-10003")  # $499 electronics, changed mind
    customer = crm.get_customer_for_order("ORD-10003")
    request = RefundRequest(order_id="ORD-10003", reason=RefundReason.CHANGED_MIND)

    decision = policy_engine.evaluate(order, customer, request)

    assert decision.decision is DecisionType.PARTIALLY_APPROVED
    assert decision.restocking_fee == Decimal("74.85")
    assert decision.approved_amount == Decimal("424.15")
    assert decision.restocking_fee + decision.approved_amount == order.amount


def test_every_decision_has_audit_trail(crm: CrmRepository, policy_engine: PolicyEngine) -> None:
    """Every decision carries the full per-rule audit trail."""
    order = crm.get_order("ORD-10001")
    customer = crm.get_customer_for_order("ORD-10001")
    request = RefundRequest(order_id="ORD-10001", reason=RefundReason.DEFECTIVE)

    decision = policy_engine.evaluate(order, customer, request)

    assert len(decision.rule_results) == 9
    assert all(r.message for r in decision.rule_results)


def test_requested_amount_is_clamped(crm: CrmRepository, policy_engine: PolicyEngine) -> None:
    """A requested amount above the remaining balance is clamped down."""
    order = crm.get_order("ORD-10001")  # $129.99 remaining
    customer = crm.get_customer_for_order("ORD-10001")
    request = RefundRequest(
        order_id="ORD-10001",
        reason=RefundReason.DEFECTIVE,
        requested_amount=Decimal("9999.00"),
    )

    decision = policy_engine.evaluate(order, customer, request)

    assert decision.approved_amount == Decimal("129.99")
