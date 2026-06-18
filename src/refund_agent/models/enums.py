"""Domain enumerations.

Using string-valued enums (rather than bare strings) gives us validation at the
boundaries, IDE auto-completion, and a single authoritative list of every legal
value. All enums inherit from ``str`` so they serialize cleanly to JSON.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "OrderStatus",
    "ProductCategory",
    "RefundReason",
    "DecisionType",
    "CustomerTier",
    "AccountStatus",
    "RuleSeverity",
]


class OrderStatus(str, Enum):
    """Lifecycle state of an order."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"
    REFUNDED = "REFUNDED"
    PARTIALLY_REFUNDED = "PARTIALLY_REFUNDED"


class ProductCategory(str, Enum):
    """High-level product category used by the policy engine."""

    ELECTRONICS = "ELECTRONICS"
    APPAREL = "APPAREL"
    HOME = "HOME"
    BEAUTY = "BEAUTY"
    GROCERY = "GROCERY"
    DIGITAL = "DIGITAL"
    GIFT_CARD = "GIFT_CARD"
    OTHER = "OTHER"


class RefundReason(str, Enum):
    """Customer-stated reason for the refund request."""

    DEFECTIVE = "DEFECTIVE"
    DAMAGED = "DAMAGED"
    WRONG_ITEM = "WRONG_ITEM"
    NOT_AS_DESCRIBED = "NOT_AS_DESCRIBED"
    ARRIVED_LATE = "ARRIVED_LATE"
    NO_LONGER_NEEDED = "NO_LONGER_NEEDED"
    CHANGED_MIND = "CHANGED_MIND"
    OTHER = "OTHER"


class DecisionType(str, Enum):
    """Final outcome of a policy evaluation."""

    APPROVED = "APPROVED"
    PARTIALLY_APPROVED = "PARTIALLY_APPROVED"
    DENIED = "DENIED"
    ESCALATED = "ESCALATED"


class CustomerTier(str, Enum):
    """Loyalty tier of the customer."""

    STANDARD = "STANDARD"
    PREMIUM = "PREMIUM"
    VIP = "VIP"


class AccountStatus(str, Enum):
    """Standing of the customer account."""

    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    CLOSED = "CLOSED"


class RuleSeverity(str, Enum):
    """Severity assigned to an individual policy rule outcome.

    The aggregation precedence is ``BLOCK`` > ``REVIEW`` > ``ADJUST`` > ``PASS``.
    """

    PASS = "PASS"        # rule satisfied, no action
    ADJUST = "ADJUST"    # eligible, but the refund amount must be adjusted (fee)
    REVIEW = "REVIEW"    # cannot auto-decide; route to a human
    BLOCK = "BLOCK"      # hard denial
