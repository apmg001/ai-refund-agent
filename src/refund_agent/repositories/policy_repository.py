"""Policy repository.

Loads the machine-readable policy rules and the human-readable policy document.
The parsed rule set is exposed as a frozen, typed object so the policy engine never
touches raw dictionaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..exceptions import DataLoadError
from ..logging_config import get_logger
from ..models.enums import OrderStatus, ProductCategory, RefundReason

__all__ = ["PolicyRuleSet", "PolicyRepository"]

_logger = get_logger(__name__)


@dataclass(frozen=True)
class PolicyRuleSet:
    """Strongly-typed, immutable view of the refund policy thresholds.

    Attributes:
        version: Policy version string.
        currency: ISO currency code.
        standard_window_days: Standard return window in days.
        extended_window_days: Extended (seller-fault) return window in days.
        high_value_threshold: Amount above which a refund needs manual review.
        restocking_fee_rate: Fractional restocking fee (e.g. ``0.15``).
        restocking_fee_categories: Categories subject to a restocking fee.
        non_refundable_categories: Categories that are never refundable.
        hygiene_categories: Categories non-refundable unless seller-fault.
        refundable_statuses: Order statuses eligible for a refund.
        seller_fault_reasons: Reasons that unlock the extended window and waive
            restrictions/fees.
    """

    version: str
    currency: str
    standard_window_days: int
    extended_window_days: int
    high_value_threshold: Decimal
    restocking_fee_rate: Decimal
    restocking_fee_categories: frozenset[ProductCategory]
    non_refundable_categories: frozenset[ProductCategory]
    hygiene_categories: frozenset[ProductCategory]
    refundable_statuses: frozenset[OrderStatus]
    seller_fault_reasons: frozenset[RefundReason] = field(default_factory=frozenset)


class PolicyRepository:
    """Loads and provides access to the refund policy."""

    def __init__(self, rules_path: Path, doc_path: Path) -> None:
        """Load both the rule set and the policy document.

        Args:
            rules_path: Path to the machine-readable rules JSON.
            doc_path: Path to the human-readable policy markdown.

        Raises:
            DataLoadError: If either file is missing or malformed.
        """
        self._rules = self._load_rules(rules_path)
        self._document = self._load_document(doc_path)
        _logger.info("Refund policy loaded (version %s)", self._rules.version)

    @staticmethod
    def _load_rules(path: Path) -> PolicyRuleSet:
        """Parse the policy rules JSON into a :class:`PolicyRuleSet`."""
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise DataLoadError(f"Unable to read policy rules '{path}': {exc}") from exc
        except json.JSONDecodeError as exc:
            raise DataLoadError(f"Policy rules '{path}' is not valid JSON: {exc}") from exc

        try:
            windows = payload["windows"]
            thresholds = payload["thresholds"]
            fees = payload["fees"]
            categories = payload["categories"]
            order = payload["order"]
            reasons = payload["reasons"]

            return PolicyRuleSet(
                version=str(payload.get("version", "unknown")),
                currency=str(payload.get("currency", "USD")),
                standard_window_days=int(windows["standard_return_window_days"]),
                extended_window_days=int(windows["extended_return_window_days"]),
                high_value_threshold=Decimal(str(thresholds["high_value_manual_review_amount"])),
                restocking_fee_rate=Decimal(str(fees["restocking_fee_rate"])),
                restocking_fee_categories=frozenset(
                    ProductCategory(c) for c in fees["restocking_fee_categories"]
                ),
                non_refundable_categories=frozenset(
                    ProductCategory(c) for c in categories["non_refundable"]
                ),
                hygiene_categories=frozenset(
                    ProductCategory(c) for c in categories["hygiene_restricted"]
                ),
                refundable_statuses=frozenset(
                    OrderStatus(s) for s in order["refundable_statuses"]
                ),
                seller_fault_reasons=frozenset(
                    RefundReason(r) for r in reasons["seller_fault"]
                ),
            )
        except (KeyError, ValueError) as exc:
            raise DataLoadError(f"Policy rules file is malformed: {exc}") from exc

    @staticmethod
    def _load_document(path: Path) -> str:
        """Read the human-readable policy document."""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DataLoadError(f"Unable to read policy document '{path}': {exc}") from exc

    @property
    def rules(self) -> PolicyRuleSet:
        """Return the typed policy rule set."""
        return self._rules

    @property
    def document(self) -> str:
        """Return the full human-readable policy text."""
        return self._document

    def search_document(self, topic: str, max_chars: int = 600) -> str:
        """Return policy sections relevant to ``topic`` for agent citations.

        A lightweight keyword match over markdown sections (split on headings). This
        avoids pulling in a vector store while still letting the agent ground its
        explanations in the written policy.

        Args:
            topic: Free-text topic to search for.
            max_chars: Maximum number of characters to return.

        Returns:
            The most relevant policy excerpt, or the document preamble if nothing
            matches.
        """
        topic_lower = topic.lower().strip()
        sections = [s.strip() for s in self._document.split("\n## ") if s.strip()]
        if topic_lower:
            for section in sections:
                if topic_lower in section.lower():
                    excerpt = section if section.startswith("#") else f"## {section}"
                    return excerpt[:max_chars]
        return self._document[:max_chars]
