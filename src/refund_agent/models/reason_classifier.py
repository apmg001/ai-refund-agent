"""Deterministic refund-reason classification.

The refund *reason* drives money: seller-fault reasons (defective, damaged, wrong
item, …) waive the restocking fee and unlock the extended return window, while
non-fault reasons do not. That makes reason determination an accuracy-critical step —
and one that must never be left to the unguarded whim of an LLM.

A model asked for a reason can emit an invalid label (e.g. ``NOT_WORKING``) that, if
silently coerced to :attr:`RefundReason.OTHER`, would wrongly charge a customer a
restocking fee on a defective product. This module is the single, deterministic source
of truth for turning any text — a customer's free-text phrase *or* a model's proposed
label — into a canonical :class:`RefundReason`. It is used by both the heuristic
provider and the agent tools so the financial outcome never depends on how a label
happened to be spelled.
"""

from __future__ import annotations

import re

from .enums import RefundReason

__all__ = ["classify_reason", "REASON_KEYWORDS"]

# Ordered most-specific first; the first matching group wins. Phrases are matched
# against text that has been lower-cased and had separators collapsed to spaces, so
# both "not working" and a label like "NOT_WORKING" resolve identically.
REASON_KEYWORDS: list[tuple[tuple[str, ...], RefundReason]] = [
    (
        (
            "defective", "not working", "doesnt work", "does not work", "won t work",
            "wont work", "stopped working", "no longer works", "not work", "faulty",
            "malfunction", "won t turn on", "wont turn on", "not turning on", "dead",
            "won t power", "wont power", "broken", "bricked",
        ),
        RefundReason.DEFECTIVE,
    ),
    (
        ("damaged", "cracked", "shattered", "dented", "scratched", "torn", "smashed"),
        RefundReason.DAMAGED,
    ),
    (
        (
            "wrong item", "wrong product", "incorrect item", "received the wrong",
            "different item", "not what i ordered", "wrong order", "mixed up",
        ),
        RefundReason.WRONG_ITEM,
    ),
    (
        (
            "not as described", "different from", "misleading", "doesnt match",
            "does not match", "not what i expected", "misrepresented",
        ),
        RefundReason.NOT_AS_DESCRIBED,
    ),
    (
        ("arrived late", "too late", "delivered late", "late delivery", "took too long"),
        RefundReason.ARRIVED_LATE,
    ),
    (
        ("no longer need", "dont need", "do not need", "not needed", "don t need"),
        RefundReason.NO_LONGER_NEEDED,
    ),
    (
        (
            "changed my mind", "change my mind", "changed mind", "dont want",
            "do not want", "don t want", "regret", "no longer want",
        ),
        RefundReason.CHANGED_MIND,
    ),
]

# Matches the separator characters used inside enum-style labels.
_SEPARATORS = re.compile(r"[_\-]+")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lower-case ``text`` and collapse separators/apostrophes to single spaces."""
    lowered = text.lower().replace("'", " ").replace("’", " ")
    lowered = _SEPARATORS.sub(" ", lowered)
    return _WHITESPACE.sub(" ", lowered).strip()


def classify_reason(raw: str | None) -> RefundReason | None:
    """Resolve arbitrary text to a canonical :class:`RefundReason`.

    Resolution order:
        1. Exact (case-insensitive) match against a valid ``RefundReason`` value.
        2. Keyword/phrase match against the normalized text.

    Args:
        raw: A model-proposed label or a customer's free-text phrase.

    Returns:
        The matched :class:`RefundReason`, or ``None`` if the text cannot be mapped to
        any known reason. Callers decide how to treat ``None`` (typically: ask the
        customer to clarify, or fall back to :attr:`RefundReason.OTHER`).
    """
    if not raw or not raw.strip():
        return None

    # 1. Exact enum value (e.g. the model already sent a valid "DEFECTIVE").
    try:
        return RefundReason(raw.strip().upper())
    except ValueError:
        pass

    # 2. Keyword/phrase match on normalized text (handles "NOT_WORKING", "it is not
    #    working", etc.).
    normalized = _normalize(raw)
    for keywords, reason in REASON_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return reason
    return None