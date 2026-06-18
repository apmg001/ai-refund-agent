"""Clock abstraction.

Time-dependent logic (such as the refund eligibility window) must never call
``datetime.now()`` directly, otherwise it becomes impossible to test
deterministically. Instead, components depend on the :class:`Clock` protocol and
receive a concrete implementation via dependency injection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "SystemClock", "FrozenClock"]


@runtime_checkable
class Clock(Protocol):
    """Provides the current time. Implementations must return timezone-aware UTC."""

    def now(self) -> datetime:  # pragma: no cover - protocol definition
        """Return the current time as a timezone-aware UTC :class:`datetime`."""
        ...


class SystemClock:
    """Production clock backed by the operating system wall clock."""

    def now(self) -> datetime:
        """Return the current UTC time.

        Returns:
            A timezone-aware :class:`datetime` in UTC.
        """
        return datetime.now(timezone.utc)


class FrozenClock:
    """Deterministic clock that always returns a fixed instant.

    Intended for unit tests where time-dependent behaviour must be reproducible.
    """

    def __init__(self, fixed: datetime) -> None:
        """Initialize the frozen clock.

        Args:
            fixed: The instant to return from :meth:`now`. If naive, it is assumed
                to be UTC.
        """
        self._fixed = fixed if fixed.tzinfo else fixed.replace(tzinfo=timezone.utc)

    def now(self) -> datetime:
        """Return the fixed instant supplied at construction time."""
        return self._fixed
