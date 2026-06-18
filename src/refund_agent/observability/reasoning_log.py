"""Agent reasoning log.

Captures a structured, timestamped trace of everything the agent does — user
messages, LLM turns, tool calls, tool results, decisions, and final answers — so the
admin dashboard (or terminal) can display *real-time agent reasoning*.

The store is an in-memory implementation behind a small interface. For a scaled,
fault-tolerant deployment it could be swapped for Redis/Kafka/a database without
touching the agent, and observers (callbacks) allow live streaming to a UI.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from ..utils.serialization import dumps

__all__ = ["EventType", "ReasoningEvent", "ReasoningLog"]


class EventType(str, Enum):
    """Category of a reasoning event."""

    USER_MESSAGE = "USER_MESSAGE"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_ERROR = "TOOL_ERROR"
    DECISION = "DECISION"
    GUARDRAIL = "GUARDRAIL"
    AGENT_RESPONSE = "AGENT_RESPONSE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ReasoningEvent:
    """A single entry in the agent's reasoning trace.

    Attributes:
        session_id: The conversation/session this event belongs to.
        sequence: Monotonic per-session sequence number.
        event_type: The category of event.
        title: Short human-readable label.
        detail: Structured payload (JSON-serializable).
        timestamp: UTC time the event was recorded.
    """

    session_id: str
    sequence: int
    event_type: EventType
    title: str
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the event."""
        return {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "title": self.title,
            "detail": self.detail,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Return the event serialized as a JSON string."""
        return dumps(self.to_dict())


# Observer signature: called synchronously with each newly-recorded event.
Observer = Callable[[ReasoningEvent], None]


class ReasoningLog:
    """Thread-safe, per-session store of :class:`ReasoningEvent` objects."""

    def __init__(self) -> None:
        """Initialize an empty reasoning log."""
        self._events: dict[str, list[ReasoningEvent]] = {}
        self._counters: dict[str, int] = {}
        self._observers: list[Observer] = []
        self._lock = threading.Lock()

    def subscribe(self, observer: Observer) -> None:
        """Register a callback invoked for every newly-recorded event.

        Args:
            observer: A callable accepting a :class:`ReasoningEvent`. Exceptions
                raised by observers are swallowed so logging never breaks the agent.
        """
        with self._lock:
            self._observers.append(observer)

    def unsubscribe(self, observer: Observer) -> None:
        """Remove a previously-registered observer.

        Safe to call with an observer that is not registered (no-op). Used by streaming
        consumers (e.g. the SSE endpoint) to detach when a client disconnects so the
        observer list does not grow unbounded.

        Args:
            observer: The callback to remove.
        """
        with self._lock:
            try:
                self._observers.remove(observer)
            except ValueError:
                pass

    def record(
        self,
        session_id: str,
        event_type: EventType,
        title: str,
        detail: dict[str, Any] | None = None,
    ) -> ReasoningEvent:
        """Record a new event for ``session_id``.

        Args:
            session_id: The session the event belongs to.
            event_type: The category of event.
            title: Short human-readable label.
            detail: Optional structured payload.

        Returns:
            The recorded :class:`ReasoningEvent`.
        """
        with self._lock:
            seq = self._counters.get(session_id, 0) + 1
            self._counters[session_id] = seq
            event = ReasoningEvent(
                session_id=session_id,
                sequence=seq,
                event_type=event_type,
                title=title,
                detail=detail or {},
            )
            self._events.setdefault(session_id, []).append(event)
            observers = list(self._observers)

        # Notify observers outside the lock; never let an observer crash the agent.
        for observer in observers:
            try:
                observer(event)
            except Exception:  # noqa: BLE001 - observers must not break logging
                pass
        return event

    def get_events(self, session_id: str) -> list[ReasoningEvent]:
        """Return all events for ``session_id`` (empty list if none)."""
        with self._lock:
            return list(self._events.get(session_id, []))

    def list_sessions(self) -> list[str]:
        """Return the ids of all sessions that have recorded events."""
        with self._lock:
            return list(self._events.keys())

    def clear(self, session_id: str) -> None:
        """Remove all events for ``session_id``."""
        with self._lock:
            self._events.pop(session_id, None)
            self._counters.pop(session_id, None)