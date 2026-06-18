"""API schemas.

Pydantic request/response models for the HTTP layer. Keeping the wire contract in
its own module separates transport concerns from the domain models, so the public
API can evolve independently of internal representations.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ReasoningEventModel",
    "SessionLogResponse",
    "CustomerSummary",
    "HealthResponse",
]


class ChatRequest(BaseModel):
    """Inbound chat message."""

    message: str = Field(..., min_length=1, description="The customer's message.")
    session_id: Optional[str] = Field(None, description="Existing session to continue.")


class ChatResponse(BaseModel):
    """Agent reply plus turn metadata."""

    session_id: str
    reply: str
    iterations: int
    tool_calls: list[str]


class ReasoningEventModel(BaseModel):
    """A single reasoning-trace event for the admin dashboard."""

    session_id: str
    sequence: int
    event_type: str
    title: str
    detail: dict[str, Any]
    timestamp: str


class SessionLogResponse(BaseModel):
    """The full reasoning trace for a session."""

    session_id: str
    events: list[ReasoningEventModel]


class CustomerSummary(BaseModel):
    """Condensed customer view for the admin customer list."""

    customer_id: str
    name: str
    email: str
    tier: str
    account_status: str
    order_ids: list[str]


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str
    app_name: str
    llm_provider: str
