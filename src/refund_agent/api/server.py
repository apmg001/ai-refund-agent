"""FastAPI server (optional bonus).

Exposes the agent over HTTP so a web chat UI and an admin dashboard can be built on
top of it:

    POST /chat                          -> send a message, get the agent's reply
    GET  /admin/customers               -> list mock customers
    GET  /admin/sessions                -> list active reasoning sessions
    GET  /admin/sessions/{id}/logs      -> full reasoning trace for a session
    GET  /healthz                       -> liveness/readiness

The single :class:`Application` object graph is built once at startup and shared
across requests (the services are thread-safe). This module is import-safe without
FastAPI installed; the dependency is resolved only when :func:`create_app` runs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ..app import Application, build_application
from ..exceptions import CustomerNotFoundError, OrderNotFoundError, RefundAgentError
from ..logging_config import get_logger
from .schemas import (
    ChatRequest,
    ChatResponse,
    CustomerSummary,
    HealthResponse,
    ReasoningEventModel,
    SessionLogResponse,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

__all__ = ["create_app"]

_logger = get_logger(__name__)


def create_app(application: Application | None = None) -> "FastAPI":
    """Build the FastAPI application.

    Args:
        application: An optional pre-wired :class:`Application` (used by tests). If
            omitted, one is constructed from environment settings.

    Returns:
        A configured FastAPI instance.

    Raises:
        ConfigurationError: If FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:  # pragma: no cover - depends on environment
        from ..exceptions import ConfigurationError

        raise ConfigurationError(
            "FastAPI is required to run the HTTP API. Install it with "
            "`pip install fastapi uvicorn`."
        ) from exc

    app_graph = application or build_application()
    api = FastAPI(title=app_graph.settings.app_name, version="1.0.0")

    # The frontend is a decoupled static app served from a different origin/port, so
    # cross-origin requests must be allowed. Permissive by default for local demos;
    # restrict `allow_origins` to known hosts in production.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Customer-facing
    # ------------------------------------------------------------------ #
    @api.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """Send a message to the agent and return its reply."""
        try:
            result = app_graph.agent.run(request.message, session_id=request.session_id)
        except RefundAgentError as exc:  # pragma: no cover - defensive
            _logger.exception("Chat handling failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return ChatResponse(
            session_id=result.session_id,
            reply=result.reply,
            iterations=result.iterations,
            tool_calls=list(result.tool_calls),
        )

    # ------------------------------------------------------------------ #
    # Admin dashboard
    # ------------------------------------------------------------------ #
    @api.get("/admin/customers", response_model=list[CustomerSummary])
    def list_customers() -> list[CustomerSummary]:
        """Return all mock customers."""
        return [
            CustomerSummary(
                customer_id=c.customer_id,
                name=c.name,
                email=c.email,
                tier=c.tier.value,
                account_status=c.account_status.value,
                order_ids=[o.order_id for o in c.orders],
            )
            for c in app_graph.crm.list_customers()
        ]

    @api.get("/admin/sessions", response_model=list[str])
    def list_sessions() -> list[str]:
        """Return the ids of all sessions that have reasoning events."""
        return app_graph.reasoning_log.list_sessions()

    @api.get("/admin/sessions/{session_id}/logs", response_model=SessionLogResponse)
    def session_logs(session_id: str) -> SessionLogResponse:
        """Return the full reasoning trace for a session."""
        events = app_graph.reasoning_log.get_events(session_id)
        if not events:
            raise HTTPException(status_code=404, detail=f"No events for session '{session_id}'.")
        return SessionLogResponse(
            session_id=session_id,
            events=[ReasoningEventModel(**e.to_dict()) for e in events],
        )

    @api.get("/admin/sessions/{session_id}/stream")
    def stream_session(session_id: str):
        """Stream reasoning events for a session in real time via Server-Sent Events.

        Subscribes an observer to the shared reasoning log and pushes each matching
        event to the client as it is recorded. Any events already recorded for the
        session are replayed first so a late-connecting client misses nothing. The
        observer is detached when the client disconnects.

        This is the real-time feed behind the admin dashboard. For a horizontally
        scaled deployment the in-process observer/queue would be replaced by a shared
        broker (e.g. Redis pub/sub) without changing the wire protocol.
        """
        import queue as _queue

        from fastapi.responses import StreamingResponse

        events_q: "_queue.Queue[dict]" = _queue.Queue()

        def _observer(event) -> None:  # noqa: ANN001 - ReasoningEvent
            if event.session_id == session_id:
                events_q.put(event.to_dict())

        app_graph.reasoning_log.subscribe(_observer)

        def _event_stream():
            try:
                for existing in app_graph.reasoning_log.get_events(session_id):
                    yield f"data: {json.dumps(existing.to_dict())}\n\n"
                while True:
                    try:
                        payload = events_q.get(timeout=15)
                        yield f"data: {json.dumps(payload)}\n\n"
                    except _queue.Empty:
                        yield ": keep-alive\n\n"
            finally:
                app_graph.reasoning_log.unsubscribe(_observer)

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------ #
    # Operational
    # ------------------------------------------------------------------ #
    @api.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        """Liveness/readiness probe."""
        return HealthResponse(
            status="ok",
            app_name=app_graph.settings.app_name,
            llm_provider=getattr(app_graph.llm, "name", "unknown"),
        )

    @api.exception_handler(OrderNotFoundError)
    @api.exception_handler(CustomerNotFoundError)
    def _not_found_handler(_request, exc):  # noqa: ANN001 - FastAPI handler
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": str(exc)})

    _logger.info("FastAPI app created")
    return api