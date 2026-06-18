"""HTTP API package (optional bonus).

A thin FastAPI layer exposing the agent over HTTP plus admin endpoints for the
reasoning-log dashboard. Importing this package does not require FastAPI to be
installed; the dependency is only needed when :func:`create_app` is called.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):  # noqa: ANN002, ANN003 - thin lazy proxy
    """Lazily import and build the FastAPI app (keeps FastAPI optional).

    Returns:
        A configured FastAPI application instance.
    """
    from .server import create_app as _create_app

    return _create_app(*args, **kwargs)
