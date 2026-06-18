"""Centralized logging configuration.

The application uses the stdlib :mod:`logging` module everywhere (never ``print``).
This module configures the root logger exactly once. When the optional ``rich``
package is installed the console output is colorized; otherwise it gracefully
degrades to a plain stream handler.
"""

from __future__ import annotations

import logging
import logging.config
from typing import Any

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging idempotently.

    Args:
        level: Logging threshold name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return

    handler: dict[str, Any]
    try:
        import rich.logging  # noqa: F401  (import test only)

        handler = {
            "class": "rich.logging.RichHandler",
            "rich_tracebacks": True,
            "show_path": False,
            "markup": False,
        }
        formatter_key = "rich"
        formatters = {"rich": {"format": "%(message)s", "datefmt": "[%X]"}}
    except ImportError:  # pragma: no cover - depends on environment
        handler = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        }
        formatter_key = "plain"
        formatters = {
            "plain": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        }

    handler["formatter"] = formatter_key
    handler["level"] = level

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": {"default": handler},
            "root": {"handlers": ["default"], "level": level},
            # Quiet noisy third-party libraries.
            "loggers": {
                "httpx": {"level": "WARNING"},
                "httpcore": {"level": "WARNING"},
            },
        }
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    return logging.getLogger(name)
