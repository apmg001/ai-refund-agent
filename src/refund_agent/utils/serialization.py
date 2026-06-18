"""JSON serialization helpers.

Tool results and reasoning-log payloads are serialized to JSON strings before being
handed to the LLM or exposed through the API. Domain objects contain ``Decimal``,
``datetime`` and ``Enum`` values that the stdlib :func:`json.dumps` cannot encode by
default, so we provide a single, reusable ``default`` hook and a convenience wrapper.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

__all__ = ["json_default", "dumps"]


def json_default(value: Any) -> Any:
    """Serialize types the stdlib JSON encoder cannot handle natively.

    Args:
        value: The object the encoder failed to serialize.

    Returns:
        A JSON-compatible representation of ``value``.

    Raises:
        TypeError: If ``value`` is of an unsupported type.
    """
    if isinstance(value, Decimal):
        # str preserves precision; floats can introduce rounding artefacts.
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):  # pydantic models
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps(obj: Any, *, indent: int | None = None) -> str:
    """Serialize ``obj`` to a JSON string using :func:`json_default`.

    Args:
        obj: The object to serialize.
        indent: Optional indentation for pretty-printing.

    Returns:
        The JSON-encoded string.
    """
    return json.dumps(obj, default=json_default, indent=indent, ensure_ascii=False)
