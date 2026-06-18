"""ShopSphere AI Refund Support Agent.

A modular, fault-tolerant backend that decides e-commerce refund requests using a
raw function-calling agent loop on top of a deterministic policy engine. The LLM
layer is pluggable: an open-source local model (Ollama) is used when available,
with a zero-dependency heuristic planner as an automatic fallback so the system is
fully testable from the terminal without any external service.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
