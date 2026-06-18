#!/usr/bin/env python3
"""Convenience launcher for the refund-agent CLI without installing the package.

This adds ``src`` to the import path and delegates to :func:`refund_agent.cli.main`,
so the agent can be run directly from a checkout::

    python run_cli.py --demo
    python run_cli.py --message "Refund ORD-10001, it's defective"
    python run_cli.py            # interactive REPL
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from refund_agent.cli import main  # noqa: E402  (path set up above)

if __name__ == "__main__":
    raise SystemExit(main())
