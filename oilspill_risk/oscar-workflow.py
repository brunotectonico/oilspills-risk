"""Compatibility wrapper for the importable :mod:`oilspill_risk.oscar_workflow` module."""

from __future__ import annotations

try:
    from .oscar_workflow import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - supports direct script execution during exploration
    from oscar_workflow import *  # type: ignore # noqa: F401,F403
