"""Deterministic engine with no ML dependencies, used by E2E and chaos tests."""

from metrl.engines.mock.engine import load

__all__ = ["load"]
