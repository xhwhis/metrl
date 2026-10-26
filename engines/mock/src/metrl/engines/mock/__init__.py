"""Deterministic engine with no ML dependencies, used by E2E and chaos tests."""

from metrl.engines.mock.sampler import Sampler

__all__ = ["Sampler"]
