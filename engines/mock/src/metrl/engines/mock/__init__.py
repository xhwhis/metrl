"""Deterministic engine with no ML dependencies, used by E2E and chaos tests."""

from metrl.engines.mock.sampler import Sampler
from metrl.engines.mock.trainer import Trainer

__all__ = ["Sampler", "Trainer"]
