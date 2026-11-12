from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from metrl._core import Base, BaseConfig
from metrl.worker.compute import ComputeThread
from metrl.worker.sampler import Sampler
from metrl.worker.trainer import Trainer


@dataclass(frozen=True, slots=True)
class Engine:
    """A loaded engine. `bases` has one entry per configured base, in order."""

    bases: Sequence[Base]
    trainer: Trainer
    sampler: Sampler


class Load(Protocol):
    """`metrl.engines.<name>.load`: builds the engine from its `[engine.<name>]`
    settings and loads every configured base, running blocking work on
    `compute`. Raise `ValueError` for settings the engine does not accept."""

    async def __call__(
        self, settings: Mapping[str, object], bases: Sequence[BaseConfig], compute: ComputeThread
    ) -> Engine: ...
