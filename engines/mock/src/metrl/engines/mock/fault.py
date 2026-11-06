import asyncio
import os
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Delay:
    """Holds the `call`-th call to `method`, counted from 1, for `seconds`."""

    method: str
    call: int
    seconds: float


class Faults:
    """Faults from `METRL_MOCK_FAULT`: comma-separated `delay:<method>:<call>:<seconds>`,
    such as `delay:forward_backward:1:35`."""

    def __init__(self, spec: str) -> None:
        self._delays = tuple(_parse(fault) for fault in spec.split(",") if fault)
        self._calls: Counter[str] = Counter()

    @classmethod
    def from_env(cls) -> Faults:
        return cls(os.environ.get("METRL_MOCK_FAULT", ""))

    async def before(self, method: str) -> None:
        """Counts a call to `method` and holds it if a fault targets it."""
        self._calls[method] += 1
        for delay in self._delays:
            if delay.method == method and delay.call == self._calls[method]:
                await asyncio.sleep(delay.seconds)


def _parse(fault: str) -> Delay:
    match fault.split(":"):
        case ["delay", method, call, seconds]:
            return Delay(method, int(call), float(seconds))
        case _:
            raise ValueError(f"METRL_MOCK_FAULT: cannot parse {fault!r}")
