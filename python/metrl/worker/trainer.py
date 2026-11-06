from typing import Protocol

from metrl._core import AdamParams, Datum, ForwardBackwardOutput


class Trainer(Protocol):
    """Holds each model's trainer state. Calls for one model never overlap.

    Raise `ValueError` for a request that cannot run as given; its message is
    shown to the user. Any other exception is reported as a server error.
    """

    async def create(self, model_id: str, base: str, lora_rank: int, seed: int) -> None: ...

    async def forward_backward(
        self, model_id: str, data: list[Datum], loss_fn: str, forward_only: bool
    ) -> ForwardBackwardOutput: ...

    async def optim_step(self, model_id: str, params: AdamParams) -> dict[str, float]:
        """Applies the gradients accumulated since the last step."""
        ...
