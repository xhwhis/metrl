from typing import Protocol

from metrl._core import AdamParams, AdapterWeights, Datum, ForwardBackwardOutput


class Trainer(Protocol):
    """Holds each model's trainer state. Calls for one model never overlap.
    Methods follow the Tinker `TrainingClient`.

    Raise `ValueError` for a request that cannot run as given; its message is
    shown to the user. Any other exception is reported as a server error.
    """

    async def create_model(self, model_id: str, base_model: str, rank: int, seed: int) -> None:
        """Creates the model's fresh LoRA state on `base_model`."""
        ...

    async def forward(
        self, model_id: str, data: list[Datum], loss_fn: str
    ) -> ForwardBackwardOutput: ...

    async def forward_backward(
        self, model_id: str, data: list[Datum], loss_fn: str
    ) -> ForwardBackwardOutput:
        """Like `forward`, and accumulates gradients for the next step."""
        ...

    async def optim_step(self, model_id: str, adam_params: AdamParams) -> dict[str, float]:
        """Applies the gradients accumulated since the last step."""
        ...

    async def save_weights_for_sampler(self, model_id: str) -> AdapterWeights:
        """The model's current weights, without gradients pending a step."""
        ...
