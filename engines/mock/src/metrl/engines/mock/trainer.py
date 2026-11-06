import hashlib
import struct
from collections.abc import Buffer
from dataclasses import dataclass

from metrl.engines.mock.fault import Faults
from metrl.worker import AdamParams, Datum, ForwardBackwardOutput, TensorData


@dataclass(slots=True)
class _Model:
    """Trainer state reduced to digests, so tests can tell whether every step
    applied exactly once."""

    state: bytes
    # Gradients accumulated since the last optimizer step.
    pending: bytes = b""


class Trainer:
    """Supports `cross_entropy` only. Outputs follow from the seed and every
    applied step alone."""

    def __init__(self) -> None:
        self._models: dict[str, _Model] = {}
        self._faults = Faults.from_env()

    async def create(self, model_id: str, base: str, lora_rank: int, seed: int) -> None:
        await self._faults.before("create")
        self._models[model_id] = _Model(
            state=_digest(base.encode(), struct.pack("<IQ", lora_rank, seed))
        )

    async def forward_backward(
        self, model_id: str, data: list[Datum], loss_fn: str, forward_only: bool
    ) -> ForwardBackwardOutput:
        await self._faults.before("forward_backward")
        if loss_fn != "cross_entropy":
            raise ValueError(f"loss_fn {loss_fn} is not supported")
        model = self._models[model_id]
        outputs: list[dict[str, TensorData]] = []
        loss = 0.0
        pending = model.pending
        for index, datum in enumerate(data):
            targets = _input(datum, "target_tokens", "int64")
            weights = _input(datum, "weights", "float32")
            target_ids = targets.view().cast("q").tolist()
            weight_values = weights.view().cast("f").tolist()
            if not len(target_ids) == len(weight_values) == len(datum.tokens):
                raise ValueError(
                    f"data[{index}]: target_tokens and weights must match the model input length"
                )
            logprobs = [
                _logprob(model.state, position, target)
                for position, target in enumerate(target_ids)
            ]
            loss -= sum(w * lp for w, lp in zip(weight_values, logprobs, strict=True))
            packed = struct.pack(f"<{len(logprobs)}f", *logprobs)
            outputs.append({"logprobs": TensorData(packed, "float32", [len(logprobs)])})
            if not forward_only:
                tokens = struct.pack(f"<{len(datum.tokens)}i", *datum.tokens)
                pending = _digest(pending, tokens, targets.view(), weights.view())
        if not forward_only:
            model.pending = pending
        return ForwardBackwardOutput(outputs, {"loss:sum": loss})

    async def optim_step(self, model_id: str, params: AdamParams) -> dict[str, float]:
        await self._faults.before("optim_step")
        model = self._models[model_id]
        hyperparameters = struct.pack(
            "<6d",
            params.learning_rate,
            params.beta1,
            params.beta2,
            params.eps,
            params.weight_decay,
            params.grad_clip_norm,
        )
        model.state = _digest(model.state, model.pending, hyperparameters)
        model.pending = b""
        # 48 bits, so the hash is exact as a float.
        return {"state_hash:unique": float(int.from_bytes(model.state[:6], "little"))}


def _input(datum: Datum, name: str, dtype: str) -> TensorData:
    tensor = datum.loss_fn_inputs.get(name)
    if tensor is None or tensor.dtype != dtype:
        raise ValueError(f"cross_entropy needs {name} as {dtype}")
    return tensor


def _logprob(state: bytes, position: int, target: int) -> float:
    """A fake log-probability in [-8, -0.125], exact in float32."""
    return -(1 + _digest(state, struct.pack("<qq", position, target))[0] % 64) / 8


def _digest(*parts: Buffer) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        view = memoryview(part)
        digest.update(struct.pack("<Q", view.nbytes))
        digest.update(view)
    return digest.digest()
