import hashlib
import math
import struct
from collections.abc import Buffer
from dataclasses import dataclass

from metrl.engines.mock import lm
from metrl.engines.mock.fault import Faults
from metrl.worker import (
    AdamParams,
    AdapterWeights,
    ComputeThread,
    Datum,
    ForwardBackwardOutput,
    TensorData,
)


@dataclass(slots=True)
class _Model:
    """Trainer state reduced to digests, so tests can tell whether every step
    applied exactly once."""

    base_model: str
    rank: int
    state: bytes
    # Gradients accumulated since the last optimizer step.
    pending: bytes = b""


class Trainer:
    """Supports `cross_entropy` and `importance_sampling`. Saved weights follow
    from the seed and every applied step alone, and logprobs are the fake
    language model's under those weights, so they match what the sampler gives
    for a save of the same state. State lives only on the compute thread."""

    def __init__(self, compute: ComputeThread, faults: Faults) -> None:
        self._compute = compute
        self._faults = faults
        self._models: dict[str, _Model] = {}

    async def create_model(self, model_id: str, base_model: str, rank: int, seed: int) -> None:
        await self._faults.before("create_model")

        def run() -> None:
            state = _digest(base_model.encode(), struct.pack("<IQ", rank, seed))
            self._models[model_id] = _Model(base_model=base_model, rank=rank, state=state)

        await self._compute.run(run)

    async def forward(
        self, model_id: str, data: list[Datum], loss_fn: str
    ) -> ForwardBackwardOutput:
        await self._faults.before("forward")
        return await self._compute.run(lambda: self._run(model_id, data, loss_fn, backward=False))

    async def forward_backward(
        self, model_id: str, data: list[Datum], loss_fn: str
    ) -> ForwardBackwardOutput:
        await self._faults.before("forward_backward")
        return await self._compute.run(lambda: self._run(model_id, data, loss_fn, backward=True))

    async def optim_step(self, model_id: str, adam_params: AdamParams) -> dict[str, float]:
        await self._faults.before("optim_step")

        def run() -> dict[str, float]:
            model = self._models[model_id]
            hyperparameters = struct.pack(
                "<6d",
                adam_params.learning_rate,
                adam_params.beta1,
                adam_params.beta2,
                adam_params.eps,
                adam_params.weight_decay,
                adam_params.grad_clip_norm,
            )
            model.state = _digest(model.state, model.pending, hyperparameters)
            model.pending = b""
            # 48 bits, so the hash is exact as a float.
            return {"state_hash:unique": float(int.from_bytes(model.state[:6], "little"))}

        return await self._compute.run(run)

    async def save_weights_for_sampler(self, model_id: str) -> AdapterWeights:
        await self._faults.before("save_weights_for_sampler")

        return await self._compute.run(lambda: _adapter(self._models[model_id]))

    def _run(
        self, model_id: str, data: list[Datum], loss_fn: str, *, backward: bool
    ) -> ForwardBackwardOutput:
        if loss_fn not in ("cross_entropy", "importance_sampling"):
            raise ValueError(f"loss_fn {loss_fn} is not supported")
        model = self._models[model_id]
        weights = lm.adapted(model.base_model, _adapter(model))
        outputs: list[dict[str, TensorData]] = []
        loss = 0.0
        pending = model.pending
        for index, datum in enumerate(data):
            targets = _input(datum, index, loss_fn, "target_tokens", "int64")
            target_ids = targets.view().cast("q").tolist()
            if any(not 0 <= target < lm.VOCAB for target in target_ids):
                raise ValueError(f"data[{index}]: target_tokens must be below {lm.VOCAB}")
            logprobs = _score(weights, datum.tokens, target_ids)
            if loss_fn == "cross_entropy":
                token_weights = _input(datum, index, loss_fn, "weights", "float32")
                inputs = [targets, token_weights]
                terms = [
                    -w * lp for w, lp in zip(token_weights.view().cast("f"), logprobs, strict=True)
                ]
            else:
                sampled = _input(datum, index, loss_fn, "logprobs", "float32")
                advantages = _input(datum, index, loss_fn, "advantages", "float32")
                inputs = [targets, sampled, advantages]
                terms = [
                    -math.exp(lp - q) * a
                    for lp, q, a in zip(
                        logprobs, sampled.view().cast("f"), advantages.view().cast("f"), strict=True
                    )
                ]
            loss += math.fsum(terms)
            packed = struct.pack(f"<{len(logprobs)}f", *logprobs)
            outputs.append({"logprobs": TensorData(packed, "float32", [len(logprobs)])})
            if backward:
                tokens = struct.pack(f"<{len(datum.tokens)}i", *datum.tokens)
                views = [tensor.view() for tensor in inputs]
                pending = _digest(pending, loss_fn.encode(), tokens, *views)
        if backward:
            model.pending = pending
        return ForwardBackwardOutput(outputs, {"loss:sum": loss})


def _input(datum: Datum, index: int, loss_fn: str, name: str, dtype: str) -> TensorData:
    """The loss input `name`, one value per model input token."""
    tensor = datum.loss_fn_inputs.get(name)
    length = len(datum.tokens)
    if tensor is None or tensor.dtype != dtype or tensor.shape != [length]:
        raise ValueError(f"data[{index}]: {loss_fn} needs {name} as {dtype} of shape ({length},)")
    return tensor


def _adapter(model: _Model) -> AdapterWeights:
    # Every row and column holds the state's bytes, exact in float32.
    width, rank = len(model.state), model.rank
    lora_a = struct.pack(f"<{width}f", *model.state) * rank
    columns = [byte for byte in model.state for _ in range(rank)]
    lora_b = struct.pack(f"<{width * rank}f", *columns)
    return AdapterWeights(
        rank=rank,
        alpha=float(rank),
        target_modules=["mock"],
        tensors={
            "base_model.model.mock.lora_A.weight": TensorData(lora_a, "float32", [rank, width]),
            "base_model.model.mock.lora_B.weight": TensorData(lora_b, "float32", [width, rank]),
        },
    )


def _score(weights: bytes, tokens: list[int], targets: list[int]) -> list[float]:
    """Each target's logprob given the model input up to and including its
    position."""
    state = lm.context(weights)
    scored = []
    for token, target in zip(tokens, targets, strict=True):
        lm.advance(state, token)
        scored.append(lm.logprobs(lm.logits(state))[target])
    return scored


def _digest(*parts: Buffer) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        view = memoryview(part)
        digest.update(struct.pack("<Q", view.nbytes))
        digest.update(view)
    return digest.digest()
