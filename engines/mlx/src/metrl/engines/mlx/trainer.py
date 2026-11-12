import math
from dataclasses import dataclass

import numpy as np
from metrl.engines.mlx import lora
from metrl.engines.mlx.model import Loaded, score
from metrl.worker import (
    AdamParams,
    AdapterWeights,
    ComputeThread,
    Datum,
    ForwardBackwardOutput,
    TensorData,
)

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map

# `a` and `b` of every wrapped layer, by its path.
type Weights = dict[str, dict[str, mx.array]]

LOSS_INPUTS = {
    "cross_entropy": ("weights",),
    "importance_sampling": ("logprobs", "advantages"),
}


@dataclass(slots=True)
class _Model:
    base: Loaded
    rank: int
    weights: Weights
    optimizer: optim.AdamW
    # Gradients summed since the last optimizer step.
    pending: Weights | None = None


@dataclass(frozen=True, slots=True)
class _Example:
    """One datum, checked and on the device."""

    tokens: mx.array
    targets: mx.array
    inputs: dict[str, mx.array]


class Trainer:
    """LoRA on the attention and MLP linears, trained with AdamW. Supports
    `cross_entropy` and `importance_sampling`; each loss is a sum over tokens,
    and gradients add up over `forward_backward` calls until `optim_step`."""

    def __init__(self, compute: ComputeThread, bases: dict[str, Loaded]) -> None:
        self._compute = compute
        self._bases = bases
        self._models: dict[str, _Model] = {}

    async def create_model(self, model_id: str, base_model: str, rank: int, seed: int) -> None:
        base = self._bases.get(base_model)
        if base is None:
            raise ValueError(f"this worker does not train {base_model}")

        def run() -> None:
            # As PEFT initializes: `a` uniform in ±1/√in and `b` zero, so a new
            # model scores exactly as its base.
            keys = mx.random.split(mx.random.key(seed), len(base.layers))
            weights = {
                path: {
                    "a": mx.random.uniform(
                        -1 / math.sqrt(layer.input_dims),
                        1 / math.sqrt(layer.input_dims),
                        (layer.input_dims, rank),
                        key=key,
                    ),
                    "b": mx.zeros((rank, layer.output_dims)),
                }
                for key, (path, layer) in zip(keys, base.layers.items(), strict=True)
            }
            mx.eval(weights)
            optimizer = optim.AdamW(learning_rate=0.0, bias_correction=True)
            self._models[model_id] = _Model(base, rank, weights, optimizer)

        await self._compute.run(run)

    async def forward(
        self, model_id: str, data: list[Datum], loss_fn: str
    ) -> ForwardBackwardOutput:
        return await self._compute.run(lambda: self._run(model_id, data, loss_fn, backward=False))

    async def forward_backward(
        self, model_id: str, data: list[Datum], loss_fn: str
    ) -> ForwardBackwardOutput:
        return await self._compute.run(lambda: self._run(model_id, data, loss_fn, backward=True))

    async def optim_step(self, model_id: str, adam_params: AdamParams) -> dict[str, float]:
        def run() -> dict[str, float]:
            model = self._models[model_id]
            if model.pending is None:
                return {}
            gradients = model.pending
            if adam_params.grad_clip_norm > 0:
                gradients, _ = optim.clip_grad_norm(gradients, adam_params.grad_clip_norm)
            optimizer = model.optimizer
            optimizer.learning_rate = adam_params.learning_rate
            optimizer.betas = [adam_params.beta1, adam_params.beta2]
            optimizer.eps = adam_params.eps
            optimizer.weight_decay = adam_params.weight_decay
            model.weights = optimizer.apply_gradients(gradients, model.weights)
            mx.eval(model.weights, optimizer.state)
            model.pending = None
            return {}

        return await self._compute.run(run)

    async def save_weights_for_sampler(self, model_id: str) -> AdapterWeights:
        def run() -> AdapterWeights:
            model = self._models[model_id]
            return lora.to_weights(_deltas(model, model.weights), model.rank)

        return await self._compute.run(run)

    def _run(
        self, model_id: str, data: list[Datum], loss_fn: str, *, backward: bool
    ) -> ForwardBackwardOutput:
        names = LOSS_INPUTS.get(loss_fn)
        if names is None:
            raise ValueError(f"loss_fn {loss_fn} is not supported")
        model = self._models[model_id]
        examples = [_example(model.base, d, i, loss_fn, names) for i, d in enumerate(data)]
        pending = model.pending
        outputs: list[dict[str, TensorData]] = []
        total = 0.0
        step = mx.value_and_grad(_objective)
        for example in examples:
            if backward:
                (loss, logprobs), gradients = step(model.weights, model, example, loss_fn)
                pending = gradients if pending is None else tree_map(mx.add, pending, gradients)
                mx.eval(loss, logprobs, pending)
            else:
                loss, logprobs = _objective(model.weights, model, example, loss_fn)
                mx.eval(loss, logprobs)
            total += float(loss)
            outputs.append(
                {"logprobs": TensorData(bytes(memoryview(logprobs)), "float32", [logprobs.size])}
            )
        if backward:
            model.pending = pending
        return ForwardBackwardOutput(outputs, {"loss:sum": total})


def _deltas(model: _Model, weights: Weights) -> dict[str, lora.Delta]:
    scale = lora.ALPHA / model.rank
    return {path: lora.Delta(w["a"], w["b"], scale) for path, w in weights.items()}


def _objective(
    weights: Weights, model: _Model, example: _Example, loss_fn: str
) -> tuple[mx.array, mx.array]:
    """The loss under `weights` and the target logprobs it came from."""
    lora.install(model.base.layers, _deltas(model, weights))
    logprobs = score(model.base.model, example.tokens, example.targets)
    inputs = example.inputs
    if loss_fn == "cross_entropy":
        loss = -(inputs["weights"] * logprobs).sum()
    else:
        ratio = mx.exp(logprobs - inputs["logprobs"])
        loss = -(ratio * inputs["advantages"]).sum()
    return loss, logprobs


def _example(
    base: Loaded, datum: Datum, index: int, loss_fn: str, names: tuple[str, ...]
) -> _Example:
    tokens = datum.tokens
    if not tokens:
        raise ValueError(f"data[{index}]: model_input is empty")
    base.check(tokens, f"data[{index}].model_input")
    targets = np.frombuffer(_input(datum, index, loss_fn, "target_tokens", "int64"), np.int64)
    base.check(targets.tolist(), f"data[{index}].target_tokens")
    inputs = {
        name: mx.array(np.frombuffer(_input(datum, index, loss_fn, name, "float32"), np.float32))
        for name in names
    }
    return _Example(mx.array(tokens), mx.array(targets, dtype=mx.int32), inputs)


def _input(datum: Datum, index: int, loss_fn: str, name: str, dtype: str) -> memoryview:
    """The bytes of loss input `name`, one value per model input token."""
    tensor = datum.loss_fn_inputs.get(name)
    length = len(datum.tokens)
    if tensor is None or tensor.dtype != dtype or tensor.shape != [length]:
        raise ValueError(f"data[{index}]: {loss_fn} needs {name} as {dtype} of shape ({length},)")
    return tensor.view()
