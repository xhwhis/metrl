"""LoRA over a shared base. Every attention and MLP linear of the base is
wrapped once at load; each call installs the low-rank update of the model or
adapter it serves, so one resident base serves every model in the process."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from metrl.worker import AdapterWeights, TensorData
from mlx_lm.models.cache import KVCache

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

# Tinker scales LoRA by 32 / rank, so its learning rate guidance applies here.
ALPHA = 32.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LanguageModel(Protocol):
    """An `mlx_lm` model: token ids of shape (batch, length) to logits of
    shape (batch, length, vocab)."""

    layers: list[nn.Module]

    def __call__(self, inputs: mx.array, cache: list[KVCache] | None = None) -> mx.array: ...


@dataclass(frozen=True, slots=True)
class Delta:
    """`scale · x @ a @ b`, with `a` of shape (in, rank) and `b` of shape (rank, out)."""

    a: mx.array
    b: mx.array
    scale: float


class Lora(nn.Module):
    """A base linear plus the installed `Delta`, if any. The base stays the
    only parameter, so gradients are taken with respect to the deltas alone."""

    def __init__(self, linear: nn.Module, input_dims: int, output_dims: int) -> None:
        super().__init__()
        self.linear = linear
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.delta: Delta | None = None

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        if self.delta is None:
            return y
        z = (x @ self.delta.a.astype(x.dtype)) @ self.delta.b.astype(x.dtype)
        return y + (self.delta.scale * z).astype(y.dtype)


def wrap(model: nn.Module) -> dict[str, Lora]:
    """Wraps every target linear of `model` and returns the wrappers by path,
    such as `model.layers.0.self_attn.q_proj`, the path PEFT names it by."""
    wrapped: dict[str, Lora] = {}
    for index, layer in enumerate(cast(list[nn.Module], model.layers)):
        replaced = []
        for name, module in layer.named_modules():
            if name.rsplit(".", 1)[-1] not in TARGET_MODULES:
                continue
            if not isinstance(module, nn.Linear | nn.QuantizedLinear):
                raise ValueError(f"cannot apply LoRA to {name}, a {type(module).__name__}")
            output_dims, input_dims = module.weight.shape
            if isinstance(module, nn.QuantizedLinear):
                input_dims = input_dims * 32 // module.bits
            lora = Lora(module, input_dims, output_dims)
            replaced.append((name, lora))
            wrapped[f"model.layers.{index}.{name}"] = lora
        layer.update_modules(tree_unflatten(replaced))
    return wrapped


def install(layers: Mapping[str, Lora], deltas: Mapping[str, Delta]) -> None:
    """Applies `deltas` and removes every other layer's, so no earlier call's
    update stays on the shared base."""
    for path, layer in layers.items():
        layer.delta = deltas.get(path)


def to_weights(deltas: Mapping[str, Delta], rank: int) -> AdapterWeights:
    """`deltas` in PEFT layout: `lora_A` of shape (rank, in), `lora_B` of shape
    (out, rank), and `lora_alpha / r` as the scale."""
    tensors = {}
    for path, delta in deltas.items():
        for name, array in (("lora_A", delta.a.T), ("lora_B", delta.b.T)):
            data = array.astype(mx.float32)
            tensors[f"base_model.model.{path}.{name}.weight"] = TensorData(
                bytes(memoryview(data)), "float32", list(data.shape)
            )
    return AdapterWeights(rank=rank, alpha=ALPHA, target_modules=TARGET_MODULES, tensors=tensors)


def from_weights(layers: Mapping[str, Lora], weights: AdapterWeights) -> dict[str, Delta]:
    """The deltas of a PEFT adapter saved by `to_weights` for the same base."""
    scale = weights.alpha / weights.rank
    tensors = weights.tensors
    deltas = {}
    for path, layer in layers.items():
        a = tensors.get(f"base_model.model.{path}.lora_A.weight")
        b = tensors.get(f"base_model.model.{path}.lora_B.weight")
        if a is None or b is None:
            continue
        expected = ([weights.rank, layer.input_dims], [layer.output_dims, weights.rank])
        if (a.shape, b.shape) != expected:
            raise ValueError(f"adapter tensors for {path} do not fit the base")
        deltas[path] = Delta(_array(a).T, _array(b).T, scale)
    if not deltas:
        raise ValueError("the adapter has no LoRA tensors for this base")
    return deltas


def _array(tensor: TensorData) -> mx.array:
    if tensor.dtype != "float32":
        raise ValueError(f"adapter tensors must be float32, not {tensor.dtype}")
    return mx.array(np.frombuffer(tensor.view(), dtype=np.float32).reshape(tensor.shape))
