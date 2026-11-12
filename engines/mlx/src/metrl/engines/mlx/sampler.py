import secrets
from typing import cast

from metrl.engines.mlx import lora
from metrl.engines.mlx.model import Loaded
from metrl.worker import (
    Adapter,
    AdapterCache,
    ComputeThread,
    SampledSequence,
    SampleResponse,
    SamplingParams,
)
from mlx_lm.models.cache import KVCache

import mlx.core as mx

# Adapters a sampler keeps loaded.
ADAPTERS = 8


class Sampler:
    """Decodes every sample of a request as one batch over the prompt's shared
    KV cache. Without `stop`, a sequence ends at the tokenizer's
    end-of-sequence tokens; stop strings end it once the decoded completion
    contains one. The stopping token is part of the completion."""

    def __init__(self, compute: ComputeThread, bases: dict[str, Loaded]) -> None:
        self._compute = compute
        self._bases = bases
        self._adapters = AdapterCache[dict[str, lora.Delta]](ADAPTERS)

    async def sample(
        self,
        model_id: str,
        base_model: str,
        adapter: Adapter | None,
        prompt: list[int],
        num_samples: int,
        sampling_params: SamplingParams,
        include_prompt_logprobs: bool,
    ) -> SampleResponse:
        base = self._bases.get(base_model)
        if base is None:
            raise ValueError(f"this worker does not sample {base_model}")
        base.check(prompt, "prompt")
        stop = sampling_params.stop
        stop_tokens = base.eos if stop is None else {t for t in stop if isinstance(t, int)}
        stop_strings = [] if stop is None else [s for s in stop if isinstance(s, str)]
        deltas = {} if adapter is None else await self._deltas(base, adapter)
        seed = secrets.randbits(64) if sampling_params.seed is None else sampling_params.seed

        def run() -> SampleResponse:
            lora.install(base.layers, deltas)
            return _sample(
                base,
                prompt,
                num_samples,
                sampling_params,
                frozenset(stop_tokens),
                stop_strings,
                seed,
                include_prompt_logprobs=include_prompt_logprobs,
            )

        return await self._compute.run(run)

    async def _deltas(self, base: Loaded, adapter: Adapter) -> dict[str, lora.Delta]:
        async def load() -> dict[str, lora.Delta]:
            weights = await adapter.load()

            def convert() -> dict[str, lora.Delta]:
                deltas = lora.from_weights(base.layers, weights)
                mx.eval([(delta.a, delta.b) for delta in deltas.values()])
                return deltas

            return await self._compute.run(convert)

        return await self._adapters.get(adapter, load)


def _sample(
    base: Loaded,
    prompt: list[int],
    num_samples: int,
    params: SamplingParams,
    stop_tokens: frozenset[int],
    stop_strings: list[str],
    seed: int,
    *,
    include_prompt_logprobs: bool,
) -> SampleResponse:
    cache = [KVCache() for _ in base.model.layers]
    logits = base.model(mx.array([prompt]), cache=cache)[0].astype(mx.float32)
    prompt_logprobs: list[float | None] | None = None
    if include_prompt_logprobs:
        before = logits[:-1]
        picked = mx.take_along_axis(before, mx.array(prompt[1:])[:, None], axis=-1)[:, 0]
        scored = cast(list[float], (picked - mx.logsumexp(before, axis=-1)).tolist())
        prompt_logprobs = [None, *scored]
    # Every sample continues from the prompt's keys and values.
    for layer in cache:
        keys, values, offset = layer.state
        layer.state = (
            mx.repeat(keys[..., :offset, :], num_samples, axis=0),
            mx.repeat(values[..., :offset, :], num_samples, axis=0),
            offset,
        )
    next_logits = mx.repeat(logits[-1:], num_samples, axis=0)
    key = mx.random.key(seed)
    tokens: list[list[int]] = [[] for _ in range(num_samples)]
    logprobs: list[list[float]] = [[] for _ in range(num_samples)]
    stopped = [False] * num_samples
    for _ in range(params.max_tokens):
        key, draw = mx.random.split(key)
        chosen, chosen_logprobs = _draw(next_logits, params, draw)
        for index, (token, logprob) in enumerate(
            zip(
                cast(list[int], chosen.tolist()),
                cast(list[float], chosen_logprobs.tolist()),
                strict=True,
            )
        ):
            if stopped[index]:
                continue
            tokens[index].append(token)
            logprobs[index].append(logprob)
            stopped[index] = token in stop_tokens or any(
                text in base.tokenizer.decode(tokens[index]) for text in stop_strings
            )
        if all(stopped):
            break
        next_logits = base.model(chosen[:, None], cache=cache)[:, -1].astype(mx.float32)
    sequences = [
        SampledSequence(
            tokens=tokens[index],
            logprobs=logprobs[index],
            stop_reason="stop" if stopped[index] else "length",
        )
        for index in range(num_samples)
    ]
    return SampleResponse(sequences, prompt_logprobs)


def _draw(logits: mx.array, params: SamplingParams, key: mx.array) -> tuple[mx.array, mx.array]:
    """One token per row, and its logprob under the distribution it was drawn
    from: after temperature, then `top_k`, then `top_p`."""
    if params.temperature == 0:
        return mx.argmax(logits, axis=-1), mx.zeros(logits.shape[0])
    scaled = logits / params.temperature
    if 0 < params.top_k < scaled.shape[-1]:
        kth = mx.topk(scaled, params.top_k, axis=-1).min(axis=-1, keepdims=True)
        scaled = mx.where(scaled < kth, -mx.inf, scaled)
    distribution = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
    if params.top_p < 1:
        # Keeps the most likely tokens until their mass reaches `top_p`.
        order = mx.argsort(-distribution, axis=-1)
        ordered = mx.take_along_axis(distribution, order, axis=-1)
        before = mx.cumsum(mx.exp(ordered), axis=-1) - mx.exp(ordered)
        kept = mx.put_along_axis(
            mx.zeros(distribution.shape, dtype=mx.bool_), order, before < params.top_p, axis=-1
        )
        distribution = mx.where(kept, distribution, -mx.inf)
        distribution = distribution - mx.logsumexp(distribution, axis=-1, keepdims=True)
    chosen = mx.random.categorical(distribution, key=key)
    return chosen, mx.take_along_axis(distribution, chosen[:, None], axis=-1)[:, 0]
