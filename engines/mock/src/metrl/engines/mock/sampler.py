import math
import random
import secrets
from itertools import accumulate

from metrl.engines.mock import lm
from metrl.worker import (
    Adapter,
    AdapterCache,
    ComputeThread,
    SampledSequence,
    SampleResponse,
    SamplingParams,
)

# Adapters a sampler keeps loaded.
ADAPTERS = 8


class Sampler:
    """Samples the fake language model in `lm`, so samples are reproducible
    for a seed and change with the adapter. There is no tokenizer and no
    end-of-sequence token: only token-id stops end a sequence early."""

    def __init__(self, compute: ComputeThread) -> None:
        self._compute = compute
        self._adapters = AdapterCache[bytes](ADAPTERS)

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
        stop = sampling_params.stop or []
        if any(isinstance(token, str) for token in stop):
            raise ValueError("the mock engine has no tokenizer; stop must be token ids")
        stop_tokens = frozenset(token for token in stop if isinstance(token, int))
        if include_prompt_logprobs and max(prompt) >= lm.VOCAB:
            raise ValueError(f"the mock engine scores token ids below {lm.VOCAB} only")
        weights = await self._weights(base_model, adapter)
        seed = secrets.randbits(64) if sampling_params.seed is None else sampling_params.seed
        return await self._compute.run(
            lambda: _sample(
                weights,
                prompt,
                num_samples,
                sampling_params,
                stop_tokens,
                seed,
                include_prompt_logprobs=include_prompt_logprobs,
            )
        )

    async def _weights(self, base_model: str, adapter: Adapter | None) -> bytes:
        """A digest of the weights the request samples."""
        if adapter is None:
            return lm.base(base_model)

        async def load() -> bytes:
            loaded = await adapter.load()
            return await self._compute.run(lambda: lm.adapted(base_model, loaded))

        return await self._adapters.get(adapter, load)


def _sample(
    weights: bytes,
    prompt: list[int],
    num_samples: int,
    params: SamplingParams,
    stop_tokens: frozenset[int],
    seed: int,
    *,
    include_prompt_logprobs: bool,
) -> SampleResponse:
    context = lm.context(weights)
    prompt_logprobs: list[float | None] = [None]
    for position, token in enumerate(prompt):
        if include_prompt_logprobs and position > 0:
            prompt_logprobs.append(lm.logprobs(lm.logits(context))[token])
        lm.advance(context, token)
    sequences = []
    for index in range(num_samples):
        rng = random.Random(f"{seed}/{index}")
        state = context.copy()
        tokens: list[int] = []
        logprobs: list[float] = []
        stop_reason = "length"
        for _ in range(params.max_tokens):
            candidates = _distribution(lm.logits(state), params)
            ranked = list(candidates)
            token = rng.choices(ranked, [math.exp(candidates[t]) for t in ranked])[0]
            tokens.append(token)
            logprobs.append(candidates[token])
            lm.advance(state, token)
            if token in stop_tokens:
                stop_reason = "stop"
                break
        sequences.append(SampledSequence(tokens=tokens, logprobs=logprobs, stop_reason=stop_reason))
    return SampleResponse(sequences, prompt_logprobs if include_prompt_logprobs else None)


def _distribution(logits: list[float], params: SamplingParams) -> dict[int, float]:
    """The logprob of every token the next draw can pick, most likely first,
    ties to the lower id."""
    ranked = sorted(range(lm.VOCAB), key=lambda token: -logits[token])
    if params.temperature == 0:
        return {ranked[0]: 0.0}
    if params.top_k > 0:
        ranked = ranked[: params.top_k]
    top = logits[ranked[0]]
    scaled = lm.logprobs([(logits[token] - top) / params.temperature for token in ranked])
    if params.top_p < 1:
        mass = list(accumulate(math.exp(logprob) for logprob in scaled))
        kept = next(
            (count for count, total in enumerate(mass, 1) if total >= params.top_p), len(mass)
        )
        ranked = ranked[:kept]
        scaled = lm.logprobs(scaled[:kept])
    return dict(zip(ranked, scaled, strict=True))
