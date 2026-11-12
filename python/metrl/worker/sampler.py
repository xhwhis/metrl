from typing import Protocol

from metrl._core import Adapter, SampleResponse, SamplingParams


class Sampler(Protocol):
    """Samples one request at a time; batching across requests is the
    engine's own. Methods follow the Tinker `SamplingClient`.

    Raise `ValueError` for a request that cannot run as given.

    Cache loaded adapters by `Adapter.key`, and let requests that arrive while
    an adapter loads wait for that load instead of starting another: after a
    save, many samples of the new adapter arrive at once.
    """

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
        """Returns `num_samples` sequences of at most `sampling_params.max_tokens`
        tokens each, sampled on `base_model` with `adapter` applied when it is
        set. `model_id` is the sampling session. With `include_prompt_logprobs`,
        also one logprob per prompt token, `None` for the first."""
        ...
