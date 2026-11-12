from dataclasses import dataclass

from metrl.engines.mlx.lora import LanguageModel, Lora
from metrl.worker import Base
from mlx_lm.tokenizer_utils import TokenizerWrapper

import mlx.core as mx

# MLX 0.32 attends at most 8 queries with kernels that have no backward pass,
# so shorter inputs are padded; causal attention keeps the padding from
# changing earlier positions.
MIN_LENGTH = 9


@dataclass(frozen=True, slots=True)
class Loaded:
    """A resident base that the trainer and sampler share. Calls on it run only
    on the compute thread, and each installs its own deltas first."""

    model: LanguageModel
    tokenizer: TokenizerWrapper
    layers: dict[str, Lora]
    vocab: int
    eos: frozenset[int]
    # What the worker reports for it.
    described: Base

    def check(self, tokens: list[int], what: str) -> None:
        """Raises for token ids the base cannot embed."""
        if any(not 0 <= token < self.vocab for token in tokens):
            raise ValueError(f"{what} must be token ids below {self.vocab}")


def score(model: LanguageModel, tokens: mx.array, targets: mx.array) -> mx.array:
    """Each target's logprob given the tokens up to and including its position,
    in float32."""
    length = tokens.size
    padded = mx.pad(tokens, (0, max(0, MIN_LENGTH - length)))
    logits = model(padded[None])[0, :length].astype(mx.float32)
    picked = mx.take_along_axis(logits, targets[:, None], axis=-1)[:, 0]
    return picked - mx.logsumexp(logits, axis=-1)
