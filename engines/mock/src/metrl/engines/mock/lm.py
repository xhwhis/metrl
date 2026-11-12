"""The mock's fake language model over token ids below `VOCAB`. A token's
logits follow from a weights digest and every token before it. The trainer
scores its current adapter and the sampler the saved one through the same
functions, so on-policy logprobs agree exactly, as they would with real
weights."""

import hashlib
import math
import struct

from metrl.worker import AdapterWeights

VOCAB = 256


def base(base_model: str) -> bytes:
    """The weights digest of a bare base."""
    return hashlib.blake2b(base_model.encode(), digest_size=16).digest()


def adapted(base_model: str, adapter: AdapterWeights) -> bytes:
    """The weights digest of `base_model` with `adapter` applied."""
    digest = hashlib.blake2b(base(base_model), digest_size=16)
    digest.update(struct.pack("<Id", adapter.rank, adapter.alpha))
    for name, tensor in adapter.tensors.items():
        header = f"{name}:{tensor.dtype}:{tensor.shape}".encode()
        digest.update(struct.pack("<Q", len(header)) + header)
        digest.update(tensor.view())
    return digest.digest()


def context(weights: bytes) -> hashlib.blake2b:
    """The state before the first token."""
    return hashlib.blake2b(weights, digest_size=16)


def advance(state: hashlib.blake2b, token: int) -> None:
    state.update(struct.pack("<i", token))


def logits(state: hashlib.blake2b) -> list[float]:
    """The next token's logits, in [0, 8)."""
    return [byte / 32 for byte in hashlib.shake_256(state.digest()).digest(VOCAB)]


def logprobs(logits: list[float]) -> list[float]:
    top = max(logits)
    shifted = [logit - top for logit in logits]
    total = math.log(math.fsum(math.exp(logit) for logit in shifted))
    return [logit - total for logit in shifted]
