import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import cast

import mlx_lm
from huggingface_hub import snapshot_download
from metrl.engines.mlx import lora
from metrl.engines.mlx.lora import LanguageModel
from metrl.engines.mlx.model import Loaded
from metrl.engines.mlx.sampler import Sampler
from metrl.engines.mlx.trainer import Trainer
from metrl.worker import Base, BaseConfig, ComputeThread, Engine
from mlx_lm.tokenizer_utils import TokenizerWrapper

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# The adapter sizes this engine trains and samples.
MAX_LORA_RANK = 64


async def load(
    settings: Mapping[str, object], bases: Sequence[BaseConfig], compute: ComputeThread
) -> Engine:
    """Takes no settings. Loads each base from a local directory, or from the
    Hugging Face repository `path` (or `name`) names, once for both roles."""
    if settings:
        raise ValueError(f"the mlx engine takes no settings, got {sorted(settings)}")
    loaded = {config.name: await compute.run(partial(_load, config)) for config in bases}
    return Engine(
        bases=[base.described for base in loaded.values()],
        trainer=Trainer(compute, loaded),
        sampler=Sampler(compute, loaded),
    )


def _load(config: BaseConfig) -> Loaded:
    source = config.path or config.name
    local = Path(source).is_dir()
    path = Path(source) if local else Path(snapshot_download(source))
    loaded = mlx_lm.load(str(path), return_config=True)
    model, tokenizer, model_config = cast(tuple[nn.Module, TokenizerWrapper, dict], loaded)
    if hasattr(model, "make_cache"):
        raise ValueError(f"{source}: only models with a plain KV cache are supported")
    # The revision follows the weights as loaded and the config, wherever the
    # files came from.
    digest = hashlib.blake2b(json.dumps(model_config, sort_keys=True).encode(), digest_size=16)
    for name, parameter in cast(list[tuple[str, mx.array]], tree_flatten(model.parameters())):
        digest.update(f"{name}:{parameter.dtype}:{parameter.shape}".encode())
        digest.update(memoryview(parameter))
    return Loaded(
        model=cast(LanguageModel, model),
        tokenizer=tokenizer,
        layers=lora.wrap(model),
        vocab=model_config["vocab_size"],
        eos=frozenset(tokenizer.eos_token_ids),
        described=Base(
            name=config.name,
            revision=digest.hexdigest(),
            arch=model_config["model_type"],
            # A local directory means nothing to clients, so they get the name.
            tokenizer_id=config.name if local else source,
            max_context_length=model_config.get("max_position_embeddings", 4096),
            max_lora_rank=MAX_LORA_RANK,
        ),
    )
