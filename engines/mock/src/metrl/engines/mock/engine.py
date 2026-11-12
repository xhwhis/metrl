from collections.abc import Mapping, Sequence

from metrl.engines.mock.fault import Faults
from metrl.engines.mock.sampler import Sampler
from metrl.engines.mock.trainer import Trainer
from metrl.worker import Base, BaseConfig, ComputeThread, Engine


async def load(
    settings: Mapping[str, object], bases: Sequence[BaseConfig], compute: ComputeThread
) -> Engine:
    """Takes no settings. Every base reports the `Qwen/Qwen3-0.6B` tokenizer, so
    SDK tokenizer calls work against it."""
    if settings:
        raise ValueError(f"the mock engine takes no settings, got {sorted(settings)}")
    faults = Faults.from_env()
    return Engine(
        bases=[
            Base(
                name=base.name,
                revision="mock-1",
                arch="mock",
                tokenizer_id="Qwen/Qwen3-0.6B",
                max_context_length=4096,
                max_lora_rank=64,
            )
            for base in bases
        ],
        trainer=Trainer(compute, faults),
        sampler=Sampler(compute),
    )
