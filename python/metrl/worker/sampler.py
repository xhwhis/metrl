from typing import Protocol


class Sampler(Protocol):
    async def sample(
        self, model_id: str, prompt: list[int], num_samples: int, max_tokens: int, seed: int
    ) -> list[list[int]]: ...
