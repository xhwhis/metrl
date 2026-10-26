from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Sampler:
    async def sample(
        self, model_id: str, prompt: list[int], num_samples: int, max_tokens: int, seed: int
    ) -> list[list[int]]:
        offset = sum(prompt) + seed
        return [
            [(offset + sequence * 17 + position) % 256 for position in range(max_tokens)]
            for sequence in range(num_samples)
        ]
