"""Minimal RL loop through the Tinker SDK, used as the MLX acceptance check:
Qwen3-0.6B learns two-digit multiplication with a boxed answer. Each round
saves the weights, samples a group of answers per problem, rewards the
correct ones, and trains with `importance_sampling` on advantages centered
per group, as tinker-cookbook's `rl_loop` does. Exits nonzero unless the
reward on held-out problems, sampled at one seed before and after training,
rose by `--margin`.

Run against a server with an MLX worker, such as `just dev` and
`just worker config/worker-mlx.toml`, with `TINKER_API_KEY` set:

    uv run --no-sync python scripts/rl_loop.py
"""

import argparse
import asyncio
import os
import random
import re
import sys
import time

import tinker
from tinker import types
from transformers import PreTrainedTokenizerBase

BASE_MODEL = "Qwen/Qwen3-0.6B"
BOXED = re.compile(r"\\boxed\{(-?\d+)\}")


def problem(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randrange(11, 100), rng.randrange(11, 100)
    return f"What is {a} * {b}? Write only the final answer as \\boxed{{...}}.", a * b


def reward(text: str, answer: int) -> float:
    found = BOXED.search(text)
    return 1.0 if found is not None and int(found.group(1)) == answer else 0.0


def prompt(tokenizer: PreTrainedTokenizerBase, question: str) -> types.ModelInput:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    return types.ModelInput.from_ints(tokenizer.encode(str(text), add_special_tokens=False))


def datum(
    question: types.ModelInput, tokens: list[int], logprobs: list[float], advantage: float
) -> types.Datum:
    """Trains on the completion only: targets over the prompt carry no advantage."""
    prompt_length = question.length - 1
    return types.Datum(
        model_input=question.append(types.EncodedTextChunk(tokens=tokens[:-1])),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[0] * prompt_length + tokens, dtype="int64"),
            "logprobs": types.TensorData(data=[0.0] * prompt_length + logprobs, dtype="float32"),
            "advantages": types.TensorData(
                data=[0.0] * prompt_length + [advantage] * len(tokens), dtype="float32"
            ),
        },
    )


async def score(
    sampler: tinker.SamplingClient,
    tokenizer: PreTrainedTokenizerBase,
    problems: list[tuple[str, int]],
    group: int,
    params: types.SamplingParams,
) -> list[tuple[types.ModelInput, int, types.SampleResponse, list[float]]]:
    """Samples `group` answers to each problem and rewards each answer."""
    questions = [prompt(tokenizer, text) for text, _ in problems]
    results = await asyncio.gather(
        *(sampler.sample(question, group, params) for question in questions)
    )
    return [
        (
            question,
            answer,
            result,
            [
                reward(str(tokenizer.decode(s.tokens, skip_special_tokens=True)), answer)
                for s in result.sequences
            ],
        )
        for question, (_, answer), result in zip(questions, problems, results, strict=True)
    ]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


async def run(args: argparse.Namespace) -> tuple[float, float]:
    """Trains for `args.rounds` and returns the held-out reward before and after."""
    # Metrl does not serve the SDK's telemetry route; the SDK reads this when
    # a client is created.
    os.environ.setdefault("TINKER_TELEMETRY", "0")
    client = tinker.ServiceClient(base_url=args.base_url)
    training = await client.create_lora_training_client(
        base_model=BASE_MODEL, rank=args.rank, seed=args.seed
    )
    tokenizer = training.get_tokenizer()
    adam = types.AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    params = types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0)
    # The same problems and seed before and after training, so only the
    # weights differ between the two scores.
    held_out_problems = [problem(random.Random(f"eval/{i}")) for i in range(args.eval_problems)]
    held_out_params = types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, seed=1)

    async def held_out() -> float:
        sampler = await training.save_weights_and_get_sampling_client()
        scored = await score(sampler, tokenizer, held_out_problems, args.group, held_out_params)
        return mean([r for *_, rewards in scored for r in rewards])

    before = await held_out()
    sys.stdout.write(f"held-out reward {before:.3f}\n")
    rng = random.Random(args.seed)
    for round_index in range(args.rounds):
        started = time.monotonic()
        sampler = await training.save_weights_and_get_sampling_client()
        problems = [problem(rng) for _ in range(args.problems)]
        scored = await score(sampler, tokenizer, problems, args.group, params)
        data: list[types.Datum] = []
        for question, _, result, rewards in scored:
            baseline = mean(rewards)
            # A group whose answers all score the same carries no signal.
            if all(r == baseline for r in rewards):
                continue
            data += [
                datum(question, s.tokens, s.logprobs or [], r - baseline)
                for s, r in zip(result.sequences, rewards, strict=True)
            ]
        if data:
            backward = training.forward_backward(data, "importance_sampling")
            step = training.optim_step(adam)
            await backward
            await step
        round_reward = mean([r for *_, rewards in scored for r in rewards])
        sys.stdout.write(
            f"round {round_index:2d}  reward {round_reward:.3f}  trained on {len(data):3d}"
            f"  {time.monotonic() - started:5.1f}s\n"
        )
        sys.stdout.flush()
    after = await held_out()
    sys.stdout.write(f"held-out reward {before:.3f} -> {after:.3f}\n")
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:7580")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--problems", type=int, default=32)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--eval-problems", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.1)
    args = parser.parse_args()
    before, after = asyncio.run(run(args))
    if after < before + args.margin:
        sys.exit(f"held-out reward rose by less than {args.margin}")


if __name__ == "__main__":
    main()
