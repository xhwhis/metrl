"""The flows of `test_training` and `test_sampling` on the MLX engine with a
tiny Qwen3. Runs only on macOS: MLX installs on Linux but its wheel there
lacks a backend, so the worker could not load it."""

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
import tinker
from conftest import Cluster

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="MLX runs only on macOS")

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
# Clients see the alias; the worker loads `TINY` from its `path`.
BASE = "tiny-qwen3"
PROMPT = [9707, 11, 1879, 0]
ADAM = tinker.AdamParams(learning_rate=1e-2, beta2=0.95, grad_clip_norm=1.0)


@pytest.fixture
def worker_config(tmp_path: Path) -> Path:
    path = tmp_path / "worker.toml"
    path.write_text(
        "\n".join(
            [
                'grpc_addr = "127.0.0.1:7590"',
                "servers = []",
                "[engine]",
                'name = "mlx"',
                "[object]",
                'url = "s3://metrl"',
                'endpoint = "http://127.0.0.1:9000"',
                'access_key_id = "metrl"',
                "[[bases]]",
                f'name = "{BASE}"',
                f'path = "{TINY}"',
                'roles = ["trainer", "sampler"]',
            ]
        )
    )
    return path


@pytest.fixture
def startup() -> float:
    # The first run downloads the model.
    return 120


def rl_datum(completion: list[int], logprobs: list[float], advantage: float) -> tinker.Datum:
    tokens = [*PROMPT, *completion]
    prompt = len(PROMPT) - 1
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=tokens[1:], dtype="int64"),
            "logprobs": tinker.TensorData(data=[0.0] * prompt + logprobs, dtype="float32"),
            "advantages": tinker.TensorData(
                data=[0.0] * prompt + [advantage] * len(completion), dtype="float32"
            ),
        },
    )


async def rl(client: tinker.ServiceClient, wanted: int) -> list[Any]:
    """Three rounds of sample, reward, and train with `importance_sampling` at
    one seed, rewarding tokens of parity `wanted`; returns what each round
    sampled and the loss it trained on."""
    training = await client.create_lora_training_client(base_model=BASE, rank=4, seed=5)
    trace: list[Any] = []
    for _ in range(3):
        sampler = await training.save_weights_and_get_sampling_client()
        sampled = await sampler.sample(
            tinker.ModelInput.from_ints(PROMPT), 4, tinker.SamplingParams(max_tokens=6, seed=1)
        )
        completions = [(s.tokens, s.logprobs or []) for s in sampled.sequences]
        rewards = [
            sum(token % 2 == wanted for token in tokens) / len(tokens) - 0.5
            for tokens, _ in completions
        ]
        data = [rl_datum(*c, r) for c, r in zip(completions, rewards, strict=True)]
        backward = training.forward_backward(data, "importance_sampling")
        step = training.optim_step(ADAM)
        output, _ = await backward, await step
        for (_, sampled_logprobs), out in zip(completions, output.loss_fn_outputs, strict=True):
            learner = out["logprobs"].tolist()[len(PROMPT) - 1 :]
            # The sampler decodes one token at a time and the trainer scores the
            # whole sequence; in bfloat16 they agree to a few ten-thousandths.
            assert learner == pytest.approx(sampled_logprobs, abs=2e-3), (
                "the trainer must score sampled tokens as the sampler drew them"
            )
        trace += [completions, output.metrics["loss:sum"]]
    return trace


def test_rl_loop_and_isolation(cluster: Cluster) -> None:
    alpha = tinker.ServiceClient(base_url=cluster.url, api_key=cluster.key("alpha"))
    beta = tinker.ServiceClient(base_url=cluster.url, api_key=cluster.key("beta"))

    async def run() -> None:
        interleaved = await asyncio.gather(rl(alpha, 1), rl(beta, 0))
        alone = [await rl(alpha, 1), await rl(beta, 0)]
        assert interleaved == alone, "one org's RL loop changed another org's"
        ours, theirs = interleaved
        assert ours[0] == theirs[0], "both orgs start from the same weights"
        assert ours[4] != theirs[4], "different rewards must train different weights"
        assert ours[4] != ours[0], "each round must sample the weights the last one trained"

    asyncio.run(run())


def test_sampling_and_short_training(cluster: Cluster) -> None:
    client = tinker.ServiceClient(base_url=cluster.url, api_key=cluster.key("alpha"))
    prompt = tinker.ModelInput.from_ints(PROMPT)

    async def run() -> None:
        sampler = await client.create_sampling_client(base_model=BASE)
        training = await client.create_lora_training_client(base_model=BASE, rank=4, seed=5)
        # Only the training client's tokenizer follows the reported
        # `tokenizer_id`; the sampling client's loads the alias itself.
        tokenizer = training.get_tokenizer()
        # A fresh adapter changes nothing, so the trainer's logprobs of the
        # prompt are the base's.
        scored = await sampler.compute_logprobs(prompt)
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(PROMPT[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(data=PROMPT[1:], dtype="int64"),
                "weights": tinker.TensorData(data=[1.0] * (len(PROMPT) - 1), dtype="float32"),
            },
        )
        output = await training.forward_backward([datum], "cross_entropy")
        assert scored[0] is None
        assert output.loss_fn_outputs[0]["logprobs"].tolist() == pytest.approx(scored[1:], abs=1e-3)
        # Inputs this short take MLX attention kernels without a backward pass.
        await training.optim_step(ADAM)
        after = await training.forward([datum], "cross_entropy")
        assert after.metrics["loss:sum"] < output.metrics["loss:sum"], "a step must lower the loss"
        assert await sampler.compute_logprobs(prompt) == scored, (
            "a trained model must not change the base other clients sample"
        )

        greedy = tinker.SamplingParams(max_tokens=8, temperature=0)
        tokens = (await sampler.sample(prompt, 1, greedy)).sequences[0].tokens
        stop = next(text for token in tokens[2:] if (text := tokenizer.decode([token])).strip())
        params = tinker.SamplingParams(max_tokens=8, temperature=0, stop=[stop])
        stopped = (await sampler.sample(prompt, 1, params)).sequences[0]
        assert stopped.stop_reason == "stop"
        assert stopped.tokens == tokens[: len(stopped.tokens)], "a stop only cuts the completion"
        assert stop in tokenizer.decode(stopped.tokens)

    asyncio.run(run())
