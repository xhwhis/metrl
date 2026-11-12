import asyncio
import itertools
import math
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
import pytest
import tinker
from conftest import SERVER, Cluster, free_ports, spawn, wait_for
from tinker import types
from tinker.proto.request_conv import forward_backward_request_to_proto
from tinker.proto.response_conv import deserialize_forward_backward_output

ADAM = tinker.AdamParams(learning_rate=1e-4)


def datum(targets: list[int]) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([2, 7, 11, 4]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=targets, dtype="int64"),
            "weights": tinker.TensorData(data=[1.0, 1.0, 0.5, 0.0], dtype="float32"),
        },
    )


async def train(client: tinker.ServiceClient, targets: list[int]) -> list[Any]:
    """Two pipelined steps, then a forward pass; returns every observable result."""
    training = await client.create_lora_training_client(base_model="mock", rank=8, seed=5)
    trace: list[Any] = []
    for _ in range(2):
        backward = training.forward_backward([datum(targets)], "cross_entropy")
        step = training.optim_step(ADAM)
        output, applied = await backward, await step
        trace += [output.loss_fn_outputs[0]["logprobs"].tolist(), output.metrics, applied.metrics]
    forward = await training.forward([datum(targets)], "cross_entropy")
    return [*trace, forward.loss_fn_outputs[0]["logprobs"].tolist()]


def test_training_sdk_and_isolation(cluster: Cluster) -> None:
    alpha_key, beta_key = cluster.key("alpha"), cluster.key("beta")
    alpha = tinker.ServiceClient(base_url=cluster.url, api_key=alpha_key)
    beta = tinker.ServiceClient(base_url=cluster.url, api_key=beta_key)

    async def run() -> None:
        interleaved = await asyncio.gather(train(alpha, [7, 11, 4, 9]), train(beta, [3, 3, 8, 1]))
        alone = [await train(alpha, [7, 11, 4, 9]), await train(beta, [3, 3, 8, 1])]
        assert interleaved == alone, "one org's tasks changed another org's model"
        trace = interleaved[0]
        assert trace != interleaved[1], "the orgs' different data must give different results"
        assert trace[0] != trace[3], "the first optimizer step did not change the model"
        assert trace[2] != trace[5], "both steps must change the model state"

    asyncio.run(run())
    training = alpha.create_lora_training_client(base_model="mock", rank=4).result()
    info = training.get_info()
    assert (info.model_data.model_name, info.model_data.tokenizer_id) == ("mock", "Qwen/Qwen3-0.6B")
    assert info.lora_rank == 4


def forward_backward(model: str, seq: int, loss_fn: types.LossFnType = "cross_entropy") -> bytes:
    """A forward-backward request body as the SDK encodes it."""
    request = types.ForwardBackwardRequest(
        forward_backward_input=types.ForwardBackwardInput(
            data=[datum([7, 11, 4, 9])], loss_fn=loss_fn
        ),
        model_id=model,
        seq_id=seq,
    )
    return forward_backward_request_to_proto(request).SerializeToString()


def result(
    client: httpx.Client, headers: dict[str, str], request: str, timeout: float = 15
) -> httpx.Response:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.post(
            "/api/v1/retrieve_future", headers=headers, json={"request_id": request}
        )
        if response.status_code != 408:
            return response
        assert response.json() == {
            "type": "try_again",
            "request_id": request,
            "queue_state": "active",
        }
        time.sleep(0.05)
    pytest.fail(f"request {request} never finished")


def test_training_routes(cluster: Cluster) -> None:
    alpha = {"x-api-key": cluster.key("alpha")}
    beta = {"x-api-key": cluster.key("beta")}
    with httpx.Client(base_url=cluster.url, timeout=30) as client:
        session = client.post("/api/v1/create_session", headers=alpha).json()["session_id"]
        body = {
            "session_id": session,
            "model_seq_id": 0,
            "base_model": "mock",
            "lora_config": {"rank": 8, "seed": 5},
            "optimizer_config": {"type": "adamw"},
        }
        created = client.post("/api/v1/create_model", headers=alpha, json=body).json()
        model = created["model_id"]
        assert client.post("/api/v1/create_model", headers=alpha, json=body).json() == created
        assert client.post("/api/v1/create_model", headers=beta, json=body).status_code == 404
        changed_retry = body | {"lora_config": {"rank": 4, "seed": 5}}
        response = client.post("/api/v1/create_model", headers=alpha, json=changed_retry)
        assert response.status_code == 400
        for invalid in (
            {"base_model": "missing"},
            {"lora_config": {"rank": 65}},
            {"lora_config": {"rank": 0}},
            {"lora_config": {"rank": 8, "train_mlp": False}},
            {"optimizer_config": {"type": "dimuon"}},
        ):
            request = body | {"model_seq_id": 1} | invalid
            response = client.post("/api/v1/create_model", headers=alpha, json=request)
            assert response.status_code == 400, invalid
        assert result(client, alpha, created["request_id"]).json() == {
            "model_id": model,
            "type": "create_model",
        }
        assert (
            client.post(
                "/api/v1/retrieve_future", headers=beta, json={"request_id": created["request_id"]}
            ).status_code
            == 404
        )
        info = client.post("/api/v1/get_info", headers=beta, json={"model_id": model})
        assert info.status_code == 404

        def submit(headers: dict[str, str], seq: int, **adam: float) -> httpx.Response:
            params = ADAM.model_dump() | adam
            step = {"model_id": model, "seq_id": seq, "adam_params": params}
            return client.post("/api/v1/optim_step", headers=headers, json=step)

        backward = client.post(
            "/api/v1/forward_backward", headers=alpha, content=forward_backward(model, 1)
        ).json()
        retried = client.post(
            "/api/v1/forward_backward", headers=alpha, content=forward_backward(model, 1)
        )
        assert retried.json() == backward, "a retried seq must return the request that took it"
        changed = client.post(
            "/api/v1/forward_backward", headers=alpha, content=forward_backward(model, 1, "ppo")
        )
        assert changed.status_code == 400
        assert (
            client.post(
                "/api/v1/forward_backward", headers=beta, content=forward_backward(model, 2)
            ).status_code
            == 404
        )
        step = submit(alpha, 2).json()
        assert submit(alpha, 2).json() == step
        assert submit(alpha, 2, learning_rate=1.0).status_code == 400
        assert submit(beta, 2).status_code == 404
        no_seq = {"model_id": model, "adam_params": ADAM.model_dump()}
        assert client.post("/api/v1/optim_step", headers=alpha, json=no_seq).status_code == 400
        # Beyond bigint; a 5xx here would make the SDK retry for minutes.
        assert submit(alpha, 2**63).status_code == 400

        stolen = {"request_id": backward["request_id"]}
        assert client.post("/api/v1/retrieve_future", headers=beta, json=stolen).status_code == 404
        output = result(client, alpha, backward["request_id"])
        assert output.headers["content-type"] == "application/x-protobuf"
        logprobs = deserialize_forward_backward_output(output.content).loss_fn_outputs[0]
        assert len(logprobs["logprobs"].tolist()) == 4
        assert "state_hash:unique" in result(client, alpha, step["request_id"]).json()["metrics"]
        unsupported = client.post(
            "/api/v1/forward_backward", headers=alpha, content=forward_backward(model, 3, "ppo")
        ).json()
        assert result(client, alpha, unsupported["request_id"]).json() == {
            "error": "loss_fn ppo is not supported",
            "category": "user",
        }

        # Seq 4 is missing, so every later task waits for it.
        waiting = [submit(alpha, seq) for seq in range(10, 74)]
        # Fails if finished tasks still count as pending.
        assert all(response.status_code == 200 for response in waiting)
        # The view may still lag behind some of them.
        seqs = itertools.count(74)
        wait_for(
            lambda: submit(alpha, next(seqs)).status_code == 429,
            "a model with 64 waiting tasks kept accepting more",
        )
        assert submit(alpha, next(seqs)).headers["retry-after"] == "1"
        assert submit(alpha, 10).json() == waiting[0].json(), (
            "a retry must not count against the limit"
        )
        # The seq the model runs next is still accepted, or the model would stall.
        unblocked = submit(alpha, 4)
        assert unblocked.status_code == 200
        assert "metrics" in result(client, alpha, unblocked.json()["request_id"]).json()

        # Above axum's 2 MiB default, so only the route's own limit lets it reach decoding.
        large = client.post("/api/v1/forward_backward", headers=alpha, content=bytes(3 << 20))
        assert large.status_code == 400
        oversized = client.post(
            "/api/v1/forward_backward", headers=alpha, content=bytes((256 << 20) + 1)
        )
        assert oversized.status_code == 413


def test_restart_keeps_models_and_ends_long_polls(cluster: Cluster, tmp_path: Path) -> None:
    alpha = {"x-api-key": cluster.key("alpha")}
    with httpx.Client(base_url=cluster.url, timeout=60) as client:
        session = client.post("/api/v1/create_session", headers=alpha).json()["session_id"]
        body = {
            "session_id": session,
            "model_seq_id": 0,
            "base_model": "mock",
            "lora_config": {"rank": 8},
        }
        created = client.post("/api/v1/create_model", headers=alpha, json=body).json()
        # Seq 2 waits for seq 1, so its future stays pending.
        pending = client.post(
            "/api/v1/forward_backward",
            headers=alpha,
            content=forward_backward(created["model_id"], 2),
        ).json()["request_id"]
        with ThreadPoolExecutor(1) as pool:
            held = pool.submit(
                client.post,
                "/api/v1/retrieve_future",
                headers=alpha,
                json={"request_id": pending},
            )
            # Long enough for the request to reach the server and start waiting.
            time.sleep(1)
            stopping = time.monotonic()
            cluster.server.send_signal(signal.SIGTERM)
            assert held.result().status_code == 408
            assert cluster.server.wait(timeout=10) == 0
        assert time.monotonic() - stopping < 5, "shutdown waited for a held retrieve_future"

        # The restarted server never saw the session, so only Postgres knows the model.
        with ExitStack() as stack, (tmp_path / "restarted.log").open("wb") as log:
            spawn(stack, [SERVER], cluster.env, log)

            def retried() -> bool:
                try:
                    response = client.post("/api/v1/create_model", headers=alpha, json=body)
                except httpx.TransportError:
                    return False
                assert response.json() == created, response.text
                return True

            wait_for(retried, "the restarted server never answered create_model")


def create(client: httpx.Client, headers: dict[str, str], session: str, model_seq_id: int) -> str:
    """A rank-8 mock model with seed 5, so models created alike train alike."""
    body = {
        "session_id": session,
        "model_seq_id": model_seq_id,
        "base_model": "mock",
        "lora_config": {"rank": 8, "seed": 5},
    }
    return client.post("/api/v1/create_model", headers=headers, json=body).json()["model_id"]


def submit(client: httpx.Client, headers: dict[str, str], model: str, seq: int) -> str:
    """Odd seqs are forward-backward passes and even seqs optimizer steps."""
    if seq % 2:
        response = client.post(
            "/api/v1/forward_backward", headers=headers, content=forward_backward(model, seq)
        )
    else:
        step = {"model_id": model, "seq_id": seq, "adam_params": ADAM.model_dump()}
        response = client.post("/api/v1/optim_step", headers=headers, json=step)
    assert response.status_code == 200, response.text
    return response.json()["request_id"]


def outcome(
    client: httpx.Client, headers: dict[str, str], request: str, timeout: float = 15
) -> object:
    """A finished task's observable result: logprobs and metrics, or the JSON body."""
    response = result(client, headers, request, timeout)
    assert response.status_code == 200, response.text
    if response.headers["content-type"] == "application/x-protobuf":
        output = deserialize_forward_backward_output(response.content)
        return [output.loss_fn_outputs[0]["logprobs"].tolist(), output.metrics]
    return response.json()


def test_takeover_returns_the_kept_result_instead_of_repeating_a_step(
    cluster: Cluster, tmp_path: Path
) -> None:
    """The owner dies after the trainer ran a step but before the outcome is
    committed. The next owner dispatches the same seq; the step must apply once."""
    alpha = {"x-api-key": cluster.key("alpha")}
    with httpx.Client(base_url=cluster.url, timeout=30) as client:
        session = client.post("/api/v1/create_session", headers=alpha).json()["session_id"]
        reference, model = create(client, alpha, session, 0), create(client, alpha, session, 1)
        expected = [
            outcome(client, alpha, submit(client, alpha, reference, seq)) for seq in range(1, 5)
        ]
        assert expected[1] != expected[3], "the second step must change the model"
        ran = [outcome(client, alpha, submit(client, alpha, model, seq)) for seq in (1, 2)]
        assert ran == expected[:2], "models with the same seed and inputs must agree"

        with ExitStack() as stack:
            cluster.psql_background(stack, "hold_model_lock", model=model)
            wait_for(
                lambda: cluster.psql("model_lock_held") == "t", "the model lock was never held"
            )
            backward = submit(client, alpha, model, 3)
            wait_for(
                lambda: cluster.psql("outcome_blocked") == "t",
                "the owner never ran seq 3 and waited to commit it",
            )
            cluster.server.kill()
            cluster.server.wait()
            # The dead owner's commit still waits on the lock; it must not land.
            cluster.psql("terminate_lock_waiters")
            cluster.psql("release_model_lock")

    with (
        ExitStack() as stack,
        (tmp_path / "takeover.log").open("wb") as log,
        httpx.Client(base_url=cluster.url, timeout=30) as client,
    ):
        spawn(stack, [SERVER], cluster.env, log)

        def serving() -> bool:
            try:
                return cluster.capabilities(alpha["x-api-key"]).status_code == 200
            except httpx.TransportError:
                return False

        wait_for(serving, "the new server never answered")
        # The lease lapses up to 10s after the dead owner's last renewal.
        assert outcome(client, alpha, backward, timeout=30) == expected[2]
        assert outcome(client, alpha, submit(client, alpha, model, 4)) == expected[3], (
            "the step that ran before the takeover applied twice or not at all"
        )


# Before the fix, the worker gave up on an engine call after 25s and reported the
# step failed while the engine still applied it; the server also gives up on a
# call after 30s and repeats it.
@pytest.mark.parametrize("mock_fault", ["delay:forward_backward:1:35"])
def test_engine_call_slower_than_the_rpc_timeouts_applies_once(cluster: Cluster) -> None:
    alpha = {"x-api-key": cluster.key("alpha")}
    with httpx.Client(base_url=cluster.url, timeout=30) as client:
        session = client.post("/api/v1/create_session", headers=alpha).json()["session_id"]
        slow, reference = create(client, alpha, session, 0), create(client, alpha, session, 1)
        # The slow model's forward-backward is the engine's first, so it is the
        # one held for 35s.
        requests = [submit(client, alpha, slow, seq) for seq in (1, 2)]
        ran = [outcome(client, alpha, request, timeout=60) for request in requests]
        expected = [outcome(client, alpha, submit(client, alpha, reference, seq)) for seq in (1, 2)]
        assert ran == expected, "a slow step must apply exactly once and report its result"


def sample(client: httpx.Client, headers: dict[str, str], sampling_session: str) -> httpx.Response:
    """Submits one seeded sample of `[2, 7, 11]` as the session's seq 0."""
    body = {
        "sampling_session_id": sampling_session,
        "seq_id": 0,
        "num_samples": 1,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [2, 7, 11]}]},
        "sampling_params": {
            "max_tokens": 4,
            "seed": 3,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
        },
    }
    return client.post("/api/v1/asample", headers=headers, json=body)


def sampled(client: httpx.Client, headers: dict[str, str], sampling_session: str) -> bytes:
    """The encoded result of `sample` on `sampling_session`."""
    future = sample(client, headers, sampling_session)
    assert future.status_code == 200, future.text
    response = result(client, headers, future.json()["request_id"])
    assert response.status_code == 200, response.text
    return response.content


async def trained(client: tinker.ServiceClient) -> tinker.TrainingClient:
    training = await client.create_lora_training_client(base_model="mock", rank=8, seed=5)
    backward = training.forward_backward([datum([7, 11, 4, 9])], "cross_entropy")
    step = training.optim_step(ADAM)
    await backward
    await step
    return training


def test_sampler_saves_sdk_and_isolation(cluster: Cluster) -> None:
    alpha = tinker.ServiceClient(base_url=cluster.url, api_key=cluster.key("alpha"))
    beta = tinker.ServiceClient(base_url=cluster.url, api_key=cluster.key("beta"))

    async def run() -> None:
        ours, theirs = await asyncio.gather(trained(alpha), trained(beta))
        # Names are per model, so another org's model may reuse one.
        saved, other = await asyncio.gather(
            ours.save_weights_for_sampler("step-1"),
            theirs.save_weights_for_sampler("step-1"),
        )
        assert saved.path == f"tinker://{ours.model_id}/sampler_weights/step-1"
        assert other.path == f"tinker://{theirs.model_id}/sampler_weights/step-1"
        # Errors are checked over HTTP: the SDK waits 30s to report each one
        # to a telemetry route Metrl does not serve.
        prompt = tinker.ModelInput.from_ints([2, 7, 11])
        params = tinker.SamplingParams(max_tokens=8, seed=3)

        async def tokens(sampler: tinker.SamplingClient) -> list[int]:
            return (await sampler.sample(prompt, 1, params)).sequences[0].tokens

        step_1, their_step_1, base = await asyncio.gather(
            tokens(await alpha.create_sampling_client(model_path=saved.path)),
            tokens(await beta.create_sampling_client(model_path=other.path)),
            tokens(await alpha.create_sampling_client(base_model="mock")),
        )
        assert step_1 != base, "sampling a checkpoint must apply its weights"
        assert their_step_1 == step_1, "models trained alike must sample alike in any org"
        unnamed = await ours.save_weights_and_get_sampling_client()
        assert await tokens(unnamed) == step_1, "the same weights must sample the same"
        await ours.forward_backward([datum([7, 11, 4, 9])], "cross_entropy")
        await ours.optim_step(ADAM)
        step_2 = await ours.save_weights_and_get_sampling_client()
        assert await tokens(step_2) != step_1, "a later checkpoint must sample its own weights"

    asyncio.run(run())


def test_sampler_save_routes(cluster: Cluster) -> None:
    alpha = {"x-api-key": cluster.key("alpha")}
    beta = {"x-api-key": cluster.key("beta")}
    with httpx.Client(base_url=cluster.url, timeout=30) as client:
        session = client.post("/api/v1/create_session", headers=alpha).json()["session_id"]
        model = create(client, alpha, session, 0)
        for seq in (1, 2):
            outcome(client, alpha, submit(client, alpha, model, seq))

        def save(headers: dict[str, str], seq: int, **fields: object) -> httpx.Response:
            body = {"model_id": model, "seq_id": seq} | fields
            return client.post("/api/v1/save_weights_for_sampler", headers=headers, json=body)

        for invalid in (
            {"path": "a", "sampling_session_seq_id": 0},
            {},
            # The name ends a tinker:// path.
            {"path": "a/b"},
            # Below the table's check, which would fail the commit forever.
            {"path": "a", "ttl_seconds": 60},
            {"path": "a", "kind": "inflight"},
        ):
            assert save(alpha, 3, **invalid).status_code == 400, invalid
        named = {"path": "step-2", "ttl_seconds": 3600, "user_metadata": {"step": "2"}}
        request = save(alpha, 3, **named).json()
        assert save(alpha, 3, **named).json() == request
        assert save(alpha, 3, path="other").status_code == 400
        assert save(beta, 3, **named).status_code == 404
        path = f"tinker://{model}/sampler_weights/step-2"
        assert outcome(client, alpha, request["request_id"]) == {
            "path": path,
            "type": "save_weights_for_sampler",
        }
        taken = outcome(client, alpha, save(alpha, 4, path="step-2").json()["request_id"])
        assert isinstance(taken, dict) and taken["category"] == "user", taken
        # The failed save must not stall the model's later tasks.
        unnamed = save(alpha, 5, sampling_session_seq_id=0).json()["request_id"]
        assert outcome(client, alpha, unnamed) == {
            "sampling_session_id": unnamed,
            "type": "save_weights_for_sampler",
        }

        def sampler(headers: dict[str, str], seq: int, model_path: str) -> httpx.Response:
            sampler_session = sessions[headers["x-api-key"]]
            body = {
                "session_id": sampler_session,
                "sampling_session_seq_id": seq,
                "model_path": model_path,
            }
            return client.post("/api/v1/create_sampling_session", headers=headers, json=body)

        sessions = {
            headers["x-api-key"]: client.post("/api/v1/create_session", headers=headers).json()[
                "session_id"
            ]
            for headers in (alpha, beta)
        }
        named_sampler = sampler(alpha, 0, path).json()
        assert sampler(alpha, 0, path).json() == named_sampler
        assert sampler(beta, 0, path).status_code == 404
        assert sampler(alpha, 1, f"tinker://{model}/weights/step-2").status_code == 400
        assert sampler(alpha, 2, f"tinker://{model}/sampler_weights/missing").status_code == 404

        assert sample(client, beta, unnamed).status_code == 404

        def sampler_record(headers: dict[str, str], sampler: str) -> httpx.Response:
            return client.get(f"/api/v1/samplers/{sampler}", headers=headers)

        named_session = named_sampler["sampling_session_id"]
        assert sampler_record(alpha, named_session).json() == {
            "sampler_id": named_session,
            "base_model": "mock",
            "model_path": path,
        }
        assert sampler_record(alpha, unnamed).json() == {
            "sampler_id": unnamed,
            "base_model": "mock",
            "model_path": None,
        }
        assert sampler_record(beta, named_session).status_code == 404
        assert sampler_record(beta, unnamed).status_code == 404
        # Both saves hold the weights after seq 2, the last step before them.
        samples = [
            sampled(client, alpha, session)
            for session in (named_sampler["sampling_session_id"], unnamed)
        ]
        assert samples[0] == samples[1]


PROMPT = [2, 7, 11]


def rl_datum(completion: list[int], logprobs: list[float], advantage: float) -> tinker.Datum:
    """An `importance_sampling` datum that trains on the completion only."""
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
    """Three rounds of sample, reward, and train with `importance_sampling`,
    rewarding tokens of parity `wanted`; returns every observable result."""
    training = await client.create_lora_training_client(base_model="mock", rank=8, seed=5)
    trace: list[Any] = []
    for _ in range(3):
        sampler = await training.save_weights_and_get_sampling_client()
        sampled = await sampler.sample(
            tinker.ModelInput.from_ints(PROMPT), 4, tinker.SamplingParams(max_tokens=6, seed=1)
        )
        completions = [(s.tokens, s.logprobs or []) for s in sampled.sequences]
        rewards = [
            sum(token % 2 == wanted for token in tokens) / len(tokens) for tokens, _ in completions
        ]
        data = [rl_datum(*c, r) for c, r in zip(completions, rewards, strict=True)]
        backward = training.forward_backward(data, "importance_sampling")
        step = training.optim_step(ADAM)
        output, applied = await backward, await step
        for (_, sampled_logprobs), out in zip(completions, output.loss_fn_outputs, strict=True):
            learner = out["logprobs"].tolist()[len(PROMPT) - 1 :]
            assert learner == pytest.approx(sampled_logprobs, abs=1e-6), (
                "the trainer must score sampled tokens as the sampler drew them"
            )
        # On-policy the ratio is 1, so the loss is minus the summed advantages.
        expected = -sum(r * len(t) for (t, _), r in zip(completions, rewards, strict=True))
        assert output.metrics["loss:sum"] == pytest.approx(expected, rel=1e-5)
        trace += [[tokens for tokens, _ in completions], output.metrics, applied.metrics]
    # Halving the sampler's probabilities doubles every ratio; `forward`
    # scores both at the same weights and leaves them as they were.
    halved = [
        rl_datum(tokens, [lp - math.log(2) for lp in logprobs], r)
        for (tokens, logprobs), r in zip(completions, rewards, strict=True)
    ]
    same = await training.forward(data, "importance_sampling")
    off_policy = await training.forward(halved, "importance_sampling")
    assert off_policy.metrics["loss:sum"] == pytest.approx(2 * same.metrics["loss:sum"], rel=1e-5)
    return trace


def test_importance_sampling_loop(cluster: Cluster) -> None:
    alpha = tinker.ServiceClient(base_url=cluster.url, api_key=cluster.key("alpha"))
    beta = tinker.ServiceClient(base_url=cluster.url, api_key=cluster.key("beta"))

    async def run() -> None:
        interleaved = await asyncio.gather(rl(alpha, 1), rl(beta, 0))
        alone = [await rl(alpha, 1), await rl(beta, 0)]
        assert interleaved == alone, "one org's RL loop changed another org's"
        ours, theirs = interleaved
        assert ours[0] == theirs[0], "both orgs start from the same weights"
        assert ours[2] != theirs[2], "different advantages must train different weights"
        assert ours[3] != ours[0], "each round must sample the weights the last one trained"

    asyncio.run(run())


def test_server_whose_view_lags_samples_a_checkpoint_saved_through_it(
    cluster: Cluster, tmp_path: Path
) -> None:
    """A second server applies no commit after its startup scan, standing for one
    whose CDC lags. It answers a save from Postgres once the model's owner ran
    it; sampling that checkpoint there right away must not return 404, which
    the SDK does not retry."""
    alpha = {"x-api-key": cluster.key("alpha")}
    http, grpc, system = free_ports(3)
    lagging_env = cluster.env | {
        "METRL_HTTP_ADDR": f"127.0.0.1:{http}",
        "METRL_GRPC_ADDR": f"127.0.0.1:{grpc}",
        "METRL_SYSTEM_ADDR": f"127.0.0.1:{system}",
        "FAILPOINTS": "cdc_skip_commits=return",
    }
    with ExitStack() as stack:
        owner = stack.enter_context(httpx.Client(base_url=cluster.url, timeout=30))
        session = owner.post("/api/v1/create_session", headers=alpha).json()["session_id"]
        model = create(owner, alpha, session, 0)
        # A finished task means the first server holds the model's lease, so the
        # lagging server never runs it.
        outcome(owner, alpha, submit(owner, alpha, model, 1))
        spawn(stack, [SERVER], lagging_env, stack.enter_context((tmp_path / "lag.log").open("wb")))
        lagging = stack.enter_context(httpx.Client(base_url=f"http://127.0.0.1:{http}", timeout=30))

        def session_on(client: httpx.Client, headers: dict[str, str]) -> httpx.Response:
            return client.post("/api/v1/create_session", headers=headers)

        def serving() -> bool:
            try:
                return session_on(lagging, alpha).status_code == 200
            except httpx.TransportError:
                return False

        wait_for(serving, "the lagging server never answered")
        late = {"x-api-key": cluster.key("late")}
        assert session_on(lagging, late).status_code == 401, (
            "the lagging server applied a commit; build it with --features failpoints"
        )

        def save(seq: int, **fields: object) -> str:
            body = {"model_id": model, "seq_id": seq} | fields
            response = lagging.post("/api/v1/save_weights_for_sampler", headers=alpha, json=body)
            assert response.status_code == 200, response.text
            return response.json()["request_id"]

        path = f"tinker://{model}/sampler_weights/fresh"
        named = save(2, path="fresh")
        assert outcome(lagging, alpha, named, timeout=30) == {
            "path": path,
            "type": "save_weights_for_sampler",
        }
        unnamed = save(3, sampling_session_seq_id=0)
        assert outcome(lagging, alpha, unnamed, timeout=30) == {
            "sampling_session_id": unnamed,
            "type": "save_weights_for_sampler",
        }

        def samples(client: httpx.Client) -> list[bytes]:
            body = {
                "session_id": session_on(client, alpha).json()["session_id"],
                "sampling_session_seq_id": 0,
                "model_path": path,
            }
            created = client.post("/api/v1/create_sampling_session", headers=alpha, json=body)
            assert created.status_code == 200, created.text
            session = created.json()["sampling_session_id"]
            return [sampled(client, alpha, session), sampled(client, alpha, unnamed)]

        assert samples(lagging) == samples(owner), "the lagging server sampled other weights"
