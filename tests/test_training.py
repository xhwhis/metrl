import asyncio
import itertools
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
import pytest
import tinker
from conftest import SERVER, Cluster, spawn, wait_for
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
    training = await client.create_lora_training_client_async(base_model="mock", rank=8, seed=5)
    trace: list[Any] = []
    for _ in range(2):
        backward = await training.forward_backward_async([datum(targets)], "cross_entropy")
        step = await training.optim_step_async(ADAM)
        output, applied = await backward, await step
        trace += [output.loss_fn_outputs[0]["logprobs"].tolist(), output.metrics, applied.metrics]
    forward = await (await training.forward_async([datum(targets)], "cross_entropy"))
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
    training = alpha.create_lora_training_client(base_model="mock", rank=4)
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
