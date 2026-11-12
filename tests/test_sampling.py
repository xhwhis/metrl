import asyncio
import time

import httpx
import pytest
import tinker
from conftest import Cluster


def test_sampling_sdk_and_isolation(cluster: Cluster) -> None:
    alpha_key, beta_key = cluster.key("alpha"), cluster.key("beta")

    async def run() -> None:
        alpha = tinker.ServiceClient(base_url=cluster.url, api_key=alpha_key)
        beta = tinker.ServiceClient(base_url=cluster.url, api_key=beta_key)
        a = await alpha.create_sampling_client(base_model="mock")
        b = await beta.create_sampling_client(base_model="mock")
        prompt = tinker.ModelInput.from_ints([2, 7, 11])
        params = tinker.SamplingParams(max_tokens=8, seed=3)
        first, other = await asyncio.gather(
            a.sample(prompt, 2, params),
            b.sample(prompt, 2, params),
        )
        repeated = await a.sample(prompt, 2, params)
        reseeded = await a.sample(prompt, 2, tinker.SamplingParams(max_tokens=8, seed=4))
        tokens = [s.tokens for s in first.sequences]
        assert [s.tokens for s in repeated.sequences] == tokens, "a seed must reproduce samples"
        assert [s.tokens for s in other.sequences] == tokens, "an interleaved org changed them"
        assert [s.tokens for s in reseeded.sequences] != tokens
        assert tokens[0] != tokens[1], "each sample must draw its own tokens"
        for sequence in first.sequences:
            assert sequence.logprobs is not None
            assert len(sequence.tokens) == len(sequence.logprobs) == 8
            assert all(logprob < 0 for logprob in sequence.logprobs)
            assert sequence.stop_reason == "length"
        assert len({s.sequence_id for s in (*first.sequences, *repeated.sequences)}) == 4

        greedy, top1, nucleus = await asyncio.gather(
            a.sample(prompt, 1, tinker.SamplingParams(max_tokens=8, temperature=0)),
            a.sample(prompt, 1, tinker.SamplingParams(max_tokens=8, seed=5, top_k=1)),
            a.sample(prompt, 1, tinker.SamplingParams(max_tokens=8, seed=6, top_p=0.01)),
        )
        best = greedy.sequences[0].tokens
        assert top1.sequences[0].tokens == best, "top_k=1 must pick the most likely token"
        assert nucleus.sequences[0].tokens == best, "a tiny top_p must keep only the top token"
        assert greedy.sequences[0].logprobs == [0.0] * 8, "a greedy draw is certain"
        stop = best[3]
        stopped = await a.sample(
            prompt, 1, tinker.SamplingParams(max_tokens=8, temperature=0, stop=[stop])
        )
        assert stopped.sequences[0].tokens == best[: best.index(stop) + 1]
        assert stopped.sequences[0].stop_reason == "stop"

        # The sampled tokens' logprobs under the default distribution are what
        # scoring the same tokens gives.
        drawn = (await a.sample(prompt, 1, tinker.SamplingParams(max_tokens=8))).sequences[0]
        scored = await a.compute_logprobs(tinker.ModelInput.from_ints([2, 7, 11, *drawn.tokens]))
        assert scored[0] is None
        assert scored[3:] == pytest.approx(drawn.logprobs, abs=1e-5)

    asyncio.run(run())
    with httpx.Client(base_url=cluster.url) as client:
        alpha = {"x-api-key": alpha_key}
        beta = {"x-api-key": beta_key}
        assert client.post("/api/v1/create_session").status_code == 401
        session = client.post("/api/v1/create_session", headers=alpha).json()["session_id"]
        body = {"session_id": session, "sampling_session_seq_id": 0, "base_model": "mock"}
        assert (
            client.post("/api/v1/create_sampling_session", headers=beta, json=body).status_code
            == 404
        )
        model = client.post("/api/v1/create_sampling_session", headers=alpha, json=body).json()
        assert (
            client.post("/api/v1/create_sampling_session", headers=alpha, json=body).json() == model
        )
        assert (
            client.post(
                "/api/v1/create_sampling_session",
                headers=alpha,
                json=body | {"base_model": "missing"},
            ).status_code
            == 400
        )
        sample = {
            "sampling_session_id": model["sampling_session_id"],
            "seq_id": 3,
            "num_samples": 2,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [2, 7, 11]}]},
            "sampling_params": {
                "max_tokens": 4,
                "seed": 3,
                "temperature": 1,
                "top_k": -1,
                "top_p": 1,
            },
        }

        def outcome(future: dict[str, str]) -> object:
            request = {"request_id": future["request_id"]}
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                response = client.post("/api/v1/retrieve_future", headers=alpha, json=request)
                if response.status_code != 408:
                    return response.json()
                time.sleep(0.05)
            pytest.fail("the request stayed pending")

        # Checked over HTTP: the SDK waits 30s to report a failed future to a
        # telemetry route Metrl does not serve.
        params = sample["sampling_params"] | {"stop": "\n"}
        unsupported = sample | {"seq_id": 5, "sampling_params": params}
        failed = client.post("/api/v1/asample", headers=alpha, json=unsupported).json()
        assert outcome(failed) == {
            "error": "the mock engine has no tokenizer; stop must be token ids",
            "category": "user",
        }
        cluster.worker.terminate()
        cluster.worker.wait(timeout=5)
        assert (
            client.post("/api/v1/create_sampling_session", headers=alpha, json=body).json() == model
        )
        assert client.post("/api/v1/asample", headers=beta, json=sample).status_code == 404
        future = client.post("/api/v1/asample", headers=alpha, json=sample).json()
        assert client.post("/api/v1/asample", headers=alpha, json=sample).json() == future
        assert (
            client.post(
                "/api/v1/asample", headers=alpha, json=sample | {"num_samples": 1}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/v1/retrieve_future", headers=beta, json={"request_id": future["request_id"]}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/v1/asample", headers=alpha, json=sample | {"num_samples": 0}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/v1/asample", headers=alpha, json=sample | {"num_samples": -1}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/v1/asample", headers=alpha, json=sample | {"sampling_session_id": "invalid"}
            ).status_code
            == 400
        )
        failed = client.post("/api/v1/asample", headers=alpha, json=sample | {"seq_id": 4}).json()
        assert outcome(failed) == {"error": "worker sampling failed", "category": "server"}
