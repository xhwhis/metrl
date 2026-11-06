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
        a = await alpha.create_sampling_client_async(base_model="mock")
        b = await beta.create_sampling_client_async(base_model="mock")
        params = tinker.SamplingParams(max_tokens=4, seed=3)
        first, other = await asyncio.gather(
            a.sample_async(tinker.ModelInput.from_ints([2, 7, 11]), 2, params),
            b.sample_async(tinker.ModelInput.from_ints([9, 1]), 1, params),
        )
        repeated = await a.sample_async(tinker.ModelInput.from_ints([2, 7, 11]), 2, params)
        changed = await b.sample_async(
            tinker.ModelInput.from_ints([9, 1]), 1, tinker.SamplingParams(max_tokens=2, seed=8)
        )
        assert [s.tokens for s in first.sequences] == [[23, 24, 25, 26], [40, 41, 42, 43]]
        assert [s.tokens for s in other.sequences] == [[13, 14, 15, 16]]
        assert [s.tokens for s in repeated.sequences] == [s.tokens for s in first.sequences]
        assert [s.tokens for s in changed.sequences] == [[18, 19]]
        assert all(s.stop_reason == "length" for s in first.sequences)
        assert len({s.sequence_id for s in (*first.sequences, *repeated.sequences)}) == 4

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
        cluster.worker.terminate()
        cluster.worker.wait(timeout=5)
        assert (
            client.post("/api/v1/create_sampling_session", headers=alpha, json=body).json() == model
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
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            response = client.post(
                "/api/v1/retrieve_future", headers=alpha, json={"request_id": failed["request_id"]}
            )
            if response.status_code != 408:
                assert response.json()["error"] == "worker sampling failed"
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker failure left the request pending")
