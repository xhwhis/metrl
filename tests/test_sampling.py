import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import tinker


@dataclass(frozen=True, slots=True)
class Runtime:
    url: str
    worker: subprocess.Popen[bytes]


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[Runtime]:
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(3)]
        for connection in sockets:
            connection.bind(("127.0.0.1", 0))
        ports = [connection.getsockname()[1] for connection in sockets]
    url = f"http://127.0.0.1:{ports[0]}"
    env = os.environ | {
        "METRL_HTTP_ADDR": f"127.0.0.1:{ports[0]}",
        "METRL_SYSTEM_ADDR": f"127.0.0.1:{ports[1]}",
        "METRL_WORKER_ENDPOINT": f"http://127.0.0.1:{ports[2]}",
        "METRL_AUTH_KEYS": "alpha:tml-alpha,beta:tml-beta",
    }
    with (
        (tmp_path / "worker.log").open("wb") as worker_log,
        (tmp_path / "server.log").open("wb") as server_log,
        subprocess.Popen(
            [
                str(Path(sys.executable).with_name("metrl-worker")),
                "--addr",
                f"127.0.0.1:{ports[2]}",
            ],
            stdout=worker_log,
            stderr=subprocess.STDOUT,
        ) as worker,
        subprocess.Popen(
            ["target/debug/metrl-server"], env=env, stdout=server_log, stderr=subprocess.STDOUT
        ) as server,
    ):
        try:
            with httpx.Client(
                base_url=url, timeout=1, headers={"x-api-key": "tml-alpha"}
            ) as client:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    assert server.poll() is None, (tmp_path / "server.log").read_text()
                    assert worker.poll() is None, (tmp_path / "worker.log").read_text()
                    try:
                        if client.get("/api/v1/get_server_capabilities").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.05)
                else:
                    pytest.fail("server and worker did not become ready")
            yield Runtime(url=url, worker=worker)
        finally:
            server.terminate()
            if worker.poll() is None:
                worker.terminate()


def test_sampling_sdk_and_isolation(runtime: Runtime) -> None:
    async def run() -> None:
        alpha = tinker.ServiceClient(base_url=runtime.url, api_key="tml-alpha")
        beta = tinker.ServiceClient(base_url=runtime.url, api_key="tml-beta")
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
    with httpx.Client(base_url=runtime.url) as client:
        alpha = {"x-api-key": "tml-alpha"}
        beta = {"x-api-key": "tml-beta"}
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
        runtime.worker.terminate()
        runtime.worker.wait(timeout=5)
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
