"""Shared E2E fixtures: real `metrl-server` and `metrl-worker` processes on a
fresh Postgres database per test."""

import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import httpx
import pytest

SERVER = Path("target/debug/metrl-server")
WORKER = Path(sys.executable).with_name("metrl-worker")
SQL = Path(__file__).with_name("sql")
ADMIN_DATABASE = "postgres"


def psql_command(database: str, **variables: str) -> list[str]:
    command = ["docker", "compose", "exec", "-T", "postgres", "psql", "-U", "metrl", "-d", database]
    options = ["-X", "-q", "-At", "-v", "ON_ERROR_STOP=1"]
    return [*command, *options, *(f"--set={name}={value}" for name, value in variables.items())]


def psql(script: str, database: str, **variables: str) -> str:
    """Runs `tests/sql/<script>.sql` in the compose Postgres and returns its output."""
    return subprocess.run(
        psql_command(database, **variables),
        input=(SQL / f"{script}.sql").read_text(),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def create_key(env: dict[str, str], org: str) -> str:
    """Creates an API key for `org` through the server CLI."""
    return subprocess.run(
        [SERVER, "key", "create", "--org", org],
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


@dataclass(frozen=True, slots=True)
class Cluster:
    url: str
    database: str
    server: subprocess.Popen[bytes]
    worker: subprocess.Popen[bytes]
    env: dict[str, str]

    def psql(self, script: str, **variables: str) -> str:
        return psql(script, self.database, **variables)

    def psql_background(self, stack: ExitStack, script: str, **variables: str) -> None:
        """Starts `tests/sql/<script>.sql` without waiting for it; `stack` stops
        the client, and the test database's drop ends its backend."""
        process = subprocess.Popen(
            psql_command(self.database, **variables),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        stack.callback(process.wait)
        stack.callback(process.kill)
        assert process.stdin is not None, "stdin is a pipe"
        process.stdin.write((SQL / f"{script}.sql").read_text())
        process.stdin.close()

    def capabilities(self, key: str) -> httpx.Response:
        return httpx.get(
            f"{self.url}/api/v1/get_server_capabilities", headers={"x-api-key": key}, timeout=2
        )

    def key(self, org: str) -> str:
        """Creates an API key for `org` and waits until CDC brings it to the server;
        the SDK does not retry a 401."""
        key = create_key(self.env, org)
        wait_for(
            lambda: self.capabilities(key).status_code == 200,
            f"the key for {org} never reached the server",
        )
        return key


def free_ports(count: int) -> list[int]:
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(count)]
        for connection in sockets:
            connection.bind(("127.0.0.1", 0))
        return [connection.getsockname()[1] for connection in sockets]


def database_url(name: str) -> str:
    return f"postgres://metrl@127.0.0.1:5432/{name}"


def spawn(
    stack: ExitStack, command: list[Path], env: dict[str, str], log: BinaryIO
) -> subprocess.Popen[bytes]:
    """Starts a process that is stopped when `stack` closes, even on failure."""
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)

    def stop() -> None:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    stack.callback(stop)
    return process


def wait_for(condition: Callable[[], bool], message: str, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(message)
        time.sleep(0.05)


@pytest.fixture(scope="session")
def template() -> Iterator[str]:
    """A database with the schema applied, cloned for each test."""
    name = f"metrl_template_{uuid.uuid4().hex[:12]}"
    psql("create_database", ADMIN_DATABASE, name=name)
    try:
        subprocess.run(
            [SERVER, "migrate"],
            env=os.environ | {"METRL_DATABASE_URL": database_url(name)},
            check=True,
        )
        yield name
    finally:
        psql("drop_database", ADMIN_DATABASE, name=name)


@pytest.fixture
def mock_fault() -> str:
    """`METRL_MOCK_FAULT` for the worker; override with `pytest.mark.parametrize`."""
    return ""


@pytest.fixture
def cluster(template: str, tmp_path: Path, mock_fault: str) -> Iterator[Cluster]:
    name = f"metrl_test_{uuid.uuid4().hex[:12]}"
    http, grpc, system, worker_grpc = free_ports(4)
    # The compose object store, under a prefix of the test's own.
    objects = {
        "METRL_OBJECT__URL": f"s3://metrl/tests/{name}",
        "METRL_OBJECT__SECRET_ACCESS_KEY": "metrl-dev-secret",
    }
    env = (
        os.environ
        | objects
        | {
            "METRL_DATABASE_URL": database_url(name),
            "METRL_HTTP_ADDR": f"127.0.0.1:{http}",
            "METRL_GRPC_ADDR": f"127.0.0.1:{grpc}",
            "METRL_SYSTEM_ADDR": f"127.0.0.1:{system}",
        }
    )
    worker_env = (
        os.environ
        | objects
        | {
            "METRL_GRPC_ADDR": f"127.0.0.1:{worker_grpc}",
            "METRL_SERVERS": f"http://127.0.0.1:{grpc}",
            "METRL_MOCK_FAULT": mock_fault,
        }
    )
    server_log, worker_log = tmp_path / "server.log", tmp_path / "worker.log"
    psql("clone_database", ADMIN_DATABASE, name=name, template=template)
    with ExitStack() as stack:
        stack.callback(psql, "drop_database", ADMIN_DATABASE, name=name)
        # Created before the server starts, so only its startup scan can load it.
        probe = create_key(env, "probe")
        server = spawn(stack, [SERVER], env, stack.enter_context(server_log.open("wb")))
        worker = spawn(stack, [WORKER], worker_env, stack.enter_context(worker_log.open("wb")))
        cluster = Cluster(
            url=f"http://127.0.0.1:{http}", database=name, server=server, worker=worker, env=env
        )

        def registered() -> bool:
            for process, log in ((server, server_log), (worker, worker_log)):
                assert process.poll() is None, log.read_text()
            try:
                response = cluster.capabilities(probe)
            except httpx.TransportError:
                return False
            return response.status_code == 200 and bool(response.json()["supported_models"])

        wait_for(registered, "the worker never appeared in get_server_capabilities")
        yield cluster
