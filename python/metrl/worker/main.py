import argparse
import importlib
import os
import socket
from typing import cast

import uvloop
from metrl import _core
from metrl.worker import Sampler


def main() -> None:
    parser = argparse.ArgumentParser(prog="metrl-worker", description="Metrl worker")
    parser.add_argument("--addr", default=os.environ.get("METRL_GRPC_ADDR", "0.0.0.0:7590"))
    parser.add_argument(
        "--engine", choices=["mock"], default=os.environ.get("METRL_ENGINE", "mock")
    )
    parser.add_argument("--base-model", default="mock")
    args = parser.parse_args()
    engine = cast(Sampler, importlib.import_module(f"metrl.engines.{args.engine}").Sampler())
    uvloop.run(_core.serve(socket.gethostname(), args.base_model, args.addr, engine))
