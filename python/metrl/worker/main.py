import argparse
import importlib
import os
import socket

import uvloop
from metrl import _core


def main() -> None:
    parser = argparse.ArgumentParser(prog="metrl-worker", description="Metrl worker")
    parser.add_argument("--addr", default=os.environ.get("METRL_GRPC_ADDR", "0.0.0.0:7590"))
    parser.add_argument("--engine", default=os.environ.get("METRL_ENGINE", "mock"))
    args = parser.parse_args()
    importlib.import_module(f"metrl.engines.{args.engine}")
    uvloop.run(_core.serve(socket.gethostname(), args.addr))
