import argparse
import asyncio
import contextlib
import importlib
import os
import signal
import uuid
from typing import cast

import uvloop
from metrl import _core
from metrl.worker import ComputeThread, Load


async def run(config: _core.Config) -> None:
    load = cast(Load, importlib.import_module(f"metrl.engines.{config.engine}").load)
    engine = await load(config.engine_settings(), config.bases, ComputeThread())
    task = asyncio.create_task(
        _core.serve(str(uuid.uuid7()), config, engine.bases, engine.sampler, engine.trainer)
    )
    loop = asyncio.get_running_loop()
    signals = (signal.SIGINT, signal.SIGTERM)

    # Cancelling the task makes the Rust runtime unregister and drain. A second
    # signal gets the default action and ends the process at once.
    def stop() -> None:
        for signum in signals:
            loop.remove_signal_handler(signum)
        task.cancel()

    for signum in signals:
        loop.add_signal_handler(signum, stop)
    with contextlib.suppress(asyncio.CancelledError):
        await task


def main() -> None:
    parser = argparse.ArgumentParser(prog="metrl-worker", description="Metrl worker")
    parser.add_argument("--config", default=os.environ.get("METRL_CONFIG", "config/worker.toml"))
    args = parser.parse_args()
    uvloop.run(run(_core.Config.load(args.config)))
