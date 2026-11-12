import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class ComputeThread:
    """The worker's one compute thread. Engines run every blocking call on it,
    so the event loop keeps serving RPCs, and calls run one at a time in the
    order they were submitted. An awaiting coroutine that is cancelled leaves
    its call running to the end."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metrl-compute")

    async def run[T](self, fn: Callable[[], T]) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn)
