import asyncio
from collections import OrderedDict
from collections.abc import Callable, Coroutine

from metrl._core import Adapter


class AdapterCache[T]:
    """Loaded adapters by `Adapter.key`, at most `size`, least recently used
    evicted first. Requests that arrive while an adapter loads wait for that
    load, since many samples of a new adapter arrive at once after a save. A
    failed load is dropped, so the next request retries it."""

    def __init__(self, size: int) -> None:
        self._size = size
        # Holding the task rather than its result lets waiting requests share it.
        self._loads: OrderedDict[str, asyncio.Task[T]] = OrderedDict()

    async def get(self, adapter: Adapter, load: Callable[[], Coroutine[object, object, T]]) -> T:
        """`adapter` as `load` returns it, loading it only on a miss."""
        key = adapter.key
        loading = self._loads.get(key)
        if loading is None:
            loading = asyncio.create_task(load())
            loading.add_done_callback(lambda done: self._drop_failed(key, done))
            self._loads[key] = loading
            if len(self._loads) > self._size:
                self._loads.popitem(last=False)
        self._loads.move_to_end(key)
        # A cancelled request must not cancel the load other requests share.
        return await asyncio.shield(loading)

    def _drop_failed(self, key: str, done: asyncio.Task[T]) -> None:
        failed = done.cancelled() or done.exception() is not None
        if failed and self._loads.get(key) is done:
            del self._loads[key]
