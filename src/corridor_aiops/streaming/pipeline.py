from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable

from corridor_aiops.schema import CorridorEvent

EventHandler = Callable[[CorridorEvent], Awaitable[None]]


async def run_pipeline(
    source: AsyncIterable[CorridorEvent],
    handler: EventHandler,
    *,
    max_queue_size: int = 1024,
) -> None:
    """Apply backpressure between asynchronous producers and a single consumer."""
    if max_queue_size < 1:
        raise ValueError("max_queue_size must be positive")
    queue: asyncio.Queue[CorridorEvent | None] = asyncio.Queue(maxsize=max_queue_size)

    async def consume() -> None:
        while True:
            event = await queue.get()
            try:
                if event is None:
                    return
                await handler(event)
            finally:
                queue.task_done()

    consumer = asyncio.create_task(consume())
    try:
        async for event in source:
            await queue.put(event)
        await queue.put(None)
        await consumer
    except BaseException:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        raise
