from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable

from corridor_aiops.schema import CorridorEvent

EventHandler = Callable[[CorridorEvent], Awaitable[None]]
_END_OF_STREAM = object()


async def run_pipeline(
    source: AsyncIterable[CorridorEvent],
    handler: EventHandler,
    *,
    max_queue_size: int = 1024,
) -> None:
    """Apply backpressure between an asynchronous producer and a single consumer.

    The producer and consumer run as sibling tasks that are monitored together.
    If either fails, the other is cancelled immediately and the original
    exception propagates. A consumer failure therefore cannot leave the
    producer blocked on a full bounded queue.
    """
    if max_queue_size < 1:
        raise ValueError("max_queue_size must be positive")
    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=max_queue_size)

    async def produce() -> None:
        async for event in source:
            await queue.put(event)
        await queue.put(_END_OF_STREAM)

    async def consume() -> None:
        while True:
            event = await queue.get()
            try:
                if event is _END_OF_STREAM:
                    return
                await handler(event)  # type: ignore[arg-type]
            finally:
                queue.task_done()

    producer = asyncio.create_task(produce(), name="corridor-pipeline-producer")
    consumer = asyncio.create_task(consume(), name="corridor-pipeline-consumer")
    tasks = (producer, consumer)
    try:
        pending: set[asyncio.Task[None]] = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
