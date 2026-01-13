import asyncio
from typing import Any, Awaitable, Callable

from .commands import parse_command
from .streams import StreamRegistry


def is_stop_command(text: str) -> bool:
    parsed = parse_command(text)
    return bool(parsed and parsed[0] == "stop")


async def request_stop(graph: Any, streams: StreamRegistry, room_id: str) -> bool:
    stopper = getattr(graph, "stop", None)
    if callable(stopper):
        return await stopper(room_id)
    return streams.stop(room_id)


async def dispatch_graph(
    graph: Any,
    streams: StreamRegistry,
    room_id: str,
    text: str,
    on_start: Callable[[], Awaitable[None]] | None = None,
    on_done: Callable[[], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    if on_start:
        await on_start()
    task = asyncio.create_task(graph.ainvoke(room_id, text))
    streams.register(room_id, task)
    try:
        return await task
    finally:
        streams.clear(room_id, task)
        if on_done:
            await on_done()
