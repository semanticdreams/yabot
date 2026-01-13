from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


async def retry_until_ok(
    fn: Callable[[], Awaitable[None]],
    delay_seconds: float,
    cancel_event: asyncio.Event | None = None,
) -> None:
    while True:
        try:
            await fn()
            return
        except Exception:
            if cancel_event is not None and cancel_event.is_set():
                return
            await asyncio.sleep(delay_seconds)
