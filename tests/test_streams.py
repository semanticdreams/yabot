import asyncio

import pytest

from yabot.streams import StreamRegistry


@pytest.mark.asyncio
async def test_stop_cancels_active_task():
    registry = StreamRegistry()
    task = asyncio.create_task(asyncio.sleep(10))
    registry.register("room1", task)

    assert registry.stop("room1") is True

    await asyncio.sleep(0)
    assert task.cancelled()
