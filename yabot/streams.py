import asyncio
from typing import Any


class StreamRegistry:
    def __init__(self) -> None:
        self._active_streams: dict[str, asyncio.Task[dict[str, Any]]] = {}

    def register(self, room_id: str, task: asyncio.Task[dict[str, Any]]) -> None:
        assert room_id, "room_id is required"
        assert isinstance(task, asyncio.Task), "task must be asyncio.Task"
        self._active_streams[room_id] = task

    def clear(self, room_id: str, task: asyncio.Task[dict[str, Any]]) -> None:
        assert room_id, "room_id is required"
        assert isinstance(task, asyncio.Task), "task must be asyncio.Task"
        if self._active_streams.get(room_id) is task:
            self._active_streams.pop(room_id, None)

    def stop(self, room_id: str) -> bool:
        assert room_id, "room_id is required"
        task = self._active_streams.get(room_id)
        if not task or task.done():
            return False
        task.cancel()
        return True
