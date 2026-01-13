from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import websockets


class RemoteGraphClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._send_lock = asyncio.Lock()
        self._receiver_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def connect(self) -> None:
        if self._ws is None or self._is_closed(self._ws):
            self._ws = await websockets.connect(self.url)
            self._receiver_task = asyncio.create_task(self._receiver_loop())

    async def close(self) -> None:
        if self._ws is not None and not self._is_closed(self._ws):
            await self._ws.close()
        if self._receiver_task is not None:
            self._receiver_task.cancel()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("Connection closed."))
        self._pending.clear()
        self._ws = None
        self._receiver_task = None

    @staticmethod
    def _is_closed(ws: Any) -> bool:
        closed = getattr(ws, "closed", None)
        if closed is not None:
            return bool(closed)
        close_code = getattr(ws, "close_code", None)
        if close_code is not None:
            return True
        state = getattr(ws, "state", None)
        if state is not None:
            name = getattr(state, "name", "")
            return name == "CLOSED"
        return False

    async def ainvoke(self, room_id: str, text: str) -> dict[str, Any]:
        await self.connect()
        assert self._ws is not None
        request_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = {"type": "message", "id": request_id, "room_id": room_id, "text": text}
        async with self._send_lock:
            await self._ws.send(json.dumps(payload))
        return await future

    async def stop(self, room_id: str) -> bool:
        await self.connect()
        assert self._ws is not None
        request_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = {"type": "stop", "id": request_id, "room_id": room_id}
        async with self._send_lock:
            await self._ws.send(json.dumps(payload))
        result = await future
        return bool(result.get("ok"))

    async def _receiver_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                request_id = message.get("id")
                if not request_id or request_id not in self._pending:
                    continue
                future = self._pending.pop(request_id)
                msg_type = message.get("type")
                if msg_type == "response":
                    future.set_result(message.get("result", {}))
                elif msg_type == "stopped":
                    future.set_result(message)
                elif msg_type == "cancelled":
                    future.set_exception(RuntimeError("Request cancelled."))
                elif msg_type == "error":
                    future.set_exception(RuntimeError(message.get("error", "Unknown error")))
                else:
                    future.set_exception(RuntimeError("Unknown response type"))
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("Connection closed."))
            self._pending.clear()
