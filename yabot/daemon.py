from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from .config import load_config
from .interaction import dispatch_graph
from .runtime import build_graph
from .streams import StreamRegistry


class YabotDaemon:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.streams = StreamRegistry()
        self.logger = logging.getLogger("yabot.daemon")

    async def handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        tasks: set[asyncio.Task[None]] = set()

        try:
            async for raw in websocket:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                    continue

                msg_type = payload.get("type")
                if msg_type == "message":
                    task = asyncio.create_task(self._handle_message(websocket, payload))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                    continue

                if msg_type == "stop":
                    room_id = str(payload.get("room_id", ""))
                    request_id = payload.get("id")
                    ok = self.streams.stop(room_id)
                    await websocket.send(
                        json.dumps({"type": "stopped", "id": request_id, "room_id": room_id, "ok": ok})
                    )
                    continue

                await websocket.send(json.dumps({"type": "error", "error": "Unknown message type"}))
        except Exception as exc:
            if isinstance(exc, ConnectionClosed):
                self.logger.info("Connection closed")
            else:
                self.logger.exception("Connection handler error: %s", exc)

        for task in tasks:
            task.cancel()

    async def _handle_message(self, websocket: websockets.WebSocketServerProtocol, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("id", ""))
        room_id = str(payload.get("room_id", ""))
        text = str(payload.get("text", ""))
        try:
            result = await dispatch_graph(self.graph, self.streams, room_id, text)
        except asyncio.CancelledError:
            await websocket.send(json.dumps({"type": "cancelled", "id": request_id, "room_id": room_id}))
        except Exception as exc:
            await websocket.send(
                json.dumps({"type": "error", "id": request_id, "room_id": room_id, "error": str(exc)})
            )
        else:
            await websocket.send(
                json.dumps({"type": "response", "id": request_id, "room_id": room_id, "result": result})
            )


async def serve(host: str, port: int, graph: Any) -> None:
    daemon = YabotDaemon(graph)
    async with websockets.serve(daemon.handler, host, port):
        await asyncio.Future()


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()
    graph = build_graph(config)
    host = "127.0.0.1"
    port = 8765
    logging.info("Starting Yabot daemon on ws://%s:%d", host, port)
    asyncio.run(serve(host, port, graph))


if __name__ == "__main__":
    run()
