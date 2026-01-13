from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from pathlib import Path

import websockets
from websockets.exceptions import ConnectionClosed

from .config import load_config
from .runtime import build_graph
from .streams import StreamRegistry
from .trace import TraceLogger
from .ws_protocol import ClientMessage, ServerMessage, parse_json


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
                    payload = parse_json(raw)
                except Exception:
                    await websocket.send(
                        ServerMessage(type="error", id="", room_id="", error="Invalid JSON").to_json()
                    )
                    continue

                msg_type = payload.get("type")
                assert msg_type in {"message", "stop"}, f"unknown message type: {msg_type}"
                if msg_type == "message":
                    assert payload.get("id"), "message id is required"
                    assert payload.get("room_id"), "room_id is required"
                    task = asyncio.create_task(self._handle_message(websocket, payload))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                    continue

                if msg_type == "stop":
                    assert payload.get("id"), "stop id is required"
                    assert payload.get("room_id"), "room_id is required"
                    room_id = str(payload.get("room_id", ""))
                    request_id = payload.get("id")
                    ok = self.streams.stop(room_id)
                    await websocket.send(
                        ServerMessage(type="stopped", id=str(request_id), room_id=room_id, ok=ok).to_json()
                    )
                    continue

                await websocket.send(
                    ServerMessage(type="error", id="", room_id="", error="Unknown message type").to_json()
                )
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
        assert request_id, "request id is required"
        assert room_id, "room_id is required"
        try:
            async def on_token(chunk: str) -> None:
                await websocket.send(
                    ServerMessage(type="stream", id=request_id, room_id=room_id, chunk=chunk).to_json()
                )

            async def run_graph() -> dict[str, Any]:
                stream_fn = getattr(self.graph, "ainvoke_stream", None)
                if callable(stream_fn):
                    return await stream_fn(room_id, text, on_token=on_token)
                return await self.graph.ainvoke(room_id, text)

            task = asyncio.create_task(run_graph())
            self.streams.register(room_id, task)
            try:
                result = await task
            finally:
                self.streams.clear(room_id, task)
        except asyncio.CancelledError:
            await websocket.send(ServerMessage(type="cancelled", id=request_id, room_id=room_id).to_json())
        except Exception as exc:
            await websocket.send(
                ServerMessage(type="error", id=request_id, room_id=room_id, error=str(exc)).to_json()
            )
        else:
            await websocket.send(
                ServerMessage(type="response", id=request_id, room_id=room_id, result=result).to_json()
            )


async def serve(host: str, port: int, graph: Any, parent_pid: int | None = None) -> None:
    daemon = YabotDaemon(graph)
    shutdown_event = asyncio.Event()
    monitor_task: asyncio.Task[None] | None = None
    if parent_pid is not None:
        monitor_task = asyncio.create_task(_monitor_parent(parent_pid, shutdown_event))
    async with websockets.serve(daemon.handler, host, port):
        await shutdown_event.wait()
    if monitor_task is not None:
        monitor_task.cancel()


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()
    _write_pidfile()
    TraceLogger(Path(config.trace_path)).log(
        "daemon_startup",
        {
            "trace_path": str(config.trace_path),
        },
    )
    graph = build_graph(config)
    host = config.daemon_host
    port = config.daemon_port
    logging.info("Starting Yabot daemon on ws://%s:%d", host, port)
    parent_pid = _parent_pid_from_env()
    asyncio.run(serve(host, port, graph, parent_pid=parent_pid))


def _write_pidfile() -> None:
    path = os.environ.get("YABOT_DAEMON_PID_PATH")
    if not path:
        return
    parent = os.environ.get("YABOT_DAEMON_PARENT_PID", "")
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(f"{os.getpid()}\n{parent}\n", encoding="utf-8")
    except OSError as exc:
        logging.getLogger("yabot.daemon").warning("Failed to write pid file %s: %s", path, exc)


def _parent_pid_from_env() -> int | None:
    raw = os.environ.get("YABOT_DAEMON_PARENT_PID")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def _monitor_parent(parent_pid: int, shutdown_event: asyncio.Event) -> None:
    while True:
        try:
            os.kill(parent_pid, 0)
        except OSError:
            shutdown_event.set()
            return
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    run()
