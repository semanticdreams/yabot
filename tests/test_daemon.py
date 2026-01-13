import asyncio
import pytest
import websockets

from yabot.daemon import YabotDaemon
from yabot.remote import RemoteGraphClient
from yabot.ws_protocol import ClientMessage, parse_json


class StatefulGraph:
    def __init__(self) -> None:
        self.state: dict[str, int] = {}

    async def ainvoke(self, room_id: str, text: str):
        count = self.state.get(room_id, 0) + 1
        self.state[room_id] = count
        return {
            "responses": [f"{text}:{count}"],
            "active": "conv",
            "conversations": {"conv": {"model": "gpt-4o-mini"}},
        }


class SlowGraph:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def ainvoke(self, room_id: str, text: str):
        self.started.set()
        await asyncio.sleep(10)
        return {"responses": ["done"]}


async def _start_server(graph):
    daemon = YabotDaemon(graph)
    server = await websockets.serve(daemon.handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"ws://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_daemon_shares_state_between_clients():
    graph = StatefulGraph()
    server, url = await _start_server(graph)
    async with server:
        client1 = RemoteGraphClient(url)
        client2 = RemoteGraphClient(url)
        result1 = await client1.ainvoke("room", "hello")
        result2 = await client2.ainvoke("room", "hi")
        await client1.close()
        await client2.close()

    assert result1["responses"] == ["hello:1"]
    assert result2["responses"] == ["hi:2"]


@pytest.mark.asyncio
async def test_daemon_stop_cancels_task():
    graph = SlowGraph()
    server, url = await _start_server(graph)
    async with server:
        async with websockets.connect(url) as ws:
            await ws.send(ClientMessage(type="message", id="1", room_id="room", text="work").to_json())
            await asyncio.wait_for(graph.started.wait(), 1)
            await ws.send(ClientMessage(type="stop", id="stop-1", room_id="room").to_json())
            stopped = parse_json(await ws.recv())
            assert stopped["type"] == "stopped"
            assert stopped["id"] == "stop-1"
            assert stopped["ok"] is True
