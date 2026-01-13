import asyncio

import pytest

from yabot.cli import ChatLog, YabotCLIApp


class DummyGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def ainvoke(self, room_id: str, text: str):
        self.calls.append((room_id, text))
        return {
            "responses": [f"echo: {text}"],
            "active": "conv-1",
            "conversations": {"conv-1": {"model": "gpt-4o-mini", "messages": []}},
        }


class SlowGraph:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def ainvoke(self, room_id: str, text: str):
        self.started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return {"responses": ["done"], "active": "conv-1", "conversations": {"conv-1": {"model": "gpt-4o-mini"}}}


@pytest.mark.asyncio
async def test_cli_app_renders_response():
    graph = DummyGraph()
    app = YabotCLIApp(graph=graph, available_models=["gpt-4o-mini"], default_model="gpt-4o-mini")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#chat-input")
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("enter")
        await pilot.pause()

        chat = app.query_one(ChatLog)
        assert ("You", "hello") in chat.messages
        assert ("Yabot", "echo: hello") in chat.messages
        assert graph.calls == [("cli", "hello")]


@pytest.mark.asyncio
async def test_cli_stop_cancels_active_task():
    graph = SlowGraph()
    app = YabotCLIApp(graph=graph)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#chat-input")
        await pilot.press("w", "o", "r", "k")
        await pilot.press("enter")
        await asyncio.wait_for(graph.started.wait(), 1)

        await pilot.click("#chat-input")
        await pilot.press("!", "s", "t", "o", "p")
        await pilot.press("enter")
        await asyncio.wait_for(graph.cancelled.wait(), 1)
        await pilot.pause()

        chat = app.query_one(ChatLog)
        assert ("System", "Stopping current response…") in chat.messages
        assert ("Yabot", "done") not in chat.messages
