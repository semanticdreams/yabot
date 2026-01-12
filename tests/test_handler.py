import asyncio

import pytest

from yabot.handler import BotHandler, StreamRegistry


class DummyRoom:
    def __init__(self, room_id: str) -> None:
        self.room_id = room_id


class DummyEvent:
    def __init__(self, sender: str, body: str) -> None:
        self.sender = sender
        self.body = body


class DummyGraph:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[tuple[str, str]] = []

    async def ainvoke(self, room_id: str, text: str):
        self.calls.append((room_id, text))
        return {"responses": list(self.responses)}


@pytest.mark.asyncio
async def test_on_message_ignored_for_disallowed_sender(messenger):
    graph = DummyGraph(responses=["ok"])
    streams = StreamRegistry()
    handler = BotHandler(
        messenger=messenger,
        graph=graph,
        streams=streams,
        allowed_users=["@alice:example.org"],
    )

    await handler.on_message(DummyRoom("room1"), DummyEvent("@bob:example.org", "hi"))

    assert graph.calls == []


@pytest.mark.asyncio
async def test_on_message_runs_llm_and_updates_state(messenger):
    graph = DummyGraph(responses=["hello there"])
    streams = StreamRegistry()
    handler = BotHandler(
        messenger=messenger,
        graph=graph,
        streams=streams,
        allowed_users=[],
    )

    await handler.on_message(DummyRoom("room1"), DummyEvent("@alice:example.org", "hello"))

    assert len(graph.calls) == 1
    assert streams.stop("room1") is False
    assert messenger.sent[-1][1] == "hello there"
