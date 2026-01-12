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


class DummyLLM:
    class Message:
        def __init__(self, content: str) -> None:
            self.role = "assistant"
            self.content = content
            self.tool_calls = []

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def create_message(self, model, messages):
        self.calls.append((model, messages))
        return self.Message(self.reply)


class DummyCommands:
    def __init__(self, handled: bool) -> None:
        self.handled = handled
        self.calls: list[tuple[str, str]] = []

    async def try_handle(self, room_id: str, text: str) -> bool:
        self.calls.append((room_id, text))
        return self.handled


@pytest.mark.asyncio
async def test_on_message_ignored_for_disallowed_sender(state_store, messenger):
    llm = DummyLLM()
    commands = DummyCommands(handled=False)
    streams = StreamRegistry()
    handler = BotHandler(
        state=state_store,
        messenger=messenger,
        llm=llm,
        commands=commands,
        default_model="gpt-4o-mini",
        streams=streams,
        allowed_users=["@alice:example.org"],
    )

    await handler.on_message(DummyRoom("room1"), DummyEvent("@bob:example.org", "hi"))

    assert llm.calls == []
    assert commands.calls == []
    assert state_store.state["rooms"] == {}


@pytest.mark.asyncio
async def test_on_message_runs_llm_and_updates_state(state_store, messenger):
    llm = DummyLLM(reply="hello there")
    commands = DummyCommands(handled=False)
    streams = StreamRegistry()
    handler = BotHandler(
        state=state_store,
        messenger=messenger,
        llm=llm,
        commands=commands,
        default_model="gpt-4o-mini",
        streams=streams,
        allowed_users=[],
    )

    await handler.on_message(DummyRoom("room1"), DummyEvent("@alice:example.org", "hello"))

    assert len(llm.calls) == 1
    assert streams.stop("room1") is False

    _, conv = state_store.room_active_conv("room1")
    assert conv["messages"][0]["content"] == "hello"
    assert conv["messages"][-1]["content"] == "hello there"
