import json
from pathlib import Path

import pytest

from yabot.handler import BotHandler, StreamRegistry


class DummyRoom:
    def __init__(self, room_id: str) -> None:
        self.room_id = room_id


class DummyEvent:
    def __init__(self, sender: str, body: str) -> None:
        self.sender = sender
        self.body = body


class DummyFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class DummyToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = DummyFunction(name, arguments)

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


class DummyMessage:
    def __init__(self, content: str = "", tool_calls: list[DummyToolCall] | None = None) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or []


class DummyLLM:
    def __init__(self, messages: list[DummyMessage]) -> None:
        self.messages = list(messages)
        self.calls: list[list[dict[str, str]]] = []

    async def create_message(self, model, messages):
        self.calls.append(messages)
        return self.messages.pop(0)


class DummyCommands:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def try_handle(self, room_id: str, text: str) -> bool:
        self.calls.append((room_id, text))
        return False


@pytest.mark.asyncio
async def test_run_shell_requires_approval_and_remembers(state_store, messenger, tmp_path: Path):
    out_path = tmp_path / "out.txt"
    args = json.dumps({"command": f"python -c \"open('{out_path}','w').write('ok')\"", "workdir": str(tmp_path)})
    tool_call = DummyToolCall("call-1", "run_shell", args)
    llm = DummyLLM([DummyMessage(tool_calls=[tool_call]), DummyMessage(content="done")])

    handler = BotHandler(
        state=state_store,
        messenger=messenger,
        llm=llm,
        commands=DummyCommands(),
        default_model="gpt-4o-mini",
        streams=StreamRegistry(),
        allowed_users=[],
    )

    await handler.on_message(DummyRoom("room1"), DummyEvent("@alice:example.org", "run"))

    assert not out_path.exists()
    assert "Approve running shell command" in messenger.sent[-1][1]

    await handler.on_message(DummyRoom("room1"), DummyEvent("@alice:example.org", " y "))

    assert out_path.exists()
    assert messenger.sent[-1][1] == "done"
    assert state_store.room_is_shell_approved("room1", json.loads(args)["command"], str(tmp_path))


@pytest.mark.asyncio
async def test_run_shell_cancelled_on_non_y(state_store, messenger, tmp_path: Path):
    out_path = tmp_path / "cancel.txt"
    args = json.dumps({"command": f"python -c \"open('{out_path}','w').write('ok')\"", "workdir": str(tmp_path)})
    tool_call = DummyToolCall("call-1", "run_shell", args)
    llm = DummyLLM([DummyMessage(tool_calls=[tool_call])])

    handler = BotHandler(
        state=state_store,
        messenger=messenger,
        llm=llm,
        commands=DummyCommands(),
        default_model="gpt-4o-mini",
        streams=StreamRegistry(),
        allowed_users=[],
    )

    await handler.on_message(DummyRoom("room1"), DummyEvent("@alice:example.org", "run"))
    await handler.on_message(DummyRoom("room1"), DummyEvent("@alice:example.org", "nope"))

    assert not out_path.exists()
    assert messenger.sent[-1][1] == "Cancelled."
    assert state_store.room_get_pending("room1") is None


@pytest.mark.asyncio
async def test_fs_tool_approval_scopes_directory(state_store, messenger, tmp_path: Path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("hi", encoding="utf-8")
    args1 = json.dumps({"path": str(file_path)})
    tool_call1 = DummyToolCall("call-1", "read_file", args1)
    args2 = json.dumps({"path": str(file_path)})
    tool_call2 = DummyToolCall("call-2", "read_file", args2)

    llm = DummyLLM(
        [
            DummyMessage(tool_calls=[tool_call1]),
            DummyMessage(content="done1"),
            DummyMessage(tool_calls=[tool_call2]),
            DummyMessage(content="done2"),
        ]
    )

    handler = BotHandler(
        state=state_store,
        messenger=messenger,
        llm=llm,
        commands=DummyCommands(),
        default_model="gpt-4o-mini",
        streams=StreamRegistry(),
        allowed_users=[],
    )

    await handler.on_message(DummyRoom("room1"), DummyEvent("@alice:example.org", "read"))

    assert "Approve access to directory" in messenger.sent[-1][1]

    await handler.on_message(DummyRoom("room1"), DummyEvent("@alice:example.org", "y"))

    assert messenger.sent[-1][1] == "done1"
    assert state_store.room_is_dir_approved("room1", str(tmp_path))

    await handler.on_message(DummyRoom("room1"), DummyEvent("@alice:example.org", "read again"))

    assert messenger.sent[-1][1] == "done2"
