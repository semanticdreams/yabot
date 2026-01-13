import json
from pathlib import Path

import pytest

from langgraph.checkpoint.memory import MemorySaver

from yabot.graph import YabotGraph
from yabot.skills import SkillRegistry


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

    async def create_message(self, model, messages, tools=None):
        self.calls.append(messages)
        return self.messages.pop(0)


class ValidatingLLM:
    def __init__(self, tool_call: DummyToolCall, final: str) -> None:
        self.tool_call = tool_call
        self.final = final
        self.calls = 0

    async def create_message(self, model, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return DummyMessage(tool_calls=[self.tool_call])

        tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        assert tool_indexes, "Expected tool messages before final LLM call."
        first_tool = tool_indexes[0]
        assistant_with_calls = any(
            m.get("role") == "assistant" and m.get("tool_calls") for m in messages[:first_tool]
        )
        assert assistant_with_calls, "Tool messages must follow an assistant tool_calls message."
        return DummyMessage(content=self.final)


@pytest.mark.asyncio
async def test_run_shell_requires_approval_and_remembers(tmp_path: Path):
    out_path = tmp_path / "out.txt"
    args = json.dumps({"command": f"python -c \"open('{out_path}','w').write('ok')\"", "workdir": str(tmp_path)})
    tool_call = DummyToolCall("call-1", "run_shell", args)
    llm = DummyLLM([DummyMessage(tool_calls=[tool_call]), DummyMessage(content="done")])
    graph = YabotGraph(
        llm=llm,
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke("room1", "run")

    assert not out_path.exists()
    assert any("Approve running shell command" in r for r in result["responses"])

    result = await graph.ainvoke("room1", " y ")

    assert out_path.exists()
    assert "done" in result["responses"]
    approvals = result["approvals"]["shell"]
    assert f"{json.loads(args)['command']}\n{str(tmp_path)}" in approvals


@pytest.mark.asyncio
async def test_run_shell_denied_with_feedback(tmp_path: Path):
    out_path = tmp_path / "cancel.txt"
    args = json.dumps({"command": f"python -c \"open('{out_path}','w').write('ok')\"", "workdir": str(tmp_path)})
    tool_call = DummyToolCall("call-1", "run_shell", args)
    llm = DummyLLM([DummyMessage(tool_calls=[tool_call]), DummyMessage(content="ok")])
    graph = YabotGraph(
        llm=llm,
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )

    await graph.ainvoke("room1", "run")
    result = await graph.ainvoke("room1", "nope")

    assert not out_path.exists()
    assert "ok" in result["responses"]
    assert result["approvals"]["pending"] is None


@pytest.mark.asyncio
async def test_fs_tool_approval_scopes_directory(tmp_path: Path):
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
    graph = YabotGraph(
        llm=llm,
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke("room1", "read")

    assert any("Approve access to directory" in r for r in result["responses"])

    result = await graph.ainvoke("room1", "y")

    assert "done1" in result["responses"]
    assert str(tmp_path) in result["approvals"]["dirs"]

    result = await graph.ainvoke("room1", "read again")

    assert "done2" in result["responses"]


@pytest.mark.asyncio
async def test_tool_messages_follow_tool_calls(tmp_path: Path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("hi", encoding="utf-8")
    args = json.dumps({"path": str(file_path)})
    tool_call = DummyToolCall("call-1", "read_file", args)
    llm = ValidatingLLM(tool_call, "done")
    graph = YabotGraph(
        llm=llm,
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke("room1", "read")
    assert any("Approve access to directory" in r for r in result["responses"])

    result = await graph.ainvoke("room1", "y")
    assert "done" in result["responses"]
