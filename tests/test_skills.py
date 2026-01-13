import json
from pathlib import Path

import pytest

from langgraph.checkpoint.memory import MemorySaver

from yabot.graph import YabotGraph
from yabot.skills import Skill, SkillRegistry, load_skills


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


class SkillLLM:
    def __init__(self, tool_name: str, expected_system: str) -> None:
        self.tool_name = tool_name
        self.expected_system = expected_system
        self.calls = 0

    async def create_message(self, model, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            tool_call = DummyToolCall("call-1", self.tool_name, "{}")
            return DummyMessage(tool_calls=[tool_call])

        system_messages = [m for m in messages if m.get("role") == "system"]
        assert any(self.expected_system in m.get("content", "") for m in system_messages)
        return DummyMessage(content="done")


def test_load_skills_from_dirs(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    builtin.mkdir()
    user.mkdir()

    (builtin / "one.md").write_text(
        "---\nname: Demo Skill\ndescription: Demo desc\n---\nBody text\n",
        encoding="utf-8",
    )
    (user / "two.md").write_text(
        "---\nname: Other Skill\ndescription: Other desc\n---\nOther body\n",
        encoding="utf-8",
    )

    registry = load_skills([builtin, user])
    names = {skill.name for skill in registry.skills}

    assert names == {"Demo Skill", "Other Skill"}
    tool_names = {skill.tool_name for skill in registry.skills}
    assert len(tool_names) == 2


@pytest.mark.asyncio
async def test_skill_tool_injects_system_message():
    skill = Skill(
        name="Demo Skill",
        description="Demo desc",
        content="Use the demo approach.",
        tool_name="skill__demo_skill",
    )
    llm = SkillLLM(skill.tool_name, "Use the demo approach.")
    graph = YabotGraph(
        llm=llm,
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([skill]),
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke("room1", "do something")
    assert result["responses"][-1] == "done"
    assert result["tool_notices"][0].startswith("Tool call: skill__demo_skill")
    conv = result["conversations"][result["active"]]["messages"]
    assert any(m.get("role") == "system" and "Use the demo approach." in m.get("content", "") for m in conv)


@pytest.mark.asyncio
async def test_ask_user_tool_flow():
    class AskLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def create_message(self, model, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return DummyMessage(
                    tool_calls=[DummyToolCall("call-1", "ask_user", json.dumps({"question": "More detail?"}))],
                )
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert tool_msgs and tool_msgs[-1].get("content") == "extra details"
            return DummyMessage(content="ok")

    graph = YabotGraph(
        llm=AskLLM(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke("room1", "start")
    assert result["responses"][0] == "More detail?"

    result = await graph.ainvoke("room1", "extra details")
    assert result["responses"][0] == "ok"
