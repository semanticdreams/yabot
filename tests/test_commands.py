import json

import pytest

from langgraph.checkpoint.memory import MemorySaver

from yabot.graph import YabotGraph
from yabot.skills import SkillRegistry


class DummyLLM:
    async def create_message(self, model, messages, tools=None):
        raise AssertionError("LLM should not be called for commands.")


class EchoLLM:
    class Message:
        def __init__(self, content: str) -> None:
            self.role = "assistant"
            self.content = content
            self.tool_calls = []

    async def create_message(self, model, messages, tools=None):
        return self.Message("ok")


class PlannerLLM:
    class Message:
        def __init__(self, content: str) -> None:
            self.role = "assistant"
            self.content = content
            self.tool_calls = []

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def create_message(self, model, messages, tools=None):
        self.calls.append(messages)
        system = messages[0].get("content", "") if messages else ""
        if "planner" in system.lower():
            return self.Message("- step one\n- step two")
        user = messages[-1].get("content", "") if messages else ""
        if "step one" in user:
            return self.Message("done one")
        if "step two" in user:
            return self.Message("done two")
        return self.Message("done")


class PlanApprovalLLM:
    class Message:
        def __init__(self, content: str | None = None, tool_calls=None) -> None:
            self.role = "assistant"
            self.content = content
            self.tool_calls = tool_calls or []

    class ToolCall:
        def __init__(self, call_id: str, name: str, arguments: str) -> None:
            self.id = call_id
            self.function = type("Fn", (), {"name": name, "arguments": arguments})()

        def model_dump(self) -> dict:
            return {
                "id": self.id,
                "function": {"name": self.function.name, "arguments": self.function.arguments},
            }

    def __init__(self, command: str) -> None:
        self.command = command
        self.calls: list[list[dict]] = []

    async def create_message(self, model, messages, tools=None):
        self.calls.append(messages)
        system = messages[0].get("content", "") if messages else ""
        if "planner" in system.lower():
            return self.Message("- step one\n- step two")

        last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        has_tool = any(m.get("role") == "tool" for m in messages)
        if "Execute step 1/2" in last_user and not has_tool:
            args = json.dumps({"command": self.command})
            return self.Message(tool_calls=[self.ToolCall("call-1", "run_shell", args)])
        if "Execute step 1/2" in last_user and has_tool:
            tool_msg = next(m for m in messages if m.get("role") == "tool")
            payload = json.loads(tool_msg.get("content", "{}"))
            assert "hi" in payload.get("stdout", "")
            return self.Message("done one")
        if "Execute step 2/2" in last_user:
            return self.Message("done two")
        return self.Message("done")


@pytest.mark.asyncio
async def test_model_command_updates_state():
    graph = YabotGraph(
        llm=DummyLLM(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini", "gpt-5.2"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke("room1", "!model gpt-5.2")

    assert result["responses"][0].startswith("Model set to `gpt-5.2`")
    active = result["active"]
    assert result["conversations"][active]["model"] == "gpt-5.2"


@pytest.mark.asyncio
async def test_new_list_use_reset_flow():
    graph = YabotGraph(
        llm=EchoLLM(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini", "gpt-5.2"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )

    await graph.ainvoke("room1", "!new")
    result = await graph.ainvoke("room1", "!new")
    new_id = result["active"]

    result = await graph.ainvoke("room1", "!new")
    newer_id = result["active"]
    assert newer_id != new_id

    list_result = await graph.ainvoke("room1", "!list")
    list_body = list_result["responses"][0]
    assert "Conversations for this room:" in list_body
    assert new_id in list_body
    assert newer_id in list_body

    await graph.ainvoke("room1", "hi")

    result = await graph.ainvoke("room1", "!reset")
    assert result["conversations"][newer_id]["messages"] == []


@pytest.mark.asyncio
async def test_remaining_context_command_returns_percentage():
    graph = YabotGraph(
        llm=DummyLLM(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke("room1", "!remaining-context-percentage")

    assert result["responses"][0].startswith("Remaining context:")


@pytest.mark.asyncio
async def test_auto_plans_complex_requests():
    llm = PlannerLLM()
    graph = YabotGraph(
        llm=llm,
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke("room1", "Please do A, then B, and then C with details.")

    assert result["responses"][0].startswith("Plan:")
    assert "step one" in result["responses"][0]
    assert "[system] Step 1/2" in result["responses"][1]
    assert "done one" in result["responses"][2]
    assert "[system] Step 2/2" in result["responses"][3]
    assert "done two" in result["responses"][4]
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_plan_resumes_after_approval_with_tool_output():
    command = "python -c \"print('hi')\""
    llm = PlanApprovalLLM(command)
    graph = YabotGraph(
        llm=llm,
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )

    result = await graph.ainvoke("room1", "Please do A, then B with details.")

    assert any("Approve running shell command" in r for r in result["responses"])

    result = await graph.ainvoke("room1", "y")

    assert any("done one" in r for r in result["responses"])
    assert any("[system] Step 2/2" in r for r in result["responses"])
    assert any("done two" in r for r in result["responses"])
