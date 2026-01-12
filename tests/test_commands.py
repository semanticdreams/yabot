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
