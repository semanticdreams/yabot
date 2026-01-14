import io

import pytest
from langgraph.checkpoint.memory import MemorySaver

from yabot.cli import YabotCLI
from yabot.graph import YabotGraph
from yabot.skills import SkillRegistry
from yabot.system_prompt import meta_system_prompt


LEGACY_META_PROMPT = "\n".join(
    [
        "You are the Yabot meta agent.",
        "You can manage and query the main agent on the user's behalf.",
        "Use agent tools when the user asks to change the main agent, ask it questions, or inspect its tool calls.",
        "You may answer directly when no delegation is needed.",
        "Be concise and precise. Ask clarifying questions only when needed.",
    ]
)


class PromptEchoLLM:
    class Message:
        def __init__(self, content: str) -> None:
            self.role = "assistant"
            self.content = content
            self.tool_calls = []

    async def create_message(self, model, messages, tools=None):
        system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        return self.Message(system)

    async def create_message_stream(self, model, messages, tools=None, on_token=None):
        system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        if on_token:
            await on_token(system)
        return {"role": "assistant", "content": system, "tool_calls": []}


class DummyInput:
    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def __call__(self, _prompt: str = "") -> str:
        try:
            return next(self._lines)
        except StopIteration as exc:
            raise EOFError from exc


class SimpleLLM:
    class Message:
        def __init__(self, content: str) -> None:
            self.role = "assistant"
            self.content = content
            self.tool_calls = []

    async def create_message(self, model, messages, tools=None):
        return self.Message("ok")

    async def create_message_stream(self, model, messages, tools=None, on_token=None):
        if on_token:
            await on_token("ok")
        return {"role": "assistant", "content": "ok", "tool_calls": []}


@pytest.mark.asyncio
async def test_cli_replaces_legacy_system_prompt():
    graph = YabotGraph(
        llm=PromptEchoLLM(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
        meta_system_prompt=meta_system_prompt(),
    )
    config = {"configurable": {"thread_id": "cli"}}
    graph.graph.update_state(
        config,
        {
            "active_agent": "meta",
            "agents": {
                "meta": {
                    "active": "conv-1",
                    "conversations": {
                        "conv-1": {
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "system", "content": LEGACY_META_PROMPT}],
                        }
                    },
                },
                "main": {"active": "conv-2", "conversations": {"conv-2": {"model": "gpt-4o-mini", "messages": []}}},
            },
            "approvals": {"shell": [], "dirs": [], "pending": None},
        },
    )
    output = io.StringIO()
    cli = YabotCLI(graph=graph, input_fn=DummyInput(["who are you"]), output=output)

    await cli.run_async()

    assert "You are meta." in output.getvalue()


@pytest.mark.asyncio
async def test_cli_clears_stale_plan_steps():
    graph = YabotGraph(
        llm=SimpleLLM(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
        meta_system_prompt=meta_system_prompt(),
    )
    config = {"configurable": {"thread_id": "cli"}}
    graph.graph.update_state(
        config,
        {
            "active_agent": "meta",
            "agents": {
                "meta": {"active": "conv-1", "conversations": {"conv-1": {"model": "gpt-4o-mini", "messages": []}}},
                "main": {"active": "conv-2", "conversations": {"conv-2": {"model": "gpt-4o-mini", "messages": []}}},
            },
            "plan_steps": ["old step 1", "old step 2", "old step 3"],
            "plan_index": 2,
            "approvals": {"shell": [], "dirs": [], "pending": None},
        },
    )
    output = io.StringIO()
    cli = YabotCLI(graph=graph, input_fn=DummyInput(["who are you"]), output=output)

    await cli.run_async()

    assert "[system] Step" not in output.getvalue()
