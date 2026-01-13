import json
from pathlib import Path

import pytest

from langgraph.checkpoint.memory import MemorySaver

from yabot.graph import YabotGraph
from yabot.skills import SkillRegistry
from yabot.trace import TraceLogger


class EchoLLM:
    class Message:
        def __init__(self, content: str) -> None:
            self.role = "assistant"
            self.content = content
            self.tool_calls = []

    async def create_message(self, model, messages, tools=None):
        return self.Message("ok")


@pytest.mark.asyncio
async def test_trace_logger_writes_events(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    tracer = TraceLogger(trace_path)
    graph = YabotGraph(
        llm=EchoLLM(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
        tracer=tracer,
    )

    await graph.ainvoke("room1", "hello")

    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "Expected trace log lines"
    payloads = [json.loads(line) for line in lines]
    events = {entry.get("event") for entry in payloads}
    assert "invoke" in events
    assert "response_final" in events or "invoke_result" in events
