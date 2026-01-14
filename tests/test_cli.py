import asyncio
import io

import pytest
from langgraph.checkpoint.memory import MemorySaver

from yabot.cli import YabotCLI
from yabot.graph import YabotGraph
from yabot.skills import SkillRegistry


class DummyInput:
    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def __call__(self, _prompt: str = "") -> str:
        try:
            return next(self._lines)
        except StopIteration as exc:
            raise EOFError from exc


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


class StreamGraph:
    async def ainvoke_stream(self, room_id: str, text: str, on_token):
        await on_token("stream")
        await on_token("ed")
        return {"responses": ["streamed"]}


class StreamGraphMismatch:
    async def ainvoke_stream(self, room_id: str, text: str, on_token):
        await on_token("hello worl")
        return {"responses": ["hello world"]}


class StreamGraphMatch:
    async def ainvoke_stream(self, room_id: str, text: str, on_token):
        await on_token("hello world")
        return {"responses": ["hello world"]}


class DummyLLM:
    async def create_message(self, model, messages, tools=None):
        raise AssertionError("LLM should not be called for commands.")


@pytest.mark.asyncio
async def test_cli_prints_response():
    graph = DummyGraph()
    output = io.StringIO()
    cli = YabotCLI(graph=graph, input_fn=DummyInput(["hello"]), output=output)

    await cli.run_async()

    assert "echo: hello" in output.getvalue()
    assert "[system] Processing request" in output.getvalue()
    assert "[system] Response complete." in output.getvalue()
    assert graph.calls == [("cli", "hello")]


@pytest.mark.asyncio
async def test_cli_stop_cancels_active_task():
    graph = SlowGraph()
    output = io.StringIO()
    cli = YabotCLI(graph=graph, input_fn=DummyInput(["work", "!stop"]), output=output)

    runner = asyncio.create_task(cli.run_async())
    await asyncio.wait_for(graph.started.wait(), 1)
    await asyncio.wait_for(graph.cancelled.wait(), 1)
    await runner

    out_text = output.getvalue()
    assert "[system] Processing request" in out_text
    assert "Stopping current response" in out_text
    assert "done" not in out_text


@pytest.mark.asyncio
async def test_cli_context_command_prints_percentage():
    graph = YabotGraph(
        llm=DummyLLM(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        skills=SkillRegistry([]),
        checkpointer=MemorySaver(),
    )
    output = io.StringIO()
    cli = YabotCLI(graph=graph, input_fn=DummyInput(["!remaining-context-percentage"]), output=output)

    await cli.run_async()

    assert "[system] Processing request" in output.getvalue()
    assert "Remaining context:" in output.getvalue()


@pytest.mark.asyncio
async def test_cli_streams_llm_output():
    output = io.StringIO()
    cli = YabotCLI(graph=StreamGraph(), input_fn=DummyInput(["go"]), output=output)

    await cli.run_async()

    out_text = output.getvalue()
    assert out_text.count("streamed") == 1


@pytest.mark.asyncio
async def test_cli_corrects_stream_mismatch():
    output = io.StringIO()
    cli = YabotCLI(graph=StreamGraphMismatch(), input_fn=DummyInput(["go"]), output=output)

    await cli.run_async()

    out_text = output.getvalue()
    assert "Output corrected from final response." in out_text
    assert "hello world" in out_text


@pytest.mark.asyncio
async def test_cli_does_not_correct_when_stream_matches():
    output = io.StringIO()
    cli = YabotCLI(graph=StreamGraphMatch(), input_fn=DummyInput(["go"]), output=output)

    await cli.run_async()

    out_text = output.getvalue()
    assert "Output corrected from final response." not in out_text
