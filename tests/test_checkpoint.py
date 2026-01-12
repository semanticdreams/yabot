from pathlib import Path

import pytest

from yabot.checkpoint import FileBackedSaver
from yabot.graph import YabotGraph


class DummyLLM:
    async def create_message(self, model, messages):
        raise AssertionError("LLM should not be called for commands.")


@pytest.mark.asyncio
async def test_file_backed_saver_persists_state(tmp_path: Path):
    path = tmp_path / "graph_state.pkl"
    graph1 = YabotGraph(
        llm=DummyLLM(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        checkpointer=FileBackedSaver(str(path)),
    )

    result1 = await graph1.ainvoke("room1", "!new")
    active_id = result1["active"]

    graph2 = YabotGraph(
        llm=DummyLLM(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=3,
        checkpointer=FileBackedSaver(str(path)),
    )
    result2 = await graph2.ainvoke("room1", "!list")

    assert path.exists()
    assert active_id in result2["responses"][0]
