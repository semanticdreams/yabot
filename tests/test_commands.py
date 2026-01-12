import pytest

from yabot.commands import CommandProcessor


@pytest.mark.asyncio
async def test_model_command_updates_state(state_store, messenger):
    processor = CommandProcessor(
        state=state_store,
        messenger=messenger,
        available_models=["gpt-4o-mini", "gpt-5.2"],
        default_model="gpt-4o-mini",
        stop_stream=lambda _room_id: False,
    )

    ok = await processor.try_handle("room1", "!model gpt-5.2")

    assert ok is True
    assert messenger.sent[-1][1].startswith("Model set to `gpt-5.2`")
    _, conv = state_store.room_active_conv("room1")
    assert conv["model"] == "gpt-5.2"


@pytest.mark.asyncio
async def test_new_list_use_reset_flow(state_store, messenger):
    processor = CommandProcessor(
        state=state_store,
        messenger=messenger,
        available_models=["gpt-4o-mini", "gpt-5.2"],
        default_model="gpt-4o-mini",
        stop_stream=lambda _room_id: False,
    )

    async with state_store.lock:
        state_store.ensure_room("room1")
        initial_id, _ = state_store.room_active_conv("room1")

    await processor.try_handle("room1", "!new")
    async with state_store.lock:
        new_id, _ = state_store.room_active_conv("room1")

    assert new_id != initial_id

    await processor.try_handle("room1", "!list")
    list_body = messenger.sent[-1][1]
    assert "Conversations for this room:" in list_body
    assert initial_id in list_body
    assert new_id in list_body

    async with state_store.lock:
        _, conv = state_store.room_active_conv("room1")
        conv["messages"] = [{"role": "user", "content": "hi"}]

    await processor.try_handle("room1", "!reset")
    async with state_store.lock:
        _, conv = state_store.room_active_conv("room1")
        assert conv["messages"] == []


@pytest.mark.asyncio
async def test_stop_command_responses(state_store, messenger):
    calls = {"count": 0}

    def stop_stream(_room_id: str) -> bool:
        calls["count"] += 1
        return calls["count"] == 1

    processor = CommandProcessor(
        state=state_store,
        messenger=messenger,
        available_models=["gpt-4o-mini", "gpt-5.2"],
        default_model="gpt-4o-mini",
        stop_stream=stop_stream,
    )

    await processor.try_handle("room1", "!stop")
    assert messenger.sent[-1][1].startswith("Stopping current response")

    await processor.try_handle("room1", "!stop")
    assert messenger.sent[-1][1] == "No active response to stop."
