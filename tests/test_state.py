import pytest

from yabot.state import StateStore


def test_trim_messages_keeps_recent_turns(tmp_path):
    state = StateStore(
        state_path=str(tmp_path / "state.json"),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=1,
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    trimmed = state.trim_messages(messages, "gpt-4o-mini")

    assert trimmed[0]["role"] == "system"
    assert trimmed[-2]["content"] == "u2"
    assert trimmed[-1]["content"] == "a2"
