from pathlib import Path

import pytest

from yabot.state import StateStore


@pytest.fixture
def state_store(tmp_path: Path) -> StateStore:
    return StateStore(
        state_path=str(tmp_path / "state.json"),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini", "gpt-5.2"],
        max_turns=3,
    )


class DummyClient:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id


class DummyMessenger:
    def __init__(self, user_id: str = "@bot:example.org") -> None:
        self.client = DummyClient(user_id)
        self.sent: list[tuple[str, str]] = []

    async def send_text(self, room_id: str, body: str) -> str:
        self.sent.append((room_id, body))
        return "event-id"


@pytest.fixture
def messenger() -> DummyMessenger:
    return DummyMessenger()
