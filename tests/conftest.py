import pytest


class DummyClient:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id


class DummyMessenger:
    def __init__(self, user_id: str = "@bot:example.org") -> None:
        self.client = DummyClient(user_id)
        self.sent: list[tuple[str, str]] = []
        self.typing: list[tuple[str, bool]] = []

    async def send_text(self, room_id: str, body: str) -> str:
        self.sent.append((room_id, body))
        return "event-id"

    async def set_typing(self, room_id: str, is_typing: bool) -> None:
        self.typing.append((room_id, is_typing))


@pytest.fixture
def messenger() -> DummyMessenger:
    return DummyMessenger()
