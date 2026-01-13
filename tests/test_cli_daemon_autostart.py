import asyncio

import pytest

from yabot.cli import ensure_daemon


class FlakyClient:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def connect(self) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("no daemon")


@pytest.mark.asyncio
async def test_ensure_daemon_autostarts_once():
    client = FlakyClient(failures=2)
    spawned = 0

    def spawn() -> None:
        nonlocal spawned
        spawned += 1

    await ensure_daemon(client, autostart=True, spawn=spawn, retries=3, delay=0)

    assert spawned == 1
    assert client.calls == 3


@pytest.mark.asyncio
async def test_ensure_daemon_no_autostart_raises():
    client = FlakyClient(failures=3)

    def spawn() -> None:
        raise AssertionError("spawn should not be called")

    with pytest.raises(RuntimeError):
        await ensure_daemon(client, autostart=False, spawn=spawn, retries=2, delay=0)
