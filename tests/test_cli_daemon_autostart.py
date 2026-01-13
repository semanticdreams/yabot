import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from yabot.cli import YabotCLIApp
from yabot.cli_runtime import ensure_daemon
from yabot.remote import RemoteGraphClient


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

    class DummyProc:
        def __init__(self) -> None:
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True

    def spawn() -> DummyProc:
        nonlocal spawned
        spawned += 1
        return DummyProc()

    proc = await ensure_daemon(client, autostart=True, spawn=spawn, retries=3, delay=0)

    assert spawned == 1
    assert client.calls == 3
    assert proc is not None


@pytest.mark.asyncio
async def test_ensure_daemon_no_autostart_raises():
    client = FlakyClient(failures=3)

    class DummyProc:
        def terminate(self) -> None:
            raise AssertionError("terminate should not be called")

    def spawn() -> DummyProc:
        raise AssertionError("spawn should not be called")

    with pytest.raises(RuntimeError):
        await ensure_daemon(client, autostart=False, spawn=spawn, retries=2, delay=0)


@pytest.mark.asyncio
async def test_cli_closes_autostarted_daemon_process(tmp_path: Path):
    pid_path = tmp_path / "daemon.pid"
    app = YabotCLIApp(
        graph=RemoteGraphClient("ws://127.0.0.1:1"),
        daemon_pid_path=pid_path,
    )
    app._daemon_autostarted = True

    proc = subprocess.Popen(["sleep", "10"])
    try:
        pid_path.write_text(f"{proc.pid}\n{os.getpid()}\n", encoding="utf-8")

        await app._close_remote()

        proc.wait(timeout=2)
        assert not pid_path.exists()
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=2)
