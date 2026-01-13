import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from yabot.remote import RemoteGraphClient


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_daemon(url: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        client = RemoteGraphClient(url)
        try:
            await client.connect()
            await client.close()
            return
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.2)
    raise RuntimeError(f"Daemon failed to start: {last_error}")


@pytest.mark.asyncio
async def test_openai_daemon_list_dir_triggers_tool_call(tmp_path: Path):
    if os.environ.get("YABOT_INTEGRATION", "") != "1":
        pytest.skip("Set YABOT_INTEGRATION=1 to run OpenAI integration test.")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set.")

    port = _pick_port()
    url = f"ws://127.0.0.1:{port}"
    trace_path = tmp_path / "trace.jsonl"
    data_home = tmp_path / "xdg_data"
    state_home = tmp_path / "xdg_state"
    env = os.environ.copy()
    env["YABOT_DAEMON_HOST"] = "127.0.0.1"
    env["YABOT_DAEMON_PORT"] = str(port)
    env["YABOT_TRACE_PATH"] = str(trace_path)
    env["XDG_DATA_HOME"] = str(data_home)
    env["XDG_STATE_HOME"] = str(state_home)

    proc = subprocess.Popen(
        [sys.executable, "-m", "yabot.daemon"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    try:
        await _wait_for_daemon(url)
        client = RemoteGraphClient(url)
        result = await client.ainvoke("room", "list files in cwd")
        await client.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    responses = result.get("responses", []) or []
    assert responses, "Expected responses from daemon."
    joined = "\n".join(responses)
    assert "Tool call:" in joined or "Approve access to directory" in joined
