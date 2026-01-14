import asyncio
import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from yabot.cli import YabotCLI
from yabot.config import load_config
from yabot.remote import RemoteGraphClient
from yabot.runtime import build_graph


class DummyInput:
    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def __call__(self, _prompt: str = "") -> str:
        try:
            return next(self._lines)
        except StopIteration as exc:
            raise EOFError from exc


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


def _load_trace_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


async def _wait_for_trace_event(path: Path, predicate, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    last_seen: dict | None = None
    while time.time() < deadline:
        events = _load_trace_events(path)
        for event in events:
            if predicate(event):
                return event
        if events:
            last_seen = events[-1]
        await asyncio.sleep(0.2)
    raise AssertionError(f"Timed out waiting for trace event. Last seen: {last_seen}")


async def _invoke_with_approvals(client: RemoteGraphClient, room_id: str, text: str, max_rounds: int = 4):
    result = await client.ainvoke(room_id, text)
    rounds = 0
    while rounds < max_rounds:
        responses = result.get("responses", []) or []
        pending = (result.get("approvals") or {}).get("pending")
        if not any("Approve " in r for r in responses) and not pending:
            return result
        result = await client.ainvoke(room_id, "y")
        rounds += 1
    return result


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


@pytest.mark.asyncio
async def test_openai_cli_streams_approval_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if os.environ.get("YABOT_INTEGRATION", "") != "1":
        pytest.skip("Set YABOT_INTEGRATION=1 to run OpenAI integration test.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set.")

    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg_data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
    monkeypatch.setenv("YABOT_TRACE_PATH", str(tmp_path / "trace.jsonl"))

    graph = build_graph(load_config())
    output = io.StringIO()
    cli = YabotCLI(
        graph=graph,
        input_fn=DummyInput(["List files in cwd, then read README.md and summarize."]),
        output=output,
    )

    await cli.run_async()

    assert "Approve access to directory" in output.getvalue()


@pytest.mark.asyncio
async def test_openai_daemon_fetches_hn_titles(tmp_path: Path):
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
        prompt = (
            "Fetch the top 10 Hacker News story titles using the HN API. "
            "Use the run_shell tool with a python snippet (urllib + json) to print "
            "one title per line to stdout. Do not use notify-send or jq. "
            "Return the 10 titles in your final response."
        )
        await _invoke_with_approvals(client, "room", prompt)
        await client.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def _is_hn_tool_result(event: dict) -> bool:
        if event.get("event") != "tool_result":
            return False
        if event.get("tool") != "run_shell":
            return False
        try:
            payload = json.loads(event.get("result", "{}"))
        except json.JSONDecodeError:
            return False
        stdout = (payload.get("stdout") or "").strip()
        lines = [line for line in stdout.splitlines() if line.strip()]
        return len(lines) >= 10 and payload.get("returncode") == 0

    event = await _wait_for_trace_event(trace_path, _is_hn_tool_result)
    payload = json.loads(event.get("result", "{}"))
    stdout = (payload.get("stdout") or "").strip()
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert payload.get("returncode") == 0
    assert len(lines) >= 10
