from __future__ import annotations

import asyncio
import atexit
import html
import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout

from .cli_runtime import ensure_daemon, spawn_daemon
from .config import load_config
from .interaction import is_stop_command, request_stop
from .remote import RemoteGraphClient
from .runtime import build_graph
from .streams import StreamRegistry
from .util import retry_until_ok


class YabotCLI:
    class DaemonState(str, Enum):
        UNKNOWN = "unknown"
        UNAVAILABLE = "unavailable"
        STARTING = "starting"
        CONNECTED = "connected"
        STARTED = "started"

    def __init__(
        self,
        graph: Any,
        room_id: str = "cli",
        available_models: list[str] | None = None,
        default_model: str | None = None,
        daemon_autostart: bool = False,
        daemon_pid_path: Path | None = None,
        input_fn: Callable[[str], str] | None = None,
        output: TextIO | None = None,
        prompt: str = "yabot> ",
    ) -> None:
        self.graph = graph
        self.room_id = room_id
        self.streams = StreamRegistry()
        self.available_models = available_models or []
        self.default_model = default_model or (self.available_models[0] if self.available_models else "-")
        self.daemon_autostart = daemon_autostart
        self._daemon_autostarted = False
        self._daemon_atexit_registered = False
        self._daemon_pid_path = daemon_pid_path
        self._daemon_retry_task: asyncio.Task[None] | None = None
        self._daemon_retry_stop = asyncio.Event()
        self._daemon_state = self.DaemonState.UNKNOWN
        self._stop_requested = False
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._print_lock = asyncio.Lock()
        self.input_fn = input_fn
        self.output = output or sys.stdout
        self.prompt = prompt
        self._prompt_pt = HTML(f"<skyblue>{html.escape(self.prompt)}</skyblue>")
        self._pt_session: PromptSession | None = None

    def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print("\n[system] Interrupted.", file=self.output)

    async def run_async(self) -> None:
        if self.input_fn is None:
            self._pt_session = PromptSession()
        try:
            await self._print("[system] Welcome to Yabot CLI. Type !help for commands.")
            await self._ensure_remote()
            if self.input_fn is None:
                await self._prompt_toolkit_loop(self._pt_session)
            else:
                await self._input_fn_loop()
        finally:
            await self._drain_pending()
            await self._close_remote()
            await self._print("[system] Shutdown complete.")
            self._pt_session = None

    async def _input_fn_loop(self) -> None:
        while True:
            try:
                line = await asyncio.to_thread(self._call_input)
            except EOFError:
                await self._print("[system] Input closed.")
                return
            if line is None:
                await self._print("[system] Input closed.")
                return
            await self._handle_line(str(line))

    async def _prompt_toolkit_loop(self, session: PromptSession) -> None:
        try:
            with patch_stdout():
                while True:
                    try:
                        line = await session.prompt_async(self._prompt_pt)
                    except EOFError:
                        await self._print("[system] Input closed.")
                        return
                    except KeyboardInterrupt:
                        await self._print("[system] Input cancelled.")
                        continue
                    await self._handle_line(line)
        finally:
            self._pt_session = None

    def _call_input(self) -> str:
        try:
            assert self.input_fn is not None
            return self.input_fn(self.prompt)
        except TypeError:
            assert self.input_fn is not None
            return self.input_fn()  # type: ignore[misc]

    async def _handle_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        await self._print(f"[you] {text}")
        if is_stop_command(text):
            await self._handle_stop()
            return
        self._start_dispatch(text)
        await asyncio.sleep(0)

    def _start_dispatch(self, text: str) -> None:
        task = asyncio.create_task(self._dispatch(text))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _dispatch(self, text: str) -> None:
        await self._print("[system] Processing request…")
        completed = False
        streamed = False
        ended_with_newline = False

        async def on_token(chunk: str) -> None:
            nonlocal streamed, ended_with_newline
            streamed = True
            ended_with_newline = chunk.endswith("\n")
            await self._write_llm(chunk)

        graph_task = asyncio.create_task(self._call_graph(text, on_token))
        self.streams.register(self.room_id, graph_task)
        try:
            result = await graph_task
        except asyncio.CancelledError:
            if not self._stop_requested:
                await self._print("[system] Cancelled.")
            raise
        except Exception as exc:
            await self._print(f"[system] Error: {exc}")
        else:
            if streamed and not ended_with_newline:
                await self._write_llm("\n")
            if not streamed:
                await self._apply_result(result)
            completed = True
        finally:
            self.streams.clear(self.room_id, graph_task)
            self._stop_requested = False
            if completed:
                await self._print("[system] Response complete.")

    async def _call_graph(self, text: str, on_token: Callable[[str], Awaitable[None]]) -> dict[str, Any]:
        assert text, "text must be non-empty"
        assert callable(on_token), "on_token must be callable"
        stream_fn = getattr(self.graph, "ainvoke_stream", None)
        if callable(stream_fn):
            result = await stream_fn(self.room_id, text, on_token=on_token)
        else:
            result = await self.graph.ainvoke(self.room_id, text)
        assert isinstance(result, dict), "graph result must be a dict"
        return result

    async def _apply_result(self, result: dict[str, Any]) -> None:
        for body in result.get("responses", []) or []:
            await self._print(body)

    async def _handle_stop(self) -> None:
        if await request_stop(self.graph, self.streams, self.room_id):
            self._stop_requested = True
            await self._print("[system] Stopping current response…")
        else:
            await self._print("[system] No active response to stop.")

    async def _print(self, text: str) -> None:
        async with self._print_lock:
            if self._pt_session is not None:
                print_formatted_text(self._format_text(text), output=self._pt_session.output)
            else:
                print(text, file=self.output, flush=True)

    async def _write_llm(self, text: str) -> None:
        async with self._print_lock:
            if self._pt_session is not None:
                print_formatted_text(self._format_llm(text), output=self._pt_session.output, end="", flush=True)
            else:
                self.output.write(text)
                self.output.flush()

    def _format_llm(self, text: str) -> HTML:
        escaped = html.escape(text)
        return HTML(f"<ansigreen>{escaped}</ansigreen>")

    def _format_text(self, text: str) -> HTML:
        escaped = html.escape(text)
        if text.startswith("[system] Error:"):
            return HTML(f"<ansired>{escaped}</ansired>")
        if text.startswith("[system]"):
            return HTML(f"<ansibrightblack>{escaped}</ansibrightblack>")
        if text.startswith("[you]"):
            return HTML(f"<ansicyan>{escaped}</ansicyan>")
        return HTML(f"<ansigreen>{escaped}</ansigreen>")

    async def _ensure_remote(self) -> None:
        if not isinstance(self.graph, RemoteGraphClient):
            await self._print("[system] Running locally.")
            return
        try:
            proc = await ensure_daemon(
                self.graph,
                self.daemon_autostart,
                lambda: spawn_daemon(self._daemon_pid_path, os.getpid()),
            )
            if proc is not None:
                self._daemon_autostarted = True
                await self._wait_for_pidfile()
                self._register_daemon_cleanup()
                await self._set_daemon_state(self.DaemonState.STARTED, "[system] Daemon started.")
            else:
                await self._set_daemon_state(self.DaemonState.CONNECTED, "[system] Daemon connected.")
        except Exception as exc:
            await self._set_daemon_state(self.DaemonState.UNAVAILABLE, f"[system] Daemon unavailable: {exc}")
            if self.daemon_autostart:
                await self._set_daemon_state(self.DaemonState.STARTING, "[system] Starting daemon…")
            self._start_daemon_retry()

    async def _close_remote(self) -> None:
        if isinstance(self.graph, RemoteGraphClient):
            await self.graph.close()
        if self._daemon_retry_task is not None:
            self._daemon_retry_task.cancel()
            self._daemon_retry_task = None
        self._daemon_retry_stop.set()
        if self._daemon_autostarted:
            self._stop_daemon_from_pidfile()
            self._daemon_autostarted = False

    async def _drain_pending(self) -> None:
        if not self._pending_tasks:
            return
        await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()

    async def _set_daemon_state(self, state: "YabotCLI.DaemonState", message: str | None = None) -> None:
        if self._daemon_state == state:
            return
        self._daemon_state = state
        if message:
            await self._print(message)

    def _start_daemon_retry(self) -> None:
        if self._daemon_retry_task is not None or not isinstance(self.graph, RemoteGraphClient):
            return
        self._daemon_retry_stop.clear()
        self._daemon_retry_task = asyncio.create_task(self._retry_daemon_connect())

    async def _retry_daemon_connect(self) -> None:
        async def attempt() -> None:
            await self.graph.connect()
            await self._set_daemon_state(self.DaemonState.CONNECTED, "[system] Daemon connected.")

        await retry_until_ok(attempt, delay_seconds=1.0, cancel_event=self._daemon_retry_stop)

    async def _wait_for_pidfile(self, retries: int = 10, delay: float = 0.1) -> None:
        if self._daemon_pid_path is None:
            return
        for _ in range(retries):
            if self._daemon_pid_path.exists():
                return
            await asyncio.sleep(delay)

    def _register_daemon_cleanup(self) -> None:
        if self._daemon_atexit_registered:
            return
        atexit.register(self._stop_daemon_from_pidfile)
        self._daemon_atexit_registered = True

    @staticmethod
    def _terminate_pid(pid: int) -> None:
        try:
            os.kill(pid, 15)
        except OSError:
            return
        for _ in range(10):
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.05)
        try:
            os.kill(pid, 9)
        except OSError:
            return

    def _stop_daemon_from_pidfile(self) -> None:
        if not self._daemon_autostarted:
            return
        if self._daemon_pid_path is None or not self._daemon_pid_path.exists():
            return
        try:
            pid_text, parent_text = self._daemon_pid_path.read_text(encoding="utf-8").splitlines()[:2]
            pid = int(pid_text)
            parent = int(parent_text)
        except Exception:
            return
        if parent != os.getpid():
            return
        self._terminate_pid(pid)
        self._daemon_pid_path.unlink(missing_ok=True)


def run() -> None:
    config = load_config()
    if config.daemon_url:
        graph = RemoteGraphClient(config.daemon_url)
    else:
        graph = build_graph(config)
    cli = YabotCLI(
        graph=graph,
        available_models=config.available_models,
        default_model=config.default_model,
        daemon_autostart=config.cli_daemon_autostart,
        daemon_pid_path=Path(config.data_dir) / "daemon.pid",
    )
    cli.run()
