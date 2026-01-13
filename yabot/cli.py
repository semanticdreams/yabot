from __future__ import annotations

import asyncio
import logging
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Static

from .commands import parse_command
from .config import load_config
from .handler import StreamRegistry
from .runtime import build_graph


class ChatLog(VerticalScroll):
    def __init__(self) -> None:
        super().__init__(id="chat-log")
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, text: str) -> None:
        role_class = {"you": "user", "yabot": "yabot", "system": "system"}.get(role.lower(), "system")
        self.messages.append((role, text))
        self.mount(Static(f"{role}: {text}", classes=f"message {role_class}"))
        self.scroll_end(animate=False)


class YabotCLIApp(App):
    CSS = """
    Screen {
        background: #101418;
        color: #e8e6e3;
    }
    #status {
        dock: top;
        padding: 0 1;
        height: 1;
        background: #20262b;
        color: #c7d2da;
    }
    #chat-log {
        padding: 1 2;
        overflow-y: auto;
        height: 1fr;
    }
    .message {
        padding: 0 0 1 0;
    }
    .message.user {
        color: #a7d1ff;
    }
    .message.yabot {
        color: #f5d08a;
    }
    .message.system {
        color: #9fb3c8;
    }
    #chat-input {
        height: 3;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+n", "new_conversation", "New"),
        ("ctrl+r", "reset_conversation", "Reset"),
        ("ctrl+l", "list_conversations", "List"),
        ("ctrl+m", "show_models", "Models"),
        ("ctrl+s", "stop", "Stop"),
        ("ctrl+h", "help", "Help"),
        ("ctrl+q", "quit", "Quit"),
    ]

    status_text = reactive("Ready")
    active_conversation = reactive("-")
    active_model = reactive("-")

    def __init__(
        self,
        graph: Any,
        room_id: str = "cli",
        available_models: list[str] | None = None,
        default_model: str | None = None,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.room_id = room_id
        self.streams = StreamRegistry()
        self.available_models = available_models or []
        self.default_model = default_model or (self.available_models[0] if self.available_models else "-")
        self._stop_requested = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="status")
        with Vertical():
            yield ChatLog()
            yield Input(placeholder="Type a message. Enter to send. !help for commands.", id="chat-input")
        yield Footer()

    def on_mount(self) -> None:
        self.active_model = self.default_model
        self._update_status()
        self.query_one("#chat-input", Input).focus()
        self._append_system("Welcome to Yabot CLI. Type !help for commands.")

    def watch_status_text(self, _old: str, _new: str) -> None:
        self._update_status()

    def watch_active_conversation(self, _old: str, _new: str) -> None:
        self._update_status()

    def watch_active_model(self, _old: str, _new: str) -> None:
        self._update_status()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        self._append_message("You", text)
        parsed = parse_command(text)
        if parsed and parsed[0] == "stop":
            self._handle_stop()
            return
        self._start_dispatch(text)

    def action_new_conversation(self) -> None:
        self._enqueue_command("!new")

    def action_reset_conversation(self) -> None:
        self._enqueue_command("!reset")

    def action_list_conversations(self) -> None:
        self._enqueue_command("!list")

    def action_show_models(self) -> None:
        self._enqueue_command("!models")

    def action_help(self) -> None:
        self._enqueue_command("!help")

    def action_stop(self) -> None:
        self._handle_stop()

    def _enqueue_command(self, command: str) -> None:
        self._append_message("You", command)
        self._start_dispatch(command)

    def _append_message(self, role: str, text: str) -> None:
        self.query_one(ChatLog).add_message(role, text)

    def _append_system(self, text: str) -> None:
        self._append_message("System", text)

    def _update_status(self) -> None:
        status = self.query_one("#status", Static)
        status.update(
            f"Status: {self.status_text} | Conversation: {self.active_conversation} | Model: {self.active_model}"
        )

    def _handle_stop(self) -> None:
        if self.streams.stop(self.room_id):
            self._stop_requested = True
            self._append_system("Stopping current response…")
        else:
            self._append_system("No active response to stop.")

    def _start_dispatch(self, text: str) -> None:
        task = asyncio.create_task(self._dispatch(text))
        self.streams.register(self.room_id, task)

    async def _dispatch(self, text: str) -> None:
        self.status_text = "Thinking…"
        task = asyncio.current_task()
        try:
            result = await self.graph.ainvoke(self.room_id, text)
        except asyncio.CancelledError:
            if not self._stop_requested:
                self._append_system("Cancelled.")
            raise
        except Exception as exc:
            self._append_system(f"Error: {exc}")
        else:
            self._apply_result(result)
        finally:
            if task is not None:
                self.streams.clear(self.room_id, task)
            self._stop_requested = False
            self.status_text = "Ready"

    def _apply_result(self, result: dict[str, Any]) -> None:
        for body in result.get("responses", []) or []:
            self._append_message("Yabot", body)

        active = result.get("active")
        conversations = result.get("conversations", {})
        if active:
            self.active_conversation = active
            conv = conversations.get(active, {})
            model = conv.get("model")
            if model:
                self.active_model = model


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()
    graph = build_graph(config)
    app = YabotCLIApp(
        graph=graph,
        available_models=config.available_models,
        default_model=config.default_model,
    )
    app.run()
