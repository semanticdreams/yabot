import re
from collections.abc import Callable

from matrix_bot.matrix import MatrixMessenger
from .state import StateStore


CMD_RE = re.compile(r"^!(\w+)(?:\s+(.*))?$")


def help_text() -> str:
    return (
        "Commands:\n"
        "!models\n"
        "!model <name>\n"
        "!new\n"
        "!list\n"
        "!use <id>\n"
        "!reset\n"
        "!stop\n"
    )


class CommandProcessor:
    def __init__(
        self,
        state: StateStore,
        messenger: MatrixMessenger,
        available_models: list[str],
        default_model: str,
        stop_stream: Callable[[str], bool],
    ) -> None:
        self.state = state
        self.messenger = messenger
        self.available_models = available_models
        self.default_model = default_model
        self.stop_stream = stop_stream

    async def try_handle(self, room_id: str, text: str) -> bool:
        m = CMD_RE.match(text)
        if not m:
            return False

        cmd = m.group(1).lower()
        arg = (m.group(2) or "").strip()

        handlers = {
            "help": self._cmd_help,
            "h": self._cmd_help,
            "?": self._cmd_help,
            "models": self._cmd_models,
            "model": self._cmd_model,
            "new": self._cmd_new,
            "list": self._cmd_list,
            "use": self._cmd_use,
            "reset": self._cmd_reset,
            "stop": self._cmd_stop,
        }

        handler = handlers.get(cmd)
        if not handler:
            await self.messenger.send_text(room_id, "Unknown command.\n" + help_text())
            return True

        await handler(room_id, arg)
        return True

    async def _cmd_help(self, room_id: str, _arg: str) -> None:
        await self.messenger.send_text(room_id, help_text())

    async def _cmd_models(self, room_id: str, _arg: str) -> None:
        await self.messenger.send_text(
            room_id,
            "Available models:\n" + "\n".join(f"- {x}" for x in self.available_models),
        )

    async def _cmd_model(self, room_id: str, arg: str) -> None:
        if not arg:
            await self.messenger.send_text(
                room_id,
                "Usage: !model <name>\n" + "\n".join(self.available_models),
            )
            return
        async with self.state.lock:
            ok = self.state.room_set_model(room_id, arg)
        if not ok:
            await self.messenger.send_text(room_id, f"Unknown model `{arg}`.\nUse !models.")
            return
        await self.state.save()
        await self.messenger.send_text(room_id, f"Model set to `{arg}` for this room’s active conversation.")

    async def _cmd_new(self, room_id: str, _arg: str) -> None:
        async with self.state.lock:
            cid = self.state.room_new_conv(room_id)
        await self.state.save()
        await self.messenger.send_text(room_id, f"Started new conversation for this room: `{cid}`")

    async def _cmd_list(self, room_id: str, _arg: str) -> None:
        async with self.state.lock:
            items = self.state.room_list_convs(room_id)
        lines = ["Conversations for this room:"]
        for cid, model, nmsgs, is_active in items:
            mark = " (active)" if is_active else ""
            lines.append(f"- `{cid}` [{model}] msgs={nmsgs}{mark}")
        await self.messenger.send_text(room_id, "\n".join(lines))

    async def _cmd_use(self, room_id: str, arg: str) -> None:
        if not arg:
            await self.messenger.send_text(room_id, "Usage: !use <conversation_id>")
            return
        async with self.state.lock:
            ok = self.state.room_use_conv(room_id, arg)
        if not ok:
            await self.messenger.send_text(room_id, f"No conversation `{arg}` in this room. Use !list.")
            return
        await self.state.save()
        await self.messenger.send_text(room_id, f"Switched active conversation to `{arg}` for this room.")

    async def _cmd_reset(self, room_id: str, _arg: str) -> None:
        async with self.state.lock:
            self.state.room_reset(room_id)
        await self.state.save()
        await self.messenger.send_text(room_id, "Cleared memory for this room’s active conversation.")

    async def _cmd_stop(self, room_id: str, _arg: str) -> None:
        if self.stop_stream(room_id):
            await self.messenger.send_text(room_id, "Stopping current response…")
        else:
            await self.messenger.send_text(room_id, "No active response to stop.")
