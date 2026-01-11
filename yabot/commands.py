import re
from collections.abc import Callable

from .matrix import MatrixMessenger
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

        if cmd in ("help", "h", "?"):
            await self.messenger.send_text(room_id, help_text())
            return True

        if cmd == "models":
            await self.messenger.send_text(
                room_id,
                "Available models:\n" + "\n".join(f"- {x}" for x in self.available_models),
            )
            return True

        if cmd == "model":
            if not arg:
                await self.messenger.send_text(
                    room_id,
                    "Usage: !model <name>\n" + "\n".join(self.available_models),
                )
                return True
            async with self.state.lock:
                ok = self.state.room_set_model(room_id, arg)
            if not ok:
                await self.messenger.send_text(room_id, f"Unknown model `{arg}`.\nUse !models.")
                return True
            await self.state.save()
            await self.messenger.send_text(room_id, f"Model set to `{arg}` for this room’s active conversation.")
            return True

        if cmd == "new":
            async with self.state.lock:
                cid = self.state.room_new_conv(room_id)
            await self.state.save()
            await self.messenger.send_text(room_id, f"Started new conversation for this room: `{cid}`")
            return True

        if cmd == "list":
            async with self.state.lock:
                items = self.state.room_list_convs(room_id)
            lines = ["Conversations for this room:"]
            for cid, model, nmsgs, is_active in items:
                mark = " (active)" if is_active else ""
                lines.append(f"- `{cid}` [{model}] msgs={nmsgs}{mark}")
            await self.messenger.send_text(room_id, "\n".join(lines))
            return True

        if cmd == "use":
            if not arg:
                await self.messenger.send_text(room_id, "Usage: !use <conversation_id>")
                return True
            async with self.state.lock:
                ok = self.state.room_use_conv(room_id, arg)
            if not ok:
                await self.messenger.send_text(room_id, f"No conversation `{arg}` in this room. Use !list.")
                return True
            await self.state.save()
            await self.messenger.send_text(room_id, f"Switched active conversation to `{arg}` for this room.")
            return True

        if cmd == "reset":
            async with self.state.lock:
                self.state.room_reset(room_id)
            await self.state.save()
            await self.messenger.send_text(room_id, "Cleared memory for this room’s active conversation.")
            return True

        if cmd == "stop":
            if self.stop_stream(room_id):
                await self.messenger.send_text(room_id, "Stopping current response…")
            else:
                await self.messenger.send_text(room_id, "No active response to stop.")
            return True

        await self.messenger.send_text(room_id, "Unknown command.\n" + help_text())
        return True
