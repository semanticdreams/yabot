import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from matrix_bot.storage import atomic_write_json, load_json
from .tokens import (
    context_window_for_model,
    estimate_message_tokens,
    estimate_messages_tokens,
    get_encoding,
    output_reserve_tokens,
)


@dataclass
class Conversation:
    conv_id: str
    model: str
    messages: List[Dict[str, str]]


State = Dict[str, Any]


class StateStore:
    def __init__(
        self,
        state_path: str,
        default_model: str,
        available_models: List[str],
        max_turns: int,
    ) -> None:
        self.state_path = state_path
        self.default_model = default_model
        self.available_models = set(available_models)
        self.max_turns = max_turns
        self.lock = asyncio.Lock()
        self.state: State = {"rooms": {}}

    async def load(self) -> None:
        loaded = load_json(self.state_path)
        if isinstance(loaded, dict) and "rooms" in loaded:
            self.state = loaded

    async def save(self) -> None:
        async with self.lock:
            atomic_write_json(self.state_path, self.state)

    def ensure_room(self, room_id: str) -> None:
        rooms = self.state.setdefault("rooms", {})
        if room_id not in rooms:
            conv_id = self._new_conv_id()
            rooms[room_id] = {
                "active": conv_id,
                "conversations": {
                    conv_id: {"model": self.default_model, "messages": []}
                },
                "approvals": {"shell": [], "dirs": [], "pending": None},
            }
        else:
            rooms[room_id].setdefault("approvals", {"shell": [], "dirs": [], "pending": None})

    def room_get_pending(self, room_id: str) -> Optional[Dict[str, Any]]:
        self.ensure_room(room_id)
        return self.state["rooms"][room_id].get("approvals", {}).get("pending")

    def room_set_pending(self, room_id: str, pending: Dict[str, Any]) -> None:
        self.ensure_room(room_id)
        self.state["rooms"][room_id].setdefault("approvals", {"shell": [], "dirs": [], "pending": None})
        self.state["rooms"][room_id]["approvals"]["pending"] = pending

    def room_clear_pending(self, room_id: str) -> None:
        self.ensure_room(room_id)
        self.state["rooms"][room_id].setdefault("approvals", {"shell": [], "dirs": [], "pending": None})
        self.state["rooms"][room_id]["approvals"]["pending"] = None

    def room_is_shell_approved(self, room_id: str, command: str, workdir: Optional[str]) -> bool:
        self.ensure_room(room_id)
        approvals = self.state["rooms"][room_id].setdefault("approvals", {"shell": [], "dirs": [], "pending": None})
        key = f"{command}\n{workdir or ''}"
        return key in approvals.get("shell", [])

    def room_approve_shell(self, room_id: str, command: str, workdir: Optional[str]) -> None:
        self.ensure_room(room_id)
        approvals = self.state["rooms"][room_id].setdefault("approvals", {"shell": [], "dirs": [], "pending": None})
        key = f"{command}\n{workdir or ''}"
        shell_list = approvals.setdefault("shell", [])
        if key not in shell_list:
            shell_list.append(key)

    def room_is_dir_approved(self, room_id: str, path: str) -> bool:
        self.ensure_room(room_id)
        approvals = self.state["rooms"][room_id].setdefault("approvals", {"shell": [], "dirs": [], "pending": None})
        dir_list = approvals.get("dirs", [])
        target = Path(path).expanduser().resolve(strict=False)
        for approved in dir_list:
            approved_path = Path(approved).expanduser().resolve(strict=False)
            if target == approved_path or target.is_relative_to(approved_path):
                return True
        return False

    def room_approve_dir(self, room_id: str, path: str) -> None:
        self.ensure_room(room_id)
        approvals = self.state["rooms"][room_id].setdefault("approvals", {"shell": [], "dirs": [], "pending": None})
        dir_list = approvals.setdefault("dirs", [])
        normalized = str(Path(path).expanduser().resolve(strict=False))
        if normalized not in dir_list:
            dir_list.append(normalized)

    def room_active_conv(self, room_id: str) -> Tuple[str, Dict[str, Any]]:
        self.ensure_room(room_id)
        room_state = self.state["rooms"][room_id]
        conv_id = room_state["active"]
        return conv_id, room_state["conversations"][conv_id]

    def room_new_conv(self, room_id: str) -> str:
        self.ensure_room(room_id)
        conv_id = self._new_conv_id()
        self.state["rooms"][room_id]["conversations"][conv_id] = {
            "model": self.default_model,
            "messages": [],
        }
        self.state["rooms"][room_id]["active"] = conv_id
        return conv_id

    def room_list_convs(self, room_id: str) -> List[Tuple[str, str, int, bool]]:
        self.ensure_room(room_id)
        room_state = self.state["rooms"][room_id]
        active = room_state["active"]
        out = []
        for cid, c in room_state["conversations"].items():
            out.append((cid, c.get("model", self.default_model), len(c.get("messages", [])), cid == active))
        out.sort(key=lambda x: (not x[3], x[0]))
        return out

    def room_use_conv(self, room_id: str, conv_id: str) -> bool:
        self.ensure_room(room_id)
        room_state = self.state["rooms"][room_id]
        if conv_id not in room_state["conversations"]:
            return False
        room_state["active"] = conv_id
        return True

    def room_set_model(self, room_id: str, model: str) -> bool:
        if model not in self.available_models:
            return False
        _, conv = self.room_active_conv(room_id)
        conv["model"] = model
        return True

    def room_reset(self, room_id: str) -> None:
        _, conv = self.room_active_conv(room_id)
        conv["messages"] = []

    def trim_messages(self, messages: List[Dict[str, str]], model: str) -> List[Dict[str, str]]:
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        other = [m for m in messages if m.get("role") != "system"]
        keep = other[-(self.max_turns * 2):]
        kept = sys_msgs + keep

        context_window = context_window_for_model(model)
        reserve = output_reserve_tokens(context_window)
        input_budget = max(0, context_window - reserve)

        encoding = get_encoding(model)
        if estimate_messages_tokens(kept, encoding) <= input_budget:
            return kept

        sys_msgs = [m for m in kept if m.get("role") == "system"]
        other = [m for m in kept if m.get("role") != "system"]

        sys_counts = [estimate_message_tokens(m, encoding) for m in sys_msgs]
        other_counts = [estimate_message_tokens(m, encoding) for m in other]
        total = sum(sys_counts) + sum(other_counts)

        while total > input_budget and other:
            total -= other_counts.pop(0)
            other.pop(0)

        while total > input_budget and sys_msgs:
            total -= sys_counts.pop(0)
            sys_msgs.pop(0)

        return sys_msgs + other

    def _new_conv_id(self) -> str:
        return str(uuid.uuid4())[:8]
