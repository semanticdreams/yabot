from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


MessageType = Literal["message", "stop"]
ResponseType = Literal["response", "stopped", "cancelled", "error"]


@dataclass(frozen=True)
class ClientMessage:
    type: MessageType
    id: str
    room_id: str
    text: str | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {"type": self.type, "id": self.id, "room_id": self.room_id}
        if self.text is not None:
            payload["text"] = self.text
        return json.dumps(payload)


@dataclass(frozen=True)
class ServerMessage:
    type: ResponseType
    id: str
    room_id: str
    result: dict[str, Any] | None = None
    ok: bool | None = None
    error: str | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {"type": self.type, "id": self.id, "room_id": self.room_id}
        if self.result is not None:
            payload["result"] = self.result
        if self.ok is not None:
            payload["ok"] = self.ok
        if self.error is not None:
            payload["error"] = self.error
        return json.dumps(payload)


def parse_json(raw: str) -> dict[str, Any]:
    return json.loads(raw)
