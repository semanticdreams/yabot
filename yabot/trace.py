from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceLogger:
    path: Path
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "_lock", threading.Lock())
        self.path.touch(exist_ok=True)

    def log(self, event: str, data: dict[str, Any], context: dict[str, Any] | None = None) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "schema_version": self.schema_version,
            "event": event,
            **(context or {}),
            **data,
        }
        line = json.dumps(payload, ensure_ascii=True, default=str)
        lock: threading.Lock = getattr(self, "_lock")
        with lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
