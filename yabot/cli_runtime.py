from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Callable

from .remote import RemoteGraphClient


async def ensure_daemon(
    client: RemoteGraphClient,
    autostart: bool,
    spawn: Callable[[], subprocess.Popen[bytes]],
    retries: int = 5,
    delay: float = 0.2,
) -> subprocess.Popen[bytes] | None:
    last_error: Exception | None = None
    spawned = False
    proc: subprocess.Popen[bytes] | None = None
    for _ in range(retries):
        try:
            await client.connect()
            return proc
        except Exception as exc:
            last_error = exc
            if autostart and not spawned:
                proc = spawn()
                spawned = True
            if delay:
                await asyncio.sleep(delay)
    if last_error:
        raise last_error
    return proc


def spawn_daemon() -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-m", "yabot.daemon"])
