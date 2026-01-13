import asyncio
import json
import subprocess

from .util import truncate


def run_shell(command: str, workdir: str | None = None) -> str:
    assert command, "command is required"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        payload = {
            "command": command,
            "workdir": workdir,
            "returncode": result.returncode,
            "stdout": truncate(result.stdout),
            "stderr": truncate(result.stderr),
        }
        return json.dumps(payload)
    except Exception as exc:
        return json.dumps({"error": str(exc), "command": command, "workdir": workdir})


async def run_shell_async(command: str, workdir: str | None = None) -> str:
    assert command, "command is required"
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise
        payload = {
            "command": command,
            "workdir": workdir,
            "returncode": proc.returncode,
            "stdout": truncate(stdout.decode(errors="replace")),
            "stderr": truncate(stderr.decode(errors="replace")),
        }
        return json.dumps(payload)
    except Exception as exc:
        return json.dumps({"error": str(exc), "command": command, "workdir": workdir})


TOOL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command and return stdout, stderr, and exit code.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "workdir": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}
