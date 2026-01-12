import json
import subprocess

from .util import truncate


def run_shell(command: str, workdir: str | None = None) -> str:
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
