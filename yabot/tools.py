import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict


MAX_OUTPUT_CHARS = 8000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...(truncated)"


def list_dir(path: str) -> str:
    try:
        entries = []
        for name in sorted(os.listdir(path)):
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path):
                kind = "dir"
            elif os.path.isfile(full_path):
                kind = "file"
            else:
                kind = "other"
            entries.append({"name": name, "type": kind})
        payload = {"path": path, "entries": entries}
        return json.dumps(payload)
    except Exception as exc:
        return json.dumps({"error": str(exc), "path": path})


def read_file(path: str) -> str:
    try:
        data = Path(path).read_bytes()
        text = data.decode("utf-8", errors="replace")
        return _truncate(text)
    except Exception as exc:
        return f"ERROR: {exc}"


def write_file(path: str, content: str) -> str:
    try:
        parent = Path(path).parent
        if parent and not parent.exists():
            return f"ERROR: parent directory does not exist: {parent}"
        Path(path).write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"ERROR: {exc}"


def create_dir(path: str, exist_ok: bool) -> str:
    try:
        Path(path).mkdir(parents=True, exist_ok=exist_ok)
        return f"OK: created {path}"
    except Exception as exc:
        return f"ERROR: {exc}"


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
            "stdout": _truncate(result.stdout),
            "stderr": _truncate(result.stderr),
        }
        return json.dumps(payload)
    except Exception as exc:
        return json.dumps({"error": str(exc), "command": command, "workdir": workdir})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries in a directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a text file to disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_dir",
            "description": "Create a directory, optionally allowing it to exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "exist_ok": {"type": "boolean", "default": False},
                },
                "required": ["path"],
            },
        },
    },
    {
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
    },
]


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    if name == "list_dir":
        return list_dir(str(arguments.get("path", "")))
    if name == "read_file":
        return read_file(str(arguments.get("path", "")))
    if name == "write_file":
        return write_file(str(arguments.get("path", "")), str(arguments.get("content", "")))
    if name == "create_dir":
        return create_dir(str(arguments.get("path", "")), bool(arguments.get("exist_ok", False)))
    if name == "run_shell":
        return run_shell(str(arguments.get("command", "")), arguments.get("workdir"))
    return f"ERROR: unknown tool {name}"
