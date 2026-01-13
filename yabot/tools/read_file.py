from pathlib import Path

from .util import truncate


def read_file(path: str) -> str:
    assert path, "path is required"
    try:
        data = Path(path).read_bytes()
        text = data.decode("utf-8", errors="replace")
        return truncate(text)
    except Exception as exc:
        return f"ERROR: {exc}"


TOOL = {
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
}
