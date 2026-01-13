from pathlib import Path


def write_file(path: str, content: str) -> str:
    assert path, "path is required"
    try:
        parent = Path(path).parent
        if parent and not parent.exists():
            return f"ERROR: parent directory does not exist: {parent}"
        Path(path).write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"ERROR: {exc}"


TOOL = {
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
}
