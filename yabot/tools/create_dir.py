from pathlib import Path


def create_dir(path: str, exist_ok: bool) -> str:
    try:
        Path(path).mkdir(parents=True, exist_ok=exist_ok)
        return f"OK: created {path}"
    except Exception as exc:
        return f"ERROR: {exc}"


TOOL = {
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
}
