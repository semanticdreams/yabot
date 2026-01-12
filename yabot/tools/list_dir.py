import json
import os


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


TOOL = {
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
}
