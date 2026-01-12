from appdirs import user_data_dir
from pathlib import Path


def get_skills_dir(app_name: str = "yabot") -> str:
    return str(Path(user_data_dir(app_name)) / "skills")


TOOL = {
    "type": "function",
    "function": {
        "name": "get_skills_dir",
        "description": "Return the local skills directory path.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "default": "yabot"},
            },
        },
    },
}
