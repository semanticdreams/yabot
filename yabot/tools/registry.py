from typing import Any, Dict

from .ask_user import TOOL as ASK_USER_TOOL, ask_user
from .create_dir import TOOL as CREATE_DIR_TOOL, create_dir
from .get_skills_dir import TOOL as GET_SKILLS_DIR_TOOL, get_skills_dir
from .list_dir import TOOL as LIST_DIR_TOOL, list_dir
from .read_file import TOOL as READ_FILE_TOOL, read_file
from .run_shell import TOOL as RUN_SHELL_TOOL, run_shell
from .write_file import TOOL as WRITE_FILE_TOOL, write_file


TOOLS = [
    ASK_USER_TOOL,
    LIST_DIR_TOOL,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    CREATE_DIR_TOOL,
    GET_SKILLS_DIR_TOOL,
    RUN_SHELL_TOOL,
]


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    if name == "list_dir":
        return list_dir(str(arguments.get("path", "")))
    if name == "ask_user":
        return ask_user(str(arguments.get("question", "")))
    if name == "read_file":
        return read_file(str(arguments.get("path", "")))
    if name == "write_file":
        return write_file(str(arguments.get("path", "")), str(arguments.get("content", "")))
    if name == "create_dir":
        return create_dir(str(arguments.get("path", "")), bool(arguments.get("exist_ok", False)))
    if name == "get_skills_dir":
        return get_skills_dir(str(arguments.get("app_name", "yabot")))
    if name == "run_shell":
        return run_shell(str(arguments.get("command", "")), arguments.get("workdir"))
    return f"ERROR: unknown tool {name}"
