import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .commands import (
    ensure_state,
    handle_command,
    parse_command,
    room_active_conv,
    trim_messages,
)
from .llm import LLMClient
from .tools import execute_tool


class GraphState(TypedDict, total=False):
    incoming: str
    responses: List[str]
    conversations: Dict[str, Dict[str, Any]]
    active: str
    approvals: Dict[str, Any]


class YabotGraph:
    def __init__(
        self,
        llm: LLMClient,
        default_model: str,
        available_models: List[str],
        max_turns: int,
        checkpointer: Any | None = None,
    ) -> None:
        self.llm = llm
        self.default_model = default_model
        self.available_models = list(available_models)
        self.max_turns = max_turns
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = self._build_graph()

    async def ainvoke(self, room_id: str, text: str) -> Dict[str, Any]:
        return await self.graph.ainvoke(
            {"incoming": text},
            config={"configurable": {"thread_id": room_id}},
        )

    def _build_graph(self):
        builder: StateGraph = StateGraph(GraphState)
        builder.add_node("process", self._process_input)
        builder.set_entry_point("process")
        builder.add_edge("process", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _tool_calls_to_dicts(self, tool_calls: Any) -> List[dict[str, Any]]:
        normalized: List[dict[str, Any]] = []
        for call in tool_calls or []:
            if hasattr(call, "model_dump"):
                data = call.model_dump()
            elif isinstance(call, dict):
                data = call
            else:
                fn = getattr(call, "function", None)
                data = {
                    "id": getattr(call, "id", ""),
                    "function": {
                        "name": getattr(fn, "name", ""),
                        "arguments": getattr(fn, "arguments", "{}"),
                    },
                }
            normalized.append(data)
        return normalized

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": message.role, "content": message.content}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            msg["tool_calls"] = self._tool_calls_to_dicts(tool_calls)
        return msg

    def _required_dir_for_tool(self, name: str, args: dict[str, Any]) -> Path | None:
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return None
        target = Path(path).expanduser().resolve(strict=False)
        if name in {"read_file", "write_file"}:
            return target.parent
        return target

    def _approval_prompt(self, request: dict[str, Any]) -> str:
        kind = request["kind"]
        if kind == "shell":
            command = request["command"]
            workdir = request.get("workdir") or ""
            suffix = f" (workdir: {workdir})" if workdir else ""
            return f"Approve running shell command: `{command}`{suffix}? Reply `y` to allow."
        if kind == "dir":
            return f"Approve access to directory `{request['dir']}` (includes descendants)? Reply `y` to allow."
        return "Approve requested action? Reply `y` to allow."

    def _shell_key(self, command: str, workdir: Optional[str]) -> str:
        return f"{command}\n{workdir or ''}"

    def _is_shell_approved(self, state: Dict[str, Any], command: str, workdir: Optional[str]) -> bool:
        approvals = state["approvals"]
        return self._shell_key(command, workdir) in approvals.get("shell", [])

    def _approve_shell(self, state: Dict[str, Any], command: str, workdir: Optional[str]) -> None:
        approvals = state["approvals"]
        shell_list = approvals.setdefault("shell", [])
        key = self._shell_key(command, workdir)
        if key not in shell_list:
            shell_list.append(key)

    def _is_dir_approved(self, state: Dict[str, Any], path: str) -> bool:
        approvals = state["approvals"]
        dir_list = approvals.get("dirs", [])
        target = Path(path).expanduser().resolve(strict=False)
        for approved in dir_list:
            approved_path = Path(approved).expanduser().resolve(strict=False)
            if target == approved_path or target.is_relative_to(approved_path):
                return True
        return False

    def _approve_dir(self, state: Dict[str, Any], path: str) -> None:
        approvals = state["approvals"]
        dir_list = approvals.setdefault("dirs", [])
        normalized = str(Path(path).expanduser().resolve(strict=False))
        if normalized not in dir_list:
            dir_list.append(normalized)

    def _first_missing_approval(self, state: Dict[str, Any], tool_calls: List[dict[str, Any]]) -> dict[str, Any] | None:
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            if name == "run_shell":
                command = str(args.get("command", ""))
                workdir = args.get("workdir")
                if not self._is_shell_approved(state, command, workdir):
                    return {"kind": "shell", "command": command, "workdir": workdir}

            if name in {"list_dir", "read_file", "write_file", "create_dir"}:
                required_dir = self._required_dir_for_tool(name, args)
                if required_dir and not self._is_dir_approved(state, str(required_dir)):
                    return {"kind": "dir", "dir": str(required_dir)}

        return None

    async def _execute_tool_calls(self, tool_calls: List[dict[str, Any]]) -> List[dict[str, Any]]:
        tool_messages: List[dict[str, Any]] = []
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                result = f"ERROR: invalid JSON arguments: {exc}"
            else:
                result = await asyncio.to_thread(execute_tool, name, arguments)
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result,
                }
            )
        return tool_messages

    async def _run_llm_loop(
        self,
        state: Dict[str, Any],
        model: str,
        messages: List[dict[str, Any]],
        initial_tool_calls: List[dict[str, Any]] | None = None,
        initial_assistant: dict[str, Any] | None = None,
    ) -> tuple[List[str], List[dict[str, Any]] | None, dict[str, Any] | None]:
        working_messages = list(messages)
        new_messages: List[dict[str, Any]] = []
        tool_calls = initial_tool_calls
        assistant_message = initial_assistant

        if tool_calls is not None:
            if assistant_message is None:
                assistant_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            new_messages.append(assistant_message)
            working_messages.append(assistant_message)

        while True:
            if tool_calls is None:
                message = await self.llm.create_message(model, working_messages)
                assistant_message = self._message_to_dict(message)
                new_messages.append(assistant_message)
                working_messages.append(assistant_message)
                tool_calls = self._tool_calls_to_dicts(getattr(message, "tool_calls", None))

            if tool_calls:
                missing = self._first_missing_approval(state, tool_calls)
                if missing:
                    return [self._approval_prompt(missing)], None, {
                        "request": missing,
                        "assistant": assistant_message,
                        "tool_calls": tool_calls,
                    }

                tool_messages = await self._execute_tool_calls(tool_calls)
                new_messages.extend(tool_messages)
                working_messages.extend(tool_messages)
                tool_calls = None
                continue

            final_text = (working_messages[-1].get("content") or "").strip()
            if final_text:
                responses = [p.strip() for p in final_text.split("\n\n") if p.strip()]
            else:
                responses = ["…(no output)"]
            return responses, new_messages, None

    async def _process_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        incoming = (state.get("incoming") or "").strip()
        responses: List[str] = []
        ensure_state(state, self.default_model)

        pending = state["approvals"].get("pending")
        if pending:
            if incoming.strip().lower() == "y":
                request = pending.get("request") or {}
                if request.get("kind") == "shell":
                    self._approve_shell(state, request.get("command", ""), request.get("workdir"))
                elif request.get("kind") == "dir":
                    self._approve_dir(state, request.get("dir", ""))

                state["approvals"]["pending"] = None
                _, conv = room_active_conv(state)
                model = conv.get("model", self.default_model)
                responses, new_messages, pending_next = await self._run_llm_loop(
                    state,
                    model,
                    conv.get("messages", []),
                    initial_tool_calls=pending.get("tool_calls"),
                    initial_assistant=pending.get("assistant"),
                )
                if pending_next:
                    state["approvals"]["pending"] = pending_next
                elif new_messages is not None:
                    conv_messages = conv.get("messages", [])
                    conv_messages.extend(new_messages)
                    conv["messages"] = trim_messages(conv_messages, model, self.max_turns)
            else:
                state["approvals"]["pending"] = None
                responses = ["Cancelled."]

            state["responses"] = responses
            return state

        parsed = parse_command(incoming)
        if parsed:
            cmd, arg = parsed
            responses = [handle_command(state, cmd, arg, self.available_models, self.default_model)]
            state["responses"] = responses
            return state

        _, conv = room_active_conv(state)
        model = conv.get("model", self.default_model)
        conv_messages = conv.get("messages", [])
        conv_messages.append({"role": "user", "content": incoming})
        conv["messages"] = trim_messages(conv_messages, model, self.max_turns)

        responses, new_messages, pending_next = await self._run_llm_loop(
            state,
            model,
            conv.get("messages", []),
        )
        if pending_next:
            state["approvals"]["pending"] = pending_next
        elif new_messages is not None:
            conv_messages = conv.get("messages", [])
            conv_messages.extend(new_messages)
            conv["messages"] = trim_messages(conv_messages, model, self.max_turns)

        state["responses"] = responses
        return state
