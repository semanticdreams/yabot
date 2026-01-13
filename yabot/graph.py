import asyncio
import contextvars
import json
import uuid
import os
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
from .skills import SkillRegistry
from .tools import TOOLS
from .tools.registry import execute_tool_async


class GraphState(TypedDict, total=False):
    incoming: str
    responses: List[str]
    conversations: Dict[str, Dict[str, Any]]
    active: str
    approvals: Dict[str, Any]
    trace: Dict[str, Any]


class YabotGraph:
    def __init__(
        self,
        llm: LLMClient,
        default_model: str,
        available_models: List[str],
        max_turns: int,
        skills: SkillRegistry,
        checkpointer: Any | None = None,
        tracer: Any | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.default_model = default_model
        self.available_models = list(available_models)
        self.max_turns = max_turns
        self.skills = skills
        self.base_tools = list(TOOLS)
        self.skill_tools = skills.tool_defs()
        self.checkpointer = checkpointer or MemorySaver()
        self.tracer = tracer
        self.system_prompt = system_prompt
        self._stream_callback: contextvars.ContextVar = contextvars.ContextVar("yabot_stream_callback", default=None)
        self.graph = self._build_graph()

    async def ainvoke(self, room_id: str, text: str) -> Dict[str, Any]:
        trace_ctx = {"trace_id": uuid.uuid4().hex, "room_id": room_id}
        if self.tracer:
            self.tracer.log("invoke", {"text": text}, context=trace_ctx)
        return await self.graph.ainvoke(
            {"incoming": text, "trace": trace_ctx},
            config={"configurable": {"thread_id": room_id}},
        )

    async def ainvoke_stream(
        self,
        room_id: str,
        text: str,
        on_token: Any,
    ) -> Dict[str, Any]:
        token = self._stream_callback.set(on_token)
        try:
            return await self.ainvoke(room_id, text)
        finally:
            self._stream_callback.reset(token)

    def _build_graph(self):
        builder: StateGraph = StateGraph(GraphState)
        builder.add_node("process", self._process_input)
        builder.set_entry_point("process")
        builder.add_edge("process", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _ensure_system_prompt(self, conv: Dict[str, Any]) -> None:
        if not self.system_prompt:
            return
        messages = conv.get("messages", [])
        for msg in messages:
            if msg.get("role") == "system" and msg.get("content") == self.system_prompt:
                return
        conv["messages"] = [{"role": "system", "content": self.system_prompt}] + list(messages)

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
        if isinstance(message, dict):
            msg: dict[str, Any] = {
                "role": message.get("role", "assistant"),
                "content": message.get("content"),
            }
            tool_calls = message.get("tool_calls")
            if tool_calls:
                msg["tool_calls"] = self._tool_calls_to_dicts(tool_calls)
            return msg
        msg: dict[str, Any] = {"role": message.role, "content": message.content}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            msg["tool_calls"] = self._tool_calls_to_dicts(tool_calls)
        return msg

    def _agents_message(self, path: Path, content: str) -> dict[str, Any]:
        return {
            "role": "system",
            "content": f"AGENTS.md instructions from {path}:\n{content}",
        }

    def _read_agents_file(self, directory: Path) -> tuple[Path, str] | None:
        agents_path = directory / "AGENTS.md"
        try:
            content = agents_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return None
        content = content.strip()
        if not content:
            return None
        return agents_path, content

    def _inject_agents_messages(
        self,
        agents_loaded: List[str],
        messages: List[dict[str, Any]],
        paths: List[Path],
    ) -> List[dict[str, Any]]:
        injected: List[dict[str, Any]] = []
        for directory in paths:
            directory = directory.expanduser().resolve(strict=False)
            result = self._read_agents_file(directory)
            if not result:
                continue
            agents_path, content = result
            key = str(agents_path)
            if key in agents_loaded:
                continue
            agents_loaded.append(key)
            msg = self._agents_message(agents_path, content)
            messages.append(msg)
            injected.append(msg)
        return injected

    def _agents_paths_for_tool_calls(self, tool_calls: List[dict[str, Any]]) -> List[Path]:
        paths: List[Path] = []
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                continue

            if name == "run_shell":
                workdir = args.get("workdir")
                if isinstance(workdir, str) and workdir:
                    paths.append(Path(workdir))
                continue

            if name in {"list_dir", "read_file", "write_file", "create_dir"}:
                required_dir = self._required_dir_for_tool(name, args)
                if required_dir:
                    paths.append(required_dir)

        return paths

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
            if self.skills.is_skill_tool(name) or name == "ask_user":
                continue
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

    async def _execute_tool_calls(
        self, tool_calls: List[dict[str, Any]], trace_ctx: dict[str, Any]
    ) -> tuple[List[dict[str, Any]], List[dict[str, Any]]]:
        tool_messages: List[dict[str, Any]] = []
        system_messages: List[dict[str, Any]] = []
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                arguments = {}
                result = f"ERROR: invalid JSON arguments: {exc}"
            else:
                if self.skills.is_skill_tool(name):
                    skill = self.skills.get_by_tool_name(name)
                    if skill:
                        system_messages.append({"role": "system", "content": skill.content})
                        result = f"Skill applied: {skill.name}"
                    else:
                        result = f"ERROR: unknown skill tool {name}"
                else:
                    result = await execute_tool_async(name, arguments)
                if self.tracer:
                    self.tracer.log(
                        "tool_result",
                        {
                            "tool": name,
                            "tool_call_id": call.get("id", ""),
                            "raw_arguments": raw_args,
                            "arguments": arguments,
                            "result": result,
                        },
                        context=trace_ctx,
                    )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result,
                }
            )
        return tool_messages, system_messages

    # shell execution lives in tools.run_shell and is dispatched via execute_tool_async

    async def _run_llm_loop(
        self,
        state: Dict[str, Any],
        model: str,
        messages: List[dict[str, Any]],
        trace_ctx: dict[str, Any],
        agents_loaded: List[str],
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
                if self.tracer:
                    self.tracer.log(
                        "llm_request",
                        {
                            "model": model,
                            "messages": working_messages,
                            "tools": [t.get("function", {}).get("name") for t in self.base_tools + self.skill_tools],
                        },
                        context=trace_ctx,
                    )
                stream_callback = self._stream_callback.get()
                if stream_callback:
                    message = await self.llm.create_message_stream(
                        model,
                        working_messages,
                        tools=self.base_tools + self.skill_tools,
                        on_token=stream_callback,
                    )
                else:
                    message = await self.llm.create_message(
                        model, working_messages, tools=self.base_tools + self.skill_tools
                    )
                assistant_message = self._message_to_dict(message)
                if self.tracer:
                    self.tracer.log("llm_response", {"message": assistant_message}, context=trace_ctx)
                new_messages.append(assistant_message)
                working_messages.append(assistant_message)
                tool_calls = self._tool_calls_to_dicts(getattr(message, "tool_calls", None))
                if self.tracer and tool_calls:
                    self.tracer.log("tool_calls", {"calls": tool_calls}, context=trace_ctx)

            if tool_calls:
                for call in tool_calls:
                    function = call.get("function") or {}
                    name = function.get("name") or ""
                    if name == "ask_user":
                        raw_args = function.get("arguments") or "{}"
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args = {}
                        question = str(args.get("question", "")).strip() or "Can you clarify?"
                        return [question], None, {
                            "request": {"kind": "ask_user", "tool_call_id": call.get("id", "")},
                            "assistant": assistant_message,
                        }

                missing = self._first_missing_approval(state, tool_calls)
                if missing:
                    if self.tracer:
                        self.tracer.log("approval_request", {"request": missing}, context=trace_ctx)
                    return [self._approval_prompt(missing)], None, {
                        "request": missing,
                        "assistant": assistant_message,
                        "tool_calls": tool_calls,
                    }

                agent_paths = self._agents_paths_for_tool_calls(tool_calls)
                if agent_paths:
                    injected = self._inject_agents_messages(agents_loaded, working_messages, agent_paths)
                    if injected:
                        new_messages.extend(injected)

                tool_messages, system_messages = await self._execute_tool_calls(tool_calls, trace_ctx)
                new_messages.extend(tool_messages)
                working_messages.extend(tool_messages)
                if system_messages:
                    new_messages.extend(system_messages)
                    working_messages.extend(system_messages)
                tool_calls = None
                continue

            final_text = (working_messages[-1].get("content") or "").strip()
            if final_text:
                responses = [p.strip() for p in final_text.split("\n\n") if p.strip()]
            else:
                responses = ["…(no output)"]
            if self.tracer:
                self.tracer.log("response_final", {"responses": responses}, context=trace_ctx)
            return responses, new_messages, None

    async def _process_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        incoming = (state.get("incoming") or "").strip()
        responses: List[str] = []
        ensure_state(state, self.default_model)
        trace_ctx = dict(state.get("trace") or {})

        conv_id, conv = room_active_conv(state)
        model = conv.get("model", self.default_model)
        agents_loaded = conv.setdefault("agents_loaded", [])
        trace_ctx.update({"conv_id": conv_id, "model": model})
        self._ensure_system_prompt(conv)
        conv_messages = conv.get("messages", [])
        self._inject_agents_messages(agents_loaded, conv_messages, [Path(os.getcwd())])
        conv["messages"] = conv_messages

        if self.tracer:
            self.tracer.log("incoming", {"text": incoming}, context=trace_ctx)

        pending = state["approvals"].get("pending")
        if pending:
            request = pending.get("request") or {}
            if request.get("kind") == "ask_user":
                tool_call_id = request.get("tool_call_id", "")
                assistant_message = pending.get("assistant")
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": incoming,
                }
                state["approvals"]["pending"] = None
                _, conv = room_active_conv(state)
                model = conv.get("model", self.default_model)
                base_messages = list(conv.get("messages", []))
                if assistant_message:
                    base_messages.append(assistant_message)
                base_messages.append(tool_message)
                responses, new_messages, pending_next = await self._run_llm_loop(
                    state,
                    model,
                    base_messages,
                    trace_ctx,
                    agents_loaded,
                )
                if pending_next:
                    state["approvals"]["pending"] = pending_next
                elif new_messages is not None:
                    conv_messages = list(conv.get("messages", []))
                    if assistant_message:
                        conv_messages.append(assistant_message)
                    conv_messages.append(tool_message)
                    conv_messages.extend(new_messages)
                    conv["messages"] = trim_messages(conv_messages, model, self.max_turns)
            elif incoming.strip().lower() == "y":
                if request.get("kind") == "shell":
                    self._approve_shell(state, request.get("command", ""), request.get("workdir"))
                elif request.get("kind") == "dir":
                    self._approve_dir(state, request.get("dir", ""))

                if self.tracer:
                    self.tracer.log("approval_response", {"request": request, "approved": True}, context=trace_ctx)
                state["approvals"]["pending"] = None
                _, conv = room_active_conv(state)
                model = conv.get("model", self.default_model)
                responses, new_messages, pending_next = await self._run_llm_loop(
                    state,
                    model,
                    conv.get("messages", []),
                    trace_ctx,
                    agents_loaded,
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
                if self.tracer:
                    self.tracer.log(
                        "approval_response",
                        {"request": request, "approved": False, "feedback": incoming},
                        context=trace_ctx,
                    )
                state["approvals"]["pending"] = None
                _, conv = room_active_conv(state)
                model = conv.get("model", self.default_model)
                base_messages = list(conv.get("messages", []))
                assistant_message = pending.get("assistant")
                if assistant_message:
                    base_messages.append(assistant_message)
                base_messages.append(
                    {
                        "role": "user",
                        "content": f"Approval denied. Feedback: {incoming}",
                    }
                )
                responses, new_messages, pending_next = await self._run_llm_loop(
                    state,
                    model,
                    base_messages,
                    trace_ctx,
                    agents_loaded,
                )
                if pending_next:
                    state["approvals"]["pending"] = pending_next
                elif new_messages is not None:
                    conv_messages = list(conv.get("messages", []))
                    if assistant_message:
                        conv_messages.append(assistant_message)
                    conv_messages.append(base_messages[-1])
                    conv_messages.extend(new_messages)
                    conv["messages"] = trim_messages(conv_messages, model, self.max_turns)

            state["responses"] = responses
            return state

        parsed = parse_command(incoming)
        if parsed:
            cmd, arg = parsed
            if self.tracer:
                self.tracer.log("command", {"command": cmd, "arg": arg}, context=trace_ctx)
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
            trace_ctx,
            agents_loaded,
        )
        if pending_next:
            state["approvals"]["pending"] = pending_next
        elif new_messages is not None:
            conv_messages = conv.get("messages", [])
            conv_messages.extend(new_messages)
            conv["messages"] = trim_messages(conv_messages, model, self.max_turns)

        state["responses"] = responses
        if self.tracer:
            self.tracer.log("invoke_result", {"responses": responses}, context=trace_ctx)
        return state
