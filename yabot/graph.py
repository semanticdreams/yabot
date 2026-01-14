import asyncio
import contextvars
import json
import uuid
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .agent_tools import META_AGENT_TOOLS
from .commands import (
    agent_active_conv,
    agent_recent_tool_calls,
    agent_record_tool_calls,
    agent_set_model,
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
    agents: Dict[str, Dict[str, Any]]
    active_agent: str
    approvals: Dict[str, Any]
    trace: Dict[str, Any]
    plan: List[str]
    plan_steps: List[str]
    plan_index: int


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
        meta_system_prompt: str | None = None,
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
        self.meta_system_prompt = meta_system_prompt
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

    def _ensure_system_prompt(self, conv: Dict[str, Any], system_prompt: str | None) -> None:
        if not system_prompt:
            return
        messages = conv.get("messages", [])
        for msg in messages:
            if msg.get("role") == "system" and msg.get("content") == system_prompt:
                return
        conv["messages"] = [{"role": "system", "content": system_prompt}] + list(messages)

    def _system_prompt_for_agent(self, agent_name: str) -> str | None:
        if agent_name == "meta":
            return self.meta_system_prompt
        return self.system_prompt

    def _tools_for_agent(self, agent_name: str) -> List[dict[str, Any]]:
        if agent_name == "meta":
            return list(META_AGENT_TOOLS)
        return self.base_tools + self.skill_tools

    def _tool_calls_to_dicts(self, tool_calls: Any) -> List[dict[str, Any]]:
        normalized: List[dict[str, Any]] = []
        for call in tool_calls or []:
            if hasattr(call, "model_dump"):
                data = call.model_dump()
            elif isinstance(call, dict):
                data = dict(call)
            else:
                fn = getattr(call, "function", None)
                data = {
                    "id": getattr(call, "id", ""),
                    "function": {
                        "name": getattr(fn, "name", ""),
                        "arguments": getattr(fn, "arguments", "{}"),
                    },
                }
            if "type" not in data:
                data["type"] = "function"
            name = (data.get("function") or {}).get("name", "")
            assert name, "tool_call missing function name"
            normalized.append(data)
        return normalized

    def _tool_call_notices(self, tool_calls: List[dict[str, Any]]) -> List[str]:
        notices: List[str] = []
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or "unknown"
            raw_args = function.get("arguments") or "{}"
            notices.append(f"[system] Tool call: {name} {raw_args}")
        return notices

    @staticmethod
    def _is_complex_task(text: str) -> bool:
        tokens = len(text.split())
        if tokens >= 24:
            return True
        if text.count("\n") >= 2:
            return True
        if text.count(".") >= 2:
            return True
        keywords = (" and ", " then ", " also ", " plus ", " after ", " before ", " while ")
        lower = f" {text.lower()} "
        return any(k in lower for k in keywords)

    @staticmethod
    def _parse_plan(text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines()]
        steps: List[str] = []
        for line in lines:
            if not line:
                continue
            if line.startswith("- "):
                steps.append(line[2:].strip())
                continue
            if line[0].isdigit():
                parts = line.split(".", 1)
                if len(parts) == 2 and parts[1].strip():
                    steps.append(parts[1].strip())
                    continue
            steps.append(line)
        return steps[:8]

    async def _auto_plan(
        self,
        incoming: str,
        model: str,
        trace_ctx: dict[str, Any],
    ) -> List[str]:
        planner_prompt = (
            "You are a planner. Return a concise todo list for the task.\n"
            "Use 3-7 bullets, each starting with '- '. No extra text."
        )
        if self.tracer:
            self.tracer.log("plan_request", {"model": model, "text": incoming}, context=trace_ctx)
        message = await self.llm.create_message(
            model,
            [{"role": "system", "content": planner_prompt}, {"role": "user", "content": incoming}],
            tools=[],
        )
        content = (getattr(message, "content", None) or "").strip()
        steps = self._parse_plan(content) if content else []
        if self.tracer:
            self.tracer.log("plan_created", {"steps": steps, "count": len(steps)}, context=trace_ctx)
        return steps

    async def _run_plan_steps(
        self,
        state: Dict[str, Any],
        model: str,
        conv: Dict[str, Any],
        trace_ctx: dict[str, Any],
        agents_loaded: List[str],
        agent_name: str,
        tools: List[dict[str, Any]],
        start_index: int | None = None,
    ) -> tuple[List[str], dict[str, Any] | None]:
        steps = list(state.get("plan_steps") or [])
        if not steps:
            return [], None
        index = start_index if start_index is not None else int(state.get("plan_index", 0))
        total = len(steps)
        responses: List[str] = []
        stream_callback = self._stream_callback.get()

        while index < total:
            step = steps[index]
            header = f"[system] Step {index + 1}/{total}: {step}"
            if stream_callback:
                await stream_callback(header + "\n")
            responses.append(header)

            user_msg = {"role": "user", "content": f"Execute step {index + 1}/{total}: {step}"}
            conv_messages = conv.get("messages", [])
            conv_messages.append(user_msg)
            conv["messages"] = trim_messages(conv_messages, model, self.max_turns)

            step_responses, new_messages, pending_next, tool_notices = await self._run_llm_loop(
                state,
                model,
                conv.get("messages", []),
                trace_ctx,
                agents_loaded,
                agent_name,
                tools,
            )
            if tool_notices:
                responses.extend(tool_notices)
            responses.extend(step_responses)

            if pending_next:
                state["approvals"]["pending"] = pending_next
                state["plan_index"] = index
                return responses, pending_next

            if new_messages is not None:
                conv_messages = conv.get("messages", [])
                conv_messages.extend(new_messages)
                conv["messages"] = trim_messages(conv_messages, model, self.max_turns)

            index += 1
            state["plan_index"] = index

        state.pop("plan_steps", None)
        state.pop("plan_index", None)
        return responses, None

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            role = message.get("role", "assistant")
            assert role in {"assistant", "system", "user", "tool"}, f"invalid role: {role}"
            msg: dict[str, Any] = {
                "role": role,
                "content": message.get("content"),
            }
            tool_calls = message.get("tool_calls")
            if tool_calls:
                msg["tool_calls"] = self._tool_calls_to_dicts(tool_calls)
            return msg
        msg: dict[str, Any] = {"role": message.role, "content": message.content}
        assert msg["role"] in {"assistant", "system", "user", "tool"}, f"invalid role: {msg['role']}"
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            msg["tool_calls"] = self._tool_calls_to_dicts(tool_calls)
        return msg

    @staticmethod
    def _assistant_without_tool_calls(message: dict[str, Any] | None) -> dict[str, Any] | None:
        if not message:
            return None
        if message.get("role") != "assistant":
            return message
        if not message.get("tool_calls"):
            return message
        content = message.get("content")
        if not content:
            return None
        return {"role": "assistant", "content": content}

    def _normalize_messages_for_llm(self, messages: List[dict[str, Any]]) -> List[dict[str, Any]]:
        normalized: List[dict[str, Any]] = []
        index = 0
        total = len(messages)
        while index < total:
            msg = messages[index]
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                assert tool_call_id, "tool message missing tool_call_id"
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = self._tool_calls_to_dicts(msg.get("tool_calls"))
                expected_ids = [c.get("id") for c in tool_calls if c.get("id")]
                tool_block: List[dict[str, Any]] = []
                tool_ids: set[str] = set()
                j = index + 1
                while j < total and messages[j].get("role") == "tool":
                    tool_block.append(messages[j])
                    tool_call_id = messages[j].get("tool_call_id")
                    if tool_call_id:
                        tool_ids.add(tool_call_id)
                    j += 1
                if expected_ids and all(tid in tool_ids for tid in expected_ids):
                    copy = dict(msg)
                    copy["tool_calls"] = tool_calls
                    normalized.append(copy)
                    normalized.extend(tool_block)
                else:
                    stripped = self._assistant_without_tool_calls(msg)
                    if stripped:
                        normalized.append(stripped)
                index = j
                continue
            if msg.get("role") == "tool":
                index += 1
                continue
            normalized.append(msg)
            index += 1
        return normalized

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
            if self.skills.is_skill_tool(name) or name in {
                "ask_user",
                "agent_ask",
                "agent_set_model",
                "agent_recent_tool_calls",
            }:
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
        self,
        state: Dict[str, Any],
        agent_name: str,
        tool_calls: List[dict[str, Any]],
        trace_ctx: dict[str, Any],
    ) -> tuple[List[dict[str, Any]], List[dict[str, Any]], List[str]]:
        tool_messages: List[dict[str, Any]] = []
        system_messages: List[dict[str, Any]] = []
        result_notices: List[str] = []
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            assert name, "tool call missing name"
            assert self.skills.is_skill_tool(name) or name in {
                "ask_user",
                "agent_ask",
                "agent_set_model",
                "agent_recent_tool_calls",
                "list_dir",
                "read_file",
                "write_file",
                "create_dir",
                "get_skills_dir",
                "run_shell",
            }, f"unknown tool: {name}"
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
                elif name == "agent_set_model":
                    target = str(arguments.get("agent", "main")).strip().lower()
                    model = str(arguments.get("model", "")).strip()
                    if not target or not model:
                        result = "ERROR: agent and model are required"
                    else:
                        updated = agent_set_model(state, target, model, self.available_models)
                        result = (
                            f"Model set to `{model}` for `{target}`."
                            if updated
                            else f"ERROR: unknown model `{model}` or agent `{target}`"
                        )
                        if self.tracer:
                            self.tracer.log(
                                "agent_set_model",
                                {"agent": target, "model": model, "ok": updated},
                                context=trace_ctx,
                            )
                elif name == "agent_recent_tool_calls":
                    target = str(arguments.get("agent", "main")).strip().lower()
                    try:
                        limit = int(arguments.get("limit", 10) or 10)
                    except (TypeError, ValueError):
                        limit = 10
                    history = agent_recent_tool_calls(state, target, limit=limit)
                    result = json.dumps(history)
                    if self.tracer:
                        self.tracer.log(
                            "agent_recent_tool_calls",
                            {"agent": target, "count": len(history)},
                            context=trace_ctx,
                        )
                elif name == "agent_ask":
                    target = str(arguments.get("agent", "main")).strip().lower()
                    text = str(arguments.get("text", "")).strip()
                    if not text:
                        result = "ERROR: text is required"
                    else:
                        result = await self._agent_ask(state, target, text, trace_ctx)
                else:
                    result = await execute_tool_async(name, arguments)
                if name == "run_shell":
                    try:
                        payload = json.loads(result)
                    except json.JSONDecodeError:
                        payload = {}
                    if payload.get("error"):
                        result_notices.append(f"[system] Shell error: {payload['error']}")
                    else:
                        returncode = payload.get("returncode")
                        stderr = (payload.get("stderr") or "").strip()
                        stdout = (payload.get("stdout") or "").strip()
                        if isinstance(returncode, int) and returncode != 0:
                            detail = stderr or stdout or "no output"
                            result_notices.append(
                                f"[system] Shell command failed (exit {returncode}): {detail}"
                            )
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
        return tool_messages, system_messages, result_notices

    # shell execution lives in tools.run_shell and is dispatched via execute_tool_async

    async def _agent_ask(
        self,
        state: Dict[str, Any],
        agent_name: str,
        text: str,
        trace_ctx: dict[str, Any],
    ) -> str:
        conv_id, conv = agent_active_conv(state, agent_name)
        if not conv_id or not conv:
            return f"ERROR: unknown agent `{agent_name}`"
        model = conv.get("model", self.default_model)
        self._ensure_system_prompt(conv, self._system_prompt_for_agent(agent_name))
        conv_messages = conv.get("messages", [])
        conv_messages.append({"role": "user", "content": text})
        conv["messages"] = trim_messages(conv_messages, model, self.max_turns)
        if self.tracer:
            self.tracer.log(
                "agent_invoke",
                {"agent": agent_name, "model": model, "text": text},
                context=trace_ctx,
            )
        message = await self.llm.create_message(model, conv.get("messages", []), tools=[])
        assistant_message = self._message_to_dict(message)
        stripped = self._assistant_without_tool_calls(assistant_message)
        if stripped:
            conv_messages = conv.get("messages", [])
            conv_messages.append(stripped)
            conv["messages"] = trim_messages(conv_messages, model, self.max_turns)
        response_text = (assistant_message.get("content") or "").strip()
        if not response_text:
            response_text = "…(no output)"
        if self.tracer:
            self.tracer.log(
                "agent_response",
                {"agent": agent_name, "response": response_text},
                context=trace_ctx,
            )
        return response_text

    async def _run_llm_loop(
        self,
        state: Dict[str, Any],
        model: str,
        messages: List[dict[str, Any]],
        trace_ctx: dict[str, Any],
        agents_loaded: List[str],
        agent_name: str,
        tools: List[dict[str, Any]],
        initial_tool_calls: List[dict[str, Any]] | None = None,
        initial_assistant: dict[str, Any] | None = None,
    ) -> tuple[List[str], List[dict[str, Any]] | None, dict[str, Any] | None, List[str]]:
        working_messages = list(messages)
        new_messages: List[dict[str, Any]] = []
        tool_notices: List[str] = []
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
                            "messages": self._normalize_messages_for_llm(working_messages),
                            "tools": [t.get("function", {}).get("name") for t in tools],
                        },
                        context=trace_ctx,
                    )
                stream_callback = self._stream_callback.get()
                safe_messages = self._normalize_messages_for_llm(working_messages)
                if stream_callback:
                    message = await self.llm.create_message_stream(
                        model,
                        safe_messages,
                        tools=tools,
                        on_token=stream_callback,
                    )
                else:
                    message = await self.llm.create_message(
                        model, safe_messages, tools=tools
                    )
                assistant_message = self._message_to_dict(message)
                if self.tracer:
                    self.tracer.log("llm_response", {"message": assistant_message}, context=trace_ctx)
                new_messages.append(assistant_message)
                working_messages.append(assistant_message)
                if isinstance(message, dict):
                    raw_tool_calls = message.get("tool_calls")
                else:
                    raw_tool_calls = getattr(message, "tool_calls", None)
                tool_calls = self._tool_calls_to_dicts(raw_tool_calls)
                if self.tracer and tool_calls:
                    self.tracer.log("tool_calls", {"calls": tool_calls}, context=trace_ctx)
                if tool_calls:
                    agent_record_tool_calls(state, agent_name, tool_calls)

            if tool_calls:
                notices = self._tool_call_notices(tool_calls)
                if notices:
                    tool_notices.extend(notices)
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
                            "agent": agent_name,
                        }, tool_notices

                missing = self._first_missing_approval(state, tool_calls)
                if missing:
                    stream_callback = self._stream_callback.get()
                    prompt = self._approval_prompt(missing)
                    if stream_callback:
                        await stream_callback(prompt + "\n")
                    if self.tracer:
                        self.tracer.log("approval_request", {"request": missing}, context=trace_ctx)
                    return [prompt], None, {
                        "request": missing,
                        "assistant": assistant_message,
                        "tool_calls": tool_calls,
                        "agent": agent_name,
                    }, tool_notices

                agent_paths = self._agents_paths_for_tool_calls(tool_calls)
                if agent_paths:
                    injected = self._inject_agents_messages(agents_loaded, working_messages, agent_paths)
                    if injected:
                        new_messages.extend(injected)

                tool_messages, system_messages, result_notices = await self._execute_tool_calls(
                    state, agent_name, tool_calls, trace_ctx
                )
                new_messages.extend(tool_messages)
                working_messages.extend(tool_messages)
                if system_messages:
                    new_messages.extend(system_messages)
                    working_messages.extend(system_messages)
                if result_notices:
                    tool_notices.extend(result_notices)
                tool_calls = None
                continue

            final_text = (working_messages[-1].get("content") or "").strip()
            if final_text:
                responses = [p.strip() for p in final_text.split("\n\n") if p.strip()]
            else:
                responses = ["…(no output)"]
            if self.tracer:
                self.tracer.log("response_final", {"responses": responses}, context=trace_ctx)
            return responses, new_messages, None, tool_notices

    async def _process_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        incoming = (state.get("incoming") or "").strip()
        responses: List[str] = []
        ensure_state(state, self.default_model)
        trace_ctx = dict(state.get("trace") or {})

        active_agent = state.get("active_agent", "meta")
        conv_id, conv = room_active_conv(state)
        model = conv.get("model", self.default_model)
        agents_loaded = conv.setdefault("agents_loaded", [])
        trace_ctx.update({"conv_id": conv_id, "model": model, "agent": active_agent})
        self._ensure_system_prompt(conv, self._system_prompt_for_agent(active_agent))
        conv_messages = conv.get("messages", [])
        self._inject_agents_messages(agents_loaded, conv_messages, [Path(os.getcwd())])
        conv["messages"] = conv_messages

        if self.tracer:
            self.tracer.log("incoming", {"text": incoming}, context=trace_ctx)

        pending = state["approvals"].get("pending")
        if pending:
            pending_agent = pending.get("agent", active_agent)
            pending_tools = self._tools_for_agent(pending_agent)
            pending_prompt = self._system_prompt_for_agent(pending_agent)
            pending_trace = dict(trace_ctx)
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
                conv_id, conv = agent_active_conv(state, pending_agent)
                model = conv.get("model", self.default_model)
                agents_loaded = conv.setdefault("agents_loaded", [])
                self._ensure_system_prompt(conv, pending_prompt)
                pending_trace.update({"conv_id": conv_id, "model": model, "agent": pending_agent})
                base_messages = list(conv.get("messages", []))
                if assistant_message:
                    base_messages.append(assistant_message)
                base_messages.append(tool_message)
                responses, new_messages, pending_next, tool_notices = await self._run_llm_loop(
                    state,
                    model,
                    base_messages,
                    pending_trace,
                    agents_loaded,
                    pending_agent,
                    pending_tools,
                )
                if tool_notices:
                    responses = tool_notices + responses
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
                    self.tracer.log("approval_response", {"request": request, "approved": True}, context=pending_trace)
                state["approvals"]["pending"] = None
                conv_id, conv = agent_active_conv(state, pending_agent)
                model = conv.get("model", self.default_model)
                agents_loaded = conv.setdefault("agents_loaded", [])
                self._ensure_system_prompt(conv, pending_prompt)
                pending_trace.update({"conv_id": conv_id, "model": model, "agent": pending_agent})
                responses, new_messages, pending_next, tool_notices = await self._run_llm_loop(
                    state,
                    model,
                    conv.get("messages", []),
                    pending_trace,
                    agents_loaded,
                    pending_agent,
                    pending_tools,
                    initial_tool_calls=pending.get("tool_calls"),
                    initial_assistant=pending.get("assistant"),
                )
                if tool_notices:
                    responses = tool_notices + responses
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
                        context=pending_trace,
                    )
                state["approvals"]["pending"] = None
                conv_id, conv = agent_active_conv(state, pending_agent)
                model = conv.get("model", self.default_model)
                agents_loaded = conv.setdefault("agents_loaded", [])
                self._ensure_system_prompt(conv, pending_prompt)
                pending_trace.update({"conv_id": conv_id, "model": model, "agent": pending_agent})
                base_messages = list(conv.get("messages", []))
                assistant_message = pending.get("assistant")
                stripped = self._assistant_without_tool_calls(assistant_message)
                if stripped:
                    base_messages.append(stripped)
                base_messages.append(
                    {
                        "role": "user",
                        "content": f"Approval denied. Feedback: {incoming}",
                    }
                )
                responses, new_messages, pending_next, tool_notices = await self._run_llm_loop(
                    state,
                    model,
                    base_messages,
                    trace_ctx,
                    agents_loaded,
                    pending_agent,
                    pending_tools,
                )
                if tool_notices:
                    responses = tool_notices + responses
                if pending_next:
                    state["approvals"]["pending"] = pending_next
                elif new_messages is not None:
                    conv_messages = list(conv.get("messages", []))
                    stripped = self._assistant_without_tool_calls(assistant_message)
                    if stripped:
                        conv_messages.append(stripped)
                    conv_messages.append(base_messages[-1])
                    conv_messages.extend(new_messages)
                    conv["messages"] = trim_messages(conv_messages, model, self.max_turns)

            state["responses"] = responses
            if pending_next is None and state.get("plan_steps"):
                state["plan_index"] = int(state.get("plan_index", 0)) + 1
                follow_responses, follow_pending = await self._run_plan_steps(
                    state,
                    model,
                    conv,
                    trace_ctx,
                    agents_loaded,
                    pending_agent,
                    pending_tools,
                )
                responses.extend(follow_responses)
                if follow_pending:
                    state["responses"] = responses
                    return state
            state["responses"] = responses
            return state

        parsed = parse_command(incoming)
        if parsed:
            cmd, arg = parsed
            if self.tracer:
                self.tracer.log("command", {"command": cmd, "arg": arg}, context=trace_ctx)
            before_agent = active_agent
            responses = [handle_command(state, cmd, arg, self.available_models, self.default_model)]
            if cmd == "become":
                after_agent = state.get("active_agent", before_agent)
                trace_ctx["agent"] = after_agent
                if self.tracer:
                    self.tracer.log(
                        "agent_switch",
                        {"from": before_agent, "to": after_agent},
                        context=trace_ctx,
                    )
            state["responses"] = responses
            return state

        _, conv = room_active_conv(state)
        model = conv.get("model", self.default_model)
        conv_messages = conv.get("messages", [])
        conv_messages.append({"role": "user", "content": incoming})
        conv["messages"] = trim_messages(conv_messages, model, self.max_turns)

        plan_steps: List[str] = []
        if self._is_complex_task(incoming):
            plan_steps = await self._auto_plan(incoming, model, trace_ctx)
            if plan_steps:
                state["plan"] = plan_steps
                state["plan_steps"] = plan_steps
                state["plan_index"] = 0
                plan_message = {"role": "system", "content": "Planned steps:\n- " + "\n- ".join(plan_steps)}
                conv_messages = conv.get("messages", [])
                conv_messages.append(plan_message)
                conv["messages"] = trim_messages(conv_messages, model, self.max_turns)
                stream_callback = self._stream_callback.get()
                if stream_callback:
                    await stream_callback("[system] Plan:\n- " + "\n- ".join(plan_steps) + "\n")

        if plan_steps:
            tools = self._tools_for_agent(active_agent)
            responses, pending_next = await self._run_plan_steps(
                state, model, conv, trace_ctx, agents_loaded, active_agent, tools, 0
            )
            if pending_next:
                state["responses"] = responses
                return state
        else:
            tools = self._tools_for_agent(active_agent)
            responses, new_messages, pending_next, tool_notices = await self._run_llm_loop(
                state,
                model,
                conv.get("messages", []),
                trace_ctx,
                agents_loaded,
                active_agent,
                tools,
            )
            if tool_notices:
                responses = tool_notices + responses
            if pending_next:
                state["approvals"]["pending"] = pending_next
            elif new_messages is not None:
                conv_messages = conv.get("messages", [])
                conv_messages.extend(new_messages)
                conv["messages"] = trim_messages(conv_messages, model, self.max_turns)

        if plan_steps and not self._stream_callback.get():
            responses = ["Plan:\n- " + "\n- ".join(plan_steps)] + responses

        state["responses"] = responses
        if self.tracer:
            self.tracer.log("invoke_result", {"responses": responses}, context=trace_ctx)
        return state
