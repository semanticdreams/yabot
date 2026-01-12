import asyncio
import json
import logging
from pathlib import Path
from typing import Any, List

from nio import InviteMemberEvent, MatrixRoom, MegolmEvent, RoomMessage, UnknownEncryptedEvent

from .commands import CommandProcessor
from .llm import LLMClient
from matrix_bot.matrix import MatrixMessenger
from .state import StateStore
from .tools import execute_tool


class StreamRegistry:
    def __init__(self) -> None:
        self._active_streams: dict[str, asyncio.Task[tuple[str | None, list[dict[str, Any]] | None, bool]]] = {}

    def register(
        self,
        room_id: str,
        task: asyncio.Task[tuple[str | None, list[dict[str, Any]] | None, bool]],
    ) -> None:
        self._active_streams[room_id] = task

    def clear(
        self,
        room_id: str,
        task: asyncio.Task[tuple[str | None, list[dict[str, Any]] | None, bool]],
    ) -> None:
        if self._active_streams.get(room_id) is task:
            self._active_streams.pop(room_id, None)

    def stop(self, room_id: str) -> bool:
        task = self._active_streams.get(room_id)
        if not task or task.done():
            return False
        task.cancel()
        return True


class BotHandler:
    def __init__(
        self,
        state: StateStore,
        messenger: MatrixMessenger,
        llm: LLMClient,
        commands: CommandProcessor,
        default_model: str,
        streams: StreamRegistry,
        allowed_users: list[str],
    ) -> None:
        self.state = state
        self.messenger = messenger
        self.llm = llm
        self.commands = commands
        self.default_model = default_model
        self.streams = streams
        self.allowed_users = set(allowed_users)
        self.logger = logging.getLogger("yabot.handler")

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

    def _first_missing_approval(self, room_id: str, tool_calls: List[dict[str, Any]]) -> dict[str, Any] | None:
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
                if not self.state.room_is_shell_approved(room_id, command, workdir):
                    return {"kind": "shell", "command": command, "workdir": workdir}

            if name in {"list_dir", "read_file", "write_file", "create_dir"}:
                required_dir = self._required_dir_for_tool(name, args)
                if required_dir and not self.state.room_is_dir_approved(room_id, str(required_dir)):
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
        room_id: str,
        model: str,
        working_messages: List[dict[str, Any]],
        new_messages: List[dict[str, Any]],
        initial_tool_calls: List[dict[str, Any]] | None = None,
    ) -> tuple[str | None, List[dict[str, Any]] | None, bool]:
        sent_parts: List[str] = []
        tool_calls = initial_tool_calls

        try:
            await self.messenger.set_typing(room_id, True)
            while True:
                if tool_calls is None:
                    message = await self.llm.create_message(model, working_messages)
                    message_dict = self._message_to_dict(message)
                    new_messages.append(message_dict)
                    working_messages.append(message_dict)
                    tool_calls = self._tool_calls_to_dicts(getattr(message, "tool_calls", None))
                if tool_calls:
                    missing = self._first_missing_approval(room_id, tool_calls)
                    if missing:
                        async with self.state.lock:
                            self.state.room_set_pending(
                                room_id,
                                {
                                    "model": model,
                                    "messages": working_messages,
                                    "new_messages": new_messages,
                                    "tool_calls": tool_calls,
                                    "request": missing,
                                },
                            )
                        await self.state.save()
                        await self.messenger.send_text(room_id, self._approval_prompt(missing))
                        return None, None, True

                    tool_messages = await self._execute_tool_calls(tool_calls)
                    new_messages.extend(tool_messages)
                    working_messages.extend(tool_messages)
                    tool_calls = None
                    continue

                final_text = (working_messages[-1].get("content") or "").strip()
                if final_text:
                    paragraphs = [p.strip() for p in final_text.split("\n\n") if p.strip()]
                    for para in paragraphs:
                        await self.messenger.send_text(room_id, para)
                        sent_parts.append(para)
                else:
                    await self.messenger.send_text(room_id, "…(no output)")

                return "\n\n".join(sent_parts).strip(), new_messages, False

        except asyncio.CancelledError:
            err = "Cancelled."
            await self.messenger.send_text(room_id, err)
            return err, [{"role": "assistant", "content": err}], False
        except Exception as e:
            err = f"LLM error: {e}"
            await self.messenger.send_text(room_id, err)
            return err, [{"role": "assistant", "content": err}], False
        finally:
            await self.messenger.set_typing(room_id, False)

    async def _resume_pending(self, room_id: str, pending: dict[str, Any]) -> None:
        model = pending["model"]
        working_messages = list(pending["messages"])
        new_messages = list(pending["new_messages"])
        tool_calls = list(pending["tool_calls"])

        stream_task = asyncio.create_task(
            self._run_llm_loop(room_id, model, working_messages, new_messages, initial_tool_calls=tool_calls)
        )
        self.streams.register(room_id, stream_task)
        try:
            assistant_text, assistant_messages, pending_set = await stream_task
        finally:
            self.streams.clear(room_id, stream_task)

        if pending_set or assistant_messages is None:
            return

        async with self.state.lock:
            _, conv = self.state.room_active_conv(room_id)
            conv_msgs = conv.get("messages", [])
            conv_msgs.extend(assistant_messages)
            conv["messages"] = self.state.trim_messages(conv_msgs, model)
        await self.state.save()

    async def on_message(self, room: MatrixRoom, event: RoomMessage) -> None:
        self.logger.info(
            "Message received room=%s sender=%s type=%s decrypted=%s verified=%s",
            room.room_id,
            event.sender,
            type(event).__name__,
            getattr(event, "decrypted", None),
            getattr(event, "verified", None),
        )
        if event.sender == self.messenger.client.user_id:
            self.logger.info("Ignoring self message sender=%s", event.sender)
            return
        if self.allowed_users and event.sender not in self.allowed_users:
            self.logger.warning("Sender not allowed sender=%s", event.sender)
            return

        room_id = room.room_id
        text = (getattr(event, "body", "") or "").strip()
        if not text:
            self.logger.info("Ignoring non-text message type=%s", type(event).__name__)
            return

        async with self.state.lock:
            self.state.ensure_room(room_id)
            pending = self.state.room_get_pending(room_id)

        if pending:
            if text.strip().lower() == "y":
                request = pending.get("request") or {}
                kind = request.get("kind")
                async with self.state.lock:
                    if kind == "shell":
                        self.state.room_approve_shell(room_id, request.get("command", ""), request.get("workdir"))
                    elif kind == "dir":
                        self.state.room_approve_dir(room_id, request.get("dir", ""))
                    self.state.room_clear_pending(room_id)
                await self.state.save()
                await self._resume_pending(room_id, pending)
            else:
                async with self.state.lock:
                    self.state.room_clear_pending(room_id)
                await self.state.save()
                await self.messenger.send_text(room_id, "Cancelled.")
            return

        if await self.commands.try_handle(room_id, text):
            self.logger.info("Handled command room=%s", room_id)
            return

        async with self.state.lock:
            _, conv = self.state.room_active_conv(room_id)
            model = conv.get("model", self.default_model)

            conv_messages = conv.get("messages", [])
            conv_messages.append({"role": "user", "content": text})
            conv["messages"] = self.state.trim_messages(conv_messages, model)

            messages_for_llm: List[dict[str, Any]] = list(conv["messages"])

        stream_task = asyncio.create_task(
            self._run_llm_loop(room_id, model, list(messages_for_llm), [], initial_tool_calls=None)
        )
        self.streams.register(room_id, stream_task)
        self.logger.info("Starting LLM response room=%s model=%s", room_id, model)
        try:
            assistant_text, assistant_messages, pending_set = await stream_task
            self.logger.info("Completed LLM response room=%s", room_id)
        finally:
            self.streams.clear(room_id, stream_task)

        if pending_set or assistant_messages is None:
            return

        async with self.state.lock:
            _, conv2 = self.state.room_active_conv(room_id)
            conv2_msgs = conv2.get("messages", [])
            conv2_msgs.extend(assistant_messages)
            conv2["messages"] = self.state.trim_messages(conv2_msgs, model)

        await self.state.save()

    async def on_decryption_failed(self, room: MatrixRoom, event: MegolmEvent) -> None:
        self.logger.warning(
            "Unable to decrypt message room=%s sender=%s session_id=%s",
            room.room_id,
            event.sender,
            getattr(event, "session_id", None),
        )

    async def on_unknown_encrypted(self, room: MatrixRoom, event: UnknownEncryptedEvent) -> None:
        self.logger.warning(
            "Unknown encrypted event room=%s sender=%s",
            room.room_id,
            event.sender,
        )

    async def on_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        self.logger.info("Invite received room=%s sender=%s", room.room_id, event.sender)
        try:
            await self.messenger.client.join(room.room_id)
            self.logger.info("Joined room=%s", room.room_id)
        except Exception as exc:
            self.logger.exception("Failed to join room=%s error=%s", room.room_id, exc)
