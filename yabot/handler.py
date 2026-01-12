import asyncio
import logging
from typing import List

from nio import InviteMemberEvent, MatrixRoom, MegolmEvent, RoomMessage, UnknownEncryptedEvent

from .commands import CommandProcessor
from .llm import LLMClient
from matrix_bot.matrix import MatrixMessenger
from .state import StateStore


class StreamRegistry:
    def __init__(self) -> None:
        self._active_streams: dict[str, asyncio.Task[str]] = {}

    def register(self, room_id: str, task: asyncio.Task[str]) -> None:
        self._active_streams[room_id] = task

    def clear(self, room_id: str, task: asyncio.Task[str]) -> None:
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

        if await self.commands.try_handle(room_id, text):
            self.logger.info("Handled command room=%s", room_id)
            return

        async with self.state.lock:
            _, conv = self.state.room_active_conv(room_id)
            model = conv.get("model", self.default_model)

            conv_messages = conv.get("messages", [])
            conv_messages.append({"role": "user", "content": text})
            conv["messages"] = self.state.trim_messages(conv_messages, model)

            messages_for_llm: List[dict[str, str]] = list(conv["messages"])

        stream_task = asyncio.create_task(
            self.llm.respond_streaming(
                self.messenger,
                room_id,
                model,
                messages_for_llm,
            )
        )
        self.streams.register(room_id, stream_task)
        self.logger.info("Starting LLM response room=%s model=%s", room_id, model)
        try:
            assistant_text = await stream_task
            self.logger.info("Completed LLM response room=%s", room_id)
        finally:
            self.streams.clear(room_id, stream_task)

        async with self.state.lock:
            _, conv2 = self.state.room_active_conv(room_id)
            conv2_msgs = conv2.get("messages", [])
            conv2_msgs.append({"role": "assistant", "content": assistant_text})
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
