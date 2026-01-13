import logging
from typing import Any

from nio import InviteMemberEvent, MatrixRoom, MegolmEvent, RoomMessage, UnknownEncryptedEvent

from .interaction import dispatch_graph, is_stop_command, request_stop
from matrix_bot.matrix import MatrixMessenger
from .streams import StreamRegistry


class BotHandler:
    def __init__(
        self,
        messenger: MatrixMessenger,
        graph: Any,
        streams: StreamRegistry,
        allowed_users: list[str],
    ) -> None:
        self.messenger = messenger
        self.graph = graph
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

        if is_stop_command(text):
            if await request_stop(self.graph, self.streams, room_id):
                await self.messenger.send_text(room_id, "Stopping current response…")
            else:
                await self.messenger.send_text(room_id, "No active response to stop.")
            return

        try:
            result = await dispatch_graph(self.graph, self.streams, room_id, text)
            self.logger.info("Completed graph response room=%s", room_id)
        except Exception as exc:
            self.logger.exception("Graph error room=%s error=%s", room_id, exc)
            await self.messenger.send_text(room_id, "Error while processing request.")
            return

        for body in result.get("responses", []) or []:
            await self.messenger.send_text(room_id, body)

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
