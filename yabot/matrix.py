import logging
from typing import Dict, Optional

from nio import AsyncClient, AsyncClientConfig, LoginResponse, MembersSyncError, RoomSendError

from .config import Config
from .storage import atomic_write_json, load_json


def load_creds(path: str) -> Optional[Dict[str, str]]:
    data = load_json(path)
    if isinstance(data, dict) and "access_token" in data and "device_id" in data and "user_id" in data:
        return data
    return None


def save_creds(path: str, access_token: str, device_id: str, user_id: str) -> None:
    atomic_write_json(
        path,
        {
            "access_token": access_token,
            "device_id": device_id,
            "user_id": user_id,
        },
    )


async def login_or_restore(config: Config) -> AsyncClient:
    if not config.bot_user:
        raise RuntimeError("MATRIX_USER is not set")

    client_config = AsyncClientConfig(
        encryption_enabled=True,
        store_sync_tokens=True,
    )

    client = AsyncClient(
        config.homeserver,
        config.bot_user,
        store_path=config.nio_store_dir,
        config=client_config,
    )

    creds = load_creds(config.creds_path)
    if creds and creds.get("user_id") == config.bot_user:
        client.access_token = creds["access_token"]
        client.user_id = creds["user_id"]
        client.device_id = creds["device_id"]
        try:
            client.load_store()
        except Exception:
            pass
        return client

    if not config.bot_password:
        raise RuntimeError("No saved creds found and MATRIX_PASSWORD is not set.")

    resp = await client.login(config.bot_password)
    if isinstance(resp, LoginResponse):
        save_creds(config.creds_path, resp.access_token, resp.device_id, resp.user_id)
        return client

    raise RuntimeError(f"Login failed: {resp}")


class MatrixMessenger:
    def __init__(self, client: AsyncClient) -> None:
        self.client = client
        self.logger = logging.getLogger("yabot.matrix")

    async def send_text(self, room_id: str, body: str) -> str:
        return await self._send_message(
            room_id,
            {
                "msgtype": "m.text",
                "body": body,
            },
        )

    async def edit_text(self, room_id: str, original_event_id: str, new_body: str) -> None:
        await self._send_message(
            room_id,
            {
                "msgtype": "m.text",
                "body": new_body,
                "m.new_content": {"msgtype": "m.text", "body": new_body},
                "m.relates_to": {"rel_type": "m.replace", "event_id": original_event_id},
            },
        )

    async def _send_message(self, room_id: str, content: Dict[str, str]) -> str:
        try:
            resp = await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
        except MembersSyncError:
            self.logger.warning("Members not synced; syncing room=%s", room_id)
            await self.client.joined_members(room_id)
            resp = await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
        except Exception as exc:
            self.logger.exception("room_send failed room=%s error=%s", room_id, exc)
            return ""

        if isinstance(resp, RoomSendError):
            self.logger.error(
                "room_send error room=%s errcode=%s message=%s",
                room_id,
                getattr(resp, "errcode", None),
                getattr(resp, "message", None),
            )
            return ""

        return getattr(resp, "event_id", "")


def auto_trust_devices(client: AsyncClient) -> None:
    for device in client.device_store:
        if device.deleted or device.blacklisted or device.verified:
            continue
        client.verify_device(device)
