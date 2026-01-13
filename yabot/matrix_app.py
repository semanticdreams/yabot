import asyncio
import logging

from nio import InviteMemberEvent, MegolmEvent, RoomMessage, SyncResponse, UnknownEncryptedEvent

from .config import load_config
from matrix_bot.cross_signing import CrossSigningManager
from .handler import BotHandler
from .remote import RemoteGraphClient
from .runtime import build_graph
from matrix_bot.matrix import MatrixMessenger, auto_trust_devices, login_or_restore
from .streams import StreamRegistry


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()
    client = await login_or_restore(
        config.homeserver,
        config.bot_user,
        config.bot_password,
        config.creds_path,
        config.nio_store_dir,
    )
    messenger = MatrixMessenger(client)
    streams = StreamRegistry()
    cross_signing = CrossSigningManager(
        config.data_dir,
        reset=config.cross_signing_reset,
        password=config.bot_password,
    )
    if config.daemon_url:
        graph = RemoteGraphClient(config.daemon_url)
        await graph.connect()
    else:
        graph = build_graph(config)
    handler = BotHandler(
        messenger,
        graph,
        streams,
        config.allowed_users,
    )

    client.add_event_callback(handler.on_message, RoomMessage)
    client.add_event_callback(handler.on_decryption_failed, MegolmEvent)
    client.add_event_callback(handler.on_unknown_encrypted, UnknownEncryptedEvent)
    client.add_event_callback(handler.on_invite, InviteMemberEvent)
    async def on_sync(_: SyncResponse) -> None:
        auto_trust_devices(client)
        await cross_signing.ensure_setup(client)
    client.add_response_callback(on_sync, SyncResponse)

    logging.info("Data dir: %s", config.data_dir)
    logging.info("Bot is running (E2EE enabled). Invite it to an encrypted room or DM it.")

    try:
        await client.sync_forever(timeout=30000, full_state=True)
    finally:
        if isinstance(graph, RemoteGraphClient):
            await graph.close()
        await client.close()


def run() -> None:
    asyncio.run(main())
