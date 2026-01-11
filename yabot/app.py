import asyncio
import logging

from nio import InviteMemberEvent, MegolmEvent, RoomMessage, SyncResponse, UnknownEncryptedEvent

from .commands import CommandProcessor
from .config import load_config
from .cross_signing import CrossSigningManager
from .handler import BotHandler, StreamRegistry
from .llm import LLMClient
from .matrix import MatrixMessenger, auto_trust_devices, login_or_restore
from .state import StateStore


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()
    state = StateStore(
        config.state_path,
        config.default_model,
        config.available_models,
        config.max_turns,
    )
    await state.load()

    client = await login_or_restore(config)
    messenger = MatrixMessenger(client)
    streams = StreamRegistry()
    cross_signing = CrossSigningManager(
        config.data_dir,
        reset=config.cross_signing_reset,
        password=config.bot_password,
    )
    llm = LLMClient(
        api_key=config.openai_api_key,
    )
    commands = CommandProcessor(
        state,
        messenger,
        config.available_models,
        config.default_model,
        streams.stop,
    )
    handler = BotHandler(
        state,
        messenger,
        llm,
        commands,
        config.default_model,
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
        await client.close()


def run() -> None:
    asyncio.run(main())
