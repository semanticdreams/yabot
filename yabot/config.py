import os
from dataclasses import dataclass
from typing import List

from appdirs import user_data_dir
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    homeserver: str
    bot_user: str | None
    bot_password: str | None
    openai_api_key: str
    bot_name: str
    available_models: List[str]
    default_model: str
    max_turns: int
    allowed_users: List[str]
    cross_signing_reset: bool
    data_dir: str
    state_path: str
    creds_path: str
    nio_store_dir: str


def load_config() -> Config:
    load_dotenv()

    homeserver = os.environ.get("MATRIX_HOMESERVER", "https://matrix.org")
    bot_user = os.environ.get("MATRIX_USER")
    bot_password = os.environ.get("MATRIX_PASSWORD")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    bot_name = os.environ.get("BOT_NAME", "yabot")
    allowed_users_raw = os.environ.get("ALLOWED_USERS", "")
    allowed_users = [u.strip() for u in allowed_users_raw.split(",") if u.strip()]
    cross_signing_reset = os.environ.get("CROSS_SIGNING_RESET", "").strip().lower() in {"1", "true", "yes"}

    available_models = [
        "gpt-4o-mini",
        "gpt-5.2",
    ]
    default_model = "gpt-4o-mini"

    max_turns = 30

    data_dir = user_data_dir(bot_name)
    os.makedirs(data_dir, exist_ok=True)

    state_path = os.path.join(data_dir, "state.json")
    creds_path = os.path.join(data_dir, "creds.json")
    nio_store_dir = os.path.join(data_dir, "nio_store")
    os.makedirs(nio_store_dir, exist_ok=True)

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    return Config(
        homeserver=homeserver,
        bot_user=bot_user,
        bot_password=bot_password,
        openai_api_key=openai_api_key,
        bot_name=bot_name,
        available_models=available_models,
        default_model=default_model,
        max_turns=max_turns,
        allowed_users=allowed_users,
        cross_signing_reset=cross_signing_reset,
        data_dir=data_dir,
        state_path=state_path,
        creds_path=creds_path,
        nio_store_dir=nio_store_dir,
    )
