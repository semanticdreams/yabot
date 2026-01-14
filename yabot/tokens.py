from pathlib import Path
from typing import Any, Dict, List
import json

import tiktoken


MODELS_DATA_PATH = Path(__file__).with_name("models.json")
_MODELS_CONTEXT_WINDOWS: dict[str, int] | None = None

TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1
PRIMING_TOKENS = 3


def context_window_for_model(model: str) -> int:
    windows = _context_windows()
    assert model in windows, f"Unknown model context window: {model}"
    return windows[model]


def _context_windows() -> dict[str, int]:
    global _MODELS_CONTEXT_WINDOWS
    if _MODELS_CONTEXT_WINDOWS is None:
        from_file = _load_models_context_windows()
        assert from_file, "models.json is missing or invalid"
        _MODELS_CONTEXT_WINDOWS = dict(from_file)
    return _MODELS_CONTEXT_WINDOWS


def _load_models_context_windows() -> dict[str, int]:
    if not MODELS_DATA_PATH.exists():
        return {}
    try:
        payload = json.loads(MODELS_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    models = _models_from_payload(payload)
    windows: dict[str, int] = {}
    for entry in models:
        model_id = _model_id_from_entry(entry)
        if not model_id:
            continue
        window = _context_window_from_entry(entry)
        if window:
            windows[model_id] = window
    return windows


def _models_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [m for m in payload if isinstance(m, dict)]
    if not isinstance(payload, dict):
        return []
    # Provider-indexed payloads (models.dev/api.json)
    provider_models: list[dict[str, Any]] = []
    for provider in payload.values():
        if not isinstance(provider, dict):
            continue
        models = provider.get("models")
        if isinstance(models, dict):
            provider_models.extend(m for m in models.values() if isinstance(m, dict))
        elif isinstance(models, list):
            provider_models.extend(m for m in models if isinstance(m, dict))
    if provider_models:
        return provider_models
    for key in ("data", "models", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [m for m in value if isinstance(m, dict)]
    return []


def _model_id_from_entry(entry: dict[str, Any]) -> str | None:
    for key in ("id", "name", "model"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _context_window_from_entry(entry: dict[str, Any]) -> int | None:
    limit = entry.get("limit")
    if isinstance(limit, dict):
        for key in ("context", "max_context", "max_context_length"):
            value = limit.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
    for key in (
        "context_length",
        "context_window",
        "context",
        "max_context_length",
        "max_context",
        "max_input_tokens",
    ):
        value = entry.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _reset_models_context_cache_for_tests() -> None:
    global _MODELS_CONTEXT_WINDOWS
    _MODELS_CONTEXT_WINDOWS = None


def output_reserve_tokens(context_window: int) -> int:
    return max(2048, int(context_window * 0.1))


def get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_message_tokens(message: Dict[str, Any], encoding: tiktoken.Encoding) -> int:
    total = TOKENS_PER_MESSAGE
    for key, value in message.items():
        if value is None:
            continue
        total += len(encoding.encode(str(value)))
        if key == "name":
            total += TOKENS_PER_NAME
    return total


def estimate_messages_tokens(messages: List[Dict[str, Any]], encoding: tiktoken.Encoding) -> int:
    if not messages:
        return 0
    total = 0
    for message in messages:
        total += estimate_message_tokens(message, encoding)
    total += PRIMING_TOKENS
    return total
