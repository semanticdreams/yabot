from typing import Any, Dict, List

import tiktoken


CONTEXT_WINDOWS = {
    "gpt-5.2": 400000,
    "gpt-4o-mini": 200000,
}
DEFAULT_CONTEXT_WINDOW = min(CONTEXT_WINDOWS.values())

TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1
PRIMING_TOKENS = 3


def context_window_for_model(model: str) -> int:
    return CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)


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
