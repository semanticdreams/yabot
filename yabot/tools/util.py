MAX_OUTPUT_CHARS = 8000


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...(truncated)"
