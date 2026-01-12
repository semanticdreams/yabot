def ask_user(question: str) -> str:
    return f"ASK: {question}"


TOOL = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": "Ask the user a clarification question and wait for their reply.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
            },
            "required": ["question"],
        },
    },
}
