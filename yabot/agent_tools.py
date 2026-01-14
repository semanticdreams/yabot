META_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "agent_ask",
            "description": "Ask another agent to respond to a user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": ["main", "meta"]},
                    "text": {"type": "string"},
                },
                "required": ["agent", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_set_model",
            "description": "Change the active model for another agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": ["main", "meta"]},
                    "model": {"type": "string"},
                },
                "required": ["agent", "model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_recent_tool_calls",
            "description": "List recent tool calls made by another agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": ["main", "meta"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["agent"],
            },
        },
    },
]
