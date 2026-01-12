from typing import Any, Dict, List

from openai import AsyncOpenAI

from .tools import TOOLS


class LLMClient:
    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)

    async def create_message(self, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None):
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or TOOLS,
            tool_choice="auto",
        )
        return response.choices[0].message
