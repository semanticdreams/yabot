from typing import Any, Dict, List

from openai import AsyncOpenAI

from .tools import TOOLS


class LLMClient:
    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)

    async def _iter_stream_content(self, model: str, messages: List[Dict[str, Any]]):
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
            except Exception:
                piece = None
            if piece:
                yield piece

    async def stream_reply(self, model: str, messages: List[Dict[str, Any]]) -> str:
        final_text_parts: List[str] = []
        async for piece in self._iter_stream_content(model, messages):
            final_text_parts.append(piece)
        return "".join(final_text_parts).strip()

    async def create_message(self, model: str, messages: List[Dict[str, Any]]):
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        return response.choices[0].message
