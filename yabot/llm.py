from typing import Any, Awaitable, Callable, Dict, List

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

    async def create_message_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or TOOLS,
            tool_choice="auto",
            stream=True,
        )
        content_chunks: List[str] = []
        tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
        async for event in response:
            choices = getattr(event, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if not delta:
                continue
            content = getattr(delta, "content", None)
            if content:
                content_chunks.append(content)
                if on_token:
                    await on_token(content)
            tool_deltas = getattr(delta, "tool_calls", None) or []
            for tool_delta in tool_deltas:
                index = int(getattr(tool_delta, "index", 0))
                entry = tool_calls_by_index.setdefault(
                    index, {"id": "", "function": {"name": "", "arguments": ""}}
                )
                tool_id = getattr(tool_delta, "id", None)
                if tool_id:
                    entry["id"] = tool_id
                function = getattr(tool_delta, "function", None)
                if function:
                    name = getattr(function, "name", None)
                    if name:
                        entry["function"]["name"] = name
                    arguments = getattr(function, "arguments", None)
                    if arguments:
                        entry["function"]["arguments"] += arguments
        tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
        message: Dict[str, Any] = {"role": "assistant", "content": "".join(content_chunks) or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message
