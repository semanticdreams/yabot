import json
from typing import Any, Awaitable, Callable, Dict, List

from openai import AsyncOpenAI

from .tools import TOOLS


class LLMClient:
    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)

    async def create_message(
        self, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None
    ):
        assert model, "model must be set"
        assert isinstance(messages, list), "messages must be a list"
        tool_choice = "auto"
        tools_payload = tools if tools is not None else TOOLS
        if tools is not None and not tools:
            tools_payload = None
            tool_choice = "none"
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tool_choice": tool_choice,
        }
        if tools_payload is not None:
            kwargs["tools"] = tools_payload
        response = await self.client.chat.completions.create(**kwargs)
        assert response.choices, "LLM response missing choices"
        assert response.choices[0].message is not None, "LLM response missing message"
        return response.choices[0].message

    async def create_message_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> Dict[str, Any]:
        assert model, "model must be set"
        assert isinstance(messages, list), "messages must be a list"
        tool_choice = "auto"
        tools_payload = tools if tools is not None else TOOLS
        if tools is not None and not tools:
            tools_payload = None
            tool_choice = "none"
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tool_choice": tool_choice,
            "stream": True,
        }
        if tools_payload is not None:
            kwargs["tools"] = tools_payload
        response = await self.client.chat.completions.create(**kwargs)
        assert response is not None, "LLM stream response is None"
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
        content = "".join(content_chunks) or None
        if tool_calls:
            for call in tool_calls:
                arguments = (call.get("function") or {}).get("arguments") or ""
                if arguments:
                    try:
                        json.loads(arguments)
                    except json.JSONDecodeError:
                        message = await self.create_message(model, messages, tools=tools)
                        return {
                            "role": message.role,
                            "content": message.content,
                            "tool_calls": getattr(message, "tool_calls", None),
                        }
        if content is None and not tool_calls:
            message = await self.create_message(model, messages, tools=tools)
            return {"role": message.role, "content": message.content, "tool_calls": getattr(message, "tool_calls", None)}
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message
