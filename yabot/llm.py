import asyncio
import json
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI

from matrix_bot.matrix import MatrixMessenger
from .tools import TOOLS, execute_tool


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

    async def _run_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        tool_messages: List[Dict[str, Any]] = []
        for call in tool_calls:
            name = call.function.name
            raw_args = call.function.arguments or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                result = f"ERROR: invalid JSON arguments: {exc}"
            else:
                result = await asyncio.to_thread(execute_tool, name, arguments)
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )
        return tool_messages

    def _message_to_dict(self, message) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_calls:
            msg["tool_calls"] = [
                call.model_dump() if hasattr(call, "model_dump") else call for call in message.tool_calls
            ]
        return msg

    async def respond_with_tools(
        self,
        model: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        working_messages = list(messages)
        new_messages: List[Dict[str, Any]] = []

        while True:
            response = await self.client.chat.completions.create(
                model=model,
                messages=working_messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            message = response.choices[0].message
            message_dict = self._message_to_dict(message)
            new_messages.append(message_dict)
            working_messages.append(message_dict)

            if message.tool_calls:
                tool_messages = await self._run_tool_calls(message.tool_calls)
                new_messages.extend(tool_messages)
                working_messages.extend(tool_messages)
                continue

            final_text = (message.content or "").strip()
            return final_text, new_messages

    async def respond_streaming(
        self,
        messenger: MatrixMessenger,
        room_id: str,
        model: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        sent_parts: List[str] = []
        new_messages: List[Dict[str, Any]] = []

        try:
            await messenger.set_typing(room_id, True)
            final_text, new_messages = await self.respond_with_tools(model, messages)

            if final_text:
                paragraphs = [p.strip() for p in final_text.split("\n\n") if p.strip()]
                for para in paragraphs:
                    await messenger.send_text(room_id, para)
                    sent_parts.append(para)
            else:
                await messenger.send_text(room_id, "…(no output)")

            return "\n\n".join(sent_parts).strip(), new_messages

        except asyncio.CancelledError:
            err = "Cancelled."
            await messenger.send_text(room_id, err)
            return err, [{"role": "assistant", "content": err}]
        except Exception as e:
            err = f"LLM error: {e}"
            await messenger.send_text(room_id, err)
            return err, [{"role": "assistant", "content": err}]
        finally:
            await messenger.set_typing(room_id, False)
