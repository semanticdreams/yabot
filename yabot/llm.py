import asyncio
from typing import Dict, List

from openai import AsyncOpenAI

from matrix_bot.matrix import MatrixMessenger


class LLMClient:
    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)

    async def _iter_stream_content(self, model: str, messages: List[Dict[str, str]]):
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

    async def stream_reply(self, model: str, messages: List[Dict[str, str]]) -> str:
        final_text_parts: List[str] = []
        async for piece in self._iter_stream_content(model, messages):
            final_text_parts.append(piece)
        return "".join(final_text_parts).strip()

    async def respond_streaming(
        self,
        messenger: MatrixMessenger,
        room_id: str,
        model: str,
        messages: List[Dict[str, str]],
    ) -> str:
        await messenger.send_text(room_id, "⏳ Thinking…")

        buf: List[str] = []
        sent_parts: List[str] = []

        try:
            async for piece in self._iter_stream_content(model, messages):
                buf.append(piece)
                text = "".join(buf)
                paragraphs = text.split("\n\n")
                if len(paragraphs) > 1:
                    for part in paragraphs[:-1]:
                        para = part.strip()
                        if para:
                            await messenger.send_text(room_id, para)
                            sent_parts.append(para)
                    buf = [paragraphs[-1]]

            final_text = "".join(buf).strip()

            if final_text:
                await messenger.send_text(room_id, final_text)
                sent_parts.append(final_text)
            else:
                await messenger.send_text(room_id, "…(no output)")

            return "\n\n".join(sent_parts).strip()

        except asyncio.CancelledError:
            err = "Cancelled."
            await messenger.send_text(room_id, err)
            return err
        except Exception as e:
            err = f"LLM error: {e}"
            await messenger.send_text(room_id, err)
            return err
