from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from cip_protocol.llm.provider import ProviderResponse


class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    @staticmethod
    def _messages(
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in chat_history or []:
            role = item.get("role", "").strip()
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _extract_text_blocks(content_blocks: list[Any] | None) -> str:
        text_chunks: list[str] = []
        for block in content_blocks or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                text_chunks.append(text)
        return "".join(text_chunks)

    async def generate(
        self,
        system_message: str,
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> ProviderResponse:
        start = time.monotonic()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=self._messages(user_message, chat_history),
        )
        return ProviderResponse(
            content=self._extract_text_blocks(getattr(response, "content", None)),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def generate_stream(
        self,
        system_message: str,
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=self._messages(user_message, chat_history),
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except AttributeError:
            # SDK variant without stream helper — fall back to full generate
            response = await self.generate(
                system_message, user_message, chat_history, max_tokens, temperature,
            )
            if response.content:
                yield response.content
