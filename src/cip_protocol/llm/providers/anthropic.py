"""Anthropic Claude provider.

Wraps the ``anthropic`` SDK's async client behind the LLMProvider protocol.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from cip_protocol.llm.provider import ProviderResponse


class AnthropicProvider:
    """Claude provider using the Anthropic SDK."""

    def __init__(
        self, api_key: str, model: str = "claude-sonnet-4-20250514"
    ) -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    @staticmethod
    def _build_messages(
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build Anthropic messages (user/assistant turns only)."""
        messages: list[dict[str, str]] = []
        for item in chat_history or []:
            role = item.get("role", "").strip()
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        return messages

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
            messages=self._build_messages(user_message, chat_history),
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        content = response.content[0].text if response.content else ""
        return ProviderResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
            latency_ms=elapsed_ms,
        )

    async def generate_stream(
        self,
        system_message: str,
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Yield streaming text chunks from Anthropic messages API."""
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=self._build_messages(user_message, chat_history),
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except AttributeError:
            # Fallback for SDK variants without stream helper.
            response = await self.generate(
                system_message=system_message,
                user_message=user_message,
                chat_history=chat_history,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if response.content:
                yield response.content
