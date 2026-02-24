from __future__ import annotations

import time
from collections.abc import AsyncIterator

from cip_protocol.llm.provider import ProviderResponse


class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    @staticmethod
    def _messages(
        system_message: str,
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_message}]
        for item in chat_history or []:
            role = item.get("role", "").strip()
            content = item.get("content", "")
            if role in {"system", "user", "assistant", "tool"} and content:
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
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=self._messages(system_message, user_message, chat_history),
        )
        choice = response.choices[0] if response.choices else None
        usage = response.usage
        return ProviderResponse(
            content=choice.message.content or "" if choice else "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
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
        stream = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=self._messages(system_message, user_message, chat_history),
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
