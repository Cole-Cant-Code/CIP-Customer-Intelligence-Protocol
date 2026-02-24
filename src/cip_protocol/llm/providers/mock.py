from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cip_protocol.llm.provider import ProviderResponse


class MockProvider:
    def __init__(self, response_content: str = "Mock LLM response.") -> None:
        self.response_content = response_content
        self.last_system_message: str = ""
        self.last_user_message: str = ""
        self.last_chat_history: list[dict[str, str]] = []
        self.call_count: int = 0

    async def generate(
        self,
        system_message: str,
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> ProviderResponse:
        self.last_system_message = system_message
        self.last_user_message = user_message
        self.last_chat_history = chat_history or []
        self.call_count += 1
        return ProviderResponse(
            content=self.response_content,
            input_tokens=len(system_message.split()) + len(user_message.split()),
            output_tokens=len(self.response_content.split()),
            model="mock",
            latency_ms=0.0,
        )

    async def generate_stream(
        self,
        system_message: str,
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        self.last_system_message = system_message
        self.last_user_message = user_message
        self.last_chat_history = chat_history or []
        self.call_count += 1

        for token in self.response_content.split():
            await asyncio.sleep(0)
            yield token + " "
