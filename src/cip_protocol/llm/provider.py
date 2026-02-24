from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

HistoryMessage = dict[str, str]


@dataclass
class ProviderResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float


@runtime_checkable
class LLMProvider(Protocol):
    async def generate(
        self,
        system_message: str,
        user_message: str,
        chat_history: list[HistoryMessage] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> ProviderResponse: ...

    async def generate_stream(
        self,
        system_message: str,
        user_message: str,
        chat_history: list[HistoryMessage] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        raise NotImplementedError


def create_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
) -> LLMProvider:
    if provider_name == "anthropic":
        from cip_protocol.llm.providers.anthropic import AnthropicProvider
        return AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-20250514")

    if provider_name == "openai":
        from cip_protocol.llm.providers.openai import OpenAIProvider
        return OpenAIProvider(api_key=api_key, model=model or "gpt-4o")

    if provider_name == "mock":
        from cip_protocol.llm.providers.mock import MockProvider
        return MockProvider()

    raise ValueError(f"Unknown LLM provider: {provider_name}")
