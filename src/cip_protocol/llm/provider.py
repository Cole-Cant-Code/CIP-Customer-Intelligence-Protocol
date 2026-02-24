"""LLM provider protocol -- abstract interface for inner LLM calls.

Defines the contract that every provider (Anthropic, OpenAI, mock) must
satisfy, plus a factory function for instantiation by name.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

HistoryMessage = dict[str, str]


@dataclass
class ProviderResponse:
    """Raw response from an LLM provider.

    Captures the essentials: generated text, token counts for cost
    tracking, the model that produced the output, and wall-clock latency.
    """

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float


@runtime_checkable
class LLMProvider(Protocol):
    """Abstract interface for inner LLM calls.

    Any class that implements ``generate`` with the correct signature
    satisfies this protocol at runtime (thanks to ``runtime_checkable``).
    """

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
        """Yield response content chunks for streaming clients."""
        raise NotImplementedError


def create_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
) -> LLMProvider:
    """Factory function to create an LLM provider by name.

    Args:
        provider_name: "anthropic", "openai", or "mock".
        api_key: API key for the provider.
        model: Model identifier override.

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If *provider_name* is not recognised.
    """
    if provider_name == "anthropic":
        from cip_protocol.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            api_key=api_key, model=model or "claude-sonnet-4-20250514"
        )
    elif provider_name == "openai":
        from cip_protocol.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key, model=model or "gpt-4o")
    elif provider_name == "mock":
        from cip_protocol.llm.providers.mock import MockProvider

        return MockProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
