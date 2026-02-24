"""Anthropic Claude provider.

Wraps the ``anthropic`` SDK's async client behind the LLMProvider protocol.
"""

from __future__ import annotations

import time

from cip_protocol.llm.provider import ProviderResponse


class AnthropicProvider:
    """Claude provider using the Anthropic SDK."""

    def __init__(
        self, api_key: str, model: str = "claude-sonnet-4-20250514"
    ) -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> ProviderResponse:
        start = time.monotonic()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=[{"role": "user", "content": user_message}],
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
