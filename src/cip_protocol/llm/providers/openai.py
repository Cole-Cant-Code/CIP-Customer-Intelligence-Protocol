"""OpenAI GPT provider.

Wraps the ``openai`` SDK's async client behind the LLMProvider protocol.
"""

from __future__ import annotations

import time

from cip_protocol.llm.provider import ProviderResponse


class OpenAIProvider:
    """OpenAI provider using the OpenAI SDK."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> ProviderResponse:
        start = time.monotonic()
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0] if response.choices else None
        content = choice.message.content or "" if choice else ""
        usage = response.usage
        return ProviderResponse(
            content=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
            latency_ms=elapsed_ms,
        )
