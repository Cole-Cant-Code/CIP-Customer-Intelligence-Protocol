"""Mock LLM provider for testing.

Returns canned responses and records call metadata so tests can assert
on the system/user messages and invocation count.
"""

from __future__ import annotations

from cip_protocol.llm.provider import ProviderResponse


class MockProvider:
    """Mock provider for testing -- returns a canned response."""

    def __init__(self, response_content: str = "Mock LLM response.") -> None:
        self.response_content = response_content
        self.last_system_message: str = ""
        self.last_user_message: str = ""
        self.call_count: int = 0

    async def generate(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> ProviderResponse:
        self.last_system_message = system_message
        self.last_user_message = user_message
        self.call_count += 1
        return ProviderResponse(
            content=self.response_content,
            input_tokens=len(system_message.split()) + len(user_message.split()),
            output_tokens=len(self.response_content.split()),
            model="mock",
            latency_ms=0.0,
        )
