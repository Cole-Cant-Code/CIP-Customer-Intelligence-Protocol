"""Tests for LLM providers."""

from types import SimpleNamespace

import pytest

from cip_protocol.llm.provider import LLMProvider, create_provider
from cip_protocol.llm.providers.anthropic import AnthropicProvider
from cip_protocol.llm.providers.mock import MockProvider


class TestMockProvider:
    @pytest.mark.asyncio
    async def test_returns_canned_response(self):
        provider = MockProvider(response_content="Hello!")
        response = await provider.generate(
            system_message="sys", user_message="usr"
        )
        assert response.content == "Hello!"
        assert response.model == "mock"
        assert response.latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_records_messages(self):
        provider = MockProvider()
        await provider.generate(
            system_message="system prompt",
            user_message="user query",
            chat_history=[{"role": "assistant", "content": "previous"}],
        )
        assert provider.last_system_message == "system prompt"
        assert provider.last_user_message == "user query"
        assert provider.last_chat_history == [{"role": "assistant", "content": "previous"}]
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_streaming_chunks(self):
        provider = MockProvider(response_content="one two three")
        chunks = []
        async for chunk in provider.generate_stream(
            system_message="sys",
            user_message="usr",
        ):
            chunks.append(chunk)
        assert "".join(chunks).strip() == "one two three"

    def test_satisfies_protocol(self):
        provider = MockProvider()
        assert isinstance(provider, LLMProvider)


class TestProviderFactory:
    def test_create_mock(self):
        provider = create_provider("mock")
        assert isinstance(provider, MockProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_provider("nonexistent")


class TestAnthropicProviderParsing:
    def test_extract_text_blocks_ignores_non_text_items(self):
        blocks = [
            SimpleNamespace(type="tool_use"),
            SimpleNamespace(type="text", text="Hello"),
            SimpleNamespace(type="text", text=" world"),
        ]
        assert AnthropicProvider._extract_text_blocks(blocks) == "Hello world"

    def test_extract_text_blocks_handles_missing_or_empty(self):
        assert AnthropicProvider._extract_text_blocks([]) == ""
        assert AnthropicProvider._extract_text_blocks(None) == ""
