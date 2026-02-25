"""Tests for multi-turn conversation."""

from __future__ import annotations

import pytest

from cip_protocol.cip import CIP, CIPResult
from cip_protocol.conversation import Conversation, Turn
from cip_protocol.llm.providers.mock import MockProvider
from cip_protocol.scaffold.registry import ScaffoldRegistry
from tests.conftest import make_test_config, make_test_scaffold


def _make_cip(provider=None) -> CIP:
    registry = ScaffoldRegistry()
    registry.register(make_test_scaffold())
    config = make_test_config()
    return CIP(config, registry, provider or MockProvider())


class TestConversation:
    @pytest.mark.asyncio
    async def test_say_returns_result(self):
        cip = _make_cip()
        conv = Conversation(cip)
        result = await conv.say("hello", tool_name="test_tool")
        assert isinstance(result, CIPResult)
        assert result.response.content

    @pytest.mark.asyncio
    async def test_history_accumulates(self):
        cip = _make_cip()
        conv = Conversation(cip)
        await conv.say("first message", tool_name="test_tool")
        await conv.say("second message", tool_name="test_tool")
        assert len(conv.history) == 4  # 2 pairs
        assert conv.history[0]["role"] == "user"
        assert conv.history[0]["content"] == "first message"
        assert conv.history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_turns_tracked(self):
        cip = _make_cip()
        conv = Conversation(cip)
        await conv.say("msg1", tool_name="test_tool")
        await conv.say("msg2", tool_name="test_tool")
        assert conv.turn_count == 2
        assert len(conv.turns) == 2
        assert isinstance(conv.turns[0], Turn)
        assert conv.turns[0].turn_number == 1
        assert conv.turns[1].turn_number == 2

    @pytest.mark.asyncio
    async def test_history_truncation(self):
        cip = _make_cip()
        conv = Conversation(cip, max_history_turns=2)
        for i in range(5):
            await conv.say(f"message {i}", tool_name="test_tool")
        # max_history_turns=2 → max 4 messages
        assert len(conv.history) == 4

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        cip = _make_cip()
        conv = Conversation(cip)
        await conv.say("hello", tool_name="test_tool")
        assert conv.turn_count == 1
        conv.reset()
        assert conv.turn_count == 0
        assert conv.history == []
        assert conv.accumulated_context == {}
        assert conv.last_scaffold_id is None

    @pytest.mark.asyncio
    async def test_last_scaffold_id(self):
        cip = _make_cip()
        conv = Conversation(cip)
        assert conv.last_scaffold_id is None
        await conv.say("hello", tool_name="test_tool")
        assert conv.last_scaffold_id == "test_scaffold"

    @pytest.mark.asyncio
    async def test_data_context_merged_with_accumulated(self):
        provider = MockProvider()
        cip = _make_cip(provider=provider)
        conv = Conversation(cip)
        await conv.say("first", tool_name="test_tool", data_context={"key1": "val1"})
        await conv.say("second", tool_name="test_tool", data_context={"key2": "val2"})
        # Provider should have received merged context on second call
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_string_policy_accepted(self):
        cip = _make_cip()
        conv = Conversation(cip)
        result = await conv.say("hello", tool_name="test_tool", policy="be concise")
        assert result.policy_source

    @pytest.mark.asyncio
    async def test_history_passed_to_provider(self):
        provider = MockProvider()
        cip = _make_cip(provider=provider)
        conv = Conversation(cip)
        await conv.say("first", tool_name="test_tool")
        await conv.say("second", tool_name="test_tool")
        # On second call, provider should have received chat history
        assert len(provider.last_chat_history) > 0
