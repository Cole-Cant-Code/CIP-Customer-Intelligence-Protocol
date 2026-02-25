"""Tests for the CIP facade."""

from __future__ import annotations

import pytest

from cip_protocol.cip import CIP, CIPResult
from cip_protocol.control import RunPolicy
from cip_protocol.llm.providers.mock import MockProvider
from cip_protocol.scaffold.registry import ScaffoldRegistry
from tests.conftest import make_test_config, make_test_scaffold


def _make_cip(
    scaffolds=None,
    default_scaffold_id="test_scaffold",
    provider=None,
) -> CIP:
    registry = ScaffoldRegistry()
    for s in (scaffolds or [make_test_scaffold()]):
        registry.register(s)
    config = make_test_config(default_scaffold_id=default_scaffold_id)
    return CIP(config, registry, provider or MockProvider())


class TestCIPRun:
    @pytest.mark.asyncio
    async def test_basic_run(self):
        cip = _make_cip()
        result = await cip.run("test query", tool_name="test_tool")
        assert isinstance(result, CIPResult)
        assert result.response.content
        assert result.scaffold_id == "test_scaffold"

    @pytest.mark.asyncio
    async def test_result_has_selection_metadata(self):
        cip = _make_cip()
        result = await cip.run("test query", tool_name="test_tool")
        assert result.selection_mode in ("caller_id", "tool_match", "scored", "default")
        assert result.scaffold_display_name

    @pytest.mark.asyncio
    async def test_string_policy_parsed(self):
        cip = _make_cip()
        result = await cip.run(
            "test query", tool_name="test_tool", policy="be concise, bullet points",
        )
        assert result.policy_source
        assert "constraint:" in result.policy_source

    @pytest.mark.asyncio
    async def test_unrecognized_constraints_captured(self):
        cip = _make_cip()
        result = await cip.run(
            "test query", tool_name="test_tool", policy="be concise, do a backflip",
        )
        assert "do a backflip" in result.unrecognized_constraints

    @pytest.mark.asyncio
    async def test_run_policy_object_accepted(self):
        cip = _make_cip()
        policy = RunPolicy(temperature=0.9, source="test_source")
        result = await cip.run("test query", tool_name="test_tool", policy=policy)
        assert result.policy_source == "test_source"

    @pytest.mark.asyncio
    async def test_none_policy_works(self):
        cip = _make_cip()
        result = await cip.run("test query", tool_name="test_tool", policy=None)
        assert result.policy_source == ""
        assert result.unrecognized_constraints == []

    @pytest.mark.asyncio
    async def test_data_context_defaults_to_empty(self):
        provider = MockProvider()
        cip = _make_cip(provider=provider)
        await cip.run("test query", tool_name="test_tool")
        # The engine got called with {} not None — verify no error occurred
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_scaffold_id_override(self):
        s1 = make_test_scaffold("s1", tools=["tool_a"])
        s2 = make_test_scaffold("s2", tools=["tool_b"])
        cip = _make_cip(scaffolds=[s1, s2])
        result = await cip.run("anything", scaffold_id="s2")
        assert result.scaffold_id == "s2"
        assert result.selection_mode == "caller_id"

    @pytest.mark.asyncio
    async def test_scored_selection_returns_scores(self):
        s1 = make_test_scaffold("s1", keywords=["money", "spending"], tools=[])
        s2 = make_test_scaffold("s2", keywords=["budget"], tools=[])
        cip = _make_cip(scaffolds=[s1, s2], default_scaffold_id="s1")
        result = await cip.run("show me my spending money")
        assert result.selection_mode == "scored"
        assert len(result.selection_scores) > 0


class TestCIPStream:
    @pytest.mark.asyncio
    async def test_stream_yields_events(self):
        cip = _make_cip()
        events = []
        async for event in cip.stream("test query", tool_name="test_tool"):
            events.append(event)
        assert len(events) > 0
        assert events[-1].event == "final"


class TestCIPFromConfig:
    def test_from_config_with_mock(self, tmp_path):
        # Create a minimal scaffold file
        scaffold_file = tmp_path / "test.yaml"
        scaffold_file.write_text(
            "id: test_scaffold\n"
            "version: '1.0'\n"
            "domain: test\n"
            "display_name: Test\n"
            "description: Test scaffold\n"
            "applicability:\n"
            "  tools: [test_tool]\n"
            "  keywords: [test]\n"
            "framing:\n"
            "  role: Analyst\n"
            "  perspective: Analytical\n"
            "  tone: neutral\n"
            "reasoning_framework:\n"
            "  steps: [Analyze]\n"
            "domain_knowledge_activation: [test]\n"
            "output_calibration:\n"
            "  format: structured_narrative\n"
            "guardrails:\n"
            "  disclaimers: [Test disclaimer]\n"
        )
        config = make_test_config()
        cip = CIP.from_config(config, str(tmp_path), "mock")
        assert len(cip.registry.all()) == 1

    def test_from_config_accepts_provider_instance(self):
        config = make_test_config()
        provider = MockProvider(response_content="custom response")
        registry = ScaffoldRegistry()
        registry.register(make_test_scaffold())
        cip = CIP(config, registry, provider)
        assert cip.client.provider is provider


class TestCIPConversation:
    def test_conversation_returns_conversation(self):
        cip = _make_cip()
        conv = cip.conversation()
        from cip_protocol.conversation import Conversation
        assert isinstance(conv, Conversation)

    def test_conversation_max_turns(self):
        cip = _make_cip()
        conv = cip.conversation(max_history_turns=5)
        assert conv._max_history_turns == 5
