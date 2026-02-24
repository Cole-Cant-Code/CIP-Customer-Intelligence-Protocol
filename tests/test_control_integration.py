"""Integration tests for RunPolicy flowing through engine, renderer, matcher, and client."""

from __future__ import annotations

import pytest
from conftest import make_test_config, make_test_scaffold

from cip_protocol.control import (
    ConstraintParser,
    RunPolicy,
)
from cip_protocol.llm.client import InnerLLMClient
from cip_protocol.llm.providers.mock import MockProvider
from cip_protocol.scaffold.engine import ScaffoldEngine
from cip_protocol.scaffold.matcher import _score_scaffolds, match_scaffold
from cip_protocol.scaffold.models import AssembledPrompt
from cip_protocol.scaffold.registry import ScaffoldRegistry
from cip_protocol.scaffold.renderer import render_scaffold
from cip_protocol.telemetry import InMemoryTelemetrySink

# ---------------------------------------------------------------------------
# Renderer integration
# ---------------------------------------------------------------------------


class TestRendererPolicy:
    def test_policy_overrides_max_length(self):
        scaffold = make_test_scaffold()
        policy = RunPolicy(max_length_guidance="under 100 words")
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "under 100 words" in result.system_message

    def test_policy_appends_must_include(self):
        scaffold = make_test_scaffold()
        policy = RunPolicy(extra_must_include=["data sources", "confidence level"])
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "data sources" in result.system_message
        assert "confidence level" in result.system_message

    def test_policy_appends_never_include(self):
        scaffold = make_test_scaffold()
        policy = RunPolicy(extra_never_include=["personal opinions"])
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "personal opinions" in result.system_message

    def test_policy_skips_disclaimers(self):
        scaffold = make_test_scaffold(disclaimers=["Not professional advice."])
        policy = RunPolicy(skip_disclaimers=True)
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "Not professional advice" not in result.system_message
        assert "Disclaimers" not in result.system_message

    def test_policy_removes_specific_prohibited_action(self):
        actions = ["Give diagnoses", "Prescribe medication"]
        scaffold = make_test_scaffold(prohibited_actions=actions)
        policy = RunPolicy(remove_prohibited_actions=["Give diagnoses"])
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "Give diagnoses" not in result.system_message
        assert "Prescribe medication" in result.system_message

    def test_policy_removes_all_prohibited_actions_wildcard(self):
        actions = ["Give diagnoses", "Prescribe medication"]
        scaffold = make_test_scaffold(prohibited_actions=actions)
        policy = RunPolicy(remove_prohibited_actions=["*"])
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "Prohibited Actions" not in result.system_message

    def test_policy_adds_prohibited_actions(self):
        scaffold = make_test_scaffold(prohibited_actions=[])
        policy = RunPolicy(extra_prohibited_actions=["Never speculate"])
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "Never speculate" in result.system_message

    def test_policy_combined_add_remove_prohibited(self):
        scaffold = make_test_scaffold(prohibited_actions=["Old rule"])
        policy = RunPolicy(
            remove_prohibited_actions=["Old rule"],
            extra_prohibited_actions=["New rule"],
        )
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "Old rule" not in result.system_message
        assert "New rule" in result.system_message

    def test_no_policy_backward_compat(self):
        scaffold = make_test_scaffold(disclaimers=["Disclaimer here."])
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={},
        )
        assert "Disclaimer here" in result.system_message

    def test_policy_source_in_metadata(self):
        scaffold = make_test_scaffold()
        policy = RunPolicy(source="preset:creative")
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert result.metadata.get("policy_source") == "preset:creative"

    def test_no_policy_no_policy_source_in_metadata(self):
        scaffold = make_test_scaffold()
        result = render_scaffold(
            scaffold=scaffold, user_query="test", data_context={},
        )
        assert "policy_source" not in result.metadata


# ---------------------------------------------------------------------------
# Matcher integration
# ---------------------------------------------------------------------------


class TestMatcherPolicy:
    def test_selection_bias_boosts_scaffold(self):
        s1 = make_test_scaffold(scaffold_id="weak", keywords=["budget"])
        s2 = make_test_scaffold(scaffold_id="strong", keywords=["budget"])
        # Without bias, both have equal keyword score — first one wins
        result_no_bias = _score_scaffolds([s1, s2], "create a budget")
        assert result_no_bias is not None

        # With bias boosting s2, s2 should win
        result_bias = _score_scaffolds(
            [s1, s2], "create a budget",
            selection_bias={"strong": 5.0},
        )
        assert result_bias is not None
        assert result_bias.id == "strong"

    def test_selection_bias_empty_dict_noop(self):
        s = make_test_scaffold(scaffold_id="s", keywords=["savings"])
        result = _score_scaffolds([s], "savings account", selection_bias={})
        assert result is not None
        assert result.id == "s"

    def test_selection_bias_unknown_id_ignored(self):
        s = make_test_scaffold(scaffold_id="s", keywords=["savings"])
        result = _score_scaffolds(
            [s], "savings account",
            selection_bias={"nonexistent": 10.0},
        )
        assert result is not None
        assert result.id == "s"

    def test_selection_bias_does_not_affect_tool_match(self):
        registry = ScaffoldRegistry()
        s = make_test_scaffold(scaffold_id="s", tools=["my_tool"])
        registry.register(s)
        # Tool match bypasses scoring, so bias is irrelevant
        result = match_scaffold(
            registry, "my_tool",
            selection_bias={"other": 10.0},
        )
        assert result is not None
        assert result.id == "s"


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class TestEnginePolicy:
    def _make_engine(self):
        registry = ScaffoldRegistry()
        scaffold = make_test_scaffold()
        registry.register(scaffold)
        config = make_test_config()
        telemetry = InMemoryTelemetrySink()
        engine = ScaffoldEngine(registry, config=config, telemetry_sink=telemetry)
        return engine, scaffold, telemetry

    def test_apply_with_policy_overrides_tone(self):
        engine, scaffold, _ = self._make_engine()
        policy = RunPolicy(tone_variant="friendly")
        result = engine.apply(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "Warm and approachable" in result.system_message

    def test_apply_with_policy_overrides_format(self):
        engine, scaffold, _ = self._make_engine()
        policy = RunPolicy(output_format="bullet_points")
        result = engine.apply(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "bullet_points" in result.system_message

    def test_apply_with_policy_compact_override(self):
        engine, scaffold, _ = self._make_engine()
        policy = RunPolicy(compact=True)
        result = engine.apply(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert "##" not in result.system_message

    def test_apply_policy_metadata_in_prompt(self):
        engine, scaffold, _ = self._make_engine()
        policy = RunPolicy(source="test_source")
        result = engine.apply(
            scaffold=scaffold, user_query="test", data_context={}, policy=policy,
        )
        assert result.metadata.get("policy_source") == "test_source"

    def test_apply_no_policy_backward_compat(self):
        engine, scaffold, _ = self._make_engine()
        result = engine.apply(
            scaffold=scaffold, user_query="test", data_context={},
        )
        assert "## Your Role" in result.system_message
        assert "policy_source" not in result.metadata

    def test_select_with_policy_bias(self):
        registry = ScaffoldRegistry()
        s1 = make_test_scaffold(scaffold_id="low", keywords=["market"], tools=[])
        s2 = make_test_scaffold(scaffold_id="high", keywords=["market"], tools=[])
        registry.register(s1)
        registry.register(s2)
        config = make_test_config(default_scaffold_id=None)
        engine = ScaffoldEngine(registry, config=config)
        policy = RunPolicy(scaffold_selection_bias={"high": 5.0})
        result = engine.select(
            tool_name="unknown", user_input="market analysis", policy=policy,
        )
        assert result.id == "high"


# ---------------------------------------------------------------------------
# Client integration
# ---------------------------------------------------------------------------


class TestClientPolicy:
    @pytest.mark.asyncio
    async def test_invoke_policy_temperature_override(self):
        provider = MockProvider(response_content="Test.")
        client = InnerLLMClient(provider, config=make_test_config())
        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")
        policy = RunPolicy(temperature=0.9)

        await client.invoke(assembled_prompt=prompt, scaffold=scaffold, policy=policy)
        assert provider.last_temperature == 0.9

    @pytest.mark.asyncio
    async def test_invoke_policy_max_tokens_override(self):
        provider = MockProvider(response_content="Test.")
        client = InnerLLMClient(provider, config=make_test_config())
        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")
        policy = RunPolicy(max_tokens=4096)

        await client.invoke(assembled_prompt=prompt, scaffold=scaffold, policy=policy)
        assert provider.last_max_tokens == 4096

    @pytest.mark.asyncio
    async def test_invoke_policy_skip_disclaimers(self):
        provider = MockProvider(response_content="Analysis here.")
        client = InnerLLMClient(provider, config=make_test_config())
        scaffold = make_test_scaffold(disclaimers=["Not professional advice."])
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")
        policy = RunPolicy(skip_disclaimers=True)

        response = await client.invoke(
            assembled_prompt=prompt, scaffold=scaffold, policy=policy,
        )
        assert "Not professional advice" not in response.content

    @pytest.mark.asyncio
    async def test_invoke_no_policy_backward_compat(self):
        provider = MockProvider(response_content="Analysis here.")
        client = InnerLLMClient(provider, config=make_test_config())
        scaffold = make_test_scaffold(disclaimers=["Not professional advice."])
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")

        response = await client.invoke(assembled_prompt=prompt, scaffold=scaffold)
        assert "Not professional advice" in response.content

    @pytest.mark.asyncio
    async def test_invoke_stream_policy_temperature(self):
        provider = MockProvider(response_content="Stream test.")
        client = InnerLLMClient(provider, config=make_test_config())
        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")
        policy = RunPolicy(temperature=0.95)

        events = []
        async for event in client.invoke_stream(
            assembled_prompt=prompt, scaffold=scaffold, policy=policy,
        ):
            events.append(event)

        assert provider.last_temperature == 0.95
        assert events[-1].event == "final"

    @pytest.mark.asyncio
    async def test_invoke_policy_telemetry_emitted(self):
        sink = InMemoryTelemetrySink()
        provider = MockProvider(response_content="Test.")
        client = InnerLLMClient(provider, config=make_test_config(), telemetry_sink=sink)
        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")
        policy = RunPolicy(source="preset:creative")

        await client.invoke(assembled_prompt=prompt, scaffold=scaffold, policy=policy)

        start_events = [e for e in sink.events if e.name == "llm.invoke.start"]
        assert len(start_events) == 1
        assert start_events[0].attributes.get("policy_source") == "preset:creative"

    @pytest.mark.asyncio
    async def test_invoke_policy_none_uses_defaults(self):
        provider = MockProvider(response_content="Test.")
        client = InnerLLMClient(provider, config=make_test_config())
        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")

        await client.invoke(assembled_prompt=prompt, scaffold=scaffold, policy=None)
        assert provider.last_temperature == 0.3
        assert provider.last_max_tokens == 2048

    @pytest.mark.asyncio
    async def test_full_pipeline_with_parsed_constraints(self):
        """End-to-end: parse constraints → build policy → invoke."""
        text = "be more creative, skip disclaimers, keep it under 200 words"
        result = ConstraintParser.parse(text)
        policy = result.policy

        provider = MockProvider(response_content="Creative analysis.")
        client = InnerLLMClient(provider, config=make_test_config())
        scaffold = make_test_scaffold(disclaimers=["Not professional advice."])
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")

        response = await client.invoke(
            assembled_prompt=prompt, scaffold=scaffold, policy=policy,
        )

        assert provider.last_temperature == 0.8
        assert "Not professional advice" not in response.content
