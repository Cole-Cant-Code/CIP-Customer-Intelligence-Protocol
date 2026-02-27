"""Tests for the scaffold engine — selection, matching, rendering."""

import pytest
from conftest import make_test_config, make_test_scaffold

from cip_protocol.scaffold.engine import ScaffoldEngine, ScaffoldNotFoundError
from cip_protocol.scaffold.loader import load_scaffold_directory
from cip_protocol.scaffold.registry import ScaffoldRegistry
from cip_protocol.telemetry import InMemoryTelemetrySink


class TestScaffoldRegistry:
    def test_register_and_get(self):
        registry = ScaffoldRegistry()
        s = make_test_scaffold("alpha")
        registry.register(s)
        assert registry.get("alpha") is s

    def test_duplicate_id_raises(self):
        registry = ScaffoldRegistry()
        registry.register(make_test_scaffold("alpha"))
        with pytest.raises(ValueError, match="Duplicate"):
            registry.register(make_test_scaffold("alpha"))

    def test_find_by_tool(self):
        registry = ScaffoldRegistry()
        s = make_test_scaffold("alpha", tools=["analyze"])
        registry.register(s)
        assert registry.find_by_tool("analyze") == [s]
        assert registry.find_by_tool("nonexistent") == []

    def test_find_by_tag(self):
        registry = ScaffoldRegistry()
        s = make_test_scaffold("alpha")
        s.tags = ["important"]
        registry.register(s)
        assert registry.find_by_tag("important") == [s]


class TestScaffoldEngine:
    def _engine_with_scaffolds(self):
        config = make_test_config()
        telemetry = InMemoryTelemetrySink()
        registry = ScaffoldRegistry()
        registry.register(make_test_scaffold(
            "test_scaffold", tools=["default_tool"]
        ))
        registry.register(make_test_scaffold(
            "special", tools=["special_tool"],
            keywords=["special"],
        ))
        engine = ScaffoldEngine(registry, config=config, telemetry_sink=telemetry)
        return engine, telemetry

    def test_select_by_tool(self):
        engine, _ = self._engine_with_scaffolds()
        scaffold = engine.select(tool_name="special_tool")
        assert scaffold.id == "special"

    def test_select_by_caller_id(self):
        engine, _ = self._engine_with_scaffolds()
        scaffold = engine.select(
            tool_name="unknown_tool", caller_scaffold_id="special"
        )
        assert scaffold.id == "special"

    def test_select_falls_back_to_default(self):
        engine, _ = self._engine_with_scaffolds()
        scaffold = engine.select(tool_name="unknown_tool")
        assert scaffold.id == "test_scaffold"

    def test_select_raises_when_no_default(self):
        config = make_test_config(default_scaffold_id=None)
        registry = ScaffoldRegistry()
        engine = ScaffoldEngine(registry, config=config)
        with pytest.raises(ScaffoldNotFoundError):
            engine.select(tool_name="nonexistent")

    def test_apply_produces_assembled_prompt(self):
        engine, _ = self._engine_with_scaffolds()
        scaffold = engine.select(tool_name="default_tool")
        prompt = engine.apply(
            scaffold=scaffold,
            user_query="Test query",
            data_context={"key": "value"},
        )
        assert "Test analyst" in prompt.system_message
        assert "Test query" in prompt.user_message
        assert "Test Data" in prompt.user_message  # from config.data_context_label

    def test_apply_uses_generic_label_without_config(self):
        registry = ScaffoldRegistry()
        registry.register(make_test_scaffold("s1", tools=["t1"]))
        engine = ScaffoldEngine(registry, config=None)
        scaffold = engine.select(tool_name="t1")
        prompt = engine.apply(
            scaffold=scaffold,
            user_query="query",
            data_context={"x": 1},
        )
        assert "Data Context" in prompt.user_message

    def test_telemetry_events_emitted(self):
        engine, telemetry = self._engine_with_scaffolds()
        scaffold = engine.select(tool_name="default_tool")
        engine.apply(scaffold=scaffold, user_query="q", data_context={"k": "v"})

        event_names = [event.name for event in telemetry.events]
        assert "scaffold.select" in event_names
        assert "scaffold.apply" in event_names


class TestScaffoldMatching:
    def test_intent_signals_score_higher_than_keywords(self):
        registry = ScaffoldRegistry()
        kw_scaffold = make_test_scaffold(
            "keyword_match",
            tools=[],
            keywords=["budget"],
        )
        intent_scaffold = make_test_scaffold(
            "intent_match",
            tools=[],
            keywords=[],
            intent_signals=["create a budget"],
        )
        registry.register(kw_scaffold)
        registry.register(intent_scaffold)

        engine = ScaffoldEngine(registry)
        scaffold = engine.select(
            tool_name="no_match",
            user_input="I want to create a budget for next month",
        )
        assert scaffold.id == "intent_match"

    def test_keyword_matching_uses_word_boundaries(self):
        registry = ScaffoldRegistry()
        registry.register(
            make_test_scaffold(
                "keyword_plan",
                tools=[],
                keywords=["plan"],
                intent_signals=[],
            )
        )
        engine = ScaffoldEngine(registry, config=make_test_config(default_scaffold_id=None))

        with pytest.raises(ScaffoldNotFoundError):
            engine.select(tool_name="no_match", user_input="planetary motion is stable")


class TestBuiltinScaffolds:
    def test_builtins_auto_register_on_engine_init(self):
        registry = ScaffoldRegistry()
        ScaffoldEngine(registry)
        assert registry.get("orchestration_layer_assessment") is not None

    def test_orchestration_selected_by_tool_name(self):
        registry = ScaffoldRegistry()
        registry.register(make_test_scaffold("fallback", tools=["other"]))
        config = make_test_config(default_scaffold_id="fallback")
        engine = ScaffoldEngine(registry, config=config)
        scaffold = engine.select(tool_name="orchestration")
        assert scaffold.id == "orchestration_layer_assessment"

    def test_orchestration_selected_by_intent_signal(self):
        registry = ScaffoldRegistry()
        registry.register(make_test_scaffold("fallback", tools=[]))
        config = make_test_config(default_scaffold_id="fallback")
        engine = ScaffoldEngine(registry, config=config)
        scaffold = engine.select(
            tool_name="no_match",
            user_input="this request requires choosing between multiple tools",
        )
        assert scaffold.id == "orchestration_layer_assessment"

    def test_domain_scaffolds_dont_clobber_builtins(self, tmp_path):
        # Load domain scaffolds first, then create engine (which loads builtins)
        registry = ScaffoldRegistry()
        scaffold_file = tmp_path / "domain.yaml"
        scaffold_file.write_text(
            "id: domain_scaffold\n"
            "version: '1.0'\n"
            "domain: test\n"
            "display_name: Domain Test\n"
            "description: A domain scaffold\n"
            "applicability:\n"
            "  tools: [domain_tool]\n"
            "  keywords: [domain]\n"
            "framing:\n"
            "  role: Domain analyst\n"
            "  perspective: Analytical\n"
            "  tone: neutral\n"
            "reasoning_framework:\n"
            "  steps: [Analyze]\n"
            "domain_knowledge_activation: [domain knowledge]\n"
            "output_calibration:\n"
            "  format: structured_narrative\n"
            "guardrails:\n"
            "  disclaimers: [Test disclaimer]\n"
        )
        load_scaffold_directory(str(tmp_path), registry)
        engine = ScaffoldEngine(registry)

        # Both should be present
        assert registry.get("domain_scaffold") is not None
        assert registry.get("orchestration_layer_assessment") is not None

        # Each selectable by its own tool name
        assert engine.select(tool_name="domain_tool").id == "domain_scaffold"
        assert engine.select(tool_name="orchestration").id == "orchestration_layer_assessment"
