"""Tests for the scaffold engine — selection, matching, rendering."""

import pytest
from conftest import make_test_config, make_test_scaffold

from cip_protocol.scaffold.engine import ScaffoldEngine, ScaffoldNotFoundError
from cip_protocol.scaffold.registry import ScaffoldRegistry


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
        registry = ScaffoldRegistry()
        registry.register(make_test_scaffold(
            "test_scaffold", tools=["default_tool"]
        ))
        registry.register(make_test_scaffold(
            "special", tools=["special_tool"],
            keywords=["special"],
        ))
        engine = ScaffoldEngine(registry, config=config)
        return engine

    def test_select_by_tool(self):
        engine = self._engine_with_scaffolds()
        scaffold = engine.select(tool_name="special_tool")
        assert scaffold.id == "special"

    def test_select_by_caller_id(self):
        engine = self._engine_with_scaffolds()
        scaffold = engine.select(
            tool_name="unknown_tool", caller_scaffold_id="special"
        )
        assert scaffold.id == "special"

    def test_select_falls_back_to_default(self):
        engine = self._engine_with_scaffolds()
        scaffold = engine.select(tool_name="unknown_tool")
        assert scaffold.id == "test_scaffold"

    def test_select_raises_when_no_default(self):
        config = make_test_config(default_scaffold_id=None)
        registry = ScaffoldRegistry()
        engine = ScaffoldEngine(registry, config=config)
        with pytest.raises(ScaffoldNotFoundError):
            engine.select(tool_name="nonexistent")

    def test_apply_produces_assembled_prompt(self):
        engine = self._engine_with_scaffolds()
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
