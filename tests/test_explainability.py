"""Tests for scaffold selection explainability."""

from __future__ import annotations

import pytest
from conftest import make_test_config, make_test_scaffold

from cip_protocol.scaffold.engine import ScaffoldEngine
from cip_protocol.scaffold.matcher import (
    ScaffoldScore,
    score_scaffolds_explained,
)
from cip_protocol.scaffold.registry import ScaffoldRegistry


class TestScoreScaffoldsExplained:
    def test_returns_scores_for_all_scaffolds(self):
        s1 = make_test_scaffold(
            "s1", keywords=["spending", "money"], intent_signals=["where is my money"],
        )
        s2 = make_test_scaffold("s2", keywords=["budget"], intent_signals=["budget overview"])
        scores = score_scaffolds_explained([s1, s2], "where is my money going?")
        assert len(scores) == 2
        assert all(isinstance(s, ScaffoldScore) for s in scores)

    def test_intent_signals_scored(self):
        s1 = make_test_scaffold("s1", intent_signals=["where is my money"])
        scores = score_scaffolds_explained([s1], "where is my money going?")
        assert scores[0].total_score > 0
        assert len(scores[0].intent_signal_scores) > 0

    def test_keyword_matching_scored(self):
        s1 = make_test_scaffold("s1", keywords=["spending"])
        scores = score_scaffolds_explained([s1], "show me my spending")
        assert scores[0].total_score > 0
        assert "spending" in scores[0].keyword_scores

    def test_no_match_returns_zero(self):
        s1 = make_test_scaffold("s1", keywords=["investing"], intent_signals=["stock portfolio"])
        scores = score_scaffolds_explained([s1], "hello world")
        assert scores[0].total_score == 0.0

    def test_selection_bias_applied(self):
        s1 = make_test_scaffold("s1", keywords=["money"])
        scores_no_bias = score_scaffolds_explained([s1], "show me money")
        scores_biased = score_scaffolds_explained(
            [s1], "show me money", selection_bias={"s1": 2.0},
        )
        assert scores_biased[0].bias_multiplier == 2.0
        assert scores_biased[0].pre_bias_score == scores_no_bias[0].pre_bias_score
        assert abs(scores_biased[0].total_score - scores_no_bias[0].total_score * 2.0) < 1e-9

    def test_empty_scaffolds_returns_empty(self):
        scores = score_scaffolds_explained([], "anything")
        assert scores == []

    def test_layer_breakdown_present(self):
        s1 = make_test_scaffold("s1", keywords=["spending"])
        scores = score_scaffolds_explained([s1], "show me my spending")
        assert hasattr(scores[0], "layers")
        assert scores[0].layers.micro > 0


class TestSelectExplained:
    def _make_engine(self, scaffolds, default_scaffold_id=None):
        registry = ScaffoldRegistry()
        for s in scaffolds:
            registry.register(s)
        config = make_test_config(default_scaffold_id=default_scaffold_id)
        return ScaffoldEngine(registry, config)

    def test_caller_id_mode(self):
        s1 = make_test_scaffold("s1")
        engine = self._make_engine([s1])
        scaffold, explanation = engine.select_explained("", caller_scaffold_id="s1")
        assert scaffold.id == "s1"
        assert explanation.selection_mode == "caller_id"
        assert explanation.selected_scaffold_id == "s1"

    def test_tool_match_mode(self):
        s1 = make_test_scaffold("s1", tools=["spending_tool"])
        engine = self._make_engine([s1])
        scaffold, explanation = engine.select_explained("spending_tool")
        assert scaffold.id == "s1"
        assert explanation.selection_mode == "tool_match"

    def test_scored_mode(self):
        s1 = make_test_scaffold("s1", keywords=["spending", "money"], tools=[])
        s2 = make_test_scaffold("s2", keywords=["budget"], tools=[])
        engine = self._make_engine([s1, s2])
        scaffold, explanation = engine.select_explained("", user_input="show me my spending money")
        assert scaffold.id == "s1"
        assert explanation.selection_mode == "scored"
        assert len(explanation.scores) >= 2

    def test_scored_mode_has_confidence(self):
        s1 = make_test_scaffold("s1", keywords=["spending", "money"], tools=[])
        engine = self._make_engine([s1])
        scaffold, explanation = engine.select_explained("", user_input="show me my spending money")
        assert explanation.confidence > 0

    def test_default_fallback_mode(self):
        s1 = make_test_scaffold("fallback", keywords=["xyz"], tools=[])
        engine = self._make_engine([s1], default_scaffold_id="fallback")
        scaffold, explanation = engine.select_explained("", user_input="unrelated query")
        assert scaffold.id == "fallback"
        assert explanation.selection_mode == "default"

    def test_no_scaffold_raises(self):
        engine = self._make_engine([], default_scaffold_id=None)
        with pytest.raises(Exception):
            engine.select_explained("", user_input="anything")

    def test_explanation_has_scores_dict(self):
        s1 = make_test_scaffold("s1", keywords=["test"], tools=[])
        engine = self._make_engine([s1])
        _, explanation = engine.select_explained("", user_input="test query")
        assert explanation.selection_mode == "scored"
        score_map = {s.scaffold_id: s.total_score for s in explanation.scores}
        assert "s1" in score_map
