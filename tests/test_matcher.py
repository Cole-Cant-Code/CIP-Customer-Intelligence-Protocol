"""Tests for scaffold/matcher.py — layered selection scoring."""

from __future__ import annotations

from conftest import make_test_scaffold

from cip_protocol.scaffold.matcher import (
    EXACT_SIGNAL_BONUS,
    INTENT_WEIGHT,
    KEYWORD_WEIGHT,
    MIN_SIGNAL_COVERAGE,
    LayerBreakdown,
    SelectionParams,
    _cache,
    _saturate,
    _score_scaffolds,
    _score_scaffolds_layered,
    _tokenize,
    clear_matcher_cache,
    match_scaffold,
    prepare_matcher_cache,
    score_scaffolds_explained,
)
from cip_protocol.scaffold.registry import ScaffoldRegistry


class TestTokenize:
    def test_basic_words(self):
        assert _tokenize("Hello World") == {"hello", "world"}

    def test_numbers_included(self):
        assert _tokenize("plan for 2024") == {"plan", "for", "2024"}

    def test_apostrophes_preserved(self):
        assert "don't" in _tokenize("I don't know")

    def test_punctuation_stripped(self):
        tokens = _tokenize("budget, savings, and debt!")
        assert tokens == {"budget", "savings", "and", "debt"}

    def test_empty_string(self):
        assert _tokenize("") == set()

    def test_case_insensitive(self):
        assert _tokenize("BUDGET Plan") == {"budget", "plan"}


class TestSaturate:
    def test_zero_input(self):
        assert _saturate(0, 1.0) == 0.0

    def test_negative_input(self):
        assert _saturate(-1, 1.0) == 0.0

    def test_positive_input(self):
        result = _saturate(1, 0.7)
        assert 0.49 < result < 0.51  # ~0.503

    def test_diminishing_returns(self):
        one = _saturate(1, 0.7)
        two = _saturate(2, 0.7)
        three = _saturate(3, 0.7)
        assert two > one
        assert three > two
        # Diminishing: gap from 1->2 > gap from 2->3
        assert (two - one) > (three - two)

    def test_approaches_one(self):
        assert _saturate(10, 1.0) > 0.99


class TestMatchScaffold:
    def _registry(self) -> ScaffoldRegistry:
        registry = ScaffoldRegistry()
        registry.register(make_test_scaffold(
            "by_tool", tools=["analyze"], keywords=[], intent_signals=[],
        ))
        registry.register(make_test_scaffold(
            "by_keyword", tools=[], keywords=["savings", "budget"], intent_signals=[],
        ))
        registry.register(make_test_scaffold(
            "by_intent", tools=[], keywords=[], intent_signals=["create a budget"],
        ))
        return registry

    def test_caller_scaffold_id_wins(self):
        registry = self._registry()
        result = match_scaffold(registry, "no_match", caller_scaffold_id="by_keyword")
        assert result is not None
        assert result.id == "by_keyword"

    def test_tool_name_match(self):
        registry = self._registry()
        result = match_scaffold(registry, "analyze")
        assert result is not None
        assert result.id == "by_tool"

    def test_falls_through_to_scoring(self):
        registry = self._registry()
        result = match_scaffold(registry, "no_match", user_input="help me create a budget")
        assert result is not None
        assert result.id == "by_intent"

    def test_no_match_returns_none(self):
        registry = self._registry()
        result = match_scaffold(registry, "no_match")
        assert result is None

    def test_no_match_with_irrelevant_input(self):
        registry = self._registry()
        result = match_scaffold(registry, "no_match", user_input="quantum physics lecture")
        assert result is None

    def test_macro_fallback_runs_when_candidate_pruning_finds_none(self):
        registry = ScaffoldRegistry()
        desc_only = make_test_scaffold(
            "desc_only",
            tools=[],
            keywords=["zz_unmatched_kw"],
            intent_signals=["yy unmatched signal"],
        ).model_copy(update={"description": "categorize spending expenses analysis"})
        registry.register(desc_only)

        result = match_scaffold(
            registry,
            "no_match",
            user_input="please categorize spending expenses",
        )
        assert result is not None
        assert result.id == "desc_only"

    def test_invalid_caller_id_falls_through(self):
        registry = self._registry()
        result = match_scaffold(registry, "analyze", caller_scaffold_id="nonexistent")
        assert result is not None
        assert result.id == "by_tool"

    def test_params_passed_through(self):
        registry = self._registry()
        params = SelectionParams(
            layer_weights={"micro": 0.80, "meso": 0.10, "macro": 0.05, "meta": 0.05},
        )
        # With micro heavily weighted, keyword match should dominate
        result = match_scaffold(
            registry, "no_match",
            user_input="help me with my budget savings",
            params=params,
        )
        assert result is not None
        assert result.id == "by_keyword"


class TestScoreScaffolds:
    def test_intent_beats_keyword(self):
        kw = make_test_scaffold(
            "kw", tools=[], keywords=["budget"], intent_signals=[],
        )
        intent = make_test_scaffold(
            "intent", tools=[], keywords=[], intent_signals=["create a budget"],
        )
        result = _score_scaffolds([kw, intent], "I want to create a budget")
        assert result is not None
        assert result.id == "intent"

    def test_no_scaffolds_returns_none(self):
        assert _score_scaffolds([], "anything") is None

    def test_zero_score_returns_none(self):
        s = make_test_scaffold("s", tools=[], keywords=["zebra"], intent_signals=[])
        assert _score_scaffolds([s], "nothing relevant here") is None

    def test_keyword_only_match(self):
        s = make_test_scaffold(
            "s", tools=[], keywords=["savings"], intent_signals=[],
        )
        result = _score_scaffolds([s], "how do I grow my savings?")
        assert result is not None
        assert result.id == "s"

    def test_multiple_keywords_accumulate(self):
        single = make_test_scaffold(
            "single", tools=[], keywords=["savings"], intent_signals=[],
        )
        multi = make_test_scaffold(
            "multi", tools=[], keywords=["savings", "budget", "plan"], intent_signals=[],
        )
        result = _score_scaffolds(
            [single, multi],
            "I need a savings plan and a budget",
        )
        assert result is not None
        assert result.id == "multi"


class TestLayeredScoring:
    def test_layer_breakdown_populated(self):
        s = make_test_scaffold("s", tools=[], keywords=["budget"], intent_signals=[])
        params = SelectionParams()
        _, scores, _, _ = _score_scaffolds_layered([s], "create a budget", params)
        assert len(scores) > 0
        top = scores[0]
        assert isinstance(top.layers, LayerBreakdown)
        assert top.layers.micro > 0  # "budget" keyword matched

    def test_meso_layer_from_intent_signal(self):
        s = make_test_scaffold("s", tools=[], keywords=[], intent_signals=["create a budget"])
        params = SelectionParams()
        _, scores, _, _ = _score_scaffolds_layered([s], "help me create a budget", params)
        top = scores[0]
        assert top.layers.meso > 0

    def test_cross_layer_reinforcement(self):
        """Score should be higher when multiple layers agree."""
        both = make_test_scaffold(
            "both", tools=[], keywords=["budget"], intent_signals=["create a budget"],
        )
        params = SelectionParams()
        _, scores, _, _ = _score_scaffolds_layered([both], "create a budget", params)
        top = scores[0]
        # Both micro and meso fired -> interaction > 1.0
        assert top.interaction_multiplier > 1.0

    def test_custom_weights_change_winner(self):
        """Custom layer weights can flip which scaffold wins."""
        kw_heavy = make_test_scaffold(
            "kw_heavy", tools=[],
            keywords=["savings", "budget", "plan", "money"],
            intent_signals=[],
        )
        intent_focused = make_test_scaffold(
            "intent_focused", tools=[],
            keywords=[],
            intent_signals=["create a savings plan"],
        )

        # Default weights: meso=0.4 > micro=0.2, so intent wins
        default_result = _score_scaffolds(
            [kw_heavy, intent_focused],
            "I need a savings plan and a budget for my money",
        )
        assert default_result is not None
        assert default_result.id == "intent_focused"

        # Flip weights: micro=0.7, meso=0.1 — keywords dominate
        params = SelectionParams(
            layer_weights={"micro": 0.70, "meso": 0.10, "macro": 0.15, "meta": 0.05},
        )
        scaffold, _, _, _ = _score_scaffolds_layered(
            [kw_heavy, intent_focused],
            "I need a savings plan and a budget for my money",
            params,
        )
        assert scaffold is not None
        assert scaffold.id == "kw_heavy"

    def test_confidence_threshold_rejects_weak_match(self):
        s = make_test_scaffold("s", tools=[], keywords=["budget"], intent_signals=[])
        params = SelectionParams(min_confidence=0.5)
        scaffold, _, confidence, _ = _score_scaffolds_layered(
            [s], "budget", params,
        )
        # Single keyword match with default weights produces ~0.10 score
        assert scaffold is None
        assert confidence < 0.5

    def test_ambiguity_detection(self):
        s1 = make_test_scaffold("s1", tools=[], keywords=["budget"], intent_signals=[])
        s2 = make_test_scaffold("s2", tools=[], keywords=["budget"], intent_signals=[])
        params = SelectionParams(ambiguity_margin=0.5)
        scaffold, _, _, ambiguous = _score_scaffolds_layered(
            [s1, s2], "budget analysis", params,
        )
        # Both scaffolds score identically — should be flagged
        assert ambiguous is True

    def test_meta_layer_domain_hint(self):
        finance = make_test_scaffold(
            "finance", domain="finance",
            tools=[], keywords=["analysis"], intent_signals=[],
        )
        health = make_test_scaffold(
            "health", domain="health",
            tools=[], keywords=["analysis"], intent_signals=[],
        )
        # With domain hint: finance should win
        params_hint = SelectionParams(context={"domain": "finance"})
        scaffold, _, _, _ = _score_scaffolds_layered(
            [finance, health], "run an analysis", params_hint,
        )
        assert scaffold is not None
        assert scaffold.id == "finance"

    def test_saturation_prevents_keyword_count_domination(self):
        """A scaffold with 10 keywords shouldn't score 10x one with 2."""
        few = make_test_scaffold(
            "few", tools=[], keywords=["budget", "plan"], intent_signals=[],
        )
        many = make_test_scaffold(
            "many", tools=[],
            keywords=["budget", "plan", "savings", "money", "finance",
                       "spending", "income", "debt", "credit", "tax"],
            intent_signals=[],
        )
        input_text = "budget plan savings money finance spending income debt credit tax"
        params = SelectionParams()
        _, scores, _, _ = _score_scaffolds_layered(
            [few, many], input_text, params,
        )
        score_map = {s.scaffold_id: s for s in scores}
        few_score = score_map["few"].total_score
        many_score = score_map["many"].total_score
        # many should win, but not by 5x — saturation limits the advantage
        assert many_score > few_score
        assert many_score < few_score * 3


class TestScoreScaffoldsExplained:
    def test_returns_scores_for_all(self):
        s1 = make_test_scaffold("s1", tools=[], keywords=["budget"], intent_signals=[])
        s2 = make_test_scaffold("s2", tools=[], keywords=["savings"], intent_signals=[])
        scores = score_scaffolds_explained([s1, s2], "budget")
        assert len(scores) == 2

    def test_selection_bias_applied(self):
        s1 = make_test_scaffold("s1", tools=[], keywords=["money"], intent_signals=[])
        scores_no_bias = score_scaffolds_explained([s1], "show me money")
        scores_biased = score_scaffolds_explained(
            [s1], "show me money", selection_bias={"s1": 2.0},
        )
        assert scores_biased[0].total_score > 0
        assert scores_biased[0].bias_multiplier == 2.0
        assert scores_biased[0].pre_bias_score == scores_no_bias[0].pre_bias_score
        assert abs(scores_biased[0].total_score - scores_no_bias[0].total_score * 2.0) < 1e-9


class TestMatcherCache:
    def test_prepare_populates_cache(self):
        registry = ScaffoldRegistry()
        s = make_test_scaffold("cached", tools=[], keywords=["test"], intent_signals=["do test"])
        registry.register(s)
        prepare_matcher_cache(registry)
        assert "cached" in _cache
        assert "do test" in _cache["cached"].signal_tokens
        assert "test" in _cache["cached"].keyword_patterns

    def test_clear_empties_cache(self):
        registry = ScaffoldRegistry()
        s = make_test_scaffold("to_clear", tools=[], keywords=["x"], intent_signals=[])
        registry.register(s)
        prepare_matcher_cache(registry)
        assert "to_clear" in _cache
        clear_matcher_cache()
        assert len(_cache) == 0

    def test_cached_results_match_uncached(self):
        """Scoring with pre-warmed cache produces identical results to cold cache."""
        kw = make_test_scaffold("kw", tools=[], keywords=["budget"], intent_signals=[])
        intent = make_test_scaffold(
            "intent", tools=[], keywords=[], intent_signals=["create a budget"],
        )
        scaffolds = [kw, intent]
        user_input = "I want to create a budget"

        clear_matcher_cache()
        cold_result = _score_scaffolds(scaffolds, user_input)
        warm_result = _score_scaffolds(scaffolds, user_input)

        assert cold_result is not None
        assert warm_result is not None
        assert cold_result.id == warm_result.id

    def test_lazy_cache_on_first_score(self):
        s = make_test_scaffold("lazy", tools=[], keywords=["data"], intent_signals=[])
        assert "lazy" not in _cache
        _score_scaffolds([s], "show me data")
        assert "lazy" in _cache

    def test_same_id_rebuilds_cache_when_keywords_change(self):
        original = make_test_scaffold("reused", tools=[], keywords=["alpha"], intent_signals=[])
        _score_scaffolds([original], "alpha")

        updated = make_test_scaffold("reused", tools=[], keywords=["beta"], intent_signals=[])
        result = _score_scaffolds([updated], "beta")

        assert result is not None
        assert result.id == "reused"

    def test_same_id_refresh_drops_old_keyword_matches(self):
        original = make_test_scaffold("reused", tools=[], keywords=["alpha"], intent_signals=[])
        _score_scaffolds([original], "alpha")

        updated = make_test_scaffold("reused", tools=[], keywords=["beta"], intent_signals=[])
        _score_scaffolds([updated], "beta")

        assert _score_scaffolds([updated], "alpha") is None


class TestConstants:
    def test_intent_weight_greater_than_keyword(self):
        assert INTENT_WEIGHT > KEYWORD_WEIGHT

    def test_coverage_threshold_is_sensible(self):
        assert 0 < MIN_SIGNAL_COVERAGE <= 1.0

    def test_exact_signal_bonus_positive(self):
        assert EXACT_SIGNAL_BONUS > 0
