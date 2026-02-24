"""Tests for scaffold/matcher.py — selection scoring, tokenization, phrase matching."""

from __future__ import annotations

from conftest import make_test_scaffold

from cip_protocol.scaffold.matcher import (
    EXACT_SIGNAL_BONUS,
    INTENT_WEIGHT,
    KEYWORD_WEIGHT,
    MIN_SIGNAL_COVERAGE,
    _cache,
    _contains_phrase,
    _score_scaffolds,
    _tokenize,
    clear_matcher_cache,
    match_scaffold,
    prepare_matcher_cache,
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


class TestContainsPhrase:
    def test_exact_match(self):
        assert _contains_phrase("create a budget", "budget")

    def test_word_boundary_no_partial(self):
        assert not _contains_phrase("planetary motion", "plan")

    def test_phrase_match(self):
        assert _contains_phrase("i want to create a budget", "create a budget")

    def test_empty_phrase_returns_false(self):
        assert not _contains_phrase("anything", "")

    def test_caller_lowercases_haystack(self):
        # _contains_phrase expects pre-lowered haystack (callers pass user_lower)
        assert _contains_phrase("create a budget", "Create A Budget")


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

    def test_invalid_caller_id_falls_through(self):
        registry = self._registry()
        result = match_scaffold(registry, "analyze", caller_scaffold_id="nonexistent")
        assert result is not None
        assert result.id == "by_tool"


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

        # Cold cache
        clear_matcher_cache()
        cold_result = _score_scaffolds(scaffolds, user_input)

        # Warm cache (already populated from first call), re-run
        warm_result = _score_scaffolds(scaffolds, user_input)

        assert cold_result is not None
        assert warm_result is not None
        assert cold_result.id == warm_result.id

    def test_lazy_cache_on_first_score(self):
        """Cache is populated lazily when _score_scaffolds encounters a scaffold."""
        s = make_test_scaffold("lazy", tools=[], keywords=["data"], intent_signals=[])
        assert "lazy" not in _cache
        _score_scaffolds([s], "show me data")
        assert "lazy" in _cache


class TestConstants:
    def test_intent_weight_greater_than_keyword(self):
        assert INTENT_WEIGHT > KEYWORD_WEIGHT

    def test_coverage_threshold_is_sensible(self):
        assert 0 < MIN_SIGNAL_COVERAGE <= 1.0

    def test_exact_signal_bonus_positive(self):
        assert EXACT_SIGNAL_BONUS > 0
