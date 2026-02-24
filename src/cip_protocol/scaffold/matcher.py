"""Multi-criteria scaffold selection.

Priority cascade:
  1. Explicit caller_scaffold_id
  2. Tool name match (first registered wins)
  3. Intent signal + keyword scoring
  4. None (caller handles fallback)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cip_protocol.scaffold.models import Scaffold
from cip_protocol.scaffold.registry import ScaffoldRegistry

INTENT_WEIGHT = 2.0
KEYWORD_WEIGHT = 1.0
EXACT_SIGNAL_BONUS = 0.5
MIN_SIGNAL_COVERAGE = 0.5


# ---------------------------------------------------------------------------
# Pre-computed cache for scaffold tokens and compiled phrase patterns
# ---------------------------------------------------------------------------

@dataclass
class _ScaffoldCache:
    signal_tokens: dict[str, set[str]] = field(default_factory=dict)
    signal_patterns: dict[str, re.Pattern[str]] = field(default_factory=dict)
    keyword_patterns: dict[str, re.Pattern[str]] = field(default_factory=dict)


_cache: dict[str, _ScaffoldCache] = {}


def _compile_phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)")


def _ensure_cached(scaffold: Scaffold) -> _ScaffoldCache:
    cached = _cache.get(scaffold.id)
    if cached is not None:
        return cached

    entry = _ScaffoldCache()
    for signal in scaffold.applicability.intent_signals:
        entry.signal_tokens[signal] = _tokenize(signal)
        lower = signal.lower()
        if lower:
            entry.signal_patterns[signal] = _compile_phrase_pattern(signal)

    for kw in scaffold.applicability.keywords:
        lower = kw.lower()
        if lower:
            entry.keyword_patterns[kw] = _compile_phrase_pattern(kw)

    _cache[scaffold.id] = entry
    return entry


def prepare_matcher_cache(registry: ScaffoldRegistry) -> None:
    """Pre-warm the matcher cache for all registered scaffolds."""
    for scaffold in registry.all():
        _ensure_cached(scaffold)


def clear_matcher_cache() -> None:
    """Clear the matcher cache. Useful in tests."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Public API (unchanged signatures)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _contains_phrase(haystack: str, phrase: str) -> bool:
    if not phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)")
    return bool(pattern.search(haystack))


def match_scaffold(
    registry: ScaffoldRegistry,
    tool_name: str,
    user_input: str = "",
    caller_scaffold_id: str | None = None,
    selection_bias: dict[str, float] | None = None,
) -> Scaffold | None:
    if caller_scaffold_id:
        scaffold = registry.get(caller_scaffold_id)
        if scaffold:
            return scaffold

    tool_matches = registry.find_by_tool(tool_name)
    if tool_matches:
        return tool_matches[0]

    if user_input:
        return _score_scaffolds(registry.all(), user_input, selection_bias=selection_bias)

    return None


def _score_scaffolds(
    scaffolds: list[Scaffold],
    user_input: str,
    selection_bias: dict[str, float] | None = None,
) -> Scaffold | None:
    user_lower = user_input.lower()
    user_tokens = _tokenize(user_input)
    best_match: Scaffold | None = None
    best_score = 0.0

    for scaffold in scaffolds:
        cache = _ensure_cached(scaffold)
        score = 0.0

        for signal in scaffold.applicability.intent_signals:
            signal_tokens = cache.signal_tokens.get(signal, set())
            if not signal_tokens:
                continue
            coverage = sum(1 for t in signal_tokens if t in user_tokens) / len(signal_tokens)
            if coverage >= MIN_SIGNAL_COVERAGE:
                score += INTENT_WEIGHT * coverage
            pat = cache.signal_patterns.get(signal)
            if pat and pat.search(user_lower):
                score += EXACT_SIGNAL_BONUS

        for kw in scaffold.applicability.keywords:
            pat = cache.keyword_patterns.get(kw)
            if pat and pat.search(user_lower):
                score += KEYWORD_WEIGHT

        if selection_bias:
            score *= selection_bias.get(scaffold.id, 1.0)

        if score > best_score:
            best_score = score
            best_match = scaffold

    return best_match if best_score > 0 else None
