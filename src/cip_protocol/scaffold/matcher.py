"""Multi-criteria scaffold selection.

Priority cascade:
  1. Explicit caller_scaffold_id
  2. Tool name match (first registered wins)
  3. Intent signal + keyword scoring
  4. None (caller handles fallback)
"""

from __future__ import annotations

import re

from cip_protocol.scaffold.models import Scaffold
from cip_protocol.scaffold.registry import ScaffoldRegistry

INTENT_WEIGHT = 2.0
KEYWORD_WEIGHT = 1.0
EXACT_SIGNAL_BONUS = 0.5
MIN_SIGNAL_COVERAGE = 0.5


def match_scaffold(
    registry: ScaffoldRegistry,
    tool_name: str,
    user_input: str = "",
    caller_scaffold_id: str | None = None,
) -> Scaffold | None:
    if caller_scaffold_id:
        scaffold = registry.get(caller_scaffold_id)
        if scaffold:
            return scaffold

    tool_matches = registry.find_by_tool(tool_name)
    if tool_matches:
        return tool_matches[0]

    if user_input:
        return _score_scaffolds(registry.all(), user_input)

    return None


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _contains_phrase(haystack: str, phrase: str) -> bool:
    if not phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)")
    return bool(pattern.search(haystack))


def _score_scaffolds(scaffolds: list[Scaffold], user_input: str) -> Scaffold | None:
    user_lower = user_input.lower()
    user_tokens = _tokenize(user_input)
    best_match: Scaffold | None = None
    best_score = 0.0

    for scaffold in scaffolds:
        score = 0.0

        for signal in scaffold.applicability.intent_signals:
            signal_tokens = _tokenize(signal)
            if not signal_tokens:
                continue
            coverage = sum(1 for t in signal_tokens if t in user_tokens) / len(signal_tokens)
            if coverage >= MIN_SIGNAL_COVERAGE:
                score += INTENT_WEIGHT * coverage
            if _contains_phrase(user_lower, signal):
                score += EXACT_SIGNAL_BONUS

        for kw in scaffold.applicability.keywords:
            if _contains_phrase(user_lower, kw):
                score += KEYWORD_WEIGHT

        if score > best_score:
            best_score = score
            best_match = scaffold

    return best_match if best_score > 0 else None
