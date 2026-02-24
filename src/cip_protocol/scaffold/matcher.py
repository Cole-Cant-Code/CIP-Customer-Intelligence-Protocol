"""Scaffold matcher -- multi-criteria scoring for scaffold selection.

Selection follows a strict priority cascade:
  1. Explicit caller_scaffold_id (sophisticated clients can name a scaffold)
  2. Tool name match (first scaffold registered for the tool wins)
  3. Intent signal + keyword scoring (fuzzy match against user input)
  4. None (caller is responsible for fallback / default handling)

Scoring uses two weights:
  - INTENT_WEIGHT (2.0): intent signals are high-confidence matches
  - KEYWORD_WEIGHT (1.0): keywords are lower-confidence but still useful
"""

from __future__ import annotations

import logging
import re

from cip_protocol.scaffold.models import Scaffold
from cip_protocol.scaffold.registry import ScaffoldRegistry

logger = logging.getLogger(__name__)

# Scoring weights
INTENT_WEIGHT = 2.0
KEYWORD_WEIGHT = 1.0
EXACT_SIGNAL_BONUS = 0.5


def match_scaffold(
    registry: ScaffoldRegistry,
    tool_name: str,
    user_input: str = "",
    caller_scaffold_id: str | None = None,
) -> Scaffold | None:
    """Select the best scaffold using multi-criteria scoring.

    Selection priority:
    1. Explicit caller_scaffold_id (sophisticated client)
    2. Tool name match
    3. Intent signal + keyword scoring
    4. None (caller handles default)
    """
    # Priority 1: Explicit client choice
    if caller_scaffold_id:
        scaffold = registry.get(caller_scaffold_id)
        if scaffold:
            logger.info("Scaffold selected by caller: %s", scaffold.id)
            return scaffold
        logger.warning(
            "Caller requested scaffold '%s' but not found, falling back",
            caller_scaffold_id,
        )

    # Priority 2: Tool name match
    tool_matches = registry.find_by_tool(tool_name)
    if tool_matches:
        scaffold = tool_matches[0]
        logger.info(
            "Scaffold selected by tool match: %s (tool=%s)", scaffold.id, tool_name
        )
        return scaffold

    # Priority 3: Scored matching (intent signals + keywords)
    if user_input:
        best = _score_scaffolds(registry.all(), user_input)
        if best:
            logger.info("Scaffold selected by scoring: %s", best.id)
            return best

    return None


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """Check phrase match with word boundaries to avoid substring false positives."""
    if not phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)")
    return bool(pattern.search(haystack))


def _score_scaffolds(scaffolds: list[Scaffold], user_input: str) -> Scaffold | None:
    """Score all scaffolds against user input and return the best match."""
    user_lower = user_input.lower()
    user_tokens = _tokenize(user_input)
    best_match: Scaffold | None = None
    best_score = 0.0

    for scaffold in scaffolds:
        score = 0.0

        # Intent signal matching (higher weight)
        for signal in scaffold.applicability.intent_signals:
            signal_tokens = _tokenize(signal)
            if not signal_tokens:
                continue
            matches = sum(1 for token in signal_tokens if token in user_tokens)
            coverage = matches / len(signal_tokens)
            if coverage >= 0.5:
                score += INTENT_WEIGHT * coverage
            if _contains_phrase(user_lower, signal):
                score += EXACT_SIGNAL_BONUS

        # Keyword matching
        for kw in scaffold.applicability.keywords:
            if _contains_phrase(user_lower, kw):
                score += KEYWORD_WEIGHT

        if score > best_score:
            best_score = score
            best_match = scaffold

    return best_match if best_score > 0 else None
