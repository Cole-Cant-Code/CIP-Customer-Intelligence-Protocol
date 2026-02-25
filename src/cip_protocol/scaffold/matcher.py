"""Layered scaffold selection.

Scores scaffolds across four signal layers (micro/meso/macro/meta),
with saturation, cross-layer reinforcement, and confidence assessment.
All scoring parameters are supplied via SelectionParams — the LLM
(or orchestrator) controls the tuning per-call.

Priority cascade (unchanged):
  1. Explicit caller_scaffold_id
  2. Tool name match (first registered wins)
  3. Layered scoring
  4. None (caller handles fallback)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from cip_protocol.scaffold.models import Scaffold
from cip_protocol.scaffold.registry import ScaffoldRegistry

# ---------------------------------------------------------------------------
# Defaults — used when the caller provides no SelectionParams.
# Maximally permissive: don't reject or flag anything by default.
# The LLM opts into strictness by setting params.
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {"micro": 0.20, "meso": 0.40, "macro": 0.30, "meta": 0.10}
_DEFAULT_SATURATION = {"micro": 0.7, "meso": 1.0}
_DEFAULT_MIN_SIGNAL_COVERAGE = 0.5
_DEFAULT_EXACT_SIGNAL_BONUS = 0.3
_DEFAULT_REINFORCEMENT = 0.15
_DEFAULT_LAYER_ACTIVATION = 0.05
_DEFAULT_MACRO_MIN_OVERLAP = 2
_DEFAULT_MIN_CONFIDENCE = 0.0    # any positive score wins (backward compat)
_DEFAULT_AMBIGUITY_MARGIN = 0.0  # don't flag ambiguity unless asked

# Backward-compat aliases — consumed by benchmark_hotpaths.py and tests
INTENT_WEIGHT = 2.0
KEYWORD_WEIGHT = 1.0
EXACT_SIGNAL_BONUS = 0.5
MIN_SIGNAL_COVERAGE = 0.5


# ---------------------------------------------------------------------------
# SelectionParams — the LLM controls this
# ---------------------------------------------------------------------------

@dataclass
class SelectionParams:
    """Every tunable knob in the layered scorer.

    The LLM (or orchestrator, or RunPolicy, or scaffold YAML itself)
    can supply these per-call. Anything left as None falls back to
    a default. The defaults are permissive starting points, not policy.

    Example — LLM deciding "this is clearly a finance question,
    lean on domain match, raise the confidence bar"::

        SelectionParams(
            layer_weights={"micro": 0.05, "meso": 0.35, "macro": 0.45, "meta": 0.15},
            min_confidence=0.20,
            context={"domain": "finance"},
        )
    """

    # Layer weights (micro/meso/macro/meta, should sum to ~1.0)
    layer_weights: dict[str, float] | None = None

    # Saturation rates per layer (higher k = faster diminishing returns)
    saturation: dict[str, float] | None = None

    # Intent signal matching
    min_signal_coverage: float | None = None
    exact_signal_bonus: float | None = None

    # Cross-layer interaction
    reinforcement: float | None = None
    layer_activation: float | None = None

    # Macro layer
    macro_min_overlap: int | None = None

    # Selection thresholds
    min_confidence: float | None = None
    ambiguity_margin: float | None = None

    # Contextual inputs for meta layer
    context: dict[str, object] | None = None

    # Per-scaffold bias (from RunPolicy)
    selection_bias: dict[str, float] | None = None

    def weights(self) -> dict[str, float]:
        return self.layer_weights or _DEFAULT_WEIGHTS

    def sat(self, layer: str) -> float:
        rates = self.saturation or _DEFAULT_SATURATION
        return rates.get(layer, 1.0)

    def signal_coverage(self) -> float:
        if self.min_signal_coverage is not None:
            return self.min_signal_coverage
        return _DEFAULT_MIN_SIGNAL_COVERAGE

    def signal_bonus(self) -> float:
        if self.exact_signal_bonus is not None:
            return self.exact_signal_bonus
        return _DEFAULT_EXACT_SIGNAL_BONUS

    def reinforce(self) -> float:
        return self.reinforcement if self.reinforcement is not None else _DEFAULT_REINFORCEMENT

    def activation(self) -> float:
        if self.layer_activation is not None:
            return self.layer_activation
        return _DEFAULT_LAYER_ACTIVATION

    def macro_overlap(self) -> int:
        if self.macro_min_overlap is not None:
            return self.macro_min_overlap
        return _DEFAULT_MACRO_MIN_OVERLAP

    def confidence(self) -> float:
        return self.min_confidence if self.min_confidence is not None else _DEFAULT_MIN_CONFIDENCE

    def ambiguity(self) -> float:
        if self.ambiguity_margin is not None:
            return self.ambiguity_margin
        return _DEFAULT_AMBIGUITY_MARGIN


# ---------------------------------------------------------------------------
# Pre-computed cache for scaffold tokens and compiled phrase patterns
# ---------------------------------------------------------------------------

@dataclass
class _ScaffoldCache:
    signal_tokens: dict[str, set[str]] = field(default_factory=dict)
    signal_patterns: dict[str, re.Pattern[str]] = field(default_factory=dict)
    keyword_patterns: dict[str, re.Pattern[str]] = field(default_factory=dict)
    match_tokens: set[str] = field(default_factory=set)
    has_tokenless_keyword: bool = False
    signature: tuple[tuple[str, ...], tuple[str, ...]] = field(default_factory=lambda: ((), ()))


_cache: dict[str, _ScaffoldCache] = {}
_token_to_scaffold_ids: dict[str, set[str]] = {}


def _compile_phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(phrase.lower())}\b")


def _normalize_phrase(phrase: str) -> str:
    return " ".join(phrase.lower().split())


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _scaffold_signature(scaffold: Scaffold) -> tuple[tuple[str, ...], tuple[str, ...]]:
    intent_signals = tuple(sorted(
        normalized
        for raw in scaffold.applicability.intent_signals
        if (normalized := _normalize_phrase(raw))
    ))
    keywords = tuple(sorted(
        normalized
        for raw in scaffold.applicability.keywords
        if (normalized := _normalize_phrase(raw))
    ))
    return intent_signals, keywords


def _evict_scaffold_tokens(scaffold_id: str, tokens: set[str]) -> None:
    for token in tokens:
        scaffold_ids = _token_to_scaffold_ids.get(token)
        if not scaffold_ids:
            continue
        scaffold_ids.discard(scaffold_id)
        if not scaffold_ids:
            _token_to_scaffold_ids.pop(token, None)


def _ensure_cached(scaffold: Scaffold) -> _ScaffoldCache:
    signature = _scaffold_signature(scaffold)
    cached = _cache.get(scaffold.id)
    if cached is not None and cached.signature == signature:
        return cached
    if cached is not None:
        _evict_scaffold_tokens(scaffold.id, cached.match_tokens)

    entry = _ScaffoldCache(signature=signature)
    for signal in scaffold.applicability.intent_signals:
        signal_tokens = _tokenize(signal)
        entry.signal_tokens[signal] = signal_tokens
        entry.match_tokens.update(signal_tokens)
        lower = signal.lower()
        if lower:
            entry.signal_patterns[signal] = _compile_phrase_pattern(signal)

    for kw in scaffold.applicability.keywords:
        keyword_tokens = _tokenize(kw)
        if keyword_tokens:
            entry.match_tokens.update(keyword_tokens)
        elif kw.strip():
            entry.has_tokenless_keyword = True
        lower = kw.lower()
        if lower:
            entry.keyword_patterns[kw] = _compile_phrase_pattern(kw)

    _cache[scaffold.id] = entry

    for token in entry.match_tokens:
        _token_to_scaffold_ids.setdefault(token, set()).add(scaffold.id)

    return entry


def prepare_matcher_cache(registry: ScaffoldRegistry) -> None:
    """Pre-warm the matcher cache for all registered scaffolds."""
    for scaffold in registry.all():
        _ensure_cached(scaffold)


def clear_matcher_cache() -> None:
    """Clear the matcher cache. Useful in tests."""
    _cache.clear()
    _token_to_scaffold_ids.clear()


def _candidate_scaffolds(scaffolds: list[Scaffold], user_tokens: set[str]) -> list[Scaffold]:
    """Fast candidate pruning: score only scaffolds sharing at least one token."""
    if not scaffolds or not user_tokens:
        return scaffolds

    for scaffold in scaffolds:
        _ensure_cached(scaffold)

    candidate_ids: set[str] = set()
    for token in user_tokens:
        candidate_ids.update(_token_to_scaffold_ids.get(token, set()))

    if not candidate_ids:
        return []

    return [
        scaffold for scaffold in scaffolds
        if scaffold.id in candidate_ids or _cache[scaffold.id].has_tokenless_keyword
    ]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class LayerBreakdown:
    """Per-layer scores in [0, 1]."""
    micro: float = 0.0
    meso: float = 0.0
    macro: float = 0.0
    meta: float = 0.0

    def active_count(self, threshold: float = _DEFAULT_LAYER_ACTIVATION) -> int:
        return sum(
            1 for v in (self.micro, self.meso, self.macro, self.meta)
            if v > threshold
        )

    def as_dict(self) -> dict[str, float]:
        return {"micro": self.micro, "meso": self.meso,
                "macro": self.macro, "meta": self.meta}


@dataclass
class ScaffoldScore:
    scaffold_id: str
    total_score: float
    layers: LayerBreakdown = field(default_factory=LayerBreakdown)
    interaction_multiplier: float = 1.0
    bias_multiplier: float = 1.0
    pre_bias_score: float = 0.0
    intent_signal_scores: dict[str, float] = field(default_factory=dict)
    keyword_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionExplanation:
    selected_scaffold_id: str | None
    selection_mode: str  # "caller_id", "tool_match", "scored", "default"
    scores: list[ScaffoldScore] = field(default_factory=list)
    tool_name: str = ""
    user_input: str = ""
    confidence: float = 0.0
    ambiguous: bool = False
    params_used: SelectionParams | None = None


# ---------------------------------------------------------------------------
# Layer scoring — pure functions, all behavior controlled by params
# ---------------------------------------------------------------------------

def _saturate(raw: float, k: float) -> float:
    """Diminishing returns: 1 - e^(-k * raw). Maps [0, inf) -> [0, 1)."""
    if raw <= 0:
        return 0.0
    return 1.0 - math.exp(-k * raw)


def _score_micro(
    scaffold: Scaffold,
    user_lower: str,
    cache: _ScaffoldCache,
    params: SelectionParams,
) -> tuple[float, dict[str, float]]:
    """Keyword surface matching with saturation."""
    kw_detail: dict[str, float] = {}
    hits = 0

    for kw in scaffold.applicability.keywords:
        pat = cache.keyword_patterns.get(kw)
        if pat and pat.search(user_lower):
            kw_detail[kw] = 1.0
            hits += 1

    return _saturate(hits, params.sat("micro")), kw_detail


def _score_meso(
    scaffold: Scaffold,
    user_tokens: set[str],
    user_lower: str,
    cache: _ScaffoldCache,
    params: SelectionParams,
) -> tuple[float, dict[str, float]]:
    """Intent signal coverage with saturation."""
    signal_detail: dict[str, float] = {}
    raw_sum = 0.0
    min_cov = params.signal_coverage()
    bonus = params.signal_bonus()

    for signal in scaffold.applicability.intent_signals:
        signal_tokens = cache.signal_tokens.get(signal, set())
        if not signal_tokens:
            continue

        coverage = sum(1 for t in signal_tokens if t in user_tokens) / len(signal_tokens)
        if coverage < min_cov:
            continue

        contribution = coverage

        pat = cache.signal_patterns.get(signal)
        if pat and pat.search(user_lower):
            contribution += bonus

        signal_detail[signal] = contribution
        raw_sum += contribution

    return _saturate(raw_sum, params.sat("meso")), signal_detail


def _score_macro(
    scaffold: Scaffold,
    user_tokens: set[str],
    params: SelectionParams,
) -> float:
    """Structural alignment via description token overlap."""
    desc_tokens = _tokenize(scaffold.description)
    if not desc_tokens:
        return 0.0

    overlap = len(user_tokens & desc_tokens)

    if overlap < params.macro_overlap():
        return 0.0

    return min(overlap / len(desc_tokens), 1.0)


def _score_meta(
    scaffold: Scaffold,
    params: SelectionParams,
) -> float:
    """Contextual signals from caller-provided context dict."""
    context = params.context
    if not context:
        return 0.0

    score = 0.0

    domain_hint = context.get("domain")
    if domain_hint and scaffold.domain == domain_hint:
        score = 0.5

    prior_id = context.get("prior_scaffold_id")
    if prior_id == scaffold.id:
        score = max(score, 0.3)

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Per-scaffold scoring
# ---------------------------------------------------------------------------

def _score_one(
    scaffold: Scaffold,
    user_tokens: set[str],
    user_lower: str,
    cache: _ScaffoldCache,
    params: SelectionParams,
) -> ScaffoldScore:
    """Score a single scaffold across all layers."""
    micro, kw_detail = _score_micro(scaffold, user_lower, cache, params)
    meso, sig_detail = _score_meso(scaffold, user_tokens, user_lower, cache, params)
    macro = _score_macro(scaffold, user_tokens, params)
    meta = _score_meta(scaffold, params)

    layers = LayerBreakdown(micro=micro, meso=meso, macro=macro, meta=meta)

    w = params.weights()
    weighted = (
        w.get("micro", 0) * micro
        + w.get("meso", 0) * meso
        + w.get("macro", 0) * macro
        + w.get("meta", 0) * meta
    )

    active = layers.active_count(params.activation())
    interaction = 1.0 + params.reinforce() * max(0, active - 1)

    pre_bias = weighted * interaction

    multiplier = 1.0
    if params.selection_bias:
        multiplier = params.selection_bias.get(scaffold.id, 1.0)

    return ScaffoldScore(
        scaffold_id=scaffold.id,
        total_score=pre_bias * multiplier,
        layers=layers,
        interaction_multiplier=interaction,
        bias_multiplier=multiplier,
        pre_bias_score=pre_bias,
        intent_signal_scores=sig_detail,
        keyword_scores=kw_detail,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _score_scaffolds_layered(
    scaffolds: list[Scaffold],
    user_input: str,
    params: SelectionParams,
) -> tuple[Scaffold | None, list[ScaffoldScore], float, bool]:
    """Score all scaffolds. Returns (best_scaffold, scores, confidence, ambiguous)."""
    if not scaffolds or not user_input:
        return None, [], 0.0, False

    user_lower = user_input.lower()
    user_tokens = _tokenize(user_input)

    # Pass 1: token-indexed candidates (fast)
    candidates = _candidate_scaffolds(scaffolds, user_tokens)
    candidate_ids = {s.id for s in candidates}

    scores: list[ScaffoldScore] = []
    non_candidate_scaffolds: list[Scaffold] = []

    for scaffold in scaffolds:
        if scaffold.id in candidate_ids:
            cache = _ensure_cached(scaffold)
            scores.append(_score_one(scaffold, user_tokens, user_lower, cache, params))
        else:
            non_candidate_scaffolds.append(scaffold)

    # Pass 2: macro fallback (only if fast path found nothing confident)
    best_so_far = max((s.total_score for s in scores), default=0.0)
    conf_threshold = params.confidence()

    if conf_threshold > 0 and best_so_far < conf_threshold and non_candidate_scaffolds:
        for scaffold in non_candidate_scaffolds:
            macro = _score_macro(scaffold, user_tokens, params)
            if macro > params.activation():
                cache = _ensure_cached(scaffold)
                scores.append(_score_one(scaffold, user_tokens, user_lower, cache, params))
            else:
                scores.append(ScaffoldScore(scaffold_id=scaffold.id, total_score=0.0))
    else:
        for scaffold in non_candidate_scaffolds:
            scores.append(ScaffoldScore(scaffold_id=scaffold.id, total_score=0.0))

    scores.sort(key=lambda s: s.total_score, reverse=True)

    if not scores or scores[0].total_score <= 0:
        return None, scores, 0.0, False

    best = scores[0]
    confidence = best.total_score

    if conf_threshold > 0 and confidence < conf_threshold:
        return None, scores, confidence, False

    ambiguous = False
    amb_margin = params.ambiguity()
    if amb_margin > 0 and len(scores) > 1 and scores[1].total_score > 0:
        margin = best.total_score - scores[1].total_score
        if margin < amb_margin:
            ambiguous = True

    selected = None
    for scaffold in scaffolds:
        if scaffold.id == best.scaffold_id:
            selected = scaffold
            break

    return selected, scores, confidence, ambiguous


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_scaffolds_explained(
    scaffolds: list[Scaffold],
    user_input: str,
    selection_bias: dict[str, float] | None = None,
    params: SelectionParams | None = None,
) -> list[ScaffoldScore]:
    """Score all scaffolds with per-scaffold breakdown."""
    p = params or SelectionParams(selection_bias=selection_bias)
    if selection_bias and not p.selection_bias:
        p.selection_bias = selection_bias
    _, scores, _, _ = _score_scaffolds_layered(scaffolds, user_input, p)
    return scores


def _score_scaffolds(
    scaffolds: list[Scaffold],
    user_input: str,
    selection_bias: dict[str, float] | None = None,
) -> Scaffold | None:
    """Return best-matching scaffold or None."""
    p = SelectionParams(selection_bias=selection_bias)
    scaffold, _, _, _ = _score_scaffolds_layered(scaffolds, user_input, p)
    return scaffold


def match_scaffold(
    registry: ScaffoldRegistry,
    tool_name: str,
    user_input: str = "",
    caller_scaffold_id: str | None = None,
    selection_bias: dict[str, float] | None = None,
    params: SelectionParams | None = None,
) -> Scaffold | None:
    """Select a scaffold via the priority cascade."""
    if caller_scaffold_id:
        scaffold = registry.get(caller_scaffold_id)
        if scaffold:
            return scaffold

    tool_matches = registry.find_by_tool(tool_name)
    if tool_matches:
        return tool_matches[0]

    if user_input:
        p = params or SelectionParams(selection_bias=selection_bias)
        if selection_bias and not p.selection_bias:
            p.selection_bias = selection_bias
        scaffold, _, _, _ = _score_scaffolds_layered(registry.all(), user_input, p)
        return scaffold

    return None
