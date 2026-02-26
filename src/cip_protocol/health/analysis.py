"""M-layer analysis: M-score, coherence, friction/emergence, cross-scaffold coupling.

Implements the mantic kernel ``M = sum(W_i * L_i * I_i) * f_time / k_n``
directly (no external dependency). Equal weights and unit interaction
coefficients are the defaults; callers can tune thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from cip_protocol.health.scoring import LAYER_NAMES, score_scaffold_layers
from cip_protocol.mantic_adapter import Backend, detect as adapter_detect
from cip_protocol.scaffold.models import Scaffold

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaffoldHealthResult:
    scaffold_id: str
    layers: dict[str, float]
    m_score: float
    coherence: float
    dominant_layer: str
    signal: str
    tension_pairs: list[tuple[str, str, float]]


@dataclass(frozen=True)
class PortfolioHealthResult:
    scaffolds: list[ScaffoldHealthResult]
    coupling: list[tuple[str, str, str, float]]  # (id_a, id_b, layer, score)
    avg_coherence: float
    portfolio_signal: str


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

_NUM_LAYERS = len(LAYER_NAMES)
_EQUAL_WEIGHT = 1.0 / _NUM_LAYERS


def interaction_score(layer_a: float, layer_b: float) -> float:
    """Symmetric interaction between two layer values: ``1 - |a - b|``."""
    return max(0.0, 1.0 - abs(layer_a - layer_b))


def compute_m_score(
    layers: dict[str, float],
    *,
    f_time: float = 1.0,
) -> float:
    """Weighted sum ``M = sum(W_i * L_i) * f_time / k_n`` with equal weights."""
    total = sum(_EQUAL_WEIGHT * layers[name] for name in LAYER_NAMES)
    k_n = math.sqrt(_NUM_LAYERS)
    return total * f_time / k_n


def compute_coherence(layers: dict[str, float], *, divisor: float = 0.5) -> float:
    """``max(0, 1 - stdev(layers) / divisor)``. High when layers are balanced."""
    vals = [layers[name] for name in LAYER_NAMES]
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    sigma = math.sqrt(variance)
    return max(0.0, 1.0 - sigma / divisor)


def detect_signal(
    layers: dict[str, float],
    *,
    detection_threshold: float = 0.4,
) -> str:
    """Return ``'friction_detected'``, ``'emergence_window'``, or ``'baseline'``."""
    vals = [layers[name] for name in LAYER_NAMES]
    spread = max(vals) - min(vals)
    if spread > detection_threshold:
        return "friction_detected"
    if min(vals) > detection_threshold:
        return "emergence_window"
    return "baseline"


def find_tension_pairs(
    layers: dict[str, float],
    *,
    tension_threshold: float = 0.5,
) -> list[tuple[str, str, float]]:
    """Return layer pairs whose agreement (interaction_score) falls below *tension_threshold*."""
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(LAYER_NAMES):
        for b in LAYER_NAMES[i + 1:]:
            agreement = interaction_score(layers[a], layers[b])
            if agreement < tension_threshold:
                pairs.append((a, b, round(agreement, 3)))
    return pairs


def dominant_layer(layers: dict[str, float]) -> str:
    return max(LAYER_NAMES, key=lambda name: layers[name])


# ---------------------------------------------------------------------------
# Per-scaffold analysis
# ---------------------------------------------------------------------------

def analyze_scaffold(
    scaffold: Scaffold,
    *,
    detection_threshold: float = 0.4,
    tension_threshold: float = 0.5,
    coherence_divisor: float = 0.5,
) -> ScaffoldHealthResult:
    layers = score_scaffold_layers(scaffold)
    return ScaffoldHealthResult(
        scaffold_id=scaffold.id,
        layers=layers,
        m_score=compute_m_score(layers),
        coherence=compute_coherence(layers, divisor=coherence_divisor),
        dominant_layer=dominant_layer(layers),
        signal=detect_signal(layers, detection_threshold=detection_threshold),
        tension_pairs=find_tension_pairs(layers, tension_threshold=tension_threshold),
    )


# ---------------------------------------------------------------------------
# Cross-scaffold coupling
# ---------------------------------------------------------------------------

def _cross_scaffold_coupling(
    results: Sequence[ScaffoldHealthResult],
) -> list[tuple[str, str, str, float]]:
    """Same-layer interaction scores between every scaffold pair."""
    coupling: list[tuple[str, str, str, float]] = []
    for i, ra in enumerate(results):
        for rb in results[i + 1:]:
            for layer in LAYER_NAMES:
                score = interaction_score(ra.layers[layer], rb.layers[layer])
                coupling.append((ra.scaffold_id, rb.scaffold_id, layer, round(score, 3)))
    # Sort by score descending so the most coupled pairs appear first.
    coupling.sort(key=lambda t: -t[3])
    return coupling


# ---------------------------------------------------------------------------
# Portfolio analysis
# ---------------------------------------------------------------------------

def analyze_portfolio(
    scaffolds: Sequence[Scaffold],
    *,
    detection_threshold: float = 0.4,
    tension_threshold: float = 0.5,
    coherence_divisor: float = 0.5,
) -> PortfolioHealthResult:
    results = [
        analyze_scaffold(
            s,
            detection_threshold=detection_threshold,
            tension_threshold=tension_threshold,
            coherence_divisor=coherence_divisor,
        )
        for s in scaffolds
    ]
    coupling = _cross_scaffold_coupling(results) if len(results) > 1 else []
    avg_coherence = (
        sum(r.coherence for r in results) / len(results) if results else 0.0
    )
    signals = {r.signal for r in results}
    if signals == {"emergence_window"}:
        portfolio_signal = "portfolio_emergence"
    elif signals == {"friction_detected"}:
        portfolio_signal = "portfolio_friction"
    elif len(signals) > 1:
        portfolio_signal = "portfolio_mixed"
    elif signals == {"baseline"}:
        portfolio_signal = "portfolio_baseline"
    else:
        portfolio_signal = "portfolio_empty"

    return PortfolioHealthResult(
        scaffolds=results,
        coupling=coupling,
        avg_coherence=round(avg_coherence, 3),
        portfolio_signal=portfolio_signal,
    )


# ---------------------------------------------------------------------------
# Backend-aware variants (delegate to mantic_adapter)
# ---------------------------------------------------------------------------

_HEALTH_HIERARCHY = {
    "micro": "Micro",
    "meso": "Meso",
    "macro": "Macro",
    "meta": "Meta",
}


def analyze_scaffold_with_backend(
    scaffold: Scaffold,
    *,
    backend: Backend = "auto",
    detection_threshold: float = 0.4,
    tension_threshold: float = 0.5,
    coherence_divisor: float = 0.5,
    domain_name: str = "cip_health",
    layer_hierarchy: dict[str, str] | None = None,
    temporal_config: dict[str, Any] | None = None,
) -> ScaffoldHealthResult:
    """Like :func:`analyze_scaffold` but routes through the mantic adapter."""
    layers = score_scaffold_layers(scaffold)
    result = adapter_detect(
        layer_names=list(LAYER_NAMES),
        layer_values=[layers[n] for n in LAYER_NAMES],
        backend=backend,
        mode="friction",
        detection_threshold=detection_threshold,
        tension_threshold=tension_threshold,
        coherence_divisor=coherence_divisor,
        domain_name=domain_name,
        layer_hierarchy=layer_hierarchy or _HEALTH_HIERARCHY,
        temporal_config=temporal_config,
    )
    return ScaffoldHealthResult(
        scaffold_id=scaffold.id,
        layers=layers,
        m_score=result.m_score,
        coherence=result.coherence,
        dominant_layer=result.dominant_layer,
        signal=result.signal,
        tension_pairs=result.tension_pairs,
    )


def analyze_portfolio_with_backend(
    scaffolds: Sequence[Scaffold],
    *,
    backend: Backend = "auto",
    detection_threshold: float = 0.4,
    tension_threshold: float = 0.5,
    coherence_divisor: float = 0.5,
    domain_name: str = "cip_health",
    layer_hierarchy: dict[str, str] | None = None,
    temporal_config: dict[str, Any] | None = None,
) -> PortfolioHealthResult:
    """Like :func:`analyze_portfolio` but routes through the mantic adapter."""
    results = [
        analyze_scaffold_with_backend(
            s,
            backend=backend,
            detection_threshold=detection_threshold,
            tension_threshold=tension_threshold,
            coherence_divisor=coherence_divisor,
            domain_name=domain_name,
            layer_hierarchy=layer_hierarchy,
            temporal_config=temporal_config,
        )
        for s in scaffolds
    ]
    coupling = _cross_scaffold_coupling(results) if len(results) > 1 else []
    avg_coherence = (
        sum(r.coherence for r in results) / len(results) if results else 0.0
    )
    signals = {r.signal for r in results}
    if signals == {"emergence_window"}:
        portfolio_signal = "portfolio_emergence"
    elif signals == {"friction_detected"}:
        portfolio_signal = "portfolio_friction"
    elif len(signals) > 1:
        portfolio_signal = "portfolio_mixed"
    elif signals == {"baseline"}:
        portfolio_signal = "portfolio_baseline"
    else:
        portfolio_signal = "portfolio_empty"

    return PortfolioHealthResult(
        scaffolds=results,
        coupling=coupling,
        avg_coherence=round(avg_coherence, 3),
        portfolio_signal=portfolio_signal,
    )
