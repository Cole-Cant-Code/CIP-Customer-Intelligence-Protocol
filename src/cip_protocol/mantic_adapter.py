"""Mantic detection adapter — unified interface to CIP-native and mantic-thinking backends.

CIP's own M-kernel (``NativeBackend``) reimplements the formula
``M = sum(W_i * L_i) * f_time / sqrt(N)`` with equal weights and unit
interaction coefficients.  When ``mantic-thinking`` is installed, the
``ManticThinkingBackend`` delegates to its richer engine (custom weights,
interaction tuning, temporal kernels, governance audit trails).

Usage::

    from cip_protocol.mantic_adapter import detect, get_backend

    result = detect(layer_names=["a", "b", "c"], layer_values=[0.8, 0.3, 0.6])
    backend = get_backend("auto")  # returns mantic if installed, else native
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Backend = Literal["auto", "cip_native", "mantic"]

_MANTIC_AVAILABLE: bool | None = None  # lazy probe cache


@dataclass(frozen=True)
class DetectionResult:
    """Unified result from either backend."""

    m_score: float
    spatial_component: float
    layer_attribution: dict[str, float]
    signal: str  # "friction_detected" | "emergence_window" | "baseline"
    dominant_layer: str
    coherence: float
    tension_pairs: list[tuple[str, str, float]]
    backend_used: str  # "cip_native" | "mantic"
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Native backend — no external deps
# ---------------------------------------------------------------------------


class NativeBackend:
    """CIP's own M-kernel: ``M = sum(W_i * L_i) * f_time / sqrt(N)``.

    Equal weights, unit interaction coefficients.  Matches the existing
    ``health.analysis`` primitives exactly.
    """

    def detect(
        self,
        *,
        layer_names: list[str] | tuple[str, ...],
        layer_values: list[float] | tuple[float, ...],
        weights: list[float] | tuple[float, ...] | None = None,
        mode: str = "friction",
        f_time: float = 1.0,
        detection_threshold: float = 0.4,
        tension_threshold: float = 0.5,
        coherence_divisor: float = 0.5,
        **kwargs: Any,
    ) -> DetectionResult:
        n = len(layer_names)
        if n != len(layer_values):
            raise ValueError("layer_names and layer_values must have the same length")
        if n < 2:
            raise ValueError("at least 2 layers required")
        if mode not in {"friction", "emergence"}:
            raise ValueError("mode must be 'friction' or 'emergence'")

        w = list(weights) if weights is not None else [1.0 / n] * n

        # M = sum(W_i * L_i) * f_time / sqrt(N)
        total = sum(w[i] * layer_values[i] for i in range(n))
        k_n = math.sqrt(n)
        m_score = total * f_time / k_n
        spatial = total

        # Attribution (percentage of spatial component)
        contributions = [w[i] * layer_values[i] for i in range(n)]
        attr_total = sum(contributions) or 1.0
        attribution = {
            layer_names[i]: round(contributions[i] / attr_total * 100, 1)
            for i in range(n)
        }

        # Coherence
        vals = list(layer_values)
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        sigma = math.sqrt(variance)
        coherence = max(0.0, 1.0 - sigma / coherence_divisor)

        # Signal
        spread = max(vals) - min(vals)
        if mode == "friction":
            if spread > detection_threshold:
                signal = "friction_detected"
            elif min(vals) > detection_threshold:
                signal = "emergence_window"
            else:
                signal = "baseline"
        elif min(vals) > detection_threshold:
            signal = "emergence_window"
        else:
            signal = "baseline"

        # Dominant layer
        dominant = layer_names[vals.index(max(vals))]

        # Tension pairs
        tension_pairs: list[tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                agreement = max(0.0, 1.0 - abs(vals[i] - vals[j]))
                if agreement < tension_threshold:
                    tension_pairs.append(
                        (layer_names[i], layer_names[j], round(agreement, 3))
                    )

        return DetectionResult(
            m_score=round(m_score, 6),
            spatial_component=round(spatial, 6),
            layer_attribution=attribution,
            signal=signal,
            dominant_layer=dominant,
            coherence=round(coherence, 6),
            tension_pairs=tension_pairs,
            backend_used="cip_native",
        )


# ---------------------------------------------------------------------------
# Mantic backend — delegates to mantic_thinking
# ---------------------------------------------------------------------------


class ManticThinkingBackend:
    """Delegates to ``mantic_thinking.tools.generic_detect.detect``."""

    def detect(
        self,
        *,
        layer_names: list[str] | tuple[str, ...],
        layer_values: list[float] | tuple[float, ...],
        weights: list[float] | tuple[float, ...] | None = None,
        mode: str = "friction",
        f_time: float = 1.0,
        detection_threshold: float = 0.4,
        tension_threshold: float = 0.5,
        coherence_divisor: float = 0.5,
        domain_name: str = "cip",
        layer_hierarchy: dict[str, str] | None = None,
        temporal_config: dict[str, Any] | None = None,
        interaction_mode: str = "dynamic",
        interaction_override: list[float] | dict[str, float] | None = None,
        interaction_override_mode: str = "scale",
        **kwargs: Any,
    ) -> DetectionResult:
        from mantic_thinking.tools.generic_detect import detect as mantic_detect

        names = list(layer_names)
        vals = list(layer_values)
        n = len(names)
        w = list(weights) if weights is not None else [1.0 / n] * n

        raw = mantic_detect(
            domain_name=domain_name,
            layer_names=names,
            weights=w,
            layer_values=vals,
            mode=mode,
            f_time=f_time,
            detection_threshold=detection_threshold,
            layer_hierarchy=layer_hierarchy,
            temporal_config=temporal_config,
            interaction_mode=interaction_mode,
            interaction_override=interaction_override,
            interaction_override_mode=interaction_override_mode,
        )

        # Map mantic signal to CIP signal vocabulary
        if mode == "friction":
            signal = "friction_detected" if raw.get("alert") is not None else "baseline"
        else:
            if raw.get("window_detected"):
                signal = "emergence_window"
            else:
                signal = "baseline"

        # Check if emergence layers are all above threshold but no spread
        # (mantic marks window_detected based on min > threshold, which aligns)
        # For friction: also check if we're in emergence territory
        if mode == "friction" and signal == "baseline":
            if min(vals) > detection_threshold:
                signal = "emergence_window"

        # Dominant layer from attribution
        attr = raw.get("layer_attribution", {})
        dominant = max(attr, key=attr.get, default=names[0]) if attr else names[0]

        # Coherence — compute locally (mantic's layer_coupling uses a different algorithm)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        sigma = math.sqrt(variance)
        coherence = max(0.0, 1.0 - sigma / coherence_divisor)

        # Tension pairs — compute locally for consistency with native backend
        tension_pairs: list[tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                agreement = max(0.0, 1.0 - abs(vals[i] - vals[j]))
                if agreement < tension_threshold:
                    tension_pairs.append((names[i], names[j], round(agreement, 3)))

        return DetectionResult(
            m_score=round(raw["m_score"], 6),
            spatial_component=round(raw["spatial_component"], 6),
            layer_attribution=attr,
            signal=signal,
            dominant_layer=dominant,
            coherence=round(coherence, 6),
            tension_pairs=tension_pairs,
            backend_used="mantic",
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _probe_mantic() -> bool:
    """Check whether ``mantic_thinking`` is importable (cached)."""
    global _MANTIC_AVAILABLE
    if _MANTIC_AVAILABLE is None:
        try:
            import mantic_thinking.tools.generic_detect  # noqa: F401

            _MANTIC_AVAILABLE = True
        except ImportError:
            _MANTIC_AVAILABLE = False
    return _MANTIC_AVAILABLE


def get_backend(backend: Backend = "auto") -> NativeBackend | ManticThinkingBackend:
    """Return the requested detection backend.

    Parameters
    ----------
    backend:
        ``"auto"`` — mantic if installed, else native.
        ``"cip_native"`` — always the built-in implementation.
        ``"mantic"`` — raises ``ImportError`` if mantic is not installed.
    """
    if backend == "cip_native":
        return NativeBackend()
    if backend == "mantic":
        if not _probe_mantic():
            raise ImportError(
                "mantic-thinking is not installed. "
                "Install it with: pip install 'cip-protocol[mantic]'"
            )
        return ManticThinkingBackend()
    # auto
    if _probe_mantic():
        return ManticThinkingBackend()
    return NativeBackend()


def detect(
    *,
    layer_names: list[str] | tuple[str, ...],
    layer_values: list[float] | tuple[float, ...],
    backend: Backend = "auto",
    **kwargs: Any,
) -> DetectionResult:
    """Convenience: detect using the specified backend (default: auto)."""
    return get_backend(backend).detect(
        layer_names=layer_names, layer_values=layer_values, **kwargs
    )


# ---------------------------------------------------------------------------
# Domain-specific convenience wrappers
# ---------------------------------------------------------------------------


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def detect_safety_friction(
    *,
    layer_values: list[float] | tuple[float, ...],
    backend: Backend = "auto",
    detection_threshold: float = 0.4,
) -> DetectionResult:
    """Safety evaluator friction detection for CIP guardrail content.

    Layer order: escalation_density, prohibited_density, topic_sensitivity,
    response_length_risk.
    """
    clamped = [_clamp(v) for v in layer_values]
    return detect(
        layer_names=[
            "escalation_density",
            "prohibited_density",
            "topic_sensitivity",
            "response_length_risk",
        ],
        layer_values=clamped,
        weights=[0.30, 0.35, 0.20, 0.15],
        mode="friction",
        domain_name="cip_safety",
        backend=backend,
        detection_threshold=detection_threshold,
    )


def detect_policy_conflict(
    *,
    layer_values: list[float] | tuple[float, ...],
    backend: Backend = "auto",
    detection_threshold: float = 0.4,
) -> DetectionResult:
    """RunPolicy conflict detection — surfaces tension between policy fields.

    Layer order: creativity, constraint_strictness, safety_priority, verbosity.
    """
    clamped = [_clamp(v) for v in layer_values]
    return detect(
        layer_names=[
            "creativity",
            "constraint_strictness",
            "safety_priority",
            "verbosity",
        ],
        layer_values=clamped,
        weights=[0.30, 0.25, 0.25, 0.20],
        mode="friction",
        domain_name="cip_policy",
        backend=backend,
        detection_threshold=detection_threshold,
    )


# ---------------------------------------------------------------------------
# Argument structure analysis
# ---------------------------------------------------------------------------

_ARGUMENT_LAYERS = [
    "premise_strength",
    "inferential_link",
    "structural_validity",
    "scope_consistency",
]

_ARGUMENT_WEIGHTS = [0.25, 0.30, 0.25, 0.20]


def detect_argument_friction(
    *,
    layer_values: list[float] | tuple[float, ...],
    backend: Backend = "auto",
    detection_threshold: float = 0.4,
) -> DetectionResult:
    """Argument structure friction detection — surfaces fallacy patterns.

    Layer order: premise_strength, inferential_link, structural_validity,
    scope_consistency.  Weights: [0.25, 0.30, 0.25, 0.20] — inferential link
    highest because most fallacies break the premise-to-conclusion link.
    """
    if len(layer_values) != 4:
        raise ValueError("exactly 4 layer values required for argument analysis")
    clamped = [_clamp(v) for v in layer_values]
    return detect(
        layer_names=_ARGUMENT_LAYERS,
        layer_values=clamped,
        weights=_ARGUMENT_WEIGHTS,
        mode="friction",
        domain_name="cip_argument",
        backend=backend,
        detection_threshold=detection_threshold,
    )


@dataclass(frozen=True)
class FallacyResult:
    """Classification result from argument structure analysis."""

    name: str
    display_name: str
    explanation: str
    is_valid: bool
    confidence: float
    matching_layers: list[str]


# Ordered most-specific first (4-constraint, then 3, then 2).
_FALLACY_SIGNATURES: list[dict[str, Any]] = [
    # 4-constraint signatures
    {
        "name": "straw_man",
        "display_name": "Straw Man",
        "explanation": (
            "The argument misrepresents the original position by shifting "
            "scope while maintaining internal logic."
        ),
        "predicate": lambda v: (
            v["premise_strength"] > 0.4
            and v["inferential_link"] > 0.4
            and v["structural_validity"] > 0.4
            and v["scope_consistency"] < 0.3
        ),
    },
    {
        "name": "false_dilemma",
        "display_name": "False Dilemma",
        "explanation": (
            "The argument presents a limited set of options while ignoring "
            "valid alternatives."
        ),
        "predicate": lambda v: (
            v["premise_strength"] > 0.5
            and v["inferential_link"] > 0.4
            and v["structural_validity"] < 0.4
            and v["scope_consistency"] < 0.3
        ),
    },
    # 3-constraint signatures
    {
        "name": "affirming_the_consequent",
        "display_name": "Affirming the Consequent",
        "explanation": (
            "The argument assumes that because the consequent is true, "
            "the antecedent must be true."
        ),
        "predicate": lambda v: (
            v["premise_strength"] > 0.6
            and v["inferential_link"] < 0.3
            and v["structural_validity"] < 0.35
        ),
    },
    {
        "name": "circular_reasoning",
        "display_name": "Circular Reasoning",
        "explanation": "The conclusion is assumed in the premises, creating a logical circle.",
        "predicate": lambda v: (
            v["inferential_link"] > 0.6
            and v["structural_validity"] < 0.3
            and v["premise_strength"] < 0.5
        ),
    },
    {
        "name": "appeal_to_authority",
        "display_name": "Appeal to Authority",
        "explanation": (
            "The argument relies on authority rather than evidence to support "
            "the premises."
        ),
        "predicate": lambda v: (
            v["premise_strength"] < 0.3
            and v["inferential_link"] > 0.5
            and v["structural_validity"] > 0.5
        ),
    },
    # 2-constraint signatures
    {
        "name": "hasty_generalization",
        "display_name": "Hasty Generalization",
        "explanation": (
            "The argument draws a broad conclusion from insufficient or "
            "unrepresentative evidence."
        ),
        "predicate": lambda v: 0.4 < v["premise_strength"] < 0.7 and v["scope_consistency"] < 0.3,
    },
    {
        "name": "non_sequitur",
        "display_name": "Non Sequitur",
        "explanation": "The conclusion does not logically follow from the premises.",
        "predicate": lambda v: (
            v["premise_strength"] > 0.4
            and v["inferential_link"] < 0.2
            and v["structural_validity"] > 0.4
        ),
    },
]


def classify_fallacy(
    result: DetectionResult,
    layer_values: dict[str, float],
) -> FallacyResult:
    """Classify a fallacy from a detection result and its layer values.

    If no friction is detected, the argument is considered valid.
    If friction is detected but no signature matches, returns
    ``unclassified_friction``.
    """
    if result.signal != "friction_detected":
        return FallacyResult(
            name="valid_argument",
            display_name="Valid Argument",
            explanation="No structural friction detected; the argument form is consistent.",
            is_valid=True,
            confidence=max(0.0, min(1.0, result.coherence)),
            matching_layers=[],
        )

    for sig in _FALLACY_SIGNATURES:
        if sig["predicate"](layer_values):
            # Confidence from coherence inverted: lower coherence = higher
            # confidence that something is wrong.
            raw_conf = max(0.0, min(1.0, 1.0 - result.coherence))
            matching = [
                name for name in _ARGUMENT_LAYERS
                if name in layer_values
            ]
            return FallacyResult(
                name=sig["name"],
                display_name=sig["display_name"],
                explanation=sig["explanation"],
                is_valid=False,
                confidence=raw_conf,
                matching_layers=matching,
            )

    # Friction detected but no known signature matched.
    return FallacyResult(
        name="unclassified_friction",
        display_name="Unclassified Friction",
        explanation="Structural friction detected but no known fallacy pattern matched.",
        is_valid=False,
        confidence=max(0.0, min(1.0, 1.0 - result.coherence)),
        matching_layers=list(layer_values.keys()),
    )
