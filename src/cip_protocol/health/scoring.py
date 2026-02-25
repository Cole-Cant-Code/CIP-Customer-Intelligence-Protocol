"""Score a scaffold across four M-layers: micro, meso, macro, meta.

Each scaffold section maps to a layer. Raw counts are divided by a generous
cap and clamped to [0, 1]. Cap-based normalization (not min-max across the
portfolio) makes each score interpretable in isolation.
"""

from __future__ import annotations

from cip_protocol.scaffold.models import Scaffold

LAYER_NAMES = ("micro", "meso", "macro", "meta")

# Caps used for normalization — chosen so a well-rounded scaffold scores ~0.7-0.8,
# and only an exceptionally rich one saturates at 1.0.
_MICRO_CAP = 15
_MESO_CAP = 12
_MACRO_CAP = 12
_META_CAP = 10


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def score_scaffold_layers(scaffold: Scaffold) -> dict[str, float]:
    """Return ``{micro, meso, macro, meta}`` scores in [0, 1] for *scaffold*."""
    app = scaffold.applicability
    micro_raw = len(app.tools) + len(app.keywords) + len(app.intent_signals)

    steps = scaffold.reasoning_framework.get("steps", [])
    dka = scaffold.domain_knowledge_activation
    meso_raw = len(steps) + min(len(dka), 5)

    oc = scaffold.output_calibration
    has_custom_format = int(oc.format != "structured_narrative")
    has_length_guidance = int(bool(oc.max_length_guidance))
    macro_raw = (
        has_custom_format
        + len(oc.format_options)
        + has_length_guidance
        + len(oc.must_include)
        + len(oc.never_include)
    )

    g = scaffold.guardrails
    meta_raw = len(g.disclaimers) + len(g.escalation_triggers) + len(g.prohibited_actions)

    return {
        "micro": _clamp(micro_raw / _MICRO_CAP),
        "meso": _clamp(meso_raw / _MESO_CAP),
        "macro": _clamp(macro_raw / _MACRO_CAP),
        "meta": _clamp(meta_raw / _META_CAP),
    }
