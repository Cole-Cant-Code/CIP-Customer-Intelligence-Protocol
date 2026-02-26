"""Layered lead scoring — bridge between flat event scoring and M-layer detection.

CIP provides the pipeline; domain code defines layer semantics via
:class:`LayerMapping`.  This module composes the existing
:func:`compute_lead_score` with the mantic adapter to produce a combined
result that includes both the familiar flat score and multi-signal detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from cip_protocol.engagement.scoring import (
    LeadEvent,
    LeadScoringConfig,
    compute_lead_score,
)
from cip_protocol.mantic_adapter import Backend, DetectionResult, detect as adapter_detect


@runtime_checkable
class LayerMapping(Protocol):
    """Domain implementations define what layers mean.

    A ``LayerMapping`` converts a list of lead events into normalized layer
    values suitable for multi-signal detection.  Each domain (automotive,
    real-estate, etc.) implements its own mapping.
    """

    @property
    def layer_names(self) -> tuple[str, ...]: ...

    @property
    def weights(self) -> tuple[float, ...]: ...

    def events_to_layers(
        self,
        events: list[LeadEvent],
        now: datetime,
        config: LeadScoringConfig,
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class LayeredScoreResult:
    """Combined flat + layered detection result."""

    flat_score: float
    detection: DetectionResult
    layer_values: dict[str, float]
    delta: float  # flat_score - detection.m_score


def score_lead_with_layers(
    *,
    events: list[LeadEvent],
    now: datetime,
    config: LeadScoringConfig,
    mapping: LayerMapping,
    backend: Backend = "auto",
    mode: str = "emergence",
    detection_threshold: float = 0.4,
    domain_name: str = "cip_engagement",
) -> LayeredScoreResult:
    """Score a lead using both flat scoring and layered detection.

    Parameters
    ----------
    events:
        Chronological engagement events.
    now:
        Reference timestamp for recency calculations.
    config:
        Domain scoring config (weights, thresholds, recency bands).
    mapping:
        Domain-specific ``LayerMapping`` that converts events to layer values.
    backend:
        Detection backend — ``"auto"``, ``"cip_native"``, or ``"mantic"``.
    mode:
        Detection mode — ``"emergence"`` (default for leads) or ``"friction"``.
    detection_threshold:
        Threshold for signal classification.
    domain_name:
        Domain identifier passed to the mantic adapter.
    """
    flat_score = compute_lead_score(events, now, config)
    layer_values = mapping.events_to_layers(events, now, config)
    detection = adapter_detect(
        layer_names=list(mapping.layer_names),
        layer_values=[layer_values[n] for n in mapping.layer_names],
        weights=list(mapping.weights),
        backend=backend,
        mode=mode,
        detection_threshold=detection_threshold,
        domain_name=domain_name,
    )
    return LayeredScoreResult(
        flat_score=flat_score,
        detection=detection,
        layer_values=layer_values,
        delta=round(flat_score - detection.m_score, 6),
    )
