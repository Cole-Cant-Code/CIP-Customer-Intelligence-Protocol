"""Tests for cip_protocol.engagement.layered_scoring."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cip_protocol.engagement.scoring import (
    LeadEvent,
    LeadScoringConfig,
    compute_lead_score,
)
from cip_protocol.engagement.layered_scoring import (
    LayerMapping,
    LayeredScoreResult,
    score_lead_with_layers,
)
from cip_protocol.mantic_adapter import DetectionResult


# ---------------------------------------------------------------------------
# Mock LayerMapping
# ---------------------------------------------------------------------------

class SimpleLayerMapping:
    """Test mapping: 3 layers derived from event counts and recency."""

    @property
    def layer_names(self) -> tuple[str, ...]:
        return ("recency", "volume", "intent")

    @property
    def weights(self) -> tuple[float, ...]:
        return (0.35, 0.30, 0.35)

    def events_to_layers(
        self,
        events: list[LeadEvent],
        now: datetime,
        config: LeadScoringConfig,
    ) -> dict[str, float]:
        if not events:
            return {"recency": 0.0, "volume": 0.0, "intent": 0.0}

        # Recency: how recent is the latest event (0-1, 1 = within 1 day)
        latest = max(ev.created_at for ev in events)
        age_days = max(0.0, (now - latest).total_seconds() / 86_400)
        recency = max(0.0, 1.0 - age_days / 30.0)

        # Volume: event count normalized to cap of 10
        volume = min(1.0, len(events) / 10.0)

        # Intent: proportion of high-value actions
        high_value = {"test_drive", "reserve_vehicle", "purchase_deposit"}
        high_count = sum(1 for ev in events if ev.action in high_value)
        intent = high_count / len(events) if events else 0.0

        return {"recency": recency, "volume": volume, "intent": intent}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEST_CONFIG = LeadScoringConfig(
    action_weights={
        "viewed": 1.0,
        "compared": 3.0,
        "test_drive": 8.0,
        "reserve_vehicle": 9.0,
        "purchase_deposit": 10.0,
    },
    status_thresholds=[(0, "new"), (10, "engaged"), (22, "qualified")],
    recency_bands=[(1, 1.0), (7, 0.70), (30, 0.30)],
    recency_default=0.0,
    terminal_statuses=frozenset({"won", "lost"}),
    scoring_window_days=30,
)

NOW = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)

SAMPLE_EVENTS = [
    LeadEvent("viewed", datetime(2025, 6, 15, 10, 0, tzinfo=timezone.utc)),
    LeadEvent("compared", datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc)),
    LeadEvent("test_drive", datetime(2025, 6, 13, 9, 0, tzinfo=timezone.utc)),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLayerMapping:
    def test_protocol_compliance(self):
        mapping = SimpleLayerMapping()
        assert isinstance(mapping, LayerMapping)

    def test_events_to_layers(self):
        mapping = SimpleLayerMapping()
        layers = mapping.events_to_layers(SAMPLE_EVENTS, NOW, TEST_CONFIG)
        assert set(layers.keys()) == {"recency", "volume", "intent"}
        assert all(0 <= v <= 1.0 for v in layers.values())

    def test_empty_events(self):
        mapping = SimpleLayerMapping()
        layers = mapping.events_to_layers([], NOW, TEST_CONFIG)
        assert all(v == 0.0 for v in layers.values())


class TestScoreLeadWithLayers:
    def test_flat_score_matches(self):
        result = score_lead_with_layers(
            events=SAMPLE_EVENTS,
            now=NOW,
            config=TEST_CONFIG,
            mapping=SimpleLayerMapping(),
            backend="cip_native",
        )
        expected_flat = compute_lead_score(SAMPLE_EVENTS, NOW, TEST_CONFIG)
        assert result.flat_score == expected_flat

    def test_detection_is_valid(self):
        result = score_lead_with_layers(
            events=SAMPLE_EVENTS,
            now=NOW,
            config=TEST_CONFIG,
            mapping=SimpleLayerMapping(),
            backend="cip_native",
        )
        assert isinstance(result.detection, DetectionResult)
        assert result.detection.backend_used == "cip_native"

    def test_delta_calculation(self):
        result = score_lead_with_layers(
            events=SAMPLE_EVENTS,
            now=NOW,
            config=TEST_CONFIG,
            mapping=SimpleLayerMapping(),
            backend="cip_native",
        )
        expected_delta = round(result.flat_score - result.detection.m_score, 6)
        assert result.delta == pytest.approx(expected_delta, abs=1e-6)

    def test_layer_values_populated(self):
        result = score_lead_with_layers(
            events=SAMPLE_EVENTS,
            now=NOW,
            config=TEST_CONFIG,
            mapping=SimpleLayerMapping(),
            backend="cip_native",
        )
        assert set(result.layer_values.keys()) == {"recency", "volume", "intent"}

    def test_result_is_frozen(self):
        result = score_lead_with_layers(
            events=SAMPLE_EVENTS,
            now=NOW,
            config=TEST_CONFIG,
            mapping=SimpleLayerMapping(),
            backend="cip_native",
        )
        with pytest.raises(AttributeError):
            result.flat_score = 999  # type: ignore[misc]

    def test_empty_events(self):
        result = score_lead_with_layers(
            events=[],
            now=NOW,
            config=TEST_CONFIG,
            mapping=SimpleLayerMapping(),
            backend="cip_native",
        )
        assert result.flat_score == 0.0
        assert result.detection.m_score == 0.0

    def test_emergence_mode(self):
        """Layered scoring produces a valid signal in emergence mode."""
        high_events = [
            LeadEvent("test_drive", datetime(2025, 6, 15, 10, 0, tzinfo=timezone.utc)),
            LeadEvent("reserve_vehicle", datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc)),
            LeadEvent("purchase_deposit", datetime(2025, 6, 15, 11, 30, tzinfo=timezone.utc)),
            LeadEvent("viewed", datetime(2025, 6, 15, 9, 0, tzinfo=timezone.utc)),
            LeadEvent("compared", datetime(2025, 6, 15, 9, 30, tzinfo=timezone.utc)),
        ]
        result = score_lead_with_layers(
            events=high_events,
            now=NOW,
            config=TEST_CONFIG,
            mapping=SimpleLayerMapping(),
            backend="cip_native",
            mode="emergence",
        )
        # With 5 events: recency ~1.0, volume 0.5, intent 0.6 → spread > 0.4
        # so friction_detected is expected. Signal is always valid.
        assert result.detection.signal in {
            "friction_detected", "emergence_window", "baseline"
        }
        assert result.flat_score > 0
