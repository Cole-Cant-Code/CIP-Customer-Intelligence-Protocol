"""Tests for cip_protocol.engagement.scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cip_protocol.engagement.scoring import (
    LeadEvent,
    LeadScoringConfig,
    compute_lead_score,
    infer_lead_status,
    lead_score_band,
    recency_multiplier,
)

# Mirror AutoCIP's exact configuration for parity testing.
AUTO_CONFIG = LeadScoringConfig(
    action_weights={
        "viewed": 1.0,
        "compared": 3.0,
        "financed": 6.0,
        "availability_check": 5.0,
        "test_drive": 8.0,
        "reserve_vehicle": 9.0,
        "contact_dealer": 4.0,
        "purchase_deposit": 10.0,
    },
    status_thresholds=[
        (0, "new"),
        (10, "engaged"),
        (22, "qualified"),
    ],
    recency_bands=[
        (1, 1.0),
        (3, 0.85),
        (7, 0.70),
        (14, 0.50),
        (30, 0.30),
    ],
    recency_default=0.0,
    score_bands=[
        (22, "hot"),
        (10, "warm"),
        (0, "cold"),
    ],
    terminal_statuses=frozenset({"won", "lost"}),
    scoring_window_days=30,
)


class TestRecencyMultiplier:
    """Verify each band boundary matches AutoCIP's _recency_multiplier exactly."""

    @pytest.mark.parametrize(
        "age, expected",
        [
            (0, 1.0),
            (0.5, 1.0),
            (1, 1.0),
            (1.5, 0.85),
            (3, 0.85),
            (5, 0.70),
            (7, 0.70),
            (10, 0.50),
            (14, 0.50),
            (20, 0.30),
            (30, 0.30),
            (31, 0.0),
            (100, 0.0),
        ],
    )
    def test_band(self, age, expected):
        assert recency_multiplier(age, AUTO_CONFIG) == expected


class TestComputeLeadScore:
    def test_empty_events(self):
        now = datetime.now(timezone.utc)
        assert compute_lead_score([], now, AUTO_CONFIG) == 0.0

    def test_single_weighted_event(self):
        now = datetime.now(timezone.utc)
        events = [LeadEvent("purchase_deposit", now)]
        # weight 10.0 * recency 1.0 = 10.0
        assert compute_lead_score(events, now, AUTO_CONFIG) == 10.0

    def test_recency_decay(self):
        now = datetime.now(timezone.utc)
        events = [LeadEvent("purchase_deposit", now - timedelta(days=5))]
        # weight 10.0 * recency 0.70 = 7.0
        assert compute_lead_score(events, now, AUTO_CONFIG) == 7.0

    def test_multiple_events(self):
        now = datetime.now(timezone.utc)
        events = [
            LeadEvent("viewed", now),           # 1.0 * 1.0 = 1.0
            LeadEvent("compared", now),          # 3.0 * 1.0 = 3.0
            LeadEvent("test_drive", now),        # 8.0 * 1.0 = 8.0
        ]
        assert compute_lead_score(events, now, AUTO_CONFIG) == 12.0

    def test_unknown_action_ignored(self):
        now = datetime.now(timezone.utc)
        events = [LeadEvent("unknown_action", now)]
        assert compute_lead_score(events, now, AUTO_CONFIG) == 0.0

    def test_window_exclusion_via_old_events(self):
        now = datetime.now(timezone.utc)
        events = [LeadEvent("purchase_deposit", now - timedelta(days=60))]
        # 60 days old -> recency_default = 0.0
        assert compute_lead_score(events, now, AUTO_CONFIG) == 0.0


class TestInferLeadStatus:
    def test_below_10_is_new(self):
        assert infer_lead_status(5.0, "new", AUTO_CONFIG) == "new"

    def test_at_10_is_engaged(self):
        assert infer_lead_status(10.0, "new", AUTO_CONFIG) == "engaged"

    def test_at_22_is_qualified(self):
        assert infer_lead_status(22.0, "new", AUTO_CONFIG) == "qualified"

    def test_above_22_is_qualified(self):
        assert infer_lead_status(50.0, "new", AUTO_CONFIG) == "qualified"

    def test_terminal_won_preserved(self):
        assert infer_lead_status(5.0, "won", AUTO_CONFIG) == "won"

    def test_terminal_lost_preserved(self):
        assert infer_lead_status(50.0, "lost", AUTO_CONFIG) == "lost"


class TestLeadScoreBand:
    @pytest.mark.parametrize(
        "score, expected",
        [
            (22.0, "hot"),
            (30.0, "hot"),
            (10.0, "warm"),
            (15.0, "warm"),
            (0.0, "cold"),
            (9.99, "cold"),
        ],
    )
    def test_bands(self, score, expected):
        assert lead_score_band(score, AUTO_CONFIG) == expected


class TestCustomConfig:
    def test_custom_weights_and_bands(self):
        config = LeadScoringConfig(
            action_weights={"click": 5.0},
            status_thresholds=[(0, "cold"), (5, "warm"), (20, "hot")],
            recency_bands=[(7, 1.0)],
            recency_default=0.5,
            score_bands=[(20, "blazing"), (5, "toasty"), (0, "chilly")],
        )
        now = datetime.now(timezone.utc)
        events = [LeadEvent("click", now)]
        assert compute_lead_score(events, now, config) == 5.0
        assert infer_lead_status(5.0, "cold", config) == "warm"
        assert lead_score_band(5.0, config) == "toasty"

    def test_recency_default_applied(self):
        config = LeadScoringConfig(
            action_weights={"click": 10.0},
            status_thresholds=[(0, "new")],
            recency_bands=[(1, 1.0)],
            recency_default=0.5,
        )
        now = datetime.now(timezone.utc)
        events = [LeadEvent("click", now - timedelta(days=30))]
        # 30 days old, outside all bands -> recency_default = 0.5
        assert compute_lead_score(events, now, config) == 5.0
