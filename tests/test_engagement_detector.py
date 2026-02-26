"""Tests for cip_protocol.engagement.detector."""

from __future__ import annotations

from cip_protocol.engagement.detector import (
    EscalationConfig,
    EscalationDetector,
    check_escalation,
)

_AUTO_TRANSITIONS = {
    ("new", "engaged"): "cold_to_warm",
    ("new", "qualified"): "cold_to_hot",
    ("engaged", "qualified"): "warm_to_hot",
}

_DEFAULT_CONFIG = EscalationConfig(
    transitions=_AUTO_TRANSITIONS,
    entity_id_field="vehicle_id",
)


def _check(**overrides):
    defaults = {
        "config": _DEFAULT_CONFIG,
        "lead_id": "lead-1",
        "old_status": "new",
        "new_status": "engaged",
        "score": 15.0,
        "entity_id": "v-100",
        "customer_name": "Alice",
        "customer_contact": "alice@test.com",
        "source_channel": "web",
        "action": "viewed",
    }
    defaults.update(overrides)
    return check_escalation(**defaults)


class TestCheckEscalation:
    def test_cold_to_warm(self):
        esc = _check(old_status="new", new_status="engaged")
        assert esc is not None
        assert esc["escalation_type"] == "cold_to_warm"

    def test_cold_to_hot(self):
        esc = _check(old_status="new", new_status="qualified")
        assert esc is not None
        assert esc["escalation_type"] == "cold_to_hot"

    def test_warm_to_hot(self):
        esc = _check(old_status="engaged", new_status="qualified")
        assert esc is not None
        assert esc["escalation_type"] == "warm_to_hot"

    def test_same_status_returns_none(self):
        assert _check(old_status="new", new_status="new") is None

    def test_unknown_transition_returns_none(self):
        assert _check(old_status="qualified", new_status="won") is None

    def test_escalation_has_required_fields(self):
        esc = _check()
        assert esc["id"].startswith("esc-")
        assert esc["lead_id"] == "lead-1"
        assert esc["score"] == 15.0
        assert esc["vehicle_id"] == "v-100"
        assert "created_at" in esc

    def test_custom_entity_id_field(self):
        config = EscalationConfig(
            transitions=_AUTO_TRANSITIONS,
            entity_id_field="property_id",
        )
        esc = check_escalation(
            config=config,
            lead_id="lead-1",
            old_status="new",
            new_status="engaged",
            score=10.0,
            entity_id="prop-42",
        )
        assert esc is not None
        assert esc["property_id"] == "prop-42"
        assert "vehicle_id" not in esc

    def test_empty_transitions_never_escalates(self):
        config = EscalationConfig(transitions={})
        esc = check_escalation(
            config=config,
            lead_id="lead-1",
            old_status="new",
            new_status="engaged",
            score=10.0,
            entity_id="x",
        )
        assert esc is None

    def test_callbacks_fired(self):
        received = []
        esc = _check(callbacks=[received.append])
        assert len(received) == 1
        assert received[0] is esc

    def test_callback_error_does_not_propagate(self):
        def bad_cb(e):
            raise RuntimeError("boom")

        esc = _check(callbacks=[bad_cb])
        assert esc is not None  # still returns the escalation


class TestEscalationDetector:
    def test_lifecycle(self):
        detector = EscalationDetector(_DEFAULT_CONFIG)
        received = []
        detector.register_callback(received.append)

        esc = detector.check(
            lead_id="lead-1",
            old_status="new",
            new_status="engaged",
            score=12.0,
            entity_id="v-1",
        )
        assert esc is not None
        assert len(received) == 1

        detector.clear_callbacks()
        detector.check(
            lead_id="lead-2",
            old_status="new",
            new_status="qualified",
            score=25.0,
            entity_id="v-2",
        )
        assert len(received) == 1  # no new callbacks

    def test_config_accessible(self):
        detector = EscalationDetector(_DEFAULT_CONFIG)
        assert detector.config is _DEFAULT_CONFIG
