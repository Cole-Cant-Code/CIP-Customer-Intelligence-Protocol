"""Parity tests: CIP native and mantic backends produce consistent signals.

These tests confirm that for identical scaffold inputs, both backends classify
signals the same way — the contract documented in
``docs/contracts/matcher-mantic-alignment.md``.
"""

from __future__ import annotations

import pytest

from cip_protocol.health.analysis import (
    analyze_scaffold,
    analyze_scaffold_with_backend,
)
from cip_protocol.mantic_adapter import NativeBackend, _probe_mantic
from cip_protocol.scaffold.models import (
    Scaffold,
    ScaffoldApplicability,
    ScaffoldFraming,
    ScaffoldGuardrails,
    ScaffoldOutputCalibration,
)
from tests.conftest import make_test_scaffold

_HAS_MANTIC = _probe_mantic()
skip_no_mantic = pytest.mark.skipif(not _HAS_MANTIC, reason="mantic-thinking not installed")


# ---------------------------------------------------------------------------
# Scaffold fixtures covering all three signal categories
# ---------------------------------------------------------------------------

def _friction_scaffold():
    """High micro, low meta → spread > 0.4 → friction_detected."""
    return make_test_scaffold(
        scaffold_id="friction_example",
        tools=["t1", "t2", "t3", "t4", "t5"],
        keywords=["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"],
        intent_signals=["i1", "i2"],
        disclaimers=[],
        escalation_triggers=[],
        prohibited_actions=[],
    )


def _emergence_scaffold():
    """All layers above threshold → emergence_window.

    Needs rich content across all four dimensions so every layer > 0.4
    and spread < 0.4.
    """
    return Scaffold(
        id="emergence_example",
        version="1.0",
        domain="test",
        display_name="Emergence Test",
        description="Test scaffold with balanced high layers",
        applicability=ScaffoldApplicability(
            tools=["t1", "t2", "t3", "t4", "t5", "t6"],
            keywords=["k1", "k2", "k3", "k4", "k5", "k6"],
            intent_signals=["i1", "i2", "i3"],
        ),
        framing=ScaffoldFraming(
            role="Test analyst",
            perspective="Analytical",
            tone="neutral",
            tone_variants={"friendly": "Warm"},
        ),
        reasoning_framework={"steps": ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]},
        domain_knowledge_activation=["dk1", "dk2", "dk3", "dk4", "dk5"],
        output_calibration=ScaffoldOutputCalibration(
            format="bullet_points",
            format_options=["bullet_points", "table", "narrative"],
            max_length_guidance="500 words",
            must_include=["key_findings", "recommendations", "caveats"],
            never_include=["legal_advice", "prices"],
        ),
        guardrails=ScaffoldGuardrails(
            disclaimers=["d1", "d2", "d3", "d4"],
            escalation_triggers=["e1", "e2", "e3"],
            prohibited_actions=["p1", "p2", "p3"],
        ),
        data_requirements=[],
    )


def _baseline_scaffold():
    """Minimal scaffold — all layers low → baseline."""
    return make_test_scaffold(scaffold_id="baseline_example")


# ---------------------------------------------------------------------------
# Layer hierarchy alignment
# ---------------------------------------------------------------------------

class TestLayerHierarchyMapping:
    """CIP's 4 layers map 1:1 to mantic's Micro/Meso/Macro/Meta."""

    def test_layer_names_are_four(self):
        from cip_protocol.health.scoring import LAYER_NAMES

        assert LAYER_NAMES == ("micro", "meso", "macro", "meta")

    def test_hierarchy_dict_covers_all_layers(self):
        from cip_protocol.health.analysis import _HEALTH_HIERARCHY
        from cip_protocol.health.scoring import LAYER_NAMES

        for layer in LAYER_NAMES:
            assert layer in _HEALTH_HIERARCHY


# ---------------------------------------------------------------------------
# Signal parity (conditional on mantic)
# ---------------------------------------------------------------------------

class TestSignalParity:
    """Both backends classify identical inputs the same way."""

    @skip_no_mantic
    @pytest.mark.parametrize(
        "scaffold_fn,expected_signal",
        [
            (_friction_scaffold, "friction_detected"),
            (_emergence_scaffold, "emergence_window"),
            (_baseline_scaffold, "baseline"),
        ],
    )
    def test_signal_consistency(self, scaffold_fn, expected_signal):
        scaffold = scaffold_fn()
        native = analyze_scaffold_with_backend(scaffold, backend="cip_native")
        mantic = analyze_scaffold_with_backend(scaffold, backend="mantic")

        assert native.signal == expected_signal, f"Native: expected {expected_signal}, got {native.signal}"
        assert mantic.signal == expected_signal, f"Mantic: expected {expected_signal}, got {mantic.signal}"

    @skip_no_mantic
    def test_dominant_layer_consistent(self):
        scaffold = _friction_scaffold()
        native = analyze_scaffold_with_backend(scaffold, backend="cip_native")
        mantic = analyze_scaffold_with_backend(scaffold, backend="mantic")
        assert native.dominant_layer == mantic.dominant_layer

    @skip_no_mantic
    def test_coherence_consistent(self):
        """Coherence uses same formula in both backends."""
        scaffold = _emergence_scaffold()
        native = analyze_scaffold_with_backend(scaffold, backend="cip_native")
        mantic = analyze_scaffold_with_backend(scaffold, backend="mantic")
        assert native.coherence == pytest.approx(mantic.coherence, abs=1e-5)

    @skip_no_mantic
    def test_tension_pairs_consistent(self):
        scaffold = _friction_scaffold()
        native = analyze_scaffold_with_backend(scaffold, backend="cip_native")
        mantic = analyze_scaffold_with_backend(scaffold, backend="mantic")
        assert native.tension_pairs == mantic.tension_pairs


# ---------------------------------------------------------------------------
# Native-only tests (always run)
# ---------------------------------------------------------------------------

class TestNativeSignals:
    """Verify native backend signal classification for scaffold archetypes."""

    def test_friction(self):
        result = analyze_scaffold(_friction_scaffold())
        assert result.signal == "friction_detected"

    def test_emergence(self):
        result = analyze_scaffold(_emergence_scaffold())
        assert result.signal == "emergence_window"

    def test_baseline(self):
        result = analyze_scaffold(_baseline_scaffold())
        assert result.signal == "baseline"
