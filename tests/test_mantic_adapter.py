"""Tests for cip_protocol.mantic_adapter — backend parity and factory logic."""

from __future__ import annotations

import math
import os

import pytest

from cip_protocol.mantic_adapter import (
    Backend,
    DetectionResult,
    ManticThinkingBackend,
    NativeBackend,
    get_backend,
    detect,
    _probe_mantic,
)
from cip_protocol.health.analysis import (
    compute_coherence,
    compute_m_score,
    detect_signal,
    dominant_layer,
    find_tension_pairs,
)
from cip_protocol.health.scoring import LAYER_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

UNIFORM_LAYERS = {"micro": 0.5, "meso": 0.5, "macro": 0.5, "meta": 0.5}
HIGH_SPREAD_LAYERS = {"micro": 0.9, "meso": 0.1, "macro": 0.8, "meta": 0.2}
ALL_HIGH_LAYERS = {"micro": 0.8, "meso": 0.7, "macro": 0.9, "meta": 0.75}
ALL_LOW_LAYERS = {"micro": 0.1, "meso": 0.2, "macro": 0.15, "meta": 0.1}

_HAS_MANTIC = _probe_mantic()
skip_no_mantic = pytest.mark.skipif(not _HAS_MANTIC, reason="mantic-thinking not installed")


def test_mantic_required_guard():
    """Fail explicitly when CI requires mantic but dependency is missing."""
    if os.getenv("CIP_REQUIRE_MANTIC", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("CIP_REQUIRE_MANTIC is not enabled")
    assert _HAS_MANTIC, "CIP_REQUIRE_MANTIC=1 but mantic-thinking is not installed"


# ---------------------------------------------------------------------------
# NativeBackend tests
# ---------------------------------------------------------------------------

class TestNativeBackend:
    """NativeBackend must match existing health.analysis primitives exactly."""

    def _run(self, layers: dict[str, float], **kw) -> DetectionResult:
        return NativeBackend().detect(
            layer_names=list(layers.keys()),
            layer_values=list(layers.values()),
            **kw,
        )

    def test_m_score_parity_uniform(self):
        result = self._run(UNIFORM_LAYERS)
        expected = compute_m_score(UNIFORM_LAYERS)
        assert result.m_score == pytest.approx(expected, abs=1e-5)

    def test_m_score_parity_spread(self):
        result = self._run(HIGH_SPREAD_LAYERS)
        expected = compute_m_score(HIGH_SPREAD_LAYERS)
        assert result.m_score == pytest.approx(expected, abs=1e-5)

    def test_coherence_parity_uniform(self):
        result = self._run(UNIFORM_LAYERS)
        expected = compute_coherence(UNIFORM_LAYERS)
        assert result.coherence == pytest.approx(expected, abs=1e-5)

    def test_coherence_parity_spread(self):
        result = self._run(HIGH_SPREAD_LAYERS)
        expected = compute_coherence(HIGH_SPREAD_LAYERS)
        assert result.coherence == pytest.approx(expected, abs=1e-5)

    def test_signal_friction(self):
        result = self._run(HIGH_SPREAD_LAYERS)
        assert result.signal == detect_signal(HIGH_SPREAD_LAYERS)
        assert result.signal == "friction_detected"

    def test_signal_emergence(self):
        result = self._run(ALL_HIGH_LAYERS)
        assert result.signal == detect_signal(ALL_HIGH_LAYERS)
        assert result.signal == "emergence_window"

    def test_signal_baseline(self):
        result = self._run(ALL_LOW_LAYERS)
        assert result.signal == detect_signal(ALL_LOW_LAYERS)
        assert result.signal == "baseline"

    def test_mode_emergence_ignores_friction_path(self):
        layers = {"micro": 0.96, "meso": 0.55, "macro": 0.55, "meta": 0.55}
        result = self._run(layers, mode="emergence")
        assert result.signal == "emergence_window"

    def test_mode_emergence_returns_baseline_when_floor_low(self):
        result = self._run(HIGH_SPREAD_LAYERS, mode="emergence")
        assert result.signal == "baseline"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be 'friction' or 'emergence'"):
            self._run(UNIFORM_LAYERS, mode="invalid")

    def test_dominant_layer(self):
        result = self._run(HIGH_SPREAD_LAYERS)
        assert result.dominant_layer == dominant_layer(HIGH_SPREAD_LAYERS)
        assert result.dominant_layer == "micro"

    def test_tension_pairs_parity(self):
        result = self._run(HIGH_SPREAD_LAYERS)
        expected = find_tension_pairs(HIGH_SPREAD_LAYERS)
        assert result.tension_pairs == expected

    def test_backend_used(self):
        result = self._run(UNIFORM_LAYERS)
        assert result.backend_used == "cip_native"

    def test_attribution_sums_to_100(self):
        result = self._run(HIGH_SPREAD_LAYERS)
        total = sum(result.layer_attribution.values())
        assert total == pytest.approx(100.0, abs=0.5)

    def test_f_time_scales(self):
        base = self._run(UNIFORM_LAYERS, f_time=1.0)
        scaled = self._run(UNIFORM_LAYERS, f_time=2.0)
        assert scaled.m_score == pytest.approx(base.m_score * 2.0, abs=1e-5)

    def test_too_few_layers(self):
        with pytest.raises(ValueError, match="at least 2"):
            NativeBackend().detect(layer_names=["x"], layer_values=[0.5])

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            NativeBackend().detect(layer_names=["a", "b"], layer_values=[0.5])

    def test_three_layers(self):
        """Works with non-standard layer count."""
        result = NativeBackend().detect(
            layer_names=["a", "b", "c"],
            layer_values=[0.8, 0.6, 0.7],
        )
        expected_m = sum([1/3 * v for v in [0.8, 0.6, 0.7]]) / math.sqrt(3)
        assert result.m_score == pytest.approx(expected_m, abs=1e-5)

    def test_raw_is_empty(self):
        result = self._run(UNIFORM_LAYERS)
        assert result.raw == {}


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_cip_native_always_works(self):
        backend = get_backend("cip_native")
        assert isinstance(backend, NativeBackend)

    def test_auto_returns_something(self):
        backend = get_backend("auto")
        assert isinstance(backend, (NativeBackend, ManticThinkingBackend))

    def test_detect_convenience(self):
        result = detect(
            layer_names=["a", "b", "c"],
            layer_values=[0.8, 0.6, 0.7],
            backend="cip_native",
        )
        assert isinstance(result, DetectionResult)
        assert result.backend_used == "cip_native"


# ---------------------------------------------------------------------------
# Signal classification consistency
# ---------------------------------------------------------------------------

class TestSignalClassification:
    """Both backends should classify identical inputs the same way."""

    @pytest.mark.parametrize(
        "layers,expected_signal",
        [
            (HIGH_SPREAD_LAYERS, "friction_detected"),
            (ALL_HIGH_LAYERS, "emergence_window"),
            (ALL_LOW_LAYERS, "baseline"),
            (UNIFORM_LAYERS, "emergence_window"),  # all 0.5 > 0.4 threshold
        ],
    )
    def test_native_signal(self, layers, expected_signal):
        result = NativeBackend().detect(
            layer_names=list(layers.keys()),
            layer_values=list(layers.values()),
        )
        assert result.signal == expected_signal

    @skip_no_mantic
    @pytest.mark.parametrize(
        "layers,expected_signal",
        [
            (HIGH_SPREAD_LAYERS, "friction_detected"),
            (ALL_HIGH_LAYERS, "emergence_window"),
            (ALL_LOW_LAYERS, "baseline"),
        ],
    )
    def test_mantic_signal(self, layers, expected_signal):
        result = ManticThinkingBackend().detect(
            layer_names=list(layers.keys()),
            layer_values=list(layers.values()),
        )
        assert result.signal == expected_signal


# ---------------------------------------------------------------------------
# ManticThinkingBackend (conditional)
# ---------------------------------------------------------------------------

class TestManticBackend:

    @skip_no_mantic
    def test_valid_result(self):
        result = ManticThinkingBackend().detect(
            layer_names=["micro", "meso", "macro", "meta"],
            layer_values=[0.8, 0.6, 0.7, 0.5],
        )
        assert isinstance(result, DetectionResult)
        assert 0 <= result.m_score <= 2.0
        assert result.backend_used == "mantic"

    @skip_no_mantic
    def test_raw_contains_audit(self):
        result = ManticThinkingBackend().detect(
            layer_names=["a", "b", "c", "d"],
            layer_values=[0.8, 0.6, 0.7, 0.5],
        )
        assert "overrides_applied" in result.raw
        assert "calibration" in result.raw

    @skip_no_mantic
    def test_get_backend_mantic(self):
        backend = get_backend("mantic")
        assert isinstance(backend, ManticThinkingBackend)

    def test_get_backend_mantic_missing(self):
        """When mantic is NOT installed, requesting it explicitly should raise."""
        if _HAS_MANTIC:
            pytest.skip("mantic is installed, can't test missing path")
        with pytest.raises(ImportError, match="mantic-thinking is not installed"):
            get_backend("mantic")
