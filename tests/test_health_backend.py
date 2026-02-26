"""Tests for backend-aware health analysis functions."""

from __future__ import annotations

import pytest

from cip_protocol.health.analysis import (
    analyze_portfolio,
    analyze_portfolio_with_backend,
    analyze_scaffold,
    analyze_scaffold_with_backend,
)
from cip_protocol.mantic_adapter import _probe_mantic
from tests.conftest import make_test_scaffold

_HAS_MANTIC = _probe_mantic()
skip_no_mantic = pytest.mark.skipif(not _HAS_MANTIC, reason="mantic-thinking not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _rich_scaffold(scaffold_id: str = "rich") -> object:
    """Scaffold with enough content to produce non-trivial layer scores."""
    return make_test_scaffold(
        scaffold_id=scaffold_id,
        tools=["tool_a", "tool_b", "tool_c"],
        keywords=["kw1", "kw2", "kw3", "kw4"],
        intent_signals=["sig1", "sig2"],
        disclaimers=["disc1", "disc2"],
        escalation_triggers=["esc1"],
        prohibited_actions=["prohib1"],
    )


def _minimal_scaffold(scaffold_id: str = "minimal") -> object:
    return make_test_scaffold(scaffold_id=scaffold_id)


# ---------------------------------------------------------------------------
# cip_native parity with existing functions
# ---------------------------------------------------------------------------

class TestNativeParity:
    """analyze_*_with_backend(backend='cip_native') must match analyze_*() exactly."""

    def test_scaffold_parity(self):
        s = _rich_scaffold()
        original = analyze_scaffold(s)
        via_backend = analyze_scaffold_with_backend(s, backend="cip_native")
        assert via_backend.scaffold_id == original.scaffold_id
        assert via_backend.layers == original.layers
        assert via_backend.m_score == pytest.approx(original.m_score, abs=1e-5)
        assert via_backend.coherence == pytest.approx(original.coherence, abs=1e-5)
        assert via_backend.dominant_layer == original.dominant_layer
        assert via_backend.signal == original.signal
        assert via_backend.tension_pairs == original.tension_pairs

    def test_scaffold_parity_minimal(self):
        s = _minimal_scaffold()
        original = analyze_scaffold(s)
        via_backend = analyze_scaffold_with_backend(s, backend="cip_native")
        assert via_backend.m_score == pytest.approx(original.m_score, abs=1e-5)
        assert via_backend.signal == original.signal

    def test_portfolio_parity(self):
        scaffolds = [_rich_scaffold("a"), _minimal_scaffold("b")]
        original = analyze_portfolio(scaffolds)
        via_backend = analyze_portfolio_with_backend(scaffolds, backend="cip_native")
        assert len(via_backend.scaffolds) == len(original.scaffolds)
        assert via_backend.avg_coherence == pytest.approx(original.avg_coherence, abs=1e-5)
        assert via_backend.portfolio_signal == original.portfolio_signal
        for orig_s, backend_s in zip(original.scaffolds, via_backend.scaffolds):
            assert backend_s.m_score == pytest.approx(orig_s.m_score, abs=1e-5)

    def test_portfolio_parity_single(self):
        scaffolds = [_rich_scaffold()]
        original = analyze_portfolio(scaffolds)
        via_backend = analyze_portfolio_with_backend(scaffolds, backend="cip_native")
        assert via_backend.portfolio_signal == original.portfolio_signal
        assert via_backend.coupling == original.coupling


# ---------------------------------------------------------------------------
# Auto backend
# ---------------------------------------------------------------------------

class TestAutoBackend:
    def test_scaffold_auto_returns_valid(self):
        s = _rich_scaffold()
        result = analyze_scaffold_with_backend(s, backend="auto")
        assert result.scaffold_id == "rich"
        assert 0 <= result.m_score <= 1.0
        assert result.signal in {"friction_detected", "emergence_window", "baseline"}

    def test_portfolio_auto_returns_valid(self):
        scaffolds = [_rich_scaffold("a"), _minimal_scaffold("b")]
        result = analyze_portfolio_with_backend(scaffolds, backend="auto")
        assert len(result.scaffolds) == 2
        assert result.portfolio_signal.startswith("portfolio_")


# ---------------------------------------------------------------------------
# Mantic backend (conditional)
# ---------------------------------------------------------------------------

class TestManticBackend:
    @skip_no_mantic
    def test_scaffold_mantic(self):
        s = _rich_scaffold()
        result = analyze_scaffold_with_backend(s, backend="mantic")
        assert result.scaffold_id == "rich"
        assert result.signal in {"friction_detected", "emergence_window", "baseline"}

    @skip_no_mantic
    def test_portfolio_mantic(self):
        scaffolds = [_rich_scaffold("a"), _minimal_scaffold("b")]
        result = analyze_portfolio_with_backend(scaffolds, backend="mantic")
        assert len(result.scaffolds) == 2
        assert result.portfolio_signal.startswith("portfolio_")
