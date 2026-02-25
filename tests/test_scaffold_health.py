"""Tests for cip scaffold-health: scoring, analysis, portfolio, report."""

from __future__ import annotations

import json

import pytest
from conftest import make_test_scaffold

from cip_protocol.health.analysis import (
    PortfolioHealthResult,
    ScaffoldHealthResult,
    analyze_portfolio,
    analyze_scaffold,
    compute_coherence,
    compute_m_score,
    detect_signal,
    dominant_layer,
    find_tension_pairs,
    interaction_score,
)
from cip_protocol.health.report import format_json, format_table
from cip_protocol.health.scoring import LAYER_NAMES, score_scaffold_layers
from cip_protocol.scaffold.models import (
    Scaffold,
    ScaffoldApplicability,
    ScaffoldFraming,
    ScaffoldGuardrails,
    ScaffoldOutputCalibration,
)


def _make_full_scaffold(
    scaffold_id: str = "full",
    *,
    tools: list[str] | None = None,
    keywords: list[str] | None = None,
    intent_signals: list[str] | None = None,
    steps: list[str] | None = None,
    dka: list[str] | None = None,
    fmt: str = "structured_narrative",
    format_options: list[str] | None = None,
    max_length_guidance: str = "",
    must_include: list[str] | None = None,
    never_include: list[str] | None = None,
    disclaimers: list[str] | None = None,
    escalation_triggers: list[str] | None = None,
    prohibited_actions: list[str] | None = None,
) -> Scaffold:
    """Build a Scaffold with full control over every layer-relevant field."""
    return Scaffold(
        id=scaffold_id,
        version="1.0",
        domain="test",
        display_name=f"Full: {scaffold_id}",
        description="Full control scaffold",
        applicability=ScaffoldApplicability(
            tools=tools or [],
            keywords=keywords or [],
            intent_signals=intent_signals or [],
        ),
        framing=ScaffoldFraming(role="Analyst", perspective="Analytical", tone="neutral"),
        reasoning_framework={"steps": steps or []},
        domain_knowledge_activation=dka or [],
        output_calibration=ScaffoldOutputCalibration(
            format=fmt,
            format_options=format_options or [fmt],
            max_length_guidance=max_length_guidance,
            must_include=must_include or [],
            never_include=never_include or [],
        ),
        guardrails=ScaffoldGuardrails(
            disclaimers=disclaimers or [],
            escalation_triggers=escalation_triggers or [],
            prohibited_actions=prohibited_actions or [],
        ),
    )


# ---------------------------------------------------------------------------
# TestScoring
# ---------------------------------------------------------------------------

class TestScoring:
    """Verify layer scores for various scaffold configurations."""

    def test_minimal_scaffold(self):
        """Default make_test_scaffold has known small counts."""
        s = make_test_scaffold()
        layers = score_scaffold_layers(s)

        # micro: 1 tool + 1 keyword + 0 intent_signals = 2 / 15
        assert layers["micro"] == pytest.approx(2 / 15, abs=1e-6)

        # meso: 2 steps + min(1 dka, 5) = 3 / 12
        assert layers["meso"] == pytest.approx(3 / 12, abs=1e-6)

        # macro: format is "structured_narrative" (not custom) = 0
        #        format_options has 2 items = 2
        #        no max_length_guidance = 0
        #        no must_include = 0, no never_include = 0
        #        total = 2 / 12
        assert layers["macro"] == pytest.approx(2 / 12, abs=1e-6)

        # meta: 1 disclaimer + 0 escalation + 0 prohibited = 1 / 10
        assert layers["meta"] == pytest.approx(1 / 10, abs=1e-6)

    def test_rich_scaffold(self):
        """Scaffold with many items in every section scores higher."""
        s = make_test_scaffold(
            tools=["a", "b", "c"],
            keywords=["k1", "k2", "k3", "k4"],
            intent_signals=["i1", "i2", "i3"],
            disclaimers=["d1", "d2", "d3"],
            escalation_triggers=["e1", "e2"],
            prohibited_actions=["p1", "p2", "p3"],
        )
        layers = score_scaffold_layers(s)

        # micro: 3 + 4 + 3 = 10 / 15
        assert layers["micro"] == pytest.approx(10 / 15, abs=1e-6)
        # meta: 3 + 2 + 3 = 8 / 10
        assert layers["meta"] == pytest.approx(8 / 10, abs=1e-6)
        # All scores in [0, 1]
        for name in LAYER_NAMES:
            assert 0.0 <= layers[name] <= 1.0

    def test_empty_guardrails(self):
        """Scaffold with no guardrails scores meta=0."""
        s = _make_full_scaffold(tools=["t"], keywords=["k"])
        layers = score_scaffold_layers(s)
        assert layers["meta"] == 0.0

    def test_over_cap_clamps_to_one(self):
        """Exceeding the cap clamps to 1.0, not above."""
        s = make_test_scaffold(
            tools=[f"t{i}" for i in range(10)],
            keywords=[f"k{i}" for i in range(10)],
            intent_signals=[f"s{i}" for i in range(10)],
        )
        layers = score_scaffold_layers(s)
        # micro raw = 30, cap = 15 → clamp to 1.0
        assert layers["micro"] == 1.0

    def test_all_layer_names_present(self):
        layers = score_scaffold_layers(make_test_scaffold())
        assert set(layers.keys()) == set(LAYER_NAMES)


# ---------------------------------------------------------------------------
# TestAnalysisMath
# ---------------------------------------------------------------------------

class TestAnalysisMath:
    """Unit tests for the pure math functions with hand-picked layer dicts."""

    def test_interaction_score_identical(self):
        assert interaction_score(0.8, 0.8) == 1.0

    def test_interaction_score_opposite(self):
        assert interaction_score(0.0, 1.0) == 0.0

    def test_interaction_score_partial(self):
        assert interaction_score(0.3, 0.7) == pytest.approx(0.6)

    def test_m_score_uniform(self):
        layers = {n: 0.5 for n in LAYER_NAMES}
        # M = sum(0.25 * 0.5) * 1.0 / sqrt(4) = 0.5 / 2.0 = 0.25
        assert compute_m_score(layers) == pytest.approx(0.25)

    def test_m_score_with_f_time(self):
        layers = {n: 0.5 for n in LAYER_NAMES}
        assert compute_m_score(layers, f_time=2.0) == pytest.approx(0.5)

    def test_coherence_perfect(self):
        layers = {n: 0.7 for n in LAYER_NAMES}
        # stdev = 0 → coherence = 1.0
        assert compute_coherence(layers) == 1.0

    def test_coherence_maximal_spread(self):
        layers = {"micro": 1.0, "meso": 0.0, "macro": 1.0, "meta": 0.0}
        # stdev = 0.5 → coherence = max(0, 1 - 0.5/0.5) = 0.0
        assert compute_coherence(layers) == 0.0

    def test_detect_signal_friction(self):
        layers = {"micro": 0.9, "meso": 0.1, "macro": 0.5, "meta": 0.5}
        assert detect_signal(layers) == "friction_detected"

    def test_detect_signal_emergence(self):
        layers = {"micro": 0.7, "meso": 0.8, "macro": 0.6, "meta": 0.5}
        assert detect_signal(layers) == "emergence_window"

    def test_detect_signal_baseline(self):
        layers = {"micro": 0.1, "meso": 0.2, "macro": 0.3, "meta": 0.1}
        # spread = 0.2, min = 0.1 → neither friction nor emergence
        assert detect_signal(layers) == "baseline"

    def test_find_tension_pairs_some(self):
        layers = {"micro": 0.9, "meso": 0.1, "macro": 0.5, "meta": 0.5}
        pairs = find_tension_pairs(layers, tension_threshold=0.5)
        pair_names = [(a, b) for a, b, _ in pairs]
        assert ("micro", "meso") in pair_names

    def test_find_tension_pairs_none(self):
        layers = {n: 0.5 for n in LAYER_NAMES}
        assert find_tension_pairs(layers, tension_threshold=0.5) == []

    def test_dominant_layer(self):
        layers = {"micro": 0.3, "meso": 0.9, "macro": 0.1, "meta": 0.5}
        assert dominant_layer(layers) == "meso"


# ---------------------------------------------------------------------------
# TestAnalyzeScaffold
# ---------------------------------------------------------------------------

class TestAnalyzeScaffold:
    """Integration: single scaffold end-to-end."""

    def test_returns_health_result(self):
        s = make_test_scaffold()
        r = analyze_scaffold(s)
        assert isinstance(r, ScaffoldHealthResult)
        assert r.scaffold_id == "test_scaffold"
        assert 0.0 <= r.m_score <= 1.0
        assert 0.0 <= r.coherence <= 1.0
        assert r.dominant_layer in LAYER_NAMES
        assert r.signal in ("friction_detected", "emergence_window", "baseline")

    def test_rich_scaffold_emergence(self):
        """A well-rounded scaffold should show emergence or at least no friction."""
        # All four layers need substantial content for emergence.
        s = _make_full_scaffold(
            scaffold_id="rich",
            tools=["a", "b", "c", "d"],
            keywords=["k1", "k2", "k3", "k4", "k5"],
            intent_signals=["i1", "i2"],
            steps=[f"s{i}" for i in range(6)],
            dka=["d1", "d2", "d3", "d4", "d5"],
            fmt="custom",
            format_options=["custom", "bullet", "table"],
            max_length_guidance="500 words",
            must_include=["summary", "action_items", "timeline"],
            never_include=["speculation"],
            disclaimers=["d1", "d2", "d3"],
            escalation_triggers=["e1", "e2", "e3"],
            prohibited_actions=["p1", "p2"],
        )
        r = analyze_scaffold(s)
        assert r.signal in ("emergence_window", "baseline")
        assert r.coherence > 0.5

    def test_imbalanced_scaffold_friction(self):
        """Scaffold heavy on micro but empty guardrails → friction."""
        s = _make_full_scaffold(
            scaffold_id="imbalanced",
            tools=[f"t{i}" for i in range(8)],
            keywords=[f"k{i}" for i in range(8)],
        )
        r = analyze_scaffold(s)
        assert r.signal == "friction_detected"
        assert r.dominant_layer == "micro"


# ---------------------------------------------------------------------------
# TestPortfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    """Multi-scaffold portfolio analysis + cross-scaffold coupling."""

    def test_portfolio_basics(self):
        scaffolds = [
            make_test_scaffold(scaffold_id="alpha"),
            make_test_scaffold(scaffold_id="beta"),
        ]
        result = analyze_portfolio(scaffolds)
        assert isinstance(result, PortfolioHealthResult)
        assert len(result.scaffolds) == 2
        assert result.avg_coherence > 0.0

    def test_portfolio_coupling_present(self):
        scaffolds = [
            make_test_scaffold(scaffold_id="alpha"),
            make_test_scaffold(scaffold_id="beta"),
        ]
        result = analyze_portfolio(scaffolds)
        # Two identical scaffolds → all coupling scores = 1.0
        assert len(result.coupling) > 0
        for _, _, _, score in result.coupling:
            assert score == pytest.approx(1.0)

    def test_portfolio_signal_emergence(self):
        """All scaffolds with emergence → portfolio_emergence."""
        rich_kwargs = dict(
            tools=["a", "b", "c", "d"],
            keywords=["k1", "k2", "k3", "k4", "k5"],
            intent_signals=["i1", "i2"],
            steps=[f"s{i}" for i in range(6)],
            dka=["d1", "d2", "d3", "d4", "d5"],
            fmt="custom",
            format_options=["custom", "bullet", "table"],
            max_length_guidance="500 words",
            must_include=["summary", "action_items", "timeline"],
            never_include=["speculation"],
            disclaimers=["d1", "d2", "d3"],
            escalation_triggers=["e1", "e2", "e3"],
            prohibited_actions=["p1", "p2"],
        )
        scaffolds = [
            _make_full_scaffold(scaffold_id="a", **rich_kwargs),
            _make_full_scaffold(scaffold_id="b", **rich_kwargs),
        ]
        result = analyze_portfolio(scaffolds)
        assert result.portfolio_signal == "portfolio_emergence"

    def test_portfolio_mixed(self):
        """One balanced (emergence) + one imbalanced (friction) → portfolio_mixed."""
        balanced = _make_full_scaffold(
            scaffold_id="balanced",
            tools=["a", "b", "c", "d"],
            keywords=["k1", "k2", "k3", "k4", "k5"],
            intent_signals=["i1", "i2"],
            steps=[f"s{i}" for i in range(6)],
            dka=["d1", "d2", "d3", "d4", "d5"],
            fmt="custom",
            format_options=["custom", "bullet", "table"],
            max_length_guidance="500 words",
            must_include=["summary", "action_items", "timeline"],
            never_include=["speculation"],
            disclaimers=["d1", "d2", "d3"],
            escalation_triggers=["e1", "e2", "e3"],
            prohibited_actions=["p1", "p2"],
        )
        imbalanced = _make_full_scaffold(
            scaffold_id="imbalanced",
            tools=[f"t{i}" for i in range(8)],
            keywords=[f"k{i}" for i in range(8)],
        )
        result = analyze_portfolio([balanced, imbalanced])
        assert result.portfolio_signal == "portfolio_mixed"

    def test_single_scaffold_no_coupling(self):
        result = analyze_portfolio([make_test_scaffold()])
        assert result.coupling == []

    def test_empty_portfolio(self):
        result = analyze_portfolio([])
        assert len(result.scaffolds) == 0
        assert result.portfolio_signal == "portfolio_empty"
        assert result.avg_coherence == 0.0


# ---------------------------------------------------------------------------
# TestReport
# ---------------------------------------------------------------------------

class TestReport:
    """Verify table contains scaffold IDs, JSON parses cleanly."""

    def _portfolio(self) -> PortfolioHealthResult:
        return analyze_portfolio([
            make_test_scaffold(scaffold_id="alpha"),
            make_test_scaffold(scaffold_id="beta"),
        ])

    def test_table_contains_ids(self):
        text = format_table(self._portfolio())
        assert "alpha" in text
        assert "beta" in text
        assert "Scaffold Health Report" in text
        assert "Layer Scores" in text
        assert "Portfolio:" in text

    def test_table_contains_signals(self):
        text = format_table(self._portfolio())
        # Each scaffold should have a signal listed
        for line in text.split("\n"):
            if "alpha" in line and "M-score" not in line and "Layer" not in line:
                assert any(
                    sig in line
                    for sig in ("friction_detected", "emergence_window", "baseline")
                )
                break

    def test_json_parses(self):
        raw = format_json(self._portfolio())
        data = json.loads(raw)
        assert "scaffolds" in data
        assert "coupling" in data
        assert "avg_coherence" in data
        assert "portfolio_signal" in data
        assert len(data["scaffolds"]) == 2

    def test_json_scaffold_fields(self):
        raw = format_json(self._portfolio())
        data = json.loads(raw)
        s = data["scaffolds"][0]
        assert "scaffold_id" in s
        assert "layers" in s
        assert "m_score" in s
        assert "coherence" in s
        assert "dominant_layer" in s
        assert "signal" in s
        assert "tension_pairs" in s
