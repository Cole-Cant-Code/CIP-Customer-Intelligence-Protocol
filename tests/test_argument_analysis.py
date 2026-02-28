"""Tests for argument structure analysis and fallacy detection."""

from __future__ import annotations

import pytest
from conftest import make_test_scaffold

from cip_protocol.llm.response import (
    ArgumentStructureEvaluator,
    GuardrailEvaluation,
    check_guardrails,
)
from cip_protocol.mantic_adapter import (
    _ARGUMENT_LAYERS,
    _ARGUMENT_WEIGHTS,
    DetectionResult,
    classify_fallacy,
    detect_argument_friction,
)
from cip_protocol.scaffold.loader import load_builtin_scaffolds
from cip_protocol.scaffold.registry import ScaffoldRegistry

# ---------------------------------------------------------------------------
# TestDetectArgumentFriction
# ---------------------------------------------------------------------------


class TestDetectArgumentFriction:
    """detect_argument_friction must wrap the adapter correctly."""

    def test_returns_detection_result(self):
        result = detect_argument_friction(layer_values=[0.8, 0.3, 0.7, 0.2])
        assert isinstance(result, DetectionResult)

    def test_uses_correct_layers(self):
        result = detect_argument_friction(layer_values=[0.5, 0.5, 0.5, 0.5])
        assert set(result.layer_attribution.keys()) == set(_ARGUMENT_LAYERS)

    def test_friction_for_high_spread(self):
        # High spread across layers → friction
        result = detect_argument_friction(layer_values=[0.9, 0.1, 0.8, 0.1])
        assert result.signal == "friction_detected"

    def test_emergence_for_all_high(self):
        # All layers above threshold → emergence
        result = detect_argument_friction(layer_values=[0.8, 0.7, 0.9, 0.75])
        assert result.signal == "emergence_window"

    def test_baseline_for_all_low(self):
        result = detect_argument_friction(layer_values=[0.1, 0.2, 0.15, 0.1])
        assert result.signal == "baseline"

    def test_clamps_out_of_range(self):
        # Values outside [0, 1] should be clamped
        result = detect_argument_friction(layer_values=[1.5, -0.3, 0.7, 0.5])
        assert isinstance(result, DetectionResult)
        # The clamped values should be [1.0, 0.0, 0.7, 0.5]
        # Since spread = 1.0 > 0.4, friction expected
        assert result.signal == "friction_detected"

    def test_requires_4_values(self):
        with pytest.raises(ValueError, match="exactly 4"):
            detect_argument_friction(layer_values=[0.5, 0.5, 0.5])

    def test_requires_4_values_too_many(self):
        with pytest.raises(ValueError, match="exactly 4"):
            detect_argument_friction(layer_values=[0.5, 0.5, 0.5, 0.5, 0.5])

    def test_custom_threshold(self):
        # With a very high threshold, even spread shouldn't trigger friction
        result = detect_argument_friction(
            layer_values=[0.8, 0.5, 0.7, 0.4],
            detection_threshold=0.9,
        )
        assert result.signal != "friction_detected"

    def test_weights_match_expected(self):
        assert _ARGUMENT_WEIGHTS == [0.25, 0.30, 0.25, 0.20]


# ---------------------------------------------------------------------------
# TestClassifyFallacy
# ---------------------------------------------------------------------------


class TestClassifyFallacy:
    """classify_fallacy must match known signatures and handle edge cases."""

    def _make_result(
        self,
        layer_values: list[float],
        **kw,
    ) -> tuple[DetectionResult, dict[str, float]]:
        result = detect_argument_friction(layer_values=layer_values, **kw)
        layer_dict = dict(zip(_ARGUMENT_LAYERS, layer_values))
        return result, layer_dict

    def test_valid_argument(self):
        result, layers = self._make_result([0.8, 0.7, 0.9, 0.75])
        fallacy = classify_fallacy(result, layers)
        assert fallacy.is_valid is True
        assert fallacy.name == "valid_argument"

    @pytest.mark.parametrize(
        "values,expected_name",
        [
            # straw_man: premise>0.4, inference>0.4, structure>0.4, scope<0.3
            ([0.7, 0.6, 0.7, 0.1], "straw_man"),
            # false_dilemma: premise>0.5, inference>0.4, structure<0.4, scope<0.3
            ([0.7, 0.6, 0.2, 0.1], "false_dilemma"),
            # affirming_the_consequent: premise>0.6, inference<0.3, structure<0.35
            ([0.8, 0.1, 0.2, 0.5], "affirming_the_consequent"),
            # circular_reasoning: inference>0.6, structure<0.3, premise<0.5
            ([0.3, 0.8, 0.1, 0.5], "circular_reasoning"),
            # appeal_to_authority: premise<0.3, inference>0.5, structure>0.5
            ([0.1, 0.8, 0.7, 0.5], "appeal_to_authority"),
            # hasty_generalization: 0.4<premise<0.7, scope<0.3
            # inference<=0.4 to avoid straw_man; spread must exceed threshold
            ([0.6, 0.35, 0.6, 0.05], "hasty_generalization"),
            # non_sequitur: premise>0.4, inference<0.2, structure>0.4
            ([0.7, 0.1, 0.7, 0.7], "non_sequitur"),
        ],
    )
    def test_fallacy_classification(self, values, expected_name):
        result, layers = self._make_result(values)
        fallacy = classify_fallacy(result, layers)
        assert fallacy.name == expected_name
        assert fallacy.is_valid is False

    def test_confidence_bounded(self):
        result, layers = self._make_result([0.9, 0.1, 0.8, 0.1])
        fallacy = classify_fallacy(result, layers)
        assert 0.0 <= fallacy.confidence <= 1.0

    def test_display_name_readable(self):
        result, layers = self._make_result([0.7, 0.6, 0.7, 0.1])
        fallacy = classify_fallacy(result, layers)
        assert " " in fallacy.display_name  # multi-word display name
        assert fallacy.display_name[0].isupper()  # capitalized

    def test_explanation_nonempty(self):
        result, layers = self._make_result([0.7, 0.6, 0.7, 0.1])
        fallacy = classify_fallacy(result, layers)
        assert len(fallacy.explanation) > 10

    def test_valid_confidence_bounded(self):
        result, layers = self._make_result([0.8, 0.7, 0.9, 0.75])
        fallacy = classify_fallacy(result, layers)
        assert 0.0 <= fallacy.confidence <= 1.0

    def test_unclassified_friction(self):
        # Craft values that cause friction but don't match any signature
        # All low-ish but with enough spread for friction
        result, layers = self._make_result([0.1, 0.1, 0.1, 0.9])
        if result.signal == "friction_detected":
            fallacy = classify_fallacy(result, layers)
            assert fallacy.name == "unclassified_friction"
            assert fallacy.is_valid is False


# ---------------------------------------------------------------------------
# TestArgumentStructureEvaluator
# ---------------------------------------------------------------------------


class TestArgumentStructureEvaluator:
    """ArgumentStructureEvaluator must integrate correctly with guardrail pipeline."""

    def _argument_scaffold(self):
        return make_test_scaffold(
            scaffold_id="arg_test",
            domain="argument_analysis",
            tags=["argument-analysis"],
        )

    def _plain_scaffold(self):
        return make_test_scaffold(
            scaffold_id="plain_test",
            domain="test_domain",
        )

    def test_skips_non_argument_scaffolds(self):
        evaluator = ArgumentStructureEvaluator()
        scaffold = self._plain_scaffold()
        result = evaluator.evaluate("some content here that is long enough to pass", scaffold)
        assert result.flags == []
        assert result.metadata == {}

    def test_activates_for_tagged_scaffolds(self):
        evaluator = ArgumentStructureEvaluator()
        scaffold = self._argument_scaffold()
        content = (
            "The argument has premise_strength: 0.8 and inferential_link: 0.2 "
            "with structural_validity: 0.7 and scope_consistency: 0.6"
        )
        result = evaluator.evaluate(content, scaffold)
        assert isinstance(result, GuardrailEvaluation)
        assert result.evaluator_name == "argument_structure"
        # Should have metadata with detection result
        assert "detection_result" in result.metadata
        assert "fallacy_result" in result.metadata

    def test_never_hard_violations(self):
        evaluator = ArgumentStructureEvaluator()
        scaffold = self._argument_scaffold()
        # Content with a clear fallacy signature (straw_man)
        content = (
            "Analysis shows premise_strength: 0.7 inferential_link: 0.6 "
            "structural_validity: 0.7 scope_consistency: 0.1"
        )
        result = evaluator.evaluate(content, scaffold)
        assert result.hard_violations == []

    def test_short_circuits_short_content(self):
        evaluator = ArgumentStructureEvaluator()
        scaffold = self._argument_scaffold()
        result = evaluator.evaluate("short", scaffold)
        assert result.flags == []
        assert result.metadata == {}

    def test_returns_empty_when_missing_layers(self):
        evaluator = ArgumentStructureEvaluator()
        scaffold = self._argument_scaffold()
        # Only 2 out of 4 layer scores present
        content = (
            "The argument has premise_strength: 0.8 and inferential_link: 0.2 "
            "but no other scores are mentioned in this text."
        )
        result = evaluator.evaluate(content, scaffold)
        assert result.flags == []

    def test_flags_friction_when_detected(self):
        evaluator = ArgumentStructureEvaluator()
        scaffold = self._argument_scaffold()
        # Straw man signature: high premise/inference/structure, low scope
        content = (
            "premise_strength: 0.7 inferential_link: 0.6 "
            "structural_validity: 0.7 scope_consistency: 0.1"
        )
        result = evaluator.evaluate(content, scaffold)
        assert any("argument_friction" in f for f in result.flags)
        assert "Straw Man" in result.flags[0]

    def test_integrates_with_check_guardrails(self):
        evaluator = ArgumentStructureEvaluator()
        scaffold = self._argument_scaffold()
        content = (
            "premise_strength: 0.7 inferential_link: 0.6 "
            "structural_validity: 0.7 scope_consistency: 0.1"
        )
        check = check_guardrails(content, scaffold, evaluators=[evaluator])
        # Should pass (soft evaluator, no hard violations)
        assert check.passed is True
        assert len(check.flags) > 0


# ---------------------------------------------------------------------------
# TestArgumentAnalysisScaffold
# ---------------------------------------------------------------------------


class TestArgumentAnalysisScaffold:
    """The argument_analysis.yaml builtin must load and validate correctly."""

    def test_yaml_loads_and_validates(self):
        registry = ScaffoldRegistry()
        count = load_builtin_scaffolds(registry)
        assert count >= 1
        scaffold = registry.get("argument_analysis")
        assert scaffold is not None

    def test_scaffold_has_correct_domain(self):
        registry = ScaffoldRegistry()
        load_builtin_scaffolds(registry)
        scaffold = registry.get("argument_analysis")
        assert scaffold.domain == "argument_analysis"

    def test_scaffold_has_tags(self):
        registry = ScaffoldRegistry()
        load_builtin_scaffolds(registry)
        scaffold = registry.get("argument_analysis")
        assert "argument-analysis" in scaffold.tags
        assert "fallacy-detection" in scaffold.tags

    def test_findable_by_tag(self):
        registry = ScaffoldRegistry()
        load_builtin_scaffolds(registry)
        results = registry.find_by_tag("argument-analysis")
        assert any(s.id == "argument_analysis" for s in results)

    def test_scaffold_has_disclaimers(self):
        registry = ScaffoldRegistry()
        load_builtin_scaffolds(registry)
        scaffold = registry.get("argument_analysis")
        assert len(scaffold.guardrails.disclaimers) > 0
        assert "structure" in scaffold.guardrails.disclaimers[0].lower()

    def test_scaffold_has_context_exports(self):
        registry = ScaffoldRegistry()
        load_builtin_scaffolds(registry)
        scaffold = registry.get("argument_analysis")
        export_names = {e.field_name for e in scaffold.context_exports}
        assert "layer_scores" in export_names
        assert "fallacy_result" in export_names
        assert "argument_verdict" in export_names
