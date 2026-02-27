"""Tests for control cockpit: presets, run policies, and constraint parser."""

from __future__ import annotations

import pytest

from cip_protocol.control import (
    BUILTIN_PRESETS,
    ConstraintParser,
    ControlPreset,
    PolicyConflictResult,
    PresetRegistry,
    RunPolicy,
    _policy_to_layer_values,
    detect_policy_conflict,
)

# ---------------------------------------------------------------------------
# ControlPreset
# ---------------------------------------------------------------------------


class TestControlPreset:
    def test_builtin_presets_exist(self):
        assert set(BUILTIN_PRESETS.keys()) == {"creative", "precise", "aggressive", "balanced"}

    def test_creative_preset_values(self):
        p = BUILTIN_PRESETS["creative"]
        assert p.temperature == 0.8
        assert p.max_length_guidance == "no length constraint"

    def test_precise_preset_values(self):
        p = BUILTIN_PRESETS["precise"]
        assert p.temperature == 0.1
        assert p.output_format == "bullet_points"
        assert p.compact is True

    def test_aggressive_preset_values(self):
        p = BUILTIN_PRESETS["aggressive"]
        assert p.skip_disclaimers is True
        assert p.remove_prohibited_actions == ["*"]

    def test_balanced_preset_values(self):
        p = BUILTIN_PRESETS["balanced"]
        assert p.temperature == 0.3
        assert p.skip_disclaimers is False

    def test_custom_preset_creation(self):
        p = ControlPreset(name="custom", temperature=0.6, skip_disclaimers=True)
        assert p.name == "custom"
        assert p.temperature == 0.6

    def test_temperature_validation_rejects_negative(self):
        with pytest.raises(ValueError, match="temperature"):
            ControlPreset(name="bad", temperature=-0.1)

    def test_temperature_validation_rejects_above_2(self):
        with pytest.raises(ValueError, match="temperature"):
            ControlPreset(name="bad", temperature=2.5)

    def test_max_tokens_validation_rejects_zero(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ControlPreset(name="bad", max_tokens=0)


# ---------------------------------------------------------------------------
# PresetRegistry
# ---------------------------------------------------------------------------


class TestPresetRegistry:
    def test_includes_builtins_by_default(self):
        reg = PresetRegistry()
        assert reg.get("creative") is not None
        assert reg.get("balanced") is not None

    def test_excludes_builtins_when_disabled(self):
        reg = PresetRegistry(include_builtins=False)
        assert reg.get("creative") is None
        assert reg.names() == []

    def test_register_and_get(self):
        reg = PresetRegistry(include_builtins=False)
        p = ControlPreset(name="custom", temperature=0.5)
        reg.register(p)
        assert reg.get("custom") is p

    def test_names_sorted(self):
        reg = PresetRegistry()
        assert reg.names() == sorted(reg.names())

    def test_register_overwrites_existing(self):
        reg = PresetRegistry()
        p = ControlPreset(name="creative", temperature=0.99)
        reg.register(p)
        assert reg.get("creative").temperature == 0.99


# ---------------------------------------------------------------------------
# RunPolicy
# ---------------------------------------------------------------------------


class TestRunPolicy:
    def test_default_policy_is_noop(self):
        p = RunPolicy()
        assert p.temperature is None
        assert p.max_tokens is None
        assert p.skip_disclaimers is False
        assert p.extra_must_include == []
        assert p.source == ""

    def test_from_preset_maps_all_fields(self):
        preset = BUILTIN_PRESETS["aggressive"]
        policy = RunPolicy.from_preset(preset)
        assert policy.temperature == 0.5
        assert policy.skip_disclaimers is True
        assert policy.remove_prohibited_actions == ["*"]
        assert policy.source == "preset:aggressive"

    def test_from_presets_last_writer_wins_scalars(self):
        policy = RunPolicy.from_presets(
            BUILTIN_PRESETS["creative"],
            BUILTIN_PRESETS["precise"],
        )
        assert policy.temperature == 0.1  # precise overwrites creative

    def test_from_presets_unions_lists(self):
        a = ControlPreset(name="a", extra_must_include=["item_a"])
        b = ControlPreset(name="b", extra_must_include=["item_b"])
        policy = RunPolicy.from_presets(a, b)
        assert "item_a" in policy.extra_must_include
        assert "item_b" in policy.extra_must_include

    def test_from_presets_empty(self):
        policy = RunPolicy.from_presets()
        assert policy.temperature is None

    def test_merge_scalar_other_wins(self):
        a = RunPolicy(temperature=0.3)
        b = RunPolicy(temperature=0.8)
        merged = a.merge(b)
        assert merged.temperature == 0.8

    def test_merge_none_does_not_override(self):
        a = RunPolicy(temperature=0.3)
        b = RunPolicy(temperature=None)
        merged = a.merge(b)
        assert merged.temperature == 0.3

    def test_merge_lists_concatenate(self):
        a = RunPolicy(extra_must_include=["x"])
        b = RunPolicy(extra_must_include=["y"])
        merged = a.merge(b)
        assert merged.extra_must_include == ["x", "y"]

    def test_merge_lists_deduplicate(self):
        a = RunPolicy(extra_must_include=["x", "y"])
        b = RunPolicy(extra_must_include=["y", "z"])
        merged = a.merge(b)
        assert merged.extra_must_include == ["x", "y", "z"]

    def test_merge_skip_disclaimers_sticky(self):
        a = RunPolicy(skip_disclaimers=True)
        b = RunPolicy(skip_disclaimers=False)
        merged = a.merge(b)
        assert merged.skip_disclaimers is True  # OR semantics

    def test_merge_sources_combined(self):
        a = RunPolicy(source="preset:creative")
        b = RunPolicy(source="constraint:brief")
        merged = a.merge(b)
        assert "preset:creative" in merged.source
        assert "constraint:brief" in merged.source

    def test_merge_scaffold_bias_other_wins_per_key(self):
        a = RunPolicy(scaffold_selection_bias={"s1": 2.0, "s2": 1.5})
        b = RunPolicy(scaffold_selection_bias={"s2": 3.0, "s3": 1.0})
        merged = a.merge(b)
        assert merged.scaffold_selection_bias == {"s1": 2.0, "s2": 3.0, "s3": 1.0}

    def test_validation_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            RunPolicy(temperature=3.0)

    def test_validation_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            RunPolicy(max_tokens=-1)


# ---------------------------------------------------------------------------
# ConstraintParser
# ---------------------------------------------------------------------------


class TestConstraintParser:
    def test_parse_empty_string(self):
        result = ConstraintParser.parse("")
        assert result.policy.temperature is None
        assert result.parsed == []
        assert result.unrecognized == []

    def test_parse_whitespace_only(self):
        result = ConstraintParser.parse("   ")
        assert result.parsed == []

    def test_parse_temperature_creative(self):
        result = ConstraintParser.parse("be more creative")
        assert result.policy.temperature == 0.8

    def test_parse_temperature_precise(self):
        result = ConstraintParser.parse("be more precise")
        assert result.policy.temperature == 0.1

    def test_parse_temperature_aggressive(self):
        result = ConstraintParser.parse("be more aggressive")
        assert result.policy.temperature == 0.5

    def test_parse_explicit_temperature(self):
        result = ConstraintParser.parse("temperature 0.7")
        assert result.policy.temperature == 0.7

    def test_parse_bullet_points(self):
        result = ConstraintParser.parse("use bullet points")
        assert result.policy.output_format == "bullet_points"

    def test_parse_structured_narrative(self):
        result = ConstraintParser.parse("use structured narrative")
        assert result.policy.output_format == "structured_narrative"

    def test_parse_under_n_words(self):
        result = ConstraintParser.parse("keep it under 200 words")
        assert result.policy.max_length_guidance == "under 200 words"

    def test_parse_keep_it_brief(self):
        result = ConstraintParser.parse("keep it brief")
        assert "concise" in result.policy.max_length_guidance

    def test_parse_be_concise(self):
        result = ConstraintParser.parse("be concise")
        assert "concise" in result.policy.max_length_guidance

    def test_parse_no_length_limit(self):
        result = ConstraintParser.parse("no length limit")
        assert "no length constraint" in result.policy.max_length_guidance

    def test_parse_skip_disclaimers(self):
        result = ConstraintParser.parse("skip disclaimers")
        assert result.policy.skip_disclaimers is True

    def test_parse_no_disclaimers(self):
        result = ConstraintParser.parse("no disclaimers")
        assert result.policy.skip_disclaimers is True

    def test_parse_skip_prohibited_actions(self):
        result = ConstraintParser.parse("skip prohibited actions")
        assert result.policy.remove_prohibited_actions == ["*"]

    def test_parse_must_include(self):
        result = ConstraintParser.parse("must include data sources")
        assert "data sources" in result.policy.extra_must_include

    def test_parse_never_include(self):
        result = ConstraintParser.parse("never include personal opinions")
        assert "personal opinions" in result.policy.extra_never_include

    def test_parse_compact_mode(self):
        result = ConstraintParser.parse("compact mode")
        assert result.policy.compact is True

    def test_parse_use_compact(self):
        result = ConstraintParser.parse("use compact")
        assert result.policy.compact is True

    def test_parse_tone_variant(self):
        result = ConstraintParser.parse("tone: friendly")
        assert result.policy.tone_variant == "friendly"

    def test_parse_max_tokens(self):
        result = ConstraintParser.parse("max 4000 tokens")
        assert result.policy.max_tokens == 4000

    def test_parse_multiple_clauses_comma(self):
        text = "be more creative, skip disclaimers, keep it under 200 words"
        result = ConstraintParser.parse(text)
        assert result.policy.temperature == 0.8
        assert result.policy.skip_disclaimers is True
        assert result.policy.max_length_guidance == "under 200 words"
        assert len(result.parsed) == 3

    def test_parse_multiple_clauses_semicolon(self):
        result = ConstraintParser.parse("be more precise; use bullet points")
        assert result.policy.temperature == 0.1
        assert result.policy.output_format == "bullet_points"

    def test_parse_unrecognized_clause(self):
        result = ConstraintParser.parse("do something weird")
        assert result.unrecognized == ["do something weird"]
        assert result.parsed == []

    def test_parse_mixed_recognized_unrecognized(self):
        result = ConstraintParser.parse("be more creative, do something weird, skip disclaimers")
        assert result.policy.temperature == 0.8
        assert result.policy.skip_disclaimers is True
        assert result.unrecognized == ["do something weird"]
        assert len(result.parsed) == 2

    def test_parse_case_insensitive(self):
        result = ConstraintParser.parse("Be More Creative")
        assert result.policy.temperature == 0.8

    def test_parse_preset_with_registry(self):
        reg = PresetRegistry()
        result = ConstraintParser.parse("preset: creative", preset_registry=reg)
        assert result.policy.temperature == 0.8
        assert len(result.parsed) == 1
        assert result.parsed[0].field == "preset"

    def test_parse_preset_without_registry(self):
        result = ConstraintParser.parse("preset: creative")
        assert result.unrecognized == ["preset: creative"]

    def test_parse_preset_unknown_name(self):
        reg = PresetRegistry()
        result = ConstraintParser.parse("preset: nonexistent", preset_registry=reg)
        assert "preset: nonexistent" in result.unrecognized

    def test_parse_source_tracks_rules(self):
        result = ConstraintParser.parse("be more creative, skip disclaimers")
        assert result.policy.source.startswith("constraint:")
        assert "creative_temp" in result.policy.source
        assert "skip_disclaimers" in result.policy.source

    def test_parse_multiple_must_include(self):
        result = ConstraintParser.parse("must include sources, must include citations")
        assert "sources" in result.policy.extra_must_include
        assert "citations" in result.policy.extra_must_include


# ---------------------------------------------------------------------------
# Policy Conflict Detection
# ---------------------------------------------------------------------------


class TestPolicyConflictDetection:
    def test_neutral_policy_no_conflict(self):
        policy = RunPolicy()
        result = detect_policy_conflict(policy)
        assert isinstance(result, PolicyConflictResult)
        assert result.m_score >= 0
        assert not result.has_conflict
        assert result.signal != "friction_detected"

    def test_contradictory_policy_detects_friction(self):
        policy = RunPolicy(
            temperature=1.8,  # very creative
            output_format="bullet_points",  # strict format
            max_length_guidance="concise, under 100 words",  # tight length
            compact=True,
            skip_disclaimers=True,
            remove_prohibited_actions=["*"],
        )
        result = detect_policy_conflict(policy)
        assert isinstance(result, PolicyConflictResult)
        # High creativity + high strictness + low safety = tension
        assert result.signal in ("friction_detected", "baseline", "emergence_window")

    def test_aligned_creative_policy_no_conflict(self):
        policy = RunPolicy(
            temperature=1.6,
            max_length_guidance="no length constraint",
        )
        result = detect_policy_conflict(policy)
        assert isinstance(result, PolicyConflictResult)

    def test_layer_values_bounded(self):
        # Extreme policy — all values should still be in [0, 1]
        policy = RunPolicy(
            temperature=2.0,
            output_format="json",
            max_length_guidance="under 10 words",
            compact=True,
            skip_disclaimers=True,
            remove_prohibited_actions=["*"],
            extra_must_include=["a", "b", "c", "d", "e"],
            extra_prohibited_actions=["x", "y", "z", "w"],
        )
        layer_values = _policy_to_layer_values(policy)
        for name, v in layer_values.items():
            assert 0.0 <= v <= 1.0, f"{name}={v} out of bounds"

    def test_summary_property(self):
        policy = RunPolicy()
        result = detect_policy_conflict(policy)
        summary = result.summary
        assert isinstance(summary, str)
        assert "m_score=" in summary

    def test_coherence_present(self):
        policy = RunPolicy(temperature=0.5)
        result = detect_policy_conflict(policy)
        assert 0.0 <= result.coherence <= 1.0

    def test_preset_aggressive_low_safety(self):
        policy = RunPolicy.from_preset(BUILTIN_PRESETS["aggressive"])
        layer_values = _policy_to_layer_values(policy)
        # aggressive: skip_disclaimers=True, remove_prohibited=["*"]
        # safety_priority should be low
        assert layer_values["safety_priority"] < 0.1
