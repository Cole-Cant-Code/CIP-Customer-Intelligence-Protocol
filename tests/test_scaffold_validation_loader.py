"""Tests for scaffold file validation and loader defaults."""

from __future__ import annotations

from pathlib import Path

from cip_protocol.scaffold.loader import load_scaffold_directory, load_scaffold_file
from cip_protocol.scaffold.registry import ScaffoldRegistry
from cip_protocol.scaffold.validator import validate_scaffold_directory, validate_scaffold_file


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_validator_allows_intent_only_applicability(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "intent_only.v1.yaml",
        """
id: intent_only
version: "1.0"
domain: test
display_name: Intent Only
description: intent-only scaffold
applicability:
  intent_signals: [create a budget]
framing:
  role: Analyst
  perspective: Grounded
  tone: neutral
reasoning_framework:
  steps: [step one]
output_calibration:
  format: structured_narrative
guardrails:
  disclaimers: [Not professional advice.]
""".strip(),
    )

    _, errors = validate_scaffold_file(path)
    assert not any("Applicability has no" in err for err in errors)


def test_validator_rejects_missing_all_applicability_signals(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "no_applicability.v1.yaml",
        """
id: no_applicability
version: "1.0"
domain: test
display_name: No Applicability
description: no applicability signals
framing:
  role: Analyst
  perspective: Grounded
  tone: neutral
reasoning_framework:
  steps: [step one]
output_calibration:
  format: structured_narrative
guardrails:
  disclaimers: [Not professional advice.]
""".strip(),
    )

    _, errors = validate_scaffold_file(path)
    assert any("Applicability has no tools, keywords, or intent signals" in err for err in errors)


def test_loader_defaults_format_options_to_format(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "format_default.v1.yaml",
        """
id: format_default
version: "1.0"
domain: test
display_name: Format Default
description: format options fallback
applicability:
  tools: [analyze]
framing:
  role: Analyst
  perspective: Grounded
  tone: neutral
reasoning_framework:
  steps: [step one]
output_calibration:
  format: bullet_points
guardrails:
  disclaimers: [Not professional advice.]
""".strip(),
    )

    scaffold = load_scaffold_file(path)
    assert scaffold.output_calibration.format == "bullet_points"
    assert scaffold.output_calibration.format_options == ["bullet_points"]


VALID_SCAFFOLD_YAML = """\
id: {id}
version: "1.0"
domain: test
display_name: Test {id}
description: test scaffold
applicability:
  tools: [{tool}]
framing:
  role: Analyst
  perspective: Grounded
  tone: neutral
reasoning_framework:
  steps: [step one]
output_calibration:
  format: structured_narrative
guardrails:
  disclaimers: [Not professional advice.]
"""


# --- load_scaffold_directory tests ---


def test_load_directory_loads_all_yamls(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "alpha.yaml",
        VALID_SCAFFOLD_YAML.format(id="alpha", tool="tool_a"),
    )
    _write_yaml(
        tmp_path / "beta.yaml",
        VALID_SCAFFOLD_YAML.format(id="beta", tool="tool_b"),
    )
    registry = ScaffoldRegistry()
    count = load_scaffold_directory(tmp_path, registry)
    assert count == 2
    assert registry.get("alpha") is not None
    assert registry.get("beta") is not None


def test_load_directory_skips_underscore_prefixed(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "valid.yaml",
        VALID_SCAFFOLD_YAML.format(id="valid", tool="tool_v"),
    )
    _write_yaml(
        tmp_path / "_draft.yaml",
        VALID_SCAFFOLD_YAML.format(id="draft", tool="tool_d"),
    )
    registry = ScaffoldRegistry()
    count = load_scaffold_directory(tmp_path, registry)
    assert count == 1
    assert registry.get("valid") is not None
    assert registry.get("draft") is None


def test_load_directory_recurses_subdirectories(tmp_path: Path) -> None:
    subdir = tmp_path / "nested"
    subdir.mkdir()
    _write_yaml(
        tmp_path / "top.yaml",
        VALID_SCAFFOLD_YAML.format(id="top", tool="tool_top"),
    )
    _write_yaml(
        subdir / "deep.yaml",
        VALID_SCAFFOLD_YAML.format(id="deep", tool="tool_deep"),
    )
    registry = ScaffoldRegistry()
    count = load_scaffold_directory(tmp_path, registry)
    assert count == 2
    assert registry.get("deep") is not None


def test_load_directory_skips_bad_yaml(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "good.yaml",
        VALID_SCAFFOLD_YAML.format(id="good", tool="tool_g"),
    )
    _write_yaml(tmp_path / "bad.yaml", "not: valid: yaml: [[[")
    registry = ScaffoldRegistry()
    count = load_scaffold_directory(tmp_path, registry)
    assert count == 1
    assert registry.get("good") is not None


def test_load_directory_skips_empty_yaml_file(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "good.yaml",
        VALID_SCAFFOLD_YAML.format(id="good", tool="tool_g"),
    )
    _write_yaml(tmp_path / "empty.yaml", "")

    registry = ScaffoldRegistry()
    count = load_scaffold_directory(tmp_path, registry)

    assert count == 1
    assert registry.get("good") is not None


def test_load_directory_skips_non_mapping_yaml_root(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "good.yaml",
        VALID_SCAFFOLD_YAML.format(id="good", tool="tool_g"),
    )
    _write_yaml(tmp_path / "list_root.yaml", "- not\n- a\n- mapping\n")

    registry = ScaffoldRegistry()
    count = load_scaffold_directory(tmp_path, registry)

    assert count == 1
    assert registry.get("good") is not None


def test_load_directory_nonexistent_returns_zero(tmp_path: Path) -> None:
    registry = ScaffoldRegistry()
    count = load_scaffold_directory(tmp_path / "does_not_exist", registry)
    assert count == 0


# --- validate_scaffold_directory tests ---


def test_validate_directory_reports_all_valid(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "alpha.yaml",
        VALID_SCAFFOLD_YAML.format(id="alpha", tool="tool_a"),
    )
    _write_yaml(
        tmp_path / "beta.yaml",
        VALID_SCAFFOLD_YAML.format(id="beta", tool="tool_b"),
    )
    count, errors = validate_scaffold_directory(tmp_path)
    assert count == 2
    assert errors == []


def test_validate_directory_detects_duplicate_ids(tmp_path: Path) -> None:
    subdir = tmp_path / "nested"
    subdir.mkdir()
    _write_yaml(
        tmp_path / "dupe.yaml",
        VALID_SCAFFOLD_YAML.format(id="dupe", tool="tool_a"),
    )
    _write_yaml(
        subdir / "dupe.yaml",
        VALID_SCAFFOLD_YAML.format(id="dupe", tool="tool_b"),
    )
    _, errors = validate_scaffold_directory(tmp_path)
    assert any("Duplicate ID" in err for err in errors)


def test_validate_directory_nonexistent(tmp_path: Path) -> None:
    count, errors = validate_scaffold_directory(tmp_path / "nope")
    assert count == 0
    assert any("not found" in err for err in errors)


def test_validate_directory_empty(tmp_path: Path) -> None:
    count, errors = validate_scaffold_directory(tmp_path)
    assert count == 0
    assert any("No scaffold YAML" in err for err in errors)
