"""Tests for scaffold file validation and loader defaults."""

from __future__ import annotations

from pathlib import Path

from cip_protocol.scaffold.loader import load_scaffold_file
from cip_protocol.scaffold.validator import validate_scaffold_file


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
