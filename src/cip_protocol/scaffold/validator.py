"""Scaffold YAML validation — required fields, applicability, guardrails, ID uniqueness."""

from __future__ import annotations

import logging
from pathlib import Path

from cip_protocol.scaffold.loader import load_scaffold_file
from cip_protocol.scaffold.models import Scaffold

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["id", "version", "domain", "display_name", "description"]


def _display(path: Path, project_root: Path | None) -> str:
    if not project_root:
        return str(path)
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def validate_scaffold_file(
    path: Path, *, project_root: Path | None = None
) -> tuple[Scaffold | None, list[str]]:
    errors: list[str] = []
    dp = _display(path, project_root)

    try:
        scaffold = load_scaffold_file(path)
    except Exception as exc:
        return None, [f"{dp}: Failed to load -- {exc}"]

    for field_name in REQUIRED_FIELDS:
        if not getattr(scaffold, field_name, None):
            errors.append(f"{dp}: Missing or empty required field '{field_name}'")

    if (
        not scaffold.applicability.tools
        and not scaffold.applicability.keywords
        and not scaffold.applicability.intent_signals
    ):
        errors.append(f"{dp}: Applicability has no tools, keywords, or intent signals")

    if not scaffold.guardrails.disclaimers:
        errors.append(f"{dp}: No guardrail disclaimers defined")

    if not scaffold.reasoning_framework.get("steps"):
        errors.append(f"{dp}: Reasoning framework has no steps")

    if scaffold.version and not all(c.isdigit() or c == "." for c in scaffold.version):
        errors.append(f"{dp}: Version '{scaffold.version}' doesn't look like a version number")

    name = path.name
    if not (name == f"{scaffold.id}.yaml" or name.startswith(f"{scaffold.id}.")):
        errors.append(
            f"{dp}: Filename '{name}' should match scaffold id "
            f"'{scaffold.id}' (expected '{scaffold.id}.*.yaml')"
        )

    return scaffold, errors


def validate_scaffold_directory(
    directory: str | Path, *, project_root: Path | None = None
) -> tuple[int, list[str]]:
    directory = Path(directory)
    if not directory.is_dir():
        return 0, [f"Scaffold directory not found: {directory}"]

    yaml_files = sorted(
        p for p in directory.rglob("*.yaml") if not p.name.startswith("_")
    )
    if not yaml_files:
        return 0, [f"No scaffold YAML files found in {directory}"]

    errors: list[str] = []
    seen_ids: dict[str, Path] = {}
    loaded = 0

    for path in yaml_files:
        scaffold, file_errors = validate_scaffold_file(path, project_root=project_root)
        if file_errors:
            errors.extend(file_errors)
            continue

        assert scaffold is not None
        loaded += 1

        if scaffold.id in seen_ids:
            errors.append(
                f"{_display(path, project_root)}: Duplicate ID '{scaffold.id}' -- "
                f"already defined in {_display(seen_ids[scaffold.id], project_root)}"
            )
        else:
            seen_ids[scaffold.id] = path

    return loaded, errors


