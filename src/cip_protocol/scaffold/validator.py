"""Scaffold YAML validator -- ensures scaffold definitions are well-formed.

Validation checks:
  - Required fields present and non-empty
  - Applicability defines at least one tool, keyword, or intent signal
  - At least one guardrail disclaimer is defined
  - Reasoning framework contains steps
  - Version string looks like a version number
  - Filename matches the scaffold ID
  - No duplicate IDs across a directory
"""

from __future__ import annotations

import logging
from pathlib import Path

from cip_protocol.scaffold.loader import load_scaffold_file
from cip_protocol.scaffold.models import Scaffold

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["id", "version", "domain", "display_name", "description"]


def validate_scaffold_file(
    path: Path, *, project_root: Path | None = None
) -> tuple[Scaffold | None, list[str]]:
    """Validate a single scaffold YAML file.

    Returns a tuple of (scaffold_or_none, list_of_errors).
    """
    errors: list[str] = []

    display_path: str
    if project_root:
        try:
            display_path = str(path.relative_to(project_root))
        except ValueError:
            display_path = str(path)
    else:
        display_path = str(path)

    try:
        scaffold = load_scaffold_file(path)
    except Exception as exc:
        return None, [f"{display_path}: Failed to load -- {exc}"]

    for field_name in REQUIRED_FIELDS:
        value = getattr(scaffold, field_name, None)
        if not value:
            errors.append(
                f"{display_path}: Missing or empty required field '{field_name}'"
            )

    if (
        not scaffold.applicability.tools
        and not scaffold.applicability.keywords
        and not scaffold.applicability.intent_signals
    ):
        errors.append(
            f"{display_path}: Applicability has no tools, keywords, or intent signals"
        )

    if not scaffold.guardrails.disclaimers:
        errors.append(f"{display_path}: No guardrail disclaimers defined")

    steps = scaffold.reasoning_framework.get("steps", [])
    if not steps:
        errors.append(f"{display_path}: Reasoning framework has no steps")

    if scaffold.version and not all(
        c.isdigit() or c == "." for c in scaffold.version
    ):
        errors.append(
            f"{display_path}: Version '{scaffold.version}' doesn't look like "
            f"a version number"
        )

    name = path.name
    if not (name == f"{scaffold.id}.yaml" or name.startswith(f"{scaffold.id}.")):
        errors.append(
            f"{display_path}: Filename '{name}' should match scaffold id "
            f"'{scaffold.id}' (expected '{scaffold.id}.*.yaml')"
        )

    return scaffold, errors


def validate_scaffold_directory(
    directory: str | Path, *, project_root: Path | None = None
) -> tuple[int, list[str]]:
    """Validate all scaffold YAML files in a directory (recursively).

    Returns a tuple of (scaffold_count, list_of_errors).
    """
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
        scaffold, file_errors = validate_scaffold_file(
            path, project_root=project_root
        )
        if file_errors:
            errors.extend(file_errors)
            continue

        assert scaffold is not None
        loaded += 1

        if scaffold.id in seen_ids:
            if project_root:
                try:
                    here = str(path.relative_to(project_root))
                except ValueError:
                    here = str(path)
                try:
                    there = str(seen_ids[scaffold.id].relative_to(project_root))
                except ValueError:
                    there = str(seen_ids[scaffold.id])
            else:
                here = str(path)
                there = str(seen_ids[scaffold.id])
            errors.append(
                f"{here}: Duplicate ID '{scaffold.id}' -- "
                f"already defined in {there}"
            )
        else:
            seen_ids[scaffold.id] = path

    return loaded, errors


def validate_scaffolds(directory: str | Path) -> tuple[int, int]:
    """Backwards-compatible API: returns (scaffold_count, error_count)."""
    count, errors = validate_scaffold_directory(directory)
    for err in errors:
        logger.error("%s", err)
    return count, len(errors)
