"""Scaffold loader -- reads YAML definitions from disk into the registry.

Supports both single-file loading (load_scaffold_file) and recursive
directory loading (load_scaffold_directory). Files starting with underscore
(e.g. _schema.yaml, _template.yaml) are skipped automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from cip_protocol.scaffold.models import (
    ContextField,
    Scaffold,
    ScaffoldApplicability,
    ScaffoldFraming,
    ScaffoldGuardrails,
    ScaffoldOutputCalibration,
)
from cip_protocol.scaffold.registry import ScaffoldRegistry

logger = logging.getLogger(__name__)


def load_scaffold_directory(directory: str | Path, registry: ScaffoldRegistry) -> int:
    """Load all YAML scaffold definitions from a directory (recursively).

    Returns the number of scaffolds successfully loaded.
    Skips files whose names start with underscore.
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("Scaffold directory does not exist: %s", directory)
        return 0

    count = 0
    for path in sorted(directory.rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        try:
            scaffold = load_scaffold_file(path)
            registry.register(scaffold)
            count += 1
            logger.info("Loaded scaffold: %s (v%s)", scaffold.id, scaffold.version)
        except Exception:
            logger.exception("Failed to load scaffold from %s", path)
    return count


def load_scaffold_file(path: Path) -> Scaffold:
    """Parse a single YAML file into a Scaffold instance.

    Handles missing optional sections gracefully by falling back to
    empty defaults. The `context_accepts` and `context_exports` lists
    support both 'field_name' and legacy 'field' key names.
    """
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    applicability_data = data.get("applicability", {})
    framing_data = data.get("framing", {})
    output_data = data.get("output_calibration", {})
    guardrails_data = data.get("guardrails", {})
    output_format = output_data.get("format", "structured_narrative")
    output_options = output_data.get("format_options")
    if not output_options:
        output_options = [output_format]

    return Scaffold(
        id=data["id"],
        version=data["version"],
        domain=data["domain"],
        display_name=data["display_name"],
        description=data["description"].strip(),
        applicability=ScaffoldApplicability(
            tools=applicability_data.get("tools", []),
            keywords=applicability_data.get("keywords", []),
            intent_signals=applicability_data.get("intent_signals", []),
        ),
        framing=ScaffoldFraming(
            role=framing_data.get("role", "").strip(),
            perspective=framing_data.get("perspective", "").strip(),
            tone=framing_data.get("tone", ""),
            tone_variants=framing_data.get("tone_variants", {}),
        ),
        reasoning_framework=data.get("reasoning_framework", {}),
        domain_knowledge_activation=data.get("domain_knowledge_activation", []),
        output_calibration=ScaffoldOutputCalibration(
            format=output_format,
            format_options=output_options,
            max_length_guidance=output_data.get("max_length_guidance", ""),
            must_include=output_data.get("must_include", []),
            never_include=output_data.get("never_include", []),
        ),
        guardrails=ScaffoldGuardrails(
            disclaimers=guardrails_data.get("disclaimers", []),
            escalation_triggers=guardrails_data.get("escalation_triggers", []),
            prohibited_actions=guardrails_data.get("prohibited_actions", []),
        ),
        context_accepts=[
            ContextField(
                field_name=c.get("field_name", c.get("field", "")),
                type=c.get("type", ""),
                description=c.get("description", ""),
            )
            for c in data.get("context_accepts", [])
        ],
        context_exports=[
            ContextField(
                field_name=c.get("field_name", c.get("field", "")),
                type=c.get("type", ""),
                description=c.get("description", ""),
            )
            for c in data.get("context_exports", [])
        ],
        tags=data.get("tags", []),
    )
