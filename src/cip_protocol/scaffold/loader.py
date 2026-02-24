"""YAML scaffold loading. Files starting with underscore are skipped."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from cip_protocol.scaffold.matcher import prepare_matcher_cache
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
    """Load all YAML scaffolds from a directory recursively. Returns count loaded."""
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
        except (yaml.YAMLError, KeyError, ValueError, TypeError) as exc:
            logger.exception("Failed to load scaffold from %s: %s", path, exc)
    if count > 0:
        prepare_matcher_cache(registry)
    return count


def load_scaffold_file(path: Path) -> Scaffold:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    app = data.get("applicability", {})
    framing = data.get("framing", {})
    output = data.get("output_calibration", {})
    guardrails = data.get("guardrails", {})
    fmt = output.get("format", "structured_narrative")
    fmt_options = output.get("format_options") or [fmt]

    def _context_fields(key: str) -> list[ContextField]:
        return [
            ContextField(
                field_name=c.get("field_name", c.get("field", "")),
                type=c.get("type", ""),
                description=c.get("description", ""),
            )
            for c in data.get(key, [])
        ]

    return Scaffold(
        id=data["id"],
        version=data["version"],
        domain=data["domain"],
        display_name=data["display_name"],
        description=data["description"].strip(),
        applicability=ScaffoldApplicability(
            tools=app.get("tools", []),
            keywords=app.get("keywords", []),
            intent_signals=app.get("intent_signals", []),
        ),
        framing=ScaffoldFraming(
            role=framing.get("role", "").strip(),
            perspective=framing.get("perspective", "").strip(),
            tone=framing.get("tone", ""),
            tone_variants=framing.get("tone_variants", {}),
        ),
        reasoning_framework=data.get("reasoning_framework", {}),
        domain_knowledge_activation=data.get("domain_knowledge_activation", []),
        output_calibration=ScaffoldOutputCalibration(
            format=fmt,
            format_options=fmt_options,
            max_length_guidance=output.get("max_length_guidance", ""),
            must_include=output.get("must_include", []),
            never_include=output.get("never_include", []),
        ),
        guardrails=ScaffoldGuardrails(
            disclaimers=guardrails.get("disclaimers", []),
            escalation_triggers=guardrails.get("escalation_triggers", []),
            prohibited_actions=guardrails.get("prohibited_actions", []),
        ),
        context_accepts=_context_fields("context_accepts"),
        context_exports=_context_fields("context_exports"),
        tags=data.get("tags", []),
    )
