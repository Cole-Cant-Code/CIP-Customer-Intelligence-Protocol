"""Scaffold renderer -- assembles scaffolds into complete LLM prompts.

The renderer takes a Scaffold, a user query, and data context, then builds
a two-part prompt (system + user) that encodes the scaffold's full reasoning
framework into the LLM's context window.

System message structure:
  - Role, Perspective, Tone (identity)
  - Reasoning Steps (ordered analytical framework)
  - Domain Knowledge (activated knowledge areas)
  - Output Format + Required/Prohibited elements (calibration)
  - Disclaimers, Prohibited Actions, Escalation Triggers (guardrails)

User message structure:
  - User Request (the original query)
  - Data Context (JSON payload from MCP tools)
  - Cross-Domain Context (optional, from other CIP domains)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from cip_protocol.scaffold.models import AssembledPrompt, ChatMessage, Scaffold

logger = logging.getLogger(__name__)


def render_scaffold(
    scaffold: Scaffold,
    user_query: str,
    data_context: dict[str, Any],
    cross_domain_context: dict[str, Any] | None = None,
    chat_history: list[ChatMessage] | None = None,
    tone_variant: str | None = None,
    output_format: str | None = None,
    data_context_label: str = "Data Context",
) -> AssembledPrompt:
    """Combine scaffold + user query + data into a complete LLM prompt.

    Args:
        scaffold: The selected scaffold to render.
        user_query: The user's original query.
        data_context: Structured data from the domain's data provider.
        cross_domain_context: Optional context from other CIP domains.
        chat_history: Optional multi-turn history to preserve conversation state.
        tone_variant: Optional tone override.
        output_format: Optional output format override.
        data_context_label: Heading for the data section in the user
            message.  Defaults to "Data Context".  Domains override this
            via DomainConfig (e.g. "Financial Data", "Health Records").
    """
    system_message = _build_system_message(scaffold, tone_variant, output_format)
    user_message = _build_user_message(
        scaffold, user_query, data_context, cross_domain_context,
        data_context_label=data_context_label,
    )

    effective_tone = tone_variant if tone_variant in scaffold.framing.tone_variants else None
    allowed_formats = scaffold.output_calibration.format_options or [
        scaffold.output_calibration.format
    ]
    effective_format = (
        output_format if output_format in allowed_formats else scaffold.output_calibration.format
    )

    return AssembledPrompt(
        system_message=system_message,
        user_message=user_message,
        metadata={
            "scaffold_id": scaffold.id,
            "scaffold_version": scaffold.version,
            "tone": effective_tone or scaffold.framing.tone,
            "output_format": effective_format,
        },
        chat_history=chat_history or [],
    )


def _build_system_message(
    scaffold: Scaffold,
    tone_variant: str | None,
    output_format: str | None,
) -> str:
    """Assemble the system message from scaffold components."""
    parts: list[str] = []

    # Identity
    parts.append(f"## Your Role\n{scaffold.framing.role}")
    parts.append(f"## Your Perspective\n{scaffold.framing.perspective}")

    # Tone (with optional variant override)
    if tone_variant and tone_variant in scaffold.framing.tone_variants:
        tone_desc = scaffold.framing.tone_variants[tone_variant]
        parts.append(f"## Communication Tone\n{tone_desc}")
    else:
        parts.append(f"## Communication Tone\n{scaffold.framing.tone}")

    # Reasoning framework
    steps = scaffold.reasoning_framework.get("steps", [])
    if steps:
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        parts.append(f"## Reasoning Steps\nFollow these steps in order:\n{steps_text}")

    # Domain knowledge
    if scaffold.domain_knowledge_activation:
        knowledge = "\n".join(f"- {k}" for k in scaffold.domain_knowledge_activation)
        parts.append(f"## Domain Knowledge to Apply\n{knowledge}")

    # Output calibration
    allowed_formats = scaffold.output_calibration.format_options or [
        scaffold.output_calibration.format
    ]
    fmt = output_format if output_format in allowed_formats else scaffold.output_calibration.format
    parts.append(f"## Output Format\nFormat: {fmt}")
    if scaffold.output_calibration.max_length_guidance:
        parts.append(
            f"Length guidance: {scaffold.output_calibration.max_length_guidance}"
        )

    if scaffold.output_calibration.must_include:
        includes = "\n".join(
            f"- {item}" for item in scaffold.output_calibration.must_include
        )
        parts.append(f"## Required Elements\nYour response MUST include:\n{includes}")

    if scaffold.output_calibration.never_include:
        excludes = "\n".join(
            f"- {item}" for item in scaffold.output_calibration.never_include
        )
        parts.append(
            f"## Prohibited Elements\nYour response must NEVER include:\n{excludes}"
        )

    # Guardrails
    if scaffold.guardrails.disclaimers:
        disclaimers = "\n".join(f"- {d}" for d in scaffold.guardrails.disclaimers)
        parts.append(
            f"## Required Disclaimers\nInclude these where appropriate:\n{disclaimers}"
        )

    if scaffold.guardrails.prohibited_actions:
        prohibited = "\n".join(
            f"- {a}" for a in scaffold.guardrails.prohibited_actions
        )
        parts.append(f"## Prohibited Actions\nYou must NEVER:\n{prohibited}")

    if scaffold.guardrails.escalation_triggers:
        triggers = "\n".join(
            f"- {t}" for t in scaffold.guardrails.escalation_triggers
        )
        parts.append(
            f"## Escalation Triggers\n"
            f"If any of these conditions are detected, "
            f"recommend the user seek professional help:\n{triggers}"
        )

    return "\n\n".join(parts)


def _build_user_message(
    scaffold: Scaffold,
    user_query: str,
    data_context: dict[str, Any],
    cross_domain_context: dict[str, Any] | None,
    data_context_label: str = "Data Context",
) -> str:
    """Assemble the user message with query and data context."""
    parts: list[str] = []

    parts.append(f"## User Request\n{user_query}")

    if data_context:
        parts.append(
            f"## {data_context_label}\n"
            f"```json\n{json.dumps(data_context, indent=2, default=str)}\n```"
        )

    if cross_domain_context:
        xd_json = json.dumps(cross_domain_context, indent=2, default=str)
        parts.append(
            f"## Context From Other Domains\n```json\n{xd_json}\n```"
        )

    return "\n\n".join(parts)
