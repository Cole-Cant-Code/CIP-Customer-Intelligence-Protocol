"""Assembles scaffolds into two-part LLM prompts (system + user)."""

from __future__ import annotations

import json
from typing import Any

from cip_protocol.scaffold.models import AssembledPrompt, ChatMessage, Scaffold


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
    parts: list[str] = [
        f"## Your Role\n{scaffold.framing.role}",
        f"## Your Perspective\n{scaffold.framing.perspective}",
    ]

    if tone_variant and tone_variant in scaffold.framing.tone_variants:
        parts.append(f"## Communication Tone\n{scaffold.framing.tone_variants[tone_variant]}")
    else:
        parts.append(f"## Communication Tone\n{scaffold.framing.tone}")

    steps = scaffold.reasoning_framework.get("steps", [])
    if steps:
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        parts.append(f"## Reasoning Steps\nFollow these steps in order:\n{steps_text}")

    if scaffold.domain_knowledge_activation:
        knowledge = "\n".join(f"- {k}" for k in scaffold.domain_knowledge_activation)
        parts.append(f"## Domain Knowledge to Apply\n{knowledge}")

    allowed = scaffold.output_calibration.format_options or [scaffold.output_calibration.format]
    fmt = output_format if output_format in allowed else scaffold.output_calibration.format
    parts.append(f"## Output Format\nFormat: {fmt}")
    if scaffold.output_calibration.max_length_guidance:
        parts.append(f"Length guidance: {scaffold.output_calibration.max_length_guidance}")

    if scaffold.output_calibration.must_include:
        items = "\n".join(f"- {item}" for item in scaffold.output_calibration.must_include)
        parts.append(f"## Required Elements\nYour response MUST include:\n{items}")

    if scaffold.output_calibration.never_include:
        items = "\n".join(f"- {item}" for item in scaffold.output_calibration.never_include)
        parts.append(f"## Prohibited Elements\nYour response must NEVER include:\n{items}")

    if scaffold.guardrails.disclaimers:
        items = "\n".join(f"- {d}" for d in scaffold.guardrails.disclaimers)
        parts.append(f"## Required Disclaimers\nInclude these where appropriate:\n{items}")

    if scaffold.guardrails.prohibited_actions:
        items = "\n".join(f"- {a}" for a in scaffold.guardrails.prohibited_actions)
        parts.append(f"## Prohibited Actions\nYou must NEVER:\n{items}")

    if scaffold.guardrails.escalation_triggers:
        items = "\n".join(f"- {t}" for t in scaffold.guardrails.escalation_triggers)
        parts.append(
            f"## Escalation Triggers\n"
            f"If any of these conditions are detected, "
            f"recommend the user seek professional help:\n{items}"
        )

    return "\n\n".join(parts)


def _build_user_message(
    scaffold: Scaffold,
    user_query: str,
    data_context: dict[str, Any],
    cross_domain_context: dict[str, Any] | None,
    data_context_label: str = "Data Context",
) -> str:
    parts: list[str] = [f"## User Request\n{user_query}"]

    if data_context:
        parts.append(
            f"## {data_context_label}\n"
            f"```json\n{json.dumps(data_context, indent=2, default=str)}\n```"
        )

    if cross_domain_context:
        parts.append(
            f"## Context From Other Domains\n"
            f"```json\n{json.dumps(cross_domain_context, indent=2, default=str)}\n```"
        )

    return "\n\n".join(parts)
