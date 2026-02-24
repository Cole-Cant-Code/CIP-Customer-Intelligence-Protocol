"""Assembles scaffolds into two-part LLM prompts (system + user)."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from cip_protocol.scaffold.models import AssembledPrompt, ChatMessage, Scaffold

if TYPE_CHECKING:
    from cip_protocol.control import RunPolicy


def render_scaffold(
    scaffold: Scaffold,
    user_query: str,
    data_context: dict[str, Any],
    cross_domain_context: dict[str, Any] | None = None,
    chat_history: list[ChatMessage] | None = None,
    tone_variant: str | None = None,
    output_format: str | None = None,
    data_context_label: str = "Data Context",
    compact: bool = False,
    policy: RunPolicy | None = None,
) -> AssembledPrompt:
    system_message = _build_system_message(scaffold, tone_variant, output_format, policy=policy)
    if compact:
        system_message = _compact_prompt(system_message)

    user_message = _build_user_message(
        scaffold, user_query, data_context, cross_domain_context,
        data_context_label=data_context_label,
        compact=compact,
    )

    effective_tone = tone_variant if tone_variant in scaffold.framing.tone_variants else None
    allowed_formats = scaffold.output_calibration.format_options or [
        scaffold.output_calibration.format
    ]
    effective_format = (
        output_format if output_format in allowed_formats else scaffold.output_calibration.format
    )

    metadata: dict[str, Any] = {
        "scaffold_id": scaffold.id,
        "scaffold_version": scaffold.version,
        "tone": effective_tone or scaffold.framing.tone,
        "output_format": effective_format,
    }
    if policy and policy.source:
        metadata["policy_source"] = policy.source

    return AssembledPrompt(
        system_message=system_message,
        user_message=user_message,
        metadata=metadata,
        chat_history=chat_history or [],
    )


def _build_system_message(
    scaffold: Scaffold,
    tone_variant: str | None,
    output_format: str | None,
    policy: RunPolicy | None = None,
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

    length_guidance = scaffold.output_calibration.max_length_guidance
    if policy and policy.max_length_guidance:
        length_guidance = policy.max_length_guidance
    if length_guidance:
        parts.append(f"Length guidance: {length_guidance}")

    must_include = list(scaffold.output_calibration.must_include)
    if policy:
        extras = [i for i in policy.extra_must_include if i not in must_include]
        must_include = must_include + extras
    if must_include:
        items = "\n".join(f"- {item}" for item in must_include)
        parts.append(f"## Required Elements\nYour response MUST include:\n{items}")

    never_include = list(scaffold.output_calibration.never_include)
    if policy:
        extras = [i for i in policy.extra_never_include if i not in never_include]
        never_include = never_include + extras
    if never_include:
        items = "\n".join(f"- {item}" for item in never_include)
        parts.append(f"## Prohibited Elements\nYour response must NEVER include:\n{items}")

    skip_disclaimers = policy.skip_disclaimers if policy else False
    if not skip_disclaimers and scaffold.guardrails.disclaimers:
        items = "\n".join(f"- {d}" for d in scaffold.guardrails.disclaimers)
        parts.append(f"## Required Disclaimers\nInclude these where appropriate:\n{items}")

    prohibited = list(scaffold.guardrails.prohibited_actions)
    if policy:
        if "*" in policy.remove_prohibited_actions:
            prohibited = []
        else:
            prohibited = [a for a in prohibited if a not in policy.remove_prohibited_actions]
        extras = [a for a in policy.extra_prohibited_actions if a not in prohibited]
        prohibited = prohibited + extras
    if prohibited:
        items = "\n".join(f"- {a}" for a in prohibited)
        parts.append(f"## Prohibited Actions\nYou must NEVER:\n{items}")

    if scaffold.guardrails.escalation_triggers:
        items = "\n".join(f"- {t}" for t in scaffold.guardrails.escalation_triggers)
        parts.append(
            f"## Escalation Triggers\n"
            f"If any of these conditions are detected, "
            f"recommend the user seek professional help:\n{items}"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Compact prompt compression
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^##\s+[^\n]+\n", re.MULTILINE)


def _compact_prompt(text: str) -> str:
    """Strip markdown headers, collapse bullet lists to semicolons, single newlines."""
    text = _HEADER_RE.sub("", text)

    lines = text.split("\n")
    compacted: list[str] = []
    bullet_buffer: list[str] = []

    def flush_bullets() -> None:
        if bullet_buffer:
            compacted.append("; ".join(bullet_buffer))
            bullet_buffer.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            bullet_buffer.append(stripped[2:].strip())
        else:
            flush_bullets()
            if stripped:
                compacted.append(stripped)

    flush_bullets()
    return "\n".join(compacted)


# ---------------------------------------------------------------------------
# User message
# ---------------------------------------------------------------------------

def _build_user_message(
    scaffold: Scaffold,
    user_query: str,
    data_context: dict[str, Any],
    cross_domain_context: dict[str, Any] | None,
    data_context_label: str = "Data Context",
    compact: bool = False,
) -> str:
    joiner = "\n" if compact else "\n\n"

    if compact:
        parts: list[str] = [user_query]
    else:
        parts = [f"## User Request\n{user_query}"]

    if data_context:
        if compact:
            parts.append(f"{data_context_label}: {json.dumps(data_context, default=str)}")
        else:
            parts.append(
                f"## {data_context_label}\n"
                f"```json\n{json.dumps(data_context, indent=2, default=str)}\n```"
            )

    if cross_domain_context:
        if compact:
            parts.append(f"Cross-domain: {json.dumps(cross_domain_context, default=str)}")
        else:
            parts.append(
                f"## Context From Other Domains\n"
                f"```json\n{json.dumps(cross_domain_context, indent=2, default=str)}\n```"
            )

    return joiner.join(parts)
