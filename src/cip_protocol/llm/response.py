"""Response parsing and guardrail enforcement for inner LLM output.

This module sits between the raw provider response and the final LLMResponse
returned by the client.  It implements three safety layers:

1. **check_guardrails** -- detects prohibited patterns and escalation triggers
2. **sanitize_content** -- redacts sentences containing prohibited phrases
3. **enforce_disclaimers** -- appends any scaffold-required disclaimers the LLM omitted

Plus a utility for extracting structured context-export fields from free text.

All prohibited patterns come from the DomainConfig — the protocol itself
has no hardcoded domain knowledge.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from cip_protocol.scaffold.models import Scaffold

logger = logging.getLogger(__name__)


@dataclass
class GuardrailCheck:
    """Result of checking a response against scaffold guardrails."""

    passed: bool
    flags: list[str] = field(default_factory=list)


def _contains_indicator(content_lower: str, pattern: str) -> bool:
    """Match indicator phrases with word boundaries and flexible whitespace."""
    normalized = " ".join(pattern.lower().split())
    if not normalized:
        return False
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    regex = re.compile(rf"(?<!\w){escaped}(?!\w)")
    return bool(regex.search(content_lower))


# ---------------------------------------------------------------------------
# Guardrail detection
# ---------------------------------------------------------------------------

def check_guardrails(
    content: str,
    scaffold: Scaffold,
    prohibited_indicators: dict[str, tuple[str, ...]] | None = None,
) -> GuardrailCheck:
    """Check LLM response content against scaffold guardrails.

    Two kinds of checks:
    * **Escalation triggers** -- if enough keywords from a trigger phrase
      appear in the content, flag it (soft: informational only).
    * **Prohibited-action patterns** -- domain-provided indicators that
      map natural-language prohibited_actions to concrete string patterns.
      A match here marks the check as *failed* so downstream sanitisation
      can redact the offending sentences.

    Args:
        content: The LLM's raw response text.
        scaffold: The scaffold that produced the response.
        prohibited_indicators: Domain-specific map of prohibition category
            to tuple of indicator phrases.  Provided by DomainConfig.
            If None, only escalation trigger checking is performed.
    """
    flags: list[str] = []
    content_lower = " ".join(content.lower().split())

    # --- escalation triggers (soft flags) ---
    for trigger in scaffold.guardrails.escalation_triggers:
        trigger_keywords = trigger.lower().split()
        matches = sum(1 for word in trigger_keywords if word in content_lower)
        if matches >= len(trigger_keywords) * 0.6:
            flags.append(f"escalation_trigger_detected: {trigger}")

    # --- prohibited-action patterns (hard flags) ---
    if prohibited_indicators:
        for action, patterns in prohibited_indicators.items():
            for pattern in patterns:
                if _contains_indicator(content_lower, pattern):
                    flags.append(
                        f"prohibited_pattern_detected: {action} ('{pattern}')"
                    )

    passed = not any(f.startswith("prohibited_pattern") for f in flags)

    if flags:
        logger.warning("Guardrail flags for scaffold %s: %s", scaffold.id, flags)

    return GuardrailCheck(passed=passed, flags=flags)


# ---------------------------------------------------------------------------
# Content sanitisation
# ---------------------------------------------------------------------------

def sanitize_content(
    content: str,
    guardrail_check: GuardrailCheck,
    redaction_message: str = "[Removed: contains prohibited content]",
) -> str:
    """Remove or redact prohibited patterns from LLM output.

    If the guardrail check passed, return content unchanged.
    Otherwise, redact sentences containing prohibited phrases.

    Args:
        content: The LLM's response text.
        guardrail_check: Result from check_guardrails.
        redaction_message: Domain-specific message to replace redacted
            sentences with.  Provided by DomainConfig.
    """
    if guardrail_check.passed:
        return content

    # Build list of prohibited phrases that were detected
    prohibited_phrases: list[str] = []
    for flag in guardrail_check.flags:
        if flag.startswith("prohibited_pattern_detected:"):
            match = re.search(r"\('([^']+)'\)", flag)
            if match:
                prohibited_phrases.append(match.group(1))

    if not prohibited_phrases:
        return content

    # Redact sentences containing prohibited phrases
    sanitized = content
    for phrase in prohibited_phrases:
        escaped_phrase = re.escape(phrase).replace(r"\ ", r"\s+")
        pattern = re.compile(
            r"[^.!?\n]*" + escaped_phrase + r"[^.!?\n]*[.!?]?",
            re.IGNORECASE,
        )
        sanitized = pattern.sub(redaction_message, sanitized)

    return sanitized


# ---------------------------------------------------------------------------
# Disclaimer enforcement
# ---------------------------------------------------------------------------

def enforce_disclaimers(content: str, scaffold: Scaffold) -> tuple[str, list[str]]:
    """Ensure scaffold-required disclaimers appear in the final response.

    Any disclaimers missing from the LLM output are appended as a footer.

    Returns:
        A tuple of (possibly modified content, list of flags).
    """
    disclaimers = [d.strip() for d in scaffold.guardrails.disclaimers if d.strip()]
    if not disclaimers:
        return content, []

    def _norm(s: str) -> str:
        return " ".join(s.lower().split())

    content_norm = _norm(content)
    missing = [d for d in disclaimers if _norm(d) not in content_norm]
    if not missing:
        return content, []

    footer = "\n\n---\nDisclaimers:\n" + "\n".join(f"- {d}" for d in missing)
    return content + footer, [f"disclaimer_appended: {d}" for d in missing]


# ---------------------------------------------------------------------------
# Context export extraction
# ---------------------------------------------------------------------------

def extract_context_exports(
    content: str,
    scaffold: Scaffold,
    data_context: dict[str, Any],
) -> dict[str, Any]:
    """Extract cross-domain context export fields from the response.

    Strategy:
    1. If the field exists in *data_context*, use it directly.
    2. Otherwise, attempt to extract from LLM content via pattern matching.
    3. If neither yields a result, skip the field.
    """
    exports: dict[str, Any] = {}

    for export_field in scaffold.context_exports:
        field_name = export_field.field_name

        if field_name in data_context:
            exports[field_name] = data_context[field_name]
            continue

        extracted = _extract_field_from_content(
            content, field_name, export_field.type
        )
        if extracted is not None:
            exports[field_name] = extracted

    return exports


def _extract_field_from_content(
    content: str, field_name: str, field_type: str
) -> Any:
    """Try to extract a named field value from LLM output text."""
    readable = field_name.replace("_", r"[\s_]")

    if field_type in ("number", "float", "int", "currency"):
        pattern = re.compile(
            readable + r"[\s:]+\$?([\d,]+\.?\d*)",
            re.IGNORECASE,
        )
        match = pattern.search(content)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                return None

    elif field_type in ("string", "str", "text"):
        pattern = re.compile(
            readable + r"[\s:]+([^\n.]+)",
            re.IGNORECASE,
        )
        match = pattern.search(content)
        if match:
            return match.group(1).strip()

    return None
