"""Pluggable guardrail evaluators, content sanitization, and disclaimer enforcement."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cip_protocol.scaffold.models import Scaffold

logger = logging.getLogger(__name__)


@dataclass
class GuardrailEvaluation:
    evaluator_name: str
    flags: list[str] = field(default_factory=list)
    hard_violations: list[str] = field(default_factory=list)
    matched_phrases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailCheck:
    passed: bool
    flags: list[str] = field(default_factory=list)
    hard_violations: list[str] = field(default_factory=list)
    matched_phrases: list[str] = field(default_factory=list)
    evaluator_findings: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class GuardrailEvaluator(Protocol):
    name: str

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation: ...


def _contains_indicator(content_lower: str, pattern: str) -> bool:
    normalized = " ".join(pattern.lower().split())
    if not normalized:
        return False
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<!\w){escaped}(?!\w)", content_lower))


class EscalationTriggerEvaluator:
    name = "escalation_trigger"

    def __init__(self, threshold_ratio: float = 0.6) -> None:
        self.threshold_ratio = threshold_ratio

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        content_lower = " ".join(content.lower().split())
        flags: list[str] = []

        for trigger in scaffold.guardrails.escalation_triggers:
            words = trigger.lower().split()
            if not words:
                continue
            if sum(1 for w in words if w in content_lower) >= len(words) * self.threshold_ratio:
                flags.append(f"escalation_trigger_detected: {trigger}")

        return GuardrailEvaluation(evaluator_name=self.name, flags=flags)


class ProhibitedPatternEvaluator:
    name = "prohibited_pattern"

    def __init__(self, indicators: dict[str, tuple[str, ...]]) -> None:
        self.indicators = indicators

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        content_lower = " ".join(content.lower().split())
        flags: list[str] = []
        violations: list[str] = []
        phrases: list[str] = []

        for action, patterns in self.indicators.items():
            for pattern in patterns:
                if _contains_indicator(content_lower, pattern):
                    flags.append(f"prohibited_pattern_detected: {action} ('{pattern}')")
                    violations.append(action)
                    phrases.append(pattern)

        return GuardrailEvaluation(
            evaluator_name=self.name,
            flags=flags,
            hard_violations=violations,
            matched_phrases=phrases,
        )


class RegexPolicyEvaluator:
    name = "regex_policy"

    def __init__(self, policy_patterns: dict[str, str]) -> None:
        self.compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in policy_patterns.items()
        }

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        flags: list[str] = []
        violations: list[str] = []

        for name, pattern in self.compiled.items():
            if pattern.search(content):
                flags.append(f"regex_policy_violation: {name}")
                violations.append(name)

        return GuardrailEvaluation(
            evaluator_name=self.name,
            flags=flags,
            hard_violations=violations,
        )


def default_guardrail_evaluators(
    prohibited_indicators: dict[str, tuple[str, ...]] | None = None,
    regex_policy_patterns: dict[str, str] | None = None,
) -> list[GuardrailEvaluator]:
    evaluators: list[GuardrailEvaluator] = [EscalationTriggerEvaluator()]
    if prohibited_indicators:
        evaluators.append(ProhibitedPatternEvaluator(prohibited_indicators))
    if regex_policy_patterns:
        evaluators.append(RegexPolicyEvaluator(regex_policy_patterns))
    return evaluators


def check_guardrails(
    content: str,
    scaffold: Scaffold,
    prohibited_indicators: dict[str, tuple[str, ...]] | None = None,
    evaluators: list[GuardrailEvaluator] | None = None,
) -> GuardrailCheck:
    active = evaluators or default_guardrail_evaluators(prohibited_indicators)

    all_flags: list[str] = []
    hard_violations: list[str] = []
    matched_phrases: list[str] = []
    findings: list[dict[str, Any]] = []

    for evaluator in active:
        result = evaluator.evaluate(content, scaffold)
        all_flags.extend(result.flags)
        hard_violations.extend(result.hard_violations)
        matched_phrases.extend(result.matched_phrases)
        findings.append({
            "evaluator": result.evaluator_name,
            "flags": result.flags,
            "hard_violations": result.hard_violations,
            "matched_phrases": result.matched_phrases,
            "metadata": result.metadata,
        })

    return GuardrailCheck(
        passed=not hard_violations,
        flags=all_flags,
        hard_violations=hard_violations,
        matched_phrases=matched_phrases,
        evaluator_findings=findings,
    )


def sanitize_content(
    content: str,
    guardrail_check: GuardrailCheck,
    redaction_message: str = "[Removed: contains prohibited content]",
) -> str:
    if guardrail_check.passed:
        return content

    phrases = list(guardrail_check.matched_phrases)
    if not phrases:
        # Try extracting from flag strings as fallback
        for flag in guardrail_check.flags:
            if flag.startswith("prohibited_pattern_detected:"):
                match = re.search(r"\('([^']+)'\)", flag)
                if match:
                    phrases.append(match.group(1))

    if not phrases:
        return redaction_message if guardrail_check.hard_violations else content

    sanitized = content
    for phrase in phrases:
        escaped = re.escape(phrase).replace(r"\ ", r"\s+")
        pattern = re.compile(
            r"[^.!?\n]*(?<!\\w)" + escaped + r"(?!\\w)[^.!?\n]*[.!?]?",
            re.IGNORECASE,
        )
        sanitized = pattern.sub(redaction_message, sanitized)

    return sanitized


def enforce_disclaimers(content: str, scaffold: Scaffold) -> tuple[str, list[str]]:
    disclaimers = [d.strip() for d in scaffold.guardrails.disclaimers if d.strip()]
    if not disclaimers:
        return content, []

    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    content_norm = norm(content)
    missing = [d for d in disclaimers if norm(d) not in content_norm]
    if not missing:
        return content, []

    footer = "\n\n---\nDisclaimers:\n" + "\n".join(f"- {d}" for d in missing)
    return content + footer, [f"disclaimer_appended: {d}" for d in missing]


def extract_context_exports(
    content: str,
    scaffold: Scaffold,
    data_context: dict[str, Any],
) -> dict[str, Any]:
    exports: dict[str, Any] = {}

    for export_field in scaffold.context_exports:
        name = export_field.field_name

        if name in data_context:
            exports[name] = data_context[name]
            continue

        extracted = _extract_field_from_content(content, name, export_field.type)
        if extracted is not None:
            exports[name] = extracted

    return exports


def _extract_field_from_content(content: str, field_name: str, field_type: str) -> Any:
    readable = field_name.replace("_", r"[\s_]")

    if field_type in ("number", "float", "int", "currency"):
        match = re.search(readable + r"[\s:]+\$?([\d,]+\.?\d*)", content, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                return None

    elif field_type in ("string", "str", "text"):
        match = re.search(readable + r"[\s:]+([^\n.]+)", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None
