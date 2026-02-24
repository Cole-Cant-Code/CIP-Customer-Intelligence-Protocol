"""Response parsing and pluggable guardrail enforcement for LLM output."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cip_protocol.scaffold.models import Scaffold

logger = logging.getLogger(__name__)


@dataclass
class GuardrailEvaluation:
    """Evaluation output produced by a single guardrail evaluator."""

    evaluator_name: str
    flags: list[str] = field(default_factory=list)
    hard_violations: list[str] = field(default_factory=list)
    matched_phrases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailCheck:
    """Aggregate result of all configured guardrail evaluators."""

    passed: bool
    flags: list[str] = field(default_factory=list)
    hard_violations: list[str] = field(default_factory=list)
    matched_phrases: list[str] = field(default_factory=list)
    evaluator_findings: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class GuardrailEvaluator(Protocol):
    """Pluggable guardrail evaluator protocol."""

    name: str

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        """Evaluate content and return findings."""
        raise NotImplementedError


def _contains_indicator(content_lower: str, pattern: str) -> bool:
    """Match indicator phrases with word boundaries and flexible whitespace."""
    normalized = " ".join(pattern.lower().split())
    if not normalized:
        return False
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    regex = re.compile(rf"(?<!\w){escaped}(?!\w)")
    return bool(regex.search(content_lower))


class EscalationTriggerEvaluator:
    """Soft evaluator that flags escalation trigger phrases."""

    name = "escalation_trigger"

    def __init__(self, threshold_ratio: float = 0.6) -> None:
        self.threshold_ratio = threshold_ratio

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        content_lower = " ".join(content.lower().split())
        flags: list[str] = []

        for trigger in scaffold.guardrails.escalation_triggers:
            trigger_keywords = trigger.lower().split()
            if not trigger_keywords:
                continue
            matches = sum(1 for word in trigger_keywords if word in content_lower)
            if matches >= len(trigger_keywords) * self.threshold_ratio:
                flags.append(f"escalation_trigger_detected: {trigger}")

        return GuardrailEvaluation(
            evaluator_name=self.name,
            flags=flags,
        )


class ProhibitedPatternEvaluator:
    """Hard evaluator based on deterministic prohibited indicators."""

    name = "prohibited_pattern"

    def __init__(self, indicators: dict[str, tuple[str, ...]]) -> None:
        self.indicators = indicators

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        _ = scaffold
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
    """Hard evaluator for domain-defined regex policy rules."""

    name = "regex_policy"

    def __init__(self, policy_patterns: dict[str, str]) -> None:
        self.compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in policy_patterns.items()
        }

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        _ = scaffold
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
    """Build the default evaluator stack."""
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
    """Run evaluator stack and aggregate guardrail results."""
    active_evaluators = evaluators or default_guardrail_evaluators(prohibited_indicators)

    all_flags: list[str] = []
    hard_violations: list[str] = []
    matched_phrases: list[str] = []
    findings: list[dict[str, Any]] = []

    for evaluator in active_evaluators:
        result = evaluator.evaluate(content, scaffold)
        all_flags.extend(result.flags)
        hard_violations.extend(result.hard_violations)
        matched_phrases.extend(result.matched_phrases)
        findings.append(
            {
                "evaluator": result.evaluator_name,
                "flags": result.flags,
                "hard_violations": result.hard_violations,
                "matched_phrases": result.matched_phrases,
                "metadata": result.metadata,
            }
        )

    passed = len(hard_violations) == 0

    if all_flags:
        logger.warning("Guardrail flags for scaffold %s: %s", scaffold.id, all_flags)

    return GuardrailCheck(
        passed=passed,
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
    """Redact sentences containing prohibited phrases from output content."""
    if guardrail_check.passed:
        return content

    prohibited_phrases = list(guardrail_check.matched_phrases)
    if not prohibited_phrases:
        for flag in guardrail_check.flags:
            if flag.startswith("prohibited_pattern_detected:"):
                match = re.search(r"\('([^']+)'\)", flag)
                if match:
                    prohibited_phrases.append(match.group(1))

    if not prohibited_phrases:
        # Hard violation without phrase-level evidence: redact full response.
        if guardrail_check.hard_violations:
            return redaction_message
        return content

    sanitized = content
    for phrase in prohibited_phrases:
        escaped_phrase = re.escape(phrase).replace(r"\ ", r"\s+")
        pattern = re.compile(
            r"[^.!?\n]*(?<!\\w)"
            + escaped_phrase
            + r"(?!\\w)[^.!?\n]*[.!?]?",
            re.IGNORECASE,
        )
        sanitized = pattern.sub(redaction_message, sanitized)

    return sanitized


def enforce_disclaimers(content: str, scaffold: Scaffold) -> tuple[str, list[str]]:
    """Ensure scaffold-required disclaimers appear in final output."""
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


def extract_context_exports(
    content: str,
    scaffold: Scaffold,
    data_context: dict[str, Any],
) -> dict[str, Any]:
    """Extract cross-domain context export fields from response content."""
    exports: dict[str, Any] = {}

    for export_field in scaffold.context_exports:
        field_name = export_field.field_name

        if field_name in data_context:
            exports[field_name] = data_context[field_name]
            continue

        extracted = _extract_field_from_content(content, field_name, export_field.type)
        if extracted is not None:
            exports[field_name] = extracted

    return exports


def _extract_field_from_content(content: str, field_name: str, field_type: str) -> Any:
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
