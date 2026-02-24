"""Pluggable guardrail evaluators, content sanitization, and disclaimer enforcement."""

from __future__ import annotations

import asyncio
import functools
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cip_protocol.scaffold.models import Scaffold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex engine: prefer google-re2 for linear-time guarantees if installed,
# with automatic fallback to stdlib re for patterns re2 can't handle.
# ---------------------------------------------------------------------------

try:
    import re2 as _re_engine  # type: ignore[import-untyped]
except ImportError:
    _re_engine = re  # type: ignore[assignment]


def _compile(pattern: str, flags: int = 0) -> re.Pattern[str]:
    """Compile with re2 if available, else stdlib re."""
    if _re_engine is not re:
        try:
            return _re_engine.compile(pattern, flags)
        except Exception:
            pass
    return re.compile(pattern, flags)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cached pattern compilation helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def _compile_indicator_pattern(normalized: str) -> re.Pattern[str]:
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    return _compile(rf"(?<!\w){escaped}(?!\w)")


@functools.lru_cache(maxsize=256)
def _compile_redaction_pattern(phrase: str) -> re.Pattern[str]:
    truncated = phrase[:500] if len(phrase) > 500 else phrase
    escaped = re.escape(truncated).replace(r"\ ", r"\s+")
    return _compile(
        r"[^.!?\n]{0,500}(?<!\w)" + escaped + r"(?!\w)[^.!?\n]{0,500}[.!?]?",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Indicator matching
# ---------------------------------------------------------------------------

def _contains_indicator(content_lower: str, pattern: str) -> bool:
    normalized = " ".join(pattern.lower().split())
    if not normalized:
        return False
    compiled = _compile_indicator_pattern(normalized)
    return bool(compiled.search(content_lower))


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

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

    async def async_evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        return self.evaluate(content, scaffold)


class ProhibitedPatternEvaluator:
    name = "prohibited_pattern"

    def __init__(self, indicators: dict[str, tuple[str, ...]]) -> None:
        self.indicators = indicators
        self._compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        for action, patterns in indicators.items():
            compiled_list: list[tuple[str, re.Pattern[str]]] = []
            for pattern in patterns:
                normalized = " ".join(pattern.lower().split())
                if not normalized:
                    continue
                compiled_list.append((pattern, _compile_indicator_pattern(normalized)))
            self._compiled[action] = compiled_list

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        content_lower = " ".join(content.lower().split())
        flags: list[str] = []
        violations: list[str] = []
        phrases: list[str] = []

        for action, compiled_patterns in self._compiled.items():
            for raw_pattern, regex in compiled_patterns:
                if regex.search(content_lower):
                    flags.append(f"prohibited_pattern_detected: {action} ('{raw_pattern}')")
                    violations.append(action)
                    phrases.append(raw_pattern)

        return GuardrailEvaluation(
            evaluator_name=self.name,
            flags=flags,
            hard_violations=violations,
            matched_phrases=phrases,
        )

    async def async_evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        return self.evaluate(content, scaffold)


class RegexPolicyEvaluator:
    name = "regex_policy"

    def __init__(self, policy_patterns: dict[str, str]) -> None:
        self.compiled = {
            name: _compile(pattern, re.IGNORECASE)
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

    async def async_evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        return self.evaluate(content, scaffold)


# ---------------------------------------------------------------------------
# Guardrail orchestration (sync + async)
# ---------------------------------------------------------------------------

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


def _aggregate_results(
    results: list[GuardrailEvaluation],
) -> GuardrailCheck:
    all_flags: list[str] = []
    hard_violations: list[str] = []
    matched_phrases: list[str] = []
    findings: list[dict[str, Any]] = []

    for result in results:
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


def check_guardrails(
    content: str,
    scaffold: Scaffold,
    prohibited_indicators: dict[str, tuple[str, ...]] | None = None,
    evaluators: list[GuardrailEvaluator] | None = None,
) -> GuardrailCheck:
    active = evaluators or default_guardrail_evaluators(prohibited_indicators)
    results = [ev.evaluate(content, scaffold) for ev in active]
    return _aggregate_results(results)


async def _run_evaluator(
    evaluator: GuardrailEvaluator, content: str, scaffold: Scaffold,
) -> GuardrailEvaluation:
    async_fn = getattr(evaluator, "async_evaluate", None)
    if async_fn is not None and callable(async_fn):
        return await async_fn(content, scaffold)
    return evaluator.evaluate(content, scaffold)


async def check_guardrails_async(
    content: str,
    scaffold: Scaffold,
    prohibited_indicators: dict[str, tuple[str, ...]] | None = None,
    evaluators: list[GuardrailEvaluator] | None = None,
) -> GuardrailCheck:
    """Async version of check_guardrails. Runs evaluators concurrently."""
    active = evaluators or default_guardrail_evaluators(prohibited_indicators)
    results = await asyncio.gather(
        *(_run_evaluator(ev, content, scaffold) for ev in active)
    )
    return _aggregate_results(list(results))


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

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
        pattern = _compile_redaction_pattern(phrase)
        sanitized = pattern.sub(redaction_message, sanitized)

    return sanitized


# ---------------------------------------------------------------------------
# Disclaimers and context export
# ---------------------------------------------------------------------------

def enforce_disclaimers(content: str, scaffold: Scaffold) -> tuple[str, list[str]]:
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
    # Escape regex metacharacters while keeping underscore/space matching flexible.
    parts = [re.escape(part) for part in re.split(r"[_\s]+", field_name.strip()) if part]
    if not parts:
        return None
    readable = r"[\s_]+".join(parts)

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
