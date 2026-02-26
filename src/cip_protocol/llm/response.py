"""Pluggable guardrail evaluators, content sanitization, and disclaimer enforcement."""

from __future__ import annotations

import asyncio
import functools
import inspect
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
    return _compile(rf"\b{escaped}\b")


@functools.lru_cache(maxsize=256)
def _compile_redaction_pattern(phrase: str) -> re.Pattern[str]:
    truncated = phrase[:500] if len(phrase) > 500 else phrase
    escaped = re.escape(truncated).replace(r"\ ", r"\s+")
    return _compile(
        r"[^.!?\n]{0,500}\b" + escaped + r"\b[^.!?\n]{0,500}[.!?]?",
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


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

class EscalationTriggerEvaluator:
    name = "escalation_trigger"

    def __init__(self, threshold_ratio: float = 0.6) -> None:
        self.threshold_ratio = threshold_ratio

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        content_lower = " ".join(content.lower().split())
        content_tokens = _tokenize(content_lower)
        flags: list[str] = []

        for trigger in scaffold.guardrails.escalation_triggers:
            trigger_tokens = _tokenize(trigger)
            if not trigger_tokens:
                continue
            if (
                sum(1 for token in trigger_tokens if token in content_tokens)
                >= len(trigger_tokens) * self.threshold_ratio
            ):
                flags.append(f"escalation_trigger_detected: {trigger}")

        return GuardrailEvaluation(evaluator_name=self.name, flags=flags)


class ProhibitedPatternEvaluator:
    name = "prohibited_pattern"

    def __init__(self, indicators: dict[str, tuple[str, ...]]) -> None:
        self.indicators = indicators
        self._compiled: dict[str, list[tuple[str, re.Pattern[str], set[str]]]] = {}

        for action, patterns in indicators.items():
            compiled_list: list[tuple[str, re.Pattern[str], set[str]]] = []
            for pattern in patterns:
                normalized = " ".join(pattern.lower().split())
                if not normalized:
                    continue
                compiled_list.append((
                    pattern,
                    _compile_indicator_pattern(normalized),
                    _tokenize(normalized),
                ))
            self._compiled[action] = compiled_list

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        _ = scaffold
        content_lower = " ".join(content.lower().split())
        content_tokens = _tokenize(content_lower)
        flags: list[str] = []
        violations: list[str] = []
        phrases: list[str] = []

        for action, compiled_patterns in self._compiled.items():
            for raw_pattern, regex, pattern_tokens in compiled_patterns:
                if pattern_tokens and not pattern_tokens.issubset(content_tokens):
                    continue
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


class RegexPolicyEvaluator:
    name = "regex_policy"

    def __init__(self, policy_patterns: dict[str, str]) -> None:
        self.compiled = {
            name: _compile(pattern, re.IGNORECASE)
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


async def check_guardrails_async(
    content: str,
    scaffold: Scaffold,
    prohibited_indicators: dict[str, tuple[str, ...]] | None = None,
    evaluators: list[GuardrailEvaluator] | None = None,
) -> GuardrailCheck:
    """Async version of check_guardrails. Runs async evaluators concurrently."""
    active = evaluators or default_guardrail_evaluators(prohibited_indicators)
    sync_results: list[GuardrailEvaluation] = []
    async_coroutines: list[Any] = []

    for evaluator in active:
        async_fn = getattr(evaluator, "async_evaluate", None)
        if async_fn is None or not callable(async_fn):
            sync_results.append(evaluator.evaluate(content, scaffold))
            continue

        maybe_awaitable = async_fn(content, scaffold)
        if inspect.isawaitable(maybe_awaitable):
            async_coroutines.append(maybe_awaitable)
        else:
            sync_results.append(maybe_awaitable)

    if async_coroutines:
        sync_results.extend(await asyncio.gather(*async_coroutines))

    return _aggregate_results(sync_results)


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


@functools.lru_cache(maxsize=256)
def _compile_export_extractor(
    field_name: str, field_type: str,
) -> tuple[re.Pattern[str] | None, str]:
    # Escape regex metacharacters while keeping underscore/space matching flexible.
    parts = [re.escape(part) for part in re.split(r"[_\s]+", field_name.strip()) if part]
    if not parts:
        return None, ""

    readable = r"[\s_]+".join(parts)
    kind = field_type.lower().strip()

    if kind in ("number", "float", "int", "currency"):
        return _compile(readable + r"[\s:]+\$?([\d,]+\.?\d*)", re.IGNORECASE), "number"
    if kind in ("string", "str", "text"):
        return _compile(readable + r"[\s:]+([^\n.]+)", re.IGNORECASE), "string"
    return None, ""


def _extract_field_from_content(content: str, field_name: str, field_type: str) -> Any:
    extractor, extractor_type = _compile_export_extractor(field_name, field_type)
    if extractor is None:
        return None

    match = extractor.search(content)
    if not match:
        return None

    if extractor_type == "number":
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    if extractor_type == "string":
        return match.group(1).strip()

    return None
