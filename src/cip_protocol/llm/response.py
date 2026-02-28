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
    mantic_safety: dict[str, Any] | None = None


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


class ManticSafetyEvaluator:
    """Mantic friction detection over guardrail content signals.

    Soft evaluator — never produces hard violations.  Opt-in only; not
    included in ``default_guardrail_evaluators()``.
    """

    name = "mantic_safety"

    def __init__(
        self,
        prohibited_indicators: dict[str, tuple[str, ...]] | None = None,
        detection_threshold: float = 0.4,
        backend: str = "auto",
    ) -> None:
        self._detection_threshold = detection_threshold
        self._backend = backend
        self._prohibited_indicators = prohibited_indicators or {}
        # Pre-compile prohibited patterns (reuses the module-level helper)
        self._compiled_prohibited: list[tuple[str, re.Pattern[str]]] = []
        for patterns in self._prohibited_indicators.values():
            for pattern in patterns:
                normalized = " ".join(pattern.lower().split())
                if normalized:
                    self._compiled_prohibited.append(
                        (pattern, _compile_indicator_pattern(normalized))
                    )

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        # Short-circuit for very short content (streaming early chunks)
        if len(content) < 50:
            return GuardrailEvaluation(evaluator_name=self.name)

        content_lower = " ".join(content.lower().split())
        content_tokens = _tokenize(content_lower)

        # --- Layer 1: escalation_density ---
        triggers = list(scaffold.guardrails.escalation_triggers)
        if triggers:
            matched_triggers = sum(
                1 for trigger in triggers
                if sum(1 for t in _tokenize(trigger) if t in content_tokens)
                >= len(_tokenize(trigger)) * 0.6
            )
            escalation_density = min(1.0, matched_triggers / len(triggers))
        else:
            escalation_density = 0.0

        # --- Layer 2: prohibited_density ---
        if self._compiled_prohibited:
            matched_prohibited = sum(
                1 for _, regex in self._compiled_prohibited
                if regex.search(content_lower)
            )
            prohibited_density = min(1.0, matched_prohibited / len(self._compiled_prohibited))
        else:
            prohibited_density = 0.0

        # --- Layer 3: topic_sensitivity (Jaccard with trigger tokens) ---
        all_trigger_tokens: set[str] = set()
        for trigger in triggers:
            all_trigger_tokens.update(_tokenize(trigger))
        if all_trigger_tokens and content_tokens:
            intersection = content_tokens & all_trigger_tokens
            union = content_tokens | all_trigger_tokens
            topic_sensitivity = len(intersection) / len(union) if union else 0.0
        else:
            topic_sensitivity = 0.0

        # --- Layer 4: response_length_risk ---
        word_count = len(content.split())
        response_length_risk = min(1.0, word_count / 2000)

        layer_values = [
            escalation_density,
            prohibited_density,
            topic_sensitivity,
            response_length_risk,
        ]

        # Local import to stay within the CI import guard
        from cip_protocol.mantic_adapter import detect_safety_friction

        result = detect_safety_friction(
            layer_values=layer_values,
            backend=self._backend,
            detection_threshold=self._detection_threshold,
        )

        flags: list[str] = []
        if result.signal == "friction_detected":
            flags.append(
                f"mantic_safety_friction: m_score={result.m_score:.3f} "
                f"dominant={result.dominant_layer}"
            )

        import dataclasses

        return GuardrailEvaluation(
            evaluator_name=self.name,
            flags=flags,
            metadata={
                "detection_result": dataclasses.asdict(result),
                "layer_values": {
                    "escalation_density": escalation_density,
                    "prohibited_density": prohibited_density,
                    "topic_sensitivity": topic_sensitivity,
                    "response_length_risk": response_length_risk,
                },
            },
        )


class ArgumentStructureEvaluator:
    """Mantic friction detection over argument structure layer scores.

    Soft evaluator — never produces hard violations.  Only activates when
    ``"argument-analysis" in scaffold.tags``.  Extracts four layer scores
    from LLM response text via regex, then delegates to
    ``detect_argument_friction`` / ``classify_fallacy``.
    """

    name = "argument_structure"

    def __init__(
        self,
        detection_threshold: float = 0.4,
        backend: str = "auto",
    ) -> None:
        self._detection_threshold = detection_threshold
        self._backend = backend

    def evaluate(self, content: str, scaffold: Scaffold) -> GuardrailEvaluation:
        # Only activate for argument-analysis scaffolds
        if "argument-analysis" not in getattr(scaffold, "tags", []):
            return GuardrailEvaluation(evaluator_name=self.name)

        # Short-circuit for very short content
        if len(content) < 50:
            return GuardrailEvaluation(evaluator_name=self.name)

        # Extract layer scores from content
        layer_names = [
            "premise_strength",
            "inferential_link",
            "structural_validity",
            "scope_consistency",
        ]
        layer_values: dict[str, float] = {}
        for layer in layer_names:
            # Match patterns like "premise_strength: 0.7" or "premise strength  0.85"
            pattern = _compile(
                layer.replace("_", r"[\s_]+") + r"[\s:]+(\d+\.?\d*)",
                re.IGNORECASE,
            )
            match = pattern.search(content)
            if match:
                val = float(match.group(1))
                layer_values[layer] = max(0.0, min(1.0, val))

        # Need all 4 layers to proceed
        if len(layer_values) != 4:
            return GuardrailEvaluation(evaluator_name=self.name)

        import dataclasses

        from cip_protocol.mantic_adapter import (
            classify_fallacy,
            detect_argument_friction,
        )

        values_list = [layer_values[n] for n in layer_names]
        detection = detect_argument_friction(
            layer_values=values_list,
            backend=self._backend,
            detection_threshold=self._detection_threshold,
        )
        fallacy = classify_fallacy(detection, layer_values)

        flags: list[str] = []
        if not fallacy.is_valid:
            flags.append(
                f"argument_friction: {fallacy.display_name} "
                f"(confidence={fallacy.confidence:.3f})"
            )

        return GuardrailEvaluation(
            evaluator_name=self.name,
            flags=flags,
            metadata={
                "detection_result": dataclasses.asdict(detection),
                "fallacy_result": dataclasses.asdict(fallacy),
                "layer_values": layer_values,
            },
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

    # Extract mantic_safety detection result if present
    mantic_safety: dict[str, Any] | None = None
    for finding in findings:
        if finding["evaluator"] == "mantic_safety":
            dr = finding["metadata"].get("detection_result")
            if dr is not None:
                mantic_safety = dr
            break

    return GuardrailCheck(
        passed=not hard_violations,
        flags=all_flags,
        hard_violations=hard_violations,
        matched_phrases=matched_phrases,
        evaluator_findings=findings,
        mantic_safety=mantic_safety,
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
