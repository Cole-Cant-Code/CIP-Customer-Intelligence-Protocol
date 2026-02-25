"""Microbenchmarks for CIP hot paths (baseline vs optimized implementations)."""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass

from cip_protocol.llm.response import (
    ProhibitedPatternEvaluator,
    check_guardrails,
    enforce_disclaimers,
    extract_context_exports,
    sanitize_content,
)
from cip_protocol.scaffold.matcher import (
    EXACT_SIGNAL_BONUS,
    INTENT_WEIGHT,
    KEYWORD_WEIGHT,
    MIN_SIGNAL_COVERAGE,
    _score_scaffolds,
    _tokenize,
    clear_matcher_cache,
)
from cip_protocol.scaffold.models import (
    ContextField,
    Scaffold,
    ScaffoldApplicability,
    ScaffoldFraming,
    ScaffoldGuardrails,
    ScaffoldOutputCalibration,
)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)")


def _build_scaffold(
    scaffold_id: str,
    *,
    keywords: list[str],
    intent_signals: list[str],
) -> Scaffold:
    return Scaffold(
        id=scaffold_id,
        version="1.0",
        domain="bench",
        display_name=f"Bench {scaffold_id}",
        description="Benchmark scaffold",
        applicability=ScaffoldApplicability(
            tools=[],
            keywords=keywords,
            intent_signals=intent_signals,
        ),
        framing=ScaffoldFraming(role="Analyst", perspective="Fast", tone="neutral"),
        reasoning_framework={"steps": ["Analyze", "Respond"]},
        domain_knowledge_activation=[],
        output_calibration=ScaffoldOutputCalibration(
            format="structured_narrative",
            format_options=["structured_narrative"],
        ),
        guardrails=ScaffoldGuardrails(
            disclaimers=["Not professional advice."],
            escalation_triggers=["severe distress"],
            prohibited_actions=[],
        ),
        context_exports=[ContextField(field_name="total_amount", type="currency")],
    )


@dataclass
class _ScaffoldCompiled:
    scaffold: Scaffold
    signal_data: list[tuple[set[str], re.Pattern[str]]]
    keyword_patterns: list[re.Pattern[str]]


def _compile_baseline_scaffolds(scaffolds: list[Scaffold]) -> list[_ScaffoldCompiled]:
    compiled: list[_ScaffoldCompiled] = []
    for scaffold in scaffolds:
        signal_data: list[tuple[set[str], re.Pattern[str]]] = []
        for signal in scaffold.applicability.intent_signals:
            signal_tokens = _tokenize(signal)
            if signal_tokens:
                signal_data.append((signal_tokens, _phrase_pattern(signal)))

        keyword_patterns = [_phrase_pattern(kw) for kw in scaffold.applicability.keywords if kw]
        compiled.append(_ScaffoldCompiled(scaffold, signal_data, keyword_patterns))
    return compiled


def _baseline_score_scaffolds(
    compiled_scaffolds: list[_ScaffoldCompiled], user_input: str,
) -> Scaffold | None:
    user_lower = user_input.lower()
    user_tokens = _tokenize(user_input)
    best_match: Scaffold | None = None
    best_score = 0.0

    for item in compiled_scaffolds:
        score = 0.0
        for signal_tokens, signal_pattern in item.signal_data:
            coverage = sum(1 for t in signal_tokens if t in user_tokens) / len(signal_tokens)
            if coverage >= MIN_SIGNAL_COVERAGE:
                score += INTENT_WEIGHT * coverage
            if signal_pattern.search(user_lower):
                score += EXACT_SIGNAL_BONUS

        for keyword_pattern in item.keyword_patterns:
            if keyword_pattern.search(user_lower):
                score += KEYWORD_WEIGHT

        if score > best_score:
            best_score = score
            best_match = item.scaffold

    return best_match if best_score > 0 else None


def _time(label: str, fn, iterations: int) -> tuple[str, float]:
    samples: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        elapsed = time.perf_counter() - start
        samples.append(elapsed / iterations)
    mean = statistics.mean(samples)
    print(f"{label:48s} {mean * 1_000:.3f} ms/op")
    return label, mean


def bench_prohibited_patterns() -> None:
    print("\n[1] Prohibited pattern detection: baseline loop vs token-prefilter engine")
    indicators = {
        f"action_{i}": (
            f"disallowed phrase {i}",
            f"never do behavior {i}",
            f"strictly avoid tactic {i}",
        )
        for i in range(120)
    }
    content = (
        " ".join(
            f"This answer mentions disallowed phrase {i} and safe text around it."
            for i in range(20, 40)
        )
        + " end."
    ).lower()

    compiled_baseline: list[tuple[str, str, re.Pattern[str]]] = []
    for action, patterns in indicators.items():
        for raw in patterns:
            normalized = " ".join(raw.lower().split())
            escaped = re.escape(normalized).replace(r"\ ", r"\s+")
            compiled_baseline.append(
                (action, raw, re.compile(rf"(?<!\w){escaped}(?!\w)"))
            )

    evaluator = ProhibitedPatternEvaluator(indicators)
    scaffold = _build_scaffold("guardrail", keywords=[], intent_signals=[])

    def baseline() -> int:
        hits = 0
        for _action, _raw, pattern in compiled_baseline:
            if pattern.search(content):
                hits += 1
        return hits

    def optimized() -> int:
        return len(evaluator.evaluate(content, scaffold).hard_violations)

    _, baseline_mean = _time("baseline_prohibited_pattern_loop", baseline, iterations=300)
    _, optimized_mean = _time("optimized_token_prefilter_evaluator", optimized, iterations=300)
    print(f"speedup: {baseline_mean / optimized_mean:.2f}x")


def bench_matcher_scaling() -> None:
    print("\n[2] Matcher scoring: full-scan baseline vs token-pruned scorer")
    scaffolds: list[Scaffold] = []
    for i in range(1500):
        scaffolds.append(_build_scaffold(
            f"s_{i}",
            keywords=[f"k{i}", f"term{i}", f"topic{i}"],
            intent_signals=[f"intent phrase {i}", f"workflow {i}"],
        ))

    scaffolds.append(_build_scaffold(
        "target",
        keywords=["budget", "spending"],
        intent_signals=["where is my money going"],
    ))

    user_input = "where is my money going and how should i budget spending?"

    clear_matcher_cache()
    compiled_baseline = _compile_baseline_scaffolds(scaffolds)

    def baseline() -> str | None:
        match = _baseline_score_scaffolds(compiled_baseline, user_input)
        return match.id if match else None

    def optimized() -> str | None:
        match = _score_scaffolds(scaffolds, user_input)
        return match.id if match else None

    baseline_id = baseline()
    optimized_id = optimized()
    if baseline_id != optimized_id:
        raise RuntimeError(
            f"Matcher mismatch: baseline={baseline_id!r}, optimized={optimized_id!r}"
        )

    _, baseline_mean = _time("baseline_full_scan_matcher", baseline, iterations=80)
    _, optimized_mean = _time("optimized_token_pruned_matcher", optimized, iterations=80)
    print(f"speedup: {baseline_mean / optimized_mean:.2f}x")


def bench_stream_chunk_pipeline() -> None:
    print("\n[3] Streaming chunk pipeline: old per-chunk postprocess vs deferred postprocess")
    scaffold = _build_scaffold("stream", keywords=["budget"], intent_signals=[])
    content_chunks = [f"chunk {i} with safe budget context. " for i in range(1, 121)]
    evaluators = []

    def old_pipeline() -> int:
        collected: list[str] = []
        flags_count = 0
        for chunk in content_chunks:
            collected.append(chunk)
            current = "".join(collected)
            check = check_guardrails(current, scaffold, evaluators=evaluators)
            current = sanitize_content(current, check)
            current, flags = enforce_disclaimers(current, scaffold)
            flags_count += len(flags)
            extract_context_exports(current, scaffold, {"total_amount": 42.0})
        return flags_count

    def new_pipeline() -> int:
        collected: list[str] = []
        for chunk in content_chunks:
            collected.append(chunk)
            current = "".join(collected)
            _ = check_guardrails(current, scaffold, evaluators=evaluators)

        final = "".join(collected)
        check = check_guardrails(final, scaffold, evaluators=evaluators)
        final = sanitize_content(final, check)
        final, flags = enforce_disclaimers(final, scaffold)
        extract_context_exports(final, scaffold, {"total_amount": 42.0})
        return len(flags)

    _, old_mean = _time("baseline_old_chunk_postprocess_every_chunk", old_pipeline, iterations=20)
    _, new_mean = _time("optimized_deferred_postprocess_pipeline", new_pipeline, iterations=20)
    print(f"speedup: {old_mean / new_mean:.2f}x")


def main() -> None:
    print("CIP Hot Path Benchmarks")
    bench_prohibited_patterns()
    bench_matcher_scaling()
    bench_stream_chunk_pipeline()


if __name__ == "__main__":
    main()
