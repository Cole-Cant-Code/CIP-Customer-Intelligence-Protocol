"""Control cockpit: presets, run policies, and constraint language."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import Field, field_validator

from cip_protocol.scaffold.models import _StrictModel

# ---------------------------------------------------------------------------
# Control Preset
# ---------------------------------------------------------------------------


class ControlPreset(_StrictModel):
    """A named behavior profile mapping to concrete parameter overrides."""

    name: str
    temperature: float | None = None
    max_tokens: int | None = None
    tone_variant: str | None = None
    output_format: str | None = None
    max_length_guidance: str | None = None
    compact: bool | None = None
    scaffold_selection_bias: dict[str, float] = Field(default_factory=dict)
    skip_disclaimers: bool = False
    extra_must_include: list[str] = Field(default_factory=list)
    extra_never_include: list[str] = Field(default_factory=list)
    extra_prohibited_actions: list[str] = Field(default_factory=list)
    remove_prohibited_actions: list[str] = Field(default_factory=list)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


BUILTIN_PRESETS: dict[str, ControlPreset] = {
    "creative": ControlPreset(
        name="creative",
        temperature=0.8,
        max_length_guidance="no length constraint",
    ),
    "precise": ControlPreset(
        name="precise",
        temperature=0.1,
        output_format="bullet_points",
        max_length_guidance="concise, under 300 words",
        compact=True,
    ),
    "aggressive": ControlPreset(
        name="aggressive",
        temperature=0.5,
        skip_disclaimers=True,
        max_length_guidance="direct and brief",
        remove_prohibited_actions=["*"],
    ),
    "balanced": ControlPreset(
        name="balanced",
        temperature=0.3,
    ),
}


# ---------------------------------------------------------------------------
# Preset Registry
# ---------------------------------------------------------------------------


class PresetRegistry:
    """Container for built-in and user-defined presets."""

    def __init__(self, include_builtins: bool = True) -> None:
        self._presets: dict[str, ControlPreset] = {}
        if include_builtins:
            self._presets.update(BUILTIN_PRESETS)

    def register(self, preset: ControlPreset) -> None:
        self._presets[preset.name] = preset

    def get(self, name: str) -> ControlPreset | None:
        return self._presets.get(name)

    def names(self) -> list[str]:
        return sorted(self._presets.keys())


# ---------------------------------------------------------------------------
# Run Policy
# ---------------------------------------------------------------------------

_LIST_FIELDS = (
    "extra_must_include",
    "extra_never_include",
    "extra_prohibited_actions",
    "remove_prohibited_actions",
)

_SCALAR_FIELDS = (
    "temperature",
    "max_tokens",
    "tone_variant",
    "output_format",
    "max_length_guidance",
    "compact",
)


class RunPolicy(_StrictModel):
    """Per-run behavior overlay. Does not modify DomainConfig or Scaffold on disk."""

    temperature: float | None = None
    max_tokens: int | None = None
    tone_variant: str | None = None
    output_format: str | None = None
    max_length_guidance: str | None = None
    compact: bool | None = None
    scaffold_selection_bias: dict[str, float] = Field(default_factory=dict)
    skip_disclaimers: bool = False
    extra_must_include: list[str] = Field(default_factory=list)
    extra_never_include: list[str] = Field(default_factory=list)
    extra_prohibited_actions: list[str] = Field(default_factory=list)
    remove_prohibited_actions: list[str] = Field(default_factory=list)
    source: str = ""

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @classmethod
    def from_preset(cls, preset: ControlPreset) -> RunPolicy:
        """Create a RunPolicy from a named preset."""
        return cls(
            temperature=preset.temperature,
            max_tokens=preset.max_tokens,
            tone_variant=preset.tone_variant,
            output_format=preset.output_format,
            max_length_guidance=preset.max_length_guidance,
            compact=preset.compact,
            scaffold_selection_bias=dict(preset.scaffold_selection_bias),
            skip_disclaimers=preset.skip_disclaimers,
            extra_must_include=list(preset.extra_must_include),
            extra_never_include=list(preset.extra_never_include),
            extra_prohibited_actions=list(preset.extra_prohibited_actions),
            remove_prohibited_actions=list(preset.remove_prohibited_actions),
            source=f"preset:{preset.name}",
        )

    @classmethod
    def from_presets(cls, *presets: ControlPreset) -> RunPolicy:
        """Merge multiple presets. Last-writer-wins for scalars, union for lists."""
        if not presets:
            return cls()
        policy = cls.from_preset(presets[0])
        for preset in presets[1:]:
            policy = policy.merge(cls.from_preset(preset))
        return policy

    def merge(self, other: RunPolicy) -> RunPolicy:
        """Compose two policies. ``other`` wins for non-None scalars; lists concatenate."""
        kwargs: dict[str, Any] = {}

        for field_name in _SCALAR_FIELDS:
            other_val = getattr(other, field_name)
            kwargs[field_name] = other_val if other_val is not None else getattr(self, field_name)

        kwargs["skip_disclaimers"] = other.skip_disclaimers or self.skip_disclaimers

        # Merge scaffold selection bias — other wins per-key
        merged_bias = dict(self.scaffold_selection_bias)
        merged_bias.update(other.scaffold_selection_bias)
        kwargs["scaffold_selection_bias"] = merged_bias

        for field_name in _LIST_FIELDS:
            self_list = getattr(self, field_name)
            other_list = getattr(other, field_name)
            seen: set[str] = set()
            deduped: list[str] = []
            for item in [*self_list, *other_list]:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            kwargs[field_name] = deduped

        sources = [s for s in (self.source, other.source) if s]
        kwargs["source"] = "+".join(sources) if sources else ""

        return RunPolicy(**kwargs)


# ---------------------------------------------------------------------------
# Constraint Parser
# ---------------------------------------------------------------------------


@dataclass
class ParsedConstraint:
    """A single parsed clause."""

    raw: str
    field: str
    value: Any
    matched_rule: str


@dataclass
class ParseResult:
    """Full parse output."""

    policy: RunPolicy
    parsed: list[ParsedConstraint] = field(default_factory=list)
    unrecognized: list[str] = field(default_factory=list)


@dataclass
class _Rule:
    pattern: re.Pattern[str]
    field: str
    extractor: Callable[[re.Match[str]], Any]
    description: str


def _const(value: Any) -> Callable[[re.Match[str]], Any]:
    """Return an extractor that always produces a constant."""
    return lambda _m: value


def _group_float(group: int = 1) -> Callable[[re.Match[str]], float]:
    return lambda m: float(m.group(group))


def _group_int(group: int = 1) -> Callable[[re.Match[str]], int]:
    return lambda m: int(m.group(group))


def _group_str(group: int = 1) -> Callable[[re.Match[str]], str]:
    return lambda m: m.group(group).strip()


def _group_list(group: int = 1) -> Callable[[re.Match[str]], list[str]]:
    return lambda m: [m.group(group).strip()]


def _length_extractor(m: re.Match[str]) -> str:
    return f"under {m.group(1)} words"


def _rule(
    pattern: str, fld: str, ext: Callable[[re.Match[str]], Any], desc: str,
) -> _Rule:
    return _Rule(re.compile(pattern), fld, ext, desc)


_RULES: list[_Rule] = [
    # Temperature presets
    _rule(r"\bmore\s+creative\b", "temperature", _const(0.8), "creative_temp"),
    _rule(r"\bmore\s+precise\b", "temperature", _const(0.1), "precise_temp"),
    _rule(r"\bmore\s+aggressive\b", "temperature", _const(0.5), "aggressive_temp"),
    _rule(r"\btemperature\s+(\d+\.?\d*)\b", "temperature", _group_float(), "explicit_temp"),
    # Output format
    _rule(r"\bbullet\s*points?\b", "output_format", _const("bullet_points"), "bullet_format"),
    _rule(
        r"\bstructured\s+narrative\b", "output_format",
        _const("structured_narrative"), "narrative_format",
    ),
    # Length
    _rule(
        r"\bunder\s+(\d+)\s+words?\b", "max_length_guidance",
        _length_extractor, "under_n_words",
    ),
    _rule(
        r"\b(?:keep\s+it\s+brief|be\s+brief|be\s+concise)\b", "max_length_guidance",
        _const("concise, under 200 words"), "brief",
    ),
    _rule(
        r"\bno\s+length\s+(?:limit|constraint)\b", "max_length_guidance",
        _const("no length constraint"), "no_length_limit",
    ),
    # Disclaimers
    _rule(
        r"\b(?:skip|no|drop)\s+disclaimers?\b", "skip_disclaimers",
        _const(True), "skip_disclaimers",
    ),
    # Prohibited actions
    _rule(
        r"\b(?:skip|no|drop)\s+prohibited\s+actions?\b", "remove_prohibited_actions",
        _const(["*"]), "skip_prohibited",
    ),
    # Must / never include
    _rule(r"\bmust\s+include\s+(.+)", "extra_must_include", _group_list(), "must_include"),
    _rule(r"\bnever\s+include\s+(.+)", "extra_never_include", _group_list(), "never_include"),
    # Compact
    _rule(r"\b(?:compact\s+mode|use\s+compact)\b", "compact", _const(True), "compact_mode"),
    # Tone
    _rule(r"\btone[:\s]+(\w+)", "tone_variant", _group_str(), "tone_variant"),
    # Max tokens
    _rule(r"\bmax\s+(\d+)\s+tokens?\b", "max_tokens", _group_int(), "max_tokens"),
    # Preset reference (handled specially)
    _rule(r"\bpreset[:\s]+(\w+)", "_preset", _group_str(), "preset_ref"),
]

_CLAUSE_SPLIT = re.compile(r"[,;]\s*")


class ConstraintParser:
    """Parse plain-English run rules into RunPolicy overrides."""

    @staticmethod
    def parse(
        text: str,
        preset_registry: PresetRegistry | None = None,
    ) -> ParseResult:
        if not text or not text.strip():
            return ParseResult(policy=RunPolicy())

        clauses = _CLAUSE_SPLIT.split(text.strip())
        parsed: list[ParsedConstraint] = []
        unrecognized: list[str] = []
        overrides: dict[str, Any] = {}

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            lower = clause.lower()
            matched = False

            for rule in _RULES:
                m = rule.pattern.search(lower)
                if m:
                    value = rule.extractor(m)

                    if rule.field == "_preset":
                        # Resolve preset reference
                        if preset_registry:
                            preset = preset_registry.get(value)
                            if preset:
                                preset_policy = RunPolicy.from_preset(preset)
                                # Merge preset fields into overrides
                                for f in _SCALAR_FIELDS:
                                    pv = getattr(preset_policy, f)
                                    if pv is not None:
                                        overrides[f] = pv
                                if preset_policy.skip_disclaimers:
                                    overrides["skip_disclaimers"] = True
                                for f in _LIST_FIELDS:
                                    existing = overrides.get(f, [])
                                    overrides[f] = existing + getattr(preset_policy, f)
                                parsed.append(ParsedConstraint(
                                    raw=clause, field="preset",
                                    value=value, matched_rule=rule.description,
                                ))
                                matched = True
                                break
                        # No registry or preset not found — unrecognized
                        unrecognized.append(clause)
                        matched = True
                        break

                    # List fields accumulate
                    if rule.field in _LIST_FIELDS:
                        existing = overrides.get(rule.field, [])
                        if isinstance(value, list):
                            overrides[rule.field] = existing + value
                        else:
                            overrides[rule.field] = existing + [value]
                    else:
                        overrides[rule.field] = value

                    parsed.append(ParsedConstraint(
                        raw=clause, field=rule.field, value=value, matched_rule=rule.description,
                    ))
                    matched = True
                    break

            if not matched:
                unrecognized.append(clause)

        source_parts = [p.matched_rule for p in parsed]
        overrides["source"] = "constraint:" + "+".join(source_parts) if source_parts else ""

        policy = RunPolicy(**overrides)
        return ParseResult(policy=policy, parsed=parsed, unrecognized=unrecognized)
