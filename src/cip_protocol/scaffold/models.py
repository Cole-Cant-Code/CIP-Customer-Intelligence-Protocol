"""Pydantic models for cognitive scaffolds and assembled prompts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    """Shared strict model settings for scaffold contracts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _normalize_string_list(values: list[str]) -> list[str]:
    """Trim whitespace and drop empty entries while preserving order."""
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized


class ScaffoldApplicability(_StrictModel):
    """Defines when a scaffold should be selected."""

    tools: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    intent_signals: list[str] = Field(default_factory=list)

    @field_validator("tools", "keywords", "intent_signals")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        return _normalize_string_list(values)


class ScaffoldFraming(_StrictModel):
    """The cognitive framing the inner LLM adopts."""

    role: str = ""
    perspective: str = ""
    tone: str = ""
    tone_variants: dict[str, str] = Field(default_factory=dict)

    @field_validator("role", "perspective", "tone")
    @classmethod
    def normalize_text_fields(cls, value: str) -> str:
        return value.strip()

    @field_validator("tone_variants")
    @classmethod
    def normalize_tone_variants(cls, variants: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, value in variants.items():
            key_norm = key.strip()
            value_norm = value.strip()
            if key_norm and value_norm:
                cleaned[key_norm] = value_norm
        return cleaned


class ScaffoldOutputCalibration(_StrictModel):
    """Controls the shape and content of LLM output."""

    format: str = "structured_narrative"
    format_options: list[str] = Field(default_factory=lambda: ["structured_narrative"])
    max_length_guidance: str = ""
    must_include: list[str] = Field(default_factory=list)
    never_include: list[str] = Field(default_factory=list)

    @field_validator("format", "max_length_guidance")
    @classmethod
    def normalize_text_fields(cls, value: str) -> str:
        return value.strip()

    @field_validator("format_options", "must_include", "never_include")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        return _normalize_string_list(values)

    @model_validator(mode="after")
    def ensure_valid_format_options(self) -> ScaffoldOutputCalibration:
        if not self.format:
            self.format = "structured_narrative"
        if not self.format_options:
            self.format_options = [self.format]
        if self.format not in self.format_options:
            self.format_options = [self.format, *self.format_options]
        return self


class ScaffoldGuardrails(_StrictModel):
    """Safety boundaries for the inner LLM."""

    disclaimers: list[str] = Field(default_factory=list)
    escalation_triggers: list[str] = Field(default_factory=list)
    prohibited_actions: list[str] = Field(default_factory=list)

    @field_validator("disclaimers", "escalation_triggers", "prohibited_actions")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        return _normalize_string_list(values)


class ContextField(_StrictModel):
    """A single cross-domain context field definition."""

    field_name: str
    type: str
    description: str = ""

    @field_validator("field_name", "type", "description")
    @classmethod
    def normalize_text_fields(cls, value: str) -> str:
        return value.strip()


class Scaffold(_StrictModel):
    """A complete cognitive scaffold reasoning framework."""

    id: str
    version: str
    domain: str
    display_name: str
    description: str
    applicability: ScaffoldApplicability
    framing: ScaffoldFraming
    reasoning_framework: dict[str, Any]
    domain_knowledge_activation: list[str]
    output_calibration: ScaffoldOutputCalibration
    guardrails: ScaffoldGuardrails
    context_accepts: list[ContextField] = Field(default_factory=list)
    context_exports: list[ContextField] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("id", "version", "domain", "display_name", "description")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("domain_knowledge_activation", "tags")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        return _normalize_string_list(values)

    @field_validator("reasoning_framework")
    @classmethod
    def validate_reasoning_framework(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("reasoning_framework must be a mapping")
        steps = value.get("steps")
        if steps is None:
            return value
        if not isinstance(steps, list):
            raise ValueError("reasoning_framework.steps must be a list")
        normalized_steps = _normalize_string_list(steps)
        value = dict(value)
        value["steps"] = normalized_steps
        return value


class ChatMessage(_StrictModel):
    """Single turn in conversation history."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        return value.strip()


class AssembledPrompt(_StrictModel):
    """The final prompt payload sent to the inner LLM."""

    system_message: str
    user_message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chat_history: list[ChatMessage] = Field(default_factory=list)

    @field_validator("system_message", "user_message")
    @classmethod
    def normalize_message_fields(cls, value: str) -> str:
        return value.strip()
