"""Data models for cognitive scaffolds.

A scaffold is a structured reasoning framework that shapes how the inner LLM
approaches a specific type of analysis. These dataclasses define the schema
for scaffold definitions loaded from YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScaffoldApplicability:
    """Defines when a scaffold should be selected.

    Tools map scaffold to specific MCP tool invocations.
    Keywords and intent signals drive scored matching against user input.
    """

    tools: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    intent_signals: list[str] = field(default_factory=list)


@dataclass
class ScaffoldFraming:
    """The cognitive framing the inner LLM adopts.

    Role and perspective define *who* the LLM pretends to be.
    Tone controls communication style, with optional named variants.
    """

    role: str = ""
    perspective: str = ""
    tone: str = ""
    tone_variants: dict[str, str] = field(default_factory=dict)


@dataclass
class ScaffoldOutputCalibration:
    """Controls the shape and content of LLM output.

    format: the default output format (e.g. "structured_narrative")
    format_options: all valid format choices for this scaffold
    must_include / never_include: hard constraints on output content
    """

    format: str = "structured_narrative"
    format_options: list[str] = field(default_factory=lambda: ["structured_narrative"])
    max_length_guidance: str = ""
    must_include: list[str] = field(default_factory=list)
    never_include: list[str] = field(default_factory=list)


@dataclass
class ScaffoldGuardrails:
    """Safety boundaries for the inner LLM.

    disclaimers: statements that must appear in output
    escalation_triggers: conditions that warrant recommending professional help
    prohibited_actions: things the LLM must never do
    """

    disclaimers: list[str] = field(default_factory=list)
    escalation_triggers: list[str] = field(default_factory=list)
    prohibited_actions: list[str] = field(default_factory=list)


@dataclass
class ContextField:
    """A single cross-domain context field definition.

    Used in context_accepts and context_exports to declare what data
    a scaffold can receive from or provide to other domains.
    """

    field_name: str
    type: str
    description: str


@dataclass
class Scaffold:
    """A complete cognitive scaffold -- a reasoning framework for the inner LLM.

    This is the central data structure of the scaffold system. Each scaffold
    encodes a domain-specific reasoning pattern: who the LLM should be,
    how it should think, what it should include, and what it must avoid.
    """

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
    context_accepts: list[ContextField] = field(default_factory=list)
    context_exports: list[ContextField] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class AssembledPrompt:
    """The final prompt sent to the inner LLM after scaffold application.

    system_message: the full system prompt with role, reasoning, guardrails
    user_message: the user query enriched with data context
    metadata: scaffold provenance information for logging and debugging
    """

    system_message: str
    user_message: str
    metadata: dict[str, Any] = field(default_factory=dict)
