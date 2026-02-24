"""Domain configuration — the contract between the protocol and a domain.

A DomainConfig is the single object a domain provides to configure the
protocol's behavior.  It contains:

- Identity: name, display name, system prompt for the inner LLM
- Scaffold defaults: which scaffold to fall back to, label for data sections
- Guardrails: prohibited patterns and redaction message

The protocol components (ScaffoldEngine, InnerLLMClient, renderer) read
from this config instead of hardcoding domain-specific values.

Example usage (finance domain)::

    config = DomainConfig(
        name="personal_finance",
        display_name="CIP Personal Finance",
        system_prompt="You are an expert in consumer finance...",
        default_scaffold_id="spending_review",
        data_context_label="Financial Data",
        prohibited_indicators={
            "recommending products": ("i recommend", "sign up for"),
            "making predictions": ("the market will", "guaranteed to"),
        },
        redaction_message="[Removed: contains prohibited financial guidance]",
    )

Example usage (health domain)::

    config = DomainConfig(
        name="health_wellness",
        display_name="CIP Health & Wellness",
        system_prompt="You are a health information specialist...",
        default_scaffold_id="symptom_overview",
        data_context_label="Health Data",
        prohibited_indicators={
            "diagnosing conditions": ("you have", "this is definitely"),
            "prescribing treatment": ("take this medication", "you should stop taking"),
        },
        redaction_message="[Removed: contains prohibited medical guidance]",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainConfig:
    """Configuration that a CIP domain provides to the protocol.

    This is the boundary between protocol and domain.  The protocol
    never imports domain-specific code — it reads everything it needs
    from this dataclass.

    Attributes:
        name: Machine-readable domain identifier (e.g. "personal_finance").
        display_name: Human-readable name (e.g. "CIP Personal Finance").
        system_prompt: Base identity prompt prepended to every scaffold
            system message.  Defines who the inner LLM is and what it
            must never do.
        default_scaffold_id: Scaffold ID to fall back to when no match
            is found.  None means raise ScaffoldNotFoundError instead.
        data_context_label: Heading used in the user message for the
            data payload (e.g. "Financial Data", "Health Records").
        prohibited_indicators: Map of prohibition category to tuple of
            phrases that indicate the LLM violated that prohibition.
            Used by the guardrail checker.
        regex_guardrail_policies: Optional map of policy name to regex
            pattern for deterministic hard-policy checks.
        redaction_message: Replacement text when a prohibited pattern
            is redacted from LLM output.
    """

    name: str
    display_name: str
    system_prompt: str
    default_scaffold_id: str | None = None
    data_context_label: str = "Data Context"
    prohibited_indicators: dict[str, tuple[str, ...]] = field(default_factory=dict)
    regex_guardrail_policies: dict[str, str] = field(default_factory=dict)
    redaction_message: str = "[Removed: contains prohibited content]"
