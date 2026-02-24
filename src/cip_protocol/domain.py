from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainConfig:
    """The boundary between protocol and domain.

    The protocol never imports domain-specific code — it reads
    everything it needs from this dataclass.
    """

    name: str
    display_name: str
    system_prompt: str
    default_scaffold_id: str | None = None
    data_context_label: str = "Data Context"
    prohibited_indicators: dict[str, tuple[str, ...]] = field(default_factory=dict)
    regex_guardrail_policies: dict[str, str] = field(default_factory=dict)
    redaction_message: str = "[Removed: contains prohibited content]"
