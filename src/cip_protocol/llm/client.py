"""Inner LLM client -- the bridge between scaffolds and LLM calls.

The InnerLLMClient owns the full invoke() pipeline:

    build system prompt (domain identity + scaffold instructions)
    -> provider.generate
    -> check_guardrails (with domain-specific prohibited indicators)
    -> sanitize_content (with domain-specific redaction message)
    -> enforce_disclaimers
    -> append provenance
    -> extract_context_exports
    -> return LLMResponse

The client is domain-agnostic: all domain-specific behavior comes from
the DomainConfig passed at construction time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from cip_protocol.domain import DomainConfig
from cip_protocol.llm.provider import LLMProvider, ProviderResponse
from cip_protocol.llm.response import (
    check_guardrails,
    enforce_disclaimers,
    extract_context_exports,
    sanitize_content,
)
from cip_protocol.scaffold.models import AssembledPrompt, Scaffold

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from the inner specialist LLM.

    Carries the final (sanitised, disclaimer-enforced) content alongside
    provenance metadata so callers can trace which scaffold produced the
    output and what guardrail actions were taken.
    """

    content: str
    scaffold_id: str
    scaffold_version: str
    guardrail_flags: list[str] = field(default_factory=list)
    context_exports: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


class InnerLLMClient:
    """Invokes the inner specialist LLM with scaffold-assembled prompts.

    This is the single entry-point for all LLM calls in a CIP server.
    The outer scaffold layer assembles a prompt; this client sends it to
    the provider and applies every safety/post-processing step before
    returning the result.

    Args:
        provider: The LLM provider to call (Anthropic, OpenAI, mock).
        config: Domain configuration.  Provides the system prompt,
            prohibited indicators, and redaction message.  If None,
            no domain system prompt is prepended and no prohibited
            pattern checking is performed.
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: DomainConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config

    def _build_full_system_prompt(self, scaffold_system_message: str) -> str:
        """Combine domain system prompt with scaffold-specific instructions.

        If no config is provided, the scaffold system message is used as-is.
        """
        if not self.config or not self.config.system_prompt:
            return scaffold_system_message

        return f"""{self.config.system_prompt}

---

{scaffold_system_message}"""

    async def invoke(
        self,
        assembled_prompt: AssembledPrompt,
        scaffold: Scaffold,
        data_context: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Call the inner specialist LLM with an assembled scaffold prompt.

        Pipeline:
            1. Prepend domain system prompt to scaffold system message.
            2. Send to the provider.
            3. Run guardrail checks and sanitise prohibited content.
            4. Enforce scaffold-required disclaimers.
            5. Append deterministic provenance (data source footer).
            6. Extract context-export fields for cross-domain sharing.
        """
        full_system = self._build_full_system_prompt(
            assembled_prompt.system_message
        )

        provider_response: ProviderResponse = await self.provider.generate(
            system_message=full_system,
            user_message=assembled_prompt.user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        logger.info(
            "Inner LLM call: scaffold=%s, model=%s, tokens=%d+%d, latency=%.0fms",
            scaffold.id,
            provider_response.model,
            provider_response.input_tokens,
            provider_response.output_tokens,
            provider_response.latency_ms,
        )

        # --- guardrail enforcement ---
        prohibited = (
            self.config.prohibited_indicators if self.config else None
        )
        guardrail_check = check_guardrails(
            provider_response.content, scaffold, prohibited_indicators=prohibited
        )

        redaction_msg = (
            self.config.redaction_message
            if self.config
            else "[Removed: contains prohibited content]"
        )
        content = sanitize_content(
            provider_response.content, guardrail_check,
            redaction_message=redaction_msg,
        )
        if not guardrail_check.passed:
            logger.warning(
                "Guardrails enforced on scaffold %s: %d prohibited patterns redacted",
                scaffold.id,
                len([f for f in guardrail_check.flags if "prohibited" in f]),
            )

        # --- disclaimer enforcement ---
        content, disclaimer_flags = enforce_disclaimers(content, scaffold)

        # --- deterministic provenance footer ---
        if data_context:
            source = data_context.get("data_source")
            note = data_context.get("data_source_note")
            if source and "Data source:" not in content:
                footer_lines = ["", "", "---", f"Data source: {source}"]
                if note:
                    footer_lines.append(f"Note: {note}")
                content += "\n".join(footer_lines)

        # --- context export extraction ---
        context_exports = extract_context_exports(
            content=content,
            scaffold=scaffold,
            data_context=data_context or {},
        )

        return LLMResponse(
            content=content,
            scaffold_id=scaffold.id,
            scaffold_version=scaffold.version,
            guardrail_flags=guardrail_check.flags + disclaimer_flags,
            context_exports=context_exports,
            usage={
                "input_tokens": provider_response.input_tokens,
                "output_tokens": provider_response.output_tokens,
            },
        )
