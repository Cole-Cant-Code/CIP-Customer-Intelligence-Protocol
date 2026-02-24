"""Inner LLM client bridging scaffold prompts, providers, guardrails, and telemetry."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from cip_protocol.domain import DomainConfig
from cip_protocol.llm.provider import HistoryMessage, LLMProvider, ProviderResponse
from cip_protocol.llm.response import (
    GuardrailEvaluator,
    check_guardrails,
    default_guardrail_evaluators,
    enforce_disclaimers,
    extract_context_exports,
    sanitize_content,
)
from cip_protocol.scaffold.models import AssembledPrompt, ChatMessage, Scaffold
from cip_protocol.telemetry import NoOpTelemetrySink, TelemetryEvent, TelemetrySink

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from the inner specialist LLM."""

    content: str
    scaffold_id: str
    scaffold_version: str
    guardrail_flags: list[str] = field(default_factory=list)
    context_exports: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """Streaming event emitted by invoke_stream."""

    event: str
    text: str = ""
    response: LLMResponse | None = None


class InnerLLMClient:
    """Invokes the inner specialist LLM with scaffold-assembled prompts."""

    def __init__(
        self,
        provider: LLMProvider,
        config: DomainConfig | None = None,
        guardrail_evaluators: list[GuardrailEvaluator] | None = None,
        telemetry_sink: TelemetrySink | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.guardrail_evaluators = guardrail_evaluators
        self.telemetry_sink = telemetry_sink or NoOpTelemetrySink()

    def _build_full_system_prompt(self, scaffold_system_message: str) -> str:
        """Combine domain system prompt with scaffold-specific instructions."""
        if not self.config or not self.config.system_prompt:
            return scaffold_system_message

        return f"""{self.config.system_prompt}

---

{scaffold_system_message}"""

    @staticmethod
    def _normalize_chat_history(
        chat_history: list[ChatMessage | HistoryMessage] | None,
    ) -> list[HistoryMessage]:
        if not chat_history:
            return []

        normalized: list[HistoryMessage] = []
        for item in chat_history:
            if isinstance(item, ChatMessage):
                role = item.role
                content = item.content
            else:
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", ""))
            if role and content:
                normalized.append({"role": role, "content": content})
        return normalized

    def _resolve_chat_history(
        self,
        assembled_prompt: AssembledPrompt,
        chat_history: list[ChatMessage | HistoryMessage] | None,
    ) -> list[HistoryMessage]:
        source = chat_history if chat_history is not None else assembled_prompt.chat_history
        return self._normalize_chat_history(source)

    def _resolve_evaluators(self) -> list[GuardrailEvaluator]:
        if self.guardrail_evaluators is not None:
            return self.guardrail_evaluators
        prohibited = self.config.prohibited_indicators if self.config else None
        regex_policies = self.config.regex_guardrail_policies if self.config else None
        return default_guardrail_evaluators(prohibited, regex_policies)

    def _emit(self, name: str, **attributes: Any) -> None:
        self.telemetry_sink.emit(TelemetryEvent(name=name, attributes=attributes))

    @staticmethod
    def _append_provenance(content: str, data_context: dict[str, Any] | None) -> str:
        if not data_context:
            return content

        source = data_context.get("data_source")
        note = data_context.get("data_source_note")
        if source and "Data source:" not in content:
            footer_lines = ["", "", "---", f"Data source: {source}"]
            if note:
                footer_lines.append(f"Note: {note}")
            return content + "\n".join(footer_lines)

        return content

    @staticmethod
    def _estimate_input_tokens(
        system_message: str,
        user_message: str,
        chat_history: list[HistoryMessage],
    ) -> int:
        text_parts = [system_message, user_message]
        for item in chat_history:
            text_parts.append(item.get("content", ""))
        return len(" ".join(text_parts).split())

    async def invoke(
        self,
        assembled_prompt: AssembledPrompt,
        scaffold: Scaffold,
        data_context: dict[str, Any] | None = None,
        chat_history: list[ChatMessage | HistoryMessage] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Call provider, apply guardrails, and return post-processed response."""
        full_system = self._build_full_system_prompt(assembled_prompt.system_message)
        normalized_history = self._resolve_chat_history(assembled_prompt, chat_history)
        evaluators = self._resolve_evaluators()

        self._emit(
            "llm.invoke.start",
            scaffold_id=scaffold.id,
            scaffold_version=scaffold.version,
            history_turns=len(normalized_history),
            max_tokens=max_tokens,
            temperature=temperature,
            evaluator_count=len(evaluators),
        )

        provider_response: ProviderResponse = await self.provider.generate(
            system_message=full_system,
            user_message=assembled_prompt.user_message,
            chat_history=normalized_history,
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

        guardrail_check = check_guardrails(
            provider_response.content,
            scaffold,
            evaluators=evaluators,
        )

        redaction_msg = (
            self.config.redaction_message
            if self.config
            else "[Removed: contains prohibited content]"
        )
        content = sanitize_content(
            provider_response.content,
            guardrail_check,
            redaction_message=redaction_msg,
        )

        if not guardrail_check.passed:
            self._emit(
                "llm.guardrail.intervention",
                scaffold_id=scaffold.id,
                hard_violations=guardrail_check.hard_violations,
                matched_phrases=guardrail_check.matched_phrases,
            )
            logger.warning(
                "Guardrails enforced on scaffold %s: %d hard violations",
                scaffold.id,
                len(guardrail_check.hard_violations),
            )

        content, disclaimer_flags = enforce_disclaimers(content, scaffold)
        content = self._append_provenance(content, data_context)

        context_exports = extract_context_exports(
            content=content,
            scaffold=scaffold,
            data_context=data_context or {},
        )

        response = LLMResponse(
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

        self._emit(
            "llm.invoke.complete",
            scaffold_id=scaffold.id,
            model=provider_response.model,
            input_tokens=provider_response.input_tokens,
            output_tokens=provider_response.output_tokens,
            latency_ms=provider_response.latency_ms,
            guardrail_flag_count=len(response.guardrail_flags),
        )

        return response

    async def invoke_stream(
        self,
        assembled_prompt: AssembledPrompt,
        scaffold: Scaffold,
        data_context: dict[str, Any] | None = None,
        chat_history: list[ChatMessage | HistoryMessage] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        """Stream provider output with incremental guardrail checks.

        Yields:
            StreamEvent(event="chunk", text=...)
            StreamEvent(event="halted", text=..., response=...) when halted early
            StreamEvent(event="final", text=..., response=...) on normal completion
        """
        full_system = self._build_full_system_prompt(assembled_prompt.system_message)
        normalized_history = self._resolve_chat_history(assembled_prompt, chat_history)
        evaluators = self._resolve_evaluators()

        started = time.monotonic()
        self._emit(
            "llm.stream.start",
            scaffold_id=scaffold.id,
            history_turns=len(normalized_history),
            evaluator_count=len(evaluators),
        )

        collected: list[str] = []
        async for chunk in self.provider.generate_stream(
            system_message=full_system,
            user_message=assembled_prompt.user_message,
            chat_history=normalized_history,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if not chunk:
                continue

            collected.append(chunk)
            current = "".join(collected)
            guardrail_check = check_guardrails(current, scaffold, evaluators=evaluators)
            if not guardrail_check.passed:
                redaction_msg = (
                    self.config.redaction_message
                    if self.config
                    else "[Removed: contains prohibited content]"
                )
                content = sanitize_content(
                    current,
                    guardrail_check,
                    redaction_message=redaction_msg,
                )
                content, disclaimer_flags = enforce_disclaimers(content, scaffold)
                content = self._append_provenance(content, data_context)
                context_exports = extract_context_exports(
                    content=content,
                    scaffold=scaffold,
                    data_context=data_context or {},
                )

                response = LLMResponse(
                    content=content,
                    scaffold_id=scaffold.id,
                    scaffold_version=scaffold.version,
                    guardrail_flags=guardrail_check.flags + disclaimer_flags,
                    context_exports=context_exports,
                    usage={
                        "input_tokens": self._estimate_input_tokens(
                            full_system,
                            assembled_prompt.user_message,
                            normalized_history,
                        ),
                        "output_tokens": len(content.split()),
                    },
                )

                self._emit(
                    "llm.stream.halted",
                    scaffold_id=scaffold.id,
                    hard_violations=guardrail_check.hard_violations,
                    elapsed_ms=(time.monotonic() - started) * 1000,
                )
                yield StreamEvent(event="halted", text=content, response=response)
                return

            yield StreamEvent(event="chunk", text=chunk)

        final_raw = "".join(collected).strip()
        guardrail_check = check_guardrails(final_raw, scaffold, evaluators=evaluators)
        redaction_msg = (
            self.config.redaction_message
            if self.config
            else "[Removed: contains prohibited content]"
        )
        content = sanitize_content(final_raw, guardrail_check, redaction_message=redaction_msg)
        content, disclaimer_flags = enforce_disclaimers(content, scaffold)
        content = self._append_provenance(content, data_context)
        context_exports = extract_context_exports(
            content=content,
            scaffold=scaffold,
            data_context=data_context or {},
        )

        response = LLMResponse(
            content=content,
            scaffold_id=scaffold.id,
            scaffold_version=scaffold.version,
            guardrail_flags=guardrail_check.flags + disclaimer_flags,
            context_exports=context_exports,
            usage={
                "input_tokens": self._estimate_input_tokens(
                    full_system,
                    assembled_prompt.user_message,
                    normalized_history,
                ),
                "output_tokens": len(content.split()),
            },
        )

        self._emit(
            "llm.stream.complete",
            scaffold_id=scaffold.id,
            output_tokens=response.usage["output_tokens"],
            elapsed_ms=(time.monotonic() - started) * 1000,
            guardrail_flag_count=len(response.guardrail_flags),
        )

        yield StreamEvent(event="final", text=response.content, response=response)
