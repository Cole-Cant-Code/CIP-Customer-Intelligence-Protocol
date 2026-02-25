from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cip_protocol.domain import DomainConfig
from cip_protocol.llm.provider import HistoryMessage, LLMProvider, ProviderResponse
from cip_protocol.llm.response import (
    GuardrailCheck,
    GuardrailEvaluator,
    check_guardrails,
    check_guardrails_async,
    default_guardrail_evaluators,
    enforce_disclaimers,
    extract_context_exports,
    sanitize_content,
)
from cip_protocol.scaffold.models import AssembledPrompt, ChatMessage, Scaffold
from cip_protocol.telemetry import NoOpTelemetrySink, TelemetryEvent, TelemetrySink

if TYPE_CHECKING:
    from cip_protocol.control import RunPolicy

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    scaffold_id: str
    scaffold_version: str
    guardrail_flags: list[str] = field(default_factory=list)
    context_exports: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class StreamEvent:
    event: str
    text: str = ""
    response: LLMResponse | None = None


class InnerLLMClient:
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
        self.telemetry = telemetry_sink or NoOpTelemetrySink()
        if guardrail_evaluators is not None:
            self._resolved_evaluators = guardrail_evaluators
        else:
            prohibited = self.config.prohibited_indicators if self.config else None
            regex = self.config.regex_guardrail_policies if self.config else None
            self._resolved_evaluators = default_guardrail_evaluators(prohibited, regex)

    def _build_system_prompt(self, scaffold_system_message: str) -> str:
        if not self.config or not self.config.system_prompt:
            return scaffold_system_message
        return f"{self.config.system_prompt}\n\n---\n\n{scaffold_system_message}"

    @staticmethod
    def _normalize_history(
        chat_history: list[ChatMessage | HistoryMessage] | None,
    ) -> list[HistoryMessage]:
        if not chat_history:
            return []
        result: list[HistoryMessage] = []
        for item in chat_history:
            if isinstance(item, ChatMessage):
                role, content = item.role, item.content
            else:
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", ""))
            if role and content:
                result.append({"role": role, "content": content})
        return result

    def _resolve_history(
        self,
        prompt: AssembledPrompt,
        chat_history: list[ChatMessage | HistoryMessage] | None,
    ) -> list[HistoryMessage]:
        source = chat_history if chat_history is not None else prompt.chat_history
        return self._normalize_history(source)

    def _resolve_evaluators(self) -> list[GuardrailEvaluator]:
        return self._resolved_evaluators

    @property
    def _redaction_message(self) -> str:
        if self.config:
            return self.config.redaction_message
        return "[Removed: contains prohibited content]"

    def _emit(self, name: str, **attrs: Any) -> None:
        self.telemetry.emit(TelemetryEvent(name=name, attributes=attrs))

    def _finalize_postprocess(
        self,
        raw_content: str,
        scaffold: Scaffold,
        guardrail_check: GuardrailCheck,
        data_context: dict[str, Any] | None,
        skip_disclaimers: bool = False,
    ) -> tuple[str, list[str], dict[str, Any], bool]:
        content = sanitize_content(
            raw_content, guardrail_check, redaction_message=self._redaction_message
        )

        if not guardrail_check.passed:
            self._emit(
                "llm.guardrail.intervention",
                scaffold_id=scaffold.id,
                hard_violations=guardrail_check.hard_violations,
                matched_phrases=guardrail_check.matched_phrases,
            )

        if skip_disclaimers:
            disclaimer_flags: list[str] = []
        else:
            content, disclaimer_flags = enforce_disclaimers(content, scaffold)
        content = self._append_provenance(content, data_context)

        context_exports = extract_context_exports(
            content=content,
            scaffold=scaffold,
            data_context=data_context or {},
        )

        flags = guardrail_check.flags + disclaimer_flags
        return content, flags, context_exports, guardrail_check.passed

    def _postprocess(
        self,
        raw_content: str,
        scaffold: Scaffold,
        evaluators: list[GuardrailEvaluator],
        data_context: dict[str, Any] | None,
        skip_disclaimers: bool = False,
    ) -> tuple[str, list[str], dict[str, Any], bool]:
        """Sync postprocess: guardrails, sanitize, disclaimers, provenance."""
        guardrail_check = check_guardrails(raw_content, scaffold, evaluators=evaluators)
        return self._finalize_postprocess(
            raw_content, scaffold, guardrail_check, data_context,
            skip_disclaimers=skip_disclaimers,
        )

    async def _postprocess_async(
        self,
        raw_content: str,
        scaffold: Scaffold,
        evaluators: list[GuardrailEvaluator],
        data_context: dict[str, Any] | None,
        skip_disclaimers: bool = False,
    ) -> tuple[str, list[str], dict[str, Any], bool]:
        """Async postprocess: runs guardrail evaluators concurrently."""
        guardrail_check = await check_guardrails_async(
            raw_content, scaffold, evaluators=evaluators
        )
        return self._finalize_postprocess(
            raw_content, scaffold, guardrail_check, data_context,
            skip_disclaimers=skip_disclaimers,
        )

    @staticmethod
    def _append_provenance(content: str, data_context: dict[str, Any] | None) -> str:
        if not data_context:
            return content
        source = data_context.get("data_source")
        if not source or "Data source:" in content:
            return content

        footer = ["", "", "---", f"Data source: {source}"]
        note = data_context.get("data_source_note")
        if note:
            footer.append(f"Note: {note}")
        return content + "\n".join(footer)

    def _build_response(
        self,
        content: str,
        scaffold: Scaffold,
        flags: list[str],
        context_exports: dict[str, Any],
        usage: dict[str, int],
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            scaffold_id=scaffold.id,
            scaffold_version=scaffold.version,
            guardrail_flags=flags,
            context_exports=context_exports,
            usage=usage,
        )

    async def invoke(
        self,
        assembled_prompt: AssembledPrompt,
        scaffold: Scaffold,
        data_context: dict[str, Any] | None = None,
        chat_history: list[ChatMessage | HistoryMessage] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        policy: RunPolicy | None = None,
    ) -> LLMResponse:
        if policy:
            if policy.temperature is not None:
                temperature = policy.temperature
            if policy.max_tokens is not None:
                max_tokens = policy.max_tokens

        skip_disclaimers = policy.skip_disclaimers if policy else False

        full_system = self._build_system_prompt(assembled_prompt.system_message)
        history = self._resolve_history(assembled_prompt, chat_history)
        evaluators = self._resolve_evaluators()

        emit_attrs: dict[str, Any] = {
            "scaffold_id": scaffold.id,
            "scaffold_version": scaffold.version,
            "history_turns": len(history),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "evaluator_count": len(evaluators),
        }
        if policy and policy.source:
            emit_attrs["policy_source"] = policy.source
        self._emit("llm.invoke.start", **emit_attrs)

        resp: ProviderResponse = await self.provider.generate(
            system_message=full_system,
            user_message=assembled_prompt.user_message,
            chat_history=history,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content, flags, exports, _ = await self._postprocess_async(
            resp.content, scaffold, evaluators, data_context,
            skip_disclaimers=skip_disclaimers,
        )

        self._emit(
            "llm.invoke.complete",
            scaffold_id=scaffold.id,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_ms=resp.latency_ms,
            guardrail_flag_count=len(flags),
        )

        return self._build_response(
            content, scaffold, flags, exports,
            {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
        )

    async def invoke_stream(
        self,
        assembled_prompt: AssembledPrompt,
        scaffold: Scaffold,
        data_context: dict[str, Any] | None = None,
        chat_history: list[ChatMessage | HistoryMessage] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        policy: RunPolicy | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield StreamEvents: chunk → (halted | final)."""
        if policy:
            if policy.temperature is not None:
                temperature = policy.temperature
            if policy.max_tokens is not None:
                max_tokens = policy.max_tokens

        skip_disclaimers = policy.skip_disclaimers if policy else False

        full_system = self._build_system_prompt(assembled_prompt.system_message)
        history = self._resolve_history(assembled_prompt, chat_history)
        evaluators = self._resolve_evaluators()

        started = time.monotonic()
        self._emit(
            "llm.stream.start",
            scaffold_id=scaffold.id,
            history_turns=len(history),
            evaluator_count=len(evaluators),
        )

        collected: list[str] = []
        async for chunk in self.provider.generate_stream(
            system_message=full_system,
            user_message=assembled_prompt.user_message,
            chat_history=history,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if not chunk:
                continue

            collected.append(chunk)
            raw_content = "".join(collected)

            # Hot path optimization: run guardrail checks per chunk, defer
            # expensive disclaimer/context/provenance processing until halt/final.
            guardrail_check = await check_guardrails_async(
                raw_content, scaffold, evaluators=evaluators
            )
            if not guardrail_check.passed:
                content, flags, exports, _passed = self._finalize_postprocess(
                    raw_content, scaffold, guardrail_check, data_context,
                    skip_disclaimers=skip_disclaimers,
                )
                self._emit(
                    "llm.stream.halted",
                    scaffold_id=scaffold.id,
                    elapsed_ms=(time.monotonic() - started) * 1000,
                )
                yield StreamEvent(
                    event="halted",
                    text=content,
                    response=self._build_response(
                        content, scaffold, flags, exports,
                        {"input_tokens": 0, "output_tokens": len(content.split())},
                    ),
                )
                return

            yield StreamEvent(event="chunk", text=chunk)

        # Final pass on complete content
        content, flags, exports, _ = await self._postprocess_async(
            "".join(collected).strip(), scaffold, evaluators, data_context,
            skip_disclaimers=skip_disclaimers,
        )

        self._emit(
            "llm.stream.complete",
            scaffold_id=scaffold.id,
            output_tokens=len(content.split()),
            elapsed_ms=(time.monotonic() - started) * 1000,
            guardrail_flag_count=len(flags),
        )

        yield StreamEvent(
            event="final",
            text=content,
            response=self._build_response(
                content, scaffold, flags, exports,
                {"input_tokens": 0, "output_tokens": len(content.split())},
            ),
        )
