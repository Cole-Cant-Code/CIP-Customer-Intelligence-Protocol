"""CIP facade — 3-line entry point for the protocol."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cip_protocol.conversation import Conversation

from cip_protocol.control import ConstraintParser, PresetRegistry, RunPolicy
from cip_protocol.domain import DomainConfig
from cip_protocol.llm.client import InnerLLMClient, LLMResponse, StreamEvent
from cip_protocol.llm.provider import LLMProvider, create_provider
from cip_protocol.llm.response import GuardrailEvaluator
from cip_protocol.scaffold.engine import ScaffoldEngine
from cip_protocol.scaffold.loader import load_scaffold_directory
from cip_protocol.scaffold.registry import ScaffoldRegistry
from cip_protocol.telemetry import NoOpTelemetrySink, TelemetrySink


@dataclass
class CIPResult:
    response: LLMResponse
    scaffold_id: str
    scaffold_display_name: str
    selection_mode: str  # "caller_id", "tool_match", "scored", "default"
    selection_scores: dict[str, float] = field(default_factory=dict)
    policy_source: str = ""
    unrecognized_constraints: list[str] = field(default_factory=list)


class CIP:
    """Single entry point that wires registry, engine, and client internally."""

    def __init__(
        self,
        config: DomainConfig,
        registry: ScaffoldRegistry,
        provider: LLMProvider,
        *,
        preset_registry: PresetRegistry | None = None,
        guardrail_evaluators: list[GuardrailEvaluator] | None = None,
        telemetry_sink: TelemetrySink | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.preset_registry = preset_registry or PresetRegistry()
        sink = telemetry_sink or NoOpTelemetrySink()
        self.engine = ScaffoldEngine(registry, config, telemetry_sink=sink)
        self.client = InnerLLMClient(
            provider, config,
            guardrail_evaluators=guardrail_evaluators,
            telemetry_sink=sink,
        )

    @classmethod
    def from_config(
        cls,
        config: DomainConfig,
        scaffold_dir: str,
        provider: str | LLMProvider = "mock",
        *,
        api_key: str = "",
        model: str = "",
        preset_registry: PresetRegistry | None = None,
        guardrail_evaluators: list[GuardrailEvaluator] | None = None,
        telemetry_sink: TelemetrySink | None = None,
    ) -> CIP:
        """Build a CIP instance from a config, scaffold directory, and provider name."""
        registry = ScaffoldRegistry()
        load_scaffold_directory(scaffold_dir, registry)

        if isinstance(provider, str):
            llm_provider = create_provider(provider, api_key=api_key, model=model)
        else:
            llm_provider = provider

        return cls(
            config, registry, llm_provider,
            preset_registry=preset_registry,
            guardrail_evaluators=guardrail_evaluators,
            telemetry_sink=telemetry_sink,
        )

    def _resolve_policy(
        self, policy: RunPolicy | str | None,
    ) -> tuple[RunPolicy | None, str, list[str]]:
        """Resolve policy from RunPolicy, constraint string, or None."""
        if policy is None:
            return None, "", []
        if isinstance(policy, str):
            result = ConstraintParser.parse(policy, self.preset_registry)
            return result.policy, result.policy.source, result.unrecognized
        return policy, policy.source, []

    async def run(
        self,
        user_input: str,
        *,
        tool_name: str = "",
        data_context: dict[str, Any] | None = None,
        policy: RunPolicy | str | None = None,
        scaffold_id: str | None = None,
        cross_domain_context: dict[str, Any] | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> CIPResult:
        resolved_policy, policy_source, unrecognized = self._resolve_policy(policy)
        data = data_context if data_context is not None else {}

        scaffold, explanation = self.engine.select_explained(
            tool_name=tool_name,
            user_input=user_input,
            caller_scaffold_id=scaffold_id,
            policy=resolved_policy,
        )

        prompt = self.engine.apply(
            scaffold, user_input, data,
            cross_domain_context=cross_domain_context,
            policy=resolved_policy,
        )

        response = await self.client.invoke(
            prompt, scaffold,
            data_context=data,
            chat_history=chat_history,
            policy=resolved_policy,
        )

        scores = {s.scaffold_id: s.total_score for s in explanation.scores}

        return CIPResult(
            response=response,
            scaffold_id=scaffold.id,
            scaffold_display_name=scaffold.display_name,
            selection_mode=explanation.selection_mode,
            selection_scores=scores,
            policy_source=policy_source,
            unrecognized_constraints=unrecognized,
        )

    async def stream(
        self,
        user_input: str,
        *,
        tool_name: str = "",
        data_context: dict[str, Any] | None = None,
        policy: RunPolicy | str | None = None,
        scaffold_id: str | None = None,
        cross_domain_context: dict[str, Any] | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        resolved_policy, _policy_source, _unrecognized = self._resolve_policy(policy)
        data = data_context if data_context is not None else {}

        scaffold, _explanation = self.engine.select_explained(
            tool_name=tool_name,
            user_input=user_input,
            caller_scaffold_id=scaffold_id,
            policy=resolved_policy,
        )

        prompt = self.engine.apply(
            scaffold, user_input, data,
            cross_domain_context=cross_domain_context,
            policy=resolved_policy,
        )

        async for event in self.client.invoke_stream(
            prompt, scaffold,
            data_context=data,
            chat_history=chat_history,
            policy=resolved_policy,
        ):
            yield event

    def conversation(self, *, max_history_turns: int = 20) -> Conversation:
        from cip_protocol.conversation import Conversation as _Conversation
        return _Conversation(self, max_history_turns=max_history_turns)
