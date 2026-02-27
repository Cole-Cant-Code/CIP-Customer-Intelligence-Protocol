"""Scaffold engine — select() finds the right scaffold, apply() renders it."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from cip_protocol.domain import DomainConfig
from cip_protocol.scaffold.loader import load_builtin_scaffolds
from cip_protocol.scaffold.matcher import (
    SelectionExplanation,
    SelectionParams,
    _score_scaffolds_layered,
    match_scaffold,
    score_scaffolds_explained,
)
from cip_protocol.scaffold.models import AssembledPrompt, ChatMessage, Scaffold
from cip_protocol.scaffold.registry import ScaffoldRegistry
from cip_protocol.scaffold.renderer import render_scaffold
from cip_protocol.telemetry import NoOpTelemetrySink, TelemetryEvent, TelemetrySink

if TYPE_CHECKING:
    from cip_protocol.control import RunPolicy

logger = logging.getLogger(__name__)


class ScaffoldNotFoundError(Exception):
    pass


class ScaffoldEngine:
    def __init__(
        self,
        registry: ScaffoldRegistry,
        config: DomainConfig | None = None,
        telemetry_sink: TelemetrySink | None = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.telemetry = telemetry_sink or NoOpTelemetrySink()
        self._last_scaffold_id: str | None = None
        load_builtin_scaffolds(registry)

    def _build_params(self, policy: RunPolicy | None = None) -> SelectionParams:
        """Build SelectionParams from policy and engine config."""
        bias = policy.scaffold_selection_bias if policy else None
        selection_params = getattr(policy, "selection_params", None) if policy else None

        if isinstance(selection_params, SelectionParams):
            p = selection_params
            if bias and not p.selection_bias:
                p.selection_bias = bias
        else:
            p = SelectionParams(selection_bias=bias)

        # Wire engine context into meta layer
        if p.context is None:
            p.context = {}
        if self.config and "domain" not in p.context:
            p.context["domain"] = self.config.name
        if self._last_scaffold_id and "prior_scaffold_id" not in p.context:
            p.context["prior_scaffold_id"] = self._last_scaffold_id

        return p

    def select(
        self,
        tool_name: str,
        user_input: str = "",
        caller_scaffold_id: str | None = None,
        policy: RunPolicy | None = None,
    ) -> Scaffold:
        params = self._build_params(policy)
        scaffold = match_scaffold(
            registry=self.registry,
            tool_name=tool_name,
            user_input=user_input,
            caller_scaffold_id=caller_scaffold_id,
            params=params,
        )

        if scaffold:
            self._last_scaffold_id = scaffold.id
            self.telemetry.emit(TelemetryEvent(
                name="scaffold.select",
                attributes={
                    "tool_name": tool_name,
                    "selected_scaffold_id": scaffold.id,
                    "selection_mode": "matched",
                },
            ))
            return scaffold

        # Fall back to domain default
        default_id = self.config.default_scaffold_id if self.config else None
        if default_id:
            default = self.registry.get(default_id)
            if default:
                self._last_scaffold_id = default.id
                self.telemetry.emit(TelemetryEvent(
                    name="scaffold.select",
                    attributes={
                        "tool_name": tool_name,
                        "selected_scaffold_id": default.id,
                        "selection_mode": "default",
                    },
                ))
                return default

        raise ScaffoldNotFoundError(
            f"No scaffold found for tool='{tool_name}', "
            f"input='{user_input[:50]}', "
            f"and no default scaffold configured"
        )

    def select_explained(
        self,
        tool_name: str,
        user_input: str = "",
        caller_scaffold_id: str | None = None,
        policy: RunPolicy | None = None,
    ) -> tuple[Scaffold, SelectionExplanation]:
        """Like select(), but returns (scaffold, explanation) with selection metadata."""
        params = self._build_params(policy)

        # Priority 1: explicit caller_scaffold_id
        if caller_scaffold_id:
            scaffold = self.registry.get(caller_scaffold_id)
            if scaffold:
                self._last_scaffold_id = scaffold.id
                explanation = SelectionExplanation(
                    selected_scaffold_id=scaffold.id,
                    selection_mode="caller_id",
                    tool_name=tool_name,
                    user_input=user_input,
                    confidence=1.0,
                    params_used=params,
                )
                self.telemetry.emit(TelemetryEvent(
                    name="scaffold.select",
                    attributes={
                        "tool_name": tool_name,
                        "selected_scaffold_id": scaffold.id,
                        "selection_mode": "caller_id",
                    },
                ))
                return scaffold, explanation

        # Priority 2: tool name match
        tool_matches = self.registry.find_by_tool(tool_name)
        if tool_matches:
            scaffold = tool_matches[0]
            self._last_scaffold_id = scaffold.id
            explanation = SelectionExplanation(
                selected_scaffold_id=scaffold.id,
                selection_mode="tool_match",
                tool_name=tool_name,
                user_input=user_input,
                confidence=1.0,
                params_used=params,
            )
            self.telemetry.emit(TelemetryEvent(
                name="scaffold.select",
                attributes={
                    "tool_name": tool_name,
                    "selected_scaffold_id": scaffold.id,
                    "selection_mode": "tool_match",
                },
            ))
            return scaffold, explanation

        # Priority 3: layered scoring
        if user_input:
            scaffold, scores, confidence, ambiguous = _score_scaffolds_layered(
                self.registry.all(), user_input, params,
            )
            if scaffold:
                self._last_scaffold_id = scaffold.id
                explanation = SelectionExplanation(
                    selected_scaffold_id=scaffold.id,
                    selection_mode="scored",
                    scores=scores,
                    tool_name=tool_name,
                    user_input=user_input,
                    confidence=confidence,
                    ambiguous=ambiguous,
                    params_used=params,
                )
                self.telemetry.emit(TelemetryEvent(
                    name="scaffold.select",
                    attributes={
                        "tool_name": tool_name,
                        "selected_scaffold_id": scaffold.id,
                        "selection_mode": "scored",
                        "confidence": confidence,
                        "ambiguous": ambiguous,
                    },
                ))
                return scaffold, explanation

        # Priority 4: domain default
        default_id = self.config.default_scaffold_id if self.config else None
        if default_id:
            default = self.registry.get(default_id)
            if default:
                self._last_scaffold_id = default.id
                scores = []
                if user_input:
                    scores = score_scaffolds_explained(
                        self.registry.all(), user_input, params=params,
                    )
                explanation = SelectionExplanation(
                    selected_scaffold_id=default.id,
                    selection_mode="default",
                    scores=scores,
                    tool_name=tool_name,
                    user_input=user_input,
                    params_used=params,
                )
                self.telemetry.emit(TelemetryEvent(
                    name="scaffold.select",
                    attributes={
                        "tool_name": tool_name,
                        "selected_scaffold_id": default.id,
                        "selection_mode": "default",
                    },
                ))
                return default, explanation

        raise ScaffoldNotFoundError(
            f"No scaffold found for tool='{tool_name}', "
            f"input='{user_input[:50]}', "
            f"and no default scaffold configured"
        )

    def apply(
        self,
        scaffold: Scaffold,
        user_query: str,
        data_context: dict[str, Any],
        cross_domain_context: dict[str, Any] | None = None,
        chat_history: list[ChatMessage] | None = None,
        tone_variant: str | None = None,
        output_format: str | None = None,
        compact: bool = False,
        policy: RunPolicy | None = None,
    ) -> AssembledPrompt:
        # Resolve effective overrides from policy
        if policy:
            tone_variant = policy.tone_variant or tone_variant
            output_format = policy.output_format or output_format
            if policy.compact is not None:
                compact = policy.compact

        label = self.config.data_context_label if self.config else "Data Context"
        self.telemetry.emit(TelemetryEvent(
            name="scaffold.apply",
            attributes={
                "scaffold_id": scaffold.id,
                "user_query_length": len(user_query),
                "data_context_keys": sorted(data_context.keys()),
            },
        ))
        return render_scaffold(
            scaffold=scaffold,
            user_query=user_query,
            data_context=data_context,
            cross_domain_context=cross_domain_context,
            chat_history=chat_history,
            tone_variant=tone_variant,
            output_format=output_format,
            data_context_label=label,
            compact=compact,
            policy=policy,
        )
