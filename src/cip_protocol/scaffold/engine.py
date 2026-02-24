"""Scaffold engine -- orchestrates selection and application of scaffolds.

This is the main entry point for the scaffold system. It implements the
Negotiated Expertise Pattern:
  1. select() finds the right scaffold for a given tool/input combination
  2. apply() renders that scaffold into a complete LLM prompt

The engine owns the fallback logic: if no scaffold matches, it tries the
domain's default_scaffold_id from the DomainConfig. If that also fails,
it raises ScaffoldNotFoundError so the caller can handle the situation.
"""

from __future__ import annotations

import logging
from typing import Any

from cip_protocol.domain import DomainConfig
from cip_protocol.scaffold.matcher import match_scaffold
from cip_protocol.scaffold.models import AssembledPrompt, Scaffold
from cip_protocol.scaffold.registry import ScaffoldRegistry
from cip_protocol.scaffold.renderer import render_scaffold

logger = logging.getLogger(__name__)


class ScaffoldNotFoundError(Exception):
    """Raised when no scaffold can be selected for a request."""


class ScaffoldEngine:
    """Selects the right scaffold and assembles it into an LLM prompt.

    This is the core of the Negotiated Expertise Pattern: the engine
    mediates between the caller's intent and the available scaffolds,
    selecting the best reasoning framework and rendering it into a
    prompt that shapes the inner LLM's behavior.

    Args:
        registry: The scaffold registry to search.
        config: Domain configuration. Provides default_scaffold_id and
            data_context_label.  If None, no default fallback is used
            and data context is labeled generically.
    """

    def __init__(
        self,
        registry: ScaffoldRegistry,
        config: DomainConfig | None = None,
    ) -> None:
        self.registry = registry
        self.config = config

    def select(
        self,
        tool_name: str,
        user_input: str = "",
        caller_scaffold_id: str | None = None,
    ) -> Scaffold:
        """Select the best scaffold for this invocation.

        Falls back to config.default_scaffold_id if no match is found.
        Raises ScaffoldNotFoundError if no scaffold matches and no default exists.
        """
        scaffold = match_scaffold(
            registry=self.registry,
            tool_name=tool_name,
            user_input=user_input,
            caller_scaffold_id=caller_scaffold_id,
        )

        if scaffold:
            return scaffold

        # Fall back to domain default
        default_id = self.config.default_scaffold_id if self.config else None
        if default_id:
            default = self.registry.get(default_id)
            if default:
                logger.info("Using default scaffold: %s", default_id)
                return default

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
        tone_variant: str | None = None,
        output_format: str | None = None,
    ) -> AssembledPrompt:
        """Combine scaffold + user query + data into a complete LLM prompt."""
        label = self.config.data_context_label if self.config else "Data Context"
        return render_scaffold(
            scaffold=scaffold,
            user_query=user_query,
            data_context=data_context,
            cross_domain_context=cross_domain_context,
            tone_variant=tone_variant,
            output_format=output_format,
            data_context_label=label,
        )
