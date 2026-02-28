"""Test fixtures for CIP Protocol tests."""

from __future__ import annotations

import pytest

from cip_protocol import DomainConfig
from cip_protocol.scaffold.matcher import clear_matcher_cache
from cip_protocol.scaffold.models import (
    DataRequirement,
    Scaffold,
    ScaffoldApplicability,
    ScaffoldFraming,
    ScaffoldGuardrails,
    ScaffoldOutputCalibration,
)


@pytest.fixture(autouse=True)
def _clear_matcher_cache():
    """Clear the matcher cache before each test to avoid cross-test pollution."""
    clear_matcher_cache()
    yield
    clear_matcher_cache()


def make_test_scaffold(
    scaffold_id: str = "test_scaffold",
    domain: str = "test_domain",
    tools: list[str] | None = None,
    keywords: list[str] | None = None,
    intent_signals: list[str] | None = None,
    disclaimers: list[str] | None = None,
    prohibited_actions: list[str] | None = None,
    escalation_triggers: list[str] | None = None,
    data_requirements: list[DataRequirement] | None = None,
    tags: list[str] | None = None,
) -> Scaffold:
    """Create a minimal scaffold for testing."""
    return Scaffold(
        id=scaffold_id,
        version="1.0",
        domain=domain,
        display_name=f"Test: {scaffold_id}",
        description=f"Test scaffold {scaffold_id}",
        applicability=ScaffoldApplicability(
            tools=tools or ["test_tool"],
            keywords=keywords or ["test"],
            intent_signals=intent_signals or [],
        ),
        framing=ScaffoldFraming(
            role="Test analyst",
            perspective="Analytical",
            tone="neutral",
            tone_variants={"friendly": "Warm and approachable"},
        ),
        reasoning_framework={"steps": ["Analyze data", "Draw conclusions"]},
        domain_knowledge_activation=["test knowledge"],
        output_calibration=ScaffoldOutputCalibration(
            format="structured_narrative",
            format_options=["structured_narrative", "bullet_points"],
        ),
        guardrails=ScaffoldGuardrails(
            disclaimers=disclaimers or ["This is for informational purposes only."],
            escalation_triggers=escalation_triggers or [],
            prohibited_actions=prohibited_actions or [],
        ),
        data_requirements=data_requirements or [],
        tags=tags or [],
    )


def make_test_config(
    name: str = "test_domain",
    default_scaffold_id: str | None = "test_scaffold",
    regex_guardrail_policies: dict[str, str] | None = None,
) -> DomainConfig:
    """Create a minimal DomainConfig for testing."""
    return DomainConfig(
        name=name,
        display_name=f"CIP Test: {name}",
        system_prompt=(
            "You are a test specialist. Analyze the provided data "
            "and give clear, actionable insights."
        ),
        default_scaffold_id=default_scaffold_id,
        data_context_label="Test Data",
        prohibited_indicators={
            "making guarantees": ("guaranteed to", "i guarantee"),
            "impersonating professionals": ("as your doctor", "as your lawyer"),
        },
        regex_guardrail_policies=regex_guardrail_policies or {},
        redaction_message="[Removed: contains prohibited test content]",
    )
