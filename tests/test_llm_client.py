"""Tests for the InnerLLMClient and guardrail pipeline."""


import pytest
from conftest import make_test_config, make_test_scaffold

from cip_protocol.llm.client import InnerLLMClient
from cip_protocol.llm.providers.mock import MockProvider
from cip_protocol.llm.response import (
    check_guardrails,
    enforce_disclaimers,
    sanitize_content,
)
from cip_protocol.scaffold.models import AssembledPrompt


class TestGuardrails:
    def test_clean_content_passes(self):
        scaffold = make_test_scaffold()
        result = check_guardrails("This is a clean response.", scaffold)
        assert result.passed
        assert result.flags == []

    def test_prohibited_pattern_detected(self):
        scaffold = make_test_scaffold()
        indicators = {
            "making guarantees": ("guaranteed to", "i guarantee"),
        }
        result = check_guardrails(
            "This is guaranteed to work!",
            scaffold,
            prohibited_indicators=indicators,
        )
        assert not result.passed
        assert any("prohibited" in f for f in result.flags)

    def test_no_indicators_means_no_checking(self):
        scaffold = make_test_scaffold()
        result = check_guardrails(
            "I guarantee this will work.",
            scaffold,
            prohibited_indicators=None,
        )
        assert result.passed

    def test_escalation_trigger_detected(self):
        scaffold = make_test_scaffold(
            escalation_triggers=["severe financial distress"]
        )
        result = check_guardrails(
            "You appear to be in severe financial distress.",
            scaffold,
        )
        assert any("escalation" in f for f in result.flags)
        # Escalation triggers are soft — they don't fail the check
        assert result.passed


class TestSanitization:
    def test_clean_content_unchanged(self):
        from cip_protocol.llm.response import GuardrailCheck

        check = GuardrailCheck(passed=True, flags=[])
        result = sanitize_content("Hello world.", check)
        assert result == "Hello world."

    def test_prohibited_content_redacted(self):
        from cip_protocol.llm.response import GuardrailCheck

        check = GuardrailCheck(
            passed=False,
            flags=["prohibited_pattern_detected: guarantees ('guaranteed to')"],
        )
        result = sanitize_content(
            "This is guaranteed to make you rich.",
            check,
            redaction_message="[REDACTED]",
        )
        assert "[REDACTED]" in result
        assert "guaranteed to" not in result

    def test_custom_redaction_message(self):
        from cip_protocol.llm.response import GuardrailCheck

        check = GuardrailCheck(
            passed=False,
            flags=["prohibited_pattern_detected: diagnosis ('you have')"],
        )
        result = sanitize_content(
            "Based on these symptoms, you have the flu.",
            check,
            redaction_message="[Removed: medical diagnosis not permitted]",
        )
        assert "medical diagnosis not permitted" in result


class TestDisclaimers:
    def test_missing_disclaimers_appended(self):
        scaffold = make_test_scaffold(
            disclaimers=["This is not professional advice."]
        )
        content, flags = enforce_disclaimers("Here is my analysis.", scaffold)
        assert "This is not professional advice." in content
        assert len(flags) == 1

    def test_present_disclaimers_not_duplicated(self):
        scaffold = make_test_scaffold(
            disclaimers=["This is for informational purposes only."]
        )
        original = "Analysis here. This is for informational purposes only."
        content, flags = enforce_disclaimers(original, scaffold)
        assert content == original
        assert flags == []


class TestInnerLLMClient:
    @pytest.mark.asyncio
    async def test_invoke_with_config(self):
        config = make_test_config()
        provider = MockProvider(response_content="Test analysis complete.")
        client = InnerLLMClient(provider, config=config)

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(
            system_message="Analyze this.",
            user_message="What do you see?",
        )

        response = await client.invoke(
            assembled_prompt=prompt,
            scaffold=scaffold,
        )

        assert response.content is not None
        assert response.scaffold_id == "test_scaffold"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_system_prompt_prepended(self):
        config = make_test_config()
        provider = MockProvider()
        client = InnerLLMClient(provider, config=config)

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(
            system_message="Scaffold instructions here.",
            user_message="Query.",
        )

        await client.invoke(assembled_prompt=prompt, scaffold=scaffold)

        # The domain system prompt should be prepended
        assert "test specialist" in provider.last_system_message
        assert "Scaffold instructions here." in provider.last_system_message

    @pytest.mark.asyncio
    async def test_no_config_skips_system_prompt(self):
        provider = MockProvider()
        client = InnerLLMClient(provider, config=None)

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(
            system_message="Just scaffold.",
            user_message="Query.",
        )

        await client.invoke(assembled_prompt=prompt, scaffold=scaffold)
        assert provider.last_system_message == "Just scaffold."

    @pytest.mark.asyncio
    async def test_guardrails_enforced_from_config(self):
        config = make_test_config()
        provider = MockProvider(
            response_content="This is guaranteed to work perfectly."
        )
        client = InnerLLMClient(provider, config=config)

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(
            system_message="Analyze.", user_message="Query."
        )

        response = await client.invoke(
            assembled_prompt=prompt, scaffold=scaffold
        )

        assert any("prohibited" in f for f in response.guardrail_flags)
        assert "guaranteed to" not in response.content

    @pytest.mark.asyncio
    async def test_provenance_footer_appended(self):
        config = make_test_config()
        provider = MockProvider(response_content="Analysis.")
        client = InnerLLMClient(provider, config=config)

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(
            system_message="Analyze.", user_message="Query."
        )

        response = await client.invoke(
            assembled_prompt=prompt,
            scaffold=scaffold,
            data_context={"data_source": "test_provider"},
        )

        assert "Data source: test_provider" in response.content
