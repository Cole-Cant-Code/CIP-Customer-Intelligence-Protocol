"""Tests for the InnerLLMClient and guardrail pipeline."""


import pytest
from conftest import make_test_config, make_test_scaffold

from cip_protocol.llm.client import InnerLLMClient
from cip_protocol.llm.providers.mock import MockProvider
from cip_protocol.llm.response import (
    GuardrailEvaluation,
    check_guardrails,
    enforce_disclaimers,
    sanitize_content,
)
from cip_protocol.scaffold.models import AssembledPrompt, ChatMessage
from cip_protocol.telemetry import InMemoryTelemetrySink


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

    def test_prohibited_pattern_requires_word_boundary(self):
        scaffold = make_test_scaffold()
        indicators = {"plan_advice": ("plan",)}
        result = check_guardrails(
            "This planetary model is useful.",
            scaffold,
            prohibited_indicators=indicators,
        )
        assert result.passed

    def test_prohibited_pattern_matches_with_flexible_whitespace(self):
        scaffold = make_test_scaffold()
        indicators = {"recommending": ("i recommend",)}
        result = check_guardrails(
            "If asked directly, I   recommend waiting a month.",
            scaffold,
            prohibited_indicators=indicators,
        )
        assert not result.passed

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

    @pytest.mark.asyncio
    async def test_chat_history_forwarded_from_prompt(self):
        provider = MockProvider(response_content="History-aware response.")
        client = InnerLLMClient(provider, config=make_test_config())

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(
            system_message="Analyze.",
            user_message="Current query.",
            chat_history=[
                ChatMessage(role="user", content="Previous user turn"),
                ChatMessage(role="assistant", content="Previous assistant turn"),
            ],
        )

        await client.invoke(assembled_prompt=prompt, scaffold=scaffold)

        assert len(provider.last_chat_history) == 2
        assert provider.last_chat_history[0]["role"] == "user"
        assert provider.last_chat_history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_chat_history_override_parameter(self):
        provider = MockProvider(response_content="Override history response.")
        client = InnerLLMClient(provider, config=make_test_config())

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(
            system_message="Analyze.",
            user_message="Current query.",
            chat_history=[ChatMessage(role="user", content="Prompt history turn")],
        )

        await client.invoke(
            assembled_prompt=prompt,
            scaffold=scaffold,
            chat_history=[{"role": "assistant", "content": "Explicit override"}],
        )

        assert len(provider.last_chat_history) == 1
        assert provider.last_chat_history[0]["content"] == "Explicit override"

    @pytest.mark.asyncio
    async def test_custom_guardrail_evaluator_pipeline(self):
        class AlwaysBlockEvaluator:
            name = "always_block"

            def evaluate(self, content: str, scaffold):  # noqa: ANN001
                _ = content, scaffold
                return GuardrailEvaluation(
                    evaluator_name=self.name,
                    flags=["regex_policy_violation: blocked"],
                    hard_violations=["blocked"],
                )

        provider = MockProvider(response_content="All good.")
        client = InnerLLMClient(
            provider,
            config=make_test_config(),
            guardrail_evaluators=[AlwaysBlockEvaluator()],
        )

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")
        response = await client.invoke(assembled_prompt=prompt, scaffold=scaffold)

        assert "prohibited test content" in response.content
        assert any("regex_policy_violation" in flag for flag in response.guardrail_flags)

    @pytest.mark.asyncio
    async def test_regex_guardrail_policy_from_config(self):
        config = make_test_config(
            regex_guardrail_policies={"dosage_directive": r"\btake\b.+\d+mg\b"}
        )
        provider = MockProvider(response_content="You should take 20mg every day.")
        client = InnerLLMClient(provider, config=config)

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")
        response = await client.invoke(assembled_prompt=prompt, scaffold=scaffold)

        assert "prohibited test content" in response.content
        assert any("regex_policy_violation" in flag for flag in response.guardrail_flags)

    @pytest.mark.asyncio
    async def test_invoke_stream_emits_final_event(self):
        provider = MockProvider(response_content="Streaming works.")
        client = InnerLLMClient(provider, config=make_test_config())

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")

        events = []
        async for event in client.invoke_stream(assembled_prompt=prompt, scaffold=scaffold):
            events.append(event)

        assert any(event.event == "chunk" for event in events)
        assert events[-1].event == "final"
        assert events[-1].response is not None

    @pytest.mark.asyncio
    async def test_invoke_stream_halts_on_guardrail_violation(self):
        provider = MockProvider(response_content="This is guaranteed to work.")
        client = InnerLLMClient(provider, config=make_test_config())

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")

        events = []
        async for event in client.invoke_stream(assembled_prompt=prompt, scaffold=scaffold):
            events.append(event)

        assert events[-1].event == "halted"
        assert events[-1].response is not None
        assert "guaranteed to" not in events[-1].response.content

    @pytest.mark.asyncio
    async def test_telemetry_events_emitted(self):
        sink = InMemoryTelemetrySink()
        provider = MockProvider(response_content="Telemetry test.")
        client = InnerLLMClient(provider, config=make_test_config(), telemetry_sink=sink)

        scaffold = make_test_scaffold()
        prompt = AssembledPrompt(system_message="Analyze.", user_message="Query.")
        await client.invoke(assembled_prompt=prompt, scaffold=scaffold)

        names = [event.name for event in sink.events]
        assert "llm.invoke.start" in names
        assert "llm.invoke.complete" in names
