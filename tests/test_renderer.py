"""Tests for the scaffold renderer."""

from conftest import make_test_scaffold

from cip_protocol.scaffold.renderer import render_scaffold


class TestRenderer:
    def test_system_message_contains_role(self):
        scaffold = make_test_scaffold()
        result = render_scaffold(
            scaffold=scaffold,
            user_query="test",
            data_context={},
        )
        assert "Test analyst" in result.system_message

    def test_user_message_contains_query(self):
        scaffold = make_test_scaffold()
        result = render_scaffold(
            scaffold=scaffold,
            user_query="What is happening?",
            data_context={},
        )
        assert "What is happening?" in result.user_message

    def test_data_context_label_default(self):
        scaffold = make_test_scaffold()
        result = render_scaffold(
            scaffold=scaffold,
            user_query="test",
            data_context={"key": "value"},
        )
        assert "## Data Context" in result.user_message

    def test_data_context_label_custom(self):
        scaffold = make_test_scaffold()
        result = render_scaffold(
            scaffold=scaffold,
            user_query="test",
            data_context={"key": "value"},
            data_context_label="Health Records",
        )
        assert "## Health Records" in result.user_message
        assert "## Data Context" not in result.user_message

    def test_tone_variant_override(self):
        scaffold = make_test_scaffold()
        result = render_scaffold(
            scaffold=scaffold,
            user_query="test",
            data_context={},
            tone_variant="friendly",
        )
        assert "Warm and approachable" in result.system_message

    def test_cross_domain_context(self):
        scaffold = make_test_scaffold()
        result = render_scaffold(
            scaffold=scaffold,
            user_query="test",
            data_context={},
            cross_domain_context={"health_score": 0.8},
        )
        assert "Context From Other Domains" in result.user_message
        assert "health_score" in result.user_message

    def test_metadata_populated(self):
        scaffold = make_test_scaffold()
        result = render_scaffold(
            scaffold=scaffold,
            user_query="test",
            data_context={},
        )
        assert result.metadata["scaffold_id"] == "test_scaffold"
        assert result.metadata["scaffold_version"] == "1.0"

    def test_guardrails_in_system_message(self):
        scaffold = make_test_scaffold(
            disclaimers=["Not professional advice."],
            prohibited_actions=["Never give diagnoses."],
            escalation_triggers=["severe distress detected"],
        )
        result = render_scaffold(
            scaffold=scaffold,
            user_query="test",
            data_context={},
        )
        assert "Not professional advice" in result.system_message
        assert "Never give diagnoses" in result.system_message
        assert "severe distress detected" in result.system_message

    def test_reasoning_steps_in_system_message(self):
        scaffold = make_test_scaffold()
        result = render_scaffold(
            scaffold=scaffold,
            user_query="test",
            data_context={},
        )
        assert "1. Analyze data" in result.system_message
        assert "2. Draw conclusions" in result.system_message
