"""Tests for DomainConfig — the protocol's domain boundary."""

from cip_protocol import DomainConfig


class TestDomainConfig:
    def test_minimal_config(self):
        config = DomainConfig(
            name="test",
            display_name="Test Domain",
            system_prompt="You are a test expert.",
        )
        assert config.name == "test"
        assert config.default_scaffold_id is None
        assert config.data_context_label == "Data Context"
        assert config.prohibited_indicators == {}
        assert config.redaction_message == "[Removed: contains prohibited content]"

    def test_full_config(self):
        config = DomainConfig(
            name="health",
            display_name="CIP Health",
            system_prompt="You are a health information specialist.",
            default_scaffold_id="symptom_overview",
            data_context_label="Health Records",
            prohibited_indicators={
                "diagnosing": ("you have", "this is definitely"),
            },
            redaction_message="[Removed: contains prohibited medical guidance]",
        )
        assert config.default_scaffold_id == "symptom_overview"
        assert config.data_context_label == "Health Records"
        assert "diagnosing" in config.prohibited_indicators
        assert "medical" in config.redaction_message

    def test_config_is_domain_agnostic(self):
        """The same DomainConfig structure works for any domain."""
        finance = DomainConfig(
            name="finance",
            display_name="CIP Finance",
            system_prompt="You are a finance expert.",
            prohibited_indicators={
                "recommending products": ("i recommend", "sign up for"),
            },
        )
        legal = DomainConfig(
            name="legal",
            display_name="CIP Legal",
            system_prompt="You are a legal information specialist.",
            prohibited_indicators={
                "giving legal advice": ("you should sue", "file a lawsuit"),
            },
        )
        assert finance.name != legal.name
        assert finance.prohibited_indicators != legal.prohibited_indicators
