"""Tests for cip_protocol.orchestration.pool."""

from __future__ import annotations

import pytest
from conftest import make_test_config, make_test_scaffold

from cip_protocol.cip import CIP
from cip_protocol.llm.provider import DEFAULT_PROVIDER_MODELS
from cip_protocol.llm.providers.mock import MockProvider
from cip_protocol.orchestration.pool import ProviderPool
from cip_protocol.scaffold.registry import ScaffoldRegistry


@pytest.fixture()
def scaffold_dir(tmp_path):
    """Write a minimal scaffold YAML so ProviderPool can build real CIP instances."""
    import yaml

    scaffold = {
        "id": "test_scaffold",
        "version": "1.0",
        "domain": "test_domain",
        "display_name": "Test",
        "description": "Test scaffold",
        "applicability": {"tools": ["test_tool"], "keywords": ["test"]},
        "framing": {
            "role": "Analyst",
            "perspective": "Analytical",
            "tone": "neutral",
            "tone_variants": {"friendly": "Warm"},
        },
        "reasoning_framework": {"steps": ["Analyze"]},
        "domain_knowledge_activation": ["test"],
        "output_calibration": {
            "format": "structured_narrative",
            "format_options": ["structured_narrative"],
        },
        "guardrails": {
            "disclaimers": ["Test only."],
            "escalation_triggers": [],
            "prohibited_actions": [],
        },
        "data_requirements": [],
    }
    (tmp_path / "test.yaml").write_text(yaml.dump(scaffold))
    return str(tmp_path)


@pytest.fixture()
def config():
    return make_test_config()


@pytest.fixture()
def pool(config, scaffold_dir, monkeypatch):
    """Pool configured with 'mock' as a known provider for test isolation."""
    monkeypatch.setenv("CIP_LLM_PROVIDER", "mock")
    return ProviderPool(
        config,
        scaffold_dir,
        key_map={"mock": "MOCK_KEY", "openai": "OPENAI_API_KEY"},
        default_models={"mock": "", "openai": "gpt-4o"},
    )


class TestLazyCreation:
    def test_pool_starts_empty(self, pool):
        assert pool._pool == {}

    def test_first_get_creates_instance(self, pool):
        cip = pool.get("mock")
        assert isinstance(cip, CIP)
        assert "mock" in pool._pool

    def test_second_get_returns_same(self, pool):
        cip1 = pool.get("mock")
        cip2 = pool.get("mock")
        assert cip1 is cip2


class TestOverride:
    def test_override_bypasses_pool(self, pool):
        registry = ScaffoldRegistry()
        registry.register(make_test_scaffold())
        mock_cip = CIP(make_test_config(), registry, MockProvider())
        pool.set_override(mock_cip)
        assert pool.get() is mock_cip

    def test_override_none_restores(self, pool):
        registry = ScaffoldRegistry()
        registry.register(make_test_scaffold())
        mock_cip = CIP(make_test_config(), registry, MockProvider())
        pool.set_override(mock_cip)
        pool.set_override(None)
        result = pool.get("mock")
        assert result is not mock_cip
        assert isinstance(result, CIP)


class TestProviderResolution:
    def test_unknown_provider_raises(self, pool):
        with pytest.raises(ValueError, match="Unknown provider"):
            pool.get("deepseek")

    def test_set_provider_updates_default(self, pool):
        msg = pool.set_provider("mock")
        assert "mock" in msg
        assert pool._default_provider == "mock"

    def test_set_provider_unknown_returns_error(self, pool):
        msg = pool.set_provider("deepseek")
        assert "Unknown provider" in msg

    def test_missing_api_key_fails_fast_for_non_mock(self, config, scaffold_dir, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        strict_pool = ProviderPool(
            config,
            scaffold_dir,
            key_map={"openai": "OPENAI_API_KEY"},
            default_models={"openai": DEFAULT_PROVIDER_MODELS["openai"]},
        )
        with pytest.raises(ValueError, match="Missing API key"):
            strict_pool.get("openai")


class TestPrepareOrchestration:
    def test_validates_scaffold(self, pool):
        with pytest.raises(ValueError, match="Unknown scaffold_id"):
            pool.prepare_orchestration(
                tool_name="test_tool",
                scaffold_id="nonexistent",
            )

    def test_normalizes_empty(self, pool):
        cip, sid, pol, ctx = pool.prepare_orchestration(
            tool_name="test_tool",
            scaffold_id="",
            policy="  ",
            context_notes="",
        )
        assert isinstance(cip, CIP)
        assert sid is None
        assert pol is None
        assert ctx is None

    def test_returns_valid_scaffold(self, pool):
        cip, sid, pol, ctx = pool.prepare_orchestration(
            tool_name="test_tool",
            scaffold_id="test_scaffold",
            policy="be concise",
            context_notes="note",
        )
        assert sid == "test_scaffold"
        assert pol == "be concise"
        assert ctx == "note"


class TestGetInfo:
    def test_format(self, pool):
        pool.get("mock")  # initialize
        info = pool.get_info()
        assert "mock" in info
        assert "pool=" in info
