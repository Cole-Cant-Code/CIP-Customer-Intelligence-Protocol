"""Tests for cip_protocol.orchestration.runner."""

from __future__ import annotations

import json

import pytest
from conftest import make_test_config, make_test_scaffold

from cip_protocol.cip import CIP
from cip_protocol.llm.providers.mock import MockProvider
from cip_protocol.orchestration.runner import (
    build_cross_domain_context,
    build_raw_response,
    run_tool_with_orchestration,
)
from cip_protocol.scaffold.registry import ScaffoldRegistry


def _make_cip(provider=None) -> CIP:
    registry = ScaffoldRegistry()
    registry.register(make_test_scaffold())
    config = make_test_config()
    return CIP(config, registry, provider or MockProvider())


class TestBuildRawResponse:
    def test_keys(self):
        result = json.loads(build_raw_response("my_tool", {"foo": 1}))
        assert result["_raw"] is True
        assert result["_tool"] == "my_tool"
        assert result["_meta"]["schema_version"] == 1
        assert result["data"] == {"foo": 1}

    def test_serializes_non_json_types(self):
        from datetime import datetime

        raw = build_raw_response("t", {"ts": datetime(2024, 1, 1)})
        parsed = json.loads(raw)
        assert "2024" in parsed["data"]["ts"]


class TestBuildCrossDomainContext:
    def test_nonempty(self):
        ctx = build_cross_domain_context("hello world")
        assert ctx == {"orchestrator_notes": "hello world"}

    def test_empty_string(self):
        assert build_cross_domain_context("") is None

    def test_none(self):
        assert build_cross_domain_context(None) is None

    def test_whitespace_only(self):
        assert build_cross_domain_context("   ") is None

    def test_strips_whitespace(self):
        ctx = build_cross_domain_context("  padded  ")
        assert ctx == {"orchestrator_notes": "padded"}


class TestRunToolWithOrchestration:
    @pytest.mark.asyncio
    async def test_raw_mode_returns_json(self):
        cip = _make_cip()
        result = await run_tool_with_orchestration(
            cip,
            user_input="query",
            tool_name="test_tool",
            data_context={"key": "val"},
            raw=True,
        )
        parsed = json.loads(result)
        assert parsed["_raw"] is True
        assert parsed["data"] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_orchestrated_calls_cip(self):
        mock = MockProvider("Orchestrated response.")
        cip = _make_cip(provider=mock)
        result = await run_tool_with_orchestration(
            cip,
            user_input="test query",
            tool_name="test_tool",
            data_context={"x": 1},
            raw=False,
        )
        assert "Orchestrated response." in result

    @pytest.mark.asyncio
    async def test_passes_scaffold_and_policy(self):
        mock = MockProvider("OK")
        cip = _make_cip(provider=mock)
        result = await run_tool_with_orchestration(
            cip,
            user_input="query",
            tool_name="test_tool",
            data_context={},
            scaffold_id="test_scaffold",
            policy="be concise",
            raw=False,
        )
        assert result  # just verify it completes without error

    @pytest.mark.asyncio
    async def test_context_notes_forwarded(self):
        mock = MockProvider("With context")
        cip = _make_cip(provider=mock)
        result = await run_tool_with_orchestration(
            cip,
            user_input="query",
            tool_name="test_tool",
            data_context={},
            context_notes="important context",
            raw=False,
        )
        assert result
