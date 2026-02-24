"""Tests for CIP_PERF_MODE environment toggle on _StrictModel."""

from __future__ import annotations

import importlib
import os
import sys

import pytest
from pydantic import ValidationError


def _reload_models(perf_mode: str | None):
    """Reload models module with the given CIP_PERF_MODE value."""
    env_key = "CIP_PERF_MODE"
    old = os.environ.get(env_key)
    try:
        if perf_mode is not None:
            os.environ[env_key] = perf_mode
        elif env_key in os.environ:
            del os.environ[env_key]

        # Remove cached modules so reload picks up new env
        mods_to_remove = [k for k in sys.modules if k.startswith("cip_protocol.scaffold.models")]
        for mod in mods_to_remove:
            del sys.modules[mod]

        return importlib.import_module("cip_protocol.scaffold.models")
    finally:
        # Restore original env
        if old is not None:
            os.environ[env_key] = old
        elif env_key in os.environ:
            del os.environ[env_key]
        # Restore original module
        mods_to_remove = [k for k in sys.modules if k.startswith("cip_protocol.scaffold.models")]
        for mod in mods_to_remove:
            del sys.modules[mod]
        importlib.import_module("cip_protocol.scaffold.models")


class TestPerfModeToggle:
    def test_strict_mode_rejects_extra_fields(self):
        models = _reload_models(None)
        with pytest.raises(ValidationError):
            models.ChatMessage(role="user", content="hi", bogus_field="oops")

    def test_perf_mode_accepts_extra_fields(self):
        models = _reload_models("1")
        msg = models.ChatMessage(role="user", content="hello", bogus_field="ignored")
        assert msg.content == "hello"

    def test_perf_mode_zero_stays_strict(self):
        models = _reload_models("0")
        with pytest.raises(ValidationError):
            models.ChatMessage(role="user", content="hi", bogus_field="oops")

    def test_perf_mode_flag_read_correctly(self):
        models = _reload_models("1")
        assert models._PERF_MODE is True

    def test_default_is_strict(self):
        models = _reload_models(None)
        assert models._PERF_MODE is False
