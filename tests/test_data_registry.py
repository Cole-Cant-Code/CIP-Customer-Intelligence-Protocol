"""Tests for DataSourceRegistry."""

from __future__ import annotations

import pytest

from cip_protocol.data.models import (
    DataField,
    DataQuery,
    DataResult,
    DataSchema,
    DataSourceSpec,
)
from cip_protocol.data.registry import DataSourceRegistry


def _make_spec(
    source_id: str = "test_source",
    domain: str = "test_domain",
    source_type: str = "api",
) -> DataSourceSpec:
    return DataSourceSpec(
        id=source_id,
        domain=domain,
        display_name=f"Test: {source_id}",
        description=f"Test source {source_id}",
        source_type=source_type,
        data_schema=DataSchema(fields=[DataField(name="value", type="string")]),
    )


class _FakeDataSource:
    """Minimal DataSource implementation for testing."""

    def __init__(self, spec: DataSourceSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> DataSourceSpec:
        return self._spec

    async def fetch(self, query: DataQuery) -> DataResult:
        return DataResult(source_id=self._spec.id, records=[], record_count=0)


class TestDataSourceRegistry:
    def test_register_spec_and_get(self):
        reg = DataSourceRegistry()
        spec = _make_spec("listings")
        reg.register_spec(spec)
        assert reg.get_spec("listings") is spec

    def test_get_returns_none_for_unknown(self):
        reg = DataSourceRegistry()
        assert reg.get("unknown") is None
        assert reg.get_spec("unknown") is None

    def test_register_live_source(self):
        reg = DataSourceRegistry()
        spec = _make_spec("listings")
        source = _FakeDataSource(spec)
        reg.register(source)
        assert reg.get("listings") is source
        assert reg.get_spec("listings") is spec

    def test_duplicate_id_raises(self):
        reg = DataSourceRegistry()
        reg.register_spec(_make_spec("listings"))
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register_spec(_make_spec("listings"))

    def test_duplicate_live_source_raises(self):
        reg = DataSourceRegistry()
        spec = _make_spec("listings")
        reg.register(_FakeDataSource(spec))
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register(_FakeDataSource(spec))

    def test_for_domain(self):
        reg = DataSourceRegistry()
        reg.register_spec(_make_spec("a", domain="real_estate"))
        reg.register_spec(_make_spec("b", domain="real_estate"))
        reg.register_spec(_make_spec("c", domain="health"))

        re_specs = reg.for_domain("real_estate")
        assert len(re_specs) == 2
        assert {s.id for s in re_specs} == {"a", "b"}

        health_specs = reg.for_domain("health")
        assert len(health_specs) == 1

        assert reg.for_domain("unknown") == []

    def test_for_type(self):
        reg = DataSourceRegistry()
        reg.register_spec(_make_spec("a", source_type="api"))
        reg.register_spec(_make_spec("b", source_type="user_provided"))
        reg.register_spec(_make_spec("c", source_type="api"))

        api_specs = reg.for_type("api")
        assert len(api_specs) == 2

        user_specs = reg.for_type("user_provided")
        assert len(user_specs) == 1

    def test_all_specs(self):
        reg = DataSourceRegistry()
        reg.register_spec(_make_spec("a"))
        reg.register_spec(_make_spec("b"))
        assert len(reg.all_specs()) == 2

    def test_all_specs_empty(self):
        reg = DataSourceRegistry()
        assert reg.all_specs() == []

    def test_spec_only_not_in_get(self):
        """register_spec() should NOT make source available via get()."""
        reg = DataSourceRegistry()
        reg.register_spec(_make_spec("listings"))
        assert reg.get("listings") is None
        assert reg.get_spec("listings") is not None
