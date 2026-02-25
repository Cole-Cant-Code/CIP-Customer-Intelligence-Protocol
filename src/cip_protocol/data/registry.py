"""In-memory triple index for data sources: by id, by domain, by source_type."""

from __future__ import annotations

from cip_protocol.data.models import DataSourceSpec
from cip_protocol.data.source import DataSource


class DataSourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, DataSource] = {}
        self._specs: dict[str, DataSourceSpec] = {}
        self._by_domain: dict[str, list[str]] = {}
        self._by_type: dict[str, list[str]] = {}

    def register(self, source: DataSource) -> None:
        """Register a live data source (with implementation)."""
        spec = source.spec
        if spec.id in self._specs:
            raise ValueError(f"Duplicate data source id registered: {spec.id!r}")
        self._sources[spec.id] = source
        self._specs[spec.id] = spec
        self._index(spec)

    def register_spec(self, spec: DataSourceSpec) -> None:
        """Register a spec without a live implementation (for validation/authoring)."""
        if spec.id in self._specs:
            raise ValueError(f"Duplicate data source id registered: {spec.id!r}")
        self._specs[spec.id] = spec
        self._index(spec)

    def _index(self, spec: DataSourceSpec) -> None:
        self._by_domain.setdefault(spec.domain, []).append(spec.id)
        self._by_type.setdefault(spec.source_type, []).append(spec.id)

    def get(self, source_id: str) -> DataSource | None:
        return self._sources.get(source_id)

    def get_spec(self, source_id: str) -> DataSourceSpec | None:
        return self._specs.get(source_id)

    def for_domain(self, domain: str) -> list[DataSourceSpec]:
        return [self._specs[sid] for sid in self._by_domain.get(domain, [])]

    def for_type(self, source_type: str) -> list[DataSourceSpec]:
        return [self._specs[sid] for sid in self._by_type.get(source_type, [])]

    def all_specs(self) -> list[DataSourceSpec]:
        return list(self._specs.values())
