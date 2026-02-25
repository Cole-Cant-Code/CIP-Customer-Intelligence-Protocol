"""DataSource protocol — implemented by consuming MCP servers, not CIP."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cip_protocol.data.models import DataQuery, DataResult, DataSourceSpec


@runtime_checkable
class DataSource(Protocol):
    """Interface for domain data providers.

    Implementations live in consuming MCP servers, not in CIP.
    Two common patterns:

    - **API source** (real estate, vehicles): ``fetch()`` calls an external API.
    - **User-provided source** (health, finance): ``fetch()`` returns from a
      local store populated during an ingestion step.
    """

    @property
    def spec(self) -> DataSourceSpec: ...

    async def fetch(self, query: DataQuery) -> DataResult: ...
