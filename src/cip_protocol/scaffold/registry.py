"""In-memory triple index: by scaffold ID, by tool name, by tag."""

from __future__ import annotations

from cip_protocol.scaffold.models import Scaffold


class ScaffoldRegistry:
    def __init__(self) -> None:
        self._scaffolds: dict[str, Scaffold] = {}
        self._by_tool: dict[str, list[str]] = {}
        self._by_tag: dict[str, list[str]] = {}

    def register(self, scaffold: Scaffold) -> None:
        if scaffold.id in self._scaffolds:
            raise ValueError(f"Duplicate scaffold id registered: {scaffold.id!r}")
        self._scaffolds[scaffold.id] = scaffold

        for tool in scaffold.applicability.tools:
            self._by_tool.setdefault(tool, []).append(scaffold.id)

        for tag in scaffold.tags:
            self._by_tag.setdefault(tag, []).append(scaffold.id)

    def register_tool_alias(self, tool_name: str, scaffold_id: str) -> None:
        """Register an additional tool name mapping for an existing scaffold."""
        if scaffold_id not in self._scaffolds:
            raise ValueError(f"Scaffold {scaffold_id!r} not registered")
        self._by_tool.setdefault(tool_name, []).append(scaffold_id)

    def get(self, scaffold_id: str) -> Scaffold | None:
        return self._scaffolds.get(scaffold_id)

    def find_by_tool(self, tool_name: str) -> list[Scaffold]:
        return [self._scaffolds[sid] for sid in self._by_tool.get(tool_name, [])]

    def find_by_tag(self, tag: str) -> list[Scaffold]:
        return [self._scaffolds[sid] for sid in self._by_tag.get(tag, [])]

    def all(self) -> list[Scaffold]:
        return list(self._scaffolds.values())
