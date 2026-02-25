"""YAML data-source-spec loading. Files starting with underscore are skipped."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from cip_protocol.data.models import (
    DataField,
    DataSchema,
    DataSourceSpec,
    PrivacyClassification,
    PrivacyPolicy,
    QueryParameter,
)
from cip_protocol.data.registry import DataSourceRegistry

logger = logging.getLogger(__name__)


def load_data_source_spec(path: Path) -> DataSourceSpec:
    with open(path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)
    if raw_data is None:
        raise ValueError(f"Empty data source YAML: {path}")
    if not isinstance(raw_data, dict):
        raise ValueError(f"Data source YAML root must be a mapping: {path}")

    data: dict[str, Any] = raw_data

    schema_raw = data.get("schema", {})
    fields = [
        DataField(
            name=f.get("name", ""),
            type=f.get("type", "string"),
            required=f.get("required", False),
            description=f.get("description", ""),
            pii=f.get("pii", False),
        )
        for f in schema_raw.get("fields", [])
    ]

    query_params = [
        QueryParameter(
            name=qp.get("name", ""),
            type=qp.get("type", "string"),
            required=qp.get("required", False),
            description=qp.get("description", ""),
        )
        for qp in data.get("query_parameters", [])
    ]

    privacy_raw = data.get("privacy", {})
    classification_str = privacy_raw.get("classification", "public")
    try:
        classification = PrivacyClassification(classification_str)
    except ValueError:
        logger.warning(
            "Unknown privacy classification %r, defaulting to PUBLIC",
            classification_str,
        )
        classification = PrivacyClassification.PUBLIC

    privacy = PrivacyPolicy(
        classification=classification,
        retention=privacy_raw.get("retention", "session"),
        pii_fields=privacy_raw.get("pii_fields", []),
        requires_consent=privacy_raw.get("requires_consent", False),
    )

    return DataSourceSpec(
        id=data["id"],
        domain=data["domain"],
        display_name=data["display_name"],
        description=data.get("description", "").strip(),
        source_type=data["source_type"],
        data_schema=DataSchema(fields=fields),
        query_parameters=query_params,
        privacy=privacy,
        tags=data.get("tags", []),
    )


def load_data_source_directory(
    directory: str | Path, registry: DataSourceRegistry,
) -> int:
    """Load all YAML data source specs from a directory recursively. Returns count loaded."""
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("Data source directory does not exist: %s", directory)
        return 0

    count = 0
    for path in sorted(directory.rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        try:
            spec = load_data_source_spec(path)
            registry.register_spec(spec)
            count += 1
        except (yaml.YAMLError, KeyError, ValueError, TypeError) as exc:
            logger.exception("Failed to load data source spec from %s: %s", path, exc)
    return count
