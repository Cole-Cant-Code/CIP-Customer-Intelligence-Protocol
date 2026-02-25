"""Data source models — schema, privacy, query, and result types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from cip_protocol.scaffold.models import _normalize_string_list, _StrictModel


class PrivacyClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"


class DataField(_StrictModel):
    name: str
    type: str  # "string", "number", "integer", "boolean", "date", "currency", "list"
    required: bool = False
    description: str = ""
    pii: bool = False

    @field_validator("name", "type", "description")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class DataSchema(_StrictModel):
    fields: list[DataField]


class QueryParameter(_StrictModel):
    name: str
    type: str
    required: bool = False
    description: str = ""

    @field_validator("name", "type", "description")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class PrivacyPolicy(_StrictModel):
    classification: PrivacyClassification = PrivacyClassification.PUBLIC
    retention: str = "session"
    pii_fields: list[str] = Field(default_factory=list)
    requires_consent: bool = False

    @field_validator("retention")
    @classmethod
    def normalize_retention(cls, value: str) -> str:
        return value.strip()

    @field_validator("pii_fields")
    @classmethod
    def normalize_pii_fields(cls, values: list[str]) -> list[str]:
        return _normalize_string_list(values)


class DataSourceSpec(_StrictModel):
    """YAML-driven spec defining a data source's shape, privacy, and query interface."""

    id: str
    domain: str
    display_name: str
    description: str
    source_type: str  # "api", "user_provided", "file", "stream"
    data_schema: DataSchema
    query_parameters: list[QueryParameter] = Field(default_factory=list)
    privacy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
    tags: list[str] = Field(default_factory=list)

    @field_validator("id", "domain", "display_name", "description", "source_type")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        return _normalize_string_list(values)


class DataQuery(_StrictModel):
    """Standardized query passed to a DataSource."""

    source_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    limit: int = 100

    @field_validator("source_id")
    @classmethod
    def normalize_source_id(cls, value: str) -> str:
        return value.strip()


class DataResult(_StrictModel):
    """Standardized response from a DataSource."""

    source_id: str
    records: list[dict[str, Any]]
    record_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id")
    @classmethod
    def normalize_source_id(cls, value: str) -> str:
        return value.strip()


class ValidationResult(_StrictModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DataRequirement(_StrictModel):
    """A scaffold's declaration that it needs data from a specific source."""

    source_id: str
    required: bool = False

    @field_validator("source_id")
    @classmethod
    def normalize_source_id(cls, value: str) -> str:
        return value.strip()
