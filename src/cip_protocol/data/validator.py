"""Schema validation for data records and queries."""

from __future__ import annotations

from typing import Any

from cip_protocol.data.models import (
    DataQuery,
    DataSchema,
    DataSourceSpec,
    ValidationResult,
)

_NUMERIC_TYPES = frozenset({"number", "integer", "currency"})
_TYPE_CHECKERS: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "date": str,  # ISO date string
    "currency": (int, float),
    "list": list,
}


def validate_records(
    records: list[dict[str, Any]], schema: DataSchema,
) -> ValidationResult:
    """Validate a list of data records against a schema."""
    errors: list[str] = []
    warnings: list[str] = []

    required_fields = [f for f in schema.fields if f.required]
    field_map = {f.name: f for f in schema.fields}
    pii_fields = [f.name for f in schema.fields if f.pii]

    for i, record in enumerate(records):
        for rf in required_fields:
            if rf.name not in record:
                errors.append(f"Record {i}: missing required field '{rf.name}'")

        for key, value in record.items():
            field_def = field_map.get(key)
            if field_def is None:
                continue  # extra fields are allowed

            if value is None:
                if field_def.required:
                    errors.append(f"Record {i}: required field '{key}' is null")
                continue

            # bool is a subclass of int in Python — reject it for numeric types
            if isinstance(value, bool) and field_def.type in _NUMERIC_TYPES:
                errors.append(
                    f"Record {i}: field '{key}' expected {field_def.type}, got bool"
                )
                continue

            expected = _TYPE_CHECKERS.get(field_def.type)
            if expected and not isinstance(value, expected):
                errors.append(
                    f"Record {i}: field '{key}' expected {field_def.type}, "
                    f"got {type(value).__name__}"
                )

    if pii_fields:
        warnings.append(f"Data contains PII fields: {', '.join(pii_fields)}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_query(query: DataQuery, spec: DataSourceSpec) -> ValidationResult:
    """Validate a query against a data source spec's query parameters."""
    errors: list[str] = []
    warnings: list[str] = []

    required_params = [qp for qp in spec.query_parameters if qp.required]
    param_map = {qp.name: qp for qp in spec.query_parameters}

    for rp in required_params:
        if rp.name not in query.parameters:
            errors.append(f"Missing required query parameter '{rp.name}'")

    for key, value in query.parameters.items():
        param_def = param_map.get(key)
        if param_def is None:
            warnings.append(f"Unknown query parameter '{key}'")
            continue

        if value is None:
            if param_def.required:
                errors.append(f"Required query parameter '{key}' is null")
            continue

        if isinstance(value, bool) and param_def.type in _NUMERIC_TYPES:
            errors.append(
                f"Query parameter '{key}' expected {param_def.type}, got bool"
            )
            continue

        expected = _TYPE_CHECKERS.get(param_def.type)
        if expected and not isinstance(value, expected):
            errors.append(
                f"Query parameter '{key}' expected {param_def.type}, "
                f"got {type(value).__name__}"
            )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
