"""Tests for data schema and query validation."""

from __future__ import annotations

from cip_protocol.data.models import (
    DataField,
    DataQuery,
    DataSchema,
    DataSourceSpec,
    QueryParameter,
)
from cip_protocol.data.validator import validate_query, validate_records


def _make_schema(*fields: DataField) -> DataSchema:
    return DataSchema(fields=list(fields))


class TestValidateRecords:
    def test_valid_records(self):
        schema = _make_schema(
            DataField(name="price", type="currency", required=True),
            DataField(name="address", type="string", required=True),
            DataField(name="bedrooms", type="integer"),
        )
        records = [
            {"price": 450000, "address": "123 Main St", "bedrooms": 3},
            {"price": 325000.50, "address": "456 Oak Ave", "bedrooms": 2},
        ]
        result = validate_records(records, schema)
        assert result.valid is True
        assert result.errors == []

    def test_missing_required_field(self):
        schema = _make_schema(
            DataField(name="price", type="currency", required=True),
            DataField(name="address", type="string", required=True),
        )
        records = [{"price": 450000}]  # missing address
        result = validate_records(records, schema)
        assert result.valid is False
        assert any("address" in e for e in result.errors)

    def test_null_required_field(self):
        schema = _make_schema(
            DataField(name="price", type="currency", required=True),
        )
        records = [{"price": None}]
        result = validate_records(records, schema)
        assert result.valid is False
        assert any("null" in e for e in result.errors)

    def test_wrong_type(self):
        schema = _make_schema(
            DataField(name="price", type="currency"),
        )
        records = [{"price": "not a number"}]
        result = validate_records(records, schema)
        assert result.valid is False
        assert any("expected currency" in e for e in result.errors)

    def test_bool_not_accepted_as_number(self):
        schema = _make_schema(
            DataField(name="count", type="number"),
        )
        records = [{"count": True}]
        result = validate_records(records, schema)
        assert result.valid is False
        assert any("bool" in e for e in result.errors)

    def test_bool_not_accepted_as_integer(self):
        schema = _make_schema(
            DataField(name="count", type="integer"),
        )
        records = [{"count": False}]
        result = validate_records(records, schema)
        assert result.valid is False

    def test_extra_fields_allowed(self):
        schema = _make_schema(
            DataField(name="price", type="currency"),
        )
        records = [{"price": 100, "extra_field": "whatever"}]
        result = validate_records(records, schema)
        assert result.valid is True

    def test_pii_warning(self):
        schema = _make_schema(
            DataField(name="medications", type="list", pii=True),
        )
        records = [{"medications": ["aspirin"]}]
        result = validate_records(records, schema)
        assert result.valid is True
        assert any("PII" in w for w in result.warnings)

    def test_empty_records(self):
        schema = _make_schema(
            DataField(name="price", type="currency", required=True),
        )
        result = validate_records([], schema)
        assert result.valid is True

    def test_multiple_errors_across_records(self):
        schema = _make_schema(
            DataField(name="a", type="string", required=True),
            DataField(name="b", type="integer", required=True),
        )
        records = [
            {"a": "ok"},       # missing b
            {"b": "wrong"},    # missing a, wrong type for b
        ]
        result = validate_records(records, schema)
        assert result.valid is False
        assert len(result.errors) >= 2

    def test_null_optional_field_ok(self):
        schema = _make_schema(
            DataField(name="notes", type="string"),
        )
        records = [{"notes": None}]
        result = validate_records(records, schema)
        assert result.valid is True

    def test_list_type_validation(self):
        schema = _make_schema(
            DataField(name="tags", type="list"),
        )
        result_ok = validate_records([{"tags": ["a", "b"]}], schema)
        assert result_ok.valid is True

        result_bad = validate_records([{"tags": "not a list"}], schema)
        assert result_bad.valid is False

    def test_date_type_accepts_string(self):
        schema = _make_schema(
            DataField(name="created", type="date"),
        )
        result = validate_records([{"created": "2025-01-01"}], schema)
        assert result.valid is True


class TestValidateQuery:
    def _make_spec_with_params(self, *params: QueryParameter) -> DataSourceSpec:
        return DataSourceSpec(
            id="test",
            domain="test",
            display_name="Test",
            description="Test",
            source_type="api",
            data_schema=DataSchema(fields=[]),
            query_parameters=list(params),
        )

    def test_valid_query(self):
        spec = self._make_spec_with_params(
            QueryParameter(name="location", type="string", required=True),
            QueryParameter(name="price_max", type="number"),
        )
        query = DataQuery(source_id="test", parameters={"location": "Denver", "price_max": 500000})
        result = validate_query(query, spec)
        assert result.valid is True

    def test_missing_required_param(self):
        spec = self._make_spec_with_params(
            QueryParameter(name="location", type="string", required=True),
        )
        query = DataQuery(source_id="test", parameters={})
        result = validate_query(query, spec)
        assert result.valid is False
        assert any("location" in e for e in result.errors)

    def test_null_required_param(self):
        spec = self._make_spec_with_params(
            QueryParameter(name="location", type="string", required=True),
        )
        query = DataQuery(source_id="test", parameters={"location": None})
        result = validate_query(query, spec)
        assert result.valid is False

    def test_wrong_type_param(self):
        spec = self._make_spec_with_params(
            QueryParameter(name="price_max", type="number"),
        )
        query = DataQuery(source_id="test", parameters={"price_max": "not a number"})
        result = validate_query(query, spec)
        assert result.valid is False

    def test_unknown_param_warning(self):
        spec = self._make_spec_with_params(
            QueryParameter(name="location", type="string"),
        )
        query = DataQuery(source_id="test", parameters={"location": "Denver", "foo": "bar"})
        result = validate_query(query, spec)
        assert result.valid is True
        assert any("Unknown" in w for w in result.warnings)

    def test_empty_params_no_required(self):
        spec = self._make_spec_with_params(
            QueryParameter(name="location", type="string"),
        )
        query = DataQuery(source_id="test", parameters={})
        result = validate_query(query, spec)
        assert result.valid is True

    def test_bool_not_accepted_as_number_param(self):
        spec = self._make_spec_with_params(
            QueryParameter(name="count", type="number"),
        )
        query = DataQuery(source_id="test", parameters={"count": True})
        result = validate_query(query, spec)
        assert result.valid is False
