"""Tests for data source models."""

from __future__ import annotations

from cip_protocol.data.models import (
    DataField,
    DataQuery,
    DataRequirement,
    DataResult,
    DataSchema,
    DataSourceSpec,
    PrivacyClassification,
    PrivacyPolicy,
    QueryParameter,
    ValidationResult,
)


class TestDataField:
    def test_basic_construction(self):
        f = DataField(name="price", type="currency")
        assert f.name == "price"
        assert f.type == "currency"
        assert f.required is False
        assert f.pii is False

    def test_required_pii_field(self):
        f = DataField(name="ssn", type="string", required=True, pii=True)
        assert f.required is True
        assert f.pii is True

    def test_whitespace_normalization(self):
        f = DataField(name="  price  ", type=" currency ", description="  a field  ")
        assert f.name == "price"
        assert f.type == "currency"
        assert f.description == "a field"


class TestDataSchema:
    def test_empty_fields(self):
        s = DataSchema(fields=[])
        assert s.fields == []

    def test_with_fields(self):
        s = DataSchema(fields=[
            DataField(name="a", type="string"),
            DataField(name="b", type="number", required=True),
        ])
        assert len(s.fields) == 2
        assert s.fields[1].required is True


class TestPrivacyClassification:
    def test_all_values(self):
        assert PrivacyClassification.PUBLIC == "public"
        assert PrivacyClassification.INTERNAL == "internal"
        assert PrivacyClassification.PERSONAL == "personal"
        assert PrivacyClassification.SENSITIVE == "sensitive"

    def test_from_string(self):
        assert PrivacyClassification("sensitive") == PrivacyClassification.SENSITIVE


class TestPrivacyPolicy:
    def test_defaults(self):
        p = PrivacyPolicy()
        assert p.classification == PrivacyClassification.PUBLIC
        assert p.retention == "session"
        assert p.pii_fields == []
        assert p.requires_consent is False

    def test_sensitive_with_consent(self):
        p = PrivacyPolicy(
            classification=PrivacyClassification.SENSITIVE,
            retention="session",
            pii_fields=["medications", "ssn"],
            requires_consent=True,
        )
        assert p.classification == PrivacyClassification.SENSITIVE
        assert p.requires_consent is True
        assert len(p.pii_fields) == 2


class TestQueryParameter:
    def test_basic(self):
        qp = QueryParameter(name="location", type="string", required=True)
        assert qp.name == "location"
        assert qp.required is True

    def test_optional(self):
        qp = QueryParameter(name="price_max", type="number")
        assert qp.required is False


class TestDataSourceSpec:
    def test_minimal_api_spec(self):
        spec = DataSourceSpec(
            id="listings",
            domain="real_estate",
            display_name="Listings",
            description="Property listings",
            source_type="api",
            data_schema=DataSchema(fields=[
                DataField(name="price", type="currency", required=True),
            ]),
        )
        assert spec.id == "listings"
        assert spec.source_type == "api"
        assert spec.privacy.classification == PrivacyClassification.PUBLIC
        assert spec.tags == []

    def test_user_provided_spec(self):
        spec = DataSourceSpec(
            id="health",
            domain="health",
            display_name="Health Records",
            description="User health data",
            source_type="user_provided",
            data_schema=DataSchema(fields=[
                DataField(name="weight", type="number"),
                DataField(name="medications", type="list", pii=True),
            ]),
            privacy=PrivacyPolicy(
                classification=PrivacyClassification.SENSITIVE,
                requires_consent=True,
                pii_fields=["medications"],
            ),
            tags=["health", "personal"],
        )
        assert spec.source_type == "user_provided"
        assert spec.privacy.requires_consent is True
        assert len(spec.tags) == 2

    def test_whitespace_normalization(self):
        spec = DataSourceSpec(
            id="  test  ",
            domain="  test  ",
            display_name="  Test  ",
            description="  desc  ",
            source_type="  api  ",
            data_schema=DataSchema(fields=[]),
        )
        assert spec.id == "test"
        assert spec.domain == "test"


class TestDataQuery:
    def test_basic_query(self):
        q = DataQuery(source_id="listings", parameters={"location": "Denver"})
        assert q.source_id == "listings"
        assert q.limit == 100

    def test_custom_limit(self):
        q = DataQuery(source_id="listings", limit=10)
        assert q.limit == 10


class TestDataResult:
    def test_basic_result(self):
        r = DataResult(
            source_id="listings",
            records=[{"price": 450000, "address": "123 Main St"}],
            record_count=1,
        )
        assert r.record_count == 1
        assert len(r.records) == 1

    def test_with_metadata(self):
        r = DataResult(
            source_id="listings",
            records=[],
            record_count=0,
            metadata={"fetched_at": "2025-01-01T00:00:00Z"},
        )
        assert "fetched_at" in r.metadata


class TestValidationResult:
    def test_valid(self):
        v = ValidationResult(valid=True)
        assert v.valid is True
        assert v.errors == []

    def test_invalid_with_errors(self):
        v = ValidationResult(valid=False, errors=["missing field 'price'"])
        assert v.valid is False
        assert len(v.errors) == 1


class TestDataRequirement:
    def test_basic(self):
        dr = DataRequirement(source_id="listings")
        assert dr.source_id == "listings"
        assert dr.required is False

    def test_required(self):
        dr = DataRequirement(source_id="listings", required=True)
        assert dr.required is True

    def test_whitespace_normalization(self):
        dr = DataRequirement(source_id="  listings  ")
        assert dr.source_id == "listings"
