"""Tests for YAML data source spec loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from cip_protocol.data.loader import load_data_source_directory, load_data_source_spec
from cip_protocol.data.models import PrivacyClassification
from cip_protocol.data.registry import DataSourceRegistry


@pytest.fixture()
def data_sources_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data_sources"
    d.mkdir()
    return d


def _write_yaml(directory: Path, filename: str, content: str) -> Path:
    path = directory / filename
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


class TestLoadDataSourceSpec:
    def test_load_api_spec(self, data_sources_dir: Path):
        path = _write_yaml(data_sources_dir, "listings.yaml", """\
            id: listings
            domain: real_estate
            display_name: Real Estate Listings
            description: Active property listings
            source_type: api

            schema:
              fields:
                - name: price
                  type: currency
                  required: true
                - name: address
                  type: string
                  required: true
                - name: bedrooms
                  type: integer

            query_parameters:
              - name: location
                type: string
                required: true
              - name: price_max
                type: number

            privacy:
              classification: public
              retention: none

            tags: [real_estate, listings]
        """)
        spec = load_data_source_spec(path)
        assert spec.id == "listings"
        assert spec.domain == "real_estate"
        assert spec.source_type == "api"
        assert len(spec.data_schema.fields) == 3
        assert spec.data_schema.fields[0].required is True
        assert len(spec.query_parameters) == 2
        assert spec.query_parameters[0].required is True
        assert spec.privacy.classification == PrivacyClassification.PUBLIC
        assert spec.privacy.retention == "none"
        assert spec.tags == ["real_estate", "listings"]

    def test_load_user_provided_spec(self, data_sources_dir: Path):
        path = _write_yaml(data_sources_dir, "health.yaml", """\
            id: health_records
            domain: health
            display_name: Health Records
            description: User-provided health data
            source_type: user_provided

            schema:
              fields:
                - name: date
                  type: date
                  required: true
                - name: weight
                  type: number
                - name: medications
                  type: list
                  pii: true

            privacy:
              classification: sensitive
              retention: session
              pii_fields: [medications]
              requires_consent: true

            tags: [health]
        """)
        spec = load_data_source_spec(path)
        assert spec.id == "health_records"
        assert spec.source_type == "user_provided"
        assert spec.privacy.classification == PrivacyClassification.SENSITIVE
        assert spec.privacy.requires_consent is True
        assert "medications" in spec.privacy.pii_fields
        assert spec.data_schema.fields[2].pii is True

    def test_load_minimal_spec(self, data_sources_dir: Path):
        path = _write_yaml(data_sources_dir, "minimal.yaml", """\
            id: minimal
            domain: test
            display_name: Minimal
            source_type: api
            schema:
              fields: []
        """)
        spec = load_data_source_spec(path)
        assert spec.id == "minimal"
        assert spec.description == ""
        assert spec.privacy.classification == PrivacyClassification.PUBLIC
        assert spec.query_parameters == []
        assert spec.tags == []

    def test_empty_yaml_raises(self, data_sources_dir: Path):
        path = _write_yaml(data_sources_dir, "empty.yaml", "")
        with pytest.raises(ValueError, match="Empty"):
            load_data_source_spec(path)

    def test_non_mapping_raises(self, data_sources_dir: Path):
        path = _write_yaml(data_sources_dir, "list.yaml", "- item1\n- item2\n")
        with pytest.raises(ValueError, match="mapping"):
            load_data_source_spec(path)

    def test_missing_required_field_raises(self, data_sources_dir: Path):
        path = _write_yaml(data_sources_dir, "bad.yaml", """\
            id: bad
            domain: test
        """)
        with pytest.raises((KeyError, TypeError)):
            load_data_source_spec(path)


class TestLoadDataSourceDirectory:
    def test_load_multiple(self, data_sources_dir: Path):
        _write_yaml(data_sources_dir, "a.yaml", """\
            id: a
            domain: test
            display_name: A
            source_type: api
            schema:
              fields: []
        """)
        _write_yaml(data_sources_dir, "b.yaml", """\
            id: b
            domain: test
            display_name: B
            source_type: file
            schema:
              fields: []
        """)
        reg = DataSourceRegistry()
        count = load_data_source_directory(data_sources_dir, reg)
        assert count == 2
        assert reg.get_spec("a") is not None
        assert reg.get_spec("b") is not None

    def test_skips_underscore_files(self, data_sources_dir: Path):
        _write_yaml(data_sources_dir, "_draft.yaml", """\
            id: draft
            domain: test
            display_name: Draft
            source_type: api
            schema:
              fields: []
        """)
        _write_yaml(data_sources_dir, "real.yaml", """\
            id: real
            domain: test
            display_name: Real
            source_type: api
            schema:
              fields: []
        """)
        reg = DataSourceRegistry()
        count = load_data_source_directory(data_sources_dir, reg)
        assert count == 1
        assert reg.get_spec("draft") is None
        assert reg.get_spec("real") is not None

    def test_nonexistent_directory(self, tmp_path: Path):
        reg = DataSourceRegistry()
        count = load_data_source_directory(tmp_path / "nope", reg)
        assert count == 0

    def test_bad_file_skipped(self, data_sources_dir: Path):
        _write_yaml(data_sources_dir, "bad.yaml", "not: valid: yaml: {{")
        _write_yaml(data_sources_dir, "good.yaml", """\
            id: good
            domain: test
            display_name: Good
            source_type: api
            schema:
              fields: []
        """)
        reg = DataSourceRegistry()
        count = load_data_source_directory(data_sources_dir, reg)
        assert count == 1
