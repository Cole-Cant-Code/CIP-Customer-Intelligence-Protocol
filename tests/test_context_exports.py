"""Tests for context export extraction."""

from conftest import make_test_scaffold

from cip_protocol.llm.response import extract_context_exports
from cip_protocol.scaffold.models import ContextField


class TestContextExports:
    def test_export_from_data_context(self):
        scaffold = make_test_scaffold()
        scaffold.context_exports = [
            ContextField(
                field_name="total_score",
                type="number",
                description="Overall score",
            )
        ]
        exports = extract_context_exports(
            content="Some LLM output.",
            scaffold=scaffold,
            data_context={"total_score": 42.0},
        )
        assert exports["total_score"] == 42.0

    def test_export_from_content_number(self):
        scaffold = make_test_scaffold()
        scaffold.context_exports = [
            ContextField(
                field_name="total_amount",
                type="currency",
                description="Total amount",
            )
        ]
        exports = extract_context_exports(
            content="The total amount: $1,234.56 was calculated.",
            scaffold=scaffold,
            data_context={},
        )
        assert exports.get("total_amount") == 1234.56

    def test_export_from_content_string(self):
        scaffold = make_test_scaffold()
        scaffold.context_exports = [
            ContextField(
                field_name="risk_level",
                type="string",
                description="Risk assessment",
            )
        ]
        exports = extract_context_exports(
            content="The risk level: moderate based on analysis.",
            scaffold=scaffold,
            data_context={},
        )
        assert exports.get("risk_level") == "moderate based on analysis"

    def test_no_exports_defined(self):
        scaffold = make_test_scaffold()
        exports = extract_context_exports(
            content="Some output.",
            scaffold=scaffold,
            data_context={"key": "value"},
        )
        assert exports == {}

    def test_data_context_takes_priority(self):
        scaffold = make_test_scaffold()
        scaffold.context_exports = [
            ContextField(
                field_name="score",
                type="number",
                description="Score",
            )
        ]
        exports = extract_context_exports(
            content="The score: 99",
            scaffold=scaffold,
            data_context={"score": 42},
        )
        # data_context value takes priority over content extraction
        assert exports["score"] == 42

    def test_export_handles_regex_metacharacters_in_field_name(self):
        scaffold = make_test_scaffold()
        scaffold.context_exports = [
            ContextField(
                field_name="risk(",
                type="string",
                description="Risk signal",
            )
        ]
        exports = extract_context_exports(
            content="The risk(: high confidence based on current inputs.",
            scaffold=scaffold,
            data_context={},
        )
        assert exports.get("risk(") == "high confidence based on current inputs"
