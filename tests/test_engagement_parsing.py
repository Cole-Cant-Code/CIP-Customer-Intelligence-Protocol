"""Tests for cip_protocol.engagement.parsing."""

from __future__ import annotations

import pytest

from cip_protocol.engagement.parsing import (
    clean_numeric_string,
    parse_float,
    parse_int,
    parse_price,
)


class TestCleanNumericString:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("$1,234.56", "1234.56"),
            ("abc", ""),
            ("-42.5", "-42.5"),
            ("  12  ", "12"),
            ("", ""),
        ],
    )
    def test_cases(self, raw, expected):
        assert clean_numeric_string(raw) == expected


class TestParsePrice:
    def test_none(self):
        assert parse_price(None) is None

    def test_bool_true(self):
        assert parse_price(True) is None

    def test_bool_false(self):
        assert parse_price(False) is None

    def test_int(self):
        assert parse_price(42) == 42.0

    def test_float(self):
        assert parse_price(3.14) == 3.14

    def test_string_numeric(self):
        assert parse_price("$29,999.00") == 29999.0

    def test_string_empty(self):
        assert parse_price("") is None

    def test_string_whitespace(self):
        assert parse_price("   ") is None

    def test_string_junk(self):
        assert parse_price("not a number") is None

    def test_string_negative(self):
        assert parse_price("-500") == -500.0

    def test_list_returns_none(self):
        assert parse_price([1, 2]) is None

    def test_zero(self):
        assert parse_price(0) == 0.0


class TestParseInt:
    def test_none(self):
        assert parse_int(None) is None

    def test_bool(self):
        assert parse_int(True) is None

    def test_int(self):
        assert parse_int(42) == 42

    def test_float(self):
        assert parse_int(3.9) == 3

    def test_string(self):
        assert parse_int("100") == 100

    def test_string_with_currency(self):
        assert parse_int("$2,500") == 2500

    def test_empty_string(self):
        assert parse_int("") is None

    def test_junk_string(self):
        assert parse_int("abc") is None


class TestParseFloat:
    def test_none(self):
        assert parse_float(None) is None

    def test_bool(self):
        assert parse_float(True) is None

    def test_int(self):
        assert parse_float(42) == 42.0

    def test_float(self):
        assert parse_float(3.14) == 3.14

    def test_string(self):
        assert parse_float("3.14") == 3.14

    def test_negative_string(self):
        assert parse_float("-97.5") == -97.5

    def test_empty_string(self):
        assert parse_float("") is None

    def test_junk_string(self):
        assert parse_float("abc") is None

    def test_preserves_sign(self):
        assert parse_float("-122.4194") == -122.4194
