"""Tests for mapping.utils - core data transformation functions."""

import math

import pandas as pd
import pytest

from mapping.utils import (
    detect_list_delimiters,
    is_missing_value,
    normalize_header,
    normalize_raw_value,
    parse_numeric_series,
    parse_numeric_value,
    safe_json_loads,
    sha256_hex,
    sha256_text,
)


class TestIsMissingValue:
    """Tests for is_missing_value function."""

    def test_none_is_missing(self):
        assert is_missing_value(None) is True

    def test_nan_float_is_missing(self):
        assert is_missing_value(float("nan")) is True

    def test_pandas_nan_is_missing(self):
        # pd.NA isn't detected directly (not a float), but np.nan is
        import numpy as np
        assert is_missing_value(np.nan) is True

    def test_empty_string_is_missing(self):
        assert is_missing_value("") is True
        assert is_missing_value("   ") is True

    def test_na_variants_are_missing(self):
        assert is_missing_value("nan") is True
        assert is_missing_value("NaN") is True
        assert is_missing_value("#N/A") is True
        assert is_missing_value("n/a") is True
        assert is_missing_value("NA") is True
        assert is_missing_value("none") is True
        assert is_missing_value("None") is True
        assert is_missing_value("null") is True
        assert is_missing_value("NULL") is True

    def test_valid_values_are_not_missing(self):
        assert is_missing_value("hello") is False
        assert is_missing_value("0") is False
        assert is_missing_value(0) is False
        assert is_missing_value(1) is False
        assert is_missing_value("false") is False


class TestNormalizeRawValue:
    """Tests for normalize_raw_value function."""

    def test_returns_none_for_missing_values(self):
        assert normalize_raw_value(None) is None
        assert normalize_raw_value("") is None
        assert normalize_raw_value("nan") is None
        assert normalize_raw_value("#N/A") is None

    def test_strips_whitespace(self):
        assert normalize_raw_value("  hello  ") == "hello"
        assert normalize_raw_value("\tworld\n") == "world"

    def test_converts_to_string(self):
        assert normalize_raw_value(123) == "123"
        assert normalize_raw_value(45.67) == "45.67"


class TestNormalizeHeader:
    """Tests for normalize_header function."""

    def test_converts_to_snake_case(self):
        assert normalize_header("Column Name") == "column_name"
        assert normalize_header("First-Last") == "first_last"

    def test_removes_special_characters(self):
        assert normalize_header("Price ($)") == "price"
        assert normalize_header("Revenue %") == "revenue"

    def test_strips_leading_trailing_underscores(self):
        assert normalize_header("  _test_  ") == "test"
        assert normalize_header("___hello___") == "hello"

    def test_handles_mixed_case(self):
        assert normalize_header("CamelCaseHeader") == "camelcaseheader"


class TestSha256Functions:
    """Tests for SHA256 hashing functions."""

    def test_sha256_hex_deterministic(self):
        data = b"test data"
        hash1 = sha256_hex(data)
        hash2 = sha256_hex(data)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex chars

    def test_sha256_text_deterministic(self):
        text = "test string"
        hash1 = sha256_text(text)
        hash2 = sha256_text(text)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_different_inputs_different_hashes(self):
        assert sha256_text("hello") != sha256_text("world")


class TestParseNumericValue:
    """Tests for parse_numeric_value function."""

    def test_returns_none_for_missing(self):
        assert parse_numeric_value(None) is None
        assert parse_numeric_value("") is None
        assert parse_numeric_value("nan") is None

    def test_parses_integers(self):
        assert parse_numeric_value(42) == 42.0
        assert parse_numeric_value("42") == 42.0

    def test_parses_floats(self):
        assert parse_numeric_value(3.14) == 3.14
        assert parse_numeric_value("3.14") == 3.14

    def test_handles_european_comma_decimal(self):
        assert parse_numeric_value("3,14") == 3.14

    def test_extracts_number_from_text(self):
        assert parse_numeric_value("$100") == 100.0
        assert parse_numeric_value("€50.00") == 50.0
        assert parse_numeric_value("about 75 units") == 75.0

    def test_averages_range(self):
        # "10 - 20" with spaces parses as ["10", "20"] and averages to 15
        result = parse_numeric_value("10 - 20")
        assert result == 15.0
        # Note: "10-20" without spaces finds ["10", "-20"] → -5 (regex matches -20)

    def test_negative_numbers(self):
        assert parse_numeric_value(-5) == -5.0
        assert parse_numeric_value("-5") == -5.0

    def test_no_number_returns_none(self):
        assert parse_numeric_value("no numbers here") is None


class TestParseNumericSeries:
    """Tests for parse_numeric_series function."""

    def test_parses_list_of_values(self):
        values = ["1", "2.5", "3"]
        parsed, failures = parse_numeric_series(values)
        assert parsed == [1.0, 2.5, 3.0]
        assert failures == 0

    def test_handles_missing_values(self):
        values = ["1", None, "3"]
        parsed, failures = parse_numeric_series(values)
        assert parsed == [1.0, None, 3.0]
        assert failures == 0  # None values don't count as failures

    def test_counts_parse_failures(self):
        values = ["1", "not a number", "3"]
        parsed, failures = parse_numeric_series(values)
        assert parsed == [1.0, None, 3.0]
        assert failures == 1


class TestDetectListDelimiters:
    """Tests for detect_list_delimiters function."""

    def test_detects_comma_delimiter(self):
        values = ["a, b, c", "x, y", "single"]
        delimiters = detect_list_delimiters(values, min_fraction=0.5)
        assert "," in delimiters

    def test_detects_semicolon_delimiter(self):
        values = ["a; b; c", "x; y; z", "m; n"]
        delimiters = detect_list_delimiters(values, min_fraction=0.5)
        assert ";" in delimiters

    def test_detects_pipe_delimiter(self):
        values = ["a|b|c", "x|y", "m|n|o"]
        delimiters = detect_list_delimiters(values, min_fraction=0.5)
        assert "|" in delimiters

    def test_returns_empty_when_no_delimiters(self):
        values = ["single", "words", "only"]
        delimiters = detect_list_delimiters(values, min_fraction=0.15)
        assert delimiters == []

    def test_returns_empty_for_empty_input(self):
        assert detect_list_delimiters([]) == []

    def test_respects_min_fraction(self):
        # Only 1 out of 10 has comma
        values = ["a, b"] + ["single"] * 9
        delimiters = detect_list_delimiters(values, min_fraction=0.15)
        assert "," not in delimiters


class TestSafeJsonLoads:
    """Tests for safe_json_loads function."""

    def test_parses_plain_json(self):
        result = safe_json_loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_array(self):
        result = safe_json_loads('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_strips_markdown_code_blocks(self):
        text = '```json\n{"key": "value"}\n```'
        result = safe_json_loads(text)
        assert result == {"key": "value"}

    def test_strips_code_blocks_without_language(self):
        text = '```\n{"key": "value"}\n```'
        result = safe_json_loads(text)
        assert result == {"key": "value"}

    def test_handles_whitespace(self):
        result = safe_json_loads('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_raises_on_invalid_json(self):
        with pytest.raises(Exception):
            safe_json_loads("not valid json")
