"""Generic data parsing utilities for engagement pipelines."""

from __future__ import annotations

from typing import Any


def clean_numeric_string(raw: str) -> str:
    """Keep only digits, ``'.'``, and ``'-'``."""
    return "".join(c for c in raw if c.isdigit() or c in {".", "-"})


def parse_price(value: Any) -> float | None:
    """Best-effort price parsing.  Returns ``None`` for unparseable input."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        cleaned = clean_numeric_string(stripped)
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def parse_int(value: Any) -> int | None:
    """Best-effort integer parsing.  Returns ``None`` for unparseable input."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        parsed = parse_price(stripped)
        if parsed is None:
            return None
        return int(parsed)
    return None


def parse_float(value: Any) -> float | None:
    """Best-effort float parsing that preserves sign (for lat/lng)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None
