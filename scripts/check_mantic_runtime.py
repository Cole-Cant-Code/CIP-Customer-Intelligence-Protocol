#!/usr/bin/env python3
"""Guardrail for CI: require mantic when CIP_REQUIRE_MANTIC=1."""

from __future__ import annotations

import os
import sys


def _require_mantic() -> bool:
    return os.getenv("CIP_REQUIRE_MANTIC", "").lower() in {"1", "true", "yes"}


def main() -> None:
    if not _require_mantic():
        print("SKIP: CIP_REQUIRE_MANTIC is not enabled")
        return

    try:
        import mantic_thinking  # noqa: F401
    except ImportError:
        print("ERROR: CIP_REQUIRE_MANTIC=1 but mantic-thinking is not installed")
        sys.exit(1)

    print("OK: mantic-thinking is available")


if __name__ == "__main__":
    main()
