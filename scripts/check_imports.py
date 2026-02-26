#!/usr/bin/env python3
"""CI enforcement: warn on mantic_thinking imports outside mantic_adapter.py.

Exit 0 (warning only) during stabilization.  Promote to exit 1 after
two release cycles with no fallback anomalies.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ALLOWED_FILES = {"mantic_adapter.py"}
SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "cip_protocol"


def check() -> list[str]:
    violations: list[str] = []
    for py_file in SRC_DIR.rglob("*.py"):
        if py_file.name in ALLOWED_FILES:
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("mantic_thinking"):
                        rel = py_file.relative_to(SRC_DIR)
                        violations.append(f"{rel}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("mantic_thinking"):
                    rel = py_file.relative_to(SRC_DIR)
                    violations.append(f"{rel}:{node.lineno}: from {node.module}")
    return violations


def main() -> None:
    violations = check()
    if violations:
        print("WARNING: mantic_thinking imports found outside mantic_adapter.py:")
        for v in violations:
            print(f"  {v}")
        # Exit 0 for now (warning). Change to sys.exit(1) after stabilization.
        sys.exit(0)
    else:
        print("OK: no mantic_thinking imports outside mantic_adapter.py")


if __name__ == "__main__":
    main()
