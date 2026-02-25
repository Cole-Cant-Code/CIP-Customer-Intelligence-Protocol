"""CLI handler for ``cip scaffold-health``."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from cip_protocol.health.analysis import analyze_portfolio
from cip_protocol.health.report import format_json, format_table
from cip_protocol.scaffold.loader import load_scaffold_file


def run_scaffold_health(args: Namespace) -> None:
    scaffold_dir = Path(args.scaffold_dir)
    if not scaffold_dir.is_dir():
        print(f"Error: scaffold directory does not exist: {scaffold_dir}", file=sys.stderr)
        sys.exit(1)

    scaffolds = []
    for path in sorted(scaffold_dir.rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        try:
            scaffolds.append(load_scaffold_file(path))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: skipping {path.name}: {exc}", file=sys.stderr)

    if not scaffolds:
        print("No scaffolds loaded.", file=sys.stderr)
        sys.exit(1)

    result = analyze_portfolio(
        scaffolds,
        detection_threshold=args.detection_threshold,
        tension_threshold=args.tension_threshold,
        coherence_divisor=args.coherence_divisor,
    )

    if args.json:
        print(format_json(result))
    else:
        print(format_table(result))
