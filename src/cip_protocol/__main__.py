"""CLI entry point: python -m cip_protocol <command>."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cip",
        description="CIP Protocol CLI",
    )
    sub = parser.add_subparsers(dest="command")

    pg = sub.add_parser("playground", help="Interactive CIP playground")
    pg.add_argument("--scaffold-dir", required=True, help="Path to scaffold YAML directory")
    pg.add_argument("--provider", default="mock", help="LLM provider (mock, anthropic, openai)")
    pg.add_argument("--api-key", default="", help="API key for the provider")
    pg.add_argument("--model", default="", help="Model name override")
    pg.add_argument("--domain", default="playground", help="Domain config name")
    pg.add_argument("--default-scaffold", default="", help="Default scaffold ID for fallback")

    args = parser.parse_args(argv)

    if args.command == "playground":
        from cip_protocol.cli.playground import run_playground
        run_playground(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
