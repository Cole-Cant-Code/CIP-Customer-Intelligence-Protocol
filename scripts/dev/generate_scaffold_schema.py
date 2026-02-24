"""Generate JSON Schema for scaffold YAML authoring and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cip_protocol.scaffold.models import Scaffold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/scaffold.schema.json"),
        help="Path to write the generated JSON schema.",
    )
    args = parser.parse_args()

    schema = Scaffold.model_json_schema()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote scaffold schema to {output_path}")


if __name__ == "__main__":
    main()
