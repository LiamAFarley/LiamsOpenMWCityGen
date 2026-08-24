"""TownBrief / CityLayout validation CLI for V2 townlayout.

Purpose
-------
Load one TownBrief or CityLayout JSON, run the Phase 1 fail-closed
validator, and write a structured issues file.

Inputs
------
``--brief`` or ``--layout`` JSON path (exactly one), plus required ``--out``.
``--emit-schema [PATH]`` writes the machine-readable JSON Schema.

Outputs
-------
Issues JSON ``{error_count, warning_count, issues, status}``.  Exit 0 when
``error_count == 0`` (warnings allowed); exit 1 when any error exists.
On errors, prints ``FAILURE: townlayout <first error message>`` to stderr.

Pipeline position
-----------------
V2 townlayout Phase 1 contracts; no generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout import (  # noqa: E402
    validate_city_layout,
    validate_town_brief,
)
from procgen.townlayout.validate import validate_fortification_product  # noqa: E402
from procgen.townlayout.schema import emit_json_schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = ROOT / "src" / "procgen" / "schemas" / "town_layout_schema_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict V2 townlayout TownBrief / CityLayout validator")
    parser.add_argument("--layout", default=None,
                        help="path to city_layout.json")
    parser.add_argument("--brief", default=None,
                        help="path to town_brief.json")
    parser.add_argument("--out", default=None,
                        help="issues JSON output path (required unless "
                             "--emit-schema)")
    parser.add_argument(
        "--emit-schema", nargs="?", const=str(DEFAULT_SCHEMA_PATH),
        default=None, metavar="PATH",
        help="write town_layout_schema_v1.json and exit "
             f"(default path: {DEFAULT_SCHEMA_PATH})",
    )
    return parser


def _issues_payload(issues: list) -> dict:
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]
    return {
        "error_count": len(errors),
        "warning_count": len(warnings),
        "issues": issues,
        "status": "ok" if not errors else "invalid",
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.emit_schema is not None:
        out = Path(args.emit_schema)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = emit_json_schema()
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"wrote schema {out}")
        return 0

    if (args.layout is None) == (args.brief is None):
        parser.error("exactly one of --layout or --brief is required")
    if args.out is None:
        parser.error("--out is required")

    src = Path(args.layout) if args.layout else Path(args.brief)
    try:
        document = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1

    if args.layout and "wall" in document and "frame" not in document:
        _doc, issues = validate_fortification_product(document)
    elif args.layout:
        _doc, issues = validate_city_layout(document)
    else:
        _doc, issues = validate_town_brief(document)

    payload = _issues_payload(issues)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if payload["error_count"]:
        first = next((i["message"] for i in issues
                      if i.get("severity") == "error"), "invalid")
        print(f"FAILURE: townlayout {first}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
