"""Author affected townlayout circulation into masterless LAND JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.townlayout.land_authoring import author_from_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circulation", type=Path, required=True)
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--source-plugin", type=Path, required=True)
    parser.add_argument("--source-land-json", type=Path,
                        help="optional remap LAND JSON layered over source-plugin LAND")
    parser.add_argument("--seated-objects", type=Path)
    parser.add_argument("--wall-doc", type=Path)
    parser.add_argument("--grading-policy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        product = author_from_paths(
            args.circulation, args.palette, args.source_plugin,
            source_land_json=args.source_land_json,
            output_path=args.out, seated_objects_path=args.seated_objects,
            wall_doc_path=args.wall_doc,
            grading_policy_path=args.grading_policy)
    except Exception as exc:
        print(f"FAILURE: land_authoring {exc}", file=sys.stderr)
        return 1
    print(json.dumps(product["authoring_evidence"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
