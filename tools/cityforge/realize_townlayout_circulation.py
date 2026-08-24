"""CLI for terrain-aware townlayout road and surface realization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.townlayout.circulation_realization import realize_from_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--seated-objects", type=Path, required=True)
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--source-roads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        product = realize_from_paths(
            args.layout, args.seated_objects, args.palette, args.survey, args.field,
            args.source_roads)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(product, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    except Exception as exc:
        print(f"FAILURE: circulation_realization {exc}", file=sys.stderr)
        return 1
    print(json.dumps(product["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
