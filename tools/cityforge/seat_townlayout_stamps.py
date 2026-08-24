"""CLI for door-first terrain seating of a townlayout stamp-object product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.townlayout.stamp_index import DEFAULT_LIBRARIES  # noqa: E402
from procgen.townlayout.terrain_seating import seat_from_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stamp-library", type=Path, action="append",
                        dest="libraries", default=None)
    args = parser.parse_args()
    libraries = tuple(args.libraries) if args.libraries else DEFAULT_LIBRARIES
    try:
        product = seat_from_paths(args.objects, args.survey, args.field, libraries)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(product, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    except Exception as exc:
        print(f"FAILURE: terrain_seating {exc}", file=sys.stderr)
        return 1
    print(json.dumps(product["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
