"""CLI to build the V2 townlayout stamp capability index.

Purpose
-------
Load kit_brief eligibility plus D-STAMP v2 libraries and write
``stamp_index.json``.  Castle Barracks is excluded.

Pipeline position
-----------------
V2 townlayout Phase 15; no parcels/placement/VTEX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.stamp_index import (  # noqa: E402
    DEFAULT_LIBRARIES,
    build_stamp_index,
    load_stamp_libraries,
)
from procgen.townlayout.validate import TownLayoutError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V2 stamp capability index")
    parser.add_argument("--kit-brief", required=True)
    parser.add_argument("--library", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not out_dir.is_dir() or any(out_dir.iterdir()):
            print("FAILURE: townlayout out-dir not empty", file=sys.stderr)
            return 1
    else:
        out_dir.mkdir(parents=True)
    lib_paths = [Path(p) for p in args.library] if args.library else list(
        DEFAULT_LIBRARIES)
    try:
        kit = json.loads(Path(args.kit_brief).read_text(encoding="utf-8"))
        libraries = load_stamp_libraries(lib_paths)
        product = build_stamp_index(kit, libraries)
    except TownLayoutError as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1
    (out_dir / "stamp_index.json").write_text(
        json.dumps(product, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
