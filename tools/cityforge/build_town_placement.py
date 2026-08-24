"""CLI to seat stamps on V2 townlayout parcels.

Purpose
-------
Consume Phase 17 ``frontage.json`` plus kit_brief / D-STAMP v2 libraries
and write ``placement.json`` plus a hull overlay PNG.

Pipeline position
-----------------
V2 townlayout Phase 18; no D-PLAN/VTEX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.place import (  # noqa: E402
    place_stamps,
    write_placement_diagnostic,
)
from procgen.townlayout.site_context import (  # noqa: E402
    build_site_context,
    resolve_topdown_png,
)
from procgen.townlayout.stamp_index import (  # noqa: E402
    DEFAULT_LIBRARIES,
    build_stamp_index,
    load_stamp_libraries,
)
from procgen.townlayout.validate import TownLayoutError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Place stamps on V2 parcels")
    parser.add_argument("--frontage", required=True)
    parser.add_argument("--kit-brief", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--survey", default="")
    parser.add_argument("--fields", default="")
    parser.add_argument("--census", default="")
    parser.add_argument("--brief", default="")
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
    try:
        product = json.loads(Path(args.frontage).read_text(encoding="utf-8"))
        kit = json.loads(Path(args.kit_brief).read_text(encoding="utf-8"))
        libs = load_stamp_libraries(DEFAULT_LIBRARIES)
        index = build_stamp_index(kit, libs)
        ctx = None
        if args.survey and args.fields and args.census and args.brief:
            brief = json.loads(Path(args.brief).read_text(encoding="utf-8"))
            ctx = build_site_context(
                survey_json=Path(args.survey),
                fields_npz=Path(args.fields),
                census_json=Path(args.census),
                town_brief=brief,
            )
        product = place_stamps(product, index, libs, ctx=ctx)
    except TownLayoutError as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1
    (out_dir / "placement.json").write_text(
        json.dumps(product, allow_nan=False) + "\n", encoding="utf-8")
    if args.survey:
        topdown = resolve_topdown_png(Path(args.survey))
        if topdown is not None and ctx is not None:
            survey = json.loads(Path(args.survey).read_text(encoding="utf-8"))
            write_placement_diagnostic(
                ctx, product, topdown_path=topdown, survey=survey,
                out_png=out_dir / "placement_diagnostic.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
