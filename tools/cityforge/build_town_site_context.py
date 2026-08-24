"""CLI to build a V2 townlayout SiteContext product and diagnostic PNG.

Purpose
-------
Load D-SITE survey + NPZ fields + D-BRIEF census + TownBrief, write
``site_context.json`` and ``site_context_diagnostic.png``.

Inputs
------
``--survey``, ``--fields``, ``--census``, ``--brief``, ``--out-dir``
(must not exist or must be empty).

Outputs
-------
``site_context.json`` (Phase 1 site_context schema) and an overlay PNG
on ``site_topdown.png`` (skipped with a stderr note if the topdown is
missing).

Pipeline position
-----------------
V2 townlayout Phase 3 SiteContext; no patches, walls, or roads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.site_context import (  # noqa: E402
    build_site_context,
    resolve_topdown_png,
    write_site_context_diagnostic,
)
from procgen.townlayout.validate import TownLayoutError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build V2 townlayout SiteContext suitability product")
    parser.add_argument("--survey", required=True)
    parser.add_argument("--fields", required=True)
    parser.add_argument("--census", required=True)
    parser.add_argument("--brief", required=True)
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

    try:
        brief = json.loads(Path(args.brief).read_text(encoding="utf-8"))
        ctx = build_site_context(
            survey_json=Path(args.survey),
            fields_npz=Path(args.fields),
            census_json=Path(args.census),
            town_brief=brief,
        )
    except TownLayoutError as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1

    json_path = out_dir / "site_context.json"
    json_path.write_text(
        json.dumps(ctx.to_json_dict(), allow_nan=False) + "\n",
        encoding="utf-8",
    )

    survey_path = Path(args.survey)
    topdown = resolve_topdown_png(survey_path)
    png_path = out_dir / "site_context_diagnostic.png"
    if topdown is None:
        print("site_context_diagnostic.png skipped: missing site_topdown.png",
              file=sys.stderr)
    else:
        survey = json.loads(survey_path.read_text(encoding="utf-8"))
        write_site_context_diagnostic(
            ctx, topdown_path=topdown, survey=survey, out_png=png_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
