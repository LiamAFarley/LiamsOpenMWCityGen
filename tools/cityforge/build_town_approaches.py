"""CLI to record the V2 rewrite domain and aligned source approaches.

Purpose
-------
Rebuild SiteContext, clip aligned roads to the rewrite-domain disk, and
write ``site_approaches.json`` plus ``approaches_diagnostic.png``.

Inputs
------
Phase 3 survey/fields/census/brief plus ``--centerlines`` (aligned
product directory). ``--out-dir`` must not exist or must be empty.

Outputs
-------
``site_approaches.json`` and an overlay PNG on ``site_topdown.png``.

Pipeline position
-----------------
V2 townlayout Phase 4 rewrite domain / approaches; no patches, walls,
or VTEX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.aligned_roads import load_aligned_network  # noqa: E402
from procgen.townlayout.approaches import (  # noqa: E402
    build_site_approaches,
    write_approaches_diagnostic,
)
from procgen.townlayout.site_context import (  # noqa: E402
    build_site_context,
    resolve_topdown_png,
)
from procgen.townlayout.validate import TownLayoutError  # noqa: E402

DEFAULT_CENTERLINES = (
    "output/mapdata/roads/tamriel_aligned_centerlines_v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build V2 rewrite domain and source approaches")
    parser.add_argument("--survey", required=True)
    parser.add_argument("--fields", required=True)
    parser.add_argument("--census", required=True)
    parser.add_argument("--brief", required=True)
    parser.add_argument("--centerlines", default=DEFAULT_CENTERLINES)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-id", default="c00")
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
        network = load_aligned_network(args.centerlines)
        product = build_site_approaches(
            ctx, network, candidate_id=str(args.candidate_id))
    except TownLayoutError as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1

    json_path = out_dir / "site_approaches.json"
    json_path.write_text(
        json.dumps(product, allow_nan=False) + "\n", encoding="utf-8")

    survey_path = Path(args.survey)
    topdown = resolve_topdown_png(survey_path)
    png_path = out_dir / "approaches_diagnostic.png"
    if topdown is None:
        print("approaches_diagnostic.png skipped: missing site_topdown.png",
              file=sys.stderr)
    else:
        survey = json.loads(survey_path.read_text(encoding="utf-8"))
        write_approaches_diagnostic(
            ctx, product, topdown_path=topdown, survey=survey, out_png=png_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
