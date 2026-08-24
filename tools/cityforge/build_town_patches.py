"""CLI to generate organic V2 macro patches and a diagnostic overlay.

Purpose
-------
Rebuild SiteContext + rewrite domain, run Phase 5A organic Voronoi
patches, write ``macro_patches.json`` and ``patches_diagnostic.png``.

Pipeline position
-----------------
V2 townlayout Phase 5A organic patches; no walls, parcels, or VTEX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.aligned_roads import AlignedRoadsError, load_aligned_network  # noqa: E402
from procgen.townlayout.approaches import (  # noqa: E402
    build_rewrite_domain,
    build_site_approaches,
)
from procgen.townlayout.patches import (  # noqa: E402
    generate_organic_patches,
    write_patches_diagnostic,
)
from procgen.townlayout.site_context import (  # noqa: E402
    build_site_context,
    resolve_topdown_png,
)
from procgen.townlayout.validate import TownLayoutError  # noqa: E402

DEFAULT_CENTERLINES = "output/mapdata/roads/tamriel_aligned_centerlines_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V2 organic macro patches")
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
        domain = build_rewrite_domain(ctx)
        network = load_aligned_network(args.centerlines)
        approaches_product = build_site_approaches(
            ctx, network, candidate_id=str(args.candidate_id))
        approaches = approaches_product.get("approaches", [])
        product = generate_organic_patches(
            ctx, domain, brief,
            master_seed=int(brief["master_seed"]),
            candidate_id=str(args.candidate_id),
            approaches=approaches,
        )
    except TownLayoutError as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1
    except AlignedRoadsError as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1

    (out_dir / "macro_patches.json").write_text(
        json.dumps(product, allow_nan=False) + "\n", encoding="utf-8")
    topdown = resolve_topdown_png(Path(args.survey))
    if topdown is None:
        print("patches_diagnostic.png skipped: missing site_topdown.png",
              file=sys.stderr)
    else:
        survey = json.loads(Path(args.survey).read_text(encoding="utf-8"))
        write_patches_diagnostic(
            ctx, product, topdown_path=topdown, survey=survey,
            out_png=out_dir / ("patches_water_diagnostic.png" if out_dir.name == "stage03_water_cropped_geometry"
                               else "patches_diagnostic.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
