"""CLI to inset buildable blocks for the V2 Gate A street map.

Purpose
-------
Rebuild through streets, run intersection cleanup, ward assignment, and
road-corridor insets. Writes ``blocks.json`` and ``blocks_diagnostic.png``.

Pipeline position
-----------------
V2 townlayout Phases 11–13; no parcels/VTEX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.aligned_roads import AlignedRoadsError, load_aligned_network  # noqa: E402
from procgen.townlayout.anchors import place_anchors  # noqa: E402
from procgen.townlayout.approaches import (  # noqa: E402
    build_rewrite_domain,
    build_site_approaches,
)
from procgen.townlayout.blocks import inset_blocks, write_blocks_diagnostic  # noqa: E402
from procgen.townlayout.domain import grow_city_domain  # noqa: E402
from procgen.townlayout.graph import build_topology_graph  # noqa: E402
from procgen.townlayout.intersections import cleanup_intersections  # noqa: E402
from procgen.townlayout.patches import generate_organic_patches  # noqa: E402
from procgen.townlayout.site_context import (  # noqa: E402
    build_site_context,
    resolve_topdown_png,
)
from procgen.townlayout.streets import assign_streets  # noqa: E402
from procgen.townlayout.validate import TownLayoutError  # noqa: E402
from procgen.townlayout.walls import build_walls_and_gates  # noqa: E402
from procgen.townlayout.wards import assign_wards  # noqa: E402

DEFAULT_CENTERLINES = "output/mapdata/roads/tamriel_aligned_centerlines_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V2 inset blocks (Gate A)")
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
        cid = str(args.candidate_id)
        domain = build_rewrite_domain(ctx)
        network = load_aligned_network(args.centerlines)
        approaches = build_site_approaches(
            ctx, network, candidate_id=cid).get("approaches", [])
        patches = generate_organic_patches(
            ctx, domain, brief, master_seed=int(brief["master_seed"]),
            candidate_id=cid, approaches=approaches)
        grown = grow_city_domain(ctx, patches, brief, approaches=approaches)
        anchored = place_anchors(
            ctx, grown, brief, approaches=approaches, candidate_id=cid)
        walled = build_walls_and_gates(
            ctx, anchored, brief, approaches=approaches, candidate_id=cid)
        graphed = build_topology_graph(ctx, walled, candidate_id=cid)
        streeted = assign_streets(
            ctx, graphed, candidate_id=cid, approaches=approaches)
        cleaned = cleanup_intersections(streeted)
        warded = assign_wards(cleaned, brief, candidate_id=cid)
        product = inset_blocks(warded, water_polygons=list(ctx.water_polygons()))
    except TownLayoutError as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1
    except AlignedRoadsError as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAILURE: townlayout {exc}", file=sys.stderr)
        return 1

    (out_dir / "blocks.json").write_text(
        json.dumps(product, allow_nan=False) + "\n", encoding="utf-8")
    topdown = resolve_topdown_png(Path(args.survey))
    if topdown is None:
        print("blocks_diagnostic.png skipped: missing site_topdown.png",
              file=sys.stderr)
    else:
        survey = json.loads(Path(args.survey).read_text(encoding="utf-8"))
        write_blocks_diagnostic(
            ctx, product, topdown_path=topdown, survey=survey,
            out_png=out_dir / ("blocks_water_diagnostic.png" if out_dir.name == "stage03_water_cropped_geometry"
                               else "blocks_diagnostic.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
