"""Run the Cityforge T1.2 houses-only placement core.

Pipeline position
------------------
This CLI is the executable bridge from the accepted T1.1 validation artifact to
the deterministic host-side placement products.  It consumes a plan,
zero-error validation result, hash-pinned site/brief/palette/stamp/centerline
inputs, and one explicit planned/final terrain field.  It writes only
``city_placement.json``, ``land_edit_requests.json``, ``solver_report.json``,
and a hash manifest.  No TES3 JSON, ESP, LAND edit, or render is authored.

Usage
-----
From the workspace root::

    python tools/cityforge/solve_city_placement.py \
      --plan path/to/city_plan.json \
      --validation path/to/city_plan.validation.json \
      --terrain-pass planned \
      --out-dir output/cityforge/phase1/t1_2_placement_fixture

Exit 0 means the placement products were written and contain at least one
accepted house lot.  Any essential input, frame/hash, selector, replay,
matrix-oracle, terrain, or collision-contract failure prints exactly
``FAILURE: cityplace <reason>`` and returns 1 without writing trusted output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen import cityplace  # noqa: E402
from procgen.cityplace_contracts import PlacementConfig, sha256_file  # noqa: E402
from procgen.cityplace_output import build_manifest, write_products  # noqa: E402
from procgen.censusio import write_deterministic  # noqa: E402


CANONICAL_SURVEY = "output/cityforge/sites/falkreath_v1/site_survey.json"
CANONICAL_BRIEF = "output/cityforge/briefs/falkreath_v1/kit_brief.json"
CANONICAL_PALETTE = "output/cityforge/briefs/falkreath_v1/region_palette.json"
CANONICAL_LIBRARIES = (
    "output/cityforge/stamps/karthgad_nord_v1.json",
    "output/cityforge/stamps/markarth_side_stone_v1.json",
)
CANONICAL_CENTERLINES = (
    "output/mapdata/roads/tamriel_aligned_centerlines_v1/"
    "tamriel_aligned_centerlines_v1.json"
)
CANONICAL_FIELD = "output/cityforge/sites/falkreath_v1/survey_fields.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cityforge T1.2 houses-only placement solver")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--validation", required=True,
                        help="zero-error T1.1 validation result; warning-only is allowed")
    parser.add_argument("--site-survey", default=CANONICAL_SURVEY)
    parser.add_argument("--kit-brief", default=CANONICAL_BRIEF)
    parser.add_argument("--region-palette", default=CANONICAL_PALETTE)
    parser.add_argument("--stamp-libraries", nargs="+", default=list(CANONICAL_LIBRARIES))
    parser.add_argument("--centerlines", default=CANONICAL_CENTERLINES)
    parser.add_argument("--terrain-field", default=CANONICAL_FIELD)
    parser.add_argument("--terrain-metadata", default=None)
    parser.add_argument("--terrain-pass", choices=("planned", "final"), required=True)
    parser.add_argument("--planned-placement", default=None,
                        help="required for final pass; prior planned city_placement.json")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = cityplace.solve_city_plan(
            plan_path=args.plan,
            validation_path=args.validation,
            site_survey_path=args.site_survey,
            kit_brief_path=args.kit_brief,
            region_palette_path=args.region_palette,
            stamp_library_paths=args.stamp_libraries,
            centerlines_path=args.centerlines,
            terrain_field_path=args.terrain_field,
            terrain_pass=args.terrain_pass,
            terrain_metadata_path=args.terrain_metadata,
            planned_placement_path=args.planned_placement,
            workspace_root=args.workspace_root,
            config=PlacementConfig(),
        )
        source_hashes = dict(result["source_hashes"])
        source_hashes["t1_1_validation"] = sha256_file(Path(args.validation))
        output_dir = Path(args.out_dir)
        output_hashes = write_products(
            output_dir,
            city_placement=result["city_placement"],
            land_edit_requests=result["land_edit_requests"],
            solver_report=result["solver_report"],
            source_hashes=source_hashes,
        )
        identity = cityplace.result_identity(result)
        manifest = build_manifest(
            source_hashes=source_hashes,
            output_hashes=output_hashes,
            plan_id=str(result["city_placement"]["plan_id"]),
            terrain_pass=args.terrain_pass,
            deterministic_identity=identity,
        )
        manifest_hash = write_deterministic(output_dir / "manifest.json", manifest)
        print(
            f"cityplace PASS accepted={result['city_placement']['counts']['accepted']} "
            f"provisional={result['city_placement']['counts']['provisional']} "
            f"rejected={result['city_placement']['counts']['rejected']} "
            f"source_members={result['solver_report']['gates']['source_replay']['members_checked']} "
            f"oracle37={result['solver_report']['gates']['multi_axis_oracle_37deg']['checked_members']} "
            f"identity={identity} manifest={manifest_hash}"
        )
        return 0
    except Exception as exc:  # the stage protocol is intentionally fail-closed
        print(f"FAILURE: cityplace {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
