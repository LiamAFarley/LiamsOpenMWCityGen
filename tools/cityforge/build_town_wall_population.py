"""Build the first wall-aware population checkpoint and full-town render."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.checkpoint import read_checkpoint, write_checkpoint  # noqa: E402
from procgen.townlayout.place import write_placement_diagnostic  # noqa: E402
from procgen.townlayout.site_context import build_site_context, resolve_topdown_png  # noqa: E402
from procgen.townlayout.stamp_index import DEFAULT_LIBRARIES, build_stamp_index, load_stamp_libraries  # noqa: E402
from procgen.townlayout.validate import TownLayoutError  # noqa: E402
from procgen.townlayout.wall_population import (  # noqa: E402
    populate_wall_front_rows,
    prepare_wall_population,
)
from procgen.visual_planner_eligibility import build_eligibility_policy  # noqa: E402

DEFAULT_PALETTE = Path(
    "output/settlement-splits/markarth-side-v2/"
    "final-markarth-extraction-2026-08-10-library/"
    "stamp_palette_v1/catalog.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Populate the accepted wall-aware town")
    parser.add_argument("--input", required=True)
    parser.add_argument("--kit-brief", required=True)
    parser.add_argument("--brief", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        print("FAILURE: R5 output directory is not empty", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = read_checkpoint(args.input, expected_stages=("r2c_minor_roads",))
        brief = json.loads(Path(args.brief).read_text(encoding="utf-8"))
        prepared = prepare_wall_population(
            source, development_policy=brief.get("development_policy"),
            has_outskirts=brief.get("has_outskirts", True),
            has_inner_wall=brief.get("has_inner_wall", None))
        kit = json.loads(Path(args.kit_brief).read_text(encoding="utf-8"))
        libraries = load_stamp_libraries(DEFAULT_LIBRARIES)
        stamp_index = build_stamp_index(kit, libraries)
        policy = build_eligibility_policy(DEFAULT_LIBRARIES, palette_path=DEFAULT_PALETTE)
        stamp_index["stamps"] = [row for row in stamp_index["stamps"]
                                  if row["stamp_id"] in policy.accepted_stamp_ids]
        survey = Path(source["identities"]["survey"]["path"])
        fields = Path(source["identities"]["fields"]["path"])
        census = Path(source["identities"]["census"]["path"])
        ctx = build_site_context(survey_json=survey, fields_npz=fields,
                                 census_json=census, town_brief=brief)
        product = populate_wall_front_rows(
            prepared, stamp_index, libraries,
            master_seed=int(brief["master_seed"]),
            candidate_id=source.get("candidate_id", "c00"))
        product["stage_id"] = "r5_wall_front_rows"
        product["preceding_checkpoint"] = str(Path(args.input).resolve())
        product["eligibility_policy"] = {
            "accepted_stamp_count": len(policy.accepted_stamp_ids),
            "rejected_stamp_ids": sorted(policy.rejected_stamp_ids),
            "metadata_hashes": policy.metadata_hashes,
        }
        write_checkpoint(product, out_dir / "wall_front_rows.json")
        if product.get("generated_stamps"):
            from procgen.townlayout.fk_house_adapter import GENERATED_LIBRARY_ID  # noqa: E402
            stamps = [product["generated_stamps"][sid] for sid in sorted(product["generated_stamps"])]
            (out_dir / "generated_fk_stamps.json").write_text(
                json.dumps(
                    {
                        "library_id": GENERATED_LIBRARY_ID,
                        "schema_version": 1,
                        "generated_by": "fk_house_adapter slice1b",
                        "stamps": stamps,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        topdown = resolve_topdown_png(survey)
        if topdown is None:
            raise TownLayoutError("R5 terrain render source is missing")
        write_placement_diagnostic(
            ctx, product, topdown_path=topdown,
            survey=json.loads(survey.read_text(encoding="utf-8")),
            out_png=out_dir / "wall_front_rows_terrain.png")
    except Exception as exc:
        print(f"FAILURE: R5 {exc}", file=sys.stderr)
        return 1
    metrics = product["population_metrics"]
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
