"""Generate the Cityforge D-SITE survey bundle for Falkreath.

Pipeline position::

    read-only source plugins and metadata
        -> procgen.citysite field/mask/LAND-road measurement
        -> output/cityforge/sites/falkreath_v1/{land_roads.json,site_survey.json,survey_fields.npz}
        -> render_site.py (Blender visual checkpoint)

The command is intentionally a host-side driver.  It never writes a source
plugin or a configured data root.  It validates the complete 7 by 7 cell
selection, streams the selected LAND records from the remap ESP and the
target-plus-perimeter records directly from ``tamriel.esm``, checks the
authoritative M0400 marker, and applies only the narrow stale Falkreath row
correction in ``town_grammars.json``.  All actual field, mask, and source
road evidence values are produced by :mod:`procgen.citysite`; this script
only resolves paths, invokes it, and prints measured counts for a run log.
The rejected vector road graph is not a command input.

Usage (from the workspace root)::

    python tools/cityforge/build_site_survey.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.citysite import TARGET_BOUNDS, build_site_survey  # noqa: E402


def _path(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(WORKSPACE))
    parser.add_argument(
        "--output-dir",
        default="output/cityforge/sites/falkreath_v1",
        help="D-SITE bundle directory, relative to --workspace",
    )
    parser.add_argument("--land-source", default="output/falkreath_landscape_texture_remap.esp")
    parser.add_argument("--base-esm", default="tamriel.esm")
    parser.add_argument("--terrain-cells", default="output/terrain_cells.json")
    parser.add_argument(
        "--remap-report",
        default="output/falkreath_landscape_texture_remap_report.json",
        help="independent raw-VTEX count report used as a fail-closed cross-check",
    )
    parser.add_argument("--settlements", default="output/mapdata/settlements.json")
    parser.add_argument("--scatter", default="output/scatter_falkreath_v6.json")
    parser.add_argument("--town-grammars", default="output/town_grammars.json")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    result = build_site_survey(
        workspace=workspace,
        land_source=_path(workspace, args.land_source),
        base_esm=_path(workspace, args.base_esm),
        terrain_cells_path=_path(workspace, args.terrain_cells),
        remap_report_path=_path(workspace, args.remap_report),
        settlements_path=_path(workspace, args.settlements),
        scatter_path=_path(workspace, args.scatter),
        output_dir=_path(workspace, args.output_dir),
        town_grammars_path=_path(workspace, args.town_grammars),
    )
    survey = result["survey"]
    grammar = result["grammar_evidence"]
    print(f"survey_id={survey['survey_id']}")
    print(f"target_bounds={TARGET_BOUNDS} cells={len(survey['cells'])}")
    print(f"water_cells={survey['stats']['water_cells']} water_tiles={survey['stats']['water_tiles']}")
    print(
        f"road_tiles_78={survey['stats']['road_tiles_78']} "
        f"components_8={survey['stats']['road_components_8']} "
        f"components_4_diagnostic={survey['stats']['road_components_4_diagnostic']} "
        f"continuation_spans={survey['stats']['road_continuation_spans']}"
    )
    print(f"scatter_refs={survey['stats']['scatter_refs_measured']} buildable_tiles={survey['stats']['buildable_tiles']}")
    print(f"base_esm_height_max_delta_thu={survey['source_crosscheck']['base_esm']['max_abs_delta_thu']}")
    print(f"survey_json={result['survey_path']}")
    print(f"survey_fields={result['fields_path']}")
    print(f"land_roads={result['land_roads_path']}")
    if grammar is not None:
        print(
            "grammar_patch="
            f"changed={grammar['changed']} replacements={grammar['replacement_count']} "
            f"before={grammar['before_sha256']} after={grammar['after_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
