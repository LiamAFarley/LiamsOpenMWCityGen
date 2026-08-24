#!/usr/bin/env python3
"""Unified settlement wilderness CLI: one run emits the town and groundcover ESPs.

Pipeline position: this is the Stage-5 entry point of the approved plan
``.opencode/runs/cityforge-scatter-groundcover-integration/plan.md``.  After
town realization, one invocation produces:

* ``<name>_town.esp`` — buildings + edited LAND + local LTEX + scatter
  (loaded as ``content=``; masterless, ``masters: []``),
* ``<name>_groundcover.esp`` — groundcover only (loaded as ``groundcover=``
  in openmw.cfg).

One script emits both so seeds, bounds, and exclusions cannot drift apart.

Internal sequence
-----------------
1. ``build_clearing()`` (Stage 1) -> ``settlement_clearing.json``
2. scatter generation (clearing + edited LAND JSON) -> ``<name>_scatter.json``
3. groundcover generation (clearing + scatter exclusions + edited LAND) ->
   ``<name>_groundcover.json``
4. town ESP assembly (``procgen.town_author.build_town_plugin``) -> tes3conv ->
   ``<name>_town.esp``
5. groundcover ESP via the existing ``groundcover_generate.author_plugin`` ->
   ``<name>_groundcover.esp``
6. ``run_summary.json``

The edited land is consumed as the city generation's tes3conv JSON (not an
ESP) via ``procgen.tes3json.land_records_from_json``.

Guards: the output directory must be fresh (non-existent or empty); protected
roots (``C:\\Modding``, configured data roots, plugin files) are refused.
Both plugins are masterless per workspace rules #24/#9.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.espland import load_land, load_ltex  # noqa: E402
from procgen.groundcover_generate import (  # noqa: E402
    author_plugin as author_groundcover_plugin,
    config_from_mapping as groundcover_config_from_mapping,
    generate_groundcover_document_with_land,
)
from procgen.groundcover_ini import parse_ini  # noqa: E402
from procgen.clearing_index import build_clearing_index  # noqa: E402
from procgen.scatter_generate import (  # noqa: E402
    TARGET_BOUNDS,
    GenerationConfig,
    generate_scatter_document,
)
from procgen.settlement_clearing import build_clearing, dump_clearing  # noqa: E402
from procgen.tes3json import (  # noqa: E402
    land_records_from_json,
    validate as validate_tes3json,
    write_json as write_tes3json,
)
from procgen.town_author import build_town_plugin  # noqa: E402
from procgen.wall_scatter import filter_scatter_document  # noqa: E402

DEFAULT_TES3CONV = WORKSPACE / "tes3conv-master" / "tes3conv.exe"
DEFAULT_TERRAIN = WORKSPACE / "tamriel.esm"
PROTECTED_ROOTS = (Path(r"C:\Modding"),)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"json-load {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"input {label} is missing: {path}")


def _check_out_dir(out_dir: Path) -> None:
    """Fresh (non-existent or empty) out dir; refuse protected roots."""
    resolved = out_dir.resolve()
    for root in PROTECTED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            pass
        else:
            raise RuntimeError(f"output dir is inside protected root {root}")
    if out_dir.exists():
        entries = list(out_dir.iterdir())
        if entries:
            raise RuntimeError(f"output dir must be fresh, found {len(entries)} entries: {out_dir}")


def _rebuild_reused_scatter(scatter: Mapping[str, Any], clearing_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the current clearing to a reused scatter product and rebuild counts."""

    clearing = build_clearing_index(clearing_doc)
    rejected = Counter()
    kept_refs: list[dict[str, Any]] = []
    filtered_cells = []
    for cell in (scatter.get("density") or {}).get("cells") or []:
        retained = []
        for raw_ref in cell.get("refs") or []:
            ref = dict(raw_ref)
            category = str(ref.get("category") or "")
            position = ref.get("position_gu")
            if not isinstance(position, list) or len(position) < 2:
                raise RuntimeError(f"reused scatter ref {ref.get('ref_id')} has no position_gu")
            x, y = float(position[0]), float(position[1])
            reason = None
            if category == "flora":
                if clearing.blocks_point(x, y):
                    reason = "flora_clearing_blocked"
            elif category in {"rocks", "cliff"}:
                aabb = (ref.get("bbox") or {}).get("world_aabb_gu") or {}
                minimum = aabb.get("min")
                maximum = aabb.get("max")
                has_aabb = (
                    isinstance(minimum, list) and len(minimum) >= 2
                    and isinstance(maximum, list) and len(maximum) >= 2
                )
                if has_aabb:
                    min_x, min_y = float(minimum[0]), float(minimum[1])
                    max_x, max_y = float(maximum[0]), float(maximum[1])
                    if clearing.blocks_city_domain_aabb(min_x, min_y, max_x, max_y):
                        reason = "city_domain_rocks_banned"
                    elif clearing.blocks_aabb(min_x, min_y, max_x, max_y):
                        reason = "rock_clearing_blocked"
                else:
                    if clearing.in_city_domain_point(x, y):
                        reason = "city_domain_rocks_banned"
                    elif clearing.blocks_point(x, y):
                        reason = "rock_clearing_blocked"
            if reason is not None:
                rejected[reason] += 1
                continue
            retained.append(ref)
            kept_refs.append(ref)
        cell_out = dict(cell)
        cell_out["refs"] = retained
        by_category = Counter(str(ref.get("category") or "unknown") for ref in retained)
        by_pass = Counter(str(ref.get("pass") or "") for ref in retained if ref.get("pass") is not None)
        stats = dict(cell.get("stats") or {})
        stats.update({
            "ref_count": len(retained),
            "flora_refs": by_category.get("flora", 0),
            "rock_refs": by_category.get("rocks", 0),
            "main_rock_refs": by_category.get("rocks", 0),
            "cliff_refs": by_category.get("cliff", 0),
            "clearing_refs": sum(1 for ref in retained if (ref.get("clearing") or {}).get("is_clearing")),
            "tree_refs": sum(1 for ref in retained if str(ref.get("flora_role") or "") == "tree"),
            "undergrowth_refs": sum(1 for ref in retained if str(ref.get("flora_role") or "") == "undergrowth"),
            "stacker_refs": sum(1 for ref in retained if (ref.get("stacking") or {}).get("enabled")),
        })
        cell_out["stats"] = stats
        filtered_cells.append(cell_out)

    by_category = Counter(str(ref.get("category") or "unknown") for ref in kept_refs)
    by_pass = Counter(str(ref.get("pass") or "") for ref in kept_refs if ref.get("pass") is not None)
    placement_stats = dict(scatter.get("placement_stats") or {})
    placement_stats.update({
        "total_refs": len(kept_refs),
        "by_category": dict(sorted(by_category.items())),
        "by_pass": dict(sorted(by_pass.items())),
        "unique_meshes": len({str(ref.get("mesh") or "") for ref in kept_refs}),
        "target_flora_refs": by_category.get("flora", 0),
        "target_rock_refs": by_category.get("rocks", 0),
        "target_cliff_refs": by_category.get("cliff", 0),
        "target_tree_refs": sum(1 for ref in kept_refs if str(ref.get("flora_role") or "") == "tree"),
        "target_undergrowth_refs": sum(1 for ref in kept_refs if str(ref.get("flora_role") or "") == "undergrowth"),
        "stackers": sum(1 for ref in kept_refs if (ref.get("stacking") or {}).get("enabled")),
    })
    output = dict(scatter)
    density = dict(scatter.get("density") or {})
    density["cells"] = filtered_cells
    output["density"] = density
    output["placement_stats"] = placement_stats
    city_clearing = dict(scatter.get("city_clearing") or {})
    city_clearing.update({
        "enabled": True,
        "reused_scatter_filter": True,
        "flora_clearing_blocked": int(rejected.get("flora_clearing_blocked", 0)),
        "rock_clearing_blocked": int(rejected.get("rock_clearing_blocked", 0)),
        "city_domain_rocks_banned": int(rejected.get("city_domain_rocks_banned", 0)),
        "reused_rejection_count": int(sum(rejected.values())),
        "reused_rejections": dict(sorted(rejected.items())),
    })
    output["city_clearing"] = city_clearing
    return output


def _load_edited_land(
    base_terrain: Path,
    land_source_json: Path,
    edited_land: Path,
) -> Mapping[tuple[int, int], Any]:
    land_records = load_land(base_terrain)
    land_records = {
        **land_records,
        **land_records_from_json(_read_json(land_source_json)),
    }
    edited = land_records_from_json(_read_json(edited_land))
    merged = dict(land_records)
    merged.update(edited)
    return merged


def _convert_esp(plugin: Any, json_path: Path, esp_path: Path, tes3conv: Path, scratch: Path) -> Path:
    issues = validate_tes3json(plugin)
    if issues:
        raise RuntimeError(
            "town plugin validation failed:\n" + "\n".join(str(issue) for issue in issues)
        )
    scratch.mkdir(parents=True, exist_ok=True)
    write_tes3json(plugin, json_path)
    result = subprocess.run(
        [str(tes3conv), "-o", "-c", str(json_path), str(esp_path)],
        cwd=str(scratch),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0 or not esp_path.is_file() or esp_path.stat().st_size <= 0:
        raise RuntimeError(
            f"tes3conv failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    return esp_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city-layout", type=Path, required=True)
    parser.add_argument("--town-placements", type=Path, required=True)
    parser.add_argument("--wall", type=Path,
                        help="composed wall JSON for wall-aware settlements")
    parser.add_argument("--edited-land", type=Path, required=True)
    parser.add_argument("--scatter-json", type=Path,
                        help="existing scatter JSON to filter/reuse")
    parser.add_argument("--scatter-analysis", type=Path)
    parser.add_argument("--cliff-analysis", type=Path)
    parser.add_argument("--bbox-cache", type=Path)
    parser.add_argument("--open-face-profiles", type=Path)
    parser.add_argument("--cliff-seating-config", type=Path,
                        help="cliff seating config JSON (paired with --cliff-seating-profiles)")
    parser.add_argument("--cliff-seating-profiles", type=Path,
                        help="cliff seating profile sidecar JSON (paired with --cliff-seating-config)")
    parser.add_argument("--groundcover-config", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, default=DEFAULT_TERRAIN)
    parser.add_argument("--land-source-json", type=Path, required=True,
                        help="corridor remap LAND JSON merged before city edits")
    parser.add_argument("--margin-gu", type=float, default=256.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--name", type=str, default="falkreath")
    parser.add_argument("--bounds", type=int, nargs=4, default=None, help="scatter bounds [min_x max_x min_y max_y] inclusive cells; defaults to TARGET_BOUNDS or derived from city layout")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tes3conv", type=Path, default=DEFAULT_TES3CONV)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path(r"C:\Users\LiamF\AppData\Local\Temp\opencode\settlement-wilderness"),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    _check_out_dir(args.out_dir)
    # Conversion runs from an isolated scratch cwd; all output paths must be
    # absolute so tes3conv can read the JSON just authored by this run.
    args.out_dir = args.out_dir.resolve()
    for label, path in (
        ("city-layout", args.city_layout),
        ("town-placements", args.town_placements),
        ("edited-land", args.edited_land),
        ("groundcover-config", args.groundcover_config),
        ("terrain", args.terrain),
        ("land-source-json", args.land_source_json),
    ):
        _require_file(path, label)
    if args.scatter_json:
        _require_file(args.scatter_json, "scatter-json")
    else:
        for label, path in (
            ("scatter-analysis", args.scatter_analysis),
            ("cliff-analysis", args.cliff_analysis),
            ("bbox-cache", args.bbox_cache),
            ("open-face-profiles", args.open_face_profiles),
        ):
            if path is None:
                raise RuntimeError(f"input {label} is required when --scatter-json is absent")
            _require_file(path, label)
        if (args.cliff_seating_config is None) != (args.cliff_seating_profiles is None):
            raise RuntimeError(
                "cliff seating config and profiles must be supplied together"
            )
        if args.cliff_seating_config is not None:
            _require_file(args.cliff_seating_config, "cliff-seating-config")
            _require_file(args.cliff_seating_profiles, "cliff-seating-profiles")
    if args.wall:
        _require_file(args.wall, "wall")
    if not args.tes3conv.is_file():
        raise RuntimeError(f"input tes3conv is missing: {args.tes3conv}")

    city_layout = _read_json(args.city_layout)
    town_placements = _read_json(args.town_placements)
    edited_land_doc = _read_json(args.edited_land)
    wall_doc = _read_json(args.wall) if args.wall else None
    locked_wall = city_layout.get("composed_wall")
    if locked_wall is not None and wall_doc is None:
        raise RuntimeError("wall-aware city layout requires its supplied composed wall")
    if wall_doc is not None and wall_doc != locked_wall:
        raise RuntimeError(
            "supplied wall is not the composed wall locked into this city layout"
        )
    if wall_doc is not None and (
        (town_placements.get("source") or {}).get("wall_stamp_id")
        != wall_doc.get("stamp_id")
    ):
        raise RuntimeError(
            "town placements were not realized from this locked wall"
        )

    # 1 -- clearing
    clearing_doc = build_clearing(
        city_layout,
        town_placements,
        margin_gu=args.margin_gu,
        wall=wall_doc,
    )
    clearing_path = args.out_dir / "settlement_clearing.json"
    dump_clearing(clearing_doc, clearing_path)

    # 2 -- scatter
    land_records = _load_edited_land(
        args.terrain, args.land_source_json, args.edited_land
    )
    _bounds = tuple(args.bounds) if args.bounds else TARGET_BOUNDS
    # If still on default but city layout is available, derive bounds from survey identities
    # to support per-settlement scattering without explicit --bounds.
    if args.bounds is None:
        try:
            _survey_path = city_layout.get("site_context", {}).get("frame", {}).get("site_survey_sha256")
            # Fallback: derive from survey file path in identities
            _id_survey = city_layout.get("identities", {}).get("survey", {}).get("path")
            if _id_survey and Path(_id_survey).is_file():
                import json as _js
                _sjs = _js.loads(Path(_id_survey).read_text(encoding="utf-8"))
                _tb = _sjs.get("target_bounds")
                if _tb and len(_tb) == 4:
                    _bounds = tuple(int(x) for x in _tb)
        except Exception:
            pass
    if args.scatter_json:
        scatter_doc = _read_json(args.scatter_json)
        scatter_doc = _rebuild_reused_scatter(scatter_doc, clearing_doc)
    else:
        scatter_doc = generate_scatter_document(
            land_records,
            _read_json(args.scatter_analysis),
            _read_json(args.cliff_analysis),
            _read_json(args.bbox_cache),
            config=GenerationConfig(master_seed=int(args.seed), bounds=_bounds),
            terrain_source=str(args.terrain.resolve()),
            open_face_profiles=_read_json(args.open_face_profiles),
            open_face_source=str(args.open_face_profiles.resolve()),
            clearing=clearing_doc,
            edited_land_source=str(args.land_source_json.resolve()),
            cliff_seating_config=(
                _read_json(args.cliff_seating_config) if args.cliff_seating_config else None
            ),
            cliff_seating_profiles=(
                _read_json(args.cliff_seating_profiles) if args.cliff_seating_profiles else None
            ),
        )
    if wall_doc is not None:
        terrain_field = town_placements.get("terrain_field")
        frame_origin = (terrain_field or {}).get("frame_origin_gu") if isinstance(terrain_field, Mapping) else None
        if not isinstance(frame_origin, list) or len(frame_origin) < 2:
            raise RuntimeError("wall-aware scatter requires seated terrain_field.frame_origin_gu")
        scatter_doc, wall_exclusion = filter_scatter_document(
            scatter_doc, wall_doc, city_layout, frame_origin
        )
    else:
        wall_exclusion = None
    scatter_path = args.out_dir / f"{args.name}_scatter.json"
    _write_json(scatter_path, scatter_doc)

    # 3 -- groundcover (clearing + scatter exclusions + edited land)
    groundcover_values = _read_json(args.groundcover_config)
    groundcover_values = {
        **groundcover_values,
        "land_plugin": str(args.terrain.resolve()),
        "edited_land_json": None,
        "clearing_json": str(clearing_path.resolve()),
        "scatter_exclusions": str(scatter_path.resolve()),
    }
    groundcover_config = groundcover_config_from_mapping(groundcover_values)
    ini = parse_ini(groundcover_config.ini_path)
    groundcover_doc = generate_groundcover_document_with_land(
        groundcover_config,
        ini,
        land_records,
        load_ltex(args.terrain),
    )
    groundcover_path = args.out_dir / f"{args.name}_groundcover.json"
    _write_json(groundcover_path, groundcover_doc)

    # 4 -- town ESP
    town_plugin = build_town_plugin(
        edited_land_doc=edited_land_doc,
        town_placements=town_placements,
        scatter_document=scatter_doc,
        wall_document=wall_doc,
        description=f"Procedural {args.name} town (buildings + land + scatter)",
    )
    town_esp = args.out_dir / f"{args.name}_town.esp"
    _convert_esp(
        town_plugin,
        args.out_dir / f"{args.name}_town_plugin.json",
        town_esp,
        args.tes3conv,
        args.scratch / "town",
    )

    # 5 -- groundcover ESP (separate file, registered via groundcover=)
    groundcover_esp = args.out_dir / f"{args.name}_groundcover.esp"
    author_groundcover_plugin(
        groundcover_doc,
        master_plugin=args.terrain,
        master_name=str(groundcover_config.master_name or ""),
        object_prefix=str(groundcover_config.object_prefix or "PTGC_"),
        output_path=groundcover_esp,
        scratch_dir=args.scratch / "groundcover",
        tes3conv_path=args.tes3conv,
    )

    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "status": "ok",
        "elapsed_s": round(elapsed, 3),
        "name": args.name,
        "out_dir": str(args.out_dir.resolve()),
        "clearing": {
            "path": str(clearing_path),
            "building_exclusions": len(clearing_doc["building_exclusions"]),
            "surface_exclusions": len(clearing_doc["surface_exclusions"]),
            "road_exclusions": len(clearing_doc["road_exclusions"]),
            "margin_gu": clearing_doc["margin_gu"],
        },
        "scatter": {
            "path": str(scatter_path),
            "total_refs": scatter_doc["placement_stats"]["total_refs"],
            "rocks_in_city": scatter_doc["city_clearing"]["accepted_rock_cliff_in_city"],
            "wall_exclusion": wall_exclusion,
        },
        "groundcover": {
            "path": str(groundcover_path),
            "total_refs": groundcover_doc["density"]["placement_stats"]["total"],
            "clearing_blocked": groundcover_doc["city_clearing"]["blocks_point_rejections"],
        },
        "town_esp": {"path": str(town_esp), "bytes": town_esp.stat().st_size},
        "groundcover_esp": {
            "path": str(groundcover_esp),
            "bytes": groundcover_esp.stat().st_size,
        },
        "inputs": {
            "city_layout": str(args.city_layout.resolve()),
            "town_placements": str(args.town_placements.resolve()),
            "edited_land": str(args.edited_land.resolve()),
            "land_source_json": str(args.land_source_json.resolve()),
            "scatter_json": str(args.scatter_json.resolve()) if args.scatter_json else None,
            "terrain": str(args.terrain.resolve()),
            "wall": str(args.wall.resolve()) if args.wall else None,
        },
    }
    _write_json(args.out_dir / "run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:
        print(f"FAILURE: settlement-wilderness {exc}", file=sys.stderr)
        return 1
    print(
        f"wrote {summary['town_esp']['path']} "
        f"({summary['town_esp']['bytes']} B) and "
        f"{summary['groundcover_esp']['path']} "
        f"({summary['groundcover_esp']['bytes']} B) in {summary['elapsed_s']}s"
    )
    print(
        "scatter refs=" + str(summary["scatter"]["total_refs"]) +
        f" rocks_in_city={summary['scatter']['rocks_in_city']} | "
        "groundcover refs=" + str(summary["groundcover"]["total_refs"]) +
        f" clearing_blocked={summary['groundcover']['clearing_blocked']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
