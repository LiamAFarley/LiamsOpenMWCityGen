"""Cityforge T1.1 synthetic validation fixture builder - D-PLAN proof plan.

Pipeline position
------------------
Proves the T1.1 plan contract and overlay on a *synthetic* plan only: this
script deterministically authors ``synthetic_not_a_falkreath_design.city_plan.json``
plus a manifest under
``output/cityforge/phase1/t1_1_validation_fixture/`` and runs the strict
validator on it (the builder refuses to write anything if the plan does not
validate with zero errors).  It never authors a real Falkreath design and
never runs placement; the real plan gate is T1.6 (lead-driven, user review).

The fixture deliberately mixes:
- explicit stamp lots and non-explicit lots resolved by the shared
  deterministic selector (both resolution modes reported by the validator);
- roads connected to real aligned-centerline edge/node ids and measured
  map-edge exits (``exit_<side>_<edge_id>`` computed from
  ``tamriel_aligned_centerlines_v1.json`` by ``cityplan.measure_map_exits``);
- a measured-capability palisade boundary with gates on the ring, each
  within 512 GU of a planned road;
- a dock feature in the water mask (the only water-position exception);
- terrain edits linked to plan elements, closed-vocabulary texture zones,
  districts, and wilderness hints;
- one lot deliberately placed far from all roads to exercise the soft
  door-to-road diagnostic (warnings are expected; errors are not).

Determinism
-----------
No randomness: lot anchors are placed on a fixed 8192-GU lattice in
row-major order; each (lattice point, lot spec) combination is accepted
only when the exact yawed hull checks pass (scope, buildable/water tile
coverage, no strict overlap, >= 3000 GU boundary gap from placed hulls).
The validator is run on the assembled plan; the manifest records the actual
resulting counts, warning codes, and input hashes.

Usage
-----
    python tools/cityforge/build_cityplan_fixture.py [--out-dir ...]

Exit codes: 0 = fixture written and self-validated (zero errors);
1 = fixture failed its own validation (nothing canonical written);
2 = configuration failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen import cityplan  # noqa: E402
from procgen.censusio import write_deterministic  # noqa: E402
from procgen.cityplan import (  # noqa: E402
    Bundle,
    SITE_SPAN_GU,
    ring_pair_status,
    tiles_covered_by_ring,
    yaw_hull,
)

CANONICAL_SURVEY = "output/cityforge/sites/falkreath_v1/site_survey.json"
CANONICAL_BRIEF = "output/cityforge/briefs/falkreath_v1/kit_brief.json"
CANONICAL_PALETTE = "output/cityforge/briefs/falkreath_v1/region_palette.json"
CANONICAL_LIBRARIES = (
    "output/cityforge/stamps/karthgad_nord_v1.json",
    "output/cityforge/stamps/markarth_side_stone_v1.json",
)
CANONICAL_CENTERLINES = ("output/mapdata/roads/tamriel_aligned_centerlines_v1/"
                         "tamriel_aligned_centerlines_v1.json")
DEFAULT_OUT_DIR = "output/cityforge/phase1/t1_1_validation_fixture"

BANNER = "SYNTHETIC VALIDATION FIXTURE \u2014 NOT A FALKREATH DESIGN"

#: Lot specs in fixed placement order.  ``stamp_id`` present = explicit
#: request; absent = non-explicit (shared deterministic selector resolves).
LOT_SPECS = [
    {"building_type": "house", "size_class": "small",
     "stamp_id": "karthgad_v1__door_094670", "mode": "explicit"},
    {"building_type": "house", "size_class": "medium", "mode": "selector"},
    {"building_type": "house", "size_class": "large",
     "stamp_id": "karthgad_v1__door_094666_door_094669", "mode": "explicit"},
    {"building_type": "tavern", "size_class": "large",
     "stamp_id": "markarth_side_v1__u31_shor_s_hearth_inn", "mode": "explicit"},
    {"building_type": "smith", "size_class": "medium",
     "stamp_id": "karthgad_v1__door_094668", "mode": "explicit"},
    {"building_type": "hall", "size_class": "large",
     "stamp_id": "karthgad_v1__door_094665", "mode": "explicit"},
    {"building_type": "shop", "size_class": "medium", "mode": "selector"},
    {"building_type": "stable", "size_class": "small",
     "stamp_id": "markarth_side_v1__u51_stables", "mode": "explicit"},
    {"building_type": "manor", "size_class": "medium", "mode": "selector"},
    {"building_type": "mill", "size_class": "small",
     "stamp_id": "markarth_side_v1__u14_windmill", "mode": "explicit"},
    {"building_type": "farm", "size_class": "small", "mode": "selector"},
    {"building_type": "guild", "size_class": "large",
     "stamp_id": "markarth_side_v1__u31_imperial_guilds", "mode": "explicit"},
]

#: Roads reference real aligned-centerline edge/node ids and the *measured*
#: exit ids (computed from the aligned product by ``cityplan.measure_map_exits``).
#: The west approach connects to the aligned edge id itself: after the +4096 GU
#: registration correction that road no longer crosses the site border (the
#: displaced frame's ``exit_west_road_edge_31938970a750dc24`` is gone; see the
#: road-authority investigation section 5.2), so no west exit exists to name.
ROAD_SPECS = [
    {"road_id": "main_street", "class": "street", "width_gu": 512.0,
     "polyline": [[2048.0, 0.0], [2048.0, 8192.0], [8192.0, 8192.0],
                  [19200.0, 16128.0]],
     "connects": ["exit_south_road_edge_f200c85cfe673343",
                  "road_node_fe5ab61f1218c960"]},
    {"road_id": "approach_west", "class": "approach", "width_gu": 1024.0,
     "polyline": [[0.0, 23087.1], [2048.0, 24576.0], [8192.0, 24576.0]],
     "connects": ["road_edge_31938970a750dc24", "main_street"]},
    {"road_id": "approach_east", "class": "approach", "width_gu": 1024.0,
     "polyline": [[47104.0, 24576.0], [51200.0, 22528.0],
                  [57344.0, 19606.3]],
     "connects": ["exit_east_road_edge_f36abb2dc60cb6fc", "main_street"]},
    {"road_id": "approach_south", "class": "approach", "width_gu": 1024.0,
     "polyline": [[24576.0, 2304.0], [32768.0, 2304.0],
                  [38503.7, 0.0]],
     "connects": ["exit_south_road_edge_ed14e373290dcd8f", "main_street"]},
    {"road_id": "market_lane", "class": "path", "width_gu": 256.0,
     "polyline": [[16384.0, 16384.0], [20480.0, 16384.0],
                  [24576.0, 20480.0]],
     "connects": ["main_street", "gate_west"]},
    {"road_id": "dock_lane", "class": "dock_lane", "width_gu": 256.0,
     "polyline": [[26624.0, 29696.0], [26624.0, 24576.0],
                  [24576.0, 16384.0]],
     "connects": ["market_lane"]},
]

PALISADE_RING = [[2048.0, 2048.0], [47104.0, 2048.0], [47104.0, 47104.0],
                 [2048.0, 47104.0], [2048.0, 2048.0]]

GATE_SPECS = [
    {"gate_id": "gate_west", "position": [2048.0, 24576.0], "heading_deg": 270.0,
     "on_road": "approach_west"},
    {"gate_id": "gate_south", "position": [24576.0, 2048.0], "heading_deg": 180.0,
     "on_road": "approach_south"},
    {"gate_id": "gate_east", "position": [47104.0, 24576.0], "heading_deg": 90.0,
     "on_road": "approach_east"},
]

ZONE_SPECS = [
    {"zone_id": "dirt_core", "classes": [
        {"texture": "settlement_dirt", "weight": 0.62},
        {"texture": "settlement_grass_dirt", "weight": 0.24},
        {"texture": "settlement_cobble", "weight": 0.14}]},
    {"zone_id": "plaza", "classes": [
        {"texture": "settlement_cobble", "weight": 0.5},
        {"texture": "settlement_grass_dirt", "weight": 0.3},
        {"texture": "base", "weight": 0.2}]},
    {"zone_id": "green", "classes": [
        {"texture": "base", "weight": 0.7},
        {"texture": "settlement_grass_dirt", "weight": 0.3}]},
]

DISTRICT_SPECS = [
    {"district_id": "core", "kind": "core", "texture_zone": "dirt_core",
     "polygon": [[8192.0, 8192.0], [28672.0, 8192.0], [28672.0, 24576.0],
                 [8192.0, 24576.0]]},
    {"district_id": "residential", "kind": "residential", "texture_zone": "green",
     "polygon": [[30720.0, 8192.0], [45056.0, 8192.0], [45056.0, 24576.0],
                 [30720.0, 24576.0]]},
    {"district_id": "market", "kind": "market", "texture_zone": "plaza",
     "polygon": [[16384.0, 16384.0], [24576.0, 16384.0], [24576.0, 20480.0],
                 [16384.0, 20480.0]]},
    {"district_id": "docks", "kind": "docks", "texture_zone": "green",
     "polygon": [[24576.0, 28672.0], [32768.0, 28672.0], [32768.0, 36864.0],
                 [24576.0, 36864.0]]},
    {"district_id": "craft", "kind": "craft", "texture_zone": "dirt_core",
     "polygon": [[8192.0, 24576.0], [16384.0, 24576.0], [16384.0, 28672.0],
                 [8192.0, 28672.0]]},
    {"district_id": "outskirts", "kind": "outskirts", "texture_zone": "green",
     "polygon": [[30720.0, 24576.0], [45056.0, 24576.0], [45056.0, 36864.0],
                 [30720.0, 36864.0]]},
]


def _anchor_fits(bundle: Bundle, x: float, y: float, yaw: float,
                 hull: list, placed: list[tuple]) -> bool:
    """One deterministic acceptance check used at fixture build time: the
    exact yawed hull must be in scope, cover only buildable non-water
    tiles, and keep >= 3000 GU boundary gap from every placed hull."""
    if not cityplan.in_scope(x, y):
        return False
    state = bundle.door_anchor_state(x, y)
    if not state["buildable"] or state["water"]:
        return False
    world = yaw_hull(hull, yaw, (x, y))
    if not all(cityplan.in_scope(p[0], p[1]) for p in world):
        return False
    for tx, ty in tiles_covered_by_ring(world):
        if not bundle.tile_buildable(tx, ty) or bundle.tile_water(tx, ty):
            return False
    for _, other in placed:
        status, dist = ring_pair_status(world, other)
        if status == "overlap" or (status == "touch" and dist < 3000.0):
            return False
        if status == "clear" and dist < 3000.0:
            return False
    return True


def _place_lots(bundle: Bundle) -> list[dict]:
    """Deterministic lattice placement (8192-GU pitch, row-major)."""
    lattice = [(4096.0 + ix * 8192.0, 4096.0 + iy * 8192.0)
               for iy in range(6) for ix in range(6)]
    lots: list[dict] = []
    placed: list[tuple[str, list]] = []
    remaining = list(LOT_SPECS)
    for point in lattice:
        if not remaining:
            break
        for spec in list(remaining):
            yaw = 0.0
            hull = bundle.stamp_geometry[spec["stamp_id"]]["footprint"]["hull_xy_rel"] \
                if spec.get("stamp_id") else None
            if hull is None:
                # selector lots: resolve with the shared selector first
                request = {"building_type": spec["building_type"],
                           "size_class": spec["size_class"]}
                candidates = cityplan._candidate_stamps(bundle, request)
                if not candidates:
                    raise cityplan.BundleError(
                        f"fixture spec {spec} has no compatible stamp")
                stamp_id = cityplan._select_stamp(bundle, candidates)
                hull = bundle.stamp_geometry[stamp_id]["footprint"]["hull_xy_rel"]
                spec = dict(spec, stamp_id=stamp_id)
            if not _anchor_fits(bundle, point[0], point[1], yaw, hull, placed):
                continue
            lot_id = f"lot_{len(lots) + 1:02d}"
            request = {"building_type": spec["building_type"],
                       "size_class": spec["size_class"]}
            if spec["mode"] == "explicit":
                # explicit requests carry the stamp_id; selector requests
                # must NOT, so the validator's shared selector resolves them
                request["stamp_id"] = spec["stamp_id"]
            lots.append({
                "lot_id": lot_id,
                "district": _district_for(spec["building_type"]),
                "position": [point[0], point[1]],
                "yaw_deg": yaw,
                "request": request,
                "terrain_policy": {"mode": "conform", "max_cut_fill_gu": 400.0},
                "access": {"face_road": "main_street"},
                "notes": f"synthetic fixture lot; resolution mode "
                         f"{spec['mode']}",
            })
            placed.append((lot_id, yaw_hull(hull, yaw, (point[0], point[1]))))
            match = next(s for s in remaining
                         if s["building_type"] == spec["building_type"]
                         and s.get("size_class") == spec.get("size_class"))
            remaining.remove(match)
    if remaining:
        raise cityplan.BundleError(
            f"fixture placement failed: {len(remaining)} lot specs unplaced: "
            f"{[r['building_type'] for r in remaining]}")
    return lots


def _district_for(building_type: str) -> str:
    if building_type in ("shop", "smith", "tavern", "guild"):
        return "core"
    if building_type == "stable":
        return "craft"
    if building_type in ("mill", "dock"):
        return "docks"
    if building_type == "farm":
        return "outskirts"
    return "residential"


def _first_water_tile_center(bundle: Bundle) -> list:
    for ty in range(cityplan.TILE_SIDE):
        for tx in range(cityplan.TILE_SIDE):
            if bundle.tile_water(tx, ty):
                return [tx * 512.0 + 256.0, ty * 512.0 + 256.0]
    raise cityplan.BundleError("fixture needs a water tile for the dock")


def _cell_median(bundle: Bundle, x: float, y: float) -> float:
    origin = bundle.survey_frame["origin_gu"]
    cx = int((origin[0] + x) // 8192.0)
    cy = int((origin[1] + y) // 8192.0)
    for cell in bundle.site_survey.get("cells", []):
        if list(cell.get("grid")) == [cx, cy]:
            return float(cell.get("elev_med_gu", 0.0))
    return 0.0


def build_fixture_plan(bundle: Bundle) -> dict:
    """Assemble the synthetic fixture plan document (deterministic)."""
    lots = _place_lots(bundle)
    dock_pos = _first_water_tile_center(bundle)
    dock_edit_target = _cell_median(bundle, dock_pos[0], dock_pos[1])
    lot_edit_target = _cell_median(bundle, lots[0]["position"][0],
                                   lots[0]["position"][1])
    plan = {
        "schema_version": 1,
        "plan_id": "synthetic_validation_fixture_v1",
        "settlement": {
            "name": "Falkreath",
            "seed_marker": "M0400",
            "anchor_cell": [-92, -10],
            "target_cells": {"min_x": -95, "max_x": -89,
                             "min_y": -11, "max_y": -5},
        },
        "frame": {
            "origin_gu": list(bundle.survey_frame["origin_gu"]),
            "units": "game_units",
            "yaw_convention": bundle.survey_frame["axis_convention"],
            "site_survey_sha256": bundle.survey_sha256,
        },
        "design_notes": (
            "SYNTHETIC VALIDATION FIXTURE - NOT A FALKREATH DESIGN. "
            "Engineered by build_cityplan_fixture.py to exercise the T1.1 "
            "plan contract: explicit + selector lots, corrected-centerline "
            "road connections, measured palisade, water dock, linked terrain "
            "edits, closed texture zones. No real Falkreath design intent."),
        "districts": DISTRICT_SPECS,
        "roads": [dict(spec, surface="road", grade_policy="conform")
                  for spec in ROAD_SPECS],
        "lots": lots,
        "boundaries": [{
            "boundary_id": "palisade_ring",
            "kind": "palisade",
            "polygon": PALISADE_RING,
            "gates": GATE_SPECS,
        }],
        "features": [
            {"feature_id": "dock_01", "kind": "dock", "position": dock_pos,
             "yaw_deg": 0.0, "on_road": "dock_lane",
             "notes": "synthetic dock in the water mask (water exception)"},
            {"feature_id": "well_01", "kind": "well",
             "position": [18432.0, 16640.0], "yaw_deg": 0.0,
             "on_road": "market_lane"},
            {"feature_id": "signpost_01", "kind": "signpost",
             "position": [3072.0, 25088.0], "yaw_deg": 0.0,
             "on_road": "approach_west"},
            {"feature_id": "keep_trees_01", "kind": "keep_trees",
             "position": [38912.0, 28672.0], "yaw_deg": 0.0},
        ],
        "terrain_edits": [
            {"edit_id": "dock_shelf", "kind": "flatten_shelf",
             "polygon": [[25600.0, 29184.0], [26624.0, 29184.0],
                         [26624.0, 30208.0], [25600.0, 30208.0]],
             "target_height_gu": dock_edit_target, "falloff_gu": 512.0,
             "linked_to": ["dock_01", "dock_lane"]},
            {"edit_id": "lot_pad_01", "kind": "flatten_shelf",
             "polygon": [[lots[0]["position"][0] - 512.0,
                          lots[0]["position"][1] - 512.0],
                         [lots[0]["position"][0] + 512.0,
                          lots[0]["position"][1] - 512.0],
                         [lots[0]["position"][0] + 512.0,
                          lots[0]["position"][1] + 512.0],
                         [lots[0]["position"][0] - 512.0,
                          lots[0]["position"][1] + 512.0]],
             "target_height_gu": lot_edit_target, "falloff_gu": 256.0,
             "linked_to": [lots[0]["lot_id"], "main_street"]},
        ],
        "texture_zones": ZONE_SPECS,
        "wilderness_hints": [
            {"hint": "cleared",
             "polygon": [[2048.0, 2048.0], [47104.0, 2048.0],
                         [47104.0, 47104.0], [2048.0, 47104.0]],
             "density": 0.0},
            {"hint": "dense_forest",
             "polygon": [[1024.0, 49152.0], [20480.0, 49152.0],
                         [20480.0, 56320.0], [1024.0, 56320.0]],
             "density": 1.2},
        ],
    }
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Cityforge T1.1 synthetic fixture builder (NOT a real "
                    "Falkreath design)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--site-survey", default=CANONICAL_SURVEY)
    parser.add_argument("--kit-brief", default=CANONICAL_BRIEF)
    parser.add_argument("--region-palette", default=CANONICAL_PALETTE)
    parser.add_argument("--stamp-libraries", nargs="+",
                        default=list(CANONICAL_LIBRARIES))
    parser.add_argument("--centerlines", default=CANONICAL_CENTERLINES)
    args = parser.parse_args(argv)

    try:
        bundle = Bundle.from_paths(
            site_survey=args.site_survey,
            kit_brief=args.kit_brief,
            region_palette=args.region_palette,
            stamp_libraries=args.stamp_libraries,
            centerlines=args.centerlines,
        )
        plan = build_fixture_plan(bundle)
        result = cityplan.validate_plan(plan, bundle)
    except cityplan.BundleError as exc:
        print(f"configuration failure: {exc}", file=sys.stderr)
        return 2

    if not result["valid"]:
        print(f"fixture failed its own validation: {result['error_count']} errors",
              file=sys.stderr)
        for issue in result["issues"]:
            if issue["severity"] == "error":
                print(f"  [{issue['code']}] {issue['path']}: {issue['message']}",
                      file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "synthetic_not_a_falkreath_design.city_plan.json"
    write_deterministic(plan_path, plan)

    manifest = {
        "label": "SYNTHETIC VALIDATION FIXTURE - NOT A FALKREATH DESIGN",
        "builder": "tools/cityforge/build_cityplan_fixture.py",
        "plan": plan_path.name,
        "plan_id": plan["plan_id"],
        "validated": result["valid"],
        "validation": {
            "issue_count": result["issue_count"],
            "error_count": result["error_count"],
            "warning_count": result["warning_count"],
            "warning_codes": sorted(result["summary"]["warning_codes"]),
        },
        "content": {
            "districts": len(plan["districts"]),
            "roads": len(plan["roads"]),
            "lots": len(plan["lots"]),
            "explicit_lots": sum(1 for r in result["summary"]["lot_resolution"]
                                 if r["mode"] == "explicit"),
            "selector_lots": sum(1 for r in result["summary"]["lot_resolution"]
                                 if r["mode"] == "selector"),
            "boundaries": len(plan["boundaries"]),
            "gates": len(plan["boundaries"][0]["gates"]),
            "features": len(plan["features"]),
            "terrain_edits": len(plan["terrain_edits"]),
            "texture_zones": len(plan["texture_zones"]),
            "wilderness_hints": len(plan["wilderness_hints"]),
        },
        "external_road_refs_used": sorted(
            ref for road in plan["roads"] for ref in road["connects"]
            if ref in bundle.edge_ids or ref in bundle.node_ids
            or ref in bundle.map_exits),
        "input_hashes": dict(sorted(bundle.hashes.items())),
    }
    manifest_path = out_dir / "synthetic_not_a_falkreath_design.manifest.json"
    write_deterministic(manifest_path, manifest)

    print(f"wrote {plan_path}")
    print(f"wrote {manifest_path}")
    print(f"fixture valid: {result['error_count']} errors, "
          f"{result['warning_count']} warnings")
    print("warning codes:", sorted(result["summary"]["warning_codes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
