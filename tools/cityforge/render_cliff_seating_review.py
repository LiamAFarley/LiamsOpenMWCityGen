#!/usr/bin/env python3
"""Render the cliff seating review board on the real heightmap context.

Stage-3 review tool of the cliff seating plan.  Two view families, both
rendered through ``tools/blender_flat_render.py``:

* Focused views — for each mesh's worst-margin accepted ref (named by the
  scatter document's ``cliff_seating_audit``), one render per configured view
  direction.  Terrain is the exact generation heightmap (the Kreathi remap
  plugin converted to a masterless ESP), and the scene contains every seated
  cliff within the configured focus radius.
* Complete-city comparison views — the accepted r18/v20 town ESP terrain with
  its seated buildings plus every seating cliff inside the town LAND cells,
  from the same configured directions.  Walls are deliberately excluded
  (the wall system is under repair elsewhere and is not the review target).

This tool renders; it never authors plugins or edits placement data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen import engine_transform  # noqa: E402
from procgen.tes3json import land_records_from_json  # noqa: E402

DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _world_to_scene(world: list[float], anchor: tuple[int, int]) -> list[float]:
    ax, ay = anchor[0] * 8192.0, anchor[1] * 8192.0
    return [
        round((float(world[0]) - ax) * 0.01, 6),
        round((float(world[1]) - ay) * 0.01, 6),
        round(float(world[2]) * 0.01, 6),
    ]


def _cliff_mesh_entry(ref: dict, anchor: tuple[int, int], index: int) -> dict:
    rotation = ref.get("rotation_radians") or [0.0, 0.0, 0.0]
    euler = engine_transform.blender_xyz_euler_for_tes3_rotation(
        [float(v) for v in rotation]
    )
    return {
        "id": f"cliff_{index:05d}_{ref.get('ref_id', '')}",
        "mesh": ref["mesh"],
        "position": _world_to_scene(ref["position_gu"], anchor),
        "rotation": [round(float(v), 6) for v in euler],
        "scale": float(ref.get("scale", 1.0)),
        "building": False,
        "source_reference": ref.get("ref_id", ""),
        "cliff": True,
    }


def _building_mesh_entries(seated: dict, survey: dict, anchor: tuple[int, int]) -> list[dict]:
    origin_x, origin_y = (float(v) for v in survey["frame"]["origin_gu"])
    entries = []
    for index, obj in enumerate(seated.get("objects", [])):
        position = obj["world_position_gu"]
        matrix = obj["rotation_matrix_3x3"]
        world = [
            float(position[0]) + origin_x,
            float(position[1]) + origin_y,
            float(position[2]),
        ]
        entries.append({
            "id": f"town_{index:04d}_{obj['reference_id']}",
            "mesh": obj["model_key"],
            "position": _world_to_scene(world, anchor),
            "rotation_z": round(math.degrees(math.atan2(
                float(matrix[1][0]), float(matrix[0][0])
            )), 6),
            "scale": float(obj.get("scale", 1.0)),
            "building": True,
            "source_reference": obj["reference_id"],
        })
    return entries


def _lighting() -> dict:
    return {
        "sun_energy": 1.35,
        "sun_angle_degrees": 32.0,
        "world_strength": 0.9,
        "fill_energy": 2200.0,
        "fill_size_factor": 2.0,
        "view_look": "AgX - Medium Low Contrast",
    }


def _scene_base(anchor: tuple[int, int], cells: list[tuple[int, int]], terrain_plugin: Path) -> dict:
    return {
        "scene_name": "CliffSeatingReview",
        "import": {
            "scale_correction": 0.01,
            "normalize_to_position": False,
            "use_existing_materials": True,
            "ignore_collision_nodes": True,
            "ignore_animations": True,
        },
        "lighting": _lighting(),
        "camera": {
            "mode": "ORTHO",
            "view": "oblique",
            "resolution": [1600, 1000],
            "margin": 1.18,
        },
        "terrain": {
            "enabled": True,
            "plugin": str(terrain_plugin.resolve()),
            "texture_plugin": str(terrain_plugin.resolve()),
            "texture_masters": [],
            "anchor_grid": list(anchor),
            "cells": [list(cell) for cell in cells],
            "decimate": 2,
            "max_seconds": 120.0,
        },
        "meshes": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scatter", type=Path, required=True,
                        help="Seating scatter JSON (carries cliff_seating_audit)")
    parser.add_argument("--terrain-esp", type=Path, required=True,
                        help="Masterless ESP of the generation heightmap (focused views)")
    parser.add_argument("--town-esp", type=Path, required=True,
                        help="Accepted town ESP supplying the complete-city LAND context")
    parser.add_argument("--town-placements", type=Path, required=True)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--edited-land", type=Path, required=True,
                        help="Town LAND tes3conv JSON (defines the town cell set)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    args = parser.parse_args(argv)

    scatter = _load(args.scatter)
    config = _load(args.config)
    seated = _load(args.town_placements)
    survey = _load(args.survey)
    town_land = land_records_from_json(_load(args.edited_land))
    town_cells = sorted(town_land)
    if not town_cells:
        raise RuntimeError("town edited LAND contains no cells")

    audit = scatter.get("cliff_seating_audit") or {}
    worst = audit.get("worst_margin_ref_by_mesh") or {}
    wanted_refs = {row["ref_id"] for row in worst.values() if isinstance(row, dict)}
    cliffs = [
        ref
        for cell in scatter["density"]["cells"]
        for ref in cell.get("refs", [])
        if ref.get("category") == "cliff"
    ]
    by_ref = {ref["ref_id"]: ref for ref in cliffs}
    missing = wanted_refs - set(by_ref)
    if missing:
        raise RuntimeError(f"audit refs missing from scatter: {sorted(missing)}")

    directions = [
        [float(v) for v in row]
        for row in config.get("visual_review", {}).get(
            "view_directions", [[0.72, -0.72, 0.62]]
        )
    ]
    focus_radius = float(config.get("visual_review", {}).get("focus_radius_gu", 8192.0))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    town_cell_set = {tuple(cell) for cell in town_cells}
    town_anchor = town_cells[0]

    jobs: list[dict] = []

    # ---- focused views ------------------------------------------------------
    for ref_id in sorted(wanted_refs):
        ref = by_ref[ref_id]
        cell = tuple(ref["cell"])
        neighborhood = [
            (cx, cy)
            for cx in range(cell[0] - 2, cell[0] + 3)
            for cy in range(cell[1] - 2, cell[1] + 3)
        ]
        anchor = min(neighborhood)
        neighbors = [
            other for other in cliffs
            if math.hypot(
                float(other["position_gu"][0]) - float(ref["position_gu"][0]),
                float(other["position_gu"][1]) - float(ref["position_gu"][1]),
            ) <= focus_radius
        ]
        bbox = (ref.get("bbox") or {}).get("world_aabb_gu") or {}
        minimum = bbox.get("min") or [0, 0, 0]
        maximum = bbox.get("max") or [0, 0, 0]
        # Frame on the EMERGED outcrop: XY extent of the seated ref, targeted
        # at the midpoint between the anchor terrain and the mesh top (the
        # seated origin is deliberately deep underground).
        xy_diagonal_gu = math.hypot(
            float(maximum[0]) - float(minimum[0]),
            float(maximum[1]) - float(minimum[1]),
        )
        span_scene = max(90.0, round(xy_diagonal_gu * 1.2 * 0.01, 3))
        anchor_terrain_z = float(ref.get("terrain", {}).get("terrain_z_gu") or 0.0)
        target_z_gu = 0.5 * (float(maximum[2]) + anchor_terrain_z)
        for view_index, direction in enumerate(directions):
            scene = _scene_base(anchor, neighborhood, args.terrain_esp)
            scene["meshes"] = [
                _cliff_mesh_entry(other, anchor, index)
                for index, other in enumerate(neighbors)
            ]
            focus_position = _world_to_scene(
                [float(ref["position_gu"][0]), float(ref["position_gu"][1]), target_z_gu],
                anchor,
            )
            scene["camera"]["view_direction"] = direction
            scene["camera"]["fixed_target_scene"] = focus_position
            scene["camera"]["fixed_span_scene"] = span_scene
            scene["camera"]["subject"] = {
                "ids": [entry["id"] for entry in scene["meshes"]],
                "include_terrain": True,
                "padding": 1.5,
            }
            scene["scene_name"] = f"CliffSeatingFocus_{ref_id}"
            scene["source"] = {
                "focus_ref": ref_id,
                "focus_mesh": ref["mesh"],
                "neighbor_cliff_count": len(neighbors),
                "terrain_plugin": str(args.terrain_esp.resolve()),
                "scatter": str(args.scatter.resolve()),
            }
            name = f"focus_{ref_id}_view{view_index}"
            jobs.append({
                "scene": scene,
                "png": out_dir / f"{name}.png",
                "label": name,
            })

    # ---- complete-city comparison views -------------------------------------
    city_cliffs = [
        ref for ref in cliffs if tuple(ref["cell"]) in town_cell_set
    ]
    building_entries = _building_mesh_entries(seated, survey, town_anchor)
    for view_index, direction in enumerate(directions):
        scene = _scene_base(town_anchor, town_cells, args.town_esp)
        scene["meshes"] = building_entries + [
            _cliff_mesh_entry(ref, town_anchor, index)
            for index, ref in enumerate(city_cliffs)
        ]
        # Fixed town-centered framing: the flat render's water plane would
        # otherwise inflate the auto-framed bounds far past the LAND cross.
        if building_entries:
            center_x = sum(entry["position"][0] for entry in building_entries) / len(building_entries)
            center_y = sum(entry["position"][1] for entry in building_entries) / len(building_entries)
            center_z = sum(entry["position"][2] for entry in building_entries) / len(building_entries)
        else:
            center_x = center_y = center_z = 0.0
        scene["camera"]["view_direction"] = direction
        scene["camera"]["fixed_target_scene"] = [center_x, center_y, center_z]
        scene["camera"]["fixed_span_scene"] = 300.0
        scene["camera"]["subject"] = {
            "ids": [entry["id"] for entry in scene["meshes"] if entry.get("building")],
            "include_terrain": True,
            "padding": 8.0,
        }
        scene["scene_name"] = f"CliffSeatingCity_view{view_index}"
        scene["source"] = {
            "town_esp": str(args.town_esp.resolve()),
            "seating_cliffs_in_town_land": len(city_cliffs),
            "building_count": len(building_entries),
            "note": "walls excluded: wall system under repair, not the review target",
        }
        name = f"city_view{view_index}"
        jobs.append({
            "scene": scene,
            "png": out_dir / f"{name}.png",
            "label": name,
        })

    rendered = 0
    for job in jobs:
        with tempfile.TemporaryDirectory(prefix="cliff_review_") as temporary:
            scene_path = Path(temporary) / "scene.json"
            scene_path.write_text(
                json.dumps(job["scene"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            command = [
                str(args.blender.resolve()), "-b", "--python",
                str(WORKSPACE / "tools" / "blender_flat_render.py"),
                "--", str(scene_path.resolve()), str(job["png"].resolve()),
                "--statics", "all",
            ]
            completed = subprocess.run(command, cwd=str(WORKSPACE), check=False)
            if completed.returncode != 0 or not job["png"].is_file():
                raise RuntimeError(f"render failed for {job['label']} (exit {completed.returncode})")
        rendered += 1
        print(f"[cliff-review] {rendered}/{len(jobs)} {job['label']}", flush=True)

    print(f"wrote {rendered} review renders under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILURE: cliff-seating-review-render {exc}", file=sys.stderr)
        raise SystemExit(1)
