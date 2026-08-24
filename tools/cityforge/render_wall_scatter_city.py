"""Render one contiguous wall-aware townlayout result.

The scene is assembled from the same city layout, seated stamp objects, fitted
wall JSON, filtered scatter JSON, and authored town ESP produced by the
JSON-first settlement pipeline. No prior city scene is used or overlaid.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from procgen import engine_transform
from procgen.tes3json import land_records_from_json
from procgen.wall_scatter import filter_scatter_document

DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe")


def _world_to_scene(world: list[float], anchor: tuple[int, int]) -> list[float]:
    ax, ay = anchor[0] * 8192.0, anchor[1] * 8192.0
    return [
        round((float(world[0]) - ax) * 0.01, 6),
        round((float(world[1]) - ay) * 0.01, 6),
        round(float(world[2]) * 0.01, 6),
    ]


def _wall_meshes(wall: dict, survey: dict, anchor: tuple[int, int]) -> list[dict]:
    origin = [float(v) for v in wall.get("origin_gu", [0.0, 0.0])]
    survey_origin = [float(v) for v in survey["frame"]["origin_gu"]]
    meshes = []
    for index, member in enumerate(wall["members"]):
        offset = member["offset_gu"]
        world = [
            origin[0] + survey_origin[0] + float(offset[0]),
            origin[1] + survey_origin[1] + float(offset[1]),
            float(offset[2]),
        ]
        rotation = member.get("rotation", [0.0, 0.0, 0.0])
        euler = engine_transform.blender_xyz_euler_for_tes3_rotation(
            [float(v) for v in rotation]
        )
        meshes.append({
            "id": f"wall_{index:04d}_{member['source_id']}",
            "mesh": member["model_key"],
            "position": _world_to_scene(world, anchor),
            "rotation": [round(float(v), 6) for v in euler],
            "scale": float(member.get("scale", 1.0)),
            "building": False,
            "source_reference": member["source_id"],
            "wall": True,
        })
    return meshes


def _ref_footprint(ref: dict):
    aabb = (ref.get("bbox") or {}).get("world_aabb_gu") or {}
    minimum = aabb.get("min")
    maximum = aabb.get("max")
    if (isinstance(minimum, list) and len(minimum) >= 2 and
            isinstance(maximum, list) and len(maximum) >= 2):
        return Polygon([
            [float(minimum[0]), float(minimum[1])],
            [float(maximum[0]), float(minimum[1])],
            [float(maximum[0]), float(maximum[1])],
            [float(minimum[0]), float(maximum[1])],
        ])
    position = ref.get("position_gu")
    if isinstance(position, list) and len(position) >= 2:
        return Point(float(position[0]), float(position[1]))
    raise ValueError(f"scatter ref has no position or measured world AABB: {ref.get('ref_id')}")


def _wall_exclusions(wall: dict, survey: dict, city_layout: dict):
    origin = [float(v) for v in wall.get("origin_gu", [0.0, 0.0])]
    survey_origin = [float(v) for v in survey["frame"]["origin_gu"]]
    wall_parts = []
    for member in wall["members"]:
        footprint = member.get("footprint_xy_rel")
        if not footprint or len(footprint) < 3:
            raise ValueError(f"wall member lacks measured footprint: {member.get('source_id')}")
        wall_parts.append(Polygon([
            [survey_origin[0] + origin[0] + float(point[0]),
             survey_origin[1] + origin[1] + float(point[1])]
            for point in footprint
        ]))
    wall_mesh = unary_union(wall_parts)
    polygon = (city_layout.get("inner_wall") or {}).get("polygon")
    if not polygon or len(polygon) < 3:
        raise ValueError("city layout lacks inner_wall.polygon for scatter exclusion")
    city_domain = Polygon([
        [survey_origin[0] + float(point[0]), survey_origin[1] + float(point[1])]
        for point in polygon
    ])
    if not city_domain.is_valid or city_domain.area <= 0.0:
        raise ValueError("city layout inner_wall.polygon is invalid")
    return wall_mesh, city_domain


def _filter_scatter_document(scatter: dict, wall_mesh, city_domain) -> tuple[dict, dict[str, int]]:
    filtered_cells = []
    excluded = {"wall_mesh": 0, "wall_domain": 0}
    for cell in scatter["density"]["cells"]:
        filtered_refs = []
        for ref in cell.get("refs", []):
            footprint = _ref_footprint(ref)
            if footprint.intersection(wall_mesh).area > 1.0:
                excluded["wall_mesh"] += 1
                continue
            if footprint.intersection(city_domain).area > 1.0:
                excluded["wall_domain"] += 1
                continue
            filtered_refs.append(ref)
        filtered_cell = dict(cell)
        filtered_cell["refs"] = filtered_refs
        filtered_cells.append(filtered_cell)
    filtered_density = dict(scatter["density"])
    filtered_density["cells"] = filtered_cells
    filtered = dict(scatter)
    filtered["density"] = filtered_density
    filtered["wall_exclusion"] = {
        "rule": "reject measured scatter AABBs intersecting wall meshes or fitted inner-wall domain",
        "excluded": dict(excluded),
    }
    return filtered, excluded


def _scatter_meshes(
    scatter: dict,
    anchor: tuple[int, int],
    cells: set[tuple[int, int]],
) -> list[dict]:
    meshes = []
    index = 0
    for cell in scatter["density"]["cells"]:
        grid = tuple(int(v) for v in cell["grid"])
        if grid not in cells:
            continue
        for ref in cell.get("refs", []):
            rotation = ref.get("rotation_radians") or [0.0, 0.0, 0.0]
            euler = engine_transform.blender_xyz_euler_for_tes3_rotation(
                [float(v) for v in rotation]
            )
            meshes.append({
                "id": f"scatter_{index:05d}_{ref.get('ref_id', '')}",
                "mesh": ref["mesh"],
                "position": _world_to_scene(ref["position_gu"], anchor),
                "rotation": [round(float(v), 6) for v in euler],
                "scale": float(ref.get("scale", 1.0)),
                "building": False,
                "source_reference": ref.get("ref_id", ""),
                "scatter": True,
            })
            index += 1
    return meshes


def _building_meshes(
    seated: dict,
    survey: dict,
    anchor: tuple[int, int],
) -> list[dict]:
    origin_x, origin_y = (float(v) for v in survey["frame"]["origin_gu"])
    meshes = []
    for index, obj in enumerate(seated.get("objects", [])):
        position = obj["world_position_gu"]
        matrix = obj["rotation_matrix_3x3"]
        world = [
            float(position[0]) + origin_x,
            float(position[1]) + origin_y,
            float(position[2]),
        ]
        meshes.append({
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
    return meshes


def _land_cells(land_json: list | dict) -> list[list[int]]:
    records = land_records_from_json(land_json)
    return [list(cell) for cell in sorted(records)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city-layout", type=Path, required=True)
    parser.add_argument("--seated-objects", type=Path, required=True)
    parser.add_argument("--edited-land", type=Path, required=True)
    parser.add_argument("--town-esp", type=Path, required=True)
    parser.add_argument("--wall", type=Path, required=True)
    parser.add_argument("--scatter", type=Path, required=True)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument(
        "--close-up", action="store_true",
        help="frame buildings and wall only instead of the full terrain extent",
    )
    parser.add_argument(
        "--view-direction", type=float, nargs=3, default=[0.72, -0.72, 0.62],
        metavar=("X", "Y", "Z"),
        help="camera-to-subject direction in scene axes",
    )
    parser.add_argument(
        "--padding", type=float, default=None,
        help="camera subject padding; defaults to 2 close / 8 overview",
    )
    args = parser.parse_args(argv)

    city_layout = json.loads(args.city_layout.read_text(encoding="utf-8"))
    seated = json.loads(args.seated_objects.read_text(encoding="utf-8"))
    wall = json.loads(args.wall.read_text(encoding="utf-8"))
    scatter = json.loads(args.scatter.read_text(encoding="utf-8"))
    survey = json.loads(args.survey.read_text(encoding="utf-8"))
    land_json = json.loads(args.edited_land.read_text(encoding="utf-8"))
    cells = {tuple(cell) for cell in _land_cells(land_json)}
    if not cells:
        raise ValueError("town LAND JSON contains no cells")
    anchor = min(cells)

    building_meshes = _building_meshes(seated, survey, anchor)
    wall_meshes = _wall_meshes(wall, survey, anchor)
    frame_origin = seated.get("terrain_field", {}).get("frame_origin_gu")
    if not isinstance(frame_origin, list) or len(frame_origin) < 2:
        raise ValueError("seated objects lack terrain_field.frame_origin_gu")
    filtered_scatter, excluded = filter_scatter_document(
        scatter, wall, city_layout, frame_origin
    )
    scatter_meshes = _scatter_meshes(filtered_scatter, anchor, cells)
    scene = {
        "scene_name": "Falkreath_R18_JSON_WallAware",
        "import": {
            "scale_correction": 0.01,
            "normalize_to_position": False,
            "use_existing_materials": True,
            "ignore_collision_nodes": True,
            "ignore_animations": True,
        },
        "lighting": {
            "sun_energy": 1.35,
            "sun_angle_degrees": 32.0,
            "world_strength": 0.9,
            "fill_energy": 2200.0,
            "fill_size_factor": 2.0,
            "view_look": "AgX - Medium Low Contrast",
        },
        "camera": {
            "mode": "ORTHO",
            "view": "oblique",
            "view_direction": [float(value) for value in args.view_direction],
            "resolution": [1600, 1000],
            "margin": 1.18,
            "subject": {"include_terrain": True, "padding": 8.0},
        },
        "terrain": {
            "enabled": True,
            "plugin": str(args.town_esp.resolve()),
            "texture_plugin": str(args.town_esp.resolve()),
            "texture_masters": [],
            "anchor_grid": list(anchor),
            "cells": [list(cell) for cell in sorted(cells)],
            "decimate": 2,
            "max_seconds": 60.0,
        },
        "meshes": building_meshes + wall_meshes + scatter_meshes,
        "source": {
            "city_layout": str(args.city_layout.resolve()),
            "seated_objects": str(args.seated_objects.resolve()),
            "town_esp": str(args.town_esp.resolve()),
            "edited_land": str(args.edited_land.resolve()),
            "fitted_wall": str(args.wall.resolve()),
            "regional_scatter": str(args.scatter.resolve()),
            "survey": str(args.survey.resolve()),
            "frame": "survey-local GU + survey origin; anchor southwest LAND cell",
        },
    }
    scene["source"]["wall_scatter_exclusion"] = excluded
    subject_ids = [
        str(mesh["id"])
        for mesh in scene["meshes"]
        if mesh.get("building") or mesh.get("wall")
    ]
    scene.setdefault("camera", {})["subject"] = {
        "ids": subject_ids,
        "include_terrain": not args.close_up,
        "padding": (
            float(args.padding)
            if args.padding is not None
            else (2.0 if args.close_up else 8.0)
        ),
    }

    args.output_scene.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_scene.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [
        str(args.blender.resolve()), "-b", "--python", str(ROOT / "tools/blender_flat_render.py"),
        "--", str(args.output_scene.resolve()), str(args.output_png.resolve()), "--statics", "all",
    ]
    result = subprocess.run(command, cwd=str(ROOT), check=False)
    if result.returncode:
        print(f"FAILURE: Blender render exited {result.returncode}")
        return result.returncode
    if not args.output_png.is_file():
        print("FAILURE: Blender produced no output PNG")
        return 1
    print(json.dumps({
        "wall_members": len(wall_meshes),
        "building_members": len(building_meshes),
        "scatter_members": len(scatter_meshes),
        "scatter_excluded": excluded,
        "filtered_scatter": str(args.scatter),
        "output_scene": str(args.output_scene),
        "output_png": str(args.output_png),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
