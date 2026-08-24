#!/usr/bin/env python3
"""Extract semantic walk-surface anchors from evaluated NIF geometry.

Pipeline position
-----------------
This Blender-side measurement tool runs after wall-kit mesh discovery and
before kit authoring. A JSON job supplies every mesh, threshold, axis, and
selection rule. The output records upward-facing surface clusters and the
selected walkway or stair entry/exit anchors in native local GU.

The tool never infers a walk surface from a full bounding box. It fails closed
when the configured surface selection is absent or ambiguous.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "tools"))

import blender_flat_render as render  # noqa: E402
import nif_thumbs  # noqa: E402

GU = 100.0


def _args() -> Path:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(values) != 1:
        raise RuntimeError("usage: blender -b --python blender_mesh_walk_anchors.py -- JOB.json")
    return Path(values[0])


def _import_mesh(mesh_key: str, roots, resolver):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    settings = {
        "scale_correction": 0.01,
        "normalize_to_position": False,
        "use_existing_materials": True,
        "ignore_collision_nodes": True,
        "ignore_animations": True,
        "reuse_meshes": False,
        "vertex_precision": 0.001,
    }
    document = {
        "scene_name": "ProcGen_Walk_Anchor_Measurement",
        "import": settings,
        "meshes": [{"id": Path(mesh_key).stem, "mesh": mesh_key.replace("/", "\\"),
                    "position": [0.0, 0.0, 0.0]}],
    }
    importer = render.setup_plugin(roots, settings)
    entries = render.resolve_meshes(document, roots, resolver)
    if entries is None:
        raise RuntimeError(f"unresolved mesh {mesh_key}")
    objects, _groups = render.import_meshes(entries, importer, settings)
    bpy.context.view_layer.update()
    return objects


def _triangle_area(a: Vector, b: Vector, c: Vector) -> float:
    return 0.5 * (b - a).cross(c - a).length * GU * GU


def _surface_triangles(objects, normal_z_min: float) -> list[dict]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rows: list[dict] = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        matrix = evaluated.matrix_world
        try:
            for polygon in mesh.polygons:
                points = [matrix @ mesh.vertices[index].co for index in polygon.vertices]
                if len(points) < 3:
                    continue
                for index in range(1, len(points) - 1):
                    triangle = (points[0], points[index], points[index + 1])
                    raw_normal = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0])
                    if raw_normal.length == 0.0:
                        continue
                    normal = raw_normal.normalized()
                    if float(normal.z) < normal_z_min:
                        continue
                    coords = [[float(value.x) * GU, float(value.y) * GU,
                               float(value.z) * GU] for value in triangle]
                    rows.append({
                        "area_gu2": _triangle_area(*triangle),
                        "points": coords,
                        "normal": [float(normal.x), float(normal.y), float(normal.z)],
                        "plane_offset_gu": sum(
                            float(normal[index]) * coords[0][index] for index in range(3)
                        ),
                        "z_gu": sum(point[2] for point in coords) / 3.0,
                    })
        finally:
            evaluated.to_mesh_clear()
    if not rows:
        raise RuntimeError("no upward-facing evaluated triangles")
    return rows


def _clusters(triangles: list[dict], tolerance: float, minimum_area: float) -> list[dict]:
    groups: list[list[dict]] = []
    for triangle in sorted(triangles, key=lambda row: float(row["z_gu"])):
        if not groups or abs(float(triangle["z_gu"]) - sum(
            float(row["z_gu"]) for row in groups[-1]
        ) / len(groups[-1])) > tolerance:
            groups.append([triangle])
        else:
            groups[-1].append(triangle)
    output = []
    for group in groups:
        area = sum(float(row["area_gu2"]) for row in group)
        if area < minimum_area:
            continue
        points = [point for row in group for point in row["points"]]
        output.append({
            "z_gu": round(sum(float(row["z_gu"]) * float(row["area_gu2"])
                              for row in group) / area, 3),
            "area_gu2": round(area, 3),
            "bounds_min_gu": [round(min(point[axis] for point in points), 3)
                              for axis in range(3)],
            "bounds_max_gu": [round(max(point[axis] for point in points), 3)
                              for axis in range(3)],
            "triangle_count": len(group),
        })
    if not output:
        raise RuntimeError("no walk-surface cluster passed minimum_area_gu2")
    return output


def _edge_anchor(cluster: dict, axis: int, edge: str, lateral_axis: int) -> list[float]:
    minimum = cluster["bounds_min_gu"]
    maximum = cluster["bounds_max_gu"]
    point = [0.0, 0.0, float(cluster["z_gu"])]
    point[axis] = float(minimum[axis] if edge == "min" else maximum[axis])
    point[lateral_axis] = 0.5 * (float(minimum[lateral_axis]) + float(maximum[lateral_axis]))
    return [round(value, 3) for value in point]


def _sloped_planes(
    triangles: list[dict], normal_tolerance: float, plane_tolerance_gu: float,
    minimum_area: float,
) -> list[dict]:
    """Group coplanar upward triangles without confusing parallel rail caps."""

    groups: list[dict] = []
    for triangle in sorted(triangles, key=lambda row: float(row["area_gu2"]), reverse=True):
        normal = Vector(triangle["normal"])
        offset = float(triangle["plane_offset_gu"])
        match = next(
            (
                group for group in groups
                if normal.dot(Vector(group["normal"])) >= 1.0 - normal_tolerance
                and abs(offset - float(group["plane_offset_gu"])) <= plane_tolerance_gu
            ),
            None,
        )
        if match is None:
            match = {
                "normal": list(triangle["normal"]),
                "plane_offset_gu": offset,
                "triangles": [],
            }
            groups.append(match)
        match["triangles"].append(triangle)

    output = []
    for group in groups:
        area = sum(float(row["area_gu2"]) for row in group["triangles"])
        if area < minimum_area:
            continue
        points = [point for row in group["triangles"] for point in row["points"]]
        output.append({
            "area_gu2": round(area, 3),
            "bounds_min_gu": [round(min(point[axis] for point in points), 3)
                              for axis in range(3)],
            "bounds_max_gu": [round(max(point[axis] for point in points), 3)
                              for axis in range(3)],
            "normal": [round(float(value), 8) for value in group["normal"]],
            "plane_offset_gu": round(float(group["plane_offset_gu"]), 3),
            "triangle_count": len(group["triangles"]),
        })
    return sorted(output, key=lambda row: float(row["area_gu2"]), reverse=True)


def _slope_anchor(plane: dict, axis: int, edge: str, lateral_axis: int) -> list[float]:
    minimum = plane["bounds_min_gu"]
    maximum = plane["bounds_max_gu"]
    point = [0.0, 0.0, 0.0]
    point[axis] = float(minimum[axis] if edge == "min" else maximum[axis])
    point[lateral_axis] = 0.5 * (float(minimum[lateral_axis]) + float(maximum[lateral_axis]))
    normal = plane["normal"]
    if abs(float(normal[2])) < 1e-6:
        raise RuntimeError("selected slope plane is vertical")
    point[2] = (
        float(plane["plane_offset_gu"])
        - float(normal[axis]) * point[axis]
        - float(normal[lateral_axis]) * point[lateral_axis]
    ) / float(normal[2])
    return [round(value, 3) for value in point]


def _measure(entry: dict, roots, resolver, defaults: dict) -> dict:
    triangles = _surface_triangles(
        _import_mesh(str(entry["model_key"]), roots, resolver),
        float(entry.get("upward_normal_z_min", defaults["upward_normal_z_min"])),
    )
    clusters = _clusters(
        triangles,
        float(entry.get("z_cluster_tolerance_gu", defaults["z_cluster_tolerance_gu"])),
        float(entry.get("minimum_cluster_area_gu2", defaults["minimum_cluster_area_gu2"])),
    )
    axis_name = str(entry["longitudinal_axis"])
    axis = {"x": 0, "y": 1}[axis_name]
    lateral_axis = 1 - axis
    role = str(entry["role"])
    result = {"model_key": str(entry["model_key"]).replace("/", "\\"),
              "role": role, "surface_clusters": clusters}
    if role == "wall_walkway":
        selected = max(clusters, key=lambda row: float(row["area_gu2"]))
        result["walkway"] = {
            "surface_z_gu": selected["z_gu"],
            "end_a_local_gu": _edge_anchor(selected, axis, "min", lateral_axis),
            "end_b_local_gu": _edge_anchor(selected, axis, "max", lateral_axis),
            "lateral_bounds_gu": [selected["bounds_min_gu"][lateral_axis],
                                  selected["bounds_max_gu"][lateral_axis]],
        }
    elif role == "stair_treads":
        entry_cluster = min(clusters, key=lambda row: float(row["z_gu"]))
        exit_cluster = max(clusters, key=lambda row: float(row["z_gu"]))
        result["treads"] = {
            "entry_local_gu": _edge_anchor(entry_cluster, axis, "min", lateral_axis),
            "exit_local_gu": _edge_anchor(exit_cluster, axis, "max", lateral_axis),
            "lateral_bounds_gu": [entry_cluster["bounds_min_gu"][lateral_axis],
                                  entry_cluster["bounds_max_gu"][lateral_axis]],
            "rise_gu": round(float(exit_cluster["z_gu"]) - float(entry_cluster["z_gu"]), 3),
        }
    elif role == "segmented_sloped_walkway":
        horizontal = [
            cluster for cluster in clusters
            if float(cluster["bounds_max_gu"][2]) - float(cluster["bounds_min_gu"][2]) <= 1.0
        ]
        if len(horizontal) < 2:
            raise RuntimeError("segmented slope requires low and high horizontal landings")
        low_landing = min(horizontal, key=lambda row: float(row["z_gu"]))
        high_landing = max(horizontal, key=lambda row: float(row["z_gu"]))
        low_center = 0.5 * (
            float(low_landing["bounds_min_gu"][axis])
            + float(low_landing["bounds_max_gu"][axis])
        )
        high_center = 0.5 * (
            float(high_landing["bounds_min_gu"][axis])
            + float(high_landing["bounds_max_gu"][axis])
        )
        rises_positive = high_center > low_center
        entry_anchor = _edge_anchor(
            low_landing, axis, "min" if rises_positive else "max", lateral_axis
        )
        exit_anchor = _edge_anchor(
            high_landing, axis, "max" if rises_positive else "min", lateral_axis
        )
        rise = float(exit_anchor[2]) - float(entry_anchor[2])
        if rise < float(entry.get("minimum_rise_gu", 64.0)):
            raise RuntimeError("segmented slope did not pass minimum_rise_gu")
        result["sloped_walkway"] = {
            "entry_local_gu": entry_anchor,
            "exit_local_gu": exit_anchor,
            "lateral_bounds_gu": [low_landing["bounds_min_gu"][lateral_axis],
                                  low_landing["bounds_max_gu"][lateral_axis]],
            "rise_gu": round(rise, 3),
            "measurement": "lowest and highest evaluated horizontal terminal landings",
        }
    elif role == "sloped_walkway":
        planes = _sloped_planes(
            triangles,
            float(entry.get("normal_tolerance", 1e-5)),
            float(entry.get("plane_tolerance_gu", 0.5)),
            float(entry.get("minimum_cluster_area_gu2", defaults["minimum_cluster_area_gu2"])),
        )
        minimum_rise = float(entry.get("minimum_rise_gu", 64.0))
        candidates = []
        for plane in planes:
            start = _slope_anchor(plane, axis, "min", lateral_axis)
            end = _slope_anchor(plane, axis, "max", lateral_axis)
            if abs(float(end[2]) - float(start[2])) >= minimum_rise:
                candidates.append((plane, start, end))
        if not candidates:
            raise RuntimeError("no sloped walk plane passed minimum_rise_gu")
        selected, start, end = max(candidates, key=lambda row: float(row[0]["area_gu2"]))
        landing_z_tolerance = float(entry.get("landing_z_tolerance_gu", 5.0))
        landing_gap_tolerance = float(entry.get("landing_gap_tolerance_gu", 32.0))
        low, high = (start, end) if start[2] <= end[2] else (end, start)
        low_edge = float(low[axis])
        high_edge = float(high[axis])
        low_landing = [
            cluster for cluster in clusters
            if float(cluster["bounds_max_gu"][2]) - float(cluster["bounds_min_gu"][2]) <= 1.0
            and abs(float(cluster["z_gu"]) - float(low[2])) <= landing_z_tolerance
            and float(cluster["bounds_max_gu"][axis]) >= low_edge - landing_gap_tolerance
            and float(cluster["bounds_min_gu"][axis]) <= low_edge + landing_gap_tolerance
        ]
        high_landing = [
            cluster for cluster in clusters
            if float(cluster["bounds_max_gu"][2]) - float(cluster["bounds_min_gu"][2]) <= 1.0
            and abs(float(cluster["z_gu"]) - float(high[2])) <= landing_z_tolerance
            and float(cluster["bounds_max_gu"][axis]) >= high_edge - landing_gap_tolerance
            and float(cluster["bounds_min_gu"][axis]) <= high_edge + landing_gap_tolerance
        ]
        if low_landing:
            selected_low = min(low_landing, key=lambda row: float(row["bounds_min_gu"][axis]))
            low[axis] = float(selected_low["bounds_min_gu"][axis])
            low[2] = float(selected_low["z_gu"])
        if high_landing:
            selected_high = max(high_landing, key=lambda row: float(row["bounds_max_gu"][axis]))
            high[axis] = float(selected_high["bounds_max_gu"][axis])
            high[2] = float(selected_high["z_gu"])
        result["sloped_walkway"] = {
            "entry_local_gu": [round(value, 3) for value in low],
            "exit_local_gu": [round(value, 3) for value in high],
            "lateral_bounds_gu": [selected["bounds_min_gu"][lateral_axis],
                                  selected["bounds_max_gu"][lateral_axis]],
            "rise_gu": round(float(high[2]) - float(low[2]), 3),
            "selected_plane": selected,
        }
    else:
        raise RuntimeError(f"unsupported role {role!r}")
    return result


def main() -> int:
    job_path = _args()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    defaults = job["defaults"]
    roots, resolver = render.load_procgen_meshcheck()
    rows = [_measure(entry, roots, resolver, defaults) for entry in job["meshes"]]
    payload = {"schema_version": 1, "unit": "gu", "meshes": rows}
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[walk-anchors] wrote {out} meshes={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
