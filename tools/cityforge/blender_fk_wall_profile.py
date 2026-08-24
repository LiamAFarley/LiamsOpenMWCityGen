#!/usr/bin/env python3
"""Extract evaluated horizontal wall sections for Falkreath kit shells.

Inputs are one or more Falkreath shell NIF keys resolved through the existing
Blender mesh-check importer. Output is raw JSON under ``output/`` containing
GU-space triangle intersection segments, source triangle normals/objects,
sample heights, and source triangle outlines for diagnostics. No topology,
hulls, facade semantics, or canonical profile data are produced here; those
belong to ``measure_fk_wall_profiles.py`` in the host Python environment.
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
SAMPLE_STEP_GU = 20.0
PLANE_EPSILON_GU = 0.01


def _args() -> tuple[Path, list[str]]:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    out = Path(values[0]) if values else WORKSPACE / "output" / "falkreath_wall_sections.raw.json"
    meshes = values[1:] or [f"sky/x/sky_FK_house_{i:02d}_a.nif" for i in range(1, 13)]
    return out, meshes


def _import_all(mesh_keys: list[str], roots, resolver):
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
    document = {"scene_name": "ProcGen_FK_Wall_Profile", "import": settings,
                "meshes": [{"id": Path(key).stem, "mesh": key.replace("/", "\\"),
                             "position": [0.0, 0.0, 0.0]} for key in mesh_keys]}
    importer = render.setup_plugin(roots, settings)
    entries = render.resolve_meshes(document, roots, resolver)
    objects, _groups = render.import_meshes(entries, importer, settings)
    bpy.context.view_layer.update()
    grouped = {key.replace("/", "\\"): [] for key in mesh_keys}
    for obj in objects:
        key = str(obj.get("procgen_source_mesh_key", ""))
        if key in grouped:
            grouped[key].append(obj)
    return grouped


def _normal(a: Vector, b: Vector, c: Vector) -> Vector:
    n = (b - a).cross(c - a)
    return n.normalized() if n.length else Vector((0.0, 0.0, 0.0))


def _point(v: Vector) -> list[float]:
    return [round(float(v.x) * GU, 4), round(float(v.y) * GU, 4), round(float(v.z) * GU, 4)]


def _triangle_hit(points: list[Vector], z: float) -> list[list[float]] | None:
    hits: list[Vector] = []
    for a, b in zip(points, points[1:] + points[:1]):
        # Section input vectors are already in GU (triangle evidence is
        # serialized after the world-space metre→GU conversion).
        da, db = a.z - z, b.z - z
        if (da < 0.0 < db) or (db < 0.0 < da):
            t = da / (da - db)
            hits.append(a + (b - a) * t)
    if len(hits) != 2:
        return None
    return [[round(float(v.x), 4), round(float(v.y), 4), round(float(v.z), 4)] for v in hits]


def _shell(mesh_key: str, objects) -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    triangles: list[dict] = []
    z_min, z_max = float("inf"), float("-inf")
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        matrix = evaluated.matrix_world
        try:
            for poly in mesh.polygons:
                if len(poly.vertices) < 3:
                    continue
                verts = [matrix @ mesh.vertices[i].co for i in poly.vertices]
                for i in range(1, len(verts) - 1):
                    tri = [verts[0], verts[i], verts[i + 1]]
                    normal = _normal(*tri)
                    coords = [_point(v) for v in tri]
                    z_values = [p[2] for p in coords]
                    z_min, z_max = min(z_min, *z_values), max(z_max, *z_values)
                    triangles.append({"object": obj.name, "points": coords,
                                      "normal": [round(float(normal.x), 6), round(float(normal.y), 6), round(float(normal.z), 6)]})
        finally:
            evaluated.to_mesh_clear()
    if not triangles:
        raise RuntimeError(f"no evaluated mesh triangles for {mesh_key}")
    vertex_events = sorted({p[2] for tri in triangles for p in tri["points"]})
    # A vertex height is a geometry-change event only when the set of
    # triangles crossing the plane changes. Coplanar tessellation seams occur
    # at many repeated heights in these NIFs and must not create hundreds of
    # redundant near-duplicate sections.
    intervals = [(min(p[2] for p in tri["points"]), max(p[2] for p in tri["points"]))
                 for tri in triangles if abs(tri["normal"][2]) <= math.sin(math.radians(15.0))]
    events = []
    for event in vertex_events:
        below = sum(lo < event - PLANE_EPSILON_GU < hi for lo, hi in intervals)
        above = sum(lo < event + PLANE_EPSILON_GU < hi for lo, hi in intervals)
        if below != above:
            events.append(event)
    samples = set()
    first = math.floor(z_min / SAMPLE_STEP_GU) * SAMPLE_STEP_GU
    z = first
    while z <= z_max:
        samples.add(round(z, 4))
        z += SAMPLE_STEP_GU
    for event in events:
        if z_min <= event <= z_max:
            samples.add(round(event - PLANE_EPSILON_GU, 4))
            samples.add(round(event + PLANE_EPSILON_GU, 4))
    sections = []
    for z in sorted(v for v in samples if z_min - 1e-4 <= v <= z_max + 1e-4):
        segments = []
        for tri in triangles:
            hit = _triangle_hit([Vector(p) for p in tri["points"]], z)
            if hit:
                segments.append({"a": hit[0], "b": hit[1], "object": tri["object"],
                                 "normal": tri["normal"], "z_gu": z})
        sections.append({"z_gu": z, "segments": segments})
    return {"model_key": mesh_key.replace("/", "\\"), "unit": "gu", "triangles": triangles,
            "z_min_gu": round(z_min, 4), "z_max_gu": round(z_max, 4), "sections": sections}


def main() -> int:
    out, meshes = _args()
    roots, resolver = render.load_procgen_meshcheck()
    shells = []
    grouped = _import_all(meshes, roots, resolver)
    for mesh in meshes:
        print(f"[fk-wall-profile] extracting {mesh}", flush=True)
        shells.append(_shell(mesh, grouped.get(mesh.replace("/", "\\"), [])))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": 1, "unit": "gu", "shells": shells},
                              indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[fk-wall-profile] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
