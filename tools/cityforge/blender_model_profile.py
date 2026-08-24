"""Blender-side native-scale model profiling for building rule kit Phase 2.

Pipeline position: measurement step of Phase 2 (spec:
``.opencode/runs/2026-08-21-building-generation-rule-kit/2026-08-22_phase2_implementation_spec.md``).
Invoked once by ``tools/cityforge/build_model_profiles.py`` through
``blender -b --python``. Follows the ``blender_wall_kit_slice.py`` job-JSON
pattern: fresh empty scene per mesh, io_scene_mw import at scale 0.01,
evaluated-depsgraph geometry in GU.

Per model it records: evaluated local bounds, vertex/face counts, a
ground-reaching XY polygon from the configured z band (z-slice method per
AGENTS.md — never a bare full-mesh AABB for footprints), robust bottom
evidence, a measurement-only principal XY axis, and an order-independent
SHA-256 geometry digest used as alias-equivalence evidence.

Unresolved or failed meshes are explicit failure rows; exit code 1 when any
mesh fails.

Job JSON::

    {
      "out": "<output json path>",
      "meshes": ["sky/x/sky_fk_house_02_a.nif", ...],
      "z_band_fractions": {"default": [0.05, 0.45], "overrides": {}},
      "bottom_percentile": 5.0,
      "digest_decimals": 4
    }
"""

from __future__ import annotations

import hashlib
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


def _argv_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round((p / 100.0) * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _convex_hull_xy(points: list[tuple[float, float]]) -> list[list[float]]:
    unique = sorted(set((round(float(x), 6), round(float(y), 6)) for x, y in points))
    if len(unique) <= 2:
        return [[round(x, 3), round(y, 3)] for x, y in unique]

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return [[round(x, 3), round(y, 3)] for x, y in lower[:-1] + upper[:-1]]


def _import_mesh(mesh: str, roots, resolver):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    config = nif_thumbs.resolved_config({}, layout="strip", resolution="1536x512")
    nif_thumbs._configure_engine(config)
    import_settings = {
        "scale_correction": 0.01,
        "normalize_to_position": False,
        "use_existing_materials": True,
        "ignore_collision_nodes": True,
        "ignore_animations": True,
        "reuse_meshes": False,
        "vertex_precision": 0.001,
    }
    document = {
        "scene_name": "ProcGen_Model_Profile",
        "import": import_settings,
        "meshes": [{"id": Path(mesh).stem, "mesh": mesh.replace("/", "\\"), "position": [0.0, 0.0, 0.0]}],
    }
    nif_import = render.setup_plugin(roots, import_settings)
    entries = render.resolve_meshes(document, roots, resolver)
    objects, _groups = render.import_meshes(entries, nif_import, import_settings)
    bpy.context.view_layer.update()
    return objects


def _collect_geometry(objects) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Evaluated world-space vertices (GU) and triangle index triples."""
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        matrix = evaluated.matrix_world
        base = len(vertices)
        try:
            mesh.calc_loop_triangles()
            for vertex in mesh.vertices:
                world = matrix @ Vector(vertex.co)
                vertices.append((float(world.x) * GU, float(world.y) * GU, float(world.z) * GU))
            for tri in mesh.loop_triangles:
                triangles.append(tuple(base + int(i) for i in tri.vertices))
        finally:
            evaluated.to_mesh_clear()
    return vertices, triangles


def _geometry_digest(vertices, triangles, decimals: int) -> str:
    """Order-independent digest: sorted rounded verts + sorted triangle coords."""
    rounded = sorted(
        tuple(round(c, decimals) for c in vertex) for vertex in vertices
    )
    tris = sorted(
        tuple(sorted(rounded_vertex for rounded_vertex in (
            tuple(round(c, decimals) for c in vertices[i]) for i in tri
        )))
        for tri in triangles
    )
    digest = hashlib.sha256()
    digest.update(repr(rounded).encode("utf-8"))
    digest.update(repr(tris).encode("utf-8"))
    return digest.hexdigest()


def _principal_axis_deg(points: list[tuple[float, float]]) -> float | None:
    """2D covariance principal axis of slice points; measurement only."""
    if len(points) < 3:
        return None
    n = float(len(points))
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points) / n
    syy = sum((p[1] - my) ** 2 for p in points) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points) / n
    if sxx == 0.0 and syy == 0.0:
        return None
    return round(math.degrees(0.5 * math.atan2(2.0 * sxy, sxx - syy)), 3)


def _measure(objects, bottom_percentile: float, digest_decimals: int, z_lo: float, z_hi: float) -> dict:
    vertices, triangles = _collect_geometry(objects)
    if not vertices:
        raise RuntimeError("no mesh geometry")
    xs_all = [v[0] for v in vertices]
    ys_all = [v[1] for v in vertices]
    zs_all = sorted(v[2] for v in vertices)
    slice_points = [(v[0], v[1]) for v in vertices if z_lo <= v[2] <= z_hi]
    if not slice_points:
        raise RuntimeError(f"empty slice band z={z_lo}..{z_hi}")
    xs = sorted(p[0] for p in slice_points)
    ys = sorted(p[1] for p in slice_points)
    z_min = zs_all[0]
    z_bottom = _percentile(zs_all, bottom_percentile)
    return {
        "vertex_count": len(vertices),
        "face_count": len(triangles),
        "bounds_local_gu": {
            "min": [round(min(xs_all), 3), round(min(ys_all), 3), round(z_min, 3)],
            "max": [round(max(xs_all), 3), round(max(ys_all), 3), round(zs_all[-1], 3)],
            "span": [round(max(xs_all) - min(xs_all), 3), round(max(ys_all) - min(ys_all), 3), round(zs_all[-1] - z_min, 3)],
        },
        "z_band_gu": [round(z_lo, 3), round(z_hi, 3)],
        "slice_vertex_count": len(slice_points),
        "ground_polygon_xy": _convex_hull_xy(slice_points),
        "slice_min_xy": [round(xs[0], 3), round(ys[0], 3)],
        "slice_max_xy": [round(xs[-1], 3), round(ys[-1], 3)],
        "slice_span_xy": [round(xs[-1] - xs[0], 3), round(ys[-1] - ys[0], 3)],
        "bottom_z_min_gu": round(z_min, 3),
        "bottom_z_percentile_gu": round(z_bottom, 3),
        "penetration_range_gu": round(z_bottom - z_min, 3),
        "principal_axis_xy_deg": _principal_axis_deg(slice_points),
        "geometry_digest": _geometry_digest(vertices, triangles, digest_decimals),
    }


def main() -> int:
    args = _argv_after_dash()
    if len(args) != 1:
        print("usage: blender -b --python blender_model_profile.py -- JOB.json", file=sys.stderr)
        return 2
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    out_path = Path(job["out"])
    meshes = list(job["meshes"])
    default_band = [float(v) for v in job.get("z_band_fractions", {}).get("default", [0.05, 0.45])]
    overrides = {
        k.replace("/", "\\").casefold(): [float(v) for v in band]
        for k, band in job.get("z_band_fractions", {}).get("overrides", {}).items()
    }
    bottom_percentile = float(job.get("bottom_percentile", 5.0))
    digest_decimals = int(job.get("digest_decimals", 4))

    roots, resolver = render.load_procgen_meshcheck()
    rows: list[dict] = []
    failures: list[str] = []
    for mesh in meshes:
        key = mesh.replace("/", "\\").casefold()
        frac_lo, frac_hi = overrides.get(key, default_band)
        print(f"[model-profile] {mesh} fracs={frac_lo}..{frac_hi}", flush=True)
        try:
            resolved = resolver(mesh, "mesh", roots=roots)
            if resolved is None:
                raise RuntimeError("unresolved under configured data roots")
            objects = _import_mesh(mesh, roots, resolver)
            zs: list[float] = []
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for obj in objects:
                if obj.type != "MESH":
                    continue
                evaluated = obj.evaluated_get(depsgraph)
                for corner in evaluated.bound_box:
                    world = evaluated.matrix_world @ Vector(corner)
                    zs.append(float(world.z) * GU)
            z_min, z_max = min(zs), max(zs)
            height = z_max - z_min
            z_lo = z_min + frac_lo * height
            z_hi = z_min + frac_hi * height
            band_fallback = False
            try:
                row = _measure(objects, bottom_percentile, digest_decimals, z_lo, z_hi)
            except RuntimeError as exc:
                if "empty slice band" not in str(exc):
                    raise
                # Recorded fallback: a model with no vertices in the body band
                # is measured over its full z range and explicitly flagged.
                band_fallback = True
                row = _measure(objects, bottom_percentile, digest_decimals, z_min, z_max)
            row["model_key"] = mesh.replace("/", "\\")
            row["resolved_path"] = str(resolved)
            row["band_fallback"] = band_fallback
            rows.append(row)
            print(
                f"[model-profile] {Path(mesh).name} verts={row['vertex_count']} faces={row['face_count']}"
                f" span={row['slice_span_xy']} fallback={band_fallback}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - report every failure explicitly
            failures.append(f"{mesh}: {exc}")
            print(f"[model-profile] FAIL {mesh}: {exc}", flush=True)

    payload = {
        "schema_version": 1,
        "unit": "gu",
        "origin": "native_nif_evaluated",
        "bottom_percentile": bottom_percentile,
        "meshes": rows,
        "failures": failures,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[model-profile] wrote {out_path} ok={len(rows)} failed={len(failures)}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
