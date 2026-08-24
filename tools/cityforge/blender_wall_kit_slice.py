"""Blender-side z-slice footprint measurement for wall-kit NIFs (W1).

Pipeline position: stage W1 of the wall kit system. Invoked by
``tools/wall_kit_extract.py`` through ``blender -b --python``. Measures each
requested mesh with the z-slice method (AGENTS.md geometry conventions):
XY extents come only from evaluated vertices inside a z-band covering the
wall body, never from the full-mesh AABB (roofs/overhangs/courtyards would
inflate footprints).

Inputs (single argv path after ``--``): a job JSON::

    {
      "out": "<output json path>",
      "meshes": ["sky/x/sky_ex_cs_re_wl_04.nif", ...],
      "z_band_fractions": {"default": [0.05, 0.45],
                           "overrides": {"sky/x/sky_ex_cs_re_gh_01.nif": [0.05, 0.6]}},
      "percentiles": [5.0, 95.0],
      "square_ratio": 1.15
    }

Output JSON: one row per mesh with full AABB (diagnostic), slice band in GU,
slice min/max and percentile XY extents, long-axis choice, and pivot-relative
2D end connection points ``end_a_local``/``end_b_local`` on the long axis.
Meshes that fail to resolve are reported as unresolved rows, never silently
dropped.
"""

from __future__ import annotations

import json
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
    """Return the evaluated slice outline, preserving circular tower profiles."""
    unique = sorted(set((round(float(x), 6), round(float(y), 6)) for x, y in points))
    if len(unique) <= 2:
        return [[round(x, 3), round(y, 3)] for x, y in unique]

    def cross(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
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
        "scene_name": "ProcGen_Wall_Kit_Slice",
        "import": import_settings,
        "meshes": [{"id": Path(mesh).stem, "mesh": mesh.replace("/", "\\"), "position": [0.0, 0.0, 0.0]}],
    }
    nif_import = render.setup_plugin(roots, import_settings)
    entries = render.resolve_meshes(document, roots, resolver)
    objects, _groups = render.import_meshes(entries, nif_import, import_settings)
    bpy.context.view_layer.update()
    return objects


def _measure(objects, z_lo: float, z_hi: float, percentiles: list[float], square_ratio: float) -> dict:
    full_min = [float("inf")] * 3
    full_max = [float("-inf")] * 3
    xs: list[float] = []
    ys: list[float] = []
    slice_points: list[tuple[float, float]] = []
    zs: list[float] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        matrix = evaluated.matrix_world
        try:
            for vertex in mesh.vertices:
                world = matrix @ Vector(vertex.co)
                x, y, z = float(world.x) * GU, float(world.y) * GU, float(world.z) * GU
                full_min[0] = min(full_min[0], x)
                full_min[1] = min(full_min[1], y)
                full_min[2] = min(full_min[2], z)
                full_max[0] = max(full_max[0], x)
                full_max[1] = max(full_max[1], y)
                full_max[2] = max(full_max[2], z)
                zs.append(z)
                if z_lo <= z <= z_hi:
                    xs.append(x)
                    ys.append(y)
                    slice_points.append((x, y))
        finally:
            evaluated.to_mesh_clear()
    if full_min[0] == float("inf"):
        raise RuntimeError("no mesh geometry")
    if not xs:
        raise RuntimeError(f"empty slice band z={z_lo}..{z_hi}")
    xs.sort()
    ys.sort()
    span_x = xs[-1] - xs[0]
    span_y = ys[-1] - ys[0]
    long_axis = "x" if span_x >= span_y else "y"
    along = xs if long_axis == "x" else ys
    across = ys if long_axis == "x" else xs
    cross_center = 0.5 * (across[0] + across[-1])
    if long_axis == "x":
        end_a = [round(xs[0], 3), round(cross_center, 3)]
        end_b = [round(xs[-1], 3), round(cross_center, 3)]
    else:
        end_a = [round(cross_center, 3), round(ys[0], 3)]
        end_b = [round(cross_center, 3), round(ys[-1], 3)]
    row = {
        "full_min": [round(v, 3) for v in full_min],
        "full_max": [round(v, 3) for v in full_max],
        "full_span": [round(full_max[i] - full_min[i], 3) for i in range(3)],
        "z_band_gu": [round(z_lo, 3), round(z_hi, 3)],
        "slice_vertex_count": len(xs),
        "slice_outline_xy": _convex_hull_xy(slice_points),
        "slice_min_xy": [round(xs[0], 3), round(ys[0], 3)],
        "slice_max_xy": [round(xs[-1], 3), round(ys[-1], 3)],
        "slice_span_xy": [round(span_x, 3), round(span_y, 3)],
        "long_axis": long_axis,
        "square": max(span_x, span_y) < square_ratio * min(span_x, span_y),
        "length_gu": round(along[-1] - along[0], 3),
        "thickness_gu": round(across[-1] - across[0], 3),
        "height_gu": round(max(zs) - min(zs), 3),
        "end_a_local": end_a,
        "end_b_local": end_b,
    }
    for p in percentiles:
        row[f"slice_p{int(p):02d}_xy"] = [
            round(_percentile(xs, p), 3),
            round(_percentile(ys, p), 3),
        ]
    return row


def main() -> int:
    args = _argv_after_dash()
    if len(args) != 1:
        print("usage: blender -b --python blender_wall_kit_slice.py -- JOB.json", file=sys.stderr)
        return 2
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    out_path = Path(job["out"])
    meshes = list(job["meshes"])
    default_band = [float(v) for v in job.get("z_band_fractions", {}).get("default", [0.05, 0.45])]
    overrides = {
        k.replace("/", "\\").casefold(): [float(v) for v in band]
        for k, band in job.get("z_band_fractions", {}).get("overrides", {}).items()
    }
    percentiles = [float(p) for p in job.get("percentiles", [5.0, 95.0])]
    square_ratio = float(job.get("square_ratio", 1.15))

    roots, resolver = render.load_procgen_meshcheck()
    rows: list[dict] = []
    failures: list[str] = []
    for mesh in meshes:
        key = mesh.replace("/", "\\").casefold()
        frac_lo, frac_hi = overrides.get(key, default_band)
        print(f"[wall-kit-slice] {mesh} fracs={frac_lo}..{frac_hi}", flush=True)
        try:
            entries_ok = resolver(mesh, "mesh", roots=roots)
            if entries_ok is None:
                raise RuntimeError("unresolved under configured data roots")
            objects = _import_mesh(mesh, roots, resolver)
            # Band needs absolute z; compute the z range from evaluated
            # bounding boxes first, then slice inside it.
            depsgraph = bpy.context.evaluated_depsgraph_get()
            z_min, z_max = float("inf"), float("-inf")
            for obj in objects:
                if obj.type != "MESH":
                    continue
                evaluated = obj.evaluated_get(depsgraph)
                for corner in evaluated.bound_box:
                    world = evaluated.matrix_world @ Vector(corner)
                    z_min = min(z_min, float(world.z) * GU)
                    z_max = max(z_max, float(world.z) * GU)
            height = z_max - z_min
            z_lo = z_min + frac_lo * height
            z_hi = z_min + frac_hi * height
            band_fallback = False
            try:
                row = _measure(objects, z_lo, z_hi, percentiles, square_ratio)
            except RuntimeError as exc:
                if "empty slice band" not in str(exc):
                    raise
                # Explicit, recorded fallback: some non-wall pieces (block or
                # bridge decks) have no vertices in the default body band.
                # Measure over the full z range instead and flag it so the
                # extraction report can show which footprints are full-AABB.
                band_fallback = True
                row = _measure(objects, z_min, z_max, percentiles, square_ratio)
            row["model_key"] = mesh.replace("/", "\\")
            row["band_fallback"] = band_fallback
            rows.append(row)
            print(
                f"[wall-kit-slice] {Path(mesh).name} len={row['length_gu']} thick={row['thickness_gu']}"
                f" h={row['height_gu']} long={row['long_axis']} square={row['square']}",
                flush=True,
            )
            if band_fallback:
                print(f"[wall-kit-slice] NOTE {Path(mesh).name}: default band empty, used full z range", flush=True)
        except Exception as exc:  # noqa: BLE001 - report every failure explicitly
            failures.append(f"{mesh}: {exc}")
            print(f"[wall-kit-slice] FAIL {mesh}: {exc}", flush=True)

    payload = {
        "schema_version": 1,
        "unit": "gu",
        "origin": "native_nif_slice",
        "meshes": rows,
        "failures": failures,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[wall-kit-slice] wrote {out_path} ok={len(rows)} failed={len(failures)}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
