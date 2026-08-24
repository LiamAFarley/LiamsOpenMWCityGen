"""Measure a wall stair's actual high/low end orientation in Blender.

Pipeline position
-----------------
This is a read-only W1 evidence tool. It imports fresh NIF geometry using the
same resolver and scale correction as ``blender_wall_kit_slice.py``, then
reports robust Z statistics at both ends of the measured long axis. It does
not decide how a composer should place the stair.

Input job JSON::

    {"out": "...", "meshes": ["sky/x/sky_ex_cs_re_st_01.nif"],
     "axis_ranges": {"sky/x/sky_ex_cs_re_st_01.nif": [-501.12, -190.08]}}

The endpoint bands are controlled by ``end_fraction`` (default ``0.15``).
Output records include endpoint medians/percentiles and the measured direction
(``high_x_end`` or ``low_x_end``), so a caller cannot silently assume that
local +X is uphill.
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


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        raise RuntimeError("no evaluated mesh vertices")
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return values[index]


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
        "scene_name": "ProcGen_Wall_Stair_Audit",
        "import": import_settings,
        "meshes": [{"id": Path(mesh).stem, "mesh": mesh.replace("/", "\\"), "position": [0.0, 0.0, 0.0]}],
    }
    nif_import = render.setup_plugin(roots, import_settings)
    entries = render.resolve_meshes(document, roots, resolver)
    if entries is None:
        raise RuntimeError("mesh unresolved under configured data roots")
    objects, _groups = render.import_meshes(entries, nif_import, import_settings)
    bpy.context.view_layer.update()
    return objects


def _measure(objects, end_fraction: float, axis_range: list[float] | None) -> dict:
    vertices: list[tuple[float, float, float]] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            matrix = evaluated.matrix_world
            vertices.extend(
                (
                    float(world.x) * GU,
                    float(world.y) * GU,
                    float(world.z) * GU,
                )
                for vertex in mesh.vertices
                for world in [matrix @ Vector(vertex.co)]
            )
        finally:
            evaluated.to_mesh_clear()
    if not vertices:
        raise RuntimeError("no evaluated mesh vertices")
    xs = [row[0] for row in vertices]
    x_min, x_max = (axis_range if axis_range is not None else [min(xs), max(xs)])
    span = x_max - x_min
    if span <= 0.0:
        raise RuntimeError("stair has no local X span")
    band = max(span * end_fraction, 1e-6)
    low = [row[2] for row in vertices if row[0] <= x_min + band]
    high = [row[2] for row in vertices if row[0] >= x_max - band]
    low_rows = [row for row in vertices if row[0] <= x_min + band]
    high_rows = [row for row in vertices if row[0] >= x_max - band]
    low_median = _percentile(low, 0.5)
    high_median = _percentile(high, 0.5)
    return {
        "unit": "gu",
        "x_span_gu": round(span, 3),
        "end_fraction": end_fraction,
        "low_x_end": {
            "y_span_gu": round(max(row[1] for row in low_rows) - min(row[1] for row in low_rows), 3),
            "z_p10_gu": round(_percentile(low, 0.1), 3),
            "z_median_gu": round(low_median, 3),
            "z_p90_gu": round(_percentile(low, 0.9), 3),
        },
        "high_x_end": {
            "y_span_gu": round(max(row[1] for row in high_rows) - min(row[1] for row in high_rows), 3),
            "z_p10_gu": round(_percentile(high, 0.1), 3),
            "z_median_gu": round(high_median, 3),
            "z_p90_gu": round(_percentile(high, 0.9), 3),
        },
        "higher_end": "high_x_end" if high_median > low_median else "low_x_end",
        "median_rise_gu": round(high_median - low_median, 3),
    }


def main() -> int:
    args = _argv_after_dash()
    if len(args) != 1:
        print("usage: blender -b --python blender_wall_stair_audit.py -- JOB.json", file=sys.stderr)
        return 2
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    end_fraction = float(job.get("end_fraction", 0.15))
    roots, resolver = render.load_procgen_meshcheck()
    rows = []
    failures = []
    for mesh in job["meshes"]:
        try:
            if resolver(mesh, "mesh", roots=roots) is None:
                raise RuntimeError("mesh unresolved under configured data roots")
            axis_range = job.get("axis_ranges", {}).get(mesh)
            rows.append({"mesh": mesh.replace("/", "\\"), **_measure(_import_mesh(mesh, roots, resolver), end_fraction, axis_range)})
        except Exception as exc:  # noqa: BLE001 - report every requested mesh
            failures.append(f"{mesh}: {exc}")
    payload = {"schema_version": 1, "meshes": rows, "failures": failures}
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[wall-stair-audit] wrote {out} ok={len(rows)} failed={len(failures)}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
