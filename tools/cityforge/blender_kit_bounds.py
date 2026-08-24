"""Blender-side AABB dump for kit NIFs, in TES3 game units.

Invoked by ``measure_fk_kit.py``. Imports each mesh through the same
io_scene_mw path as nif_thumbs, unions evaluated world bounds, writes JSON.
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

GU_PER_BLENDER = 100.0


def _argv_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def _union_bounds_gu(objects) -> dict[str, list[float]] | None:
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            world = evaluated.matrix_world @ Vector(corner)
            for axis in range(3):
                value = float(world[axis]) * GU_PER_BLENDER
                mn[axis] = min(mn[axis], value)
                mx[axis] = max(mx[axis], value)
    if mn[0] == float("inf"):
        return None
    return {
        "min": [round(v, 3) for v in mn],
        "max": [round(v, 3) for v in mx],
        "span": [round(mx[i] - mn[i], 3) for i in range(3)],
        "center": [round(0.5 * (mn[i] + mx[i]), 3) for i in range(3)],
    }


def measure_one(mesh: str, roots, resolver, nif_import) -> dict:
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
    relative = mesh.replace("/", "\\")
    document = {
        "scene_name": "ProcGen_Kit_Bounds",
        "import": import_settings,
        "meshes": [{"id": Path(mesh).stem, "mesh": relative, "position": [0.0, 0.0, 0.0]}],
    }
    nif_import = render.setup_plugin(roots, import_settings)
    entries = render.resolve_meshes(document, roots, resolver)
    objects, _groups = render.import_meshes(entries, nif_import, import_settings)
    bpy.context.view_layer.update()
    bounds = _union_bounds_gu(objects)
    if bounds is None:
        raise RuntimeError(f"no mesh bounds for {mesh}")
    span = bounds["span"]
    thin_axis = min(range(3), key=lambda axis: span[axis])
    return {
        "model_key": relative,
        "bounds_gu": bounds,
        "thin_axis": ["x", "y", "z"][thin_axis],
        "thin_span_gu": span[thin_axis],
    }


def main() -> int:
    args = _argv_after_dash()
    if len(args) < 2:
        print("usage: blender -b --python blender_kit_bounds.py -- OUT.json mesh [mesh...]", file=sys.stderr)
        return 2
    out_path = Path(args[0])
    meshes = args[1:]
    roots, resolver = render.load_procgen_meshcheck()
    nif_import = None
    rows = []
    for mesh in meshes:
        print(f"[kit-bounds] {mesh}", flush=True)
        rows.append(measure_one(mesh, roots, resolver, nif_import))
    payload = {"schema_version": 1, "unit": "gu", "origin": "native_nif", "meshes": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[kit-bounds] wrote {out_path} count={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
