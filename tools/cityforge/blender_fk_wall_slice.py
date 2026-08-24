"""Measure roof AABB vs first-floor wall AABB for Falkreath house shells.

Writes JSON with per-face overhang (roof AABB face minus wall-slice face).
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

SLICES = {
    "sky/x/sky_FK_house_01_a.nif": (-20.0, 120.0),
    "sky/x/sky_FK_house_02_a.nif": (-20.0, 120.0),
    "sky/x/sky_FK_house_03_a.nif": (-20.0, 120.0),
    "sky/x/sky_FK_house_04_a.nif": (-20.0, 140.0),
    "sky/x/sky_FK_house_05_a.nif": (0.0, 140.0),
    "sky/x/sky_FK_house_06_a.nif": (0.0, 140.0),
    "sky/x/sky_FK_house_07_a.nif": (0.0, 180.0),
    "sky/x/sky_FK_house_08_a.nif": (160.0, 320.0),
    "sky/x/sky_FK_house_09_a.nif": (0.0, 180.0),
    "sky/x/sky_FK_house_10_a.nif": (0.0, 220.0),
    "sky/x/sky_FK_house_11_a.nif": (0.0, 140.0),
    "sky/x/sky_FK_house_12_a.nif": (0.0, 140.0),
}


def _argv_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def _import(mesh: str, roots, resolver):
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
        "scene_name": "ProcGen_FK_Wall_Slice",
        "import": import_settings,
        "meshes": [{"id": Path(mesh).stem, "mesh": mesh.replace("/", "\\"), "position": [0.0, 0.0, 0.0]}],
    }
    nif_import = render.setup_plugin(roots, import_settings)
    entries = render.resolve_meshes(document, roots, resolver)
    objects, _groups = render.import_meshes(entries, nif_import, import_settings)
    bpy.context.view_layer.update()
    return objects


def _full_and_slice(objects, z_lo: float, z_hi: float) -> dict:
    full_min = [float("inf")] * 3
    full_max = [float("-inf")] * 3
    wall_min = [float("inf")] * 2
    wall_max = [float("-inf")] * 2
    depsgraph = bpy.context.evaluated_depsgraph_get()
    n_slice = 0
    xs: list[float] = []
    ys: list[float] = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        matrix = evaluated.matrix_world
        try:
            for vertex in mesh.vertices:
                world = matrix @ vertex.co
                x, y, z = float(world.x) * GU, float(world.y) * GU, float(world.z) * GU
                full_min[0] = min(full_min[0], x)
                full_min[1] = min(full_min[1], y)
                full_min[2] = min(full_min[2], z)
                full_max[0] = max(full_max[0], x)
                full_max[1] = max(full_max[1], y)
                full_max[2] = max(full_max[2], z)
                if z_lo <= z <= z_hi:
                    n_slice += 1
                    wall_min[0] = min(wall_min[0], x)
                    wall_min[1] = min(wall_min[1], y)
                    wall_max[0] = max(wall_max[0], x)
                    wall_max[1] = max(wall_max[1], y)
                    xs.append(x)
                    ys.append(y)
        finally:
            evaluated.to_mesh_clear()
    xs.sort()
    ys.sort()

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, max(0, int(round((p / 100.0) * (len(values) - 1)))))
        return round(values[idx], 3)

    overhang = {
        "neg_x": round(wall_min[0] - full_min[0], 3),
        "pos_x": round(full_max[0] - wall_max[0], 3),
        "neg_y": round(wall_min[1] - full_min[1], 3),
        "pos_y": round(full_max[1] - wall_max[1], 3),
    }
    return {
        "z_slice_gu": [z_lo, z_hi],
        "slice_vertex_count": n_slice,
        "full_min": [round(v, 3) for v in full_min],
        "full_max": [round(v, 3) for v in full_max],
        "wall_min_xy": [round(v, 3) for v in wall_min],
        "wall_max_xy": [round(v, 3) for v in wall_max],
        "wall_p05_xy": [pct(xs, 5), pct(ys, 5)],
        "wall_p95_xy": [pct(xs, 95), pct(ys, 95)],
        "overhang_gu": overhang,
    }


def main() -> int:
    args = _argv_after_dash()
    out_path = Path(args[0]) if args else WORKSPACE / "configs" / "kits" / "falkreath" / "wall_overhang.json"
    roots, resolver = render.load_procgen_meshcheck()
    rows = []
    for mesh, (z_lo, z_hi) in SLICES.items():
        print(f"[wall-slice] {mesh} z={z_lo}..{z_hi}", flush=True)
        objects = _import(mesh, roots, resolver)
        row = {"model_key": mesh.replace("/", "\\"), **_full_and_slice(objects, z_lo, z_hi)}
        print(json.dumps(row["overhang_gu"]), flush=True)
        rows.append(row)
    payload = {"schema_version": 1, "unit": "gu", "meshes": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[wall-slice] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
