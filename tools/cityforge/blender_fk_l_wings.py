"""Cluster first-floor wall planes for L / U / recessed Falkreath shells.

AABB faces cut through courtyards. X/Y clusters are the actual wall planes.
Inner courtyard faces are the interior clusters, not the AABB extremes.

    blender -b --python tools/cityforge/blender_fk_l_wings.py -- configs/kits/falkreath/house_wings.json

Optional extra meshes after the out path. Default list is the non-rectangle
shells plus remaining houses that still need a plane check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy  # type: ignore

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "tools"))

import blender_flat_render as render  # noqa: E402
import nif_thumbs  # noqa: E402

GU = 100.0
MESH_SLICES = {
    "sky/x/sky_FK_house_04_a.nif": (-20.0, 140.0),
    "sky/x/sky_FK_house_06_a.nif": (0.0, 140.0),
    "sky/x/sky_FK_house_07_a.nif": (0.0, 180.0),
    "sky/x/sky_FK_house_08_a.nif": (160.0, 320.0),
    "sky/x/sky_FK_house_09_a.nif": (0.0, 180.0),
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
        "scene_name": "ProcGen_FK_L_Wings",
        "import": import_settings,
        "meshes": [{"id": Path(mesh).stem, "mesh": mesh.replace("/", "\\"), "position": [0.0, 0.0, 0.0]}],
    }
    nif_import = render.setup_plugin(roots, import_settings)
    entries = render.resolve_meshes(document, roots, resolver)
    objects, _groups = render.import_meshes(entries, nif_import, import_settings)
    bpy.context.view_layer.update()
    return objects


def _slice_xy(objects, z_lo: float, z_hi: float) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
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
                if z_lo <= z <= z_hi:
                    xs.append(x)
                    ys.append(y)
        finally:
            evaluated.to_mesh_clear()
    return xs, ys


def _cluster_axis(values: list[float], bin_gu: float = 20.0) -> list[dict]:
    buckets: dict[int, list[float]] = {}
    for value in values:
        key = int(round(value / bin_gu))
        buckets.setdefault(key, []).append(value)
    rows = []
    for key, group in sorted(buckets.items()):
        if len(group) < 3:
            continue
        rows.append(
            {
                "count": len(group),
                "mean": round(sum(group) / len(group), 3),
                "min": round(min(group), 3),
                "max": round(max(group), 3),
            }
        )
    return rows


def _measure(mesh: str, xs: list[float], ys: list[float], z_slice: tuple[float, float]) -> dict:
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    x_walls = _cluster_axis(xs)
    y_walls = _cluster_axis(ys)
    print(f"[l-wings] {mesh} verts={len(xs)} wall=({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f})", flush=True)
    print(f"[l-wings] X {[(r['mean'], r['count']) for r in x_walls]}", flush=True)
    print(f"[l-wings] Y {[(r['mean'], r['count']) for r in y_walls]}", flush=True)
    x_means = [row["mean"] for row in x_walls]
    y_means = [row["mean"] for row in y_walls]
    x_outer = [x_means[0], x_means[-1]] if x_means else [x0, x1]
    y_outer = [y_means[0], y_means[-1]] if y_means else [y0, y1]
    x_inner = [
        value for value in x_means if abs(value - x_outer[0]) > 80 and abs(value - x_outer[1]) > 80
    ]
    y_inner = [
        value for value in y_means if abs(value - y_outer[0]) > 80 and abs(value - y_outer[1]) > 80
    ]
    return {
        "model_key": mesh.replace("/", "\\"),
        "z_slice_gu": list(z_slice),
        "slice_vertex_count": len(xs),
        "wall_min_xy": [round(x0, 3), round(y0, 3)],
        "wall_max_xy": [round(x1, 3), round(y1, 3)],
        "x_clusters": x_walls,
        "y_clusters": y_walls,
        "x_inner_means": [round(v, 3) for v in x_inner],
        "y_inner_means": [round(v, 3) for v in y_inner],
    }


def main() -> int:
    args = _argv_after_dash()
    out_path = Path(args[0]) if args else WORKSPACE / "configs" / "kits" / "falkreath" / "house_wings.json"
    extra = args[1:]
    meshes = extra or list(MESH_SLICES)
    roots, resolver = render.load_procgen_meshcheck()
    rows = []
    for mesh in meshes:
        z_slice = MESH_SLICES.get(mesh, (0.0, 140.0))
        objects = _import(mesh, roots, resolver)
        xs, ys = _slice_xy(objects, *z_slice)
        if not xs:
            print(f"[l-wings] skip {mesh}: no slice verts z={z_slice}", flush=True)
            continue
        rows.append(_measure(mesh, xs, ys, z_slice))
    payload = {"schema_version": 1, "unit": "gu", "meshes": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[l-wings] wrote {out_path} count={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
