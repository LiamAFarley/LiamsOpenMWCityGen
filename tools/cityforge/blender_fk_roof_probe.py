"""Dump evaluated Falkreath shell/dormer object bounds for roof-contact fitting.

Inputs are native mesh paths after ``--``. Output is JSON containing each
imported object's evaluated bounds and high-Z vertex samples. This is a
measurement tool only; generation consumes its reviewed results, never Blender.
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


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main() -> int:
    args = args_after_dash()
    if len(args) < 2:
        print("usage: blender -b --python blender_fk_roof_probe.py -- OUT.json MESH...", file=sys.stderr)
        return 2
    out = Path(args[0])
    meshes = args[1:]
    roots, resolver = render.load_procgen_meshcheck()
    import_settings = {
        "scale_correction": 0.01,
        "normalize_to_position": False,
        "use_existing_materials": False,
        "ignore_collision_nodes": True,
        "ignore_animations": True,
        "reuse_meshes": False,
        "vertex_precision": 0.001,
    }
    result = {"unit": "gu", "meshes": []}
    for mesh in meshes:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        config = nif_thumbs.resolved_config({}, layout="strip", resolution="1536x512")
        nif_thumbs._configure_engine(config)
        document = {
            "scene_name": "ProcGen_FK_RoofProbe",
            "import": import_settings,
            "meshes": [{"id": Path(mesh).stem, "mesh": mesh.replace("/", "\\"), "position": [0, 0, 0]}],
        }
        nif_import = render.setup_plugin(roots, import_settings)
        entries = render.resolve_meshes(document, roots, resolver)
        objects, _groups = render.import_meshes(entries, nif_import, import_settings)
        bpy.context.view_layer.update()
        rows = []
        for obj in objects:
            if obj.type != "MESH":
                continue
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            corners = [evaluated.matrix_world @ Vector(corner) * 100.0 for corner in evaluated.bound_box]
            vertices = [evaluated.matrix_world @ vertex.co * 100.0 for vertex in evaluated.data.vertices]
            if not vertices:
                continue
            max_z = max(v.z for v in vertices)
            top = [v for v in vertices if v.z >= max_z - 20.0]
            rows.append({
                "name": obj.name,
                "bounds_gu": {
                    "min": [round(min(v[i] for v in corners), 3) for i in range(3)],
                    "max": [round(max(v[i] for v in corners), 3) for i in range(3)],
                },
                "top_vertices_gu": [[round(v.x, 3), round(v.y, 3), round(v.z, 3)] for v in top[:300]],
                "vertices_gu": [[round(v.x, 3), round(v.y, 3), round(v.z, 3)] for v in vertices[:5000]],
            })
        result["meshes"].append({"model": mesh, "objects": rows})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
