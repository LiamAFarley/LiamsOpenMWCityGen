#!/usr/bin/env python3
"""Print imported Falkreath access-piece object bounds for placement tuning.

Invoked by Blender with ``-- OUT.json mesh [mesh...]``. It uses the same
io_scene_mw import path as the kit-bound measurement tool and reports each
imported object's evaluated world AABB, making deck/landing Z references
visible instead of guessing from the total mesh bounds.
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


def _argv_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main() -> int:
    args = _argv_after_dash()
    if len(args) < 2:
        print("usage: blender -b --python blender_fk_access_profile.py -- OUT.json mesh [mesh...]", file=sys.stderr)
        return 2
    out_path = Path(args[0])
    roots, resolver = render.load_procgen_meshcheck()
    import_settings = {
        "scale_correction": 0.01,
        "normalize_to_position": False,
        "use_existing_materials": True,
        "ignore_collision_nodes": True,
        "ignore_animations": True,
        "reuse_meshes": False,
        "vertex_precision": 0.001,
    }
    rows = []
    for mesh in args[1:]:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        config = nif_thumbs.resolved_config({}, layout="strip", resolution="1536x512")
        nif_thumbs._configure_engine(config)
        document = {
            "scene_name": "ProcGen_Access_Profile",
            "import": import_settings,
            "meshes": [{"id": Path(mesh).stem, "mesh": mesh.replace("/", "\\"), "position": [0.0, 0.0, 0.0]}],
        }
        nif_import = render.setup_plugin(roots, import_settings)
        entries = render.resolve_meshes(document, roots, resolver)
        objects, _groups = render.import_meshes(entries, nif_import, import_settings)
        bpy.context.view_layer.update()
        for obj in objects:
            if obj.type != "MESH":
                continue
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            corners = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
            rows.append({
                "mesh": mesh.replace("/", "\\"),
                "object": obj.name,
                "min": [round(min(float(v[i]) for v in corners) * 100.0, 3) for i in range(3)],
                "max": [round(max(float(v[i]) for v in corners) * 100.0, 3) for i in range(3)],
            })
    out_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[access-profile] wrote {out_path} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
