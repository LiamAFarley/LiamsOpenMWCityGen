"""Blender-side raw triangle evidence export for Phase 3 wall/mount profiles.

Pipeline position: evidence step of Phase 3a (spec:
``.opencode/runs/2026-08-21-building-generation-rule-kit/2026-08-22_phase3a_implementation_spec.md``).
Spawned once by ``tools/cityforge/build_wall_mount_profiles.py`` with a job
JSON. All facade reconstruction, mount framing, and door pairing logic is pure
Python host-side; this script only exports evaluated geometry facts.

Per model: evaluated triangles (3 rounded GU vertices each), per-triangle
area-weighted normal and area, evaluated bounds, and resolved source path.

Job JSON::

    {"out": "<output json>", "meshes": ["sky/x/sky_fk_house_02_a.nif", ...],
     "decimals": 4}
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


def _argv_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


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
        "scene_name": "ProcGen_Wall_Mount_Evidence",
        "import": import_settings,
        "meshes": [{"id": Path(mesh).stem, "mesh": mesh.replace("/", "\\"), "position": [0.0, 0.0, 0.0]}],
    }
    nif_import = render.setup_plugin(roots, import_settings)
    entries = render.resolve_meshes(document, roots, resolver)
    objects, _groups = render.import_meshes(entries, nif_import, import_settings)
    bpy.context.view_layer.update()
    return objects


def _export_model(mesh: str, objects, decimals: int) -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    triangles: list[dict] = []
    bmin = [float("inf")] * 3
    bmax = [float("-inf")] * 3
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        data = evaluated.to_mesh()
        matrix = evaluated.matrix_world
        try:
            data.calc_loop_triangles()
            verts = []
            for vertex in data.vertices:
                world = matrix @ Vector(vertex.co)
                p = (float(world.x) * GU, float(world.y) * GU, float(world.z) * GU)
                verts.append(p)
                for axis in range(3):
                    bmin[axis] = min(bmin[axis], p[axis])
                    bmax[axis] = max(bmax[axis], p[axis])
            for tri in data.loop_triangles:
                a, b, c = (verts[int(i)] for i in tri.vertices)
                ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
                ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
                cross = (
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                )
                area2 = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
                if area2 == 0.0:
                    continue
                triangles.append({
                    "verts": [[round(c_, decimals) for c_ in p] for p in (a, b, c)],
                    "normal": [round(cross[i] / area2, 6) for i in range(3)],
                    "area": round(area2 / 2.0, 6),
                    "centroid": [round((a[i] + b[i] + c[i]) / 3.0, decimals) for i in range(3)],
                })
        finally:
            evaluated.to_mesh_clear()
    if not triangles:
        raise RuntimeError("no mesh geometry")
    return {
        "model_key": mesh.replace("/", "\\"),
        "bounds_local_gu": {
            "min": [round(v, 3) for v in bmin],
            "max": [round(v, 3) for v in bmax],
        },
        "triangle_count": len(triangles),
        "triangles": triangles,
    }


def main() -> int:
    args = _argv_after_dash()
    if len(args) != 1:
        print("usage: blender -b --python blender_wall_mount_evidence.py -- JOB.json", file=sys.stderr)
        return 2
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    out_path = Path(job["out"])
    decimals = int(job.get("decimals", 4))
    roots, resolver = render.load_procgen_meshcheck()
    rows: list[dict] = []
    failures: list[str] = []
    for mesh in list(job["meshes"]):
        print(f"[wm-evidence] {mesh}", flush=True)
        try:
            resolved = resolver(mesh, "mesh", roots=roots)
            if resolved is None:
                raise RuntimeError("unresolved under configured data roots")
            row = _export_model(mesh, _import_mesh(mesh, roots, resolver), decimals)
            row["resolved_path"] = str(resolved)
            rows.append(row)
            print(f"[wm-evidence] {Path(mesh).name} tris={row['triangle_count']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - explicit failure rows
            failures.append(f"{mesh}: {exc}")
            print(f"[wm-evidence] FAIL {mesh}: {exc}", flush=True)
    payload = {"schema_version": 1, "unit": "gu", "models": rows, "failures": failures}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[wm-evidence] wrote {out_path} ok={len(rows)} failed={len(failures)}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
