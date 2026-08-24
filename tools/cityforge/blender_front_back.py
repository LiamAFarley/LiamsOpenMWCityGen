"""Blender front/back orthographic pair for one attachment model (Phase 3a).

Spawned by ``tools/cityforge/render_profile_diagnostics.py``. Renders the
model from +n and -n of its measured mount axis (local frame), matte
no-specular EEVEE, so the mount profile's front/back occupancy evidence has a
visual review artifact.

Job JSON::

    {"mesh": "sky/x/sky_fk_window_06a.nif", "normal_axis": 0,
      "out_plus": "...png", "out_minus": "...png", "resolution": 1024}
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


def _look_at(camera, target: Vector, location: Vector) -> None:
    direction = target - location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _render_view(mesh: str, sign: float, axis: int, out_path: Path, resolution: int, roots, resolver) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    config = nif_thumbs.resolved_config({}, layout="strip", resolution=f"{resolution}x{resolution}")
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
        "scene_name": "ProcGen_Front_Back",
        "import": import_settings,
        "meshes": [{"id": Path(mesh).stem, "mesh": mesh.replace("/", "\\"), "position": [0.0, 0.0, 0.0]}],
    }
    nif_import = render.setup_plugin(roots, import_settings)
    entries = render.resolve_meshes(document, roots, resolver)
    objects, _groups = render.import_meshes(entries, nif_import, import_settings)
    bpy.context.view_layer.update()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    bmin = Vector((float("inf"),) * 3)
    bmax = Vector((float("-inf"),) * 3)
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            world = evaluated.matrix_world @ Vector(corner)
            for i in range(3):
                bmin[i] = min(bmin[i], world[i])
                bmax[i] = max(bmax[i], world[i])
    center = (bmin + bmax) / 2.0
    size = max((bmax - bmin).length, 0.001)
    span_axes = [bmax[i] - bmin[i] for i in range(3)]
    fit_span = max(span_axes[(axis + 1) % 3], span_axes[2], 0.001)

    direction = Vector((0.0, 0.0, 0.0))
    direction[axis] = sign
    distance = size * 2.0 + 1.0
    location = center + direction * distance

    camera_data = bpy.data.cameras.new("FrontBackCam")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = fit_span * 1.25
    camera_data.clip_start = 0.01
    camera_data.clip_end = distance + size * 4.0
    camera = bpy.data.objects.new("FrontBackCam", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = location
    _look_at(camera, center, location)
    bpy.context.scene.camera = camera

    world = bpy.data.worlds.new("FrontBackWorld")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs[0].default_value = (0.15, 0.15, 0.17, 1.0)
    background.inputs[1].default_value = 0.7
    bpy.context.scene.world = world

    span = max(size, 0.001)
    # mesh_thumbs' studio energies are tuned for building-scale meshes. Scale
    # them by imported scene span so small windows do not clip to white.
    light_scale = max(min((span / 10.0) ** 2, 1.0), 0.01)
    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y, center.z + span * 1.8))
    top = bpy.context.object
    top.data.energy = 1200.0 * light_scale
    top.data.shape = "DISK"
    top.data.size = span * 1.1
    top.rotation_euler = (0.0, 0.0, 0.0)
    for dx, dy in ((span * 1.35, 0.0), (-span * 1.35, 0.0), (0.0, span * 1.35), (0.0, -span * 1.35)):
        bpy.ops.object.light_add(type="AREA", location=(center.x + dx, center.y + dy, center.z + span * 0.35))
        fill = bpy.context.object
        fill.data.energy = 900.0 * light_scale
        fill.data.shape = "DISK"
        fill.data.size = span * 0.9
        fill.rotation_euler = (center - fill.location).to_track_quat("-Z", "Y").to_euler()
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Standard"

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.filepath = str(out_path)
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


def main() -> int:
    args = _argv_after_dash()
    if len(args) != 1:
        print("usage: blender -b --python blender_front_back.py -- JOB.json", file=sys.stderr)
        return 2
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    roots, resolver = render.load_procgen_meshcheck()
    axis = int(job["normal_axis"])
    resolution = int(job.get("resolution", 1024))
    _render_view(job["mesh"], +1.0, axis, Path(job["out_plus"]), resolution, roots, resolver)
    _render_view(job["mesh"], -1.0, axis, Path(job["out_minus"]), resolution, roots, resolver)
    print(f"[front-back] wrote {job['out_plus']} and {job['out_minus']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
