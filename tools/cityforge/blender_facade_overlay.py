"""Blender facade overlay render for one shell model (Phase 3a diagnostics).

Spawned by ``tools/cityforge/render_profile_diagnostics.py``. Imports the
shell, draws each facade's measured convex polygon in its `(u, z)` frame at
the signed facade plane offset, nudged outward, as a distinct emissive n-gon
with the facade ID color, and renders four orthographic views
(north/east/south/west) so every facade plane is visible against the real mesh.

Job JSON::

    {"mesh": "sky/x/sky_fk_house_02_a.nif",
     "facades": [{"facade_id": "f001", "color": [0.9, 0.3, 0.3],
                    "n": [1, 0, 0], "u": [0, 1, 0], "offset_gu": 239.0,
                    "polygon_uz": [[-364.0, 0.3], [364.0, 0.3],
                                    [364.0, 239.0], [-364.0, 239.0]]}],
     "out_prefix": "<dir>/sky_fk_house_02_a_overlay", "resolution": 1024}
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
VIEWS = {
    "north": Vector((0.0, 1.0, 0.2)),
    "east": Vector((1.0, 0.0, 0.2)),
    "south": Vector((0.0, -1.0, 0.2)),
    "west": Vector((-1.0, 0.0, 0.2)),
}


def _setup_overcast_rig(center: Vector, span: float) -> None:
    """Mirror of the mesh_thumbs overcast rig: world + top softbox + 4 fills."""
    scene = bpy.context.scene
    world = bpy.data.worlds.new("OverlayWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.12, 0.12, 0.14, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.7
    scene.world = world
    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y, center.z + span * 1.8))
    top = bpy.context.object
    top.data.energy = 1200.0
    top.data.shape = "DISK"
    top.data.size = span * 1.1
    top.rotation_euler = (0.0, 0.0, 0.0)
    side_distance = span * 1.35
    side_height = center.z + span * 0.35
    for dx, dy in ((side_distance, 0.0), (-side_distance, 0.0), (0.0, side_distance), (0.0, -side_distance)):
        bpy.ops.object.light_add(type="AREA", location=(center.x + dx, center.y + dy, side_height))
        fill = bpy.context.object
        fill.data.energy = 900.0
        fill.data.shape = "DISK"
        fill.data.size = span * 0.9
        fill.rotation_euler = (Vector((0, 0, side_height)) - fill.location).to_track_quat("-Z", "Y").to_euler()
    scene.view_settings.view_transform = "Standard"


def _argv_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def _look_at(camera, target: Vector, location: Vector) -> None:
    camera.rotation_euler = (target - location).to_track_quat("-Z", "Y").to_euler()


def _plane_point(n: Vector, u: Vector, offset: float, u_coord: float, z: float) -> Vector:
    horizontal = Vector((n.x, n.y, 0.0))
    horizontal_length_sq = horizontal.length_squared
    if horizontal_length_sq <= 1e-12:
        raise ValueError("facade normal has no horizontal component")
    along_horizontal = (offset - n.z * z) / horizontal_length_sq
    return horizontal * along_horizontal + u * u_coord + Vector((0.0, 0.0, z))


def _add_facade_polygon(facade: dict, epsilon: float = 2.0) -> None:
    n = Vector(facade["n"])
    u = Vector(facade["u"])
    offset = float(facade["offset_gu"]) + epsilon
    polygon = facade.get("polygon_uz", [])
    if len(polygon) < 3:
        return
    corners_gu = [_plane_point(n, u, offset, float(point[0]), float(point[1])) for point in polygon]
    verts = [tuple(c / GU) for c in corners_gu]
    mesh_data = bpy.data.meshes.new(f"quad_{facade['facade_id']}")
    mesh_data.from_pydata(verts, [], [tuple(range(len(verts)))])
    obj = bpy.data.objects.new(f"quad_{facade['facade_id']}", mesh_data)
    bpy.context.scene.collection.objects.link(obj)
    material = bpy.data.materials.new(f"mat_{facade['facade_id']}")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*[float(c) for c in facade["color"]], 1.0)
    emission.inputs["Strength"].default_value = 1.0
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    material.surface_render_method = "DITHERED"
    obj.data.materials.append(material)
    obj.show_transparent = True


def main() -> int:
    args = _argv_after_dash()
    if len(args) != 1:
        print("usage: blender -b --python blender_facade_overlay.py -- JOB.json", file=sys.stderr)
        return 2
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    roots, resolver = render.load_procgen_meshcheck()
    resolution = int(job.get("resolution", 1024))
    out_prefix = Path(job["out_prefix"])

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
        "scene_name": "ProcGen_Facade_Overlay",
        "import": import_settings,
        "meshes": [{"id": Path(job["mesh"]).stem, "mesh": job["mesh"].replace("/", "\\"), "position": [0.0, 0.0, 0.0]}],
    }
    nif_import = render.setup_plugin(roots, import_settings)
    entries = render.resolve_meshes(document, roots, resolver)
    objects, _groups = render.import_meshes(entries, nif_import, import_settings)
    bpy.context.view_layer.update()

    for facade in job["facades"]:
        _add_facade_polygon(facade)

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
    spans = [float(bmax[i] - bmin[i]) for i in range(3)]

    _setup_overcast_rig(center, size)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    camera_data = bpy.data.cameras.new("OverlayCam")
    camera_data.type = "ORTHO"
    camera_data.clip_end = size * 6.0
    camera = bpy.data.objects.new("OverlayCam", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    for name, direction in VIEWS.items():
        direction.normalize()
        # Fit the view plane spans (not the full diagonal) so walls fill frame.
        if name in ("north", "south"):
            view_span = max(spans[0], spans[2])
        else:
            view_span = max(spans[1], spans[2])
        camera_data.ortho_scale = view_span * 1.2
        camera.location = center + direction * size * 2.5
        _look_at(camera, center, camera.location)
        scene.render.filepath = str(out_prefix.parent / f"{out_prefix.name}_{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[overlay] wrote {scene.render.filepath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
