"""Blender diagnostic worker for native-resolution roof overlays.

The worker imports the requested NIF and draws measured patch boundaries and
canonical frame arrows as separate scene geometry.  It never changes source
meshes or makes eligibility decisions.
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

GU_TO_SCENE = 0.01
VIEWS = {
    "north": Vector((0.0, 1.0, 0.22)),
    "east": Vector((1.0, 0.0, 0.22)),
    "south": Vector((0.0, -1.0, 0.22)),
    "west": Vector((-1.0, 0.0, 0.22)),
    "top_down": Vector((0.0, 0.0, 1.0)),
    "isometric": Vector((1.0, -1.0, 0.70)),
}


def _args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def _scene_point(point_gu: Vector | tuple[float, float, float] | list[float]) -> Vector:
    return Vector(point_gu) * GU_TO_SCENE


def _plane_point(patch: dict, u_coord: float, v_coord: float, normal_offset_gu: float) -> Vector:
    return Vector(patch["origin_gu"]) + (
        Vector(patch["u"]) * u_coord
        + Vector(patch["v"]) * v_coord
        + Vector(patch["n"]) * normal_offset_gu
    )


def _material(name: str, color: list[float], alpha: float = 1.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*[float(c) for c in color], float(alpha))
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    material.diffuse_color = (*[float(c) for c in color], float(alpha))
    try:
        material.surface_render_method = "DITHERED"
    except AttributeError:
        pass
    return material


def _curve(name: str, points_gu: list[Vector], material, bevel_gu: float = 2.0) -> None:
    if len(points_gu) < 2:
        return
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = bevel_gu * GU_TO_SCENE
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(len(points_gu) - 1)
    for point, value in zip(spline.points, points_gu):
        point.co = (*_scene_point(value), 1.0)
    object_ = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(object_)
    object_.data.materials.append(material)


def _triangles(name: str, triangles_uv: list[list[list[float]]], patch: dict, material, normal_offset_gu: float) -> None:
    vertices = []
    faces = []
    for triangle in triangles_uv:
        base = len(vertices)
        vertices.extend(
            tuple(_scene_point(_plane_point(patch, float(point[0]), float(point[1]), normal_offset_gu)))
            for point in triangle
        )
        faces.append((base, base + 1, base + 2))
    if not faces:
        return
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    object_ = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(object_)
    object_.data.materials.append(material)


def _add_patch_overlay(patch: dict, color: list[float], index: int) -> None:
    fill = _material(f"roof_fill_{patch['patch_id']}", color, 0.28)
    inset = _material(f"roof_inset_{patch['patch_id']}", [1.0, 1.0, 1.0], 1.0)
    _triangles(f"roof_fill_{patch['patch_id']}", patch.get("fill_triangles_uv", []), patch, fill, 0.8)
    for piece_index, piece in enumerate(patch.get("usable_region_uv", [])):
        outer = [_plane_point(patch, float(point[0]), float(point[1]), 1.2) for point in piece.get("outer", [])]
        _curve(f"roof_usable_{patch['patch_id']}_{piece_index}", outer + outer[:1], inset, 2.0)
        for hole_index, hole in enumerate(piece.get("holes", [])):
            points = [_plane_point(patch, float(point[0]), float(point[1]), 1.2) for point in hole]
            _curve(f"roof_usable_{patch['patch_id']}_{piece_index}_hole_{hole_index}", points + points[:1], inset, 2.0)
    classification_colors = {
        "eave": color,
        "ridge": [1.0, 0.85, 0.1],
        "gable": [0.1, 0.85, 1.0],
        "valley": [1.0, 0.35, 0.9],
        "unresolved": [1.0, 1.0, 1.0],
    }
    for segment_index, segment in enumerate(patch.get("boundary_segments", [])):
        segment_material = _material(
            f"roof_segment_{patch['patch_id']}_{segment_index}",
            classification_colors.get(segment.get("classification"), [1.0, 1.0, 1.0]),
            1.0,
        )
        a = _plane_point(patch, *segment["start_uv"], 1.5)
        b = _plane_point(patch, *segment["end_uv"], 1.5)
        _curve(f"roof_boundary_{patch['patch_id']}_{segment_index}", [a, b], segment_material, 2.5)
    anchor = Vector(patch["frame_anchor_uv"])
    anchor_surface = _plane_point(patch, anchor.x, anchor.y, 2.0)
    u_length = float(patch.get("frame_u_length_gu", patch.get("frame_arrow_length_gu", 96.0)))
    v_length = float(patch.get("frame_v_length_gu", patch.get("frame_arrow_length_gu", 96.0)))
    normal_length = float(patch.get("frame_normal_length_gu", 32.0))
    _curve(
        f"roof_u_{patch['patch_id']}",
        [anchor_surface, _plane_point(patch, anchor.x + u_length, anchor.y, 2.0)],
        _material(f"roof_u_mat_{index}", [0.0, 0.9, 1.0]),
        2.5,
    )
    _curve(
        f"roof_v_{patch['patch_id']}",
        [anchor_surface, _plane_point(patch, anchor.x, anchor.y + v_length, 2.0)],
        _material(f"roof_v_mat_{index}", [1.0, 0.8, 0.0]),
        2.5,
    )
    _curve(
        f"roof_n_{patch['patch_id']}",
        [anchor_surface, anchor_surface + Vector(patch["n"]) * normal_length],
        _material(f"roof_n_mat_{index}", [1.0, 1.0, 1.0]),
        2.5,
    )


def _add_marker(point_gu: list[float], material) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=8.0 * GU_TO_SCENE, location=_scene_point(point_gu))
    marker = bpy.context.object
    marker.name = "dormer_contact_marker"
    marker.data.materials.append(material)


def _import_scene(job: dict):
    roots, resolver = render.load_procgen_meshcheck()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    resolution = int(job["resolution"])
    settings = {
        "scale_correction": 0.01,
        "normalize_to_position": False,
        "use_existing_materials": True,
        "ignore_collision_nodes": True,
        "ignore_animations": True,
        "reuse_meshes": True,
        "vertex_precision": 0.001,
    }
    placements = []
    for index, placement in enumerate(job["placements"]):
        placements.append({
            "id": f"placement_{index}",
            "mesh": str(placement["mesh"]).replace("/", "\\"),
            "position": [float(value) * GU_TO_SCENE for value in placement.get("position_gu", [0.0, 0.0, 0.0])],
            "rotation": [float(value) for value in placement.get("rotation_blender", [0.0, 0.0, 0.0])],
            "scale": float(placement.get("scale", 1.0)),
        })
    document = {"scene_name": "ProcGen_Roof_Diagnostic", "import": settings, "meshes": placements}
    nif_thumbs._configure_engine(nif_thumbs.resolved_config({}, layout="strip", resolution=f"{resolution}x{resolution}"))
    entries = render.resolve_meshes(document, roots, resolver)
    objects, _groups = render.import_meshes(entries, render.setup_plugin(roots, settings), settings)
    bpy.context.view_layer.update()
    return objects


def _bounds(objects) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    for object_ in objects:
        if object_.type != "MESH":
            continue
        evaluated = object_.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            world = evaluated.matrix_world @ Vector(corner)
            minimum.x = min(minimum.x, world.x)
            minimum.y = min(minimum.y, world.y)
            minimum.z = min(minimum.z, world.z)
            maximum.x = max(maximum.x, world.x)
            maximum.y = max(maximum.y, world.y)
            maximum.z = max(maximum.z, world.z)
    if not math.isfinite(minimum.x):
        raise RuntimeError("imported diagnostic scene has no mesh bounds")
    return minimum, maximum


def _setup_lighting(center: Vector, span: float) -> None:
    world = bpy.data.worlds.new("RoofDiagnosticWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.11, 0.12, 0.15, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.75
    bpy.context.scene.world = world
    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y, center.z + span * 1.8))
    top = bpy.context.object
    top.data.energy = 1200.0
    top.data.shape = "DISK"
    top.data.size = span * 1.1
    top.rotation_euler = (0.0, 0.0, 0.0)
    for dx, dy in ((span * 1.35, 0.0), (-span * 1.35, 0.0), (0.0, span * 1.35), (0.0, -span * 1.35)):
        bpy.ops.object.light_add(type="AREA", location=(center.x + dx, center.y + dy, center.z + span * 0.35))
        fill = bpy.context.object
        fill.data.energy = 900.0
        fill.data.shape = "DISK"
        fill.data.size = span * 0.9
        fill.rotation_euler = (center - fill.location).to_track_quat("-Z", "Y").to_euler()


def _render(job: dict, objects) -> None:
    minimum, maximum = _bounds(objects)
    center = (minimum + maximum) / 2.0
    span = max((maximum - minimum).length, 0.001)
    _setup_lighting(center, span)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = int(job["resolution"])
    scene.render.resolution_y = int(job["resolution"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    camera_data = bpy.data.cameras.new("RoofDiagnosticCamera")
    camera_data.type = "ORTHO"
    camera_data.clip_end = span * 8.0
    camera = bpy.data.objects.new("RoofDiagnosticCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    output_prefix = Path(job["out_prefix"])
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, direction in VIEWS.items():
        direction = direction.normalized()
        camera_data.ortho_scale = span * (1.35 if name != "top_down" else 1.25)
        camera.location = center + direction * span * 2.5
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(output_prefix.parent / f"{output_prefix.name}_{name}.png")
        bpy.ops.render.render(write_still=True)


def main() -> int:
    arguments = _args()
    if len(arguments) != 1:
        print("usage: blender -b --python blender_roof_overlay.py -- JOB.json", file=sys.stderr)
        return 2
    job = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    objects = _import_scene(job)
    for index, patch in enumerate(job.get("patches", [])):
        _add_patch_overlay(patch, patch.get("color", [1.0, 0.2, 0.1]), index)
    marker = job.get("marker_gu")
    if marker is not None:
        _add_marker(marker, _material("dormer_marker", [1.0, 0.1, 0.8]))
    bpy.context.view_layer.update()
    _render(job, objects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
