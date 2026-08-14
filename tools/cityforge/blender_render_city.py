"""Blender worker for the Cityforge T1.5 terrain-backed review renderer.

This file is intentionally a worker, not the public entry point.  Invoke it
through ``tools/cityforge/render_city.py`` or directly inside Blender with a
fully validated ``render_scene.json``.  It uses the existing
``blender_flat_render`` resolver and ``io_scene_mw`` importer, but does not use
the old flat-ground or giant-water-plane paths.

Worker contract
===============
* every scene ref is imported from the host-generated final T1.2 placement
  list; no ref is synthesized, centered, grounded, or road-rotated;
* image-backed NIF materials are flattened to ordinary Principled nodes only
  when the importer supplied a real source image; a missing/placeholder image
  is fatal;
* one 449x449 mesh is built from the T1.3 final float64 field and the 49 local
  LAND VTEX grids.  Its material slots are the T1.3 local LTEX records;
* water is a triangle-by-triangle z=0 clip of that terrain mesh.  No plane,
  skirt, blue rectangle, or fallback terrain is created;
* all eleven base images plus one focused door-height detail image per
  synthetic lot are rendered in one scene/build.  Street and detail cameras
  reuse the host's deterministic door/road candidate contract, then repeat
  the terrain LOS and finite-edge checks against the imported scene;
* the machine-readable worker audit is written only after every view succeeds.

The worker writes no source or production mod data.  Its only files are the
PNG views and ``blender_worker_audit.json`` under the caller's fresh output
directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
TOOLS = WORKSPACE / "tools"
SRC = WORKSPACE / "src"
for _path in (TOOLS, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import bpy  # type: ignore  # noqa: E402
from mathutils import Vector  # type: ignore  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # type: ignore  # noqa: E402

import blender_flat_render as flat  # type: ignore  # noqa: E402
from procgen import cityrender, espland, tes3json  # noqa: E402


STAGE = cityrender.STAGE
SCENE_UNITS_PER_GAME_UNIT = cityrender.SCENE_UNITS_PER_GAME_UNIT
MATRIX_TOLERANCE = cityrender.MATRIX_TOLERANCE
TRANSLATION_FLOAT_STORAGE_TOLERANCE = 2.0e-5
FIELD_SIDE = cityrender.FIELD_SIDE
CELL_SIZE_GU = 8192.0
FIELD_SPACING_GU = 128.0
TILE_SIDE = 16
TILE_FACES_PER_SIDE = 4


class BlenderRenderFailure(RuntimeError):
    """A required worker-side import, material, geometry, or view gate failed."""


def fail(message: str) -> None:
    raise BlenderRenderFailure(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read worker scene JSON {path}: {exc}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )


def _matrix_error(actual: Any, expected: Any) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        fail(f"matrix shape mismatch {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b)))


def _all_mesh_objects(objects: Iterable[Any]) -> list[Any]:
    return [obj for obj in objects if getattr(obj, "type", None) == "MESH"]


def _object_tree_root(obj: Any) -> Any:
    root = obj
    seen: set[int] = set()
    while getattr(root, "parent", None) is not None:
        if id(root) in seen:
            fail(f"cyclic imported object hierarchy at {obj.name}")
        seen.add(id(root))
        root = root.parent
    return root


def _group_roots(objects: Sequence[Any]) -> list[Any]:
    roots = {_object_tree_root(obj) for obj in objects}
    return sorted(roots, key=lambda obj: str(obj.name))


def _world_bounds(objects: Iterable[Any]) -> tuple[Vector, Vector, list[Vector]]:
    points: list[Vector] = []
    for obj in objects:
        if getattr(obj, "type", None) != "MESH" or bool(getattr(obj, "hide_render", False)):
            continue
        try:
            points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
        except Exception:
            continue
    if not points:
        fail("render selection has no visible mesh bounds")
    minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return minimum, maximum, points


def _box_corners(minimum: Vector, maximum: Vector) -> list[Vector]:
    return [Vector((x, y, z)) for x in (minimum.x, maximum.x) for y in (minimum.y, maximum.y) for z in (minimum.z, maximum.z)]


def _configure_render_scene(scene: Any, settings: Mapping[str, Any]) -> None:
    """Configure one neutral, matte Eevee scene with no fallback engine."""

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception as exc:
        fail(f"Blender Eevee Next is unavailable; no degraded render engine is permitted: {exc}")
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.look = str(settings.get("view_look", "AgX - Medium High Contrast"))
    scene.view_settings.exposure = float(settings.get("exposure", 0.7))
    world = scene.world or bpy.data.worlds.new("Cityforge_T15_World")
    scene.world = world
    world.use_nodes = True
    color = tuple(float(value) for value in settings.get("world_color", [0.055, 0.07, 0.095, 1.0]))
    background = world.node_tree.nodes.get("Background")
    if background is None:
        fail("Blender world has no Background node")
    background.inputs["Color"].default_value = color
    background.inputs["Strength"].default_value = float(settings.get("world_strength", 0.8))
    world.color = color[:3]


def _add_area_light(name: str, location: Sequence[float], energy: float, size: float, target: Vector) -> Any:
    bpy.ops.object.light_add(type="AREA", location=tuple(float(value) for value in location))
    light = bpy.context.object
    light.name = name
    light.data.energy = float(energy)
    light.data.shape = "DISK"
    light.data.size = float(size)
    light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()
    return light


def _setup_lighting(scene: Any, minimum: Vector, maximum: Vector) -> dict[str, Any]:
    """Add broad, neutral lights; no high-specular directional glare."""

    target = (minimum + maximum) / 2.0
    span = max(float(maximum.x - minimum.x), float(maximum.y - minimum.y), float(maximum.z - minimum.z), 10.0)
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT" and str(obj.name).startswith("Cityforge_T15_Light"):
            bpy.data.objects.remove(obj, do_unlink=True)
    _add_area_light("Cityforge_T15_Light_Top", (target.x, target.y, maximum.z + span * 1.6), 3200.0, span * 1.8, target)
    for index, (dx, dy) in enumerate(((span, 0.0), (-span, 0.0), (0.0, span), (0.0, -span)), 1):
        _add_area_light(
            f"Cityforge_T15_Light_Fill_{index}",
            (target.x + dx, target.y + dy, target.z + span * 0.6),
            1300.0,
            span * 1.4,
            target,
        )
    return {
        "mode": "broad_neutral_area_rig",
        "top_energy": 3200.0,
        "fill_energy": 1300.0,
        "fill_count": 4,
        "material_policy": "roughness=0.88+ and specular=0",
    }


def _image_absolute_path(image: Any) -> Path | None:
    packed = getattr(image, "packed_file", None)
    if packed is not None:
        return None
    raw = str(getattr(image, "filepath", ""))
    if not raw:
        return None
    try:
        return Path(bpy.path.abspath(raw)).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _material_image(material: Any) -> tuple[Any, Path]:
    if material is None or not bool(getattr(material, "use_nodes", False)):
        fail("imported mesh has a non-node or empty material")
    tree = getattr(material, "node_tree", None)
    if tree is None:
        fail(f"material {material.name} has no node tree")
    images = list(flat._material_images(tree))
    candidates: list[tuple[Any, Path]] = []
    for image in images:
        path = _image_absolute_path(image)
        if path is not None and path.is_file():
            candidates.append((image, path))
    if not candidates:
        fail(f"material {material.name} has no resolved real image texture")
    # Match the existing helper's preference order, but make the filesystem
    # resolution gate explicit instead of accepting an importer placeholder.
    preferred = [
        pair
        for pair in candidates
        if not any(token in str(getattr(pair[0], "name", "")).casefold() for token in ("lightnessgrading", "normal", "_n.", "_n_", "specular", "_s.", "glow", "mask"))
    ]
    return (preferred or candidates)[0]


def _flatten_image_material(material: Any, image: Any) -> None:
    """Make the importer image feed a standard Eevee Principled shader."""

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    try:
        image.colorspace_settings.name = "sRGB"
    except (AttributeError, TypeError, ValueError):
        pass
    texture.interpolation = "Linear"
    links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    emission_color = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    if emission_color is not None:
        links.new(texture.outputs["Color"], emission_color)
        if shader.inputs.get("Emission Strength") is not None:
            shader.inputs["Emission Strength"].default_value = 0.22
    if shader.inputs.get("Alpha") is not None and texture.outputs.get("Alpha") is not None:
        links.new(texture.outputs["Alpha"], shader.inputs["Alpha"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    shader.inputs["Roughness"].default_value = 0.88
    if shader.inputs.get("Specular IOR Level") is not None:
        shader.inputs["Specular IOR Level"].default_value = 0.0
    elif shader.inputs.get("Specular") is not None:
        shader.inputs["Specular"].default_value = 0.0
    if shader.inputs.get("Metallic") is not None:
        shader.inputs["Metallic"].default_value = 0.0
    material.diffuse_color = (0.55, 0.55, 0.55, 1.0)


def _audit_and_prepare_building_materials(objects: Sequence[Any]) -> dict[str, Any]:
    """Audit every visible NIF material and retain only real image-backed ones."""

    seen: dict[int, dict[str, Any]] = {}
    object_count = 0
    for obj in _all_mesh_objects(objects):
        if bool(getattr(obj, "hide_render", False)):
            continue
        object_count += 1
        if len(obj.material_slots) == 0:
            fail(f"visible imported mesh {obj.name} has no material slots")
        for slot in obj.material_slots:
            material = slot.material
            if material is None:
                fail(f"visible imported mesh {obj.name} has an empty material slot")
            material_id = id(material)
            if material_id in seen:
                continue
            image, path = _material_image(material)
            _flatten_image_material(material, image)
            seen[material_id] = {
                "material": str(material.name),
                "image": str(getattr(image, "name", "")),
                "path": str(path),
            }
    require(object_count > 0, "NIF import produced no visible textured mesh objects")
    return {
        "visible_mesh_object_count": object_count,
        "material_count": len(seen),
        "resolved_image_count": len(seen),
        "unresolved_image_count": 0,
        "placeholder_material_count": 0,
        "materials": [seen[key] for key in sorted(seen, key=lambda key: seen[key]["material"])],
    }


def _import_buildings(scene_document: Mapping[str, Any]) -> tuple[list[Any], list[tuple[str, list[Any]]], dict[str, Any]]:
    """Import every scene ref through the existing resolver/import machinery."""

    roots, resolver = flat.load_procgen_meshcheck(scene_document["source_paths"]["procgen_config"])
    # blender_flat_render's accepted importer API consumes ``meshes`` entries;
    # the T1.5 host contract calls them ``refs`` so the transform evidence is
    # not confused with a legacy thumbnail scene.  This is a lossless adapter:
    # every field used by import_meshes comes directly from one host ref row.
    import_document = dict(scene_document)
    import_document["meshes"] = [
        {
            "id": str(row["ref_key"]),
            "mesh": str(row["model_key"]),
            "position": list(row["position"]),
            "rotation": list(row["rotation"]),
            "scale": float(row["scale"]),
            "source_id": str(row["source_id"]),
            "record_type": str(row["record_type"]),
        }
        for row in scene_document["refs"]
    ]
    resolved_entries = flat.resolve_meshes(import_document, roots, resolver)
    require(len(resolved_entries) == len(scene_document["refs"]), "mesh resolver did not resolve every T1.2 ref")
    nif_import = flat.setup_plugin(roots, scene_document.get("import", {}))
    before = set(bpy.data.objects)
    try:
        render_objects, render_groups = flat.import_meshes(
            resolved_entries,
            nif_import,
            import_document.get("import", {}),
        )
    except Exception as exc:
        fail(f"io_scene_mw NIF import failed: {exc}")
    require(len(render_groups) == len(scene_document["refs"]), f"NIF import groups {len(render_groups)} != expected refs {len(scene_document['refs'])}")
    require(all(objects for _label, objects in render_groups), "at least one placed ref imported no mesh objects")
    imported_objects = [obj for obj in bpy.data.objects if obj not in before]
    # The existing helper has a source-proven duplicate/helper exclusion.  It
    # is allowed to hide only those evaluated objects; it never makes a missing
    # building succeed.
    duplicate_audit = flat.hide_textureless_duplicate_geometry(render_objects)
    visible_objects = [obj for obj in render_objects if not bool(obj.get("procgen_render_excluded", False))]
    require(visible_objects, "duplicate/helper filter removed every visible placed mesh")
    material_audit = _audit_and_prepare_building_materials(visible_objects)
    return visible_objects, render_groups, {
        "resolved_entries": len(resolved_entries),
        "imported_objects": len(imported_objects),
        "visible_objects": len(visible_objects),
        "duplicate_filter": duplicate_audit,
        "material_audit": material_audit,
    }


def _load_exact_field(terrain: Mapping[str, Any]) -> tuple[np.ndarray, dict[tuple[int, int], Any], dict[int, Any]]:
    field_path = Path(str(terrain["final_field_npz"]))
    metadata_path = Path(str(terrain["final_field_metadata"]))
    try:
        with np.load(field_path, allow_pickle=False) as archive:
            field = np.asarray(archive["height_gu"], dtype=np.float64)
    except Exception as exc:
        fail(f"cannot load exact T1.3 final field in Blender worker: {exc}")
    require(field.shape == (FIELD_SIDE, FIELD_SIDE) and np.isfinite(field).all(), "Blender worker received a non-449x449/non-finite T1.3 field")
    metadata = read_json(metadata_path)
    require(metadata.get("terrain_field_sha256") == cityrender.terrain_field_content_hash(field), "Blender worker final field hash disagrees with T1.3 metadata")
    plugin = Path(str(terrain["plugin"]))
    require(plugin.is_file(), f"scratch terrain plugin is missing in Blender worker: {plugin}")
    records = espland.load_land(plugin, max_seconds=180.0)
    ltex = espland.load_ltex(plugin, max_seconds=180.0)
    require(len(records) == 49 and len(ltex) == 7, f"scratch terrain plugin decoded LAND/LTEX counts {len(records)}/{len(ltex)}, expected 49/7")
    cells = sorted((int(cell[0]), int(cell[1])) for cell in terrain["cells"])
    xs = sorted({cell[0] for cell in cells})
    ys = sorted({cell[1] for cell in cells})
    mismatch = 0
    quantized = np.rint(field / 8.0) * 8.0
    for cell in cells:
        record = records.get(cell)
        require(record is not None and record.heights_gu is not None, f"scratch terrain plugin has no complete LAND {cell}")
        x0 = (cell[0] - xs[0]) * 64
        y0 = (cell[1] - ys[0]) * 64
        mismatch += int(np.count_nonzero(np.asarray(record.heights_gu, dtype=np.float64) != quantized[y0 : y0 + 65, x0 : x0 + 65]))
    require(mismatch == 0, f"Blender worker detected {mismatch} final field/LAND height mismatches")
    return field, records, ltex


def _image_material(name: str, image_path: Path, *, roughness: float = 0.94) -> Any:
    require(image_path.is_file(), f"required terrain texture is missing: {image_path}")
    try:
        image = bpy.data.images.load(str(image_path), check_existing=True)
    except Exception as exc:
        fail(f"terrain texture failed to load {image_path}: {exc}")
    try:
        image.colorspace_settings.name = "sRGB"
    except (AttributeError, TypeError, ValueError):
        pass
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Linear"
    links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    emission_color = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    if emission_color is not None:
        links.new(texture.outputs["Color"], emission_color)
        if shader.inputs.get("Emission Strength") is not None:
            shader.inputs["Emission Strength"].default_value = 0.36
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    shader.inputs["Roughness"].default_value = float(roughness)
    if shader.inputs.get("Specular IOR Level") is not None:
        shader.inputs["Specular IOR Level"].default_value = 0.0
    elif shader.inputs.get("Specular") is not None:
        shader.inputs["Specular"].default_value = 0.0
    material.diffuse_color = (0.42, 0.42, 0.42, 1.0)
    return material


def _terrain_vertices(field: np.ndarray) -> list[tuple[float, float, float]]:
    return [
        (
            float(x) * FIELD_SPACING_GU * SCENE_UNITS_PER_GAME_UNIT,
            float(y) * FIELD_SPACING_GU * SCENE_UNITS_PER_GAME_UNIT,
            float(field[y, x]) * SCENE_UNITS_PER_GAME_UNIT,
        )
        for y in range(FIELD_SIDE)
        for x in range(FIELD_SIDE)
    ]


def _terrain_index(x: int, y: int) -> int:
    return y * FIELD_SIDE + x


def _make_terrain(scene_document: Mapping[str, Any]) -> tuple[Any, dict[str, Any], list[tuple[Vector, Vector, Vector]], np.ndarray]:
    """Build the exact final field mesh and collect triangles for water clipping."""

    terrain_spec = scene_document["terrain"]
    field, records, ltex = _load_exact_field(terrain_spec)
    cells = sorted((int(cell[0]), int(cell[1])) for cell in terrain_spec["cells"])
    xs = sorted({cell[0] for cell in cells})
    ys = sorted({cell[1] for cell in cells})
    vertices = _terrain_vertices(field)
    faces: list[tuple[int, int, int, int]] = []
    face_raw: list[int] = []
    face_uvs: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    triangles: list[tuple[Vector, Vector, Vector]] = []
    for cell_y, cell in enumerate(sorted(cells, key=lambda item: (item[1], item[0]))):
        record = records[cell]
        for local_y in range(64):
            for local_x in range(64):
                gx = (cell[0] - xs[0]) * 64 + local_x
                gy = (cell[1] - ys[0]) * 64 + local_y
                a = _terrain_index(gx, gy)
                b = _terrain_index(gx + 1, gy)
                c = _terrain_index(gx + 1, gy + 1)
                d = _terrain_index(gx, gy + 1)
                faces.append((a, b, c, d))
                raw = int(record.texture_index(min(15, local_x // 4), min(15, local_y // 4)))
                face_raw.append(raw)
                # One TES3 LAND texture slot covers a 512-GU tile, i.e. four
                # of this exact 128-GU final-field intervals on each axis.
                # Keep each 128-GU face in its proper quarter of the source
                # image instead of stretching the whole image over every face.
                sub_x = local_x % 4
                sub_y = local_y % 4
                u0, u1 = sub_x / 4.0, (sub_x + 1) / 4.0
                v0, v1 = sub_y / 4.0, (sub_y + 1) / 4.0
                face_uvs.append(((u0, v0), (u1, v0), (u1, v1), (u0, v1)))
                triangles.append((Vector(vertices[a]), Vector(vertices[b]), Vector(vertices[c])))
                triangles.append((Vector(vertices[a]), Vector(vertices[c]), Vector(vertices[d])))
    texture_rows = {int(row["index"]): row for row in terrain_spec["ltex"]}
    positive_raw = sorted(set(face_raw) - {0})
    material_keys = [None] if 0 in face_raw else []
    material_keys.extend(raw - 1 for raw in positive_raw)
    materials: list[Any] = []
    material_slots: dict[int | None, int] = {}
    for key in material_keys:
        if key is None:
            # T1.3's canonical synthetic output has no raw zero, but keeping a
            # strict base sentinel makes future final fields fail rather than
            # silently use a semantic flat color.
            base_path = flat.load_procgen_meshcheck(scene_document["source_paths"]["procgen_config"])[1]("_land_default.dds", "texture")
            if base_path is None:
                fail("T1.3 terrain contains raw VTEX 0 but _land_default.dds cannot resolve")
            material = _image_material("Cityforge_T15_Terrain_Base", Path(base_path))
        else:
            row = texture_rows.get(int(key))
            if row is None:
                fail(f"terrain raw VTEX {int(key) + 1} has no local LTEX row")
            path = Path(str(row["resolved_path"]))
            material = _image_material(f"Cityforge_T15_Terrain_LTex_{int(key):04d}", path)
        material_slots[key] = len(materials)
        materials.append(material)

    mesh = bpy.data.meshes.new("Cityforge_T15_Exact_Terrain_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    terrain_object = bpy.data.objects.new("Cityforge_T15_Exact_Terrain", mesh)
    bpy.context.collection.objects.link(terrain_object)
    for material in materials:
        mesh.materials.append(material)
    for polygon, raw in zip(mesh.polygons, face_raw):
        polygon.material_index = material_slots[None if raw == 0 else raw - 1]
        polygon.use_smooth = True
    uv_layer = mesh.uv_layers.new(name="Cityforge_T15_TerrainTileUV")
    for polygon, polygon_uvs in zip(mesh.polygons, face_uvs):
        for loop_index, uv in zip(
            polygon.loop_indices,
            polygon_uvs,
        ):
            uv_layer.data[loop_index].uv = uv

    # Apply the final-field geometry's deterministic smooth normals.  If the
    # Blender API exposes custom vertex normals, use T1.3's per-cell VNML bytes
    # as the rendered payload; otherwise the same exact field geometry remains
    # the only normal source (never the base ESM field).
    normal_mode = "smooth_final_field_geometry"
    try:
        normals = np.zeros((FIELD_SIDE, FIELD_SIDE, 3), dtype=np.float32)
        normal_filled = np.zeros((FIELD_SIDE, FIELD_SIDE), dtype=bool)
        for cell in cells:
            record = records[cell]
            if record.vertex_normals is None:
                continue
            payload = np.frombuffer(record.vertex_normals, dtype=np.int8).reshape((65, 65, 3)).astype(np.float32) / 127.0
            x0 = (cell[0] - xs[0]) * 64
            y0 = (cell[1] - ys[0]) * 64
            for local_y in range(65):
                for local_x in range(65):
                    gx, gy = x0 + local_x, y0 + local_y
                    if not normal_filled[gy, gx]:
                        vector = payload[local_y, local_x]
                        length = float(np.linalg.norm(vector))
                        if length > 1.0e-6 and np.isfinite(length):
                            normals[gy, gx] = vector / length
                            normal_filled[gy, gx] = True
        if bool(normal_filled.all()) and hasattr(mesh, "normals_split_custom_set_from_vertices"):
            mesh.normals_split_custom_set_from_vertices([Vector(tuple(float(value) for value in normals[y, x])) for y in range(FIELD_SIDE) for x in range(FIELD_SIDE)])
            normal_mode = "T1.3_final_VNML_payload"
    except Exception as exc:
        # A Blender version without the custom-normal setter is not allowed to
        # switch to a base terrain.  Smooth normals from the exact final field
        # are the explicit, recorded geometry convention.
        print(f"[cityforge-worker] custom VNML setter unavailable; using exact field geometry normals: {exc}", flush=True)

    minimum, maximum, _ = _world_bounds([terrain_object])
    terrain_object["cityforge_field_hash"] = str(terrain_spec["field_hash"])
    terrain_object["cityforge_cell_count"] = 49
    terrain_object["cityforge_ltex_count"] = len(ltex)
    terrain_object["cityforge_texture_resolved_count"] = len(texture_rows)
    terrain_object["cityforge_texture_missing_count"] = 0
    terrain_object["cityforge_opaque"] = True
    terrain_object["cityforge_normal_mode"] = normal_mode
    terrain_object["cityforge_flat_fallback"] = False
    return terrain_object, {
        "field_hash": str(terrain_spec["field_hash"]),
        "cell_count": 49,
        "ltex_count": len(ltex),
        "texture_resolved_count": len(texture_rows),
        "texture_missing_count": 0,
        "face_count": len(faces),
        "vertex_count": len(vertices),
        "bounds_scene_units": {"min": list(map(float, minimum)), "max": list(map(float, maximum))},
        "opaque": True,
        "uv_mode": "T1.3 VTEX tile with four-by-four 128-GU subface UVs",
        "normal_mode": normal_mode,
        "field_record_mismatch_count": 0,
    }, triangles, field


def _clip_triangle_to_water(triangle: Sequence[Vector]) -> list[Vector]:
    """Clip one terrain triangle to the closed half-space z<=0."""

    output = [Vector(point) for point in triangle]
    result: list[Vector] = []
    for index, current in enumerate(output):
        previous = output[index - 1]
        current_inside = current.z <= 0.0
        previous_inside = previous.z <= 0.0
        if current_inside != previous_inside:
            denominator = current.z - previous.z
            if abs(denominator) <= 1.0e-15:
                intersection = Vector((current.x, current.y, 0.0))
            else:
                fraction = -previous.z / denominator
                intersection = previous + fraction * (current - previous)
                intersection.z = 0.0
            result.append(intersection)
        if current_inside:
            point = Vector(current)
            point.z = 0.0
            result.append(point)
    return result


def _make_water(triangles: Sequence[Sequence[Vector]], terrain_bounds: tuple[Vector, Vector]) -> tuple[Any, dict[str, Any]]:
    """Create only clipped z=0 water triangles inside the terrain footprint."""

    minimum, maximum = terrain_bounds
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    outside = 0
    for triangle in triangles:
        clipped = _clip_triangle_to_water(triangle)
        if len(clipped) < 3:
            continue
        for point in clipped:
            if point.x < minimum.x - 1.0e-8 or point.x > maximum.x + 1.0e-8 or point.y < minimum.y - 1.0e-8 or point.y > maximum.y + 1.0e-8:
                outside += 1
            vertices.append((float(point.x), float(point.y), 0.0))
        base = len(vertices) - len(clipped)
        for offset in range(1, len(clipped) - 1):
            faces.append((base, base + offset, base + offset + 1))
    require(outside == 0, f"water clipping produced {outside} vertices outside the terrain footprint")
    require(faces, "final T1.3 field has no submerged terrain triangles for water")
    mesh = bpy.data.meshes.new("Cityforge_T15_Clipped_Water_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    water = bpy.data.objects.new("Cityforge_T15_Clipped_Water_z0", mesh)
    bpy.context.collection.objects.link(water)
    material = bpy.data.materials.new("Cityforge_T15_Water_Opaque")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is None:
        fail("water material has no Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.035, 0.16, 0.30, 1.0)
    shader.inputs["Roughness"].default_value = 0.48
    if shader.inputs.get("Specular IOR Level") is not None:
        shader.inputs["Specular IOR Level"].default_value = 0.0
    elif shader.inputs.get("Specular") is not None:
        shader.inputs["Specular"].default_value = 0.0
    material.diffuse_color = (0.035, 0.16, 0.30, 1.0)
    water.data.materials.append(material)
    z_errors = sum(1 for vertex in vertices if abs(float(vertex[2])) > 1.0e-12)
    bounds = _world_bounds([water])[0:2]
    water["cityforge_water_z_scene"] = 0.0
    water["cityforge_water_triangle_count"] = len(faces)
    water["cityforge_rectangular_plane_used"] = False
    return water, {
        "triangle_count": len(faces),
        "vertex_count": len(vertices),
        "z_mismatch_count": z_errors,
        "outside_terrain_footprint_count": outside,
        "rectangular_plane_used": False,
        "external_skirt_used": False,
        "bounds_scene_units": {"min": list(map(float, bounds[0])), "max": list(map(float, bounds[1]))},
        "z_scene_units": 0.0,
        "clip_source": "final T1.3 terrain triangles",
    }


def _camera_from_target(name: str, target: Vector, location: Vector, mode: str) -> Any:
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = name
    camera.data.type = mode
    camera.data.clip_start = 0.01
    camera.data.clip_end = 10000.0
    camera.rotation_euler = (target - location).to_track_quat("-Z", "Y").to_euler()
    return camera


def _fit_ortho(camera: Any, points: Sequence[Vector], width: int, height: int, margin: float = 1.14) -> float:
    rotation = camera.matrix_world.to_quaternion()
    right = rotation @ Vector((1.0, 0.0, 0.0))
    up = rotation @ Vector((0.0, 1.0, 0.0))
    right_values = [point.dot(right) for point in points]
    up_values = [point.dot(up) for point in points]
    horizontal = max(max(right_values) - min(right_values), 0.01)
    vertical = max(max(up_values) - min(up_values), 0.01)
    aspect = max(float(width) / float(height), 0.01)
    scale = max(vertical, horizontal / aspect) * float(margin)
    camera.data.ortho_scale = scale
    return scale


def _set_clip_from_points(camera: Any, points: Sequence[Vector]) -> None:
    inverse = camera.matrix_world.inverted()
    depths = [-(inverse @ point).z for point in points]
    camera.data.clip_start = max(0.005, min(depths) - 1.0)
    camera.data.clip_end = max(camera.data.clip_start + 10.0, max(depths) + 1.0)


def _ndc_bounds(scene: Any, camera: Any, points: Sequence[Vector]) -> dict[str, Any]:
    if not points:
        return {
            "point_count": 0,
            "min": None,
            "max": None,
            "span_width": 0.0,
            "span_height": 0.0,
            "span_area": 0.0,
            "in_frame": False,
        }
    values = [world_to_camera_view(scene, camera, point) for point in points]
    minimum = [min(float(value[index]) for value in values) for index in range(3)]
    maximum = [max(float(value[index]) for value in values) for index in range(3)]
    span_width = maximum[0] - minimum[0]
    span_height = maximum[1] - minimum[1]
    return {
        "point_count": len(points),
        "min": [round(value, 9) for value in minimum],
        "max": [round(value, 9) for value in maximum],
        "span_width": round(span_width, 9),
        "span_height": round(span_height, 9),
        "span_area": round(span_width * span_height, 9),
        "in_frame": minimum[0] >= 0.0 and maximum[0] <= 1.0 and minimum[1] >= 0.0 and maximum[1] <= 1.0,
    }


def _terrain_edge_points(field: np.ndarray) -> list[Vector]:
    """Return sampled vertices on the finite final-field perimeter.

    This is deliberately the actual 449x449 field perimeter, not a proxy box
    or an enlarged horizon.  Candidate cameras use these points to reject a
    finite-terrain edge projected immediately beside the imported subject.
    """

    side = (cityrender.FIELD_SIDE - 1) * cityrender.FIELD_SPACING_GU * SCENE_UNITS_PER_GAME_UNIT
    points: list[Vector] = []
    sample_count = 32
    for index in range(sample_count + 1):
        fraction = index / float(sample_count)
        coordinate = side * fraction
        for x_scene, y_scene in ((coordinate, 0.0), (coordinate, side), (0.0, coordinate), (side, coordinate)):
            height = cityrender.terrain_height_scene(field, x_scene, y_scene)
            require(height is not None, "finite terrain edge leaves the exact final field")
            points.append(Vector((x_scene, y_scene, float(height))))
    return points


def _terrain_edge_intrusion(
    scene: Any,
    camera: Any,
    content_points: Sequence[Vector],
    edge_points: Sequence[Vector],
) -> dict[str, Any]:
    """Measure perimeter samples projected into the subject's near-frame box.

    A horizontal camera/subject clearance alone does not prove that the finite
    terrain boundary is absent from the lower subject framing.  This second
    gate is intentionally conservative: any in-frustum perimeter sample within
    the configured NDC margin of the imported lot bounds is rejected.  It is a
    camera gate only; no terrain or building geometry is altered.
    """

    content = _ndc_bounds(scene, camera, content_points)
    margin = cityrender.TERRAIN_EDGE_SUBJECT_MARGIN_NDC
    minimum = content["min"] or [0.0, 0.0, 0.0]
    maximum = content["max"] or [0.0, 0.0, 0.0]
    intruding: list[dict[str, Any]] = []
    for point in edge_points:
        projection = world_to_camera_view(scene, camera, point)
        x = float(projection.x)
        y = float(projection.y)
        depth = float(projection.z)
        # A finite-field edge below the imported subject is still a defect:
        # it produces the characteristic empty-sky gap under a foundation even
        # when the edge is outside the subject's strict vertical NDC interval.
        if (
            0.0 <= depth <= 1.0
            and minimum[0] - margin <= x <= maximum[0] + margin
            and 0.0 <= y <= maximum[1] + margin
        ):
            intruding.append({"ndc": [round(x, 9), round(y, 9), round(depth, 9)], "point_scene": [round(float(value), 9) for value in point]})
    return {
        "passed": not intruding,
        "sample_count": len(edge_points),
        "intrusion_count": len(intruding),
        "subject_ndc_margin": margin,
        "intrusions": intruding,
    }


def _camera_audit(scene: Any, camera: Any, content_points: Sequence[Vector], fit_points: Sequence[Vector], view: Mapping[str, Any]) -> dict[str, Any]:
    all_points = list(content_points) + list(fit_points)
    _set_clip_from_points(camera, all_points)
    content_ndc = _ndc_bounds(scene, camera, content_points)
    fit_ndc = _ndc_bounds(scene, camera, fit_points)
    if not content_ndc["in_frame"] or not fit_ndc["in_frame"]:
        print(
            f"[cityforge-worker] CAMERA_MISMATCH {view['view_id']} content={content_ndc} fit={fit_ndc} "
            f"ortho={getattr(camera.data, 'ortho_scale', None)} location={[round(float(v), 4) for v in camera.location]} rotation={[round(float(v), 4) for v in camera.rotation_euler]}",
            flush=True,
        )
    require(bool(content_ndc["in_frame"]), f"{view['view_id']}: placed content is outside the camera frustum")
    require(bool(fit_ndc["in_frame"]), f"{view['view_id']}: fitted approach terrain is outside the camera frustum")
    projection = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=int(view["resolution"][0]),
        y=int(view["resolution"][1]),
        scale_x=1.0,
    )
    return {
        "mode": str(camera.data.type),
        "location": [round(float(value), 9) for value in camera.location],
        "rotation_euler": [round(float(value), 9) for value in camera.rotation_euler],
        "matrix_world": [[round(float(value), 9) for value in row] for row in camera.matrix_world],
        "projection_matrix": [[round(float(value), 9) for value in row] for row in projection],
        "clip_start": round(float(camera.data.clip_start), 9),
        "clip_end": round(float(camera.data.clip_end), 9),
        "ortho_scale": round(float(camera.data.ortho_scale), 9) if camera.data.type == "ORTHO" else None,
        "lens": round(float(camera.data.lens), 9) if camera.data.type == "PERSP" else None,
        "content_in_frame": content_ndc,
        "fit_in_frame": fit_ndc,
    }


def _street_targets(
    view: Mapping[str, Any],
    group_objects: Sequence[Any],
    field: np.ndarray,
    render_origin_gu: Sequence[float],
) -> tuple[Vector, dict[str, Vector]]:
    """Build actual imported-subject targets for the terrain LOS gate."""

    door_world = [float(value) for value in view["door_world_gu"]]
    door = Vector(
        (
            (door_world[0] - float(render_origin_gu[0])) * SCENE_UNITS_PER_GAME_UNIT,
            (door_world[1] - float(render_origin_gu[1])) * SCENE_UNITS_PER_GAME_UNIT,
            door_world[2] * SCENE_UNITS_PER_GAME_UNIT,
        )
    )
    minimum, maximum, _ = _world_bounds(group_objects)
    terrain_at_door = cityrender.terrain_height_scene(field, door.x, door.y)
    require(terrain_at_door is not None, f"{view['view_id']}: door anchor is outside the final terrain field")
    # The door anchor is authoritative.  The facade XY is measured from the
    # imported hierarchy, while the lower target is just above the actual
    # final terrain at that anchor; no building or terrain is moved to create
    # the review target.
    ground = Vector(
        (
            door.x,
            door.y,
            max(
                door.z + cityrender.STREET_TARGET_OFFSETS_SCENE[0],
                float(terrain_at_door) + cityrender.STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE + 0.05,
            ),
        )
    )
    door_center = Vector((door.x, door.y, door.z + cityrender.STREET_TARGET_OFFSETS_SCENE[1]))
    door_lower_threshold = Vector((door.x, door.y, door.z + cityrender.STREET_DOOR_LOS_FLOOR_OFFSET_SCENE))
    facade_z = min(maximum.z - 0.10, door.z + cityrender.STREET_TARGET_OFFSETS_SCENE[2])
    facade = Vector(((minimum.x + maximum.x) * 0.5, (minimum.y + maximum.y) * 0.5, facade_z))
    return door, {
        "building_ground_interface": ground,
        "door_lower_threshold": door_lower_threshold,
        "door_center": door_center,
        "facade_center": facade,
    }


def _perspective_points_fit(camera: Any, points: Sequence[Vector], view: Mapping[str, Any]) -> bool:
    """Check perspective fit without changing the candidate location."""

    bpy.context.view_layer.update()
    inverse = camera.matrix_world.inverted()
    local = [inverse @ point for point in points]
    visible = [point for point in local if point.z < -camera.data.clip_start]
    if not visible:
        return False
    half_horizontal = math.atan(camera.data.sensor_width / (2.0 * camera.data.lens))
    aspect = float(view["resolution"][0]) / float(view["resolution"][1])
    half_vertical = math.atan(math.tan(half_horizontal) / aspect)
    max_horizontal = max(abs(math.atan2(point.x, -point.z)) for point in visible)
    max_vertical = max(abs(math.atan2(point.y, -point.z)) for point in visible)
    return (
        max_horizontal <= half_horizontal * cityrender.STREET_FIT_MARGIN
        and max_vertical <= half_vertical * cityrender.STREET_FIT_MARGIN
    )


def _perspective_content_span(camera: Any, points: Sequence[Vector]) -> tuple[float, float, float]:
    """Return normalized projected content width, height, and area.

    Candidate cameras are already hard-gated for terrain LOS, finite-edge
    intrusion, and perspective fit.  This separate metric keeps the final
    choice focused: among valid views, prefer the one that gives the imported
    lot the largest stable projected footprint instead of rewarding a high but
    distant terrain sample that makes the subject microscopic.
    """

    inverse = camera.matrix_world.inverted()
    local = [inverse @ point for point in points]
    visible = [point for point in local if point.z < -camera.data.clip_start]
    if not visible:
        return (0.0, 0.0, 0.0)
    half_horizontal = math.atan(camera.data.sensor_width / (2.0 * camera.data.lens))
    aspect = 1.0
    # The ratio is only used for a comparable area score; fit itself remains
    # resolution-aware in _perspective_points_fit.
    half_vertical = math.atan(math.tan(half_horizontal) / aspect)
    xs = [math.atan2(point.x, -point.z) / half_horizontal for point in visible]
    ys = [math.atan2(point.y, -point.z) / half_vertical for point in visible]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return (float(width), float(height), float(width * height))


def _terrain_lower_support_audit(scene: Any, camera: Any, terrain: Any, content_points: Sequence[Vector]) -> dict[str, Any]:
    """Measure exact-terrain support below the imported subject in camera space.

    A perimeter-distance check cannot see the finite-field silhouette that
    appears immediately below a below-grade foundation.  This audit casts a
    small deterministic NDC lattice through the lower subject band against the
    imported final terrain mesh.  It is diagnostic here; the caller applies the
    hard threshold so a camera cannot pass merely because the strict subject
    bounds themselves are in frame.
    """

    content = _ndc_bounds(scene, camera, content_points)
    minimum = content.get("min")
    maximum = content.get("max")
    if not minimum or not maximum:
        return {"passed": False, "sample_count": 0, "terrain_hit_count": 0, "terrain_hit_fraction": 0.0, "reason": "no_content_bounds"}
    x0 = max(0.0, float(minimum[0]) - cityrender.TERRAIN_EDGE_SUBJECT_MARGIN_NDC)
    x1 = min(1.0, float(maximum[0]) + cityrender.TERRAIN_EDGE_SUBJECT_MARGIN_NDC)
    y0 = max(0.0, float(minimum[1]) - 0.12)
    y1 = min(1.0, float(minimum[1]) + 0.02)
    frame = camera.data.view_frame(scene=scene)
    # Blender's view_frame corners are ordered around the image rectangle.  A
    # direct min/max interpolation avoids depending on that winding order.
    left = min(float(point.x) for point in frame)
    right = max(float(point.x) for point in frame)
    bottom = min(float(point.y) for point in frame)
    top = max(float(point.y) for point in frame)
    camera_origin = camera.matrix_world.translation.copy()
    inverse_world = terrain.matrix_world.inverted()
    inverse_rotation = terrain.matrix_world.to_3x3().inverted()
    sample_count = 0
    hit_count = 0
    for x_index in range(7):
        x_ndc = x0 + (x1 - x0) * (x_index / 6.0)
        for y_index in range(5):
            y_ndc = y0 + (y1 - y0) * (y_index / 4.0)
            local_point = Vector((left + (right - left) * x_ndc, bottom + (top - bottom) * y_ndc, frame[0].z))
            world_point = camera.matrix_world @ local_point
            direction_world = (world_point - camera_origin).normalized()
            local_origin = inverse_world @ camera_origin
            local_direction = inverse_rotation @ direction_world
            _hit, _location, _normal, _index = terrain.ray_cast(local_origin, local_direction, distance=10000.0)
            sample_count += 1
            if _hit:
                hit_count += 1
    fraction = hit_count / float(sample_count or 1)
    return {
        "passed": True,
        "sample_count": sample_count,
        "terrain_hit_count": hit_count,
        "terrain_hit_fraction": round(fraction, 9),
        "ndc_band": [round(x0, 9), round(y0, 9), round(x1, 9), round(y1, 9)],
        "reason": None,
    }


def _terrain_building_occlusion_audit(
    scene: Any,
    camera: Any,
    terrain: Any,
    building_objects: Sequence[Any],
    content_points: Sequence[Vector],
) -> dict[str, Any]:
    """Compare terrain/building ray depth in the door/facade image band.

    LOS to four semantic points is necessary but not sufficient: a terrain
    ribbon can still cross the lower facade between those points.  This audit
    samples actual imported NIF geometry in the projected content bounds and
    records whether final terrain is the nearer hit.  The lower 12% of the
    bounds is excluded because below-grade foundations are allowed to be
    hidden; the next 58% is the readable door/facade band and is hard-gated.
    """

    content = _ndc_bounds(scene, camera, content_points)
    minimum = content.get("min")
    maximum = content.get("max")
    if not minimum or not maximum:
        return {"passed": False, "sample_count": 0, "building_hit_count": 0, "terrain_occluded_hit_count": 0, "terrain_occluded_fraction": 1.0, "reason": "no_content_bounds"}
    x0 = max(0.0, float(minimum[0]) - 0.01)
    x1 = min(1.0, float(maximum[0]) + 0.01)
    height = max(0.01, float(maximum[1]) - float(minimum[1]))
    y0 = max(0.0, float(minimum[1]) + height * 0.12)
    y1 = min(1.0, float(minimum[1]) + height * 0.70)
    frame = camera.data.view_frame(scene=scene)
    left = min(float(point.x) for point in frame)
    right = max(float(point.x) for point in frame)
    bottom = min(float(point.y) for point in frame)
    top = max(float(point.y) for point in frame)
    origin = camera.matrix_world.translation.copy()
    terrain_inverse = terrain.matrix_world.inverted()
    terrain_inverse_rotation = terrain.matrix_world.to_3x3().inverted()
    sample_count = 0
    building_hit_count = 0
    terrain_occluded_hit_count = 0
    for x_index in range(9):
        x_ndc = x0 + (x1 - x0) * (x_index / 8.0)
        for y_index in range(9):
            y_ndc = y0 + (y1 - y0) * (y_index / 8.0)
            local_point = Vector((left + (right - left) * x_ndc, bottom + (top - bottom) * y_ndc, frame[0].z))
            world_point = camera.matrix_world @ local_point
            direction = (world_point - origin).normalized()
            terrain_origin = terrain_inverse @ origin
            terrain_direction = terrain_inverse_rotation @ direction
            terrain_hit, terrain_location, _terrain_normal, _terrain_index = terrain.ray_cast(
                terrain_origin,
                terrain_direction,
                distance=10000.0,
            )
            terrain_distance = None
            if terrain_hit:
                terrain_world_location = terrain.matrix_world @ terrain_location
                terrain_distance = (terrain_world_location - origin).length
            building_distance = None
            for obj in building_objects:
                inverse = obj.matrix_world.inverted()
                local_origin = inverse @ origin
                local_direction = obj.matrix_world.to_3x3().inverted() @ direction
                hit, location, _normal, _index = obj.ray_cast(local_origin, local_direction, distance=10000.0)
                if hit:
                    world_location = obj.matrix_world @ location
                    distance = (world_location - origin).length
                    if building_distance is None or distance < building_distance:
                        building_distance = distance
            sample_count += 1
            if building_distance is not None:
                building_hit_count += 1
                if terrain_distance is not None and terrain_distance + 0.01 < building_distance:
                    terrain_occluded_hit_count += 1
    fraction = terrain_occluded_hit_count / float(building_hit_count or 1)
    return {
        "passed": True,
        "sample_count": sample_count,
        "building_hit_count": building_hit_count,
        "terrain_occluded_hit_count": terrain_occluded_hit_count,
        "terrain_occluded_fraction": round(fraction, 9),
        "ndc_band": [round(x0, 9), round(y0, 9), round(x1, 9), round(y1, 9)],
        "reason": None,
    }


def _terrain_door_band_audit(
    scene: Any,
    camera: Any,
    terrain: Any,
    building_objects: Sequence[Any],
    content_points: Sequence[Vector],
    targets: Mapping[str, Vector],
) -> dict[str, Any]:
    """Compare terrain/building depth specifically across the door/facade band."""

    content = _ndc_bounds(scene, camera, content_points)
    if not content.get("min") or not content.get("max"):
        return {"passed": False, "sample_count": 0, "building_hit_count": 0, "terrain_occluded_hit_count": 0, "terrain_occluded_fraction": 1.0, "reason": "no_content_bounds"}
    lower = world_to_camera_view(scene, camera, targets["door_lower_threshold"])
    center = world_to_camera_view(scene, camera, targets["door_center"])
    facade = world_to_camera_view(scene, camera, targets["facade_center"])
    x0 = max(0.0, float(content["min"][0]) - 0.01)
    x1 = min(1.0, float(content["max"][0]) + 0.01)
    y0 = max(0.0, min(float(lower.y), float(center.y)) - 0.025)
    y1 = min(1.0, max(float(center.y), float(facade.y)) + 0.025)
    frame = camera.data.view_frame(scene=scene)
    left = min(float(point.x) for point in frame)
    right = max(float(point.x) for point in frame)
    bottom = min(float(point.y) for point in frame)
    top = max(float(point.y) for point in frame)
    origin = camera.matrix_world.translation.copy()
    terrain_inverse = terrain.matrix_world.inverted()
    terrain_inverse_rotation = terrain.matrix_world.to_3x3().inverted()
    sample_count = 0
    building_hit_count = 0
    terrain_occluded_hit_count = 0
    for x_index in range(9):
        x_ndc = x0 + (x1 - x0) * (x_index / 8.0)
        for y_index in range(9):
            y_ndc = y0 + (y1 - y0) * (y_index / 8.0)
            local_point = Vector((left + (right - left) * x_ndc, bottom + (top - bottom) * y_ndc, frame[0].z))
            world_point = camera.matrix_world @ local_point
            direction = (world_point - origin).normalized()
            terrain_origin = terrain_inverse @ origin
            terrain_direction = terrain_inverse_rotation @ direction
            terrain_hit, terrain_location, _terrain_normal, _terrain_index = terrain.ray_cast(
                terrain_origin,
                terrain_direction,
                distance=10000.0,
            )
            terrain_distance = None
            if terrain_hit:
                terrain_distance = ((terrain.matrix_world @ terrain_location) - origin).length
            building_distance = None
            for obj in building_objects:
                inverse = obj.matrix_world.inverted()
                hit, location, _normal, _index = obj.ray_cast(
                    inverse @ origin,
                    obj.matrix_world.to_3x3().inverted() @ direction,
                    distance=10000.0,
                )
                if hit:
                    world_location = obj.matrix_world @ location
                    distance = (world_location - origin).length
                    if building_distance is None or distance < building_distance:
                        building_distance = distance
            sample_count += 1
            if building_distance is not None:
                building_hit_count += 1
                if terrain_distance is not None and terrain_distance + 0.01 < building_distance:
                    terrain_occluded_hit_count += 1
    fraction = terrain_occluded_hit_count / float(building_hit_count or 1)
    return {
        "passed": True,
        "sample_count": sample_count,
        "building_hit_count": building_hit_count,
        "terrain_occluded_hit_count": terrain_occluded_hit_count,
        "terrain_occluded_fraction": round(fraction, 9),
        "ndc_band": [round(x0, 9), round(y0, 9), round(x1, 9), round(y1, 9)],
        "target_ndc": {
            "door_lower_threshold": [round(float(lower.x), 9), round(float(lower.y), 9), round(float(lower.z), 9)],
            "door_center": [round(float(center.x), 9), round(float(center.y), 9), round(float(center.z), 9)],
            "facade_center": [round(float(facade.x), 9), round(float(facade.y), 9), round(float(facade.z), 9)],
        },
        "reason": None,
    }


def _street_camera(
    scene: Any,
    view: Mapping[str, Any],
    group_objects: Sequence[Any],
    render_origin_gu: Sequence[float],
    field: np.ndarray,
    fit_points: Sequence[Vector],
    terrain: Any,
) -> tuple[Any, dict[str, Any]]:
    """Select and construct the best deterministic terrain-clear street camera.

    The host contract supplies candidate XY positions derived from the actual
    door/road/tangent anchors.  This worker repeats the LOS and finite-edge
    tests against the exact loaded T1.3 height field and the actual imported
    NIF bounds, then selects by measured clearance rather than candidate order
    alone.  A candidate that cannot frame the imported subject is rejected
    rather than moved after selection.
    """

    contract = view.get("street_camera_contract")
    require(isinstance(contract, Mapping), f"{view['view_id']}: missing host street camera contract")
    door, targets = _street_targets(view, group_objects, field, render_origin_gu)
    minimum, maximum, content_points = _world_bounds(group_objects)
    all_fit_points = list(content_points) + list(fit_points)
    terrain_perimeter_points = _terrain_edge_points(field)
    candidates = contract.get("candidates")
    require(isinstance(candidates, list) and candidates, f"{view['view_id']}: street camera contract has no candidates")
    accepted: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        xy = candidate.get("camera_position_scene")
        if not isinstance(xy, list) or len(xy) != 3:
            fail(f"{view['view_id']}: malformed street candidate {candidate_id}")
        x_scene, y_scene = float(xy[0]), float(xy[1])
        terrain_height = cityrender.terrain_height_scene(field, x_scene, y_scene)
        reasons: list[str] = []
        if terrain_height is None:
            reasons.append("camera_outside_finite_terrain_field")
            camera_location = None
        else:
            camera_location = Vector((x_scene, y_scene, float(terrain_height) + cityrender.STREET_EYE_HEIGHT_SCENE))
        subject_edge_points = [door, *targets.values(), Vector((x_scene, y_scene, 0.0))]
        # Include the lower subject envelope in the edge metric so an image
        # cannot pass with the camera/door inside while the visible foundation
        # sits at the finite field boundary.
        subject_edge_points.extend(
            Vector((corner.x, corner.y, 0.0))
            for corner in _box_corners(minimum, Vector((maximum.x, maximum.y, minimum.z)))
        )
        edge_clearance = min(cityrender.terrain_edge_clearance_scene(point.x, point.y) for point in subject_edge_points)
        if edge_clearance < cityrender.STREET_EDGE_MARGIN_SCENE:
            reasons.append("camera_subject_or_foundation_too_close_to_terrain_edge")
        los_rows: dict[str, Any] = {}
        minimum_clearance = float("inf")
        minimum_door_clearance = float("inf")
        if camera_location is not None:
            for name, target in targets.items():
                required_clearance = (
                    cityrender.STREET_GROUND_INTERFACE_LOS_CLEARANCE_SCENE
                    if name == "building_ground_interface"
                    else cityrender.STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE
                )
                los = cityrender.terrain_line_of_sight(field, camera_location, target, minimum_clearance=required_clearance)
                los_rows[name] = los
                value = los.get("minimum_clearance_scene")
                if isinstance(value, (int, float)):
                    minimum_clearance = min(minimum_clearance, float(value))
                    if name != "building_ground_interface":
                        minimum_door_clearance = min(minimum_door_clearance, float(value))
                if not bool(los.get("passed")):
                    reasons.append(f"terrain_occludes_{name}")
        else:
            camera_location = Vector((x_scene, y_scene, 0.0))
        fit_lens = None
        content_span = (0.0, 0.0, 0.0)
        fit_candidate_seen = False
        edge_audit: dict[str, Any] = {
            "passed": False,
            "sample_count": len(terrain_perimeter_points),
            "intrusion_count": 0,
            "subject_ndc_margin": cityrender.TERRAIN_EDGE_SUBJECT_MARGIN_NDC,
            "intrusions": [],
        }
        if not reasons:
            for lens in (96.0, 84.0, 72.0, 60.0, 52.0, 48.0, 44.0, 40.0, 36.0, 32.0, 28.0, 24.0, 20.0, 18.0, 16.0, 14.0):
                facade_target = targets["facade_center"]
                composition_target = Vector(
                    (facade_target.x, facade_target.y, facade_target.z + cityrender.STREET_COMPOSITION_TARGET_RAISE_SCENE)
                )
                temporary = _camera_from_target(f"{view['view_id']}_candidate_{candidate_id}", composition_target, camera_location, "PERSP")
                temporary.data.lens = lens
                temporary.data.sensor_width = 36.0
                fits = _perspective_points_fit(temporary, all_fit_points, view)
                if fits:
                    fit_candidate_seen = True
                    edge_audit = _terrain_edge_intrusion(scene, temporary, content_points, terrain_perimeter_points)
                if fits and bool(edge_audit["passed"]):
                    fit_lens = lens
                    content_span = _perspective_content_span(temporary, content_points)
                    support_audit = _terrain_lower_support_audit(scene, temporary, terrain, content_points)
                    door_band_audit = _terrain_door_band_audit(scene, temporary, terrain, group_objects, content_points, targets)
                    if float(door_band_audit["terrain_occluded_fraction"]) > cityrender.STREET_MAX_DOOR_BAND_OCCLUSION_FRACTION:
                        reasons.append("terrain_occludes_readable_door_band")
                        fit_lens = None
                    else:
                        break
                bpy.data.objects.remove(temporary, do_unlink=True)
            if fit_lens is None:
                if fit_candidate_seen and not bool(edge_audit["passed"]):
                    reasons.append("finite_terrain_edge_intrudes_near_subject")
                else:
                    reasons.append("placed_subject_or_approach_outside_perspective_fit")
        passed = not reasons
        row = {
            "candidate_id": candidate_id,
            "family": str(candidate.get("family", "")),
            "host_passed": bool(candidate.get("passed")),
            "host_rejection_reasons": list(candidate.get("rejection_reasons", [])),
            "camera_position_scene": [round(float(value), 9) for value in (camera_location if camera_location is not None else (x_scene, y_scene, 0.0))],
            "terrain_height_scene": None if terrain_height is None else round(float(terrain_height), 9),
            "terrain_edge_clearance_scene": round(float(edge_clearance), 9),
            "terrain_los": los_rows,
            "minimum_terrain_los_clearance_scene": None if not math.isfinite(minimum_clearance) else round(minimum_clearance, 9),
            "minimum_door_terrain_los_clearance_scene": None if not math.isfinite(minimum_door_clearance) else round(minimum_door_clearance, 9),
            "fit_lens": fit_lens,
            "content_span_ndc": [round(value, 9) for value in content_span],
            "terrain_edge_intrusion": edge_audit,
            "terrain_lower_support": support_audit if fit_lens is not None else {
                "passed": False,
                "sample_count": 0,
                "terrain_hit_count": 0,
                "terrain_hit_fraction": 0.0,
                "reason": "no_fitted_lens",
            },
            "terrain_door_band": door_band_audit if fit_lens is not None else {
                "passed": False,
                "sample_count": 0,
                "building_hit_count": 0,
                "terrain_occluded_hit_count": 0,
                "terrain_occluded_fraction": 1.0,
                "reason": "no_fitted_lens",
            },
            "passed": passed,
            "rejection_reasons": sorted(set(reasons)),
            "order": int(candidate.get("order", 0)),
        }
        evidence.append(row)
        if passed:
            accepted.append(row)
    if not accepted:
        summary = "; ".join(
            f"{row['candidate_id']}[edge={row['terrain_edge_clearance_scene']}]={','.join(row['rejection_reasons']) or 'unknown'}"
            for row in evidence
        )
        fail(f"{view['view_id']}: no terrain-clear finite-edge street camera candidate remains after actual NIF fit; {summary}")
    selected = max(
        accepted,
        key=lambda row: (
            0 if str(row.get("family", "")) != "far_side_access_escape" else -1,
            float(row["content_span_ndc"][2]),
            float(row["content_span_ndc"][0]) + float(row["content_span_ndc"][1]),
            float(row["terrain_height_scene"]),
            float(row["minimum_terrain_los_clearance_scene"]),
            float(row["terrain_edge_clearance_scene"]),
            -int(row["order"]),
        ),
    )
    selected_location = Vector(tuple(float(value) for value in selected["camera_position_scene"]))
    # Use the same raised composition aim that was hard-gated during candidate
    # selection.  A lower final aim would reintroduce the terrain ribbon across
    # the door/facade even though the candidate's LOS test passed.
    facade_target = targets["facade_center"]
    target = Vector(
        (
            facade_target.x,
            facade_target.y,
            facade_target.z + cityrender.STREET_COMPOSITION_TARGET_RAISE_SCENE,
        )
    )
    camera = _camera_from_target(str(view["view_id"]), target, selected_location, "PERSP")
    camera.data.lens = float(selected["fit_lens"])
    camera.data.sensor_width = 36.0
    camera["cityforge_anchor_role"] = "actual_door_road_tangent_terrain_clear_candidate"
    camera["cityforge_road_id"] = str(view["road_id"])
    selected_los = selected["terrain_los"]
    selected_clearances = [
        float(row["minimum_clearance_scene"])
        for row in selected_los.values()
        if isinstance(row.get("minimum_clearance_scene"), (int, float))
    ]
    return camera, {
        "passed": True,
        "candidate_count": len(evidence),
        "rejected_candidate_count": sum(1 for row in evidence if not row["passed"]),
        "host_selected_candidate_id": str(contract.get("selected_candidate_id")),
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_minimum_terrain_los_clearance_scene": float(selected["minimum_terrain_los_clearance_scene"]),
        "selected_minimum_door_terrain_los_clearance_scene": float(selected["minimum_door_terrain_los_clearance_scene"]),
        "selected_edge_clearance_scene": float(selected["terrain_edge_clearance_scene"]),
        "minimum_clearance_scene": min(selected_clearances),
        "minimum_door_clearance_scene": min(
            float(row["minimum_clearance_scene"])
            for name, row in selected_los.items()
            if name != "building_ground_interface"
        ),
        "minimum_edge_clearance_scene": float(selected["terrain_edge_clearance_scene"]),
        "terrain_edge_intrusion": selected["terrain_edge_intrusion"],
        "terrain_edge_intrusion_count": int(selected["terrain_edge_intrusion"]["intrusion_count"]),
        "terrain_lower_support": selected["terrain_lower_support"],
        "terrain_lower_support_fraction": float(selected["terrain_lower_support"]["terrain_hit_fraction"]),
        "terrain_door_band": selected["terrain_door_band"],
        "terrain_door_band_occlusion_fraction": float(selected["terrain_door_band"]["terrain_occluded_fraction"]),
        "terrain_occluded_target_count": sum(1 for row in selected_los.values() if not bool(row.get("passed"))),
        "target_names": sorted(targets),
        "targets_scene": {name: [round(float(value), 9) for value in target] for name, target in targets.items()},
        "candidates": evidence,
    }


def _terrain_window_points(field: np.ndarray, minimum: Vector, maximum: Vector) -> list[Vector]:
    """Return a deterministic 3x3 exact-field patch for camera framing."""

    points: list[Vector] = []
    for x in (minimum.x, (minimum.x + maximum.x) * 0.5, maximum.x):
        for y in (minimum.y, (minimum.y + maximum.y) * 0.5, maximum.y):
            height = cityrender.terrain_height_scene(field, x, y)
            require(height is not None, "camera framing window leaves the finite final terrain field")
            points.append(Vector((x, y, float(height))))
    return points


def _render_view(
    scene: Any,
    scene_document: Mapping[str, Any],
    view: Mapping[str, Any],
    output_dir: Path,
    building_groups: Mapping[str, list[Any]],
    building_objects: Sequence[Any],
    terrain: Any,
    water: Any,
    field: np.ndarray,
) -> dict[str, Any]:
    """Create, fit, audit, render, and remove one camera."""

    _, terrain_max, terrain_points = _world_bounds([terrain])
    terrain_min, _, _ = _world_bounds([terrain])
    city_min, city_max, city_points = _world_bounds(building_objects)
    camera: Any
    if view["kind"] == "detail":
        focus_lot_ids = {str(value) for value in view.get("focus_lot_ids", [])}
        selected = [obj for lot_id in sorted(focus_lot_ids) for obj in building_groups.get(lot_id, [])]
        require(selected, f"{view['view_id']}: focused view has no imported objects for its lot ids")
        content_points = _world_bounds(selected)[2]
    elif view["kind"] == "street":
        selected = building_groups.get(str(view["lot_id"]), [])
        # Group mapping is keyed by lot id below; this branch is replaced by
        # the exact member key union when a lot has multiple members.
        if not selected:
            selected = [obj for key, objects in building_groups.items() if key in set(view["member_ref_keys"]) for obj in objects]
        require(selected, f"{view['view_id']}: selected door/lot has no imported objects")
        content_points = _world_bounds(selected)[2]
    else:
        content_points = city_points
    padding = 12.0 if view["kind"] == "detail" else 8.0 if view["kind"] == "street" else 20.0
    subject_min, subject_max, _ = _world_bounds(selected if view["kind"] in {"street", "detail"} else building_objects)
    fit_min_xy = Vector((max(terrain_min.x, subject_min.x - padding), max(terrain_min.y, subject_min.y - padding), 0.0))
    fit_max_xy = Vector((min(terrain_max.x, subject_max.x + padding), min(terrain_max.y, subject_max.y + padding), 0.0))
    require(fit_min_xy.x < fit_max_xy.x and fit_min_xy.y < fit_max_xy.y, f"{view['view_id']}: finite terrain cannot frame its content window")
    approach_points = _terrain_window_points(field, fit_min_xy, fit_max_xy)
    fit_points = list(content_points) + approach_points
    fit_min_z = min(point.z for point in fit_points)
    fit_max_z = max(point.z for point in fit_points)
    fit_min = Vector((fit_min_xy.x, fit_min_xy.y, fit_min_z))
    fit_max = Vector((fit_max_xy.x, fit_max_xy.y, max(fit_max_z, subject_max.z)))
    if view["kind"] in {"street", "detail"} and isinstance(view.get("street_camera_contract"), Mapping):
        door_world = [float(value) for value in view["door_world_gu"]]
        road_world = [float(value) for value in view["road_anchor_world_gu"]]
        street_approach_points: list[Vector] = []
        for fraction in (0.0, 0.5, 1.0):
            x_scene = (
                (door_world[0] + (road_world[0] - door_world[0]) * fraction) - float(scene_document["render_origin_gu"][0])
            ) * SCENE_UNITS_PER_GAME_UNIT
            y_scene = (
                (door_world[1] + (road_world[1] - door_world[1]) * fraction) - float(scene_document["render_origin_gu"][1])
            ) * SCENE_UNITS_PER_GAME_UNIT
            height = cityrender.terrain_height_scene(field, x_scene, y_scene)
            require(height is not None, f"{view['view_id']}: door-to-road approach leaves the finite terrain field")
            street_approach_points.append(Vector((x_scene, y_scene, float(height))))
        # A street camera fits the subject and the measured door-to-road
        # approach only.  The broad focused window above is for orthographic
        # views; forcing all of it into an eye-height perspective frustum would
        # put the camera back behind the very ridge the LOS gate rejects.
        fit_points = street_approach_points
        camera, terrain_los = _street_camera(scene, view, selected, scene_document["render_origin_gu"], field, fit_points, terrain)
    else:
        target = Vector(((fit_min.x + fit_max.x) / 2.0, (fit_min.y + fit_max.y) / 2.0, (fit_min.z + fit_max.z) / 2.0))
        if view["kind"] == "overview" or "horizontal_view_direction" not in view:
            span = max(fit_max.x - fit_min.x, fit_max.y - fit_min.y, 10.0)
            location = Vector((target.x, target.y, fit_max.z + span * 1.7))
        else:
            sx, sy = (float(value) for value in view["horizontal_view_direction"])
            view_direction = Vector((sx, sy, -0.62)).normalized()
            distance = max(fit_max.x - fit_min.x, fit_max.y - fit_min.y, fit_max.z - fit_min.z, 10.0) * 2.2
            location = target - view_direction * distance
        camera = _camera_from_target(str(view["view_id"]), target, location, "ORTHO")
        _fit_ortho(
            camera,
            fit_points,
            int(view["resolution"][0]),
            int(view["resolution"][1]),
            1.55 if view["kind"] == "detail" else 1.22,
        )
        terrain_los = None
    scene.camera = camera
    scene.render.resolution_x = int(view["resolution"][0])
    scene.render.resolution_y = int(view["resolution"][1])
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(output_dir / str(view["file"]))
    bpy.context.view_layer.update()
    # Street candidates already hard-gate the exact door-to-road terrain
    # segment in ``terrain_los``.  The perspective frustum itself is audited
    # against the placed subject; requiring the entire road segment to be
    # visible would reward a camera pulled away from the door-facing view.
    camera_fit_points = content_points if view["kind"] in {"street", "detail"} else fit_points
    camera_audit = _camera_audit(scene, camera, content_points, camera_fit_points, view)
    bpy.ops.render.render(write_still=True)
    output = output_dir / str(view["file"])
    require(output.is_file() and output.stat().st_size > 1024, f"{view['view_id']}: Blender did not write a non-trivial PNG")
    result = {
        "view_id": str(view["view_id"]),
        "kind": str(view["kind"]),
        "file": output.name,
        "resolution": [int(view["resolution"][0]), int(view["resolution"][1])],
        "selection_reason": view.get("selection_reason"),
        "lot_id": view.get("lot_id"),
        "door_ref_key": view.get("door_ref_key"),
        "road_id": view.get("road_id"),
        "camera": camera_audit,
        "content_basis": "placed NIF bounds; terrain fit is exact final field local bounds",
        "focused": bool(view.get("focused", False)),
        "focus_lot_ids": list(view.get("focus_lot_ids", [])),
    }
    if terrain_los is not None:
        result["terrain_los"] = terrain_los
    bpy.data.objects.remove(camera, do_unlink=True)
    return result


def run(scene_path: Path, output_dir: Path) -> dict[str, Any]:
    """Execute one complete scene/build and write the worker audit."""

    scene_document = read_json(scene_path)
    require(isinstance(scene_document, Mapping), "render scene must be a JSON object")
    require(scene_document.get("stage") == STAGE, "render scene has the wrong stage")
    require(scene_document.get("normalize_to_position") is False, "render scene normalization must be false")
    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = str(scene_document.get("scene_name", "Cityforge_T15"))
    _configure_render_scene(scene, scene_document["render"])

    ref_rows = list(scene_document.get("refs", []))
    require(ref_rows, "render scene has no placed refs")
    building_objects, render_groups, import_summary = _import_buildings(scene_document)
    group_map = {str(label): objects for label, objects in render_groups}
    require(set(group_map) == {str(row["ref_key"]) for row in ref_rows}, "imported ref group ids do not equal the host scene contract")
    bpy.context.view_layer.update()

    # Matrix audit happens against each imported hierarchy root.  A ref can
    # contain several roots; every root must carry the same exact placement.
    matrix_rows: list[dict[str, Any]] = []
    mismatch_count = 0
    max_error = 0.0
    max_translation_storage_error = 0.0
    ref_by_key = {str(row["ref_key"]): row for row in ref_rows}
    for label, objects in render_groups:
        row = ref_by_key[str(label)]
        expected = np.asarray(row["expected_relative_matrix"], dtype=np.float64)
        roots = _group_roots(objects)
        require(roots, f"{label}: imported group has no hierarchy root")
        root_rotation_errors = [
            _matrix_error(np.asarray(root.matrix_world, dtype=np.float64)[:3, :3], expected[:3, :3])
            for root in roots
        ]
        root_translation_errors = [
            float(np.max(np.abs(np.asarray(root.matrix_world, dtype=np.float64)[:3, 3] - expected[:3, 3])))
            for root in roots
        ]
        error = max(root_rotation_errors)
        translation_error = max(root_translation_errors)
        max_error = max(max_error, error)
        max_translation_storage_error = max(max_translation_storage_error, translation_error)
        if error > MATRIX_TOLERANCE or translation_error > TRANSLATION_FLOAT_STORAGE_TOLERANCE:
            mismatch_count += 1
            print(
                f"[cityforge-worker] MATRIX_MISMATCH {label} rotation_error={error:.9e} translation_storage_error={translation_error:.9e} "
                f"roots={[str(root.name) for root in roots]} actual={[[round(float(value), 6) for value in row] for row in roots[0].matrix_world]} "
                f"expected={[[round(float(value), 6) for value in row] for row in expected]}",
                flush=True,
            )
        matrix_rows.append(
            {
                "ref_key": str(label),
                "lot_id": str(row["lot_id"]),
                "source_id": str(row["source_id"]),
                "root_count": len(roots),
                "root_names": [str(root.name) for root in roots],
                "rotation_error": error,
                "translation_storage_error": translation_error,
                "full_matrix_storage_error": max(error, translation_error),
                "passed": error <= MATRIX_TOLERANCE and translation_error <= TRANSLATION_FLOAT_STORAGE_TOLERANCE,
            }
        )
    require(mismatch_count == 0, f"{mismatch_count} placed refs failed the Blender matrix gate")

    terrain, terrain_summary, terrain_triangles, field = _make_terrain(scene_document)
    terrain_min, terrain_max, _ = _world_bounds([terrain])
    water, water_summary = _make_water(terrain_triangles, (terrain_min, terrain_max))
    _setup_lighting(scene, terrain_min, Vector((terrain_max.x, terrain_max.y, max(terrain_max.z, _world_bounds(building_objects)[1].z))))
    bpy.context.view_layer.update()

    # Make an explicit lot-id map for street views while keeping each ref group
    # addressable by its exact host ref key.
    groups_by_lot: dict[str, list[Any]] = {}
    for row in ref_rows:
        groups_by_lot.setdefault(str(row["lot_id"]), []).extend(group_map[str(row["ref_key"])])
    render_groups_for_view = {**group_map, **groups_by_lot}
    view_rows: list[dict[str, Any]] = []
    for view in scene_document["views"]:
        view_rows.append(_render_view(scene, scene_document, view, output_dir, render_groups_for_view, building_objects, terrain, water, field))
    require(len(view_rows) == int(scene_document["counts"]["expected_view_count"]), "worker rendered view count disagrees with the scene contract")
    for row in view_rows:
        camera_audit = row.get("camera", {})
        require(bool(camera_audit.get("content_in_frame", {}).get("in_frame")), f"{row['view_id']}: content is outside the hard worker frustum gate")
        require(bool(camera_audit.get("fit_in_frame", {}).get("in_frame")), f"{row['view_id']}: approach terrain is outside the hard worker frustum gate")
        if row.get("kind") == "street":
            los = row.get("terrain_los")
            require(isinstance(los, Mapping) and bool(los.get("passed")), f"{row['view_id']}: street terrain LOS did not pass")
            require(int(sum(1 for candidate in los.get("candidates", []) if candidate.get("passed") and any(not result.get("passed") for result in candidate.get("terrain_los", {}).values()))) == 0, f"{row['view_id']}: selected candidate evidence contains an occluded target")
            require(bool(los.get("terrain_edge_intrusion", {}).get("passed")), f"{row['view_id']}: finite terrain edge intrudes near the subject")

    counts = {
        "expected_ref_count": len(ref_rows),
        "emitted_ref_count": len(ref_rows),
        "imported_ref_count": len(render_groups),
        "unique_model_count": len(scene_document.get("resolved_models", {})),
        "empty_import_count": 0,
        "unresolved_model_count": 0,
        "placeholder_material_count": int(import_summary["material_audit"]["placeholder_material_count"]),
        "unresolved_texture_count": int(import_summary["material_audit"]["unresolved_image_count"]),
        "building_texture_resolved_count": int(import_summary["material_audit"]["resolved_image_count"]),
        "building_texture_missing_count": int(import_summary["material_audit"]["unresolved_image_count"]),
        "matrix_checked_count": len(matrix_rows),
        "matrix_mismatch_count": mismatch_count,
        "max_matrix_error": max_error,
        "max_translation_storage_error": max_translation_storage_error,
        "multi_axis_canary_count": len(scene_document["matrix_canaries"]["multi_axis_refs"]),
        "near_gimbal_canary_count": len(scene_document["matrix_canaries"]["near_gimbal_refs"]),
        "expected_terrain_cell_count": 49,
        "terrain_cell_count": int(terrain_summary["cell_count"]),
        "terrain_field_record_mismatch_count": int(terrain_summary["field_record_mismatch_count"]),
        "terrain_texture_missing_count": int(terrain_summary["texture_missing_count"]),
        "expected_view_count": len(scene_document["views"]),
        "rendered_view_count": len(view_rows),
        "required_base_view_count": int(scene_document["counts"].get("required_base_view_count", cityrender.REQUIRED_BASE_VIEW_COUNT)),
        "focused_detail_view_count": int(scene_document["counts"].get("focused_detail_view_count", cityrender.FOCUSED_DETAIL_VIEW_COUNT)),
        "terrain_los_view_count": sum(1 for row in view_rows if row.get("kind") in {"street", "detail"} and "terrain_los" in row),
        "terrain_los_street_view_count": sum(1 for row in view_rows if row.get("kind") == "street" and "terrain_los" in row),
        "terrain_los_failed_view_count": sum(1 for row in view_rows if row.get("kind") in {"street", "detail"} and "terrain_los" in row and not row.get("terrain_los", {}).get("passed", False)),
        "terrain_door_band_failed_view_count": sum(
            1
            for row in view_rows
            if row.get("kind") in {"street", "detail"}
            and "terrain_los" in row
            and float(row.get("terrain_los", {}).get("terrain_door_band_occlusion_fraction", 1.0)) > cityrender.STREET_MAX_DOOR_BAND_OCCLUSION_FRACTION
        ),
        "terrain_edge_intrusion_failed_view_count": sum(
            1
            for row in view_rows
            if row.get("kind") in {"street", "detail"}
            and "terrain_los" in row
            and not row.get("terrain_los", {}).get("terrain_edge_intrusion", {}).get("passed", False)
        ),
        "terrain_edge_intrusion_sample_count": sum(
            int(row.get("terrain_los", {}).get("terrain_edge_intrusion", {}).get("sample_count", 0))
            for row in view_rows
            if row.get("kind") in {"street", "detail"} and "terrain_los" in row
        ),
        "proxy_geometry_count": 0,
        "flat_terrain_fallback_count": 0,
    }
    worker_audit = {
        "schema_version": cityrender.SCHEMA_VERSION,
        "stage": STAGE,
        "diagnostic_scope": scene_document.get("diagnostic_scope"),
        "synthetic_banner": scene_document.get("synthetic_banner"),
        "plan_id": scene_document.get("plan_id"),
        "build_hash": scene_document.get("build_hash"),
        "counts": counts,
        "imports": {
            "expected_ref_count": len(ref_rows),
            "resolved_entry_count": import_summary["resolved_entries"],
            "imported_group_count": len(render_groups),
            "imported_object_count": import_summary["imported_objects"],
            "visible_object_count": import_summary["visible_objects"],
            "unique_model_count": len(scene_document.get("resolved_models", {})),
            "duplicate_geometry_filter": import_summary["duplicate_filter"],
            "groups": [
                {
                    "ref_key": str(label),
                    "mesh_object_count": len(objects),
                    "visible_mesh_object_count": len([obj for obj in objects if not bool(obj.get("procgen_render_excluded", False))]),
                }
                for label, objects in render_groups
            ],
        },
        "textures": {
            "building": import_summary["material_audit"],
            "terrain": {
                "local_ltex_count": terrain_summary["ltex_count"],
                "resolved_count": terrain_summary["texture_resolved_count"],
                "missing_count": terrain_summary["texture_missing_count"],
                "fallback_count": 0,
                "opaque": True,
            },
        },
        "matrices": {
            "tolerance": MATRIX_TOLERANCE,
            "checked_count": len(matrix_rows),
            "mismatch_count": mismatch_count,
            "max_error": max_error,
            "rows": matrix_rows,
            "multi_axis_refs": list(scene_document["matrix_canaries"]["multi_axis_refs"]),
            "near_gimbal_refs": list(scene_document["matrix_canaries"]["near_gimbal_refs"]),
        },
        "terrain": {
            **terrain_summary,
            "field_record_mismatch_count": int(terrain_summary["field_record_mismatch_count"]),
            "opaque": True,
            "object_name": terrain.name,
            "rendered_from": "T1.3 final field + T1.3 local LAND VTEX/LTEX",
        },
        "water": water_summary,
        "views": view_rows,
    }
    write_json(output_dir / "blender_worker_audit.json", worker_audit)
    print(f"[cityforge-worker] ALL_DONE refs={len(ref_rows)} terrain_cells=49 views={len(view_rows)} water_triangles={water_summary['triangle_count']}", flush=True)
    return worker_audit


def main(argv: Sequence[str] | None = None) -> int:
    values = list(argv if argv is not None else (sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(values)
    run(args.scene.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BlenderRenderFailure as exc:
        print(f"FAILURE: render_city {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
