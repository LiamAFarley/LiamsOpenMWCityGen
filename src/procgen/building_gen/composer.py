"""Phase 6 deterministic minimum building composer.

Pipeline position
------------------
Consumes the Phase 5 compiled rule-kit and complete lot/access/terrain requests
and emits a complete composition-local generated stamp. This bounded phase
supports exact observed-template replay, one-shell primary access, and a named
secondary-access revision. Blender, TownLayout, ESP authoring, windows, roof
attachments, porches, tents, and new multi-shell graph search remain outside
this module.

The composer makes geometry decisions only from compiled evaluated-model
profiles and request-owned polygons. Native model bounds and ground polygons
are transformed with the shared OpenMW engine convention; no raw Euler or
parallel rotation math is used here. Failed extension requests return the
unchanged prior stamp with rejection evidence rather than a partial mutation.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from ..engine_transform import matrix_to_tes3_euler, rotate_reference_point, tes3_euler_to_matrix
from .contracts import (
    validate_building_extension_request,
    validate_building_request,
    validate_generated_building,
)
from .normalize import canonicalize, to_template_local


class ComposerError(ValueError):
    """Hard Phase 6 composition failure."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComposerError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ComposerError(f"{label} must be finite")
    return result


def _xy(value: Sequence[float], label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) < 2:
        raise ComposerError(f"{label} must contain at least two values")
    return (_finite(value[0], f"{label}[0]"), _finite(value[1], f"{label}[1]"))


def _vec3(value: Sequence[float], label: str) -> np.ndarray:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ComposerError(f"{label} must contain three values")
    result = np.asarray([_finite(item, f"{label}[{index}]") for index, item in enumerate(value)], dtype=np.float64)
    return result


def _unit(value: Sequence[float], label: str) -> np.ndarray:
    result = _vec3(value, label)
    length = float(np.linalg.norm(result))
    if length <= 1.0e-12:
        raise ComposerError(f"{label} must not be zero")
    return result / length


def _canonical_model_key(value: str) -> str:
    return str(value).replace("/", "\\").casefold()


def _index_rows(kit: Mapping[str, Any], key: str) -> dict[str, Mapping[str, Any]]:
    rows = kit.get(key)
    if not isinstance(rows, list):
        raise ComposerError(f"compiled kit has no {key} list")
    return {
        str(row["profile_id"]): row
        for row in rows
        if isinstance(row, Mapping) and row.get("selectable") is True and isinstance(row.get("profile_id"), str)
    }


def _compiled_indexes(kit: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    return {
        "models": {
            _canonical_model_key(str(row["model_key"])): row["profile"]
            for row in kit.get("model_profiles", [])
            if isinstance(row, Mapping)
            and isinstance(row.get("model_key"), str)
            and isinstance(row.get("profile"), Mapping)
        },
        "shells": _index_rows(kit, "shell_profiles"),
        "mounts": _index_rows(kit, "mount_profiles"),
        "access": _index_rows(kit, "access_bundles"),
        "templates": _index_rows(kit, "templates"),
    }


def _require(index: Mapping[str, Mapping[str, Any]], profile_id: str, label: str) -> Mapping[str, Any]:
    row = index.get(profile_id)
    if row is None:
        raise ComposerError(f"{label} is not selectable: {profile_id}")
    return row


def _ring(value: Sequence[Sequence[float]], label: str) -> list[tuple[float, float]]:
    points = [_xy(point, f"{label}[{index}]") for index, point in enumerate(value)]
    if len(points) < 3:
        raise ComposerError(f"{label} needs at least three points")
    return points


def _polygon(value: Sequence[Sequence[float]], label: str) -> Polygon:
    polygon = Polygon(_ring(value, label))
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
        raise ComposerError(f"{label} is not a valid positive-area polygon")
    return polygon


def _request_polygons(request: Mapping[str, Any]) -> tuple[Polygon, Polygon, list[Polygon]]:
    lot = _polygon(request["lot_polygon"], "request.lot_polygon")
    setback = _polygon(request["setback_polygon"], "request.setback_polygon")
    reserved = [_polygon(value, f"request.occupied_reserved_polygons[{index}]") for index, value in enumerate(request["occupied_reserved_polygons"])]
    return lot, setback, reserved


def _heading_vector(heading_deg: float) -> np.ndarray:
    radians = math.radians(_finite(heading_deg, "heading_deg"))
    return np.asarray([math.cos(radians), math.sin(radians), 0.0], dtype=np.float64)


def _heading_deg(vector: Sequence[float]) -> float:
    unit = _unit(vector, "heading vector")
    if abs(float(unit[0])) <= 1.0e-12 and abs(float(unit[1])) <= 1.0e-12:
        raise ComposerError("heading vector has no horizontal component")
    result = math.degrees(math.atan2(float(unit[1]), float(unit[0]))) % 360.0
    return 0.0 if abs(result) <= 1.0e-12 else result


def _member_profile(indexes: Mapping[str, Mapping[str, Mapping[str, Any]]], model_key: str) -> Mapping[str, Any]:
    profile = indexes["models"].get(_canonical_model_key(model_key))
    if profile is None:
        raise ComposerError(f"model profile is missing: {model_key}")
    return profile


def _transform_point(point: Sequence[float], offset: Sequence[float], rotation: Sequence[float], scale: float) -> np.ndarray:
    local = _vec3(point, "local point") * _finite(scale, "member scale")
    rotated = np.asarray(rotate_reference_point(local.tolist(), rotation), dtype=np.float64)
    return _vec3(offset, "member offset") + rotated


def _transform_xy(point: Sequence[float], offset: Sequence[float], rotation: Sequence[float], scale: float) -> tuple[float, float]:
    transformed = _transform_point([float(point[0]), float(point[1]), 0.0], offset, rotation, scale)
    return (float(transformed[0]), float(transformed[1]))


def _native_bounds_corners(bounds: Mapping[str, Any]) -> list[list[float]]:
    minimum = _vec3(bounds["min"], "bounds.min")
    maximum = _vec3(bounds["max"], "bounds.max")
    return [
        [x, y, z]
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]


def _rotate_frame(frame: Mapping[str, Any], rotation: Sequence[float], scale: float, offset: Sequence[float]) -> dict[str, Any]:
    engine = tes3_euler_to_matrix(rotation)
    result: dict[str, Any] = {}
    for key in ("n", "u", "v"):
        vector = engine @ _vec3(frame[key], f"facade frame {key}")
        result[key] = vector.tolist()
    plane_point = _vec3(frame["n"], "facade frame n") * _finite(frame["plane_offset_gu"], "facade plane offset")
    transformed = _transform_point(plane_point.tolist(), offset, rotation, scale)
    result["plane_point"] = transformed.tolist()
    result["plane_offset_gu"] = float(np.dot(transformed, _unit(result["n"], "rotated facade normal")))
    return result


def _facade_center_clear(facade: Mapping[str, Any], mount_profile: Mapping[str, Any] | None) -> bool:
    if mount_profile is None:
        return True
    region = facade.get("usable_region_uz")
    if not isinstance(region, Mapping):
        return False
    center_u = (float(region["u"][0]) + float(region["u"][1])) * 0.5
    center_v = (float(region["z"][0]) + float(region["z"][1])) * 0.5
    envelope = mount_profile.get("occupied_envelope_gu")
    if not isinstance(envelope, Mapping):
        raise ComposerError("access mount has no occupied envelope")
    tangent = envelope.get("tangent_gu")
    up = envelope.get("up_gu")
    if not isinstance(tangent, Sequence) or not isinstance(up, Sequence):
        raise ComposerError("access mount occupied envelope is incomplete")
    for occupied in facade.get("occupied_regions", []):
        if not isinstance(occupied, Mapping) or not isinstance(occupied.get("u_gu"), (int, float)) or not isinstance(occupied.get("z_gu"), (int, float)):
            continue
        delta_u = float(occupied["u_gu"]) - center_u
        delta_v = float(occupied["z_gu"]) - center_v
        if float(tangent[0]) <= delta_u <= float(tangent[1]) and float(up[0]) <= delta_v <= float(up[1]):
            return False
    return True


def _facade_candidate(shell_profile: Mapping[str, Any], heading_deg: float, min_body_z_gu: float, mount_profile: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    target = _heading_vector(heading_deg)
    candidates: list[tuple[float, str, Mapping[str, Any]]] = []
    for facade in shell_profile.get("facades", []):
        if not isinstance(facade, Mapping):
            continue
        region = facade.get("usable_region_uz")
        frame = facade.get("outward_frame")
        if not isinstance(region, Mapping) or not isinstance(frame, Mapping):
            continue
        z_interval = region.get("z")
        if not isinstance(z_interval, Sequence) or len(z_interval) != 2 or float(z_interval[1]) < min_body_z_gu:
            continue
        if not _facade_center_clear(facade, mount_profile):
            continue
        normal = _unit(frame.get("n"), f"facade {facade.get('facade_id')} normal")
        horizontal = np.asarray([normal[0], normal[1], 0.0], dtype=np.float64)
        horizontal_length = float(np.linalg.norm(horizontal))
        if horizontal_length <= 1.0e-9:
            continue
        score = float(np.dot(horizontal / horizontal_length, target))
        candidates.append((-score, str(facade.get("facade_id")), facade))
    if not candidates:
        raise ComposerError("no usable body facade is available for the requested access heading")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _facade_point(facade: Mapping[str, Any]) -> np.ndarray:
    frame = facade["outward_frame"]
    region = facade["usable_region_uz"]
    u_value = (float(region["u"][0]) + float(region["u"][1])) * 0.5
    v_value = (float(region["z"][0]) + float(region["z"][1])) * 0.5
    return (
        _unit(frame["n"], "facade normal") * float(frame["plane_offset_gu"])
        + _unit(frame["u"], "facade tangent") * u_value
        + _unit(frame["v"], "facade up") * v_value
    )


def _basis(columns: Sequence[Sequence[float]], label: str) -> np.ndarray:
    matrix = np.column_stack([_unit(column, f"{label}[{index}]") for index, column in enumerate(columns)])
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or abs(abs(determinant) - 1.0) > 1.0e-4:
        raise ComposerError(f"{label} is not an orthonormal basis")
    return matrix


def _mount_transform(mount_profile: Mapping[str, Any], facade_frame: Mapping[str, Any]) -> tuple[list[float], list[float], list[float]]:
    mount_frame = mount_profile["mount_frame"]
    source = _basis([mount_frame["u_tangent"], mount_frame["v_up"], mount_frame["n"]], "mount frame")
    target = _basis([facade_frame["u"], facade_frame["v"], facade_frame["n"]], "facade frame")
    engine_matrix = target @ source.T
    rotation = list(matrix_to_tes3_euler(engine_matrix))
    contact = mount_profile.get("contact_polygon_uv")
    if not isinstance(contact, list) or not contact:
        interval = mount_profile.get("contact_interval_uv")
        if not isinstance(interval, Mapping):
            raise ComposerError("access mount has no measured contact geometry")
        contact = [
            [interval["u"][0], interval["v"][0]],
            [interval["u"][1], interval["v"][0]],
            [interval["u"][1], interval["v"][1]],
            [interval["u"][0], interval["v"][1]],
        ]
    contact_center = np.mean(np.asarray([_xy(point, "mount contact point") for point in contact], dtype=np.float64), axis=0)
    target_point = _vec3(facade_frame["plane_point"], "facade target point")
    local_contact = np.asarray([contact_center[0], contact_center[1], 0.0], dtype=np.float64)
    offset = target_point - engine_matrix @ local_contact
    return offset.tolist(), rotation, target[2].tolist()


def _output_member(
    *,
    source_id: str,
    object_id: str,
    model_key: str,
    record_type: str,
    offset: Sequence[float],
    rotation: Sequence[float],
    scale: float,
    structural_role: str,
    is_door: bool = False,
    outward_heading_deg: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "category": "door" if is_door else "exterior",
        "is_door": is_door,
        "model_key": str(model_key),
        "object_id": str(object_id),
        "offset_gu": [float(value) for value in offset],
        "outward_heading_deg": outward_heading_deg,
        "record_type": str(record_type),
        "rotation": [float(value) for value in rotation],
        "scale": float(scale),
        "source_id": str(source_id),
        "structural_role": str(structural_role),
    }
    if is_door:
        result["door"] = {
            "destination_cell": None,
            "destination_position_gu": None,
            "destination_rotation": None,
        }
    return result


def _rebase_members(members: Sequence[Mapping[str, Any]], seed_id: str) -> list[dict[str, Any]]:
    seed = next((member for member in members if member.get("source_id") == seed_id), None)
    if seed is None:
        raise ComposerError(f"seed door is not in emitted members: {seed_id}")
    origin = seed["offset_gu"]
    frame = tes3_euler_to_matrix(seed["rotation"])
    result: list[dict[str, Any]] = []
    for member in members:
        offset, rotation = to_template_local(member["offset_gu"], member["rotation"], origin, frame)
        row = copy.deepcopy(dict(member))
        row["offset_gu"] = offset
        row["rotation"] = rotation
        result.append(row)
    return result


def _canonical_ring(points: Sequence[Sequence[float]]) -> list[list[float]]:
    raw = [(round(float(point[0]), 6), round(float(point[1]), 6)) for point in points]
    if raw and raw[0] == raw[-1]:
        raw.pop()
    if len(raw) < 3:
        raise ComposerError("polygon ring has fewer than three points")
    choices: list[tuple[tuple[float, float], ...]] = []
    for direction in (raw, list(reversed(raw))):
        for index in range(len(direction)):
            choices.append(tuple(direction[index:] + direction[:index]))
    return [[float(x), float(y)] for x, y in min(choices)]


def _geometry_products(members: Sequence[Mapping[str, Any]], indexes: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> tuple[Polygon | MultiPolygon, list[list[list[float]]], dict[str, list[float]], dict[str, list[float]]]:
    ground_polygons: list[Polygon] = []
    bounds_points: list[np.ndarray] = []
    for member in members:
        profile = _member_profile(indexes, str(member["model_key"]))
        offset = member["offset_gu"]
        rotation = member["rotation"]
        scale = float(member["scale"])
        bounds = profile.get("bounds_local_gu")
        if not isinstance(bounds, Mapping):
            raise ComposerError(f"model has no native bounds: {member['model_key']}")
        bounds_points.extend(_transform_point(corner, offset, rotation, scale) for corner in _native_bounds_corners(bounds))
        polygon = profile.get("ground_polygon_xy")
        if isinstance(polygon, list) and len(polygon) >= 3:
            transformed = [_transform_xy(point, offset, rotation, scale) for point in polygon]
            candidate = Polygon(transformed)
            if candidate.is_empty or not candidate.is_valid or candidate.area <= 0.0:
                raise ComposerError(f"invalid transformed ground polygon: {member['model_key']}")
            ground_polygons.append(candidate)
    if not ground_polygons:
        raise ComposerError("generated building has no measured ground-reaching components")
    union = unary_union(ground_polygons)
    if union.is_empty or not union.is_valid:
        raise ComposerError("ground component union is invalid")
    components = list(union.geoms) if isinstance(union, MultiPolygon) else [union]
    components.sort(key=lambda item: (round(item.bounds[0], 6), round(item.bounds[1], 6), round(-item.area, 6)))
    component_rings = [[_canonical_ring(component.exterior.coords) for component in components]]
    # Keep the public D-STAMP component shape as a list of rings, while the
    # audit field retains whether the conservative hull was necessary.
    rings = component_rings[0]
    hull = union.exterior if union.geom_type == "Polygon" else union.convex_hull.exterior
    hull_ring = _canonical_ring(hull.coords)
    minimum = np.min(np.asarray(bounds_points), axis=0)
    maximum = np.max(np.asarray(bounds_points), axis=0)
    bounds = {"min": minimum.tolist(), "max": maximum.tolist()}
    return union, rings, bounds, {"hull": hull_ring, "used_convex_hull": union.geom_type != "Polygon"}


def _sample_terrain(context: Mapping[str, Any], points: Sequence[tuple[float, float, str]]) -> tuple[list[dict[str, Any]], float, float, float]:
    if context.get("mode") != "plane":
        raise ComposerError("Phase 6 requires an explicit plane terrain context")
    origin = _xy(context.get("origin_gu", [0.0, 0.0]), "terrain_context.origin_gu")
    bounds = context.get("bounds_gu")
    if not isinstance(bounds, Sequence) or len(bounds) != 2:
        raise ComposerError("terrain_context.bounds_gu is required")
    min_bound, max_bound = _xy(bounds[0], "terrain_context.bounds_gu.min"), _xy(bounds[1], "terrain_context.bounds_gu.max")
    height = _finite(context.get("height_gu"), "terrain_context.height_gu")
    slope_x = _finite(context.get("slope_x"), "terrain_context.slope_x")
    slope_y = _finite(context.get("slope_y"), "terrain_context.slope_y")
    slope_deg = math.degrees(math.atan(math.hypot(slope_x, slope_y)))
    unique = sorted({(round(float(x), 6), round(float(y), 6), str(kind)) for x, y, kind in points})
    samples: list[dict[str, Any]] = []
    for x, y, kind in unique:
        if x < min_bound[0] or x > max_bound[0] or y < min_bound[1] or y > max_bound[1]:
            raise ComposerError(f"terrain context does not cover sample {x},{y}")
        sample_height = height + slope_x * (x - origin[0]) + slope_y * (y - origin[1])
        samples.append({"kind": kind, "x_gu": x, "y_gu": y, "height_gu": sample_height, "slope_deg": slope_deg})
    heights = [float(sample["height_gu"]) for sample in samples]
    return samples, (max(heights) - min(heights) if heights else 0.0), slope_deg, max(heights) if heights else height


def _surface_heading(surface: Mapping[str, Any], label: str) -> float:
    if "heading_deg" not in surface:
        raise ComposerError(f"{label}.heading_deg is required")
    return _finite(surface["heading_deg"], f"{label}.heading_deg")


def _check_spatial_contract(request: Mapping[str, Any], hull: Polygon | MultiPolygon, point: Sequence[float], free_space: Sequence[Polygon] = ()) -> None:
    lot, setback, reserved = _request_polygons(request)
    tolerance = _finite(request.get("runtime_caps", {}).get("containment_tolerance_gu", 0.0), "containment tolerance")
    envelope = hull.buffer(tolerance) if tolerance else hull
    if not lot.covers(envelope):
        raise ComposerError("lot_containment_failed")
    if not setback.covers(envelope):
        raise ComposerError("setback_containment_failed")
    for index, polygon in enumerate(reserved):
        if envelope.intersects(polygon) and envelope.intersection(polygon).area > tolerance:
            raise ComposerError(f"occupied_reserved_conflict:{index}")
    if free_space and not any(polygon.covers(Point(float(point[0]), float(point[1]))) for polygon in free_space):
        raise ComposerError("secondary_access_free_space_failed")


def _facade_frame_for_member(
    facade: Mapping[str, Any], member: Mapping[str, Any], indexes: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> tuple[dict[str, Any], np.ndarray]:
    scale = float(member["scale"])
    rotated = _rotate_frame(facade["outward_frame"], member["rotation"], scale, member["offset_gu"])
    # _rotate_frame's point uses the shell plane point; its frame vectors are
    # already engine-rotated, so use them as the current composition frame.
    region = facade["usable_region_uz"]
    target = _vec3(rotated["plane_point"], "transformed facade point")
    target += _unit(rotated["u"], "transformed facade u") * ((float(region["u"][0]) + float(region["u"][1])) * 0.5) * scale
    target += _unit(rotated["v"], "transformed facade v") * ((float(region["z"][0]) + float(region["z"][1])) * 0.5) * scale
    rotated["plane_point"] = target.tolist()
    return rotated, target


def _access_bundle_mount(
    access: Mapping[str, Any], indexes: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    profile = access["profile"]
    frame = profile["frame_member"]
    mount_id = f"mount:{frame['model_key']}"
    mount_row = _require(indexes["mounts"], mount_id, "access frame mount")
    mount_profile = mount_row["profile"]
    return profile, mount_row, mount_profile


def _base_members(request: Mapping[str, Any], indexes: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = str(request["composition_mode"])
    access_row = _require(indexes["access"], str(request["access_profile_id"]), "primary access profile")
    access, _, mount = _access_bundle_mount(access_row, indexes)
    if mode == "observed_template":
        template_row = _require(indexes["templates"], str(request["template_profile_id"]), "observed template")
        template = template_row["profile"]
        members: list[dict[str, Any]] = []
        for source in template["members"]:
            is_door = bool(source.get("is_door"))
            members.append(_output_member(
                source_id=str(source["source_id"]),
                object_id=str(source["object_id"]),
                model_key=str(source["model_key"]),
                record_type=str(source["record_type"]),
                offset=source["offset_local_gu"],
                rotation=source["rotation_local_rad"],
                scale=float(source["scale"]),
                structural_role=str(source["structural_role"]),
                is_door=is_door,
                outward_heading_deg=float(access["outward_heading_in_slot_deg"]) if is_door else None,
            ))
        source_door = next(member for member in members if member["source_id"] == template["seed_door"])
        if source_door["model_key"] != access["door_member"]["model_key"] or abs(float(source_door["scale"]) - float(access["door_member"]["scale"])) > 1.0e-6:
            raise ComposerError("template_access_door_mismatch")
        frame_members = [member for member in members if member["structural_role"] == "doorframe"]
        if not frame_members or frame_members[0]["model_key"] != access["frame_member"]["model_key"]:
            raise ComposerError("template_access_frame_mismatch")
        return members, {
            "mode": mode,
            "access_profile_id": access_row["profile_id"],
            "template_profile_id": template_row["profile_id"],
            "primary_door_id": template["seed_door"],
            "primary_heading_deg": float(access["outward_heading_in_slot_deg"]),
            "access_profile": access,
            "mount_profile": mount,
        }
    if mode != "single_shell":
        raise ComposerError(f"unsupported composition_mode: {mode}")
    shell_row = _require(indexes["shells"], str(request["shell_profile_id"]), "shell profile")
    shell = shell_row["profile"]
    shell_scale = _finite(request.get("shell_scale"), "request.shell_scale")
    target_heading = _surface_heading(request["primary_access_surface"], "primary_access_surface")
    facade = _facade_candidate(shell, target_heading, _finite(request["body_min_z_gu"], "body_min_z_gu"), mount)
    facade_frame = dict(facade["outward_frame"])
    facade_frame["plane_point"] = _facade_point(facade).tolist()
    frame_offset, frame_rotation, target_normal = _mount_transform(mount, facade_frame)
    frame = _output_member(
        source_id=f"generated:{request['request_id']}:primary_frame",
        object_id=str(access["frame_member"]["model_key"]),
        model_key=str(access["frame_member"]["model_key"]),
        record_type="STAT",
        offset=frame_offset,
        rotation=frame_rotation,
        scale=float(access["frame_member"]["scale"]),
        structural_role="doorframe",
    )
    door = _output_member(
        source_id=f"generated:{request['request_id']}:primary_door",
        object_id=str(access["door_member"]["model_key"]),
        model_key=str(access["door_member"]["model_key"]),
        record_type="DOOR",
        offset=frame_offset,
        rotation=frame_rotation,
        scale=float(access["door_member"]["scale"]),
        structural_role="door",
        is_door=True,
        outward_heading_deg=_heading_deg(target_normal),
    )
    shell_member = _output_member(
        source_id=f"generated:{request['request_id']}:shell",
        object_id=str(shell["model_key"]),
        model_key=str(shell["model_key"]),
        record_type="STAT",
        offset=[0.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 0.0],
        scale=shell_scale,
        structural_role="shell",
    )
    members = _rebase_members([shell_member, frame, door], door["source_id"])
    return members, {
        "mode": mode,
        "access_profile_id": access_row["profile_id"],
        "shell_profile_id": shell_row["profile_id"],
        "primary_door_id": door["source_id"],
        "primary_heading_deg": next(member for member in members if member["source_id"] == door["source_id"])["outward_heading_deg"],
        "access_profile": access,
        "mount_profile": mount,
        "selected_facade_id": facade["facade_id"],
        "requested_heading_deg": target_heading,
    }


def _all_facades(members: Sequence[Mapping[str, Any]], indexes: Mapping[str, Mapping[str, Mapping[str, Any]]], heading_deg: float, body_min_z_gu: float, mount_profile: Mapping[str, Any]) -> list[tuple[float, str, Mapping[str, Any], Mapping[str, Any], np.ndarray]]:
    target = _heading_vector(heading_deg)
    candidates: list[tuple[float, str, Mapping[str, Any], Mapping[str, Any], np.ndarray]] = []
    for member in members:
        if member.get("structural_role") != "shell":
            continue
        shell_row = _require(indexes["shells"], f"shell:{member['model_key']}", "shell facade profile")
        shell = shell_row["profile"]
        for facade in shell.get("facades", []):
            region = facade.get("usable_region_uz") if isinstance(facade, Mapping) else None
            if not isinstance(region, Mapping) or float(region["z"][1]) < body_min_z_gu:
                continue
            if not _facade_center_clear(facade, mount_profile):
                continue
            transformed, point = _facade_frame_for_member(facade, member, indexes)
            normal = _unit(transformed["n"], "transformed facade normal")
            horizontal = np.asarray([normal[0], normal[1], 0.0], dtype=np.float64)
            if float(np.linalg.norm(horizontal)) <= 1.0e-9:
                continue
            score = float(np.dot(horizontal / np.linalg.norm(horizontal), target))
            candidates.append((-score, f"{member['source_id']}:{facade['facade_id']}", facade, transformed, point))
    candidates.sort(key=lambda row: (row[0], row[1]))
    if not candidates:
        raise ComposerError("no exposed facade is available for secondary access")
    return candidates


def _append_access(
    members: list[dict[str, Any]],
    request: Mapping[str, Any],
    access_surface: Mapping[str, Any],
    indexes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ordinal: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    access_id = access_surface.get("access_profile_id")
    if not isinstance(access_id, str):
        raise ComposerError("secondary access surface has no access_profile_id")
    access_row = _require(indexes["access"], access_id, "secondary access profile")
    access, _, mount = _access_bundle_mount(access_row, indexes)
    heading = _surface_heading(access_surface, "secondary_access_surface")
    candidates = _all_facades(members, indexes, heading, _finite(request["body_min_z_gu"], "body_min_z_gu"), mount)
    free_space_polygons = [_polygon(value, f"extension.free_space_polygons[{index}]") for index, value in enumerate(request["free_space_polygons"])]
    access_clearance = _finite(request["secondary_access_clearance_gu"], "secondary_access_clearance_gu")
    if access_clearance < 0.0:
        raise ComposerError("secondary_access_clearance_gu must be non-negative")
    existing_door_points = [
        np.asarray(member["offset_gu"], dtype=np.float64)
        for member in members
        if bool(member.get("is_door"))
    ]
    free_space_hits = 0
    clearance_rejections = 0
    for _, facade_key, _, facade_frame, target in candidates:
        if not any(poly.covers(Point(float(target[0]), float(target[1]))) for poly in free_space_polygons):
            continue
        free_space_hits += 1
        if any(float(np.linalg.norm(target[:2] - point[:2])) < access_clearance for point in existing_door_points):
            clearance_rejections += 1
            continue
        frame_offset, frame_rotation, target_normal = _mount_transform(mount, facade_frame)
        frame_id = f"generated:{request['request_id']}:secondary_frame_{ordinal:03d}"
        door_id = f"generated:{request['request_id']}:secondary_door_{ordinal:03d}"
        frame = _output_member(
            source_id=frame_id,
            object_id=str(access["frame_member"]["model_key"]),
            model_key=str(access["frame_member"]["model_key"]),
            record_type="STAT",
            offset=frame_offset,
            rotation=frame_rotation,
            scale=float(access["frame_member"]["scale"]),
            structural_role="doorframe",
        )
        door = _output_member(
            source_id=door_id,
            object_id=str(access["door_member"]["model_key"]),
            model_key=str(access["door_member"]["model_key"]),
            record_type="DOOR",
            offset=frame_offset,
            rotation=frame_rotation,
            scale=float(access["door_member"]["scale"]),
            structural_role="door",
            is_door=True,
            outward_heading_deg=_heading_deg(target_normal),
        )
        return members + [frame, door], {
            "access_profile_id": access_row["profile_id"],
            "facade_key": facade_key,
            "door_id": door_id,
            "heading_deg": _heading_deg(target_normal),
            "access_profile": access,
        }
    if free_space_hits == 0:
        raise ComposerError("secondary_access_free_space_failed")
    if clearance_rejections:
        raise ComposerError("secondary_access_clearance_failed")
    raise ComposerError("secondary_access_free_space_failed")


def _finalize(
    request: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    indexes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    door_steps: Mapping[str, float],
    revision_id: str,
    prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = [copy.deepcopy(dict(member)) for member in members]
    seed_id = str(metadata["primary_door_id"] if prior is None else next(member["source_id"] for member in normalized if member.get("source_id") == prior["source"]["seed_door"]))
    if prior is None and any(member["source_id"] == seed_id and member["offset_gu"] != [0.0, 0.0, 0.0] for member in normalized):
        normalized = _rebase_members(normalized, seed_id)
    union, components, bounds, hull_meta = _geometry_products(normalized, indexes)
    hull = Polygon(hull_meta["hull"])
    _check_spatial_contract(request, hull, [0.0, 0.0])
    door_members = [member for member in normalized if bool(member.get("is_door"))]
    if not door_members:
        raise ComposerError("door_frame_missing")
    sample_points: list[tuple[float, float, str]] = []
    ring = list(hull.exterior.coords)[:-1]
    sample_points.extend((x, y, "hull_vertex") for x, y in ring)
    closed = ring + [ring[0]]
    sample_points.extend(((closed[index][0] + closed[index + 1][0]) * 0.5, (closed[index][1] + closed[index + 1][1]) * 0.5, "edge_midpoint") for index in range(len(ring)))
    sample_points.extend((float(member["offset_gu"][0]), float(member["offset_gu"][1]), "member_origin") for member in normalized)
    sample_points.extend((float(member["offset_gu"][0]), float(member["offset_gu"][1]), "door_origin") for member in door_members)
    terrain_samples, relief, slope_deg, max_height = _sample_terrain(request["terrain_context"], sample_points)
    bounds_min_z = float(bounds["min"][2])
    burial = max_height - bounds_min_z
    terrain_policy = request["terrain_edit_policy"]
    max_burial = terrain_policy.get("max_burial_gu", request.get("runtime_caps", {}).get("max_burial_gu"))
    if max_burial is not None and burial > _finite(max_burial, "max_burial_gu"):
        raise ComposerError("terrain_penetration_exceeded")
    step_values = [float(door_steps.get(str(member["source_id"]), 0.0)) for member in door_members]
    primary_heading = float(next(member for member in door_members if member["source_id"] == str(metadata["primary_door_id"]))["outward_heading_deg"])
    building = {
        "access_heading_rad": math.radians(primary_heading),
        "bounds_rel_gu": bounds,
        "building_type": str(request["building_type"]),
        "composition": {
            "mode": metadata["mode"],
            "selected_access_profile_id": metadata["access_profile_id"],
            "selected_shell_profile_id": metadata.get("shell_profile_id"),
            "selected_template_profile_id": metadata.get("template_profile_id"),
            "selected_facade_id": metadata.get("selected_facade_id"),
            "requested_primary_heading_deg": request["primary_access_surface"].get("heading_deg"),
            "primary_heading_deg": primary_heading,
            "used_convex_hull": bool(hull_meta["used_convex_hull"]),
        },
        "door_count": len(door_members),
        "footprint": {
            "aabb_rel": {"min": [bounds["min"][0], bounds["min"][1], 0.0], "max": [bounds["max"][0], bounds["max"][1], 0.0]},
            "components_xy_rel": components,
            "hull_xy_rel": hull_meta["hull"],
        },
        "members": normalized,
        "multi_shell": sum(member.get("structural_role") == "shell" for member in normalized) > 1,
        "revision_id": revision_id,
        "size_class": str(request["requested_size"]),
        "source": {
            "seed_door": str(metadata["primary_door_id"]),
            "phase": 6,
            "palette_id": request["palette_id"],
        },
        "stamp_id": str(prior.get("stamp_id")) if prior is not None else f"generated:{request['request_id']}",
        "terrain_envelope": {
            "burial_depth_gu": burial,
            "door_step_heights_gu": step_values,
            "footprint_relief_gu": relief,
            "footprint_slope_deg": slope_deg,
        },
        "terrain_evidence": {
            "context_id": request["terrain_context"].get("field_id"),
            "samples": terrain_samples,
            "edit_policy": copy.deepcopy(dict(terrain_policy)),
        },
    }
    if prior is None:
        building["request_context"] = {
            "building_type": request["building_type"],
            "lot_polygon": copy.deepcopy(request["lot_polygon"]),
            "setback_polygon": copy.deepcopy(request["setback_polygon"]),
            "primary_access_surface": copy.deepcopy(request["primary_access_surface"]),
            "requested_size": request["requested_size"],
            "use_tags": copy.deepcopy(request["use_tags"]),
            "runtime_caps": copy.deepcopy(request["runtime_caps"]),
        }
    validate_generated_building(building)
    return canonicalize(building)


def compose_base(request: Mapping[str, Any], compiled_kit: Mapping[str, Any]) -> dict[str, Any]:
    """Compose one Phase 6 base request into a complete generated stamp."""
    validate_building_request(request)
    indexes = _compiled_indexes(compiled_kit)
    members, metadata = _base_members(request, indexes)
    steps = {metadata["primary_door_id"]: float(metadata["access_profile"]["grade_support"]["door_step_heights_gu"][0])}
    return _finalize(
        request,
        members,
        metadata,
        indexes,
        door_steps=steps,
        revision_id=f"generated:{request['request_id']}:r{int(request['revision'])}",
    )


def compose_extension(request: Mapping[str, Any], compiled_kit: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one named secondary-access revision or return unchanged output."""
    validate_building_extension_request(request)
    indexes = _compiled_indexes(compiled_kit)
    previous = copy.deepcopy(dict(request["previous_generated_building"]))
    rejection_base = {"status": "rejected", "request_id": request["request_id"], "building": previous, "rejections": []}
    allowed = {str(value) for value in request["allowed_extension_kinds"]}
    if "secondary_access" not in allowed:
        rejection_base["rejections"].append({"code": "secondary_access_not_allowed"})
        return canonicalize(rejection_base)
    if int(request["revision"]) > int(request["palette_caps"].get("max_revisions", 0)):
        rejection_base["rejections"].append({"code": "revision_cap_exceeded"})
        return canonicalize(rejection_base)
    try:
        if not request["new_access_surfaces"]:
            raise ComposerError("secondary_access_surface_missing")
        members = copy.deepcopy(previous["members"])
        members, metadata = _append_access(members, request, request["new_access_surfaces"][0], indexes, len([m for m in members if m.get("is_door")]))
        previous_doors = [member for member in previous["members"] if bool(member.get("is_door"))]
        steps = {str(member["source_id"]): float(value) for member, value in zip(previous_doors, previous["terrain_envelope"]["door_step_heights_gu"])}
        steps[metadata["door_id"]] = float(metadata["access_profile"]["grade_support"]["door_step_heights_gu"][0])
        base_metadata = {
            "mode": "secondary_access_revision",
            "access_profile_id": previous.get("composition", {}).get("selected_access_profile_id", metadata["access_profile_id"]),
            "primary_door_id": previous["source"]["seed_door"],
            "selected_shell_profile_id": previous.get("composition", {}).get("selected_shell_profile_id"),
            "selected_template_profile_id": previous.get("composition", {}).get("selected_template_profile_id"),
        }
        working_request = copy.deepcopy(dict(previous.get("request_context", {})))
        working_request.update({
            "request_id": request["request_id"],
            "revision": request["revision"],
            "palette_id": request["palette_id"],
            "terrain_context": request["terrain_context"],
            "terrain_edit_policy": request["terrain_edit_policy"],
            "occupied_reserved_polygons": request["occupied_reserved_polygons"],
            "free_space_polygons": request["free_space_polygons"],
            "body_min_z_gu": request.get("body_min_z_gu", 0.0),
        })
        revised = _finalize(
            working_request,
            members,
            base_metadata,
            indexes,
            door_steps=steps,
            revision_id=f"{previous['stamp_id']}:r{int(request['revision'])}",
            prior=previous,
        )
        revised["composition"]["secondary_access"] = metadata
        return canonicalize({"status": "accepted", "request_id": request["request_id"], "building": revised, "rejections": []})
    except (ComposerError, ValueError, KeyError) as exc:
        rejection_base["rejections"].append({"code": str(exc).split(":", 1)[0], "message": str(exc)})
        return canonicalize(rejection_base)


__all__ = ["ComposerError", "compose_base", "compose_extension"]
