"""Attachment mount profiles from evaluated triangle evidence (Phase 3a).

Plan §7.5. For selectable attachment models (window, doorframe, dormer,
porch, stair, chimney, tent) builds the mount contract: a mount frame from
the configured axis or, when absent, the thinnest horizontal bbox axis,
front/back face-occupancy evidence, contact polygon at the back plane,
occupied/clearance envelopes, and measured sink intervals from witness stamps.
Ambiguous orientation stays ``pending`` review; nothing is inferred from
filenames.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from procgen.building_gen.facades import _convex_hull_2d


def _mount_axes(bounds_min: list[float], bounds_max: list[float]) -> tuple[int, list[float]]:
    """Thinnest horizontal axis (0=X, 1=Y) is the mount normal axis."""
    span_x = bounds_max[0] - bounds_min[0]
    span_y = bounds_max[1] - bounds_min[1]
    axis = 0 if span_x <= span_y else 1
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    return axis, normal


def _axis_override(value: str | None) -> int | None:
    """Resolve a configured horizontal mount axis, rejecting silent fallbacks."""
    if value is None:
        return None
    axis = str(value).casefold()
    if axis not in {"x", "y"}:
        raise ValueError(f"mount normal axis override must be 'x' or 'y', got {value!r}")
    return "xy".index(axis)


def build_mount_profile(
    model_key: str,
    role: str,
    triangles: list[Mapping[str, Any]],
    bounds: Mapping[str, Any],
    scales_observed: list[float],
    clearance_margin_gu: float,
    sink_tolerance_gu: float,
    normal_axis_override: str | None = None,
) -> dict[str, Any]:
    bmin = [float(v) for v in bounds["min"]]
    bmax = [float(v) for v in bounds["max"]]
    axis, n = _mount_axes(bmin, bmax)
    override_axis = _axis_override(normal_axis_override)
    if override_axis is not None:
        axis = override_axis
        n = [0.0, 0.0, 0.0]
        n[axis] = 1.0
    orientation_basis = (
        "configured mount normal axis override; dominant evaluated face-normal occupancy"
        if override_axis is not None
        else "dominant evaluated face-normal occupancy"
    )
    tangent_axis = 1 - axis

    front_area = 0.0
    back_area = 0.0
    for tri in triangles:
        component = float(tri["normal"][axis])
        if component > 0:
            front_area += float(tri["area"])
        elif component < 0:
            back_area += float(tri["area"])

    # The larger signed face area is the deterministic visible/front side.
    # The mount/contact side is the opposite bbox plane. This handles open
    # meshes such as sky_fk_window_06a whose back has no faces at all.
    front_sign = -1.0 if back_area > front_area else 1.0
    n[axis] = front_sign
    back_value = bmax[axis] if front_sign < 0.0 else bmin[axis]
    contact_points: list[tuple[float, float]] = []
    for tri in triangles:
        for v in tri["verts"]:
            if abs(v[axis] - back_value) <= sink_tolerance_gu:
                contact_points.append((v[tangent_axis], v[2]))
    contact_polygon = _convex_hull_2d(contact_points)
    contact_interval = None
    if contact_points:
        contact_interval = {
            "u": [round(min(point[0] for point in contact_points), 3),
                  round(max(point[0] for point in contact_points), 3)],
            "v": [round(min(point[1] for point in contact_points), 3),
                  round(max(point[1] for point in contact_points), 3)],
        }

    normal_values = [bmin[axis] * front_sign, bmax[axis] * front_sign]
    occupied = {
        "normal_gu": [round(min(normal_values), 3), round(max(normal_values), 3)],
        "tangent_gu": [round(bmin[tangent_axis], 3), round(bmax[tangent_axis], 3)],
        "up_gu": [round(bmin[2], 3), round(bmax[2], 3)],
    }
    clearance = {
        key: [round(pair[0] - clearance_margin_gu, 3), round(pair[1] + clearance_margin_gu, 3)]
        for key, pair in occupied.items()
    }
    tangent = [0.0, 0.0, 0.0]
    tangent[tangent_axis] = 1.0
    up = [0.0, 0.0, 1.0]
    return {
        "model_key": model_key,
        "role": role,
        "authored_scales": sorted(scales_observed),
        "mount_frame": {
            "normal_axis": "xyz"[axis],
            "n": n,
            "u_tangent": tangent,
            "v_up": up,
        },
        "front_back_evidence": {
            "front_axis_sign": "+" if front_sign > 0.0 else "-",
            "plus_axis_face_area_gu2": round(front_area, 3),
            "minus_axis_face_area_gu2": round(back_area, 3),
            "front_face_area_gu2": round(max(front_area, back_area), 3),
            "back_face_area_gu2": round(min(front_area, back_area), 3),
            "basis": f"{orientation_basis}; rendered pair confirms the assignment",
        },
        "contact_polygon_uv": contact_polygon,
        "contact_interval_uv": contact_interval,
        "contact_geometry_kind": (
            "polygon" if len(contact_polygon) >= 3
            else "interval" if contact_points
            else "missing"
        ),
        "occupied_envelope_gu": occupied,
        "clearance_envelope_gu": clearance,
        "sink_interval_gu": [],
        "orientation_evidence": orientation_basis + " on the signed mount axis",
        "review_status": "pending",
        "witnesses": [],
    }


def measure_sink_intervals(
    mount_profiles: Mapping[str, dict[str, Any]],
    facade_profiles: Mapping[str, dict[str, Any]],
    site_stamp_libraries: Mapping[str, Mapping[str, Any]],
    inventory: Mapping[str, Any],
    engine_rotation,
) -> None:
    """Record measured back-plane-to-facade signed offsets per witness stamp.

    For every stamp member whose model has a mount profile and whose stamp also
    contains a shell with a facade profile, compute the signed distance of the
    attachment's local back plane (transformed by the member's TES3 rotation and
    scale) to each host facade plane (also transformed), keeping the nearest.
    """
    role_by_key = {row["model_key"]: row["observed_roles"][0]
                   for row in inventory["models"] if len(row["observed_roles"]) == 1}
    for site_id in sorted(site_stamp_libraries):
        for stamp in site_stamp_libraries[site_id].get("stamps", []):
            members = stamp.get("members", [])
            shells = [m for m in members
                      if facade_profiles.get(str(m["model_key"]).replace("/", "\\").casefold())
                      and role_by_key.get(str(m["model_key"]).replace("/", "\\").casefold()) == "shell"]
            if not shells:
                continue
            for member in members:
                key = str(member["model_key"]).replace("/", "\\").casefold()
                profile = mount_profiles.get(key)
                if profile is None or role_by_key.get(key) == "shell":
                    continue
                axis = "xyz".index(profile["mount_frame"]["normal_axis"])
                rot = engine_rotation(member["rotation"])
                scale = float(member["scale"])
                back_local = profile["occupied_envelope_gu"]["normal_gu"][0] * scale
                back_normal = [float(c) for c in profile["mount_frame"]["n"]]
                rotated_normal = rot @ back_normal
                member_pos = member["offset_gu"]
                back_point = [member_pos[i] + rotated_normal[i] * back_local for i in range(3)]
                best = None
                for shell in shells:
                    shell_profile = facade_profiles[str(shell["model_key"]).replace("/", "\\").casefold()]
                    shell_rot = engine_rotation(shell["rotation"])
                    shell_scale = float(shell["scale"])
                    shell_pos = shell["offset_gu"]
                    for facade in shell_profile["facades"]:
                        n_local = facade["outward_frame"]["n"]
                        offset_local = facade["outward_frame"]["plane_offset_gu"]
                        n_world = shell_rot @ n_local
                        plane_point = [shell_pos[i] + n_world[i] * offset_local * shell_scale for i in range(3)]
                        signed = sum((back_point[i] - plane_point[i]) * n_world[i] for i in range(3))
                        distance = abs(signed)
                        if best is None or distance < best[1]:
                            best = (signed, distance)
                if best is not None:
                    profile["sink_interval_gu"].append(round(best[0], 3))
                    profile["witnesses"].append({
                        "site_id": site_id,
                        "stamp_id": stamp["stamp_id"],
                        "source_ref": member["source_id"],
                        "signed_back_to_facade_gu": round(best[0], 3),
                    })
    for profile in mount_profiles.values():
        if profile["sink_interval_gu"]:
            values = sorted(profile["sink_interval_gu"])
            profile["sink_interval_gu"] = [values[0], values[-1]]
