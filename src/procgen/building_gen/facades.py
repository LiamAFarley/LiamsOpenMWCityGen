"""Pure-Python facade reconstruction from evaluated triangle evidence.

Pipeline position: Phase 3a of the building rule kit (spec:
``2026-08-22_phase3a_implementation_spec.md``, plan §7.3). Consumes the raw
triangle export from ``tools/cityforge/blender_wall_mount_evidence.py`` and
reconstructs persistent facade profiles: vertical-interval wall segments with
stable IDs, outward frames, usable insets, and witness occupancy.

All thresholds arrive via the config mapping; nothing is hardcoded per model.
Semantic roles stay ``unresolved`` in Phase 3a.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _convex_hull_2d(points: list[tuple[float, float]]) -> list[list[float]]:
    unique = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(unique) <= 2:
        return [[round(x, 3), round(y, 3)] for x, y in unique]

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return [[round(x, 3), round(y, 3)] for x, y in lower[:-1] + upper[:-1]]


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return (hi - lo) / union if union > 0 else 0.0


def _span_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    shorter = min(a[1] - a[0], b[1] - b[0])
    if hi <= lo or shorter <= 0:
        return 0.0
    return (hi - lo) / shorter


def build_facade_profile(
    model_key: str,
    triangles: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    bounds_min_z: float,
    bounds_max_z: float,
    band_base_z: float | None = None,
) -> dict[str, Any]:
    """Reconstruct facade segments for one shell model from raw triangles.

    The wall band is anchored at ``band_base_z`` (the Phase 2 robust body
    bottom) so buried foundation skirts cannot capture the band; fractions
    span upward from that anchor to the evaluated z max.
    """
    max_tilt = math.radians(float(config["max_wall_tilt_deg"]))
    band_lo_f, band_hi_f = (float(v) for v in config["wall_band_lo_hi"])
    quantum = float(config["azimuth_quantum_deg"])
    offset_tol = float(config["plane_offset_tolerance_gu"])
    weld = float(config["vertex_weld_gu"])
    merge_iou = float(config["merge_interval_iou"])
    merge_gap = float(config.get("merge_vertical_gap_gu", 0.0))
    merge_horizontal_gap = float(config["merge_horizontal_gap_gu"])
    inset = float(config["facade_inset_gu"])
    min_area_fraction = float(config.get("min_facade_area_fraction", 0.0))
    max_start_offset = config.get("max_facade_start_offset_gu")

    base = bounds_min_z if band_base_z is None else float(band_base_z)
    height = bounds_max_z - base
    z_lo = base + band_lo_f * height
    z_hi = base + band_hi_f * height

    candidates: list[dict[str, Any]] = []
    for tri in triangles:
        n = tri["normal"]
        if abs(n[2]) > math.sin(max_tilt):
            continue
        c = tri["centroid"]
        if not (z_lo <= c[2] <= z_hi):
            continue
        az = math.degrees(math.atan2(n[1], n[0]))
        candidates.append({
            "verts": tri["verts"], "normal": n, "area": float(tri["area"]),
            "centroid": c, "azimuth": az,
            "z_min": min(v[2] for v in tri["verts"]),
            "z_max": max(v[2] for v in tri["verts"]),
        })

    # Model centroid over ALL input triangles, for outward orientation.
    model_c = [0.0, 0.0, 0.0]
    model_count = 0
    for t in triangles:
        for v in t["verts"]:
            model_c[0] += v[0]; model_c[1] += v[1]; model_c[2] += v[2]
            model_count += 1
    if model_count:
        model_c = [c / model_count for c in model_c]

    # Pass 1: azimuth quantum + plane-offset buckets.
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for tri in candidates:
        q = int(round(tri["azimuth"] / quantum))
        rad = math.radians(q * quantum)
        n_ref = (math.cos(rad), math.sin(rad), 0.0)
        offset = _dot(tri["centroid"], n_ref)
        buckets.setdefault((q, int(round(offset / offset_tol))), []).append(tri)

    # Pass 2: connected components inside each bucket via welded vertices.
    groups: list[list[dict[str, Any]]] = []
    for bucket in buckets.values():
        remaining = list(bucket)
        while remaining:
            seed = remaining.pop()
            component = [seed]
            welded = {tuple(round(c / weld) for c in v) for v in seed["verts"]}
            changed = True
            while changed:
                changed = False
                for tri in list(remaining):
                    keys = {tuple(round(c / weld) for c in v) for v in tri["verts"]}
                    if welded & keys:
                        component.append(tri)
                        remaining.remove(tri)
                        welded |= keys
                        changed = True
            groups.append(component)

    # Pass 3: merge nearby coplanar groups. A finite horizontal gap is required
    # even for exact-offset panels so disconnected wings cannot be convex-hulled
    # across empty intervals. Vertical merging uses gap adjacency, not IoU:
    # log walls appear as stacked thin near-vertical strips (one per log crown)
    # whose z intervals are disjoint but adjacent.
    def group_stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        area = sum(t["area"] for t in group)
        nx = sum(t["normal"][0] * t["area"] for t in group) / area
        ny = sum(t["normal"][1] * t["area"] for t in group) / area
        nz = sum(t["normal"][2] * t["area"] for t in group) / area
        nlen = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        n = (nx / nlen, ny / nlen, nz / nlen)
        offset = sum(_dot(t["centroid"], n) * t["area"] for t in group) / area
        z_interval = (min(t["z_min"] for t in group), max(t["z_max"] for t in group))
        ux, uy = -n[1], n[0]
        ulen = math.hypot(ux, uy) or 1.0
        u = (ux / ulen, uy / ulen, 0.0)
        us = [min(_dot(v, u) for v in t["verts"]) for t in group]
        ue = [max(_dot(v, u) for v in t["verts"]) for t in group]
        return {"n": n, "u": u, "offset": offset, "z": z_interval,
                "u_span": (min(us), max(ue)), "area": area, "tris": group}

    stats = [group_stats(g) for g in groups]

    # Drop inner wall skins: a group whose measured normal points toward the
    # model centroid AND has an opposite-normal group within one wall-skin gap
    # further outward. The gap bound confines this to actual wall skins and
    # preserves courtyard/notch facades that face the building mass across a
    # wide opening.
    skin_gap = float(config["max_wall_skin_gap_gu"])
    dropped_inward = 0
    outward_stats = []
    for g in stats:
        plane_g = (g["n"][0] * g["offset"], g["n"][1] * g["offset"], 0.0)
        toward = (model_c[0] - plane_g[0], model_c[1] - plane_g[1], 0.0)
        if _dot(g["n"], toward) <= 0:
            outward_stats.append(g)
            continue
        is_skin = False
        for h in stats:
            if h is g:
                continue
            if _dot(g["n"], h["n"]) > -math.cos(math.radians(quantum)):
                continue
            plane_h = (h["n"][0] * h["offset"], h["n"][1] * h["offset"], 0.0)
            outward = (plane_h[0] - plane_g[0], plane_h[1] - plane_g[1], 0.0)
            d_out = _dot(outward, (-g["n"][0], -g["n"][1], 0.0))
            if 0.0 < d_out <= skin_gap:
                is_skin = True
                break
        if is_skin:
            dropped_inward += 1
        else:
            outward_stats.append(g)
    stats = outward_stats

    # A tall triangle can cross the body band while its actual surface starts
    # on a roof/rake above the wall base. Phase 3b supplies this cutoff;
    # Phase 3a leaves it unset to preserve its accepted contract.
    dropped_above_body_start = 0
    if max_start_offset is not None:
        cutoff = base + float(max_start_offset)
        retained = [g for g in stats if g["z"][0] <= cutoff]
        dropped_above_body_start = len(stats) - len(retained)
        stats = retained

    merged = True
    while merged:
        merged = False
        for i in range(len(stats)):
            if merged:
                break
            for j in range(i + 1, len(stats)):
                a, b = stats[i], stats[j]
                na, nb = a["n"], b["n"]
                if _dot(na, nb) < math.cos(math.radians(quantum)):
                    continue
                if abs(a["offset"] - b["offset"]) > offset_tol:
                    continue
                gap = max(b["z"][0] - a["z"][1], a["z"][0] - b["z"][1], 0.0)
                if gap > merge_gap and _interval_iou(a["z"], b["z"]) < merge_iou:
                    continue
                horizontal_gap = max(
                    b["u_span"][0] - a["u_span"][1],
                    a["u_span"][0] - b["u_span"][1],
                    0.0,
                )
                if horizontal_gap > merge_horizontal_gap:
                    continue
                stats[i] = group_stats(a["tris"] + b["tris"])
                del stats[j]
                merged = True
                break

    # Frames, polygons, stable IDs. Small trim slivers (beams, ledges) are not
    # facades; they are counted and dropped.
    stats.sort(key=lambda s: (-s["area"], math.degrees(math.atan2(s["n"][1], s["n"][0]))))
    dropped_small = 0
    if stats and min_area_fraction > 0:
        largest = stats[0]["area"]
        kept = [s for s in stats if s["area"] >= min_area_fraction * largest]
        dropped_small = len(stats) - len(kept)
        stats = kept
    facades = []
    for index, s in enumerate(stats, start=1):
        n = s["n"]
        plane_point = (n[0] * s["offset"], n[1] * s["offset"], n[2] * s["offset"])
        if _dot(n, _sub(plane_point, model_c)) < 0:
            n = (-n[0], -n[1], -n[2])
            s["offset"] = -s["offset"]
        ulen = math.hypot(n[1], n[0]) or 1.0
        u = (-n[1] / ulen, n[0] / ulen, 0.0)
        v_axis = (n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0])
        points_uz = []
        for t in s["tris"]:
            for v in t["verts"]:
                points_uz.append((_dot(v, u), v[2]))
        polygon_uz = _convex_hull_2d(points_uz)
        us = [p[0] for p in points_uz]
        zs = [p[1] for p in points_uz]
        # group_stats measured its span before the final outward-normal flip.
        # Recompute it in the persisted frame so asymmetric panels cannot move
        # to the opposite side when a facade normal is corrected.
        s["u_span"] = (min(us), max(us))
        usable = {
            "u": [round(min(us) + inset, 3), round(max(us) - inset, 3)],
            "z": [round(min(zs) + inset, 3), round(max(zs) - inset, 3)],
        }
        facades.append({
            "facade_id": f"f{index:03d}",
            "semantic_role": "unresolved",
            "area_gu2": round(s["area"], 3),
            "triangle_count": len(s["tris"]),
            "outward_frame": {
                "n": [round(c, 6) for c in n],
                "u": [round(c, 6) for c in u],
                "v": [round(c, 6) for c in v_axis],
                "plane_offset_gu": round(s["offset"], 3),
            },
            "z_interval_gu": [round(s["z"][0], 3), round(s["z"][1], 3)],
            "u_span_gu": [round(s["u_span"][0], 3), round(s["u_span"][1], 3)],
            "polygon_uz": polygon_uz,
            "usable_region_uz": usable,
            "occupied_regions": [],
        })
    return {
        "model_key": model_key,
        "wall_band_gu": [round(z_lo, 3), round(z_hi, 3)],
        "candidate_triangles": len(candidates),
        "dropped_inward_groups": dropped_inward,
        "dropped_small_groups": dropped_small,
        "dropped_above_body_start_groups": dropped_above_body_start,
        "facade_count": len(facades),
        "facades": facades,
    }


def record_witness_occupancy(
    facade_profiles: Mapping[str, dict[str, Any]],
    site_stamp_libraries: Mapping[str, Mapping[str, Any]],
    inventory: Mapping[str, Any],
    engine_rotation,
    max_offset_gu: float = 64.0,
    bounds_tolerance_gu: float = 0.0,
) -> None:
    """Project observed attachment positions into host facade frames.

    For every stamp containing a profiled shell, each non-shell member's
    position is transformed into the shell's local frame and projected onto
    the nearest finite facade's (u, z) coordinates, recording an occupied point
    per facade only when the projection lies within that facade's measured
    extent. Evidence only; no role or opening semantics are inferred.
    """
    import numpy as np

    role_by_key = {row["model_key"]: row["observed_roles"][0]
                   for row in inventory["models"] if len(row["observed_roles"]) == 1}
    for site_id in sorted(site_stamp_libraries):
        for stamp in site_stamp_libraries[site_id].get("stamps", []):
            members = stamp.get("members", [])
            shells = [m for m in members
                      if str(m["model_key"]).replace("/", "\\").casefold() in facade_profiles
                      and role_by_key.get(str(m["model_key"]).replace("/", "\\").casefold()) == "shell"]
            for shell in shells:
                shell_key = str(shell["model_key"]).replace("/", "\\").casefold()
                shell_profile = facade_profiles[shell_key]
                shell_rot = engine_rotation(shell["rotation"])
                inv_rot = np.asarray(shell_rot, dtype=np.float64).T
                shell_pos = np.asarray(shell["offset_gu"], dtype=np.float64)
                shell_scale = float(shell["scale"])
                for member in members:
                    if member is shell:
                        continue
                    member_key = str(member["model_key"]).replace("/", "\\").casefold()
                    if role_by_key.get(member_key) == "shell":
                        continue
                    local = (inv_rot @ (np.asarray(member["offset_gu"], dtype=np.float64) - shell_pos)) / shell_scale
                    best = None
                    for facade in shell_profile["facades"]:
                        frame = facade["outward_frame"]
                        n = np.asarray(frame["n"], dtype=np.float64)
                        u = np.asarray(frame["u"], dtype=np.float64)
                        offset = float(frame["plane_offset_gu"])
                        signed = float(np.dot(local - n * offset, n))
                        u_coord = float(np.dot(local - n * offset, u))
                        distance = abs(signed)
                        u_min, u_max = (float(v) for v in facade["u_span_gu"])
                        z_min, z_max = (float(v) for v in facade["z_interval_gu"])
                        if not (
                            u_min - bounds_tolerance_gu <= u_coord <= u_max + bounds_tolerance_gu
                            and z_min - bounds_tolerance_gu <= float(local[2]) <= z_max + bounds_tolerance_gu
                        ):
                            continue
                        if best is None or distance < best[1]:
                            best = (facade, distance, u_coord, float(local[2]), member_key, member["source_id"])
                    if best is not None and best[1] <= max_offset_gu:
                        facade, distance, u_coord, z_coord, member_key, ref = best
                        facade["occupied_regions"].append({
                            "u_gu": round(u_coord, 3),
                            "z_gu": round(z_coord, 3),
                            "signed_offset_gu": round(distance, 3),
                            "model_key": member_key,
                            "source_ref": ref,
                            "site_id": site_id,
                            "stamp_id": stamp["stamp_id"],
                        })
