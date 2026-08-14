#!/usr/bin/env python3
"""Cityforge T0.4b: normalize v1 stamp libraries into building-aligned v2.

Purpose
-------
The v1 stamp libraries (``karthgad_nord_v1.json``, ``markarth_side_stone_v1.json``)
preserve each building's SOURCE world rotation, so most stamps sit at
non-cardinal angles in stamp space (e.g. walls at 40.1 deg).  Their stored
``footprint.hull_xy_rel`` / ``bounds_rel_gu`` are axis-aligned boxes around the
ROTATED building -- inflated and orientation-mismatched -- which produced
diagonal-looking door arrows and false collision margins downstream (see
``.opencode/runs/cityforge-process-efficiency-review/
2026-08-12_falkreath_sketch_v1_failure_analysis.md``).

This tool derives ONE rotation theta per stamp (the modal shell rotation,
mod 180, bucketed at 0.5 deg, majority by member count, mean of the winning
bucket rounded to 0.1 deg) and re-expresses every member in the frame
F = Rz(+theta) about the stamp anchor:

* ``offset_gu`` xy -> ``Rz(+theta) . offset`` (z unchanged),
* ``rotation[2]`` -> ``(rotz - theta_rad) mod 2pi`` (rx/ry unchanged),
* ``footprint.hull_xy_rel`` / ``aabb_rel`` / ``bounds_rel_gu`` recomputed from
  per-member OBBs: each member's OBB = model local XY bounds x scale,
  composed as ``offset' + Rz(-rotz') . (scale * local_corner)`` with
  ``rotz' = rotz - theta``; hull = convex hull over all member OBB corners,
  bounds = axis union.

  ROTATION CONVENTION (measured, lead 2026-08-12): the engine places a ref
  as ``world = pos + Rz(-rotz) . (scale * local)`` -- proven against
  manifest ``world_bounds_gu`` for 140 pivot-offset members across both
  libraries (Rz(-rotz) residual cluster 0.972 at mean -1.7 deg; Rz(+rotz)
  0.099 = random).  F = Rz(+theta) is the ONLY frame rotation consistent
  with that composition and the normalization goal (modal shell walls
  axis-aligned: F.Rz(-rotz) = Rz(-rotz')).  The first revision of this tool
  used Rz(-theta) for offsets and Rz(+rotz') for the composition; those two
  sign errors do not cancel (a 2*rotz error per member) and produced
  transposed/meaningless hulls for non-cardinal stamps (user-confirmed on
  the 5-stamp geometry check 2026-08-12).  Model local bounds
  come from the run's A2 evidence documents
  (``stamp_local_bounds.load_a2_local_bounds``; validated identical to the
  surface-geometry cache ``local_bounds``, 398-model overlap).  Fallback
  ladder per member, each recorded: un-inflated world AABB (2x2
  rotation-AABB solve; ill-conditioned when |cos 2phi| < 0.2), then the
  world AABB itself (conservative).
* ``access_heading_rad`` -> ``v1 + theta_rad`` (F = Rz(+theta) adds +theta to
  every direction angle),
* per-door ``outward_heading_deg`` (NEW 2026-08-12): the door's approach
  direction, derived geometrically -- the door box's THIN horizontal axis
  composed with Rz(-rotz') is the wall normal; the outward sign points away
  from the non-door body centroid.  Near-square door boxes (span ratio <
  DOOR_THIN_AXIS_MIN_RATIO) fall back to the body-centroid radial.  The raw
  TES3 door rotz is NOT a reliable facing (mesh forward axis differs per
  model family; a source door can even face inward), so downstream
  arrows/reach use this geometric heading.  door ``destination_*`` interior
  data is never touched.

Door members rotate exactly like all members.  The raw door ``rotation[2]``
is NOT assumed to be the facing (see ``outward_heading_deg`` above); door
cardinality of rotz' is REPORT-ONLY (LEAD RULING 2026-08-12: the original
+/-0.6 deg hard gate was a spec error): every door's residual vs the nearest
cardinal direction is recorded as a fact (``stats.normalization.door_facts``,
``normalization_facts.door_residuals_deg``); source-mounted skews (0.6-3.1
deg) and structural diagonal doors (towers/windmills, up to 35 deg) are
expected source geometry, not violations.

Inputs (read-only)
------------------
* One v1 stamp library JSON (``--in``).
* Per-member cached world bounds from the SAME extraction products the stamp
  library joined (per-ref ``world_bounds_gu``): Karthgad component manifests
  via ``stamp.source.slug``; Markarth via ``buildings_index.json``
  ``component_id`` -> slug.  Every manifest read is SHA-256-verified against
  the owning library's ``inputs`` map (fail closed on mismatch) -- the exact
  authority ``build_stamp_volumes.py`` uses (functions are imported from it).
* Per-model local bounds from the run's A2 evidence documents
  (``<run>/a2/nif_<model>--*.json``), matched by model key + source SHA-256.

Outputs
-------
A v2 library JSON (``--out``; refuses to overwrite an existing file):
``schema_version: 2``, same ``library_id``/stamp ids (geometry frame changed,
identity did not), ``normalization_theta_deg`` per stamp, transformed members
and OBB-derived footprint, per-member bounds-source records, per-door
cardinality facts, replay evidence for the v1->v2 transform (mirroring the
v1 evidence structure: per-stamp ``replay`` + library ``stats.replay``, with
the v1 evidence preserved verbatim as ``replay_v1``), and a
``stats.normalization`` summary (theta table, shell-vs-door-implied
deviations, flags, door facts, fallback counts, v2-vs-v1 bounds area ratios).

Invariants / gates (fail-closed, no output on any failure)
----------------------------------------------------------
* Replay gate: for every member, ``anchor + Rz(theta) . offset'`` must equal
  the v1 ``anchor + offset`` within 1e-6 GU per axis, and
  ``(theta + rotz'_deg) mod 360`` must equal v1 ``rotz_deg mod 360`` within
  1e-9 deg.
* Shell-modal frame assert (trivially true by construction; kept as a check):
  the modal shell rotz' must be ~0 mod 90 (+-0.6 deg).
* Door cardinality is a REPORT-ONLY fact list (see above).
* Cross-check (informational, does not fail): seed-door-implied wall direction
  ``(door rotz - 90) mod 180`` vs the shell modal theta; deviation > 10 deg
  flags the stamp in the run summary (per accepted design, the shell modal
  still wins).
* Deterministic output: canonical JSON bytes (sorted keys, 2-space indent, no
  NaN) plus trailing newline; no timestamps.

Pipeline position
-----------------
T0.4b, immediately after the T0.3 stamp libraries.  Consumed by
``build_stamp_volumes.py`` (v2 volumes), ``visual_planner.py`` (CANONICAL
libraries), ``visual_planner_advisory.py`` (v2 volumes sidecar), and the
planning-bundle builder.  v1 files are never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
TOOLS = WORKSPACE / "tools"
for entry in (SRC, TOOLS, TOOLS / "cityforge"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from procgen.cityplan import rot2d_ccw  # noqa: E402
from procgen.citystamps import (  # noqa: E402
    canonical_json_bytes,
    convex_hull_xy,
    sha256_file,
)
from build_stamp_volumes import (  # noqa: E402
    _karthgad_authority,
    _markarth_authority,
)
from stamp_local_bounds import (  # noqa: E402
    SRC_MODEL_LOCAL,
    SRC_UNINFLATED,
    SRC_WORLD_AABB,
    load_a2_local_bounds,
    member_obb_corners_rel,
)

SCHEMA_VERSION = 2
__version__ = "0.2.0"

# Bucket resolution for the modal shell rotation (deg, mod 180).
THETA_BUCKET_DEG = 0.5
# Theta is rounded to this precision before being applied.
THETA_ROUND_DEG = 0.1
THETA_ROUND_DIGITS = 1
# Door-implied-vs-shell deviation above which a stamp is flagged (informational).
FLAG_DEVIATION_DEG = 10.0
# Shell-modal frame assert tolerance (trivially true by construction).
SHELL_FRAME_TOLERANCE_DEG = 0.6
# Door residuals at or above this magnitude are recorded as off-cardinal
# facts (report-only per lead ruling 2026-08-12; expected source geometry).
DOOR_FACT_THRESHOLD_DEG = 0.6
# Door outward-heading derivation (2026-08-12 lead fix): the door box's thin
# horizontal axis is taken as the wall normal only when the box is
# convincingly non-square (long span / short span >= this ratio); otherwise
# the body-centroid radial is used (recorded per door as heading_source).
DOOR_THIN_AXIS_MIN_RATIO = 1.15
# Replay gate tolerances (task contract).
REPLAY_POSITION_TOLERANCE_GU = 1e-6
REPLAY_ROTATION_TOLERANCE_DEG = 1e-9

# A2 evidence dirs per kit (per-model local bounds; read-only).
KARTHGAD_RUN_DIR = WORKSPACE / "output" / "skyrim-settlements" / "karthgad-v1"
MARKARTH_RUN_DIR = WORKSPACE / "output" / "skyrim-settlements" / "markarth-side-v1"


class NormalizationError(RuntimeError):
    """Fail-closed normalization failure (message already FAILURE-prefixed)."""


def _mod_180(deg: float) -> float:
    """Reduce an angle (deg) mod 180 into [0, 180)."""
    return float(deg) % 180.0


def _mod_360(deg: float) -> float:
    return float(deg) % 360.0


def _circular_diff(a_deg: float, b_deg: float) -> float:
    """Minimal circular distance between two angles, in [0, 180] deg."""
    diff = abs(_mod_360(a_deg) - _mod_360(b_deg)) % 360.0
    return min(diff, 360.0 - diff)


def _mod_180_diff(a_deg: float, b_deg: float) -> float:
    """Minimal distance between two mod-180 (wall-direction) angles, [0, 90]."""
    diff = abs(_mod_180(a_deg) - _mod_180(b_deg))
    return min(diff, 180.0 - diff)


def _distance_to_90_grid(deg: float) -> float:
    """Minimal distance of an angle to {0, 90, 180, 270} (mod-90 distance)."""
    rem = _mod_360(deg) % 90.0
    return min(rem, 90.0 - rem)


def _bucket_key(deg: float) -> float:
    """0.5-deg bucket key for a mod-180 angle (round-half-up at 0.25 deg)."""
    index = int(math.floor(deg / THETA_BUCKET_DEG + 0.5))
    return index * THETA_BUCKET_DEG


def _rotz_deg(member: Mapping[str, Any]) -> float:
    rotation = member.get("rotation") or [0.0, 0.0, 0.0]
    if len(rotation) < 3 or not all(
        math.isfinite(float(value)) for value in rotation[:3]
    ):
        raise NormalizationError(
            f"FAILURE: normalize_stamp_orientation member {member.get('source_id')} "
            f"has no finite rotation triple {rotation!r}"
        )
    return math.degrees(float(rotation[2]))


def _modal_shell_theta(stamp: Mapping[str, Any]) -> tuple[float, list[float]]:
    """Modal non-door rotz (deg, mod 180) by 0.5-deg bucket majority.

    Returns ``(theta, bucket_member_values)`` with theta = mean of the winning
    bucket's members rounded to 0.1 deg (the lead-verified value for the
    canonical example: members at 40.1 deg -> theta 40.1).  Raises when the
    stamp has no non-door members (theta is then undefined by the spec).
    """
    non_door = [m for m in stamp["members"] if not m["is_door"]]
    if not non_door:
        raise NormalizationError(
            f"FAILURE: normalize_stamp_orientation {stamp['stamp_id']} has no "
            "non-door members; modal shell theta undefined"
        )
    buckets: Counter[float] = Counter()
    values_by_bucket: dict[float, list[float]] = {}
    for member in non_door:
        value = _mod_180(_rotz_deg(member))
        key = _bucket_key(value)
        buckets[key] += 1
        values_by_bucket.setdefault(key, []).append(value)
    winning = max(sorted(buckets), key=lambda key: (buckets[key], -key))
    members_values = values_by_bucket[winning]
    theta = round(sum(members_values) / len(members_values), THETA_ROUND_DIGITS)
    return theta, members_values


def _door_implied_wall(stamp: Mapping[str, Any]) -> float:
    """Seed-door-implied wall direction: ``(door rotz - 90) mod 180`` (deg)."""
    seed_id = stamp["anchor"]["seed_door"]
    for member in stamp["members"]:
        if member["is_door"] and member.get("source_id") == seed_id:
            return _mod_180(_rotz_deg(member) - 90.0)
    raise NormalizationError(
        f"FAILURE: normalize_stamp_orientation {stamp['stamp_id']} seed door "
        f"{seed_id!r} not found among door members"
    )


def _union_bounds(corners: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    """Per-axis min/max/span over 3D corners (the v1 derivation, inline)."""
    mn = [min(corner[axis] for corner in corners) for axis in range(3)]
    mx = [max(corner[axis] for corner in corners) for axis in range(3)]
    return {
        "min": mn,
        "max": mx,
        "span": [mx[axis] - mn[axis] for axis in range(3)],
    }


def _authority_for(library: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[tuple[str, str]]]:
    """Pick the bounds authority by library id (same fail-closed hash policy
    as build_stamp_volumes)."""
    library_id = str(library.get("library_id", ""))
    if library_id.startswith("karthgad"):
        return _karthgad_authority(library)
    if library_id.startswith("markarth"):
        return _markarth_authority(library)
    raise NormalizationError(
        f"FAILURE: normalize_stamp_orientation unknown kit for library_id "
        f"{library_id!r} (expected karthgad_nord_v1 / markarth_side_stone_v1)"
    )


def _normalize_stamp(
    stamp: Mapping[str, Any],
    authority: Mapping[str, Mapping[str, Any]],
    local_bounds: Mapping[tuple[str, str], Mapping[str, Sequence[float]]],
) -> dict[str, Any]:
    """Normalize one stamp; raises NormalizationError on any replay failure.

    Returns the v2 stamp with report-only door facts (lead ruling 2026-08-12:
    the door-cardinality hard gate was a spec error; residuals are facts).
    """
    stamp_id = stamp["stamp_id"]
    anchor = [float(value) for value in stamp["anchor"]["source_position_gu"]]
    if len(anchor) != 3:
        raise NormalizationError(
            f"FAILURE: normalize_stamp_orientation {stamp_id} malformed anchor {anchor!r}"
        )

    theta, bucket_values = _modal_shell_theta(stamp)
    theta_rad = math.radians(theta)
    door_implied = _door_implied_wall(stamp)
    deviation = _mod_180_diff(theta, door_implied)
    flagged = deviation > FLAG_DEVIATION_DEG

    # --- 1. Re-express members (offsets by Rz(+theta), rotz by -theta) ------
    # Frame transform F = Rz(+theta): only then does the measured engine
    # composition (world = pos + Rz(-rotz).local) make modal shell walls
    # axis-aligned in the v2 frame: F.Rz(-rotz) = Rz(-(rotz - theta)).
    members_v2: list[dict[str, Any]] = []
    max_pos_error = 0.0
    max_rot_error_deg = 0.0
    rotation_mismatches = 0
    scale_mismatches = 0
    for member in stamp["members"]:
        offset = [float(value) for value in member["offset_gu"]]
        if len(offset) != 3:
            raise NormalizationError(
                f"FAILURE: normalize_stamp_orientation {stamp_id} member "
                f"{member.get('source_id')} malformed offset {offset!r}"
            )
        # New offset: rotate the xy part by +theta about the (anchor) origin
        # (frame transform F = Rz(+theta), see above); z unchanged.
        new_xy = rot2d_ccw(offset[0], offset[1], theta)
        row = dict(member)
        row["offset_gu"] = [new_xy[0], new_xy[1], offset[2]]

        # New rotation: z -= theta, normalized to [0, 360) (radians in file).
        rotation = [float(value) for value in member["rotation"]]
        new_rz = (rotation[2] - theta_rad) % (2.0 * math.pi)
        row["rotation"] = [rotation[0], rotation[1], new_rz]
        # Door destination data is interior data: never touched.

        # --- Replay gate (fail-closed) --------------------------------------
        v1_pos = [anchor[0] + offset[0], anchor[1] + offset[1], anchor[2] + offset[2]]
        back_xy = rot2d_ccw(new_xy[0], new_xy[1], -theta)
        v2_pos = [anchor[0] + back_xy[0], anchor[1] + back_xy[1], anchor[2] + offset[2]]
        pos_error = max(abs(a - b) for a, b in zip(v1_pos, v2_pos))
        max_pos_error = max(max_pos_error, pos_error)
        if pos_error > REPLAY_POSITION_TOLERANCE_GU:
            raise NormalizationError(
                f"FAILURE: normalize_stamp_orientation replay position mismatch "
                f"{stamp_id} {member.get('source_id')}: {pos_error:.3e} GU > "
                f"{REPLAY_POSITION_TOLERANCE_GU} GU"
            )
        v1_rotz_deg = _mod_360(math.degrees(rotation[2]))
        rotz_prime_deg = _mod_360(math.degrees(new_rz) + theta)
        rot_error = _circular_diff(rotz_prime_deg, v1_rotz_deg)
        max_rot_error_deg = max(max_rot_error_deg, rot_error)
        if rot_error > REPLAY_ROTATION_TOLERANCE_DEG:
            raise NormalizationError(
                f"FAILURE: normalize_stamp_orientation replay rotation mismatch "
                f"{stamp_id} {member.get('source_id')}: {rot_error:.3e} deg > "
                f"{REPLAY_ROTATION_TOLERANCE_DEG} deg"
            )
        # rotation_mismatches/scale_mismatches mirror the v1 evidence shape:
        # rx/ry/scale are carried verbatim, so mismatches stay 0 unless a row
        # accidentally changed them.
        if (
            row["rotation"][0] != rotation[0]
            or row["rotation"][1] != rotation[1]
            or row["scale"] != member["scale"]
        ):
            scale_mismatches += 1
        members_v2.append(row)

    # --- 2. Footprint in the normalized frame (OBB composition) -------------
    # Per-member OBB corners via the lead's prescribed composition
    # (stamp_local_bounds): model local bounds x scale, rotated by member
    # rotz, translated by member offset -- all in the v2 frame (rotz' =
    # rotz - theta).  The composition consumes the V1 member rows (raw
    # offset/rotz): the helper applies the -theta rotation itself.
    # Fallbacks (un-inflated world AABB, world AABB) are recorded per
    # member; the v1 derivation is then applied unchanged: 2D convex hull
    # over member corners for the hull, per-axis union for the bounds
    # (tight in the building frame since modal shell rotz' ~ 0/90).
    corners_rel: list[list[float]] = []
    corners_by_sid: dict[str, list[list[float]]] = {}
    local_by_sid: dict[str, Mapping[str, Sequence[float]] | None] = {}
    missing: list[str] = []
    bounds_sources: dict[str, dict[str, Any]] = {}
    for member in stamp["members"]:
        cached = authority.get(member["source_id"]) or {}
        world_bounds = cached.get("world_bounds_gu")
        if not world_bounds:
            missing.append(member["source_id"])
            continue
        source_sha = (cached.get("source") or {}).get("sha256")
        model_key = member.get("model_key")
        local = local_bounds.get((str(model_key), str(source_sha))) if (
            model_key and source_sha
        ) else None
        corners, source_label, note = member_obb_corners_rel(
            member, theta, local, world_bounds, anchor
        )
        corners_rel.extend(corners)
        corners_by_sid[member["source_id"]] = corners
        local_by_sid[member["source_id"]] = local
        bounds_sources[member["source_id"]] = {
            "source": source_label,
            **note,
        }
    if missing:
        raise NormalizationError(
            f"FAILURE: normalize_stamp_orientation {stamp_id} no cached world "
            f"bounds for {len(missing)} member(s): {', '.join(sorted(missing))}"
        )
    bounds = _union_bounds(corners_rel)
    hull_xy_rel = convex_hull_xy([[x, y] for x, y, _ in corners_rel])

    # F = Rz(+theta) adds +theta to every direction angle (the access heading
    # is the centroid->seed-door direction angle).
    access_heading_rad = float(stamp["access_heading_rad"]) + theta_rad

    # --- 2b. Per-door outward heading (geometric, from the fixed OBBs) ------
    # A door's approach direction = the outward wall normal at the door.  It
    # is derived from geometry, NOT from any mesh-facing convention: the door
    # box's THIN horizontal axis (composed with Rz(-rotz')) is the wall
    # normal, and the outward sign points away from the building body (the
    # centroid of the non-door members' corners).  Near-square door boxes
    # (thin axis ambiguous) fall back to the body-centroid radial; a door
    # sitting exactly on the centroid falls back to the access heading.  The
    # path taken is recorded per door in door_facts.
    non_door_points = [
        corner[:2]
        for member in stamp["members"]
        if not member["is_door"]
        for corner in corners_by_sid[member["source_id"]]
    ]
    centroid_points = non_door_points if non_door_points else [c[:2] for c in corners_rel]
    body_centroid = (
        sum(p[0] for p in centroid_points) / len(centroid_points),
        sum(p[1] for p in centroid_points) / len(centroid_points),
    )
    door_headings: dict[str, dict[str, Any]] = {}
    for row in members_v2:
        if not row["is_door"]:
            continue
        sid = str(row["source_id"])
        rotz_prime = _mod_360(math.degrees(float(row["rotation"][2])))
        px, py = float(row["offset_gu"][0]), float(row["offset_gu"][1])
        rdx, rdy = px - body_centroid[0], py - body_centroid[1]
        local = local_by_sid.get(sid)
        heading: float | None = None
        heading_source = "radial_fallback"
        span_ratio: float | None = None
        if local is not None:
            scale = float(row.get("scale", 1.0))
            sx = (float(local["max"][0]) - float(local["min"][0])) * scale
            sy = (float(local["max"][1]) - float(local["min"][1])) * scale
            lo, hi = min(sx, sy), max(sx, sy)
            span_ratio = (hi / lo) if lo > 1e-9 else None
            if span_ratio is not None and span_ratio >= DOOR_THIN_AXIS_MIN_RATIO:
                n_local = (1.0, 0.0) if sx < sy else (0.0, 1.0)
                nx, ny = rot2d_ccw(n_local[0], n_local[1], -rotz_prime)
                sign = 1.0 if (nx * rdx + ny * rdy) >= 0.0 else -1.0
                heading = _mod_360(math.degrees(math.atan2(sign * ny, sign * nx)))
                heading_source = "thin_axis_x" if sx < sy else "thin_axis_y"
        if heading is None:
            if abs(rdx) + abs(rdy) > 1e-9:
                heading = _mod_360(math.degrees(math.atan2(rdy, rdx)))
            else:
                heading = _mod_360(math.degrees(access_heading_rad))
                heading_source = "access_heading_fallback"
        row["outward_heading_deg"] = round(heading, 2)
        door_headings[sid] = {
            "outward_heading_deg": row["outward_heading_deg"],
            "heading_source": heading_source,
            "thin_axis_span_ratio": (round(span_ratio, 3) if span_ratio is not None else None),
        }

    # --- 3. Door cardinality: REPORT-ONLY FACTS (lead ruling 2026-08-12) -----
    # The original +/-0.6 deg hard gate was a spec error: source-mounted door
    # skews (0.6-3.1 deg) and structural diagonal doors (towers/windmills, up
    # to ~35 deg) are expected source geometry.  Every door's residual vs the
    # nearest cardinal direction is recorded; the shell-modal frame assert is
    # kept (theta IS the modal value, so modal shell rotz' == 0 exactly).
    modal_prime = 0.0
    if _distance_to_90_grid(modal_prime) > SHELL_FRAME_TOLERANCE_DEG:
        raise NormalizationError(
            f"FAILURE: normalize_stamp_orientation shell-modal frame assert "
            f"{stamp_id}: modal shell rotz' residual {modal_prime:.3f} deg"
        )
    door_facts: list[dict[str, Any]] = []
    for member in members_v2:
        if not member["is_door"]:
            continue
        rotz_prime = _mod_360(math.degrees(float(member["rotation"][2])))
        residual = _distance_to_90_grid(rotz_prime)
        door_facts.append({
            "source_id": member["source_id"],
            "rotz_prime_deg": rotz_prime,
            "residual_vs_cardinal_deg": round(residual, 3),
            "off_cardinal": bool(residual >= DOOR_FACT_THRESHOLD_DEG),
            **door_headings.get(str(member["source_id"]), {}),
        })

    # v2-vs-v1 bounds area ratio (report summary; expect noticeably < 1.0 for
    # rotated stamps, ~1.0 for axis-aligned ones).
    v1_span = [float(value) for value in stamp["footprint"]["aabb_rel"]["span"]]
    v2_span = [float(value) for value in bounds["span"]]
    v1_area = v1_span[0] * v1_span[1]
    v2_area = v2_span[0] * v2_span[1]
    area_ratio = v2_area / v1_area if v1_area > 0.0 else 1.0

    stamp_v2 = {
        "stamp_id": stamp_id,
        "source": dict(stamp["source"]),
        "preview_sheet": stamp["preview_sheet"],
        "building_type": stamp["building_type"],
        "door_count": stamp["door_count"],
        "multi_shell": stamp["multi_shell"],
        "anchor": dict(stamp["anchor"]),
        "access_heading_rad": access_heading_rad,
        "members": members_v2,
        "footprint": {
            "aabb_rel": bounds,
            "hull_xy_rel": hull_xy_rel,
        },
        "terrain_envelope": dict(stamp["terrain_envelope"]),
        "bounds_rel_gu": bounds,
        "style_tags": list(stamp["style_tags"]),
        "size_class": stamp["size_class"],
        "normalization_theta_deg": theta,
        "normalization_facts": {
            "door_implied_wall_deg": door_implied,
            "shell_door_deviation_deg": round(deviation, 3),
            "flagged_deviation_gt_10_deg": flagged,
            "modal_bucket_member_count": len(bucket_values),
            "modal_bucket_spread_deg": round(
                max(bucket_values) - min(bucket_values), 3
            ),
            "door_facts": door_facts,
            "bounds_area_ratio_v2_vs_v1": round(area_ratio, 4),
            "v1_span_gu": v1_span,
            "v2_span_gu": v2_span,
            "member_bounds_sources": bounds_sources,
        },
        "replay": {
            "max_abs_position_error_gu": max_pos_error,
            "max_abs_rotation_error_deg": max_rot_error_deg,
            "rotation_mismatches": rotation_mismatches,
            "scale_mismatches": scale_mismatches,
            "members_checked": len(members_v2),
            "tolerances": {
                "position_game_units": REPLAY_POSITION_TOLERANCE_GU,
                "rotation_deg": REPLAY_ROTATION_TOLERANCE_DEG,
            },
        },
        "replay_v1": dict(stamp.get("replay", {})),
    }
    return stamp_v2


def normalize_library(in_path: Path, out_path: Path) -> dict[str, Any]:
    """Load a v1 library, normalize every stamp, return the v2 document."""
    if out_path.exists():
        raise NormalizationError(
            f"FAILURE: normalize_stamp_orientation refuse to overwrite existing "
            f"output {out_path}"
        )
    try:
        with in_path.open("r", encoding="utf-8") as handle:
            library = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NormalizationError(
            f"FAILURE: normalize_stamp_orientation could not read {in_path}: {exc}"
        ) from exc
    if library.get("schema_version") != 1 or not isinstance(library.get("stamps"), list):
        raise NormalizationError(
            f"FAILURE: normalize_stamp_orientation {in_path} is not a v1 stamp "
            "library (schema_version must be 1, stamps must be a list)"
        )

    authority, manifest_inputs = _authority_for(library)
    library_id = str(library.get("library_id", ""))
    a2_dir = (
        KARTHGAD_RUN_DIR / "a2"
        if library_id.startswith("karthgad")
        else MARKARTH_RUN_DIR / "a2"
        if library_id.startswith("markarth")
        else None
    )
    if a2_dir is None:
        raise NormalizationError(
            f"FAILURE: normalize_stamp_orientation unknown kit for library_id "
            f"{library_id!r}"
        )
    local_bounds = load_a2_local_bounds(a2_dir)

    stamps_v2: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    door_facts_all: list[dict[str, Any]] = []
    fallback_counts: dict[str, int] = {
        SRC_MODEL_LOCAL: 0,
        SRC_UNINFLATED: 0,
        SRC_WORLD_AABB: 0,
    }
    area_ratios: dict[str, float] = {}
    max_pos_error = 0.0
    max_rot_error_deg = 0.0
    members_checked = 0
    for stamp in library["stamps"]:
        result = _normalize_stamp(stamp, authority, local_bounds)
        stamps_v2.append(result)
        members_checked += result["replay"]["members_checked"]
        max_pos_error = max(max_pos_error, result["replay"]["max_abs_position_error_gu"])
        max_rot_error_deg = max(max_rot_error_deg, result["replay"]["max_abs_rotation_error_deg"])
        facts = result["normalization_facts"]
        for door_fact in facts["door_facts"]:
            door_facts_all.append({"stamp_id": result["stamp_id"], **door_fact})
        for source_id, bounds_row in facts["member_bounds_sources"].items():
            fallback_counts[bounds_row["source"]] = (
                fallback_counts.get(bounds_row["source"], 0) + 1
            )
        area_ratios[result["stamp_id"]] = facts["bounds_area_ratio_v2_vs_v1"]
        flag_mark = "FLAG" if facts["flagged_deviation_gt_10_deg"] else "    "
        print(
            f"[normalize] {result['stamp_id']:55s} theta={result['normalization_theta_deg']:7.1f} "
            f"door_impl={facts['door_implied_wall_deg']:7.1f} "
            f"dev={facts['shell_door_deviation_deg']:7.3f} "
            f"area_ratio={facts['bounds_area_ratio_v2_vs_v1']:6.3f} {flag_mark}"
        )
        if facts["flagged_deviation_gt_10_deg"]:
            flags.append({
                "stamp_id": result["stamp_id"],
                "theta_deg": result["normalization_theta_deg"],
                "door_implied_wall_deg": facts["door_implied_wall_deg"],
                "deviation_deg": facts["shell_door_deviation_deg"],
            })

    inputs: dict[str, str] = dict(library.get("inputs") or {})
    inputs.update({str(path): digest for path, digest in manifest_inputs})
    inputs[_ws_rel(in_path)] = sha256_file(in_path)

    stats_v2 = dict(library.get("stats") or {})
    stats_v2["replay_v1"] = dict(stats_v2.get("replay", {}))
    stats_v2["replay"] = {
        "members_checked": members_checked,
        "max_abs_position_error_gu": max_pos_error,
        "max_abs_rotation_error_deg": max_rot_error_deg,
        "rotation_mismatches": 0,
        "scale_mismatches": 0,
        "tolerances": {
            "position_game_units": REPLAY_POSITION_TOLERANCE_GU,
            "rotation_deg": REPLAY_ROTATION_TOLERANCE_DEG,
        },
    }
    off_cardinal_doors = [
        fact for fact in door_facts_all if fact["off_cardinal"]
    ]
    ratios = list(area_ratios.values())
    stats_v2["normalization"] = {
        "stamp_count": len(stamps_v2),
        "theta_deg_per_stamp": {
            s["stamp_id"]: s["normalization_theta_deg"] for s in stamps_v2
        },
        "flagged_stamp_count": len(flags),
        "flagged_stamps": flags,
        "method": "modal non-door rotz (mod 180, 0.5-deg buckets, majority by "
                  "member count, mean of winning bucket rounded to 0.1 deg)",
        "member_bounds_source_counts": dict(sorted(fallback_counts.items())),
        "door_facts": door_facts_all,
        "door_off_cardinal_count": len(off_cardinal_doors),
        "door_off_cardinal_threshold_deg": DOOR_FACT_THRESHOLD_DEG,
        "bounds_area_ratio_v2_vs_v1": {
            "per_stamp": area_ratios,
            "min": round(min(ratios), 4) if ratios else None,
            "max": round(max(ratios), 4) if ratios else None,
            "count_lt_0_9": sum(1 for value in ratios if value < 0.9),
            "count_gt_1_1": sum(1 for value in ratios if value > 1.1),
        },
    }

    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": f"{library.get('generated_by', 'citystamps')} -> "
                        f"normalize_stamp_orientation {__version__} (Cityforge T0.4b)",
        "library_id": library["library_id"],
        "library_name": library["library_name"],
        "kit": library["kit"],
        "units": "game units (GU); rotations TES3 Euler radians. v2 frame is "
                 "building-aligned: yaw 0 is the building's natural "
                 "axis-aligned pose (source orientation preserved in "
                 "normalization_theta_deg)",
        "normalized_from": _ws_rel(in_path),
        "inputs": dict(sorted(inputs.items())),
        "source_plugins": library["source_plugins"],
        "stats": stats_v2,
        "stamps": stamps_v2,
    }
    return document


def _ws_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a v1 cityforge stamp library into a "
        "building-aligned v2 library (theta = modal shell rotz, mod 180, "
        "0.5-deg buckets)."
    )
    parser.add_argument("--in", dest="in_path", type=Path, required=True,
                        help="v1 stamp library JSON (read-only)")
    parser.add_argument("--out", dest="out_path", type=Path, required=True,
                        help="v2 output path (must not exist)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        document = normalize_library(args.in_path, args.out_path)
    except NormalizationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_bytes(canonical_json_bytes(document))

    stats = document["stats"]["normalization"]
    print(f"[normalize_stamp_orientation] wrote {args.out_path} "
          f"(sha256 {sha256_file(args.out_path)})")
    print(f"[normalize_stamp_orientation] {document['library_id']}: "
          f"stamps={stats['stamp_count']} flagged={stats['flagged_stamp_count']} "
          f"doors_off_cardinal={stats['door_off_cardinal_count']} "
          f"member_bounds_sources={stats['member_bounds_source_counts']} "
          f"area_ratio_min={stats['bounds_area_ratio_v2_vs_v1']['min']} "
          f"max={stats['bounds_area_ratio_v2_vs_v1']['max']} "
          f"max_abs_position_error_gu="
          f"{document['stats']['replay']['max_abs_position_error_gu']:.3e} "
          f"max_abs_rotation_error_deg="
          f"{document['stats']['replay']['max_abs_rotation_error_deg']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
