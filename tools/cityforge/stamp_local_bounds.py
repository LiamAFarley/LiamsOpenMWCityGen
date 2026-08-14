#!/usr/bin/env python3
"""Cityforge T0.4b: per-model local bounds + member OBB composition.

Purpose
-------
Single shared source for the building-aligned (OBB) geometry derivation used
by the v2 stamp-normalization tool and the v2 volumes tool:

* ``load_a2_local_bounds(a2_dir)`` indexes the per-model A2 evidence
  documents of one extraction run (``<run>/a2/nif_<model>--*.json``) by
  ``(model_key, source_sha256)`` and unions each document's per-shape
  ``evaluated_world_bounds_game_units`` into the model's LOCAL (unrotated,
  unscaled, placement-independent) bounding box in game units.  These are
  the same documents the ``a2-evidence`` ContentCache namespace stores
  (``tools/settlement_pipeline/a2_cache.py``), and their bounds are
  byte-identical to the surface-geometry cache's per-model ``local_bounds``
  (``tools/settlement_pipeline/surface_geometry_cache.py``; validated on
  the 398-model overlap of the Markarth run, max deviation 3e-5 GU =
  float32 noise).  Coverage is asserted at runtime: every member of both
  stamp kits matched (744/744) in the 2026-08-12 run.
* ``member_obb_corners_rel(member, theta_deg, local_bounds, world_bounds)``
  composes ONE member's oriented bounding box in the NORMALIZED (v2) frame,
  relative to the stamp anchor:

      offset'  = Rz(+theta) . offset_gu           (xy; z unchanged)
      rotz'    = rotz - theta
      corner   = offset' + Rz(-rotz') . (scale * local_corner)

  CONVENTION (measured 2026-08-12, lead): the engine places a ref as
  ``world = pos + Rz(-rotz) . (scale * local)`` -- proven against manifest
  ``world_bounds_gu`` over 140 pivot-offset members of both libraries
  (residual cluster 0.972 at mean -1.7 deg for Rz(-rotz); 0.099 = random
  for Rz(+rotz)).  The normalization frame is therefore F = Rz(+theta) (so
  walls at the modal shell rotz become axis-aligned: F.Rz(-rotz) =
  Rz(-(rotz - theta)) = Rz(-rotz')).  An earlier revision of this module
  used Rz(-theta) for offsets and Rz(+rotz') for the composition; the two
  sign errors do not cancel (2*rotz per member) and produced transposed /
  meaningless hulls for non-cardinal stamps.  Z uses the same composition
  (local z x scale + offset z; rotation never touches z).

Fallback ladder per member (each recorded):
  1. ``model_local_bounds`` -- primary (above).
  2. ``uninflated_world_aabb`` -- numeric un-inflation: solve the 2x2
     rotation-AABB system ``X = w|c| + d|s|, Y = w|s| + d|c|`` for the mesh
     dims from the member world-AABB dims + its own rotz; the box keeps the
     world-AABB center and the member orientation.  Ill-conditioned
     (``|cos 2*phi| < 0.2``) or non-positive dims fall through.
  3. ``world_aabb`` -- conservative: the member's world AABB corners rotated
     by -theta about the anchor (an outer bound of the true geometry).

Invariants
----------
* Deterministic: sorted doc scan, first doc per (model_key, source_sha256)
  wins (duplicates are identical payloads in the same run).
* Fail closed on missing local bounds only through the caller's coverage
  gate (the fallback ladder is the documented degradation, never silent).
* No TES3 authoring; read-only inputs.

Pipeline position
-----------------
Shared by ``tools/cityforge/normalize_stamp_orientation.py`` (v2 library
hull/aabb/bounds) and ``tools/cityforge/build_stamp_volumes.py`` (v2 member
boxes); both consume the same extraction products as the v1 stamp library.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
for entry in (SRC,):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from procgen.cityplan import rot2d_ccw  # noqa: E402

# Un-inflation is ill-conditioned when the rotated box's AABB is nearly
# square (|cos 2*phi| < this threshold the two dims are not separable).
UNINFLATE_MIN_ABS_COS_2PHI = 0.2

# Fallback labels recorded per member.
SRC_MODEL_LOCAL = "model_local_bounds"
SRC_UNINFLATED = "uninflated_world_aabb"
SRC_WORLD_AABB = "world_aabb"


def load_a2_local_bounds(a2_dir: str | Path) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Index one run's A2 evidence documents by (model_key, source sha256).

    Returns ``{(model_key, source_sha256): {"min": [x,y,z], "max": [x,y,z]}}``
    in game units, where the box is the union over every evaluated source
    shape object's ``evaluated_world_bounds_game_units`` (the model's local
    bounds in its own frame, including any root-node offset).
    """
    result: dict[tuple[str, str], dict[str, list[float]]] = {}
    a2_path = Path(a2_dir)
    if not a2_path.is_dir():
        raise RuntimeError(
            f"FAILURE: stamp_local_bounds A2 evidence dir missing: {a2_path}"
        )
    for doc_path in sorted(a2_path.glob("nif_*.json")):
        try:
            with doc_path.open("r", encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"FAILURE: stamp_local_bounds cannot read A2 doc {doc_path}: {exc}"
            ) from exc
        model_key = doc.get("model_key")
        source = doc.get("source") or {}
        source_sha = source.get("sha256")
        if not model_key or not source_sha:
            continue
        key = (str(model_key), str(source_sha))
        if key in result:
            continue  # duplicate doc for the same model version; identical payload
        mn = [float("inf")] * 3
        mx = [float("-inf")] * 3
        for shape in doc.get("source_shapes", []):
            if not isinstance(shape, Mapping):
                continue
            for obj in shape.get("blender_objects", []):
                wb = obj.get("evaluated_world_bounds_game_units")
                if not isinstance(wb, Mapping):
                    continue
                wb_min = wb.get("min")
                wb_max = wb.get("max")
                if not (isinstance(wb_min, Sequence) and isinstance(wb_max, Sequence)):
                    continue
                for axis in range(3):
                    mn[axis] = min(mn[axis], float(wb_min[axis]))
                    mx[axis] = max(mx[axis], float(wb_max[axis]))
        if mn[0] == float("inf"):
            continue
        result[key] = {
            "min": [float(value) for value in mn],
            "max": [float(value) for value in mx],
        }
    return result


def _rotz_deg(member: Mapping[str, Any]) -> float:
    rotation = member.get("rotation") or [0.0, 0.0, 0.0]
    if len(rotation) < 3 or not all(
        math.isfinite(float(value)) for value in rotation[:3]
    ):
        raise RuntimeError(
            f"FAILURE: stamp_local_bounds member {member.get('source_id')} "
            f"has no finite rotation triple {rotation!r}"
        )
    return math.degrees(float(rotation[2]))


def _uninflated_dims(world_bounds: Mapping[str, Sequence[float]], rotz_deg: float) -> tuple[float, float] | None:
    """Solve the 2x2 rotation-AABB system for the mesh dims (w, d).

    ``X = w|c| + d|s|``, ``Y = w|s| + d|c|`` with phi = rotz mod 90.  Returns
    None when ill-conditioned (|cos 2*phi| < 0.2) or the dims are not
    positive finite (non-box mesh).
    """
    phi = math.radians(float(rotz_deg) % 90.0)
    c, s = abs(math.cos(phi)), abs(math.sin(phi))
    denom = c * c - s * s
    if abs(denom) < UNINFLATE_MIN_ABS_COS_2PHI:
        return None
    x = float(world_bounds["max"][0]) - float(world_bounds["min"][0])
    y = float(world_bounds["max"][1]) - float(world_bounds["min"][1])
    w = (x * c - y * s) / denom
    d = (y * c - x * s) / denom
    if not (math.isfinite(w) and math.isfinite(d)) or w <= 0.0 or d <= 0.0:
        return None
    return (w, d)


def member_obb_corners_rel(
    member: Mapping[str, Any],
    theta_deg: float,
    local_bounds: Mapping[str, Sequence[float]] | None,
    world_bounds: Mapping[str, Sequence[float]],
    anchor: Sequence[float],
) -> tuple[list[list[float]], str, dict[str, Any]]:
    """One member's OBB corners in the v2 (normalized) frame, relative to the
    anchor, via the fallback ladder.

    Returns ``(corners, source_label, note)``: 8 corners
    ``[x, y, z]`` per corner.  ``note`` carries the fallback detail
    (e.g. the un-inflation condition value) for the per-member record.
    ``world_bounds`` is the member's world-AABB (absolute world GU; used by
    fallbacks 2/3 and rotated about the anchor).
    """
    scale = float(member.get("scale", 1.0))
    offset = [float(value) for value in member.get("offset_gu", [0.0, 0.0, 0.0])]
    rotz_deg = _rotz_deg(member)
    rotz_prime = (rotz_deg - theta_deg) % 360.0
    anchor_xy = (float(anchor[0]), float(anchor[1]))
    anchor_z = float(anchor[2])

    def compose_box(local_min: Sequence[float], local_max: Sequence[float]) -> list[list[float]]:
        """offset' + Rz(-rotz') . (scale * local_corner), relative to anchor."""
        ox, oy = rot2d_ccw(offset[0], offset[1], theta_deg)
        corners: list[list[float]] = []
        for lx in (local_min[0], local_max[0]):
            for ly in (local_min[1], local_max[1]):
                rx, ry = rot2d_ccw(scale * float(lx), scale * float(ly), -rotz_prime)
                corners.append([ox + rx, oy + ry, offset[2] + scale * float(local_min[2])])
                corners.append([ox + rx, oy + ry, offset[2] + scale * float(local_max[2])])
        return corners

    if local_bounds is not None:
        corners = compose_box(local_bounds["min"], local_bounds["max"])
        if all(math.isfinite(value) for corner in corners for value in corner):
            return corners, SRC_MODEL_LOCAL, {}

    # Fallback 2: numeric un-inflation from the world AABB dims + rotz.
    dims = _uninflated_dims(world_bounds, rotz_deg)
    if dims is not None:
        w, d = dims
        cx = (float(world_bounds["min"][0]) + float(world_bounds["max"][0])) / 2.0
        cy = (float(world_bounds["min"][1]) + float(world_bounds["max"][1])) / 2.0
        cz = (float(world_bounds["min"][2]) + float(world_bounds["max"][2])) / 2.0
        # Box center in the v2 frame: rotate the world center about the anchor
        # (frame transform F = Rz(+theta); see module header).
        center = rot2d_ccw(cx - anchor_xy[0], cy - anchor_xy[1], theta_deg)
        half_z = (float(world_bounds["max"][2]) - float(world_bounds["min"][2])) / 2.0
        corners: list[list[float]] = []
        for lx in (-w / 2.0, w / 2.0):
            for ly in (-d / 2.0, d / 2.0):
                rx, ry = rot2d_ccw(lx, ly, -rotz_prime)
                corners.append([center[0] + rx, center[1] + ry, cz - anchor_z - half_z])
                corners.append([center[0] + rx, center[1] + ry, cz - anchor_z + half_z])
        phi = rotz_deg % 90.0
        c, s = abs(math.cos(math.radians(phi))), abs(math.sin(math.radians(phi)))
        note = {
            "condition_cos_2phi": round(c * c - s * s, 4),
            "dims_gu": [round(w, 3), round(d, 3)],
        }
        return corners, SRC_UNINFLATED, note

    # Fallback 3: the world AABB corners rotated by +theta about the anchor
    # (frame transform F = Rz(+theta); conservative outer bound of the true
    # geometry).
    corners = []
    for x in (world_bounds["min"][0], world_bounds["max"][0]):
        for y in (world_bounds["min"][1], world_bounds["max"][1]):
            rx, ry = rot2d_ccw(float(x) - anchor_xy[0], float(y) - anchor_xy[1], theta_deg)
            corners.append([rx, ry, float(world_bounds["min"][2]) - anchor_z])
            corners.append([rx, ry, float(world_bounds["max"][2]) - anchor_z])
    return corners, SRC_WORLD_AABB, {}
