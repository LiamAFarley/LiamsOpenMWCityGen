"""Derive a full format-v1 visual plan from a MINIMAL cityforge sketch.

Pipeline position
------------------
This CLI sits between the vision-capable design agent and the existing
visual-planning toolchain.  The agent authors only a minimal ``sketch.json``
(roads, spaces, lots) against a planning bundle; this tool derives the
complete versioned visual-plan extension (``visual_plan.json``), runs the
existing hard-error analysis (``visual_planner_advisory.analyze_plan``),
writes a filtered checks file, and renders ONE composite PNG on top of the
identical planning-canvas background (``planning_canvas.render_planning_canvas``
with the bundle's exact world rectangle, so the sketch overlay matches the
canvas the agent drew on).

Sketch coordinates are TES3 WORLD GU (x east, y north) -- the design agent
reads them off the bundle canvas graticule.  The derived visual plan uses the
accepted ``site_survey_plan_gu`` frame, so every derived coordinate is
``world - survey_origin``.

Lot semantics (binding): each lot's ``x, y`` is the footprint **centroid**
(what the rules.md designer is told), NOT the stamp anchor.  Stamps are
anchor-at-seed-door by library design, so the derivation internally computes
the hull_xy_rel 2D centroid (``cityplan.polygon_centroid``), rotates it by the
lot yaw, and places the anchor at ``(x, y) - rot2d_ccw(centroid_rel, yaw)``.
The derived visual_plan.json keeps storing the ANCHOR position (downstream
format unchanged); only the sketch-frame interpretation changed.

Inputs
------
* ``--bundle <dir>``  planning bundle (canvas.png, stamps.json, site.json,
  bundle_manifest.json).  The bundle manifest pins the survey dir, the
  aligned-road product dir, and the D-STAMP library paths used to load the
  exact terrain/network/stamp geometry.
* ``--sketch <file>`` sketch v1 JSON (schema below).
* ``--out <dir>``     fresh output directory (protected-root refusal).

Sketch schema v1
---------------
``{"site": <bundle site_name>,
  "roads":  [{"id", "kind": "street"|"alley", "width_gu", "points": [[x,y],...]}],
  "spaces": [{"id", "kind": "plaza"|"court", "polygon": [[x,y],...]}],
  "lots":   [{"id", "stamp": <bundle stamp id>, "x", "y", "yaw_deg", "note"?}],
  "notes":  "free text"}``
Unknown keys anywhere are fatal.  Street width 256-1024 GU, alley 128-512 GU;
>=2 road points, >=3 polygon points; every lot stamp must exist in the
bundle's ``stamps.json`` (the bundle is eligibility-filtered, so this is the
fail-closed quarantine gate); lot positions (the footprint centroids) must
lie inside the bundle ``rectangle_gu``.  Violations print
``FAILURE: sketch <reason>`` and exit 1 with no render.

Outputs (all in ``--out``)
-------------------------
``visual_plan.json``   derived format-v1 extension (validated).
``checks.json``        ONLY hard_errors (all fired codes), per-door facts,
                       per-space footprint-touch facts, and per-pair
                       subterranean-overlap facts (Z-aware building-overlap
                       excuses from the advisory's member-volume analysis).
                       Advisory codes are deliberately deferred to the
                       placement stage.
``plan.png``           one composite: identical base canvas + authored
                       streets/alleys, plaza/court fills, yawed kit-colored
                       footprint hulls with lot-id labels, and intent-colored
                       door arrows (unconnected doors flagged red).
``sketch.copy.json``   canonical copy of the parsed sketch.
``log.json``           tool-written bookkeeping: UTC timestamp + sha256 of
                       bundle files, sketch, outputs + wall clock.

Derivation rules
----------------
* ``rectangle``        copied from site.json (cells, margin, world bounds).
* ``existing_source_roads``  one record per site.json source-road edge, with
                       the edge's clipped chain endpoints as connection points.
* ``authored_roads`` / ``alleys``  from sketch roads; ``connection_targets``
                       are derived per polyline ENDPOINT against the COMPLETE
                       circulation map -- every existing source road, every
                       authored road/alley, every space polygon -- built
                       before any endpoint connection is derived, so
                       connection validity is independent of the sketch's
                       road array order (a shared fork endpoint always sees
                       its sibling road); only the road's own id is excluded
                       (no trivial self-snap).  The endpoint snaps to the
                       nearest candidate within SNAP_DISTANCE_GU (1536).  A
                       snapped endpoint declares ``at_plan_gu`` = its own
                       coordinate (distance 0 to its own polyline, so the
                       advisory's geometric check passes).  An endpoint with
                       no target in range still declares the nearest target,
                       but with ``at_plan_gu`` = the nearest point ON that
                       target -- which is > 768 GU from the road, so the
                       existing ``road_disconnected`` hard check reports it.
* ``road_surface_polygons`` / ``shared_courts``  from sketch spaces; courts
                       carry an empty ``connection_targets`` array (required
                       by the format; court connections are not hard-gated).
* ``stamps``           position/yaw from the lot; the stored position is the
                       seed-door ANCHOR, derived from the lot's centroid
                       semantics: ``anchor = lot.xy - rot2d_ccw(hull-centroid_rel,
                       yaw)`` (see "Lot semantics" above).  ``door_intents`` are
                       derived per measured door: the bundle door offset (dx,dy)
                       is rotated by the lot yaw with the SAME transform the
                       stamp libraries use (``cityplan.rot2d_ccw``), then the
                       nearest circulation target (existing source road,
                       authored road/alley centerline, or space polygon) within
                       the existing DOOR_REACH_GU (768) decides intent: public
                       for street/plaza/existing road, service for alley/court.
                       Unconnected doors keep intent public with NO target (the
                       existing door advisories report them).  ``access_links``
                       are straight stubs door -> nearest point on target.
* ``districts`` / ``annotations`` / ``advisory_overrides``  empty arrays: the
                       format requires only the KEYS; no record semantics are
                       derivable from a minimal sketch, so none are invented.
                       ``render_options`` is omitted (format-optional; the
                       renderer uses the bundle projection directly).
                       ``design_notes`` carries the sketch's free text verbatim.

Invariants
----------
* Deterministic for identical inputs; no randomness, no Blender.
* The rendered base is byte-identical to the bundle canvas (same terrain
  product, network, rectangle, title, and px density).
* No original file is modified; outputs only land in the fresh ``--out`` dir.
* Exit 0 iff the derived plan has ZERO hard errors; the PNG is ALWAYS written
  once derivation succeeds (the image is the diagnostic).
* This tool never authors TES3 records and never changes survey semantics.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from procgen import aligned_roads, cityplan, frontage_targets  # noqa: E402
from procgen.aligned_roads import SOURCE_ROAD_PRACTICAL_PATH_FRACTION  # noqa: E402
from procgen.cityplan import (  # noqa: E402
    point_in_ring,
    point_seg_distance,
    polygon_centroid,
    ring_min_distance,
    rot2d_ccw,
)
from procgen.planning_canvas import render_planning_canvas  # noqa: E402
from procgen.visual_planner_advisory import (  # noqa: E402
    DOOR_REACH_GU,
    _corridor_rings,
    analyze_plan,
)
from procgen.visual_planner_format import (  # noqa: E402
    SCHEMA_VERSION,
    KIND,
    COORDINATE_FRAME,
    canonical_json_bytes,
    require_valid_extension,
)
from procgen.visual_planner_symbols import (  # noqa: E402
    COLORS,
    LabelPlacer,
    _composite_polygon,
    _draw_styled_road,
    _font,
    _rect_from_points,
    resolve_stamps,
)
from procgen.visual_planner_terrain import TerrainBundle  # noqa: E402

from build_planning_bundle import (  # noqa: E402
    _door_members,
    refuse_unless_fresh,
)
from visual_planner import load_stamp_geometry  # noqa: E402

#: Road polyline endpoint snap distance (GU) for connection_targets.
SNAP_DISTANCE_GU = 1536.0
#: Road connection tolerance written into derived targets (matches the
#: advisory's default ROAD_CONNECTION_TOLERANCE_GU).
CONNECTION_TOLERANCE_GU = 768.0
#: Facing-cone half-angle (2026-08-12): a door whose outward heading deviates
#: from the direction to every in-reach circulation target by more than this
#: is rendered as "faces away" (magenta) — the plain reach check is
#: direction-blind and cannot tell a facing door from a sideways one.
FACING_CONE_DEG = 60.0
#: Space-fact footprint buffer for "lot touches space polygon".
SPACE_TOUCH_BUFFER_GU = 512.0

#: Sketch kind -> derived road class.
_ROAD_CLASS = {"street": "street", "alley": "service"}
#: Sketch kind -> derived surface string.
_ROAD_SURFACE = {"street": "road", "alley": "settlement_dirt"}
#: Sketch space kind -> derived record kind.
_SPACE_KIND = {"plaza": "plaza", "court": "shared_court"}
_SPACE_SURFACE = {"plaza": "settlement_cobble", "court": "settlement_grass_dirt"}
#: Circulation target kinds that count as PUBLIC-facing for door intents.
_PUBLIC_KINDS = {"existing_source_road", "authored_road", "road_surface_polygon"}


class SketchError(ValueError):
    """Fatal sketch schema violation (printed as ``FAILURE: sketch ...``)."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label} {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


# --------------------------------------------------------------------------
# Sketch validation (strict; unknown keys are fatal)
# --------------------------------------------------------------------------

def _require_keys(record: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise SketchError(f"{path} has unknown keys {unknown}")


def _finite_pair(value: Any, path: str) -> tuple[float, float]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2 and
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and
                math.isfinite(float(v)) for v in value)):
        raise SketchError(f"{path} must be a finite [x, y] pair")
    return float(value[0]), float(value[1])


def _points(value: Any, minimum: int, path: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or len(value) < minimum:
        raise SketchError(f"{path} must be a list of >= {minimum} points")
    return [_finite_pair(point, f"{path}[{index}]") for index, point in enumerate(value)]


def _validate_road(record: Any, index: int, seen: set[str]) -> dict[str, Any]:
    path = f"$.roads[{index}]"
    if not isinstance(record, dict):
        raise SketchError(f"{path} must be an object")
    _require_keys(record, {"id", "kind", "width_gu", "points"}, path)
    road_id = record.get("id")
    if not isinstance(road_id, str) or not road_id:
        raise SketchError(f"{path}.id must be a non-empty string")
    if road_id in seen:
        raise SketchError(f"duplicate road id {road_id!r}")
    seen.add(road_id)
    kind = record.get("kind")
    if kind not in ("street", "alley"):
        raise SketchError(f"{path}.kind must be 'street' or 'alley'")
    width = record.get("width_gu")
    lo, hi = (256.0, 1024.0) if kind == "street" else (128.0, 512.0)
    if not (isinstance(width, (int, float)) and not isinstance(width, bool) and
            math.isfinite(float(width)) and lo <= float(width) <= hi):
        raise SketchError(f"{path}.width_gu must be {lo:g}-{hi:g} for kind {kind!r}")
    points = _points(record.get("points"), 2, f"{path}.points")
    return {"id": road_id, "kind": kind, "width_gu": float(width), "points": points}


def _validate_space(record: Any, index: int, seen: set[str]) -> dict[str, Any]:
    path = f"$.spaces[{index}]"
    if not isinstance(record, dict):
        raise SketchError(f"{path} must be an object")
    _require_keys(record, {"id", "kind", "polygon"}, path)
    space_id = record.get("id")
    if not isinstance(space_id, str) or not space_id:
        raise SketchError(f"{path}.id must be a non-empty string")
    if space_id in seen:
        raise SketchError(f"duplicate space id {space_id!r}")
    seen.add(space_id)
    if record.get("kind") not in ("plaza", "court"):
        raise SketchError(f"{path}.kind must be 'plaza' or 'court'")
    polygon = _points(record.get("polygon"), 3, f"{path}.polygon")
    return {"id": space_id, "kind": record["kind"], "polygon": polygon}


def _validate_lot(record: Any, index: int, seen: set[str], stamp_ids: set[str],
                  rect_gu: Sequence[float],
                  door_ids_by_stamp: Mapping[str, set[str]] | None = None,
                  target_ids: set[str] | None = None) -> dict[str, Any]:
    path = f"$.lots[{index}]"
    if not isinstance(record, dict):
        raise SketchError(f"{path} must be an object")
    _require_keys(record, {"id", "stamp", "x", "y", "yaw_deg", "note", "door_targets"}, path)
    lot_id = record.get("id")
    if not isinstance(lot_id, str) or not lot_id:
        raise SketchError(f"{path}.id must be a non-empty string")
    if lot_id in seen:
        raise SketchError(f"duplicate lot id {lot_id!r}")
    seen.add(lot_id)
    stamp_id = record.get("stamp")
    if not isinstance(stamp_id, str) or stamp_id not in stamp_ids:
        raise SketchError(
            f"{path}.stamp {stamp_id!r} is not in the bundle stamps.json "
            f"(quarantined or unknown stamp ids are rejected fail-closed)")
    for key in ("x", "y", "yaw_deg"):
        value = record.get(key)
        if not (isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(float(value))):
            raise SketchError(f"{path}.{key} must be finite")
    x, y = float(record["x"]), float(record["y"])
    if not (rect_gu[0] <= x <= rect_gu[2] and rect_gu[1] <= y <= rect_gu[3]):
        raise SketchError(
            f"{path} position ({x:g},{y:g}) lies outside rectangle_gu {list(rect_gu)}")
    if "note" in record and not isinstance(record["note"], str):
        raise SketchError(f"{path}.note must be a string")
    door_targets = record.get("door_targets")
    if door_targets is not None:
        if not isinstance(door_targets, list):
            raise SketchError(f"{path}.door_targets must be an array")
        seen_doors: set[str] = set()
        allowed_doors = (door_ids_by_stamp or {}).get(stamp_id)
        for dindex, target in enumerate(door_targets):
            dpath = f"{path}.door_targets[{dindex}]"
            if not isinstance(target, dict):
                raise SketchError(f"{dpath} must be an object")
            _require_keys(target, {"door_id", "target_id", "intent"}, dpath)
            door_id = target.get("door_id")
            target_id = target.get("target_id")
            if not isinstance(door_id, str) or not door_id:
                raise SketchError(f"{dpath}.door_id must be a non-empty string")
            if door_id in seen_doors:
                raise SketchError(f"duplicate explicit door target {door_id!r} in lot {lot_id!r}")
            seen_doors.add(door_id)
            if allowed_doors is not None and door_id not in allowed_doors:
                raise SketchError(
                    f"{dpath}.door_id {door_id!r} is not a door on stamp {stamp_id!r}")
            if not isinstance(target_id, str) or not target_id:
                raise SketchError(f"{dpath}.target_id must be a non-empty string")
            if target_ids is not None and target_id not in target_ids:
                raise SketchError(
                    f"{dpath}.target_id {target_id!r} is not an authored/source target")
            if target.get("intent") not in ("public", "service"):
                raise SketchError(f"{dpath}.intent must be 'public' or 'service'")
    normalized = {"id": lot_id, "stamp": stamp_id, "x": x, "y": y,
                  "yaw_deg": float(record["yaw_deg"]), "note": record.get("note")}
    if door_targets is not None:
        normalized["door_targets"] = [
            {"door_id": str(target["door_id"]), "target_id": str(target["target_id"]),
             "intent": str(target["intent"])}
            for target in door_targets
        ]
    return normalized


def validate_sketch(sketch: Any, site_name: str, stamp_ids: set[str],
                    rect_gu: Sequence[float],
                    door_ids_by_stamp: Mapping[str, set[str]] | None = None,
                    target_ids: set[str] | None = None) -> dict[str, Any]:
    """Strictly validate a sketch v1 document; raise SketchError on any issue."""
    if not isinstance(sketch, dict):
        raise SketchError("sketch must be a JSON object")
    _require_keys(sketch, {"schema_version", "site", "roads", "spaces", "lots", "notes"}, "$")
    if "schema_version" in sketch and sketch["schema_version"] != 1:
        raise SketchError("$.schema_version must be integer 1")
    if sketch.get("site") != site_name:
        raise SketchError(
            f"$.site {sketch.get('site')!r} does not match bundle site {site_name!r}")
    if not isinstance(sketch.get("roads"), list) or not isinstance(sketch.get("spaces"), list) \
            or not isinstance(sketch.get("lots"), list):
        raise SketchError("$.roads/$.spaces/$.lots must be arrays")
    if not isinstance(sketch.get("notes"), str):
        raise SketchError("$.notes must be a string")
    roads = [_validate_road(record, index, set())
             for index, record in enumerate(sketch["roads"])]
    spaces = [_validate_space(record, index, set())
              for index, record in enumerate(sketch["spaces"])]
    lots = [_validate_lot(record, index, set(), stamp_ids, rect_gu,
                          door_ids_by_stamp, target_ids)
            for index, record in enumerate(sketch["lots"])]
    normalized = {"site": sketch["site"], "roads": roads, "spaces": spaces,
            "lots": lots, "notes": sketch["notes"]}
    return normalized


# --------------------------------------------------------------------------
# Circulation targets (plan-frame) and nearest-target helpers
# --------------------------------------------------------------------------

def _nearest_polyline(point: Sequence[float],
                      polyline: Sequence[Sequence[float]]) -> tuple[float, tuple[float, float]]:
    return frontage_targets.nearest_polyline(point, polyline)


def _nearest_point_on_ring(point: Sequence[float],
                           ring: Sequence[Sequence[float]]) -> tuple[float, tuple[float, float]]:
    """Nearest ring point/distance; a point inside the ring is at distance 0."""
    return frontage_targets.nearest_point_on_ring(point, ring)


def _reach_distance(point: Sequence[float], target: Mapping[str, Any]) -> float:
    """Door-reach distance to one target (advisory-compatible metric).

    Polylines measure the nearest CENTERLINE distance, EXCEPT existing
    source roads: their VTEX-blended corridors overstate the walkable path
    (~2.5x), so source-road reach is measured to the PRACTICAL PATH edge
    (a door at the path edge is at the road).  This mirrors
    ``visual_planner_advisory._target_distance`` exactly; for authored
    roads/alleys (half-width <= 512 GU < 768 reach) centerline reach already
    subsumes corridor reach.
    """
    return frontage_targets.reach_distance(point, target)


def _nearest_target(point: Sequence[float],
                    targets: Mapping[str, Mapping[str, Any]]) -> tuple[float, str, dict[str, Any]]:
    """Nearest target with advisory tie-breaking (distance, then id)."""
    return frontage_targets.nearest_target(point, targets)


def build_target_map(site: Mapping[str, Any], origin: Sequence[float],
                     network: Any) -> dict[str, dict[str, Any]]:
    """Plan-frame circulation targets: existing source roads only.

    Authored roads/alleys/spaces are appended by the derivation stage once
    their plan polylines exist; the map shape matches the advisory's
    ``_plan_target_map`` records (kind/polyline/polygon/width_gu).
    """
    return frontage_targets.build_target_map(site, origin, network)


def _road_target(record: Mapping[str, Any], kind: str) -> dict[str, Any]:
    return {"kind": kind, "polyline": record["polyline_plan_gu"],
            "width_gu": float(record["width_gu"])}


# --------------------------------------------------------------------------
# Overlap painting + auto-face helpers
# --------------------------------------------------------------------------

def _clip_convex(subject: Sequence[Sequence[float]],
                 clip: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip of a convex subject ring by a convex clip ring.

    Clip-ring orientation is detected from its signed area, so callers may
    pass either winding.  Returns the (possibly empty) intersection ring.
    Used only for PAINTING hard-error overlap regions red on plan.png.
    """
    if len(subject) < 3 or len(clip) < 3:
        return []
    area = sum(float(clip[i][0]) * float(clip[(i + 1) % len(clip)][1])
               - float(clip[(i + 1) % len(clip)][0]) * float(clip[i][1])
               for i in range(len(clip)))
    if abs(area) < 1e-9:
        return []
    sign = 1.0 if area > 0.0 else -1.0

    def inside(p: tuple[float, float], a: Sequence[float], b: Sequence[float]) -> bool:
        cross = ((float(b[0]) - float(a[0])) * (p[1] - float(a[1]))
                 - (float(b[1]) - float(a[1])) * (p[0] - float(a[0])))
        return cross * sign >= -1e-9

    def intersect(s: tuple[float, float], e: tuple[float, float],
                  a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
        d1 = ((float(b[0]) - float(a[0])) * (s[1] - float(a[1]))
              - (float(b[1]) - float(a[1])) * (s[0] - float(a[0])))
        d2 = ((float(b[0]) - float(a[0])) * (e[1] - float(a[1]))
              - (float(b[1]) - float(a[1])) * (e[0] - float(a[0])))
        t = d1 / (d1 - d2) if abs(d1 - d2) > 1e-12 else 0.0
        return (s[0] + (e[0] - s[0]) * t, s[1] + (e[1] - s[1]) * t)

    output = [(float(p[0]), float(p[1])) for p in subject]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        if not output:
            break
        new_output: list[tuple[float, float]] = []
        for j in range(len(output)):
            s, e = output[j], output[(j + 1) % len(output)]
            if inside(e, a, b):
                if not inside(s, a, b):
                    new_output.append(intersect(s, e, a, b))
                new_output.append(e)
            elif inside(s, a, b):
                new_output.append(intersect(s, e, a, b))
        output = new_output
    return output


def _overlap_polys_world(advisory: Mapping[str, Any], records: Sequence[Any],
                         targets: Mapping[str, Mapping[str, Any]],
                         origin: Sequence[float]) -> list[list[tuple[float, float]]]:
    """World-frame polygons of hard-error overlap regions, painted red.

    building_overlap: intersection of the two lots' transformed hulls.
    building_road_overlap: intersection of the lot hull with the target's
    corridor quads (practical path width for source roads, declared width
    for authored circulation -- the same corridors the checks use).
    """
    by_lot = {record.lot_id: record for record in records}
    polys: list[list[tuple[float, float]]] = []
    for err in advisory.get("hard_errors", []):
        if not isinstance(err, Mapping):
            continue
        code = err.get("code")
        lots = [by_lot.get(str(lot_id)) for lot_id in err.get("lot_ids", [])]
        lots = [record for record in lots if record is not None and record.hull]
        pieces: list[list[tuple[float, float]]] = []
        if code == "building_overlap" and len(lots) == 2:
            inter = _clip_convex(lots[0].hull, lots[1].hull)
            if inter:
                pieces.append(inter)
        elif code == "building_road_overlap" and lots:
            for target_id in err.get("target_ids", []):
                target = targets.get(str(target_id))
                if target is None:
                    continue
                for quad in _corridor_rings(target):
                    inter = _clip_convex(lots[0].hull, quad)
                    if inter:
                        pieces.append(inter)
        for piece in pieces:
            polys.append([_plan_to_world(p, origin) for p in piece])
    return polys


def _authored_targets(sketch: Mapping[str, Any],
                      origin: Sequence[float]) -> dict[str, dict[str, Any]]:
    """Circulation targets for the sketch's authored roads/spaces (plan frame)."""
    return frontage_targets.authored_targets(sketch, origin)


def apply_auto_face(sketch: Mapping[str, Any], site: Mapping[str, Any],
                    geometry: Mapping[str, Mapping[str, Any]],
                    network: Any, origin: Sequence[float]) -> dict[str, Any]:
    """Return a sketch copy with every lot's yaw set so its primary door
    faces the nearest circulation target.

    Rule: desired facing = direction from the door's world position to the
    nearest point on the nearest circulation target (source roads, authored
    streets/alleys, spaces); yaw = desired - door's outward_heading_deg.
    Two relaxation passes (the door position itself moves with yaw).  The
    primary door is the stamp's seed door (first door fallback).  This is a
    mechanical convenience for the designer, never a correctness gate.
    """
    targets = build_target_map(site, origin, network)
    targets.update(_authored_targets(sketch, origin))
    adjusted = json.loads(json.dumps(sketch))
    for lot in adjusted.get("lots", []):
        stamp = geometry.get(lot["stamp"])
        if stamp is None:
            continue
        doors = [m for m in stamp.get("members", []) if m.get("is_door")]
        if not doors:
            continue
        seed_id = (stamp.get("anchor") or {}).get("seed_door")
        door = next((m for m in doors if m.get("source_id") == seed_id), doors[0])
        heading_local = door.get("outward_heading_deg")
        if heading_local is None:
            heading_local = math.degrees(float(door.get("rotation", [0, 0, 0])[2])) % 360.0
        centroid_rel = _stamp_centroid_rel(stamp)
        d_local = (float(door["offset_gu"][0]), float(door["offset_gu"][1]))
        yaw = float(lot.get("yaw_deg", 0.0))
        for _ in range(2):  # relax: door position depends on yaw
            rcx, rcy = rot2d_ccw(centroid_rel[0], centroid_rel[1], yaw)
            anchor = (float(lot["x"]) - rcx, float(lot["y"]) - rcy)
            rdx, rdy = rot2d_ccw(d_local[0], d_local[1], yaw)
            door_plan = (anchor[0] + rdx - origin[0], anchor[1] + rdy - origin[1])
            best = (float("inf"), None)
            for target_id in sorted(targets):
                target = targets[target_id]
                distance = _reach_distance(door_plan, target)
                if distance >= best[0]:
                    continue
                polyline = target.get("polyline")
                if isinstance(polyline, list) and len(polyline) >= 2:
                    point = _nearest_polyline(door_plan, polyline)[1]
                else:
                    polygon = target.get("polygon")
                    if isinstance(polygon, list) and len(polygon) >= 3:
                        if point_in_ring(door_plan, polygon):
                            point = door_plan
                        else:
                            point = _nearest_point_on_ring(door_plan, polygon)[1]
                    else:
                        continue
                best = (distance, point)
            if best[1] is None:
                break
            desired = math.degrees(math.atan2(best[1][1] - door_plan[1],
                                              best[1][0] - door_plan[0]))
            yaw = (desired - float(heading_local)) % 360.0
        lot["yaw_deg"] = round(yaw, 1)
    return adjusted


# --------------------------------------------------------------------------
# Derive: sketch -> format-v1 visual plan
# --------------------------------------------------------------------------

def _derive_connections(record: dict[str, Any], *, polyline_plan_gu: list[list[float]],
                        targets: Mapping[str, Mapping[str, Any]],
                        own_id: str) -> list[dict[str, Any]]:
    """One connection_target per polyline endpoint (see module derivation rules).

    In-range endpoints declare their own coordinate (distance 0 to their own
    polyline -> the advisory's geometric check passes).  Out-of-range
    endpoints declare the nearest point ON the nearest target, which sits
    > tolerance from the road, so ``road_disconnected`` fires with a measured
    distance instead of the connection being silently dropped.
    """
    connections: list[dict[str, Any]] = []
    candidates = {target_id: target for target_id, target in targets.items()
                  if target_id != own_id}
    for endpoint in (polyline_plan_gu[0], polyline_plan_gu[-1]):
        distance, target_id, target = _nearest_target(endpoint, candidates)
        if not target_id:
            continue
        if distance <= SNAP_DISTANCE_GU:
            connections.append({
                "target_id": target_id,
                "at_plan_gu": [round(endpoint[0], 1), round(endpoint[1], 1)],
                "tolerance_gu": CONNECTION_TOLERANCE_GU,
                "reason": f"endpoint snaps to {target_id} at {distance:.0f} GU",
            })
        else:
            _, point_on_target = (
                _nearest_polyline(endpoint, target["polyline"])
                if target.get("polyline") else
                _nearest_point_on_ring(endpoint, target.get("polygon", [])))
            connections.append({
                "target_id": target_id,
                "at_plan_gu": [round(point_on_target[0], 1), round(point_on_target[1], 1)],
                "tolerance_gu": CONNECTION_TOLERANCE_GU,
                "reason": f"no target within {SNAP_DISTANCE_GU:g} GU "
                          f"(nearest {target_id} at {distance:.0f} GU); left for road_disconnected",
            })
    return connections


def _derive_lot_doors(lot: Mapping[str, Any], stamp: Mapping[str, Any],
                      bundle_entry: Mapping[str, Any], origin: Sequence[float],
                      targets: Mapping[str, Mapping[str, Any]],
                      anchor: Sequence[float],
                      ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive door_intents + access_links + per-door facts for one lot.

    ``anchor`` is the seed-door anchor in TES3 WORLD GU derived from the lot's
    centroid semantics (``lot.xy - rot2d_ccw(hull_centroid_rel, yaw)``); door
    offsets are relative to it by library design.  Door offsets are rotated
    with the SAME transform the stamp libraries use
    (``cityplan.rot2d_ccw``, as in ``visual_planner_symbols._transform_door``).
    """
    members = _door_members(stamp)
    if not members:
        raise SketchError(
            f"lot {lot['id']!r} stamp {lot['stamp']!r} has no door members in the library")
    declared = int(bundle_entry.get("door_count", -1))
    if declared != len(members):
        raise SketchError(
            f"lot {lot['id']!r} stamp {lot['stamp']!r} door_count {declared} "
            f"!= library door members {len(members)}")
    intents: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    explicit_targets = {
        str(row["door_id"]): row
        for row in lot.get("door_targets", [])
        if isinstance(row, Mapping) and isinstance(row.get("door_id"), str)
    }
    for member in members:
        door_id = str(member.get("source_id", ""))
        offset = member.get("offset_gu", [0.0, 0.0, 0.0])
        rx, ry = rot2d_ccw(float(offset[0]), float(offset[1]), float(lot["yaw_deg"]))
        plan_door = (anchor[0] - origin[0] + rx, anchor[1] - origin[1] + ry)
        outward = member.get("outward_heading_deg")
        if outward is None:
            outward = math.degrees(float(member.get("rotation", [0, 0, 0])[2]))
        heading = (float(outward) + float(lot["yaw_deg"])) % 360.0
        assignment = explicit_targets.get(door_id)
        assignment_mode = "explicit" if assignment is not None else "nearest_legacy"
        if assignment is not None:
            target_id = str(assignment["target_id"])
            target = targets.get(target_id)
            if target is None:
                # Structural validation accepts any site.json source-road edge
                # id, but build_target_map silently drops edges the aligned
                # network cannot resolve; such a row must fail closed here
                # with a named SketchError instead of a raw KeyError.
                raise SketchError(
                    f"lot {lot['id']!r} explicit door {door_id!r} target {target_id!r} "
                    f"is declared but cannot be resolved to circulation geometry")
            distance = _reach_distance(plan_door, target)
        else:
            distance, target_id, target = _nearest_target(plan_door, targets)
        target_kind = str(target.get("kind", "")) if target else None
        target_distance, target_point = frontage_targets.target_nearest_point(plan_door, target)
        facing_deg: float | None = None
        if target_point is not None:
            if target_distance < 1.0:
                facing_deg = 0.0
            else:
                to_target = math.degrees(math.atan2(target_point[1] - plan_door[1],
                                                    target_point[0] - plan_door[0]))
                facing_deg = round(abs((heading - to_target + 180.0) % 360.0 - 180.0), 1)
        # Legacy rows retain the historical nearest-target assignment.  An
        # explicit row is never retargeted, even when it is outside reach.
        # distance_gu, the access link, and facing_deviation_deg all describe
        # the SAME assigned target for both modes: nearest_legacy facing is
        # measured against the assigned nearest target, NOT the pre-v1
        # minimum over all in-reach targets.  That min-any-target mismatch is
        # deliberately not restored (plan §5).
        if distance <= DOOR_REACH_GU:
            intent = ("public" if target.get("kind") in _PUBLIC_KINDS else "service")
            intent = str(assignment["intent"]) if assignment is not None else intent
            intents.append({"door_id": door_id, "intent": intent,
                            "target_id": target_id,
                            "reason": (f"explicit circulation {target_id} "
                                       if assignment is not None else
                                       f"nearest circulation {target_id} ")
                                      + f"({target.get('kind')}) at {distance:.0f} GU"})
            if target_point is not None:
                links.append({"door_id": door_id, "target_id": target_id,
                              "polyline_plan_gu": [
                                  [round(plan_door[0], 1), round(plan_door[1], 1)],
                                  [round(target_point[0], 1), round(target_point[1], 1)]]})
            facts.append({"lot_id": lot["id"], "door_id": door_id, "intent": intent,
                           "target_id": target_id, "distance_gu": round(distance, 1),
                           "unconnected": False,
                           "facing_deviation_deg": facing_deg,
                           "target_kind": target_kind,
                           "assignment_mode": assignment_mode})
        else:
            # Explicit out-of-reach targets remain named in the intent/facts;
            # only legacy nearest behavior omits an unavailable target.
            intent = str(assignment["intent"]) if assignment is not None else "public"
            intent_row: dict[str, Any] = {
                "door_id": door_id, "intent": intent,
                "reason": (f"explicit target {target_id} is outside "
                            if assignment is not None else "no circulation target within ")
                           + f"{DOOR_REACH_GU:g} GU ({distance:.0f} GU)"}
            if assignment is not None:
                intent_row["target_id"] = target_id
            intents.append(intent_row)
            if assignment is not None and target_point is not None:
                links.append({"door_id": door_id, "target_id": target_id,
                              "polyline_plan_gu": [
                                  [round(plan_door[0], 1), round(plan_door[1], 1)],
                                  [round(target_point[0], 1), round(target_point[1], 1)]]})
            facts.append({"lot_id": lot["id"], "door_id": door_id, "intent": intent,
                          "target_id": target_id if assignment is not None else None,
                          "distance_gu": round(distance, 1),
                          "unconnected": True,
                          "facing_deviation_deg": facing_deg,
                          "target_kind": target_kind,
                          "assignment_mode": assignment_mode})
    return intents, links, facts


def _stamp_centroid_rel(stamp: Mapping[str, Any]) -> tuple[float, float]:
    """2D centroid of the stamp's ``footprint.hull_xy_rel`` (anchor-relative).

    Degenerate/missing hulls fall back to the anchor itself, which makes the
    lot position the anchor -- the pre-centroid behavior -- rather than
    inventing an offset.
    """
    hull = stamp.get("footprint", {}).get("hull_xy_rel", [])
    if not isinstance(hull, list) or len(hull) < 3:
        return (0.0, 0.0)
    return polygon_centroid([[float(p[0]), float(p[1])] for p in hull])


def derive_plan(sketch: Mapping[str, Any], site: Mapping[str, Any], bundle_stamps: Any,
                geometry: Mapping[str, Mapping[str, Any]], network: Any,
                terrain: TerrainBundle) -> dict[str, Any]:
    """Build the full format-v1 visual plan document from a validated sketch."""
    origin = terrain.origin_gu
    rect_gu = [float(value) for value in site["rectangle_gu"]]
    plan_rect = {
        "cell_bounds": list(site["cells"]),
        "context_margin_gu": float(site["margin_gu"]),
        "full_site_inset": False,
        "world_bounds_gu": [[rect_gu[0], rect_gu[1]], [rect_gu[2], rect_gu[3]]],
    }
    world_to_plan = lambda p: (p[0] - origin[0], p[1] - origin[1])  # noqa: E731

    existing_source_roads = []
    for row in site.get("source_roads", []):
        edge_id = row.get("edge_id")
        if not isinstance(edge_id, str) or not row.get("points_gu"):
            continue
        chain = [world_to_plan(p) for p in row["points_gu"]]
        existing_source_roads.append({
            "edge_id": edge_id,
            "label": f"aligned source road {edge_id[-10:]}",
            "hierarchy": "regional_approach",
            "show_corridor": True,
            "corridor_margin_gu": 0.0,
            "connection_points": [[round(p[0], 1), round(p[1], 1)] for p in (chain[0], chain[-1])],
        })

    targets = build_target_map(site, origin, network)
    authored_roads: list[dict[str, Any]] = []
    alleys: list[dict[str, Any]] = []
    # Pass 1: register EVERY authored road/alley in the target map before any
    # endpoint connection is derived.  Connection validity must not depend on
    # the sketch's JSON array order: at a fork shared by two authored roads
    # the road listed first could not see the road listed later and was
    # falsely emitted as ``road_disconnected`` (2026-08-13 shared-fork fix).
    pending_connections: list[tuple[dict[str, Any], str, list[list[float]]]] = []
    for road in sketch["roads"]:
        polyline_plan_gu = [[round(p[0] - origin[0], 1), round(p[1] - origin[1], 1)]
                            for p in road["points"]]
        record = {
            "road_id" if road["kind"] == "street" else "alley_id": road["id"],
            "class": _ROAD_CLASS[road["kind"]],
            "width_gu": road["width_gu"],
            "surface": _ROAD_SURFACE[road["kind"]],
            "polyline_plan_gu": polyline_plan_gu,
        }
        own_id = road["id"]
        targets[own_id] = _road_target(record, "authored_road" if road["kind"] == "street" else "alley")
        pending_connections.append((record, own_id, polyline_plan_gu))
        (authored_roads if road["kind"] == "street" else alleys).append(record)

    road_surface_polygons: list[dict[str, Any]] = []
    shared_courts: list[dict[str, Any]] = []
    for space in sketch["spaces"]:
        polygon_plan_gu = [[round(p[0] - origin[0], 1), round(p[1] - origin[1], 1)]
                           for p in space["polygon"]]
        if space["kind"] == "plaza":
            road_surface_polygons.append({
                "region_id": space["id"], "kind": "plaza",
                "surface": _SPACE_SURFACE["plaza"], "polygon_plan_gu": polygon_plan_gu})
            targets[space["id"]] = {"kind": "road_surface_polygon",
                                    "polygon": polygon_plan_gu, "width_gu": 0.0}
        else:
            shared_courts.append({
                "court_id": space["id"], "surface": _SPACE_SURFACE["court"],
                "polygon_plan_gu": polygon_plan_gu, "connection_targets": []})
            targets[space["id"]] = {"kind": "shared_court",
                                    "polygon": polygon_plan_gu, "width_gu": 0.0}

    # Pass 2: derive endpoint connections against the now-complete target map
    # (source roads + every authored road/alley + spaces).  Each road's
    # candidate set is that complete map minus its own id, so the nearest
    # target, tie break, and fail-closed out-of-range record are identical
    # regardless of JSON array order.
    for record, own_id, polyline_plan_gu in pending_connections:
        record["connection_targets"] = _derive_connections(
            record, polyline_plan_gu=polyline_plan_gu, targets=targets,
            own_id=own_id)

    bundle_by_id = {entry["id"]: entry for entry in bundle_stamps.get("stamps", [])}
    stamps: list[dict[str, Any]] = []
    door_facts: list[dict[str, Any]] = []
    for lot in sketch["lots"]:
        bundle_entry = bundle_by_id[lot["stamp"]]
        stamp_geometry = geometry[lot["stamp"]]
        # Lot x,y is the footprint CENTROID (rules.md semantics).  The stamp
        # anchor sits at the rotated centroid offset from it; the derived plan
        # keeps storing the anchor position (downstream format unchanged).
        centroid_rel = _stamp_centroid_rel(stamp_geometry)
        rotated_centroid = rot2d_ccw(centroid_rel[0], centroid_rel[1],
                                     float(lot["yaw_deg"]))
        anchor = (float(lot["x"]) - rotated_centroid[0],
                  float(lot["y"]) - rotated_centroid[1])
        intents, links, facts = _derive_lot_doors(
            lot, stamp_geometry, bundle_entry, origin, targets, anchor)
        door_facts.extend(facts)
        stamps.append({
            "lot_id": lot["id"],
            "stamp_id": lot["stamp"],
            "position_plan_gu": [round(anchor[0] - origin[0], 1),
                                 round(anchor[1] - origin[1], 1)],
            "yaw_deg": lot["yaw_deg"],
            "kit": bundle_entry.get("kit") or "unknown",
            "category": bundle_entry.get("building_type") or "building",
            "door_intents": intents,
            "access_links": links,
        })

    seed = int(hashlib.sha256(f"{site['site_name']}|plan_sketch_v1".encode()).hexdigest()[:8], 16)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "plan_id": f"{site['site_name']}_sketch_v1",
        "base_t1_1_plan_id": site["site_name"],
        "seed": seed,
        "coordinate_frame": COORDINATE_FRAME,
        "rectangle": plan_rect,
        "existing_source_roads": existing_source_roads,
        "authored_roads": authored_roads,
        "alleys": alleys,
        "road_surface_polygons": road_surface_polygons,
        "shared_courts": shared_courts,
        "stamps": stamps,
        "districts": [],
        "annotations": [],
        "advisory_overrides": [],
        "design_notes": sketch["notes"],
    }, door_facts


# --------------------------------------------------------------------------
# checks.json (hard errors only + facts; advisories deferred by design)
# --------------------------------------------------------------------------

def _space_facts(sketch: Mapping[str, Any], plan: Mapping[str, Any],
                 records: Sequence[Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    sections = {"plaza": ("road_surface_polygons", "region_id"),
                "court": ("shared_courts", "court_id")}
    for space in sketch["spaces"]:
        section, id_key = sections[space["kind"]]
        record = next(row for row in plan[section] if row[id_key] == space["id"])
        polygon = [list(p) for p in record["polygon_plan_gu"]]
        touching = [record_.lot_id for record_ in records
                    if record_.hull and ring_min_distance(
                        [list(p) for p in record_.hull], polygon) <= SPACE_TOUCH_BUFFER_GU]
        facts.append({"space_id": space["id"], "kind": space["kind"],
                      "touch_buffer_gu": SPACE_TOUCH_BUFFER_GU,
                      "touching_lots": sorted(touching),
                      "touch_count": len(touching)})
    return facts


def build_checks(plan: Mapping[str, Any], sketch: Mapping[str, Any],
                 advisory: Mapping[str, Any], door_facts: list[dict[str, Any]],
                 records: Sequence[Any]) -> dict[str, Any]:
    """checks.json: the full hard-error list + door/space/subterranean facts ONLY.

    Advisory codes (terrain/slope/repetition/orientation/frontage/tandem/
    circulation) are intentionally NOT included; they are deferred to the
    later placement stage.  ``hard_errors`` is the advisory's complete list
    (any of the hard-error codes that fired), gating acceptance.
    ``subterranean_overlap_facts`` reports XY-overlapping lot pairs whose 3D
    member conflicts lie entirely below the target terrain surface (facts,
    not counts -- the per-conflict measured numbers are the evidence).
    ``stamp_usage`` reports how often each stamp is used (variety is a
    design goal: 50+ eligible stamps, so heavy repetition is visible here).
    """
    usage: dict[str, int] = {}
    for lot in sketch.get("lots", []):
        stamp_id = str(lot.get("stamp", ""))
        usage[stamp_id] = usage.get(stamp_id, 0) + 1
    return {
        "schema_version": 1,
        "kind": "cityforge_sketch_derived_checks",
        "plan_id": plan["plan_id"],
        "hard_errors": list(advisory.get("hard_errors", [])),
        "hard_error_count": len(advisory.get("hard_errors", [])),
        "door_facts": door_facts,
        "space_facts": _space_facts(sketch, plan, records),
        "subterranean_overlap_facts": list(advisory.get("subterranean_overlap_facts", [])),
        "stamp_usage": dict(sorted(usage.items())),
        "stamps_used_count": len(usage),
    }


# --------------------------------------------------------------------------
# Render: one composite on the identical planning-canvas base
# --------------------------------------------------------------------------

def _canvas_title(site: Mapping[str, Any]) -> str:
    min_x, max_x, min_y, max_y = site["cells"]
    x0, y0, x1, y1 = site["rectangle_gu"]
    cells_label = f"x={min_x}..{max_x}, y={min_y}..{max_y}"
    return (f"{site['site_name']} — cells {cells_label} — GU [{x0:.0f},{y0:.0f}].."
            f"[{x1:.0f},{y1:.0f}] (margin {site['margin_gu']} GU)")


#: Overlay legend strip height in pixels, appended below the base legend band.
OVERLAY_LEGEND_PX = 32
#: Strip background; matches the base canvas legend band (planning_canvas).
_LEGEND_STRIP_BG = (28, 34, 40)


def _blend_over(color: tuple[int, int, int, int],
                bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Opaque RGB of an RGBA color alpha-composited over a background.

    The overlay fills are translucent; drawing them raw on the legend strip
    would not match the composited look, so the swatches use the exact
    blended appearance the overlay produces on the band background.
    """
    a = color[3] / 255.0
    return tuple(round(a * channel + (1.0 - a) * bg_channel)
                 for channel, bg_channel in zip(color[:3], bg))


def _extend_overlay_legend(image: Image.Image) -> Image.Image:
    """Append one flat legend strip keying every overlay element.

    Swatches replicate the overlay styles at miniature scale with the exact
    colors the composite draws: styled edge/centre road bands, translucent
    outlined polygon fills, and haloed intent-colored door arrows.
    """
    extended = Image.new("RGBA", (image.width, image.height + OVERLAY_LEGEND_PX),
                         (0, 0, 0, 0))
    extended.alpha_composite(image, (0, 0))
    draw = ImageDraw.Draw(extended)
    top = image.height
    draw.rectangle((0, top, extended.width, extended.height), fill=_LEGEND_STRIP_BG)
    draw.line((0, top, extended.width, top), fill=(0, 0, 0, 255))
    font = _font(11)
    cy = top + OVERLAY_LEGEND_PX / 2.0

    # (style, swatch color(s), label); all nine elements the overlay can draw.
    items: list[tuple[str, Any, str]] = [
        ("road", (COLORS["authored_major"], COLORS["authored_center"],
                  COLORS["authored_edge"]), "authored street"),
        ("road", (COLORS["alley"], COLORS["alley_center"], COLORS["authored_edge"]),
         "authored alley"),
        ("poly", (COLORS["plaza"], (255, 219, 130, 205)), "plaza"),
        ("poly", (COLORS["court"], (136, 219, 151, 220)), "court"),
        ("poly", (COLORS["footprint_karthgad"], COLORS["footprint_outline"]),
         "Karthgad footprint"),
        ("poly", (COLORS["footprint_markarth"], COLORS["footprint_outline"]),
         "Markarth footprint"),
        ("door", (COLORS["door_public"],), "door public"),
        ("door", (COLORS["door_service"],), "door service"),
        ("door", (COLORS["door_unconnected"],), "door unconnected"),
        ("door", (COLORS["door_faces_away"],), "door faces away (>60 deg)"),
        ("poly", (COLORS["overlap_bad"], COLORS["overlap_bad_outline"]),
         "overlap (hard error)"),
    ]
    cursor = 14.0
    for style, colours, label in items:
        swatch_w = 24.0 if style == "road" else 18.0
        if style == "road":
            # Blend every road layer over the band: the overlay composites
            # translucent layers, but PIL draws raw RGBA, so raw swatches
            # would display against the viewer background, not this band.
            fill, centre, edge = colours
            fill = _blend_over(fill, _LEGEND_STRIP_BG)
            centre = _blend_over(centre, _LEGEND_STRIP_BG)
            edge = _blend_over(edge, _LEGEND_STRIP_BG)
            _draw_styled_road(draw, [(cursor, cy), (cursor + swatch_w, cy)], 4,
                              fill, centre, edge)
        elif style == "poly":
            fill, outline = colours
            draw.rectangle((cursor, cy - 7.0, cursor + swatch_w, cy + 7.0),
                           fill=_blend_over(fill, _LEGEND_STRIP_BG),
                           outline=_blend_over(outline, _LEGEND_STRIP_BG), width=1)
        else:
            _door_arrow(draw, (cursor + 4.0, cy), 0.0, 16.0, colours[0])
            swatch_w = 26.0
        cursor += swatch_w + 8.0
        draw.text((cursor, cy), label, font=font, fill=(214, 220, 226),
                  anchor="lm", stroke_width=1, stroke_fill=(8, 16, 20, 230))
        box = draw.textbbox((0, 0), label, font=font)
        cursor += (box[2] - box[0]) + 22.0
    return extended


def _door_arrow(draw: ImageDraw.ImageDraw, start_px: tuple[float, float],
                heading_deg: float, length_px: float, colour: tuple[int, int, int, int]) -> None:
    """Door arrow in the same visual convention as the symbols renderer."""
    angle = math.radians(float(heading_deg))
    tip = (start_px[0] + length_px * math.cos(angle),
           start_px[1] - length_px * math.sin(angle))
    draw.line((start_px[0], start_px[1], tip[0], tip[1]), fill=(9, 20, 24, 245), width=10)
    draw.line((start_px[0], start_px[1], tip[0], tip[1]), fill=colour, width=6)
    head = max(10.0, min(16.0, length_px * 0.28))
    back = math.atan2(-(tip[1] - start_px[1]), tip[0] - start_px[0])
    left = (tip[0] - head * math.cos(back - 0.52), tip[1] + head * math.sin(back - 0.52))
    right = (tip[0] - head * math.cos(back + 0.52), tip[1] + head * math.sin(back + 0.52))
    draw.polygon([tip, left, right], fill=(9, 20, 24, 245))
    inner = max(6.0, head - 4.0)
    left = (tip[0] - inner * math.cos(back - 0.52), tip[1] + inner * math.sin(back - 0.52))
    right = (tip[0] - inner * math.cos(back + 0.52), tip[1] + inner * math.sin(back + 0.52))
    draw.polygon([tip, left, right], fill=colour)
    draw.ellipse((start_px[0] - 7, start_px[1] - 7, start_px[0] + 7, start_px[1] + 7),
                 fill=COLORS["door"], outline=COLORS["door_outline"], width=2)


def _plan_to_world(point: Sequence[float], origin: Sequence[float]) -> tuple[float, float]:
    """Plan-frame GU -> TES3 world GU (world = plan + survey origin)."""
    return (point[0] + origin[0], point[1] + origin[1])


def _door_fact_map(door_facts: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Index door facts by ``(lot_id, door_id)``.

    Different lots reusing the same stamp share source door ids, so door id
    alone cannot key viability; the lot-scoped key is the render contract
    (plan §5).
    """
    return {(str(fact.get("lot_id")), str(fact.get("door_id"))): fact
            for fact in door_facts}


def _door_viability_colour(fact: Mapping[str, Any] | None,
                           door: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Arrow colour from the lot's own fact (unconnected, then facing cone).

    Unconnected doors render red (colour read from the fact's ``unconnected``
    value, never inferred from a preserved target id); connected doors beyond
    the facing cone render magenta; otherwise the door's intent colour.
    """
    if fact is None:
        unconnected = not door.get("target_id")
    else:
        unconnected = bool(fact.get("unconnected"))
    if unconnected:
        return COLORS["door_unconnected"]
    deviation = fact.get("facing_deviation_deg") if fact is not None else None
    if deviation is not None and float(deviation) > FACING_CONE_DEG:
        return COLORS["door_faces_away"]
    return COLORS.get(f"door_{door.get('intent', 'public')}", COLORS["door_public"])


def render_composite(plan: Mapping[str, Any], sketch: Mapping[str, Any], site: Mapping[str, Any],
                     terrain: TerrainBundle, network: Any,
                     geometry: Mapping[str, Mapping[str, Any]],
                     out_path: Path,
                     overlap_polys: Sequence[Sequence[Sequence[float]]] = (),
                     door_facts: Sequence[Mapping[str, Any]] = ()) -> tuple[Image.Image, list[Any]]:
    """Render plan.png: identical base canvas + sketch-derived overlay."""
    world_rect = [float(value) for value in site["rectangle_gu"]]
    base, projection = render_planning_canvas(
        terrain, world_rect, network, site_name=site["site_name"],
        title=_canvas_title(site))
    records = resolve_stamps(plan, geometry)
    scale = projection.map_width_px / max(projection.width_gu, 1.0)
    origin = terrain.origin_gu
    to_px = projection.world_to_px

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Authored streets vs alleys (distinct colors and widths).
    for road in plan.get("authored_roads", []):
        points = [to_px(_plan_to_world(p, origin)) for p in road["polyline_plan_gu"]]
        width_px = max(4, int(round(float(road["width_gu"]) * scale)))
        _draw_styled_road(draw, points, width_px, COLORS["authored_major"],
                          COLORS["authored_center"], COLORS["authored_edge"])
    for alley in plan.get("alleys", []):
        points = [to_px(_plan_to_world(p, origin)) for p in alley["polyline_plan_gu"]]
        width_px = max(3, int(round(float(alley["width_gu"]) * scale)))
        _draw_styled_road(draw, points, width_px, COLORS["alley"],
                          COLORS["alley_center"], COLORS["authored_edge"])

    # Plaza vs court polygons (translucent fills + outlines).
    for region in plan.get("road_surface_polygons", []):
        points = [to_px(_plan_to_world(p, origin)) for p in region["polygon_plan_gu"]]
        _composite_polygon(overlay, points, COLORS["plaza"],
                           outline=(255, 219, 130, 205), width=2)
    for court in plan.get("shared_courts", []):
        points = [to_px(_plan_to_world(p, origin)) for p in court["polygon_plan_gu"]]
        _composite_polygon(overlay, points, COLORS["court"],
                           outline=(136, 219, 151, 220), width=2)

    # Yawed footprint hulls, kit-colored, with lot-id labels.
    label_font = _font(10)
    forbidden: list[tuple[float, float, float, float]] = []
    for record in records:
        if not record.hull:
            continue
        hull_px = [to_px(_plan_to_world(p, origin)) for p in record.hull]
        colour = (COLORS["footprint_karthgad"] if "karthgad" in record.kit.casefold()
                  else COLORS["footprint_markarth"])
        _composite_polygon(overlay, hull_px, colour, outline=COLORS["footprint_outline"], width=2)
        forbidden.append(_rect_from_points(hull_px, pad=4.0))

    # Hard-error overlap regions painted red ON TOP of the footprints: the
    # exact intersecting area (building<->building or building<->corridor),
    # so the designer sees precisely what must move.
    for poly in overlap_polys:
        points = [to_px(p) for p in poly]
        if len(points) >= 3:
            _composite_polygon(overlay, points, COLORS["overlap_bad"],
                               outline=COLORS["overlap_bad_outline"], width=3)

    # Door arrows by intent; unconnected doors flagged in a distinct color
    # and connected-but-sideways doors in a fourth color (facing deviation).
    # Facts and viability colour are keyed by (lot_id, door_id), not door id
    # alone, because different lots using the same stamp share door ids.
    fact_map = _door_fact_map(door_facts)
    arrow_data: list[tuple[str, dict[str, Any], tuple[float, float]]] = []
    for record in records:
        for door in record.doors:
            start = to_px(_plan_to_world(door["position_plan_gu"], origin))
            arrow_data.append((record.lot_id, door, start))
            tip = (start[0] + 44.0 * math.cos(math.radians(float(door["heading_deg"]))),
                   start[1] - 44.0 * math.sin(math.radians(float(door["heading_deg"]))))
            forbidden.append(_rect_from_points([start, tip], pad=8.0))
    for lot_id, door, start in arrow_data:
        fact = fact_map.get((str(lot_id), str(door.get("door_id"))))
        colour = _door_viability_colour(fact, door)
        _door_arrow(draw, start, float(door["heading_deg"]), 44.0, colour)

    # Lot-id labels through the deterministic collision gate.
    labeler = LabelPlacer(draw, (0, projection.title_band_px,
                                 projection.map_width_px,
                                 projection.title_band_px + projection.map_height_px),
                          forbidden)
    for record in records:
        if not record.hull:
            continue
        hull_px = [to_px(_plan_to_world(p, origin)) for p in record.hull]
        bbox = _rect_from_points(hull_px)
        labeler.place(record.lot_id, (bbox[0], bbox[1] - 22), COLORS["annotation"],
                      label_font, candidates=((bbox[0], bbox[1] - 22), (bbox[2] + 6, bbox[1]),
                                              (bbox[0], bbox[3] + 6), (bbox[0] - 120, bbox[1])),
                      leader_from=(bbox[0], bbox[1]))
    base.alpha_composite(overlay)
    base = _extend_overlay_legend(base)
    base.save(out_path, format="PNG", compress_level=6)
    return base, records


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_bundle(bundle_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Path]]:
    """Load bundle artifacts; returns (site, stamps, manifest, bundle_files)."""
    site_path = bundle_dir / "site.json"
    stamps_path = bundle_dir / "stamps.json"
    manifest_path = bundle_dir / "bundle_manifest.json"
    canvas_path = bundle_dir / "canvas.png"
    for path in (site_path, stamps_path, manifest_path, canvas_path):
        if not path.is_file():
            raise ValueError(f"bundle is missing required file: {path}")
    site = _load_json(site_path, "site.json")
    stamps = _load_json(stamps_path, "stamps.json")
    manifest = _load_json(manifest_path, "bundle_manifest.json")
    if not isinstance(site, dict) or not isinstance(stamps, dict) or not isinstance(manifest, dict):
        raise ValueError("bundle site.json/stamps.json/bundle_manifest.json must be objects")
    bundle_files = sorted(
        path for path in bundle_dir.iterdir() if path.is_file())
    return site, stamps, manifest, bundle_files


def load_products(manifest: Mapping[str, Any]) -> tuple[TerrainBundle, Any, dict[str, dict[str, Any]]]:
    """Load terrain/network/stamp geometry via the paths pinned in the manifest."""
    inputs = manifest.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("bundle_manifest.json has no inputs map")
    resolved: dict[str, Path] = {}
    for raw, _ in inputs.items():
        path = (ROOT / str(raw)).resolve()
        if not path.is_file():
            raise ValueError(f"manifest input is missing: {path}")
        resolved[raw] = path
    survey = next((path for raw, path in resolved.items() if raw.endswith("site_survey.json")), None)
    fields = next((path for raw, path in resolved.items() if raw.endswith("survey_fields.npz")), None)
    roads_dir = next((path.parent for raw, path in resolved.items()
                      if raw.endswith("tamriel_aligned_centerlines_v1.json")), None)
    libraries = [path for raw, path in sorted(resolved.items())
                 if "/stamps/" in raw.replace("\\", "/") and raw.endswith(".json")
                 and "catalog" not in raw]
    if survey is None or fields is None or roads_dir is None or not libraries:
        raise ValueError("manifest inputs do not pin survey/fields/roads/stamp libraries")
    terrain = TerrainBundle.from_paths(survey, fields)
    network = aligned_roads.load_aligned_network(roads_dir)
    geometry = load_stamp_geometry(tuple(libraries))
    return terrain, network, geometry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="planning bundle directory (canvas.png, site.json, stamps.json, manifest)")
    parser.add_argument("--sketch", type=Path, required=True,
                        help="sketch v1 JSON (world GU coordinates)")
    parser.add_argument("--out", type=Path, required=True,
                        help="fresh output directory (refused when non-empty or under a data root)")
    parser.add_argument("--auto-face", action="store_true",
                        help="rotate every lot so its primary door faces the nearest "
                             "circulation target before deriving (mechanical helper; "
                             "the adjusted sketch is written as sketch.copy.json)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    try:
        site, stamps, manifest, bundle_files = load_bundle(args.bundle)
        stamp_ids = {entry.get("id") for entry in stamps.get("stamps", [])}
        sketch_raw = _load_json(args.sketch, "sketch")
        terrain, network, geometry = load_products(manifest)
        door_ids_by_stamp = {
            str(stamp_id): {
                str(member.get("source_id")) for member in stamp.get("members", [])
                if isinstance(member, Mapping) and member.get("is_door")
                and isinstance(member.get("source_id"), str)
            }
            for stamp_id, stamp in geometry.items()
            if isinstance(stamp, Mapping)
        }
        target_ids = {
            str(row.get("edge_id")) for row in site.get("source_roads", [])
            if isinstance(row, Mapping) and isinstance(row.get("edge_id"), str)
        }
        if isinstance(sketch_raw, Mapping):
            target_ids.update(
                str(row.get("id")) for row in sketch_raw.get("roads", [])
                if isinstance(row, Mapping) and isinstance(row.get("id"), str)
            )
            target_ids.update(
                str(row.get("id")) for row in sketch_raw.get("spaces", [])
                if isinstance(row, Mapping) and isinstance(row.get("id"), str)
            )
        sketch = validate_sketch(sketch_raw, site["site_name"], stamp_ids,
                                 site["rectangle_gu"], door_ids_by_stamp,
                                 target_ids)
        refuse_unless_fresh(args.out)
        if args.auto_face:
            sketch = apply_auto_face(sketch, site, geometry, network,
                                     terrain.origin_gu)
        plan, door_facts = derive_plan(sketch, site, stamps, geometry, network, terrain)
        require_valid_extension(plan)
        rectangle = terrain.rectangle(
            world_bounds_gu=site["rectangle_gu"],
            context_margin_gu=float(site["margin_gu"]), full_site_inset=False)
        advisory = analyze_plan(plan, terrain, rectangle, aligned_network=network,
                                stamp_geometry=geometry)
        targets = build_target_map(site, terrain.origin_gu, network)
        targets.update(_authored_targets(sketch, terrain.origin_gu))
        records_probe = resolve_stamps(plan, geometry)
        overlap_polys = _overlap_polys_world(advisory, records_probe, targets,
                                             terrain.origin_gu)
        image, records = render_composite(plan, sketch, site, terrain, network,
                                          geometry, args.out / "plan.png",
                                          overlap_polys=overlap_polys,
                                          door_facts=door_facts)
        checks = build_checks(plan, sketch, advisory, door_facts, records)
        _write_json(args.out / "visual_plan.json", plan)
        _write_json(args.out / "checks.json", checks)
        _write_json(args.out / "sketch.copy.json", sketch)
        elapsed_ms = int(round((time.perf_counter() - started) * 1000.0))
        log = {
            "schema_version": 1,
            "kind": "cityforge_plan_sketch_run",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "site_name": site["site_name"],
            "sketch": str(args.sketch),
            "bundle_dir": str(args.bundle.resolve()),
            "sha256": {
                "bundle": {path.name: _sha256(path) for path in bundle_files},
                "sketch": _sha256(args.sketch),
                "outputs": {name: _sha256(args.out / name)
                            for name in ("visual_plan.json", "checks.json",
                                         "plan.png", "sketch.copy.json")},
            },
            "wall_clock_ms": elapsed_ms,
            "canvas_size_px": list(image.size),
            "hard_error_count": len(advisory["hard_errors"]),
        }
        _write_json(args.out / "log.json", log)
        print(f"site: {site['site_name']}  plan: {plan['plan_id']}")
        print(f"hard_errors: {len(advisory['hard_errors'])}  "
              f"wall_clock_ms: {elapsed_ms}")
        print(f"outputs: {args.out.resolve()}")
        return 0 if not advisory["hard_errors"] else 1
    except SketchError as exc:
        print(f"FAILURE: sketch {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - exact failure is the CLI contract
        print(f"FAILURE: plan_sketch {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
