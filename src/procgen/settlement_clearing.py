"""settlement_clearing.py — build the city ``clearing`` document (Stage 1).

Purpose
-------
Given an accepted Cityforge town layout and its world-seated placements,
compute the exact 2-D exclusion geometry that scatter and groundcover
generators must avoid inside a town: buffered building footprints, exact
circulation surfaces, and road centreline corridors.  This is the shared,
authoritative ``clearing`` contract that both the scatter integration (plan
Stage 3) and the groundcover integration (plan Stage 4) consume, so seeds,
bounds and exclusions cannot drift apart between the two emitted plugins.

The clearing is a *pure function of its two inputs* (the layout and the
seated placements) expressed in absolute **plan game units (GU)**, 2-D only
(no terrain sampling).  It does not author TES3 and does not touch plugins.

Pipeline position
-----------------
Approved plan: ``.opencode/runs/cityforge-scatter-groundcover-integration/
plan.md``, Stage 1.  Consumed by ``procgen.clearing_index`` (Stage 2) and by
the scatter/groundcover generators (Stages 3-4) and the unified CLI (Stage 5).

Inputs
------
* ``city_layout`` (Mapping): the accepted layout document, e.g.
  ``output/cityforge/townlayout/falkreath_phase21_city80/r14_dense_final3/
  13_city_layout/city_layout.json``.  Required keys:
    - ``placements``: list of placed stamps; each has ``placement_id`` or
      ``parcel_id``, ``stamp_id``, and ``hull`` (the world-transformed
      footprint ring in absolute plan GU).
    - ``roads``: list with ``road_id``, ``polyline`` (list of ``[x, y]``) and
      ``clear_width_gu``.
    - ``surfaces``: list with ``surface_id`` and ``polygon`` (list of rings,
      each ring a list of ``[x, y]``; ring 0 is the exterior, later rings are
      holes).
    - ``city_domain``: the city boundary ring (list of ``[x, y]``, may be
      unclosed).
* ``town_placements``: the realized / world-seated placements document
  (contract per plan; realisation stages pending).  Best-effort source of the
  plan-frame origin (``frame_origin_gu``) used to map TES3 cell grid
  coordinates into plan GU for the city-domain rasterization rule.  May also
  be a list.  Only used for frame-origin discovery and (when the expected
  shape is present) placement cross-validation; the building geometry itself
  is sourced from ``city_layout``.

Output
------
A ``ClearingDoc`` (a JSON-serialisable dict) with the documented schema below,
ready to be written as ``settlement_clearing.json``.

Invariants
----------
* All geometry is absolute plan GU, 2-D, no terrain sampling.
* Deterministic ordering: building/surface/road exclusion lists are sorted by
  their ``source_id``.
* Building footprints are the layout's world ``hull`` (which is exactly
  ``footprint.hull_xy_rel`` transformed by ``Rz(+yaw_deg)`` about the world
  anchor and translated to the anchor — verified numerically against the
  v2 stamp libraries) buffered outward by ``margin_gu``.
* Surfaces are taken exactly (no buffer).  Roads are stored as centreline
  polylines with a baked ``half_width_gu = clear_width_gu / 2 + margin_gu``;
  the buffer is applied at query time so the stored geometry stays a
  centreline.

Schema (schema_version 1, units GU)
-----------------------------------
    {
      "schema_version": 1, "units": "gu", "margin_gu": 256,
      "frame_origin_gu": [x, y] | null,
      "city_domain": [[x, y], ...],
      "building_exclusions": [
        {"kind": "polygon", "rings": [[[x, y], ...], ...], "source_id": "..."}
      ],
      "surface_exclusions": [
        {"kind": "polygon", "rings": [[[x, y], ...], ...], "source_id": "..."}
      ],
      "road_exclusions": [
        {"kind": "polyline", "points": [[x, y], ...], "half_width_gu": 512.0}
      ]
    }

Notes on the approved plan schema (documented deviations)
---------------------------------------------------------
* The plan example shows a single ``ring`` per polygon.  Real surface
  polygons contain holes (11/47 in the accepted Falkreath layout have two
  rings), so polygon geometry is stored as ``rings`` (a list of rings: ring 0
  exterior, later rings holes), matching the plan text "surfaces[].polygon
  rings".  Buildings are convex hulls and contribute exactly one ring.
* ``surface_exclusions`` and ``road_exclusions`` carry a ``source_id``
  (surface_id / road_id) for determinism and diagnostic traceability; the
  plan example only shows it on building exclusions.
* ``frame_origin_gu`` is an added top-level field (absent from the plan
  example) because ``ClearingIndex.in_city_domain`` needs the cell-grid ->
  plan-GU mapping, which is settlement-specific.  It is read from
  ``town_placements`` when present (null otherwise).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from shapely.geometry import MultiPolygon, Polygon

DEFAULT_MARGIN_GU = 256.0
SCHEMA_VERSION = 1
UNITS = "gu"


class ClearingDoc(dict):
    """Typed alias for the clearing document (a plain JSON dict)."""

    pass


def _is_ring(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) < 3:
        return False
    for pt in value:
        if not isinstance(pt, Sequence) or isinstance(pt, (str, bytes)) or len(pt) < 2:
            return False
        try:
            float(pt[0])
            float(pt[1])
        except (TypeError, ValueError):
            return False
    return True


def _closed_ring(ring: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return a closed ring (first point appended) as a list of ``[x, y]``."""
    out = [[float(pt[0]), float(pt[1])] for pt in ring]
    if out and out[0] != out[-1]:
        out.append(list(out[0]))
    return out


def _is_polyline(value: object) -> bool:
    """A polyline is a sequence of >= 2 ``[x, y]`` points (open, not a ring)."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) < 2:
        return False
    for pt in value:
        if not isinstance(pt, Sequence) or isinstance(pt, (str, bytes)) or len(pt) < 2:
            return False
        try:
            float(pt[0])
            float(pt[1])
        except (TypeError, ValueError):
            return False
    return True


def _rings_to_polygon(rings: Sequence[Sequence[Sequence[float]]]) -> Polygon:
    """Build a Shapely polygon from ring 0 (exterior) + holes (later rings)."""
    if not rings:
        return Polygon()
    exterior = _closed_ring(rings[0])
    holes = [_closed_ring(r) for r in rings[1:]]
    try:
        return Polygon(exterior, holes)
    except Exception as exc:  # shapely raises for self-intersecting etc.
        raise ValueError(f"invalid polygon rings for clearing: {exc}") from exc


def _polygon_rings(poly: Polygon) -> list[list[list[float]]]:
    """Extract a Shapely polygon (or multipolygon) back into ring lists."""
    polys: list[Polygon] = []
    if isinstance(poly, MultiPolygon):
        polys = list(poly.geoms)
    elif not poly.is_empty:
        polys = [poly]
    rings: list[list[list[float]]] = []
    for p in polys:
        rings.append(_closed_ring(list(p.exterior.coords)))
        for interior in p.interiors:
            rings.append(_closed_ring(list(interior.coords)))
    return rings


def _extract_frame_origin(town_placements: object) -> list[float] | None:
    """Best-effort plan-frame origin in world GU from the seated document.

    The seated/realisation document carries ``terrain_field.frame_origin_gu``
    (e.g. ``[-778240.0, -90112.0]`` for the accepted Falkreath layout = cell
    (-95, -11)).  Returns ``None`` when unavailable.
    """
    if isinstance(town_placements, Mapping):
        tf = town_placements.get("terrain_field")
        if isinstance(tf, Mapping):
            origin = tf.get("frame_origin_gu")
            if isinstance(origin, Sequence) and not isinstance(origin, (str, bytes)) and len(origin) >= 2:
                try:
                    return [float(origin[0]), float(origin[1])]
                except (TypeError, ValueError):
                    pass
        origin = town_placements.get("frame_origin_gu")
        if isinstance(origin, Sequence) and not isinstance(origin, (str, bytes)) and len(origin) >= 2:
            try:
                return [float(origin[0]), float(origin[1])]
            except (TypeError, ValueError):
                pass
    return None


def _seated_placements(town_placements: object) -> list[Mapping[str, Any]]:
    """Normalise town_placements to a list of placement mappings (best-effort)."""
    if isinstance(town_placements, Mapping):
        rows = town_placements.get("placements")
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, Mapping)]
        return []
    if isinstance(town_placements, list):
        return [r for r in town_placements if isinstance(r, Mapping)]
    return []


def build_clearing(
    city_layout: Mapping[str, Any],
    town_placements: object,
    *,
    margin_gu: float = DEFAULT_MARGIN_GU,
    wall: Mapping[str, Any] | None = None,
) -> ClearingDoc:
    """Build the settlement clearing document (see module docstring)."""
    if not isinstance(city_layout, Mapping):
        raise ValueError("city_layout must be a mapping")
    margin = float(margin_gu)
    if not (margin >= 0.0):
        raise ValueError(f"margin_gu must be >= 0, got {margin_gu!r}")

    placements = city_layout.get("placements")
    if not isinstance(placements, list):
        raise ValueError("city_layout has no 'placements' list")
    roads = city_layout.get("roads")
    if not isinstance(roads, list):
        raise ValueError("city_layout has no 'roads' list")
    surfaces = city_layout.get("surfaces")
    if not isinstance(surfaces, list):
        raise ValueError("city_layout has no 'surfaces' list")
    city_domain = city_layout.get("city_domain")
    if not _is_ring(city_domain):
        raise ValueError("city_layout.city_domain must be a list of [x, y] points")

    # --- building exclusions: world hull buffered by margin ---
    building_exclusions: list[dict[str, Any]] = []
    for p in placements:
        hull = p.get("hull")
        if not _is_ring(hull):
            raise ValueError(f"placement {p.get('placement_id')!r} has invalid 'hull'")
        source_id = str(
            p.get("placement_id") or p.get("parcel_id") or p.get("stamp_id") or ""
        )
        poly = Polygon(_closed_ring(hull))
        if poly.is_empty:
            raise ValueError(f"placement {source_id!r} produced an empty footprint hull")
        buffered = poly.buffer(margin)
        rings = _polygon_rings(buffered)
        building_exclusions.append(
            {"kind": "polygon", "rings": rings, "source_id": source_id}
        )

    # --- surface exclusions: exact polygons (no buffer) ---
    surface_exclusions: list[dict[str, Any]] = []
    for s in surfaces:
        rings_in = s.get("polygon")
        if not isinstance(rings_in, list) or not rings_in or not _is_ring(rings_in[0]):
            raise ValueError(f"surface {s.get('surface_id')!r} has invalid 'polygon'")
        source_id = str(s.get("surface_id") or "")
        poly = _rings_to_polygon(rings_in)
        rings = _polygon_rings(poly)
        surface_exclusions.append(
            {"kind": "polygon", "rings": rings, "source_id": source_id}
        )
    if wall is not None:
        from .wall_scatter import wall_surface_exclusions

        surface_exclusions.extend(wall_surface_exclusions(wall))

    # --- road exclusions: centreline + baked half width ---
    road_exclusions: list[dict[str, Any]] = []
    for r in roads:
        polyline = r.get("polyline")
        if not _is_polyline(polyline):
            raise ValueError(f"road {r.get('road_id')!r} has invalid 'polyline'")
        clear_width = r.get("clear_width_gu")
        if clear_width is None:
            raise ValueError(f"road {r.get('road_id')!r} has no 'clear_width_gu'")
        half_width = float(clear_width) / 2.0 + margin
        road_exclusions.append(
            {
                "kind": "polyline",
                "points": [[float(pt[0]), float(pt[1])] for pt in polyline],
                "half_width_gu": half_width,
                "source_id": str(r.get("road_id") or ""),
            }
        )

    # --- deterministic ordering ---
    building_exclusions.sort(key=lambda e: e["source_id"])
    surface_exclusions.sort(key=lambda e: e["source_id"])
    road_exclusions.sort(key=lambda e: e["source_id"])

    # --- city domain (closed) + frame origin ---
    domain_closed = _closed_ring(city_domain)
    frame_origin = _extract_frame_origin(town_placements)

    # --- best-effort placement cross-validation (only when shape matches) ---
    seated = _seated_placements(town_placements)
    if seated:
        seated_by_id: dict[str, Mapping[str, Any]] = {}
        for row in seated:
            pid = row.get("placement_id")
            if pid is not None:
                seated_by_id[str(pid)] = row
        for p in placements:
            pid = str(p.get("placement_id") or p.get("parcel_id") or "")
            srow = seated_by_id.get(pid)
            if srow is None:
                continue
            layout_anchor = p.get("anchor")
            world_anchor = srow.get("anchor_world_gu")
            if (
                isinstance(layout_anchor, Sequence)
                and len(layout_anchor) >= 2
                and isinstance(world_anchor, Sequence)
                and len(world_anchor) >= 2
            ):
                dx = abs(float(layout_anchor[0]) - float(world_anchor[0]))
                dy = abs(float(layout_anchor[1]) - float(world_anchor[1]))
                if max(dx, dy) > 1.0:
                    raise ValueError(
                        f"placement {pid} anchor drift between layout and seated "
                        f"placements: plan ({layout_anchor[0]:.3f}, "
                        f"{layout_anchor[1]:.3f}) vs world "
                        f"({world_anchor[0]:.3f}, {world_anchor[1]:.3f})"
                    )
            layout_stamp = p.get("stamp_id")
            seated_stamp = srow.get("stamp_id")
            if (
                layout_stamp is not None
                and seated_stamp is not None
                and str(layout_stamp) != str(seated_stamp)
            ):
                raise ValueError(
                    f"placement {pid} stamp_id mismatch: layout {layout_stamp!r} "
                    f"vs seated {seated_stamp!r}"
                )

    doc = ClearingDoc(
        schema_version=SCHEMA_VERSION,
        units=UNITS,
        margin_gu=margin,
        frame_origin_gu=frame_origin,
        city_domain=domain_closed,
        building_exclusions=building_exclusions,
        surface_exclusions=surface_exclusions,
        road_exclusions=road_exclusions,
    )
    return doc


def dump_clearing(doc: ClearingDoc, path: str | object) -> None:
    """Serialise a clearing document to JSON (utf-8, newline-terminated)."""
    import os

    from pathlib import Path

    target = Path(os.fspath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    target.write_text(payload, encoding="utf-8")
