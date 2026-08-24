"""Plazas, parks, and keep-forecourt open spaces for V2 townlayout (Phase 14).

Purpose
-------
Keep the Phase 7 market plaza as a protected frontage target.  If a keep
anchor exists, inset its patch by ``KEEP_FORECOURT_INSET_GU`` and reserve
the ring as a plaza.  Courts wait for Phase 16 ``court_probability``.
Parks are not in the Falkreath v1 mix.

Inputs
------
A Phase 13 candidate (blocks + wards + anchors + existing open_spaces)
and the TownBrief.

Outputs
-------
Updated ``open_spaces``, optional ``keep_buildable`` inner polygon, and a
``protected_space_ids`` list that later parcel subdivision must not
overwrite.

Pipeline position
-----------------
V2 townlayout Phase 14 open spaces; no parcels/VTEX.
"""

from __future__ import annotations

from typing import Any, Optional

from shapely.geometry import LineString, Polygon

from .constants import KEEP_FORECOURT_INSET_GU
from .geometry import normalize_ring, polygon_from_ring
from .validate import TownLayoutError


def _ring(poly: Polygon) -> list[list[float]]:
    return normalize_ring([[c[0], c[1]] for c in poly.exterior.coords])["ring"]


def _largest_polygon(geom) -> Optional[Polygon]:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area > 0 else None
    if geom.geom_type == "MultiPolygon":
        parts = [g for g in geom.geoms if g.geom_type == "Polygon" and g.area > 0]
        if not parts:
            return None
        parts.sort(key=lambda g: g.area, reverse=True)
        return parts[0]
    return None


def _as_simple_polygons(geom) -> list[Polygon]:
    """Cover ``geom`` with simple (no-hole) polygons.  1 GU slits only."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "MultiPolygon":
        parts = []
        for item in geom.geoms:
            parts.extend(_as_simple_polygons(item))
        return parts
    if geom.geom_type != "Polygon" or geom.area <= 0:
        return []
    if not geom.interiors:
        return [geom]
    remaining = geom
    for hole in geom.interiors:
        cut = LineString([hole.coords[0], geom.exterior.coords[0]]).buffer(
            1.0, cap_style=2, join_style=2)
        remaining = remaining.difference(cut)
    return _as_simple_polygons(remaining)


def protected_open_space_ids(candidate: dict) -> list[str]:
    """IDs of plazas/courts/parks that parcel subdivision must not overwrite."""
    ids = []
    for space in candidate.get("open_spaces") or []:
        if space.get("kind") in ("plaza", "court", "park"):
            ids.append(space["space_id"])
    return ids


def finalize_open_spaces(
    candidate: dict,
    town_brief: dict,
    *,
    candidate_id: str = "c00",
) -> dict[str, Any]:
    """Protect plazas and optionally cut a keep forecourt ring."""
    spaces = list(candidate.get("open_spaces") or [])
    reports = list(candidate.get("reports") or [])
    keep_buildable = None
    plaza_ids = {s["space_id"] for s in spaces if s.get("kind") == "plaza"}

    if town_brief.get("ward_mix", {}).get("park"):
        reports.append({
            "stage": "open_spaces",
            "status": "ok",
            "message": "parks skipped (not in Falkreath v1 mix)",
        })

    court_requested = any(
        float(w.get("court_probability") or 0.0) > 0.0
        for w in (candidate.get("wards") or [])
        if isinstance(w, dict)
    )
    if court_requested:
        reports.append({
            "stage": "open_spaces",
            "status": "ok",
            "message": "courts deferred to Phase 16 ward grammar",
        })

    keep_anchor = next(
        (a for a in (candidate.get("anchors") or []) if a.get("kind") == "keep"),
        None,
    )
    if keep_anchor is not None:
        poly = polygon_from_ring(keep_anchor["polygon"])
        inner = _largest_polygon(
            poly.buffer(-KEEP_FORECOURT_INSET_GU, join_style=2, cap_style=2))
        if inner is None:
            keep_mode = town_brief.get("anchors", {}).get("keep", "absent")
            if keep_mode == "required":
                raise TownLayoutError(
                    "invalid_polygon: keep patch too small for 200 GU forecourt")
            reports.append({
                "stage": "open_spaces",
                "status": "ok",
                "message": "keep forecourt skipped (inset empty)",
            })
            keep_buildable = {
                "block_id": f"keep_block_{candidate_id}",
                "patch_id": keep_anchor["patch_id"],
                "polygon": keep_anchor["polygon"],
            }
        else:
            ring_parts = _as_simple_polygons(poly.difference(inner))
            keep_buildable = {
                "block_id": f"keep_block_{candidate_id}",
                "patch_id": keep_anchor["patch_id"],
                "polygon": _ring(inner),
            }
            added = 0
            for seq, part in enumerate(ring_parts):
                if part.area <= 1.0:
                    continue
                space_id = f"space_{candidate_id}_keep_forecourt_{seq:04d}"
                spaces.append({
                    "space_id": space_id,
                    "kind": "plaza",
                    "polygon": _ring(part),
                })
                plaza_ids.add(space_id)
                added += 1
            if added:
                reports.append({
                    "stage": "open_spaces",
                    "status": "ok",
                    "message": (
                        f"keep forecourt parts={added} "
                        f"inset={KEEP_FORECOURT_INSET_GU:g} GU"
                    ),
                })

    if not plaza_ids and town_brief.get("anchors", {}).get("market") == "required":
        raise TownLayoutError("missing_anchor: required market plaza missing")

    reports.append({
        "stage": "open_spaces",
        "status": "ok",
        "message": (
            f"plazas={sum(1 for s in spaces if s.get('kind') == 'plaza')} "
            f"courts={sum(1 for s in spaces if s.get('kind') == 'court')} "
            f"parks={sum(1 for s in spaces if s.get('kind') == 'park')} "
            f"verges={sum(1 for s in spaces if s.get('kind') == 'verge')}"
        ),
    })
    out = dict(candidate)
    out["open_spaces"] = spaces
    out["protected_space_ids"] = [
        s["space_id"] for s in spaces if s.get("kind") in ("plaza", "court", "park")
    ]
    if keep_buildable is not None:
        out["keep_buildable"] = keep_buildable
    out["reports"] = reports
    return out
