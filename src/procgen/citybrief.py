"""Cityforge D-BRIEF stamp census and kit-brief engine (T0.5, dispatch 5).

Pipeline position
------------------
Consumes the two hash-pinned D-STAMP libraries
(``output/cityforge/stamps/karthgad_nord_v1.json`` and
``markarth_side_stone_v1.json``), the accepted final Markarth terrain-backed
palette (``.../final-markarth-extraction-2026-08-10-library/stamp_palette_v1/
catalog.json`` + ``render_library_manifest.json``), and the measured Karthgad
survey artifacts, and produces the planner-facing *kit brief* plus the raw
measured stamp/spacing/door-step census vectors.  Sits one stage upstream of
plan validation (D-PLAN consumes ``kit_brief.json`` enums); the CLI
``tools/cityforge/build_city_brief.py`` writes the four canonical outputs.

Invariants
----------
- Eligibility: 54 stamps = 11 Karthgad + 44 Markarth - the user-reported
  defective Castle Barracks stamp (``markarth_side_v1__u114_castle_barracks``,
  exact reason ``user-reported defective Castle Barracks extraction``).  The
  barracks is excluded BEFORE any type/count/quantile/spacing computation and
  appears exactly once in the exclusion ledger.
- Preview resolution: every eligible Markarth stamp resolves by split-record
  identity (``source.unit_id`` == palette ``split_record_id``) to exactly one
  eligible ``building_unit`` palette entry; ambiguity or a missing eligible
  preview is a hard failure.  Karthgad previews are hash-verified against
  their source paths and cross-checked with ``catalog_v1/index.json``.
- Spacing: source-world footprints are reconstructed as
  ``anchor.source_position_gu`` + ``footprint.hull_xy_rel``; boundary gaps are
  measured only within one source run (never Karthgad<->Markarth); pairs that
  intersect or touch are separated (gap exactly 0.0) from positive gaps.
- Every number carries provenance to a hash-pinned file, source record, or
  exact survey section/line range.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .censusio import (
    PinnedFile,
    quantile_summary,
    sha256_file,
)

# --------------------------------------------------------------------------
# Eligibility policy (user-accepted 2026-08-10)
# --------------------------------------------------------------------------

BARRACKS_STAMP_ID = "markarth_side_v1__u114_castle_barracks"
BARRACKS_REASON = "user-reported defective Castle Barracks extraction"

#: D-STAMP library files in canonical order (Karthgad first, then Markarth).
STAMP_LIBRARY_PATHS: dict[str, str] = {
    "karthgad_nord_v1": "output/cityforge/stamps/karthgad_nord_v1.json",
    "markarth_side_stone_v1": "output/cityforge/stamps/markarth_side_stone_v1.json",
}

CATALOG_INDEX_PATH = "output/cityforge/stamps/catalog_v1/index.json"

FINAL_LIBRARY_ROOT = (
    "output/settlement-splits/markarth-side-v2/"
    "final-markarth-extraction-2026-08-10-library"
)
FINAL_PALETTE_CATALOG = f"{FINAL_LIBRARY_ROOT}/stamp_palette_v1/catalog.json"
FINAL_RENDER_MANIFEST = f"{FINAL_LIBRARY_ROOT}/render_library_manifest.json"

#: The stale preview family that must never reach the emitted brief.
STALE_PREVIEW_MARKER = "split-render-v6"

SURVEY_KIT = (
    ".opencode/runs/karthgad-city-authoring/2026-08-04_karthgad_city_kit_survey.md"
)
SURVEY_REGION = (
    ".opencode/runs/karthgad-city-authoring/2026-08-04_region_palette_and_siting_survey.md"
)
GROUND_RULES_PATH = "output/skyrim_ground_rules.json"


class CensusError(RuntimeError):
    """Hard failure of a census stage (caller must abort, never degrade)."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CensusError(message)


# --------------------------------------------------------------------------
# Libraries and eligibility
# --------------------------------------------------------------------------

def load_stamp_libraries(root: str | Path) -> dict[str, dict[str, Any]]:
    """Load both D-STAMP libraries and return parsed payloads."""
    libraries: dict[str, dict[str, Any]] = {}
    for library_id, relative in STAMP_LIBRARY_PATHS.items():
        path = Path(root) / relative
        _require(path.is_file(), f"stamp library missing: {path}")
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        _require(payload.get("library_id") == library_id,
                 f"library_id mismatch in {path}")
        libraries[library_id] = payload
    return libraries


def select_eligible_stamps(
    libraries: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition all stamp records into eligible and excluded.

    Exclusion rule (user-accepted): the Castle Barracks D-STAMP is quarantined
    with the exact reason above.  Everything else in both libraries is
    eligible.  Duplicate stamp ids are a hard failure.
    """
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for library_id, payload in libraries.items():
        for stamp in payload.get("stamps", []):
            stamp_id = stamp.get("stamp_id")
            _require(isinstance(stamp_id, str) and stamp_id,
                     f"stamp without id in {library_id}")
            _require(stamp_id not in seen, f"duplicate stamp_id {stamp_id}")
            seen.add(stamp_id)
            record = dict(stamp)
            record["_library_id"] = library_id
            if stamp_id == BARRACKS_STAMP_ID:
                record["_exclusion_reason"] = BARRACKS_REASON
                excluded.append(record)
            else:
                eligible.append(record)
    return eligible, excluded


def summarize_upstream_exclusions(
    libraries: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize the upstream (D-STAMP recorded) exclusion ledgers.

    These were decided by the D-STAMP derivation stage (boundary/vehicle
    audit, bounds-missing candidates, doorless components); they are
    preserved and summarized here so the brief does not reintroduce them.
    """
    summary: list[dict[str, Any]] = []
    for library_id, payload in libraries.items():
        stats = payload.get("stats", {})
        if library_id == "markarth_side_stone_v1":
            audit = stats.get("audit_exclusions") or {}
            decisions = audit.get("decisions") or []
            for decision in decisions:
                summary.append({
                    "library": library_id,
                    "unit_id": decision.get("unit_id"),
                    "reason": decision.get("reason"),
                    "scope": "non_building_audit",
                    "evidence_file": audit.get("file"),
                    "preview_evidence": decision.get("preview_evidence"),
                    "source": "D-STAMP stats.audit_exclusions (hash-pinned library)",
                })
        for excluded in stats.get("excluded", []):
            summary.append({
                "library": library_id,
                "unit_id": excluded.get("unit_id"),
                "reason": excluded.get("reason"),
                "scope": excluded.get("scope"),
                "source": "D-STAMP stats.excluded (hash-pinned library)",
            })
        for entry in stats.get("source_recorded_exclusions", []):
            summary.append({
                "library": library_id,
                "candidate_id": entry.get("candidate_id"),
                "reason": entry.get("reason"),
                "scope": entry.get("scope"),
                "source": "D-STAMP stats.source_recorded_exclusions (hash-pinned library)",
            })
    return summary


# --------------------------------------------------------------------------
# Preview resolution
# --------------------------------------------------------------------------

def _palette_catalog(root: str | Path) -> dict[str, Any]:
    path = Path(root) / FINAL_PALETTE_CATALOG
    _require(path.is_file(), f"final palette catalog missing: {path}")
    return __import__("json").loads(path.read_text(encoding="utf-8"))


def _render_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / FINAL_RENDER_MANIFEST
    _require(path.is_file(), f"final render manifest missing: {path}")
    return __import__("json").loads(path.read_text(encoding="utf-8"))


def _stamp_catalog_index(root: str | Path) -> dict[str, Any]:
    path = Path(root) / CATALOG_INDEX_PATH
    _require(path.is_file(), f"stamp catalog index missing: {path}")
    return __import__("json").loads(path.read_text(encoding="utf-8"))


def resolve_markarth_previews(
    stamps: Iterable[Mapping[str, Any]],
    root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Resolve every eligible Markarth preview to the final terrain-backed
    palette by split-record identity.

    Resolution: palette ``entries`` keyed by ``split_record_id`` must contain
    exactly one entry for the stamp's ``source.unit_id``, with category
    ``building_unit`` and status ``eligible``.  The preview is
    ``<library_root>/stamp_palette_v1/<entry.file>``; the on-disk file must
    exist and hash to the catalog sha256.  The render manifest is
    cross-checked for the same sheet sha256.
    """
    catalog = _palette_catalog(root)
    manifest = _render_manifest(root)
    # The palette catalog lives in ``stamp_palette_v1/`` but its sheet files
    # sit in the library root (entry links are "../<file>"); the preview
    # therefore resolves to ``<library_root>/<file>``.
    palette_dir = Path(root) / FINAL_LIBRARY_ROOT
    by_id: dict[str, list[dict[str, Any]]] = {}
    for entry in catalog.get("entries", []):
        by_id.setdefault(entry.get("split_record_id"), []).append(entry)
    manifest_by_id = {
        b["split_record_id"]: b for b in manifest.get("buildings", [])
    }

    resolved: dict[str, dict[str, Any]] = {}
    for stamp in stamps:
        stamp_id = stamp["stamp_id"]
        if stamp.get("_library_id") != "markarth_side_stone_v1":
            continue
        unit_id = (stamp.get("source") or {}).get("unit_id")
        _require(isinstance(unit_id, str) and unit_id,
                 f"{stamp_id}: missing source.unit_id for preview resolution")
        entries = by_id.get(unit_id, [])
        _require(len(entries) == 1,
                 f"{stamp_id}: expected exactly 1 palette entry for {unit_id}, "
                 f"found {len(entries)}")
        entry = entries[0]
        _require(entry.get("category") == "building_unit",
                 f"{stamp_id}: palette entry {unit_id} category is "
                 f"{entry.get('category')!r}, not building_unit")
        _require(entry.get("status") == "eligible",
                 f"{stamp_id}: palette entry {unit_id} status is "
                 f"{entry.get('status')!r}, not eligible")
        file_name = entry.get("file")
        _require(isinstance(file_name, str) and file_name,
                 f"{stamp_id}: palette entry {unit_id} has no file")
        preview = palette_dir / file_name
        _require(preview.is_file(),
                 f"{stamp_id}: final palette preview missing: {preview}")
        on_disk = sha256_file(preview)
        _require(on_disk == entry.get("sha256"),
                 f"{stamp_id}: palette preview sha256 mismatch for {preview}")
        building = manifest_by_id.get(unit_id)
        # The manifest cross-check must never be silently skipped: every
        # eligible Markarth split record is expected in the final manifest.
        _require(building is not None,
                 f"{stamp_id}: split_record_id {unit_id} missing from "
                 f"render_library_manifest.json; cross-check cannot run")
        _require(
            building.get("sheet_sha256") == entry.get("sha256"),
            f"{stamp_id}: render manifest sheet_sha256 disagrees with "
            f"palette catalog for {unit_id}",
        )
        resolved[stamp_id] = {
            "split_record_id": unit_id,
            "preview_file": file_name,
            "preview_path": str(preview),
            "preview_path_relative": f"{FINAL_LIBRARY_ROOT}/{file_name}",
            "sha256": entry.get("sha256"),
            "catalog_entry": "stamp_palette_v1/catalog.json#entries[split_record_id="
                             f"{unit_id}]",
            "replaced_source": stamp.get("preview_sheet"),
            "cross_check_manifest_sha256": building.get("sheet_sha256")
            if building is not None else None,
        }
    return resolved


def verify_karthgad_previews(
    stamps: Iterable[Mapping[str, Any]],
    root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Hash-verify Karthgad preview sheets against their source paths and the
    catalog index (which records the same hash for source and copy)."""
    index = _stamp_catalog_index(root)
    by_id = {s["stamp_id"]: s for s in index.get("stamps", [])}
    verified: dict[str, dict[str, Any]] = {}
    for stamp in stamps:
        stamp_id = stamp["stamp_id"]
        if stamp.get("_library_id") != "karthgad_nord_v1":
            continue
        source = stamp.get("preview_sheet")
        _require(isinstance(source, str) and source,
                 f"{stamp_id}: missing preview_sheet")
        path = Path(root) / source
        _require(path.is_file(), f"{stamp_id}: preview missing: {path}")
        on_disk = sha256_file(path)
        catalog = by_id.get(stamp_id)
        _require(catalog is not None,
                 f"{stamp_id}: absent from catalog_v1/index.json")
        _require(catalog.get("verified") is True,
                 f"{stamp_id}: catalog index does not mark preview verified")
        _require(on_disk == catalog.get("sha256"),
                 f"{stamp_id}: preview sha256 {on_disk} disagrees with catalog "
                 f"{catalog.get('sha256')}")
        verified[stamp_id] = {
            "preview_path": str(path),
            "preview_path_relative": source,
            "sha256": on_disk,
            "catalog_entry": "catalog_v1/index.json#stamps[stamp_id]",
            "catalog_verified": True,
        }
    return verified


# --------------------------------------------------------------------------
# Stamp aggregation
# --------------------------------------------------------------------------

def condensed_stamp_record(
    stamp: Mapping[str, Any],
    preview: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """One planner-facing record per eligible stamp."""
    source = stamp.get("source", {})
    bounds = stamp.get("bounds_rel_gu", {})
    span = bounds.get("span")
    aabb = stamp.get("footprint", {}).get("aabb_rel", {})
    hull = stamp.get("footprint", {}).get("hull_xy_rel")
    terrain = stamp.get("terrain_envelope", {})
    return {
        "stamp_id": stamp["stamp_id"],
        "library_id": stamp["_library_id"],
        "source_run": source.get("run"),
        "source_slug": source.get("slug"),
        "source_unit_id": source.get("unit_id"),
        "source_cell": source.get("source_cell"),
        "seed_door": (stamp.get("anchor") or {}).get("seed_door"),
        "style_tags": sorted(stamp.get("style_tags", [])),
        "building_type": stamp.get("building_type"),
        "size_class": stamp.get("size_class"),
        "door_count": stamp.get("door_count"),
        "multi_shell": bool(stamp.get("multi_shell")),
        "footprint_span_gu": list(span) if span else None,
        "footprint_aabb_area_gu2": (
            float(span[0]) * float(span[1]) if span else None
        ),
        "footprint_hull_area_gu2": _hull_area(hull),
        "access_heading_rad": stamp.get("access_heading_rad"),
        "terrain": {
            "burial_depth_gu": terrain.get("burial_depth_gu"),
            "door_step_heights_gu": list(terrain.get("door_step_heights_gu", [])),
            "footprint_relief_gu": terrain.get("footprint_relief_gu"),
            "footprint_slope_deg": terrain.get("footprint_slope_deg"),
        },
        "preview_sheet": (preview or {}).get(
            "preview_path_relative", stamp.get("preview_sheet")),
        "preview_path_absolute": (preview or {}).get("preview_path"),
        "preview_sha256": (preview or {}).get("sha256"),
        "preview_provenance": (preview or {}).get("catalog_entry"),
        "preview_replaced_source": (preview or {}).get("replaced_source"),
    }


def _hull_area(hull: Any) -> float | None:
    if not hull or len(hull) < 3:
        return None
    total = 0.0
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def world_polygon(stamp: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Source-world footprint: stored anchor xy + relative hull xy."""
    anchor = stamp.get("anchor", {})
    position = anchor.get("source_position_gu")
    _require(isinstance(position, list) and len(position) >= 2,
             f"{stamp.get('stamp_id')}: anchor source_position_gu missing")
    hull = stamp.get("footprint", {}).get("hull_xy_rel")
    _require(isinstance(hull, list) and len(hull) >= 3,
             f"{stamp.get('stamp_id')}: hull_xy_rel missing or degenerate")
    origin_x, origin_y = float(position[0]), float(position[1])
    return [(origin_x + float(hx), origin_y + float(hy)) for hx, hy in hull]


def derive_building_type_enum(
    eligible: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Closed planner vocabulary: sorted union of eligible stamp types.

    This deliberately replaces the stale hard-coded example enum; ``lodge``
    and ``shack`` are absent because no eligible stamp carries them, and no
    type is coerced into another.
    """
    return sorted({stamp.get("building_type") for stamp in eligible})


def capability_gaps(eligible: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Machine-readable unavailability list (measured-data limitations)."""
    available = {stamp.get("building_type") for stamp in eligible}
    declared = {
        "lodge": "no measured lodge stamp exists in either D-STAMP library; "
                 "a manor is not relabeled",
        "stone_wall": "no measured stone-wall boundary census exists yet "
                      "(deferred to the citywalls analysis arc)",
        "fence_spacing": "farm fence pieces are measured (counts) but no "
                         "fence spacing rule is measured",
    }
    gaps: list[dict[str, Any]] = []
    for type_name, reason in sorted(declared.items()):
        gaps.append({
            "type": type_name,
            "available": type_name in available,
            "reason": reason if type_name not in available else None,
        })
    return gaps


# --------------------------------------------------------------------------
# Footprint quantiles
# --------------------------------------------------------------------------

def footprint_quantiles(
    eligible: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recomputed footprint quantiles (global + per type) over eligible
    stamps only; old D-STAMP class thresholds are never copied."""
    records = list(eligible)
    areas = []
    hull_areas = []
    spans = []
    for record in records:
        span = (record.get("bounds_rel_gu") or {}).get("span")
        if isinstance(span, list) and len(span) >= 2:
            width, depth = float(span[0]), float(span[1])
            areas.append(width * depth)
            spans.append([width, depth])
        hull = record.get("footprint", {}).get("hull_xy_rel")
        hull_area = _hull_area(hull)
        if hull_area is not None:
            hull_areas.append(hull_area)
    per_type: dict[str, dict[str, Any]] = {}
    for type_name in sorted({r["building_type"] for r in records}):
        type_areas = []
        for r in records:
            if r["building_type"] != type_name:
                continue
            span = (r.get("bounds_rel_gu") or {}).get("span")
            if isinstance(span, list) and len(span) >= 2:
                type_areas.append(float(span[0]) * float(span[1]))
        per_type[type_name] = {
            "count": len(type_areas),
            **quantile_summary(type_areas),
        }
    return {
        "basis": "eligible stamps only; Castle Barracks excluded before "
                 "quantile computation",
        "unit": "GU^2 (footprint AABB area = span_x * span_y)",
        "global_aabb_area_gu2": quantile_summary(areas),
        "global_hull_area_gu2": quantile_summary(hull_areas),
        "span_x_gu": quantile_summary([s[0] for s in spans]),
        "span_y_gu": quantile_summary([s[1] for s in spans]),
        "per_type_aabb_area_gu2": per_type,
    }


# --------------------------------------------------------------------------
# Spacing measurement
# --------------------------------------------------------------------------

def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _point_segment_distance_sq(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    qx, qy = a[0] + t * dx, a[1] + t * dy
    return (p[0] - qx) ** 2 + (p[1] - qy) ** 2


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Proper-or-touching segment intersection (orientation tests)."""
    d1 = _cross((p2[0] - p1[0], p2[1] - p1[1]), (p3[0] - p1[0], p3[1] - p1[1]))
    d2 = _cross((p2[0] - p1[0], p2[1] - p1[1]), (p4[0] - p1[0], p4[1] - p1[1]))
    d3 = _cross((p4[0] - p3[0], p4[1] - p3[1]), (p1[0] - p3[0], p1[1] - p3[1]))
    d4 = _cross((p4[0] - p3[0], p4[1] - p3[1]), (p2[0] - p3[0], p2[1] - p3[1]))
    return ((d1 * d2 <= 0.0) and (d3 * d4 <= 0.0)
            and (d1 != 0.0 or d2 != 0.0 or d3 != 0.0 or d4 != 0.0))


def _point_in_polygon(
    p: tuple[float, float], polygon: Sequence[tuple[float, float]]
) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > p[1]) != (yj > p[1])) and (
            p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def polygons_intersect_or_touch(
    a: Sequence[tuple[float, float]],
    b: Sequence[tuple[float, float]],
) -> bool:
    """True when the two polygons intersect, touch, or one contains the
    other (boundary gap is then exactly 0.0)."""
    for i in range(len(a)):
        for j in range(len(b)):
            if _segments_intersect(a[i], a[(i + 1) % len(a)],
                                   b[j], b[(j + 1) % len(b)]):
                return True
    # containment: either polygon fully inside the other
    for vertex in a:
        if _point_in_polygon(vertex, b):
            return True
    for vertex in b:
        if _point_in_polygon(vertex, a):
            return True
    return False


def polygon_boundary_gap(
    a: Sequence[tuple[float, float]],
    b: Sequence[tuple[float, float]],
) -> float:
    """Minimum boundary gap between two source-world footprints.

    Returns exactly ``0.0`` when the polygons intersect or touch (zero-gap
    class); otherwise the minimum Euclidean distance between any two polygon
    edges.
    """
    if polygons_intersect_or_touch(a, b):
        return 0.0
    best = float("inf")
    for i in range(len(a)):
        a1, a2 = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            b1, b2 = b[j], b[(j + 1) % len(b)]
            best = min(best, _point_segment_distance_sq(a1, b1, b2))
            best = min(best, _point_segment_distance_sq(a2, b1, b2))
            best = min(best, _point_segment_distance_sq(b1, a1, a2))
            best = min(best, _point_segment_distance_sq(b2, a1, a2))
    return math.sqrt(best)


def spacing_census(
    eligible: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Same-run nearest-neighbor boundary-gap census.

    For every run (``karthgad-v1``, ``markarth-side-v1``) all same-run stamp
    pairs are measured; the per-stamp nearest neighbor (minimum boundary gap)
    is the planner prior sample set.  Zero-gap stamps (nearest neighbor
    intersects/touches) are counted separately.  All pairwise gaps are also
    preserved as raw samples.
    """
    stamps = list(eligible)
    runs: dict[str, list[Mapping[str, Any]]] = {}
    for stamp in stamps:
        runs.setdefault(stamp.get("source", {}).get("run"), []).append(stamp)

    run_stats: dict[str, Any] = {}
    nn_samples: list[float] = []
    nn_records: list[dict[str, Any]] = []
    pair_samples: list[dict[str, Any]] = []
    total_pairs = 0
    total_zero_gap_pairs = 0

    for run in sorted(runs):
        group = sorted(runs[run], key=lambda s: s["stamp_id"])
        polygons = {s["stamp_id"]: world_polygon(s) for s in group}
        ids = [s["stamp_id"] for s in group]
        pairs: list[dict[str, Any]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                gap = polygon_boundary_gap(polygons[ids[i]], polygons[ids[j]])
                pairs.append({
                    "stamp_a": ids[i],
                    "stamp_b": ids[j],
                    "gap_gu": gap,
                    "zero_gap": gap == 0.0,
                })
        pairs.sort(key=lambda p: (p["stamp_a"], p["stamp_b"]))
        pair_samples.extend(pairs)
        total_pairs += len(pairs)
        total_zero_gap_pairs += sum(1 for p in pairs if p["zero_gap"])
        # nearest neighbor per stamp
        run_nn: list[dict[str, Any]] = []
        for stamp_id in ids:
            candidates = [
                (p["gap_gu"], p["stamp_b"] if p["stamp_a"] == stamp_id else p["stamp_a"])
                for p in pairs
                if stamp_id in (p["stamp_a"], p["stamp_b"])
            ]
            if candidates:
                gap, neighbor = min(candidates, key=lambda c: (c[0], c[1]))
            else:
                gap, neighbor = None, None  # single-stamp run has no pair
            run_nn.append({
                "stamp_id": stamp_id,
                "nearest_neighbor": neighbor,
                "gap_gu": gap,
                "zero_gap": bool(gap is not None and gap == 0.0),
            })
            if gap is not None:
                nn_samples.append(gap)
                nn_records.append(run_nn[-1])
        positive = [p["gap_gu"] for p in pairs if not p["zero_gap"]]
        run_nn_values = [p["gap_gu"] for p in run_nn if p["gap_gu"] is not None]
        run_nn_positive = [g for g in run_nn_values if g > 0.0]
        run_stats[run] = {
            "stamp_count": len(group),
            "pair_count": len(pairs),
            "zero_gap_pair_count": sum(1 for p in pairs if p["zero_gap"]),
            "positive_gap_pair_count": len(positive),
            "pairwise_gap_gu": quantile_summary(positive),
            "pairwise_all_gaps_gu": quantile_summary([p["gap_gu"] for p in pairs]),
            "nearest_neighbor_gap_gu": quantile_summary(run_nn_values),
            "nearest_neighbor_positive_gap_gu": quantile_summary(run_nn_positive),
            "nearest_neighbor_zero_gap_stamp_count": len(run_nn_values)
            - len(run_nn_positive),
        }

    positive_nn = [g for g in nn_samples if g > 0.0]
    positive_nn_by_run = {
        run: stats["nearest_neighbor_positive_gap_gu"]["n"]
        for run, stats in run_stats.items()
    }
    return {
        "method": (
            "source-world footprint = stored seed-door anchor xy + "
            "footprint.hull_xy_rel; polygon boundary gap = minimum Euclidean "
            "distance between polygon edges; pairs that intersect or touch "
            "are classified zero-gap (gap exactly 0.0) and separated from "
            "positive gaps; measured only within the same source run "
            "(never across Karthgad<->Markarth)"
        ),
        "unit": "GU",
        "scopes": "same-source-run pairs only",
        "granularity": {
            "karthgad-v1": {
                "grouping": "door-seeded complete-building grouping "
                            "(D-STAMP seed-door anchors; multi-door stamps "
                            "merge doors of one component)",
                "unit_id_available": False,
                "note": "Karthgad stamps carry no source.unit_id, so "
                        "fused/duplicate source-building identification is "
                        "impossible at this census level; all 11 stamps are "
                        "units of one dense door-anchored core in cell "
                        "(-102,11)",
            },
            "markarth-side-v1": {
                "grouping": "split-unit grouping (manual-corrections-v1 "
                            "split units)",
                "unit_id_available": True,
                "note": "Markarth stamps are per-house split units with "
                        "unit_id provenance",
            },
        },
        "interpretation": [
            "hulls are D-STAMP source-world footprint hulls (approximated "
            "unit envelopes, not exact contact geometry)",
            "zero gaps reflect overlaps/touches of these approximated unit "
            "envelopes (39/54 stamps have a zero-gap nearest neighbor)",
            f"the positive-only mixed nearest-neighbor sample "
            f"(n={len(positive_nn)}: "
            f"{positive_nn_by_run.get('karthgad-v1', 0)} Karthgad door-level "
            f"+ {positive_nn_by_run.get('markarth-side-v1', 0)} Markarth "
            f"building-level, computed per run) is exploratory measured "
            f"guidance, not a universal hard clearance",
            "collision clearance is the geometry solver's exact-hull "
            "domain; the layout spacing prior is a separate measured "
            "quantity and is not a hard minimum",
        ],
        "runs": run_stats,
        "combined": {
            "stamp_count": len(stamps),
            "pair_count": total_pairs,
            "zero_gap_pair_count": total_zero_gap_pairs,
            "positive_gap_pair_count": total_pairs - total_zero_gap_pairs,
            "nearest_neighbor_gap_gu": quantile_summary(nn_samples),
            "nearest_neighbor_positive_gap_gu": quantile_summary(positive_nn),
            "nearest_neighbor_zero_gap_stamp_count": len(nn_samples)
            - len(positive_nn),
        },
        "nearest_neighbor_samples": nn_records,
        "pairwise_samples": pair_samples,
    }


# --------------------------------------------------------------------------
# Door-step priors
# --------------------------------------------------------------------------

def aggregate_door_steps(
    eligible: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Door-step priors aggregated from the eligible stamps' own terrain
    envelopes (each door sample carries its stamp id)."""
    samples: list[dict[str, Any]] = []
    for stamp in eligible:
        terrain = stamp.get("terrain_envelope", {})
        for height in terrain.get("door_step_heights_gu", []):
            samples.append({
                "stamp_id": stamp["stamp_id"],
                "door_step_height_gu": float(height),
            })
    samples.sort(key=lambda s: (s["stamp_id"], s["door_step_height_gu"]))
    values = [s["door_step_height_gu"] for s in samples]
    return {
        "source": "eligible D-STAMP terrain_envelope.door_step_heights_gu "
                  "(one sample per door of every eligible stamp)",
        "sample_count": len(values),
        **quantile_summary(values),
        "samples": samples,
    }


def ground_rules_door_steps(root: str | Path) -> dict[str, Any]:
    """Consumed (not recomputed) ground-rule door-step statistics."""
    path = Path(root) / GROUND_RULES_PATH
    _require(path.is_file(), f"ground rules missing: {path}")
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    stats = payload.get("stats", {})
    return {
        "file": GROUND_RULES_PATH,
        "path_absolute": str(path),
        "sha256": sha256_file(path),
        "measured_by": "skyrim_ground_rules.json (601 clusters / 8681 members "
                       "aggregated by that product's tool)",
        "door_step_count": stats.get("door_step_count"),
        "door_step_min_game_units": stats.get("door_step_min_game_units"),
        "door_step_max_game_units": stats.get("door_step_max_game_units"),
        "door_step_p10_game_units": stats.get("door_step_p10_game_units"),
        "door_step_p50_game_units": stats.get("door_step_p50_game_units"),
        "door_step_p90_game_units": stats.get("door_step_p90_game_units"),
        "status": "survey_measured",  # consumed, not recomputed here
    }


# --------------------------------------------------------------------------
# Kit brief assembly
# --------------------------------------------------------------------------

def _survey_pin(root: str | Path, relative: str) -> dict[str, Any]:
    path = Path(root) / relative
    return {
        "file": relative,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def build_kit_brief(
    *,
    eligible: Sequence[Mapping[str, Any]],
    excluded_records: Sequence[Mapping[str, Any]],
    upstream_exclusions: Sequence[Mapping[str, Any]],
    building_type_enum: Sequence[str],
    gaps: Sequence[Mapping[str, Any]],
    spacing: Mapping[str, Any],
    door_steps: Mapping[str, Any],
    ground_rules: Mapping[str, Any],
    previews: Mapping[str, Mapping[str, Any]],
    library_pins: Mapping[str, dict[str, object]],
    root: str | Path,
    date: str,
) -> dict[str, Any]:
    """Assemble ``kit_brief.json`` — the planner's build vocabulary."""
    kit_pin = _survey_pin(root, SURVEY_KIT)
    region_pin = _survey_pin(root, SURVEY_REGION)
    stamps = [
        condensed_stamp_record(stamp, previews.get(stamp["stamp_id"]))
        for stamp in eligible
    ]
    stamps.sort(key=lambda s: s["stamp_id"])
    house_count = sum(1 for s in stamps if s["building_type"] == "house")
    return {
        "schema_version": 1,
        "kit_id": "falkreath_karthgad_nord_stone_v1",
        "date": date,
        "style_tags": sorted(
            set().union(*(s.get("style_tags", []) for s in stamps))
        ),
        "sources": {
            "runs": ["karthgad-v1", "markarth-side-v1"],
            "stamp_library_karthgad": {
                "library_id": "karthgad_nord_v1",
                **library_pins["karthgad_nord_v1"],
            },
            "stamp_library_markarth": {
                "library_id": "markarth_side_stone_v1",
                **library_pins["markarth_side_stone_v1"],
            },
            "final_palette_authority": (
                "stamp_palette_v1/catalog.json is preview/provenance authority "
                "for Markarth sheets; D-STAMP JSON remains stamp geometry "
                "authority"
            ),
        },
        "building_type_enum": list(building_type_enum),
        "building_type_counts": {
            type_name: sum(1 for s in stamps if s["building_type"] == type_name)
            for type_name in building_type_enum
        },
        "capability_gaps": list(gaps),
        "exclusion_ledger": {
            "user_reported": [
                {
                    "stamp_id": record["stamp_id"],
                    "unit_id": record.get("source", {}).get("unit_id"),
                    "reason": record["_exclusion_reason"],
                    "library": record["_library_id"],
                }
                for record in excluded_records
            ],
            "upstream_summary": list(upstream_exclusions),
            "note": "upstream D-STAMP exclusions (walls/ship false positives, "
                    "bounds-missing candidates, doorless components) are "
                    "preserved from the hash-pinned libraries and are not "
                    "reintroduced from any render manifest",
        },
        "stamp_count": len(stamps),
        "stamps": stamps,
        "spacing_priors": {
            "collision_clearance": {
                "provider": "geometry solver (D-PLACE exact hull/contact "
                            "evaluation)",
                "hard_minimum_gu": 0.0,
                "basis": "measured building-to-building contact graph: "
                         "strict_intersection at 0.0 GU (Karthgad kit survey "
                         "sections 2.4/3; AABB-derived for palisades only)",
                "provenance": {**kit_pin, "sections": ["2.4", "3"],
                               "lines": [106, 143, 163]},
            },
            "inter_building_gap_gu": {
                "p10": spacing["combined"]["nearest_neighbor_positive_gap_gu"]["p10"],
                "p50": spacing["combined"]["nearest_neighbor_positive_gap_gu"]["p50"],
                "p90": spacing["combined"]["nearest_neighbor_positive_gap_gu"]["p90"],
                "usable_as_hard_minimum": False,
                "evidence_class": "measured guidance (not a hard minimum; "
                                  "see granularity notes)",
                "basis": "same-run nearest-neighbor polygon boundary gaps, "
                         "positive gaps only (intersecting/touching pairs are "
                         "the separate zero-gap class)",
                "granularity_notes": [
                    f"mixed sample: {spacing['runs']['karthgad-v1']['nearest_neighbor_positive_gap_gu']['n']} "
                    "Karthgad door-seeded units of one dense core + "
                    f"{spacing['runs']['markarth-side-v1']['nearest_neighbor_positive_gap_gu']['n']} "
                    "Markarth split units; exploratory, not a universal "
                    "clearance",
                    "39/54 stamps have a zero-gap nearest neighbor; zero "
                    "gaps reflect overlaps/touches of the approximated "
                    "D-STAMP hulls",
                    "per-run separated distributions in "
                    "census.json#spacing.runs",
                ],
                "method": spacing["method"],
                "sample_count": spacing["combined"]["nearest_neighbor_positive_gap_gu"]["n"],
                "zero_gap_stamp_count": spacing["combined"][
                    "nearest_neighbor_zero_gap_stamp_count"
                ],
                "census_ref": "census.json#spacing",
            },
            "inter_building_gap_including_zero_gu": {
                "p10": spacing["combined"]["nearest_neighbor_gap_gu"]["p10"],
                "p50": spacing["combined"]["nearest_neighbor_gap_gu"]["p50"],
                "p90": spacing["combined"]["nearest_neighbor_gap_gu"]["p90"],
                "usable_as_hard_minimum": False,
                "basis": "same-run nearest-neighbor gaps over all eligible "
                         "stamps; zero-gap stamps included (measured "
                         "zero-gap mass is a source fact, not filtered)",
                "sample_count": spacing["combined"]["stamp_count"],
                "zero_gap_stamp_count": spacing["combined"][
                    "nearest_neighbor_zero_gap_stamp_count"
                ],
                "census_ref": "census.json#spacing",
            },
            "tree_clearance_from_shell_gu": {
                "value": 600,
                "basis": "measured Karthgad tree-to-shell distances: 0 trees "
                         "<300 GU, 3 <600 GU, median 1235 GU",
                "provenance": {
                    **kit_pin,
                    "section": "4. Trees / flora placement",
                    "lines": [170, 171],
                },
            },
            "door_step_height_gu": {
                "p10": door_steps["p10"],
                "p50": door_steps["p50"],
                "p90": door_steps["p90"],
                "basis": "aggregated from eligible stamp terrain envelopes "
                         "(per-door samples)",
                "sample_count": door_steps["sample_count"],
                "census_ref": "census.json#door_steps",
                "ground_rules_crosscheck": {
                    "p10": ground_rules["door_step_p10_game_units"],
                    "p50": ground_rules["door_step_p50_game_units"],
                    "p90": ground_rules["door_step_p90_game_units"],
                    "sample_count": ground_rules["door_step_count"],
                    "status": ground_rules["status"],
                    "provenance": {
                        "file": ground_rules["file"],
                        "sha256": ground_rules["sha256"],
                    },
                },
            },
        },
        "boundary_pieces": {
            "palisade": {
                "modules": [
                    "sky\\x\\sky_ex_nord_f_wl_01.nif",
                    "sky\\x\\sky_ex_nord_f_wl_02.nif",
                    "sky\\x\\sky_ex_nord_f_wl_03.nif",
                ],
                "module_length_gu": [310, 361],
                "origin_spacing_gu": 295,
                "spacing_band_gu": [250, 320],
                "overlap_gu": [35, 70],
                "corner_step_max_deg": 30,
                "walkway_offset_gu": 282,
                "module_reach_below_origin_gu": 800,
                "gatehouse": {
                    "model": "sky\\x\\sky_ex_nord_f_gt_01.nif",
                    "footprint_gu": [860, 860],
                    "replaces_modules": 3,
                },
                "tower": {
                    "model": "sky\\x\\sky_ex_nord_f_tw_01.nif",
                    "footprint_gu": [766, 757],
                },
                "stairs": [
                    "sky\\x\\sky_ex_nord_f_l_02.nif",
                    "sky\\x\\sky_ex_nord_f_l_03.nif",
                ],
                "ladder_pair": {
                    "up": "sky\\d\\sky_ex_n_f_l_01_up.nif",
                    "bot": "sky\\d\\sky_ex_n_f_l_01_lw.nif",
                },
                "method_labels": {
                    "overlap": "AABB-derived (contact graph intentionally "
                               "excludes palisades)",
                    "spacing": "68 wall-to-wall origin pairs across 13 runs",
                },
                "provenance": {
                    **kit_pin,
                    "sections": ["1.3", "2.1", "2.2", "2.3", "2.4"],
                    "lines": [73, 153],
                },
            },
            "fence": {
                "pieces": [
                    {"object_id": "T_Nor_SetFarm_X_FencePost_01", "count": 15},
                    {"object_id": "T_Nor_SetFarm_X_FenceSlope_01", "count": 11},
                    {"object_id": "T_Nor_SetFarm_X_FenceLent_01", "count": 1},
                ],
                "spacing_rule": None,
                "spacing_note": "no fence spacing rule measured; capability "
                                "gap emitted (capability_gaps.fence_spacing)",
                "provenance": {**kit_pin, "section": "1.4", "lines": [95]},
            },
            "stone_wall": {
                "available": False,
                "note": "no measured Markarth stone-wall census exists yet; "
                        "deferred to the citywalls analysis arc "
                        "(capability_gaps.stone_wall)",
            },
        },
        "street_furniture": [
            {
                "kind": "lantern_hook_pair",
                "pieces": [
                    "T_Nor_Var_LanternStat_01",
                    "T_Nor_Var_WoodHook_01",
                ],
                "rule": "hook 8.5 GU from lantern, 4.5 GU below, wall-mounted",
                "measured": "21 lanterns, 19/21 with hook within 8.5 GU, "
                            "hook 4.5 GU below",
                "provenance": {**kit_pin, "section": "1.4", "lines": [97]},
            },
            {
                "kind": "signpost",
                "pieces": ["T_Nor_Set_Signpost_02", "T_Nor_Set_SignWay*"],
                "rule": "way boards stacked 24 GU apart; at road junctions "
                        "~6.4k and ~11.3k GU outside town",
                "provenance": {**kit_pin, "section": "1.4", "lines": [98]},
            },
            {
                "kind": "well",
                "pieces": ["T_Nor_Set_Well_01"],
                "measured": "1 in Karthgad market square",
                "provenance": {**kit_pin, "section": "1.4", "lines": [96]},
            },
            {
                "kind": "banner",
                "pieces": ["T_Nor_Set_BannerTownKarthgad_01"],
                "rule": "mounted on gatehouses (2) and one wall stretch (3 total)",
                "provenance": {**kit_pin, "section": "1.4", "lines": [99]},
            },
        ],
        "docks": {
            "pieces": [
                {"object_id": "T_Nor_Set_DocksEnd_03", "count": 12},
                {"object_id": "T_Nor_Set_DocksPiling_01", "count": 11},
                {"object_id": "T_Nor_Set_DocksSteps_02", "count": 6},
                {"object_id": "T_Nor_Set_DocksCleat_01", "count": 3},
                {"object_id": "T_Nor_Set_DocksCenter_01", "count": 1},
                {"object_id": "T_Nor_Set_DocksCenter_03", "count": 1},
                {"object_id": "T_Nor_Set_DocksCenter_04", "count": 1},
                {"object_id": "T_Nor_Set_Docks3Way_01", "count": 1},
                {"object_id": "T_Nor_Set_DocksCorner_10", "count": 1},
            ],
            "rules": "deck z constant per platform; steps -240 GU per 328 GU "
                     "run; pilings under deck; all contacts 0.0 GU strict "
                     "(contact graph)",
            "provenance": {**kit_pin, "section": "3", "lines": [158, 163]},
        },
        "flora_rock_palette_ref": "region_palette.json#flora_rock",
        "semantic_surfaces_used": [
            "base",
            "settlement_dirt",
            "settlement_grass_dirt",
            "settlement_cobble",
            "road",
            "water_edge_sand",
        ],
        "semantic_surfaces_used_note": (
            "machine-readable surface references of this brief; a plan "
            "validator must check them against "
            "region_palette.json#semantic_surfaces and fail on any surface "
            "absent from that closed vocabulary"
        ),
        "groundcover": {
            "ini": "configs/groundcover_falkreath_v1_currenttextures.ini",
            "road_mask_regex": ".*hr_oh_road.*",
            "note": "road mask couples to LTEX record ids; raw 78 identity is "
                    "the scatter/groundcover road gate",
            "provenance": {**region_pin, "section": "2.8", "lines": [202, 209]},
        },
        "coverage": {
            "house_count": house_count,
            "requirement_house_min": 15,
            "tavern_count": sum(1 for s in stamps if s["building_type"] == "tavern"),
            "smith_count": sum(1 for s in stamps if s["building_type"] == "smith"),
            "shop_count": sum(1 for s in stamps if s["building_type"] == "shop"),
            "farm_count": sum(1 for s in stamps if s["building_type"] == "farm"),
        },
        "survey_refs": {
            "kit_survey": kit_pin,
            "region_survey": region_pin,
        },
    }


def build_census_stamp_section(
    *,
    eligible: Sequence[Mapping[str, Any]],
    excluded_records: Sequence[Mapping[str, Any]],
    upstream_exclusions: Sequence[Mapping[str, Any]],
    building_type_enum: Sequence[str],
    library_counts: Mapping[str, int],
    markarth_previews: Mapping[str, Mapping[str, Any]],
    karthgad_previews: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Raw stamp vectors and exclusion provenance for ``census.json``.

    ``library_counts`` must be derived from the loaded (hash-verified)
    D-STAMP payloads (``len(payload["stamps"])``) by the caller; it is never
    hardcoded here.
    """
    markarth_ids = sorted(markarth_previews)
    karthgad_ids = sorted(karthgad_previews)
    return {
        "libraries": {
            library_id: {
                "stamp_count_raw": library_counts[library_id],
                "stamp_count_source": "len(payload['stamps']) of the "
                                      "hash-pinned D-STAMP library",
            }
            for library_id in sorted(library_counts)
        },
        "eligible_count": len(eligible),
        "eligible": [
            condensed_stamp_record(
                stamp, {**markarth_previews, **karthgad_previews}.get(
                    stamp["stamp_id"]))
            for stamp in eligible
        ],
        "building_type_enum": list(building_type_enum),
        "excluded": {
            "count": len(excluded_records),
            "records": [
                {
                    "stamp_id": r["stamp_id"],
                    "unit_id": r.get("source", {}).get("unit_id"),
                    "library": r["_library_id"],
                    "reason": r["_exclusion_reason"],
                }
                for r in excluded_records
            ],
        },
        "upstream_exclusion_summary": list(upstream_exclusions),
        "preview_verification": {
            "karthgad": {
                "count": len(karthgad_ids),
                "records": {
                    stamp_id: karthgad_previews[stamp_id]
                    for stamp_id in karthgad_ids
                },
            },
            "markarth": {
                "count": len(markarth_ids),
                "records": {
                    stamp_id: {k: v for k, v in markarth_previews[stamp_id].items()
                               if k != "replaced_source"}
                    for stamp_id in markarth_ids
                },
            },
            "note": "merged all-stamp preview verification with explicit "
                    "per-library subcounts (Karthgad hash-verified source "
                    "renders; Markarth resolved to final terrain-backed "
                    "palette sheets)",
        },
    }
