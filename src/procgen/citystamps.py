"""D-STAMP v1 unit-stamp library derivation (Cityforge T0.3).

Purpose
-------
Deterministically derive complete, guaranteed-valid building *unit stamps*
from existing extraction/split products only (no new extraction runs):

* Karthgad Nord kit: component manifests
  (``output/skyrim-settlements/karthgad-v1/components/buildings/<slug>/manifest.json``),
  the B1 placement manifest, per-building landscape terrain evidence, and
  source-plugin LAND re-measurement via :mod:`procgen.espland`.
* Markarth Side stone kit: the approved split subset
  (``output/settlement-splits/markarth-side-v2/manual-corrections-v1/split/units.json``),
  with every split member joined to the authoritative Markarth placement
  manifest by ``source_id``, plus the per-component manifests for member
  structure/bounds.

Every number in a stamp traces to a measured source value: member offsets are
exact float64 differences of source positions, rotations/scales are copied
verbatim from the source Euler triples, and the terrain envelope is
re-measured against source-plugin LAND with ``espland``.  The module never
invents values and never patches a broken candidate: units with ghost members,
protocol failures, no doors, missing bounds, missing LAND, or no existing
preview are excluded with exactly one reason each.

Pipeline position
-----------------
* Feeds: extraction products listed above + ``Sky_Main.esm`` (Karthgad LAND),
  ``Sky_Markarth.esm`` (Markarth LAND), both read-only.
* Consumed by: ``tools/cityforge/stamp_library.py`` (CLI that loads products,
  calls the builders below, writes ``output/cityforge/stamps/*.json`` and the
  browsable catalog).
* Consumer (later task): ``src/procgen/cityplace.py`` / T0.1
  ``engine_transform.py`` — stamps copy source Euler triples and compute
  world-aligned offsets only; all Euler arithmetic is forbidden here by
  design (see D-STAMP spec §5.2).

Invariants
----------
* Deterministic output: sorted keys, stable stamp ordering by ``stamp_id``,
  members ordered doors-first then by ``source_id``; no timestamps.
* Anchor = the seed door's world position; offsets are pure subtraction in
  game units (world-aligned, no baked rotation).
* Rotation and scale are copied verbatim from the source transform.
* ``access_heading_rad`` is measured from the footprint's 2D convex-hull
  polygon centroid toward the seed door (never from a mesh-facing
  convention); degenerate hulls fall back to the mean of member-bounds
  corners.
* Audited non-building exclusions (``non_building_boundary`` /
  ``non_building_vehicle``) are an exact, hash-pinned review list applied by
  the CLI before derivation; they take priority over derivation failures and
  are recorded in the ledger with the inspected preview evidence.
* Zero accepted ghost / protocol-failure / doorless candidates.
* Every excluded candidate appears exactly once with one reason.
* Canonical JSON bytes use ``json.dumps(..., indent=2, sort_keys=True)`` plus
  a trailing newline, matching the settlement-cache discipline.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import espland


SCHEMA_VERSION = 1
__version__ = "0.1.0"

THU_TO_GU = espland.THU_TO_GU
# Height-field sampling parameters must match the landscape product
# (``landscape/.../terrain.json``) so cross-checks are apples-to-apples.
HEIGHT_FIELD_MARGIN_GU = 256.0
HEIGHT_FIELD_SIDE = 65
# Documented replay tolerance: offsets are exact float64 differences of
# float32-derived dyadic source values, so reconstruction is exact in
# practice; the documented bound is generous and still far below any
# float32 precision concern at these magnitudes.
POSITION_REPLAY_TOLERANCE_GU = 0.001
# Terrain cross-check agreement bound from the D-STAMP spec (§3.4): a
# disagreement of more than 1 GU between the landscape product and the
# uniform LAND re-derivation is an explicit report row, never silent.
TERRAIN_CROSSCHECK_TOLERANCE_GU = 1.0

# Audited review exclusions: an exact, hash-pinned list of split units that
# the lead inspection determined are NOT complete building stamps (wall/
# boundary piece networks, vehicles).  These are not derivation failures:
# they are deterministic review overrides that take priority over every
# derivation reason below, and each is recorded once with the inspected
# preview evidence.  The CLI refuses to apply them to any other source hash.
AUDITED_EXCLUSION_REASONS = (
    "non_building_boundary",   # wall/boundary piece network, not a building
    "non_building_vehicle",    # vehicle (e.g. ship); dock/vehicle solvers later
)

# Canonical exclusion reasons, ordered by derivation severity.  A candidate
# receives exactly one reason: the first (highest-priority) condition that
# applies, so every excluded candidate appears exactly once.
EXCLUSION_REASONS = (
    "ghost_members",          # member ref missing/unresolved in placement manifest
    "protocol_failure",       # building-level protocol_failures non-empty
    "no_door",                # no door members / empty door_refs or seed_door_refs
    "land_missing",           # source LAND absent under footprint/door samples
    "bounds_missing",         # no bounds_gu in the split product
    "preview_missing",        # no existing preview asset in the render library
)
# Source-run components that never enter stamp derivation (recorded by the
# extraction products themselves) get these scoped reasons.
SOURCE_RECORDED_EXCLUSION_REASONS = (
    "doorless_component_source_recorded",
    "access_target_only",
)
EXCLUSION_PRIORITY = {
    reason: index for index, reason in enumerate(EXCLUSION_REASONS)
}
# Audited reasons rank before every derivation reason.
AUDITED_EXCLUSION_PRIORITY = {
    reason: -index - 1 for index, reason in enumerate(AUDITED_EXCLUSION_REASONS)
}

BUILDING_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Priority order: the first keyword family found in the joined
    # destination names wins.  More specific families precede generic ones
    # (e.g. "barracks" before "castle", "trader" before "house").
    ("tavern", ("tavern", "inn", "lodge")),
    ("shop", ("shop", "trader", "goods", "market", "emporium", "store", "warehouse")),
    ("smith", ("smith", "forge")),
    ("guild", ("guild",)),
    ("temple", ("temple", "chapel", "shrine")),
    ("barracks", ("barracks",)),
    ("stable", ("stable", "stables")),
    ("mill", ("mill", "windmill")),
    ("mine", ("mine",)),
    ("keep", ("keep",)),
    ("castle", ("castle",)),
    ("hall", ("hall",)),
    ("manor", ("manor", "estate")),
    ("farm", ("farm",)),
    ("house", ("house", "home", "hut", "den", "cottage")),
)

# Structural-role vocabulary emitted by the kit-role registry pipeline.
SHELL_ROLE = "shell"
ACCESS_ROLE = "access"

_SIZE_CLASSES = ("small", "medium", "large")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a value to canonical, deterministic JSON bytes.

    Sorted keys, two-space indent, no NaN/Infinity, trailing newline — the
    same discipline as the settlement caches, so identical input products
    always produce byte-identical libraries.
    """

    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file, streamed (read-only access)."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vec_sub(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def _vec_add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(a) + float(b) for a, b in zip(left, right)]


def _vec_span(minimum: Sequence[float], maximum: Sequence[float]) -> list[float]:
    return [float(mx) - float(mn) for mn, mx in zip(minimum, maximum)]


def _as_ws_path(path: str | Path) -> str:
    """Normalize a workspace-relative path to forward slashes for JSON."""
    return str(path).replace("\\", "/")


def classify_building_type(destination_names: Sequence[str]) -> str:
    """Infer a planner-facing building type from destination interior names.

    Deterministic keyword table (``BUILDING_TYPE_KEYWORDS``), first match in
    priority order over the casefolded joined names; explicit ``unknown``
    bucket when nothing matches.  Never inferred from the mesh taxonomy.
    """

    joined = " ".join(str(name) for name in destination_names).casefold()
    for family, keywords in BUILDING_TYPE_KEYWORDS:
        for keyword in keywords:
            if keyword in joined:
                return family
    return "unknown"


def size_class_quantiles(areas_gu2: Sequence[float]) -> dict[str, float]:
    """Return nearest-rank 1/3 and 2/3 quantiles of footprint areas.

    With ``n`` sorted areas, ``q(p) = areas[floor(p * (n - 1))]`` (nearest
    rank, deterministic and dependency-free).  ``small`` = area below the 1/3
    quantile, ``large`` = area above the 2/3 quantile, otherwise ``medium``.
    """

    sorted_areas = sorted(float(area) for area in areas_gu2)
    if not sorted_areas:
        return {"small_max_area_gu2": 0.0, "large_min_area_gu2": 0.0, "n": 0}
    n = len(sorted_areas)
    third = int(math.floor((n - 1) / 3.0))
    two_thirds = int(math.floor(2.0 * (n - 1) / 3.0))
    return {
        "small_max_area_gu2": sorted_areas[third],
        "large_min_area_gu2": sorted_areas[two_thirds],
        "n": n,
    }


def size_class_for_area(area_gu2: float, quantiles: Mapping[str, float]) -> str:
    if area_gu2 < float(quantiles["small_max_area_gu2"]):
        return "small"
    if area_gu2 > float(quantiles["large_min_area_gu2"]):
        return "large"
    return "medium"


def convex_hull_xy(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """2D convex hull (monotone chain) of an XY point set.

    Deterministic: points are deduplicated, sorted lexicographically, and the
    hull is returned counter-clockwise starting from the lexicographically
    smallest point.  Degenerate inputs (0/1 points, collinear) return the
    sorted unique points.  Used as the cheap collision oracle for member
    bounds corners.
    """

    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 2:
        return [[x, y] for x, y in unique]

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return [[x, y] for x, y in hull]


def polygon_centroid_xy(points: Sequence[Sequence[float]]) -> list[float]:
    """Area-weighted centroid of a closed polygon (deterministic).

    Uses the standard shoelace-based centroid for a vertex list in order
    (the monotone-chain hull from :func:`convex_hull_xy` is counter-
    clockwise).  Degenerate fallback is explicit: fewer than three distinct
    vertices, or a signed area below ``1e-12`` (collinear/zero-area), returns
    the arithmetic mean of the vertices; an empty point list returns
    ``[0.0, 0.0]``.  For a rectangle this equals the AABB midpoint, so only
    non-rectangular footprints change the measured heading.
    """

    vertices = [(float(x), float(y)) for x, y in points]
    if not vertices:
        return [0.0, 0.0]
    count = len(vertices)
    if count < 3:
        return [sum(v[0] for v in vertices) / count, sum(v[1] for v in vertices) / count]
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for index in range(count):
        x0, y0 = vertices[index]
        x1, y1 = vertices[(index + 1) % count]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area2) < 1e-12:
        return [sum(v[0] for v in vertices) / count, sum(v[1] for v in vertices) / count]
    return [cx / (3.0 * area2), cy / (3.0 * area2)]


def best_fit_plane_slope_deg(samples_xy_z_gu: Sequence[Sequence[float]]) -> float:
    """Slope (degrees) of the least-squares plane through XY-Z samples.

    Fits ``z = a*x + b*y + c`` via the 3x3 normal equations over the sampled
    terrain under the footprint (game units); returns
    ``degrees(atan(sqrt(a*a + b*b)))`` — the maximum rise per horizontal unit.
    Empty or degenerate samples return 0.0.
    """

    if len(samples_xy_z_gu) < 3:
        return 0.0
    n = float(len(samples_xy_z_gu))
    sum_x = sum_y = sum_z = sum_xx = sum_yy = sum_xy = sum_xz = sum_yz = 0.0
    for x, y, z in samples_xy_z_gu:
        x, y, z = float(x), float(y), float(z)
        sum_x += x
        sum_y += y
        sum_z += z
        sum_xx += x * x
        sum_yy += y * y
        sum_xy += x * y
        sum_xz += x * z
        sum_yz += y * z
    # Normal equations (symmetric 3x3), solved with Cramer's rule expanding
    # along the first row.  System:
    # [ n  Sx Sy ] [c]   [Sz ]
    # [Sx Sxx Sxy] [a] = [Sxz]
    # [Sy Sxy Syy] [b]   [Syz]
    m00, m01, m02 = n, sum_x, sum_y
    m11, m12 = sum_xx, sum_xy
    m22 = sum_yy
    b0, b1, b2 = sum_z, sum_xz, sum_yz
    det = (
        m00 * (m11 * m22 - m12 * m12)
        - m01 * (m01 * m22 - m12 * m02)
        + m02 * (m01 * m12 - m11 * m02)
    )
    if abs(det) < 1e-12:
        return 0.0
    # Replace the coefficient-a column (second) with the right-hand side:
    # [[m00 b0 m02], [m01 b1 m12], [m02 b2 m22]]
    det_a = (
        m00 * (b1 * m22 - m12 * b2)
        - b0 * (m01 * m22 - m12 * m02)
        + m02 * (m01 * b2 - b1 * m02)
    )
    # Replace the coefficient-b column (third) with the right-hand side:
    # [[m00 m01 b0], [m01 m11 b1], [m02 m12 b2]]
    det_b = (
        m00 * (m11 * b2 - b1 * m12)
        - m01 * (m01 * b2 - b1 * m02)
        + b0 * (m01 * m12 - m11 * m02)
    )
    a = det_a / det
    b = det_b / det
    return math.degrees(math.atan(math.hypot(a, b)))


def _sample_field(
    land: Mapping[tuple[int, int], espland.LandRecord],
    min_xy: Sequence[float],
    max_xy: Sequence[float],
) -> dict[str, Any]:
    """Deterministic 65x65 terrain field over bbox + 256 GU margin.

    Mirrors the landscape product's sampling (same side/margin/resolution).
    Returns the footprint-only sample list (game units) plus counts of
    missing samples; a single missing sample is reported, never synthesized.
    """

    field = espland.sample_height_field(
        land,
        list(min_xy),
        list(max_xy),
        margin_game_units=HEIGHT_FIELD_MARGIN_GU,
        side=HEIGHT_FIELD_SIDE,
    )
    values_gu = field["values_game_units"]  # type: ignore[index]
    spacing = field["spacing_game_units"]  # type: ignore[index]
    field_min = field["field_bbox_xy_game_units"]["min"]  # type: ignore[index]
    footprint_min = list(min_xy)
    footprint_max = list(max_xy)
    footprint_samples: list[list[float]] = []
    missing = 0
    for row in range(HEIGHT_FIELD_SIDE):
        game_y = float(field_min[1]) + row * float(spacing[1])
        for column in range(HEIGHT_FIELD_SIDE):
            game_x = float(field_min[0]) + column * float(spacing[0])
            value = values_gu[row][column]
            if value is None:
                # Only count misses inside the actual footprint bbox; margin
                # cells beyond the source plugin's LAND scope are acceptable
                # as long as the footprint itself is covered.
                if footprint_min[0] <= game_x <= footprint_max[0] and footprint_min[1] <= game_y <= footprint_max[1]:
                    missing += 1
                continue
            if footprint_min[0] <= game_x <= footprint_max[0] and footprint_min[1] <= game_y <= footprint_max[1]:
                footprint_samples.append([game_x, game_y, float(value)])
    return {
        "footprint_samples_gu": footprint_samples,
        "missing_footprint_samples": missing,
        "field_min_gu": [float(field_min[0]), float(field_min[1])],
        "spacing_gu": [float(spacing[0]), float(spacing[1])],
    }


def derive_terrain_envelope(
    land: Mapping[tuple[int, int], espland.LandRecord],
    min_xy: Sequence[float],
    max_xy: Sequence[float],
    bounds_min_z: float,
    doors_xy_z: Sequence[Sequence[float]],
) -> dict[str, Any] | None:
    """Re-derive the measured terrain envelope from source-plugin LAND.

    * ``door_step_height_gu`` per door = door origin z minus the bilinear LAND
      height at the door XY (``espland.height_at_game_position``, THU to GU).
    * ``footprint_relief_gu`` = max - min terrain height under the footprint
      bbox (65x65 sampled field, 256 GU margin, footprint samples only).
    * ``burial_depth_gu`` = max terrain under the bbox minus the building
      bounds minimum z (same definition as the landscape product).
    * ``footprint_slope_deg`` from the footprint best-fit plane.

    Returns ``None`` when any footprint sample or door sample is missing —
    the caller must exclude the stamp, never invent terrain.
    """

    field = _sample_field(land, min_xy, max_xy)
    samples = field["footprint_samples_gu"]
    if field["missing_footprint_samples"] or not samples:
        return None
    heights = [float(sample[2]) for sample in samples]
    door_steps: list[float] = []
    for door_xy_z in doors_xy_z:
        terrain_thu = espland.height_at_game_position(land, door_xy_z[:2])
        if terrain_thu is None:
            return None
        terrain_gu = float(terrain_thu) * THU_TO_GU
        door_steps.append(float(door_xy_z[2]) - terrain_gu)
    return {
        "door_step_heights_gu": door_steps,
        "footprint_relief_gu": float(max(heights) - min(heights)),
        "footprint_slope_deg": best_fit_plane_slope_deg(samples),
        "burial_depth_gu": float(max(heights)) - float(bounds_min_z),
        "footprint_sample_count": len(samples),
    }


def _member_bounds_corners_xy(world_bounds: Mapping[str, Sequence[float]] | None) -> list[list[float]]:
    if not world_bounds:
        return []
    mn = world_bounds["min"]
    mx = world_bounds["max"]
    return [
        [float(mn[0]), float(mn[1])],
        [float(mx[0]), float(mn[1])],
        [float(mx[0]), float(mx[1])],
        [float(mn[0]), float(mx[1])],
    ]


def _union_bounds(corners_xyz: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    minimum = [min(float(corner[axis]) for corner in corners_xyz) for axis in range(3)]
    maximum = [max(float(corner[axis]) for corner in corners_xyz) for axis in range(3)]
    return {
        "min": minimum,
        "max": maximum,
        "span": _vec_span(minimum, maximum),
    }


# A member record is a plain mapping with these keys:
#   source_id, object_id, record_type, model_key, category, structural_role,
#   is_door, position_gu, rotation, scale, world_bounds_gu (optional),
#   destination (optional mapping with destination_cell/destination_position/
#   destination_rotation/door_to_interior).
MEMBER_KEYS = (
    "source_id",
    "object_id",
    "record_type",
    "model_key",
    "category",
    "structural_role",
    "is_door",
    "position_gu",
    "rotation",
    "scale",
    "world_bounds_gu",
    "destination",
)


def derive_stamp(
    *,
    library_stamp_prefix: str,
    run: str,
    slug: str,
    component_id: int | None,
    seed_door_refs: Sequence[str],
    door_refs: Sequence[str],
    members: Sequence[Mapping[str, Any]],
    bounds_xy: Sequence[float] | None,
    bounds_min_z: float | None,
    named_destination_interiors: Sequence[str],
    multi_shell: bool,
    land: Mapping[tuple[int, int], espland.LandRecord],
    preview_sheet: str | None,
    extra_source: Mapping[str, Any] | None = None,
    matrix_builder: Callable[[Sequence[float], Sequence[float] | None, object], Sequence[Sequence[float]]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Derive exactly one stamp from already-loaded, joined source records.

    Returns ``(stamp_dict, None)`` on success or ``(None, reason)`` on
    exclusion.  Every member must already carry its authoritative placement
    (joined by the caller); missing members are the caller's ``ghost_members``
    concern — this function still guards and reports them.

    All derivation is world-aligned subtraction: offsets in GU relative to the
    seed-door anchor, rotation/scale copied verbatim from source, and the
    terrain envelope re-measured from source LAND.  The optional
    ``matrix_builder`` (the read-only oracle's ``placement_scene_matrix``) is
    used only for the golden-rule replay evidence.
    """

    member_by_id = {member["source_id"]: member for member in members}
    if len(member_by_id) != len(members):
        return None, "ghost_members"  # duplicate member ids: unrecoverable
    seed_door = seed_door_refs[0] if seed_door_refs else None
    if seed_door is None or seed_door not in member_by_id:
        return None, "ghost_members"
    for ref in door_refs:
        if ref not in member_by_id:
            return None, "ghost_members"
    anchor = member_by_id[seed_door]["position_gu"]
    if anchor is None:
        return None, "ghost_members"
    doors = [member_by_id[ref] for ref in door_refs if member_by_id[ref]["is_door"]]
    if not doors:
        return None, "no_door"

    # Member offsets (exact float64 subtraction; no rotation applied).
    # ``ordered_source_members`` keeps the source records in the same
    # canonical order as ``offset_members``/``final_members`` so replay
    # checks pair each row with its own source record.
    ordered_source_members = sorted(
        members, key=lambda item: (0 if item["is_door"] else 1, item["source_id"])
    )
    offset_members: list[dict[str, Any]] = []
    for member in ordered_source_members:
        source_position = member["position_gu"]
        if source_position is None:
            return None, "ghost_members"
        rotation = list(member["rotation"]) if member["rotation"] is not None else [0.0, 0.0, 0.0]
        scale = 1.0 if member["scale"] is None else float(member["scale"])
        offset_members.append(
            {
                "source_id": member["source_id"],
                "object_id": member["object_id"],
                "record_type": member["record_type"],
                "model_key": member["model_key"],
                "category": member.get("category"),
                "structural_role": member.get("structural_role"),
                "is_door": bool(member["is_door"]),
                "offset_gu": _vec_sub(source_position, anchor),
                "rotation": rotation,
                "scale": scale,
            }
        )

    # Footprint: 3D union AABB + 2D hull from member world-bounds corners.
    corners_xyz: list[list[float]] = []
    for member in members:
        world_bounds = member.get("world_bounds_gu")
        if world_bounds:
            mn = world_bounds["min"]
            mx = world_bounds["max"]
            corners_xyz.extend(
                [
                    [mn[0], mn[1], mn[2]],
                    [mx[0], mn[1], mn[2]],
                    [mx[0], mx[1], mx[2]],
                    [mn[0], mx[1], mx[2]],
                ]
            )
    if not corners_xyz:
        return None, "bounds_missing"
    bounds = _union_bounds(corners_xyz)
    hull_xy = convex_hull_xy([[x, y] for x, y, _ in corners_xyz])
    # Access heading uses the actual 2D footprint centroid: the area-weighted
    # centroid of the hull polygon (equals the AABB midpoint only for
    # rectangular hulls; degenerate hulls fall back to the vertex mean).
    centroid_xy = polygon_centroid_xy(hull_xy)
    door_xy = member_by_id[seed_door]["position_gu"][:2]
    access_heading_rad = math.atan2(
        float(door_xy[1]) - centroid_xy[1], float(door_xy[0]) - centroid_xy[0]
    )

    # Terrain envelope from source LAND.
    if bounds_xy is None or bounds_min_z is None:
        return None, "bounds_missing"
    envelope = derive_terrain_envelope(
        land,
        [bounds_xy[0], bounds_xy[1]],
        [bounds_xy[2], bounds_xy[3]],
        bounds_min_z,
        [member_by_id[ref]["position_gu"] for ref in door_refs],
    )
    if envelope is None:
        return None, "land_missing"

    # Per-door provenance (placement manifest join, provenance only).
    door_entries: list[dict[str, Any]] = []
    for ref in door_refs:
        door = member_by_id[ref]
        destination = door.get("destination") or {}
        door_entries.append(
            {
                "source_id": ref,
                "object_id": door["object_id"],
                "record_type": "DOOR" if door["record_type"] == "DOOR" else door["record_type"],
                "model_key": door["model_key"],
                "offset_gu": _vec_sub(door["position_gu"], anchor),
                "rotation": list(door["rotation"]) if door["rotation"] is not None else [0.0, 0.0, 0.0],
                "scale": 1.0 if door["scale"] is None else float(door["scale"]),
                "step_height_gu": envelope["door_step_heights_gu"][
                    list(door_refs).index(ref)
                ],
                "destination_cell": destination.get("destination_cell"),
                "destination_position_gu": destination.get("destination_position"),
                "destination_rotation": destination.get("destination_rotation"),
            }
        )

    # Build the door sub-objects into the member rows (canonical: door
    # metadata appears on the door member rows).
    door_by_id = {entry["source_id"]: entry for entry in door_entries}
    final_members: list[dict[str, Any]] = []
    for member_row in offset_members:
        row = dict(member_row)
        if member_row["is_door"]:
            door_info = door_by_id[member_row["source_id"]]
            row["door"] = {
                "step_height_gu": door_info["step_height_gu"],
                "destination_cell": door_info["destination_cell"],
                "destination_position_gu": door_info["destination_position_gu"],
                "destination_rotation": door_info["destination_rotation"],
            }
        final_members.append(row)

    relative_bounds = {
        "min": _vec_sub(bounds["min"], anchor),
        "max": _vec_sub(bounds["max"], anchor),
        "span": bounds["span"],
    }
    footprint_aabb_rel = {
        "min": _vec_sub(bounds["min"], anchor),
        "max": _vec_sub(bounds["max"], anchor),
        "span": bounds["span"],
    }
    hull_xy_rel = [[float(x) - float(anchor[0]), float(y) - float(anchor[1])] for x, y in hull_xy]

    # Replay evidence for this stamp (position reconstruction + verbatim
    # rotation/scale equality; oracle matrix check is optional).
    max_pos_error = 0.0
    rotation_mismatches = 0
    scale_mismatches = 0
    multi_axis = 0
    for member, member_row in zip(ordered_source_members, final_members):
        reconstructed = _vec_add(anchor, member_row["offset_gu"])
        max_pos_error = max(max_pos_error, max(abs(a - b) for a, b in zip(reconstructed, member["position_gu"])))
        source_rotation = member["rotation"] if member["rotation"] is not None else [0.0, 0.0, 0.0]
        if any(abs(a - b) > 1e-12 for a, b in zip(member_row["rotation"], source_rotation)):
            rotation_mismatches += 1
        source_scale = 1.0 if member["scale"] is None else float(member["scale"])
        if abs(member_row["scale"] - source_scale) > 1e-12:
            scale_mismatches += 1
        if abs(source_rotation[0]) > 1e-9 or abs(source_rotation[1]) > 1e-9:
            multi_axis += 1

    stamp = {
        "stamp_id": f"{library_stamp_prefix}__{slug}",
        "source": {
            "run": run,
            "slug": slug,
            "component_id": component_id,
            "seed_door": seed_door,
            **(extra_source or {}),
        },
        "preview_sheet": preview_sheet,
        "building_type": classify_building_type(named_destination_interiors),
        "door_count": len(door_refs),
        "multi_shell": bool(multi_shell),
        "anchor": {
            "kind": "seed_door",
            "seed_door": seed_door,
            "source_position_gu": [float(value) for value in anchor],
        },
        "access_heading_rad": access_heading_rad,
        "members": final_members,
        "footprint": {
            "aabb_rel": footprint_aabb_rel,
            "hull_xy_rel": hull_xy_rel,
        },
        "terrain_envelope": {
            "door_step_heights_gu": envelope["door_step_heights_gu"],
            "footprint_relief_gu": envelope["footprint_relief_gu"],
            "footprint_slope_deg": envelope["footprint_slope_deg"],
            "burial_depth_gu": envelope["burial_depth_gu"],
        },
        "bounds_rel_gu": relative_bounds,
        "replay": {
            "max_abs_position_error_gu": max_pos_error,
            "rotation_mismatches": rotation_mismatches,
            "scale_mismatches": scale_mismatches,
            "multi_axis_member_count": multi_axis,
        },
    }
    if matrix_builder is not None:
        matrix_mismatches = 0
        matrix_max_linear = 0.0
        matrix_max_translation_gu = 0.0
        for member, member_row in zip(ordered_source_members, final_members):
            expected = member.get("source_placement_scene_matrix")
            if not expected:
                continue
            built = matrix_builder(
                _vec_add(anchor, member_row["offset_gu"]),
                member_row["rotation"],
                member_row["scale"],
            )
            linear_error = max(
                abs(float(built[row][col]) - float(expected[row][col]))
                for row in range(3)
                for col in range(3)
            )
            translation_error_gu = max(
                abs(float(built[row][3]) * 100.0 - float(expected[row][3]) * 100.0)
                for row in range(3)
            )
            matrix_max_linear = max(matrix_max_linear, linear_error)
            matrix_max_translation_gu = max(matrix_max_translation_gu, translation_error_gu)
            if linear_error > 0.0001 or translation_error_gu > 0.01:
                matrix_mismatches += 1
        stamp["replay"]["oracle_matrix"] = {
            "checked_members": sum(
                1 for member in members if member.get("source_placement_scene_matrix")
            ),
            "max_linear_error": matrix_max_linear,
            "max_translation_error_gu": matrix_max_translation_gu,
            "mismatches": matrix_mismatches,
            "tolerances": {
                "linear_matrix_element": 0.0001,
                "translation_game_units": 0.01,
            },
        }
    return stamp, None


def _excluded_row(
    candidate_id: str,
    reason: str,
    detail: Mapping[str, Any] | None = None,
    *,
    scope: str = "split_subset",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "reason": reason,
        "scope": scope,
    }
    if detail:
        row.update({str(key): value for key, value in detail.items()})
    return row


def _pick_exclusion(
    candidate_id: str,
    conditions: Mapping[str, bool],
    detail: Mapping[str, Any] | None = None,
    *,
    scope: str = "split_subset",
) -> dict[str, Any] | None:
    """Return the highest-priority satisfied exclusion row, or ``None``.

    Audited review reasons (``AUDITED_EXCLUSION_REASONS``) are checked first:
    they are deterministic overrides pinned to the hash-verified source, so
    they dominate any derivation failure.
    """
    for reason in AUDITED_EXCLUSION_REASONS:
        if conditions.get(reason):
            return _excluded_row(candidate_id, reason, detail, scope=scope)
    for reason in EXCLUSION_REASONS:
        if conditions.get(reason):
            return _excluded_row(candidate_id, reason, detail, scope=scope)
    return None


def build_library(
    *,
    library_id: str,
    library_name: str,
    kit: Mapping[str, Any],
    stamp_prefix: str,
    inputs: Mapping[str, str],
    source_plugins: Mapping[str, Mapping[str, str]],
    land: Mapping[tuple[int, int], espland.LandRecord],
    candidates: Sequence[Mapping[str, Any]],
    source_recorded_exclusions: Sequence[Mapping[str, Any]] = (),
    extra_stats: Mapping[str, Any] | None = None,
    matrix_builder: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Derive a complete library from loaded candidates and write nothing.

    ``candidates`` entries must carry the ``derive_stamp`` keyword inputs as
    keys plus ``candidate_id`` (for the exclusion ledger) and optional
    ``exclusion_conditions`` (pre-validated conditions such as ghost members,
    missing previews) and ``exclusion_detail``.

    Deterministic: stamps sorted by ``stamp_id``; the exclusion ledger sorted
    by ``(reason, candidate_id)``; every accepted stamp gets a size class from
    the library's own footprint-area quantiles.
    """

    stamps: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        conditions = dict(candidate.get("exclusion_conditions") or {})
        conditions.setdefault("preview_missing", not candidate.get("preview_sheet"))
        excluded_row = _pick_exclusion(candidate_id, conditions, candidate.get("exclusion_detail"))
        if excluded_row is not None:
            excluded.append(excluded_row)
            continue
        stamp, reason = derive_stamp(
            library_stamp_prefix=stamp_prefix,
            run=candidate["run"],
            slug=candidate["slug"],
            component_id=candidate.get("component_id"),
            seed_door_refs=candidate["seed_door_refs"],
            door_refs=candidate["door_refs"],
            members=candidate["members"],
            bounds_xy=candidate.get("bounds_xy"),
            bounds_min_z=candidate.get("bounds_min_z"),
            named_destination_interiors=candidate.get("named_destination_interiors") or [],
            multi_shell=bool(candidate.get("multi_shell")),
            land=land,
            preview_sheet=candidate.get("preview_sheet"),
            extra_source=candidate.get("extra_source"),
            matrix_builder=matrix_builder,
        )
        if stamp is None:
            excluded.append(
                _excluded_row(
                    candidate_id,
                    reason or "unknown",
                    candidate.get("exclusion_detail"),
                )
            )
            continue
        stamps.append(stamp)

    stamps.sort(key=lambda stamp: stamp["stamp_id"])
    excluded.sort(key=lambda row: (row["reason"], row["candidate_id"]))

    quantiles = size_class_quantiles(
        [
            float(stamp["footprint"]["aabb_rel"]["span"][0])
            * float(stamp["footprint"]["aabb_rel"]["span"][1])
            for stamp in stamps
        ]
    )
    for stamp in stamps:
        area = float(stamp["footprint"]["aabb_rel"]["span"][0]) * float(
            stamp["footprint"]["aabb_rel"]["span"][1]
        )
        stamp["size_class"] = size_class_for_area(area, quantiles)
        stamp["style_tags"] = list(kit.get("style_tags") or [])

    per_type: dict[str, int] = {}
    per_size: dict[str, int] = {}
    door_count_total = 0
    multi_shell_count = 0
    multi_axis_members = 0
    max_pos_error = 0.0
    rotation_mismatches = 0
    scale_mismatches = 0
    matrix_checked = 0
    matrix_mismatches = 0
    matrix_max_linear = 0.0
    matrix_max_translation_gu = 0.0
    for stamp in stamps:
        per_type[stamp["building_type"]] = per_type.get(stamp["building_type"], 0) + 1
        per_size[stamp["size_class"]] = per_size.get(stamp["size_class"], 0) + 1
        door_count_total += stamp["door_count"]
        multi_shell_count += 1 if stamp["multi_shell"] else 0
        replay = stamp["replay"]
        multi_axis_members += replay["multi_axis_member_count"]
        max_pos_error = max(max_pos_error, replay["max_abs_position_error_gu"])
        rotation_mismatches += replay["rotation_mismatches"]
        scale_mismatches += replay["scale_mismatches"]
        oracle = replay.get("oracle_matrix")
        if oracle:
            matrix_checked += oracle["checked_members"]
            matrix_mismatches += oracle["mismatches"]
            matrix_max_linear = max(matrix_max_linear, oracle["max_linear_error"])
            matrix_max_translation_gu = max(
                matrix_max_translation_gu, oracle["max_translation_error_gu"]
            )

    stats: dict[str, Any] = {
        "stamp_count": len(stamps),
        "per_type": dict(sorted(per_type.items())),
        "per_size_class": dict(sorted(per_size.items())),
        "size_class_quantiles": quantiles,
        "door_count_total": door_count_total,
        "multi_shell_count": multi_shell_count,
        "excluded_count": len(excluded),
        "excluded": excluded,
        "replay": {
            "position_tolerance_gu": POSITION_REPLAY_TOLERANCE_GU,
            "members_checked": sum(len(stamp["members"]) for stamp in stamps),
            "max_abs_position_error_gu": max_pos_error,
            "rotation_mismatches": rotation_mismatches,
            "scale_mismatches": scale_mismatches,
            "multi_axis_member_count": multi_axis_members,
            "has_multi_axis_canary": multi_axis_members > 0,
            "oracle_matrix": {
                "checked_members": matrix_checked,
                "max_linear_error": matrix_max_linear,
                "max_translation_error_gu": matrix_max_translation_gu,
                "mismatches": matrix_mismatches,
                "tolerances": {
                    "linear_matrix_element": 0.0001,
                    "translation_game_units": 0.01,
                },
            },
        },
    }
    if extra_stats:
        stats.update(extra_stats)

    library: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "library_id": library_id,
        "library_name": library_name,
        "generated_by": f"citystamps {__version__} (Cityforge T0.3)",
        "kit": dict(kit),
        "inputs": dict(sorted(inputs.items())),
        "source_plugins": dict(sorted(source_plugins.items())),
        "units": "game units (GU); rotations TES3 Euler radians (source-authored)",
        "stamps": stamps,
        "stats": stats,
    }
    if source_recorded_exclusions:
        library["stats"]["source_recorded_exclusions"] = sorted(
            source_recorded_exclusions, key=lambda row: (row["reason"], row["candidate_id"])
        )
    return library


__all__ = [
    "ACCESS_ROLE",
    "AUDITED_EXCLUSION_PRIORITY",
    "AUDITED_EXCLUSION_REASONS",
    "BUILDING_TYPE_KEYWORDS",
    "EXCLUSION_PRIORITY",
    "EXCLUSION_REASONS",
    "HEIGHT_FIELD_MARGIN_GU",
    "HEIGHT_FIELD_SIDE",
    "POSITION_REPLAY_TOLERANCE_GU",
    "SCHEMA_VERSION",
    "SHELL_ROLE",
    "SOURCE_RECORDED_EXCLUSION_REASONS",
    "TERRAIN_CROSSCHECK_TOLERANCE_GU",
    "THU_TO_GU",
    "__version__",
    "best_fit_plane_slope_deg",
    "build_library",
    "canonical_json_bytes",
    "classify_building_type",
    "convex_hull_xy",
    "derive_stamp",
    "derive_terrain_envelope",
    "polygon_centroid_xy",
    "sha256_bytes",
    "sha256_file",
    "size_class_for_area",
    "size_class_quantiles",
]
