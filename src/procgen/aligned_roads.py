"""Aligned Tamriel road centerlines: the single supported consumer contract.

Pipeline position
------------------
This module is the **only** supported planner/generator entry point for road
geometry.  It consumes the committed, aligned canonical product

    output/mapdata/roads/tamriel_aligned_centerlines_v1/

which is derived from the accepted source-space bundle

    output/mapdata/roads/tamriel_source_centerlines_v1/

by applying exactly ``(+4096 GU, +0 GU)`` to every world coordinate (the
measured registration correction against ``tamriel.esm`` LAND/VTEX-78; see
``.opencode/runs/cityforge-road-authority-alignment/``).  Consumers must
never read the source-space bundle coordinates or the XCF/BMP rasters; the
source bundle is topology/provenance storage only.

What is here
------------
* Pinned contract constants (source bundle hashes, ``tamriel.esm`` hash,
  topology counts, the correction vector, transform equations).
* :class:`AlignedNetwork` -- the loaded product with per-id lookup, world-GU
  rectangle queries, site-local frame conversion, corridor width/provenance,
  nearest centerline point/tangent/distance, and corridor polygons for plan
  collision checks.
* ``load_aligned_network()`` -- the fail-closed loader: it refuses
  source-space paths, verifies the manifest/hash chain, the declared
  translation, topology counts, and per-coordinate pixel round-trip
  invariants.
* Direct-LAND helpers used by the build CLI and tests:
  ``load_esm78_tiles``, ``nearest_road_tile_distance``,
  ``skeleton_registration_stats``, ``edge_corridor_report``,
  ``registration_agreement``.  These read ``tamriel.esm`` through
  :mod:`procgen.espland` (read-only) and never open the XCF/BMP.

Invariants (binding)
--------------------
* The aligned product is derived from the pinned source bundle hash
  ``057d5853...``; any other source hash is drift and fails closed.
* Translation is exactly ``dx_gu = +4096``, ``dy_gu = +0``; the loader
  verifies the product's declared alignment and the pixel round-trip of every
  node/edge coordinate at the corrected registration.
* Topology is identical to the source: 3847 nodes / 4142 edges, same stable
  IDs (node/edge/component/bridge) -- the loader pins these counts.
* No numpy and no third-party dependencies; this module is stdlib-only so the
  T1.1 validator can import it without new installs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .espland import LAND_TEXTURE_SIDE, iter_land

# ---------------------------------------------------------------------------
# Pinned contract (measured 2026-08-11; see the road-authority investigation)
# ---------------------------------------------------------------------------

#: Alignment product name and loader-recognized alignment version marker.
ALIGNMENT_VERSION = "alignment_v1"
PRODUCT_CANONICAL_NAME = "tamriel_aligned_centerlines_v1.json"
MANIFEST_NAME = "alignment_manifest.json"

#: The measured rigid registration correction (world GU) applied to every
#: source coordinate.  Equivalently the source raster is registered 8 px
#: (+8 site tiles of 512 GU) east of the in-game LAND/VTEX grid.
ALIGNMENT_DX_GU = 4096
ALIGNMENT_DY_GU = 0

#: Topology counts of the accepted source bundle (must never drift).
SOURCE_NODE_COUNT = 3847
SOURCE_EDGE_COUNT = 4142

#: SHA-256 of the committed source canonical bundle (pinned).
SOURCE_CANONICAL_SHA256 = "057d5853dbcff3d3b68933fe1493d10ce9762b694f342d4fd8b272ba72e0faef"
#: SHA-256 of the committed source audit document (pinned).
SOURCE_AUDIT_SHA256 = "fbcfc17aec8b56d8428ada6ce96ab393aaf12f143fee4139e09e4aa61767bc53"
#: SHA-256 of the source effective-alpha array content (pinned).
SOURCE_EFFECTIVE_ALPHA_SHA256 = "845cd301d5f349b47baa455fea3ee46663e81da4e458a221a808f18292df24fe"
#: SHA-256 of the read-only ``tamriel.esm`` used for the registration proof.
TAMRIEL_ESM_SHA256 = "9f8f3ce92dfd198bc54f5f5d46dd63b850b714a414304a305eaeec5b423bdc01"

#: Raw VTEX class that is the in-game road identity (LTEX[77]).
RAW_VTEX_ROAD = 78
#: TES3 world tile size in GU (LAND VTEX tile).
TILE_SIZE_GU = 512
#: TES3 cell size in GU.
CELL_SIZE_GU = 8192

#: Falkreath 7x7 site window, inclusive cell bounds (x, y).
FALKREATH_CELL_BOUNDS = (-95, -89, -11, -5)
#: Five separated Falkreath canary junctions from the investigation; each
#: must sit at exactly 0 GU distance to a LAND road tile after correction.
FALKREATH_CANARY_NODE_IDS = (
    "road_node_4f749ce466e2528b",
    "road_node_214e87cebd5e7392",
    "road_node_93a5d39d6461a86d",
    "road_node_0cabe8ee6fe6c6a0",
    "road_node_fe5ab61f1218c960",
)

#: Corrected (aligned) pixel-center -> world GU equations.
#:   GU_x = (px - 4055.5) * 512          GU_y = (959.5 - py) * 512
#: Inverse (used by the loader's coordinate invariant):
#:   px = GU_x / 512 + 4055.5            py = 959.5 - GU_y / 512
_PX_TO_GU_X_OFFSET = -4055.5
_PX_TO_GU_Y_OFFSET = 959.5

#: Tolerance (GU) for the pixel round-trip invariant.  Raw-chain coordinates
#: are exact multiples of 512 and round-trip exactly; smooth polylines were
#: sampled at fractional pixel positions, so a tiny float tolerance applies.
ROUND_TRIP_TOLERANCE_GU = 1e-6

#: Neighborhood radius (cells) used for nearest-road-tile distance queries;
#: covers the 8-tile (4096 GU) zero-shift displacement case.
_NEIGHBORHOOD_CELLS = 4


class AlignedRoadsError(Exception):
    """Fatal aligned-road product problem (fail-closed contract violation).

    Raised by the loader and the direct-LAND helpers whenever the product,
    manifest, hashes, translation, topology counts, or coordinate invariants
    drift, or when a consumer points at the source-space bundle.
    """


# ---------------------------------------------------------------------------
# Path guards
# ---------------------------------------------------------------------------

def _canonical_paths(product_dir: Path) -> tuple[Path, Path]:
    return product_dir / PRODUCT_CANONICAL_NAME, product_dir / MANIFEST_NAME


def is_source_space_path(path: str | Path) -> bool:
    """True when ``path`` points into the source-space v1 bundle.

    Consumers must never load road geometry from the source bundle; the
    loader uses this guard to fail closed.
    """
    resolved = Path(path).resolve()
    return "tamriel_source_centerlines_v1" in resolved.parts or (
        resolved.name == "tamriel_road_centerlines_v1.json"
    )


def is_aligned_product_path(path: str | Path) -> bool:
    """True when ``path`` names the aligned canonical product (dir or file)."""
    resolved = Path(path).resolve()
    if resolved.name == PRODUCT_CANONICAL_NAME:
        return "tamriel_aligned_centerlines_v1" in resolved.parent.parts
    return resolved.name == "tamriel_aligned_centerlines_v1"


def resolve_product_dir(path: str | Path | None) -> Path:
    """Resolve a consumer-supplied path to the aligned product directory.

    Accepts the product directory itself or its canonical JSON file.  Any
    source-space path (directory ``tamriel_source_centerlines_v1`` or file
    ``tamriel_road_centerlines_v1.json``) is refused with
    :class:`AlignedRoadsError` -- the fail-closed source-space gate.
    """
    if path is None:
        return DEFAULT_ALIGNED_PRODUCT_DIR
    resolved = Path(path).resolve()
    if is_source_space_path(resolved):
        raise AlignedRoadsError(
            f"refusing source-space road bundle as consumer input: {resolved}. "
            "Consumers must use the aligned product "
            "(output/mapdata/roads/tamriel_aligned_centerlines_v1) through "
            "load_aligned_network(); the source bundle is topology/provenance "
            "storage only."
        )
    if resolved.is_dir():
        product_dir = resolved
    elif resolved.name == PRODUCT_CANONICAL_NAME:
        product_dir = resolved.parent
    else:
        raise AlignedRoadsError(
            f"expected the aligned product directory or {PRODUCT_CANONICAL_NAME}, "
            f"got {resolved}"
        )
    if not is_aligned_product_path(product_dir):
        raise AlignedRoadsError(
            f"product directory name must be tamriel_aligned_centerlines_v1, "
            f"got {product_dir}"
        )
    return product_dir


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def aligned_gu_from_px(px: float, py: float) -> tuple[float, float]:
    """World GU at the corrected registration for source pixel center (px,py).

    ``GU_x = (-254*16 + px + 8 + 0.5) * 512 = (px - 4055.5) * 512`` and
    ``GU_y = (959.5 - py) * 512`` (origin cell (-254,-130), 16 px/cell,
    512 GU/px, row 0 north, +8 px registration correction).
    """
    return ((px - 4055.5) * 512.0,
            (959.5 - py) * 512.0)


def px_from_aligned_gu(x: float, y: float) -> tuple[float, float]:
    """Inverse of :func:`aligned_gu_from_px` (pixel center coordinates)."""
    return (x / 512.0 + 4055.5, 959.5 - y / 512.0)


def world_to_tile(x: float, y: float) -> tuple[int, int, int, int]:
    """World GU -> (cell_x, cell_y, tile_x, tile_y) using floor semantics."""
    cell_x = math.floor(x / CELL_SIZE_GU)
    cell_y = math.floor(y / CELL_SIZE_GU)
    tile_x = math.floor((x - cell_x * CELL_SIZE_GU) / TILE_SIZE_GU)
    tile_y = math.floor((y - cell_y * CELL_SIZE_GU) / TILE_SIZE_GU)
    return cell_x, cell_y, tile_x, tile_y


def tile_rect(cell_x: int, cell_y: int, tile_x: int, tile_y: int) -> tuple[float, float, float, float]:
    """World-GU rect (min_x, min_y, max_x, max_y) of one 512-GU tile."""
    x0 = cell_x * CELL_SIZE_GU + tile_x * TILE_SIZE_GU
    y0 = cell_y * CELL_SIZE_GU + tile_y * TILE_SIZE_GU
    return x0, y0, x0 + TILE_SIZE_GU, y0 + TILE_SIZE_GU


def point_rect_distance(x: float, y: float, rect: Sequence[float]) -> float:
    """Distance from a point to an axis-aligned rect (0 inside)."""
    min_x, min_y, max_x, max_y = rect
    dx = max(min_x - x, 0.0, x - max_x)
    dy = max(min_y - y, 0.0, y - max_y)
    return math.hypot(dx, dy)


#: Length threshold (fraction of the segment) below which a Liang-Barsky
#: clip result is treated as a single contact point rather than a span.
_SEGMENT_RECT_POINT_EPSILON = 1e-12


def segment_intersects_rect(
    a: Sequence[float],
    b: Sequence[float],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> bool:
    """True when segment ``ab`` has a point inside ``[x_min, x_max) x [y_min, y_max)``.

    Deterministic true-intersection semantics for :meth:`AlignedNetwork.edges_in_rect`:

    1. the segment is clipped to the **closed** rectangle with Liang-Barsky;
       an empty clip means no intersection;
    2. a degenerate (single-contact-point) clip is accepted only when that
       point satisfies the half-open membership test (so touching the
       inclusive corner ``(x_min, y_min)`` counts, touching the exclusive
       corner ``(x_max, y_max)`` does not);
    3. a non-degenerate clip is accepted when any of its two end points or
       its midpoint satisfies the half-open membership test.  A span lying
       entirely on an exclusive boundary (e.g. collinear with
       ``y = y_max``) fails all three samples and is excluded; a span that
       crosses the rectangle, or touches an inclusive boundary or corner,
       is caught by an endpoint or midpoint sample.

    A segment that merely *touches* the closed rectangle at a point outside
    the half-open interior (or runs along ``x == x_max`` / ``y == y_max``)
    returns ``False``; a segment whose vertices are all outside but which
    passes through the rectangle returns ``True``.
    """
    if len(a) != 2 or len(b) != 2:
        raise ValueError("segment endpoints must be [x, y] pairs")
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    # Liang-Barsky against the closed rectangle [x_min, x_max] x [y_min, y_max].
    for p, q in ((-dx, ax - x_min), (dx, x_max - ax),
                 (-dy, ay - y_min), (dy, y_max - ay)):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        ratio = q / p
        if p < 0.0:
            if ratio > t1:
                return False
            if ratio > t0:
                t0 = ratio
        else:
            if ratio < t0:
                return False
            if ratio < t1:
                t1 = ratio
    if t1 < t0:
        return False
    if t1 - t0 <= _SEGMENT_RECT_POINT_EPSILON:
        px = ax + t0 * dx
        py = ay + t0 * dy
        return x_min <= px < x_max and y_min <= py < y_max
    # Non-degenerate clip: the clipped span lies inside the CLOSED rectangle;
    # it intersects the half-open rectangle iff any of its end points or its
    # midpoint satisfies the half-open membership test.  The midpoint can
    # equal an exclusive boundary only when the entire span lies on that
    # boundary (e.g. a segment collinear with y == y_max), which the
    # endpoint samples then also fail; a span that merely touches an
    # inclusive corner is caught by its boundary endpoint.
    for t in (t0, 0.5 * (t0 + t1), t1):
        px = ax + t * dx
        py = ay + t * dy
        if x_min <= px < x_max and y_min <= py < y_max:
            return True
    return False


# ---------------------------------------------------------------------------
# Direct tamriel.esm LAND/VTEX-78 evidence (read-only)
# ---------------------------------------------------------------------------

def load_esm78_tiles(
    esm_path: str | Path,
    cells: Optional[Sequence[Sequence[int]]] = None,
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Return raw-VTEX-78 occupancy directly from ``tamriel.esm`` LAND.

    Result maps ``(cell_x, cell_y)`` -> set of ``(tile_x, tile_y)`` in
    OpenMW-normalized row-major order (via :func:`procgen.espland.iter_land`).
    With ``cells=None`` the full map is scanned; otherwise only the given
    inclusive ``[min_x, max_x, min_y, max_y]`` window is retained.  A missing
    VTEX payload is an error: silent emptiness would corrupt the proof.
    """
    tiles: dict[tuple[int, int], set[tuple[int, int]]] = {}
    bounds: Optional[tuple[int, int, int, int]] = None
    if cells is not None:
        if len(cells) != 4:
            raise ValueError("cells must be [min_x, max_x, min_y, max_y]")
        bounds = tuple(int(v) for v in cells)  # type: ignore[assignment]
    for record in iter_land(esm_path):
        if record.texture_indices is None:
            raise AlignedRoadsError(
                f"LAND {record.grid} has no VTEX payload; direct-LAND proof "
                "cannot treat missing data as non-road"
            )
        cx, cy = record.grid
        if bounds is not None and not (
            bounds[0] <= cx <= bounds[1] and bounds[2] <= cy <= bounds[3]
        ):
            continue
        cell_set = tiles.setdefault((cx, cy), set())
        for tile_y in range(LAND_TEXTURE_SIDE):
            base = tile_y * LAND_TEXTURE_SIDE
            for tile_x in range(LAND_TEXTURE_SIDE):
                if record.texture_indices[base + tile_x] == RAW_VTEX_ROAD:
                    cell_set.add((tile_x, tile_y))
    return tiles


def esm78_tile_count(tiles: Mapping[tuple[int, int], Any]) -> int:
    """Total occupied-tile count across the loaded tile map."""
    return sum(len(members) for members in tiles.values())


def nearest_road_tile_distance(
    x: float, y: float, tiles: Mapping[tuple[int, int], set[tuple[int, int]]]
) -> float | None:
    """Nearest distance (GU) from (x, y) to any occupied road tile.

    ``0.0`` means the point lies inside a road tile.  ``None`` means no road
    tile exists within the ``_NEIGHBORHOOD_CELLS`` cell radius (the query
    bound covers the 8-tile zero-shift displacement).
    """
    cx, cy, _, _ = world_to_tile(x, y)
    best: float | None = None
    for off_y in range(-_NEIGHBORHOOD_CELLS, _NEIGHBORHOOD_CELLS + 1):
        for off_x in range(-_NEIGHBORHOOD_CELLS, _NEIGHBORHOOD_CELLS + 1):
            members = tiles.get((cx + off_x, cy + off_y))
            if not members:
                continue
            for tile_x, tile_y in members:
                distance = point_rect_distance(
                    x, y, tile_rect(cx + off_x, cy + off_y, tile_x, tile_y)
                )
                if distance == 0.0:
                    return 0.0
                if best is None or distance < best:
                    best = distance
    return best


def registration_stats(
    points: Iterable[Sequence[float]],
    tiles: Mapping[tuple[int, int], set[tuple[int, int]]],
    *,
    dx_gu: float,
) -> dict[str, Any]:
    """Register a point set against the direct LAND road tiles.

    Every point is shifted by ``dx_gu`` in world X (the correction being
    tested) and classified as inside (distance 0) or outside an occupied
    tile.  Returns deterministic counts plus mean/max nearest distance for
    outside points.  Used for the +4096 GU registration gate and the
    no-shift canary (``dx_gu=0`` must not register).
    """
    total = 0
    inside = 0
    outside_distances: list[float] = []
    unresolved = 0
    for point in points:
        total += 1
        distance = nearest_road_tile_distance(point[0] + dx_gu, point[1], tiles)
        if distance == 0.0:
            inside += 1
        elif distance is None:
            unresolved += 1
        else:
            outside_distances.append(distance)
    outside = len(outside_distances)
    return {
        "dx_gu": float(dx_gu),
        "point_count": total,
        "inside_tile_count": inside,
        "outside_tile_count": outside,
        "no_tile_within_4_cells_count": unresolved,
        "inside_fraction": round(inside / total, 6) if total else None,
        "outside_mean_distance_gu": (
            round(sum(outside_distances) / outside, 3) if outside else None
        ),
        "outside_max_distance_gu": round(max(outside_distances), 3) if outside else None,
    }


def _network_point_sets(network: "AlignedNetwork") -> tuple[list[list[float]], list[list[float]]]:
    node_positions = [list(node.position_gu) for node in network.nodes.values()]
    raw_points: list[list[float]] = []
    for edge in network.edges.values():
        raw_points.extend(list(point) for point in edge.raw_gu_chain)
    return node_positions, raw_points


def skeleton_registration_stats(
    network: "AlignedNetwork",
    tiles: Mapping[tuple[int, int], set[tuple[int, int]]],
    *,
    dx_gu: float,
) -> dict[str, Any]:
    """Registration stats over all node positions and raw-chain points."""
    node_positions, raw_points = _network_point_sets(network)
    return {
        "nodes": registration_stats(node_positions, tiles, dx_gu=dx_gu),
        "raw_chain_points": registration_stats(raw_points, tiles, dx_gu=dx_gu),
    }


def edge_corridor_report(
    network: "AlignedNetwork",
    tiles: Mapping[tuple[int, int], set[tuple[int, int]]],
) -> dict[str, Any]:
    """Per-edge raw-chain agreement with LAND corridors.

    Repaired (bridge-carrying) edges are reported separately: their spans
    intentionally cross source-painted gaps and are *not* required to occupy
    source-painted tiles (they are topology repair, not paint evidence).
    """
    rows: list[dict[str, Any]] = []
    for edge in network.edges.values():
        inside = 0
        outside_distances: list[float] = []
        unresolved = 0
        for point in edge.raw_gu_chain:
            distance = nearest_road_tile_distance(point[0], point[1], tiles)
            if distance == 0.0:
                inside += 1
            elif distance is None:
                unresolved += 1
            else:
                outside_distances.append(distance)
        total = len(edge.raw_gu_chain)
        rows.append(
            {
                "edge_id": edge.id,
                "source_status": edge.source_status,
                "bridge_ids": list(edge.bridge_ids),
                "point_count": total,
                "inside_tile_count": inside,
                "outside_tile_count": len(outside_distances),
                "no_tile_within_4_cells_count": unresolved,
                "inside_fraction": round(inside / total, 6) if total else None,
                "outside_mean_distance_gu": (
                    round(sum(outside_distances) / len(outside_distances), 3)
                    if outside_distances
                    else None
                ),
                "outside_max_distance_gu": (
                    round(max(outside_distances), 3) if outside_distances else None
                ),
            }
        )
    rows.sort(key=lambda row: row["edge_id"])
    repaired = [row for row in rows if row["bridge_ids"]]
    source_only = [row for row in rows if not row["bridge_ids"]]

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        total = sum(row["point_count"] for row in group)
        inside = sum(row["inside_tile_count"] for row in group)
        return {
            "edge_count": len(group),
            "point_count": total,
            "inside_tile_count": inside,
            "inside_fraction": round(inside / total, 6) if total else None,
        }

    return {
        "source_derived_edges": summarize(source_only),
        "repaired_bridge_edges": summarize(repaired),
        "edge_rows": rows,
    }


def registration_agreement(
    tiles: Mapping[tuple[int, int], set[tuple[int, int]]],
    points: Iterable[Sequence[float]],
    *,
    dx_gu: float,
) -> tuple[int, int]:
    """Count (matching, total) for a point set at a candidate shift.

    A point matches when it lies inside an occupied tile after the shift.
    The no-shift canary (``dx_gu=0``) must return a strict minority; the
    corrected shift (+4096) must return exact/near-exact agreement.
    """
    matching = 0
    total = 0
    for point in points:
        total += 1
        if nearest_road_tile_distance(point[0] + dx_gu, point[1], tiles) == 0.0:
            matching += 1
    return matching, total


# ---------------------------------------------------------------------------
# Loaded network model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignedNode:
    """One node of the aligned network (stable source ID, aligned position)."""

    id: str
    component_id: str
    degree: int
    kind: str
    position_gu: tuple[float, float]          # aligned world GU
    position_px: tuple[int, int]              # identical to source v1
    skeleton_pixels: tuple[tuple[int, int], ...]
    synthetic_loop_anchor: bool

    @property
    def aligned_position_gu(self) -> tuple[float, float]:
        """Alias kept for API clarity: this product is the aligned view."""
        return self.position_gu


@dataclass(frozen=True)
class AlignedEdge:
    """One edge of the aligned network (stable source ID, aligned chains)."""

    id: str
    from_node: str
    to_node: str
    component_id: str
    source_status: str
    bridge_ids: tuple[str, ...]
    estimated_width_gu: float
    width_gu_p10: float | None
    width_gu_p90: float | None
    raw_gu_chain: tuple[tuple[float, float], ...]        # aligned world GU
    smooth_gu_polyline: tuple[tuple[float, float], ...]  # aligned world GU
    raw_pixel_chain: tuple[tuple[int, int], ...]         # identical to source
    smooth_pixel_polyline: tuple[tuple[float, float], ...]
    raw_length_gu: float
    length_gu: float
    provenance: Mapping[str, Any]
    smoothing: Mapping[str, Any]

    @property
    def aligned_raw_gu_chain(self) -> tuple[tuple[float, float], ...]:
        return self.raw_gu_chain

    @property
    def aligned_smooth_gu_polyline(self) -> tuple[tuple[float, float], ...]:
        return self.smooth_gu_polyline

    @property
    def is_repaired(self) -> bool:
        """True when this edge carries accepted repair bridge pixels."""
        return bool(self.bridge_ids)

    def corridor_width(self) -> dict[str, Any]:
        """Estimated width plus provenance/source-vs-repair status."""
        return {
            "edge_id": self.id,
            "estimated_width_gu": self.estimated_width_gu,
            "width_gu_p10": self.width_gu_p10,
            "width_gu_p90": self.width_gu_p90,
            "provenance_method": self.provenance.get("method"),
            "source_status": self.source_status,
            "bridge_ids": list(self.bridge_ids),
            "source_vs_repair": "repair_bridge" if self.is_repaired else "source_derived",
        }


@dataclass(frozen=True)
class NearestResult:
    """Nearest centerline sample to a world-GU query point."""

    point: tuple[float, float]          # nearest point on the centerline (GU)
    tangent: tuple[float, float]        # unit tangent at ``point`` (GU/GU)
    distance_gu: float                  # Euclidean distance (GU)
    edge_id: str
    segment_index: int
    along_gu: float                     # arc length along the edge to ``point``

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_gu": list(self.point),
            "tangent": list(self.tangent),
            "distance_gu": round(self.distance_gu, 6),
            "edge_id": self.edge_id,
            "segment_index": self.segment_index,
            "along_gu": round(self.along_gu, 6),
        }


def _nearest_on_segment(
    p: Sequence[float], a: Sequence[float], b: Sequence[float]
) -> tuple[tuple[float, float], float, float]:
    """Nearest point, parameter t, and distance from p to segment ab."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return (float(a[0]), float(a[1])), 0.0, math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq))
    point = (a[0] + t * dx, a[1] + t * dy)
    distance = math.hypot(p[0] - point[0], p[1] - point[1])
    return point, t, distance


def _segment_length(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


#: Fraction of an aligned source road's VTEX-derived ``estimated_width_gu``
#: that is the practical walkable path.  The blended road TEXTURE band in
#: game looks ~2-3x wider than the actually-clear path (user-measured
#: 2026-08-12), so plan checks and the planning canvas use the shrunk width
#: for source corridors only; authored streets/alleys keep their declared
#: widths.  Buildings may sit on road texture; they must never block the
#: practical path.
SOURCE_ROAD_PRACTICAL_PATH_FRACTION = 0.4


class AlignedNetwork:
    """The loaded aligned road network (one canonical consumer object).

    Every loaded network exposes its alignment version and the source hashes
    it was derived from; callers can therefore prove which product an edge or
    node came from without re-reading files.
    """

    def __init__(
        self,
        *,
        product_dir: Path,
        product_sha256: str,
        schema_version: int,
        alignment: Mapping[str, Any],
        manifest: Mapping[str, Any],
        nodes: Sequence[AlignedNode],
        edges: Sequence[AlignedEdge],
        source_hash_verified: bool,
    ) -> None:
        self.product_dir = product_dir
        self.product_sha256 = product_sha256
        self.schema_version = schema_version
        self.alignment = dict(alignment)
        self.manifest = dict(manifest)
        self.source_hash_verified = source_hash_verified
        self.nodes: dict[str, AlignedNode] = {node.id: node for node in nodes}
        self.edges: dict[str, AlignedEdge] = {edge.id: edge for edge in edges}
        self._node_ids = frozenset(self.nodes)
        self._edge_ids = frozenset(self.edges)

    # -- identity ----------------------------------------------------------

    @property
    def alignment_version(self) -> str:
        return str(self.alignment.get("alignment_version", ""))

    @property
    def dx_gu(self) -> int:
        return int(self.alignment.get("dx_gu", 0))

    @property
    def dy_gu(self) -> int:
        return int(self.alignment.get("dy_gu", 0))

    @property
    def source_hashes(self) -> dict[str, str]:
        """Source-side hashes this product was derived from (see manifest)."""
        return {
            "source_canonical_sha256": str(
                self.alignment.get("source_canonical_sha256", "")
            ),
            "source_audit_sha256": str(self.alignment.get("source_audit_sha256", "")),
            "source_effective_alpha_sha256": str(
                self.alignment.get("source_effective_alpha_sha256", "")
            ),
            "tamriel_esm_sha256": str(self.alignment.get("tamriel_esm_sha256", "")),
        }

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def topology(self) -> dict[str, int]:
        return {"node_count": self.node_count, "edge_count": self.edge_count}

    # -- lookup ------------------------------------------------------------

    def node(self, node_id: str) -> AlignedNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise AlignedRoadsError(f"unknown aligned node id {node_id!r}") from exc

    def edge(self, edge_id: str) -> AlignedEdge:
        try:
            return self.edges[edge_id]
        except KeyError as exc:
            raise AlignedRoadsError(f"unknown aligned edge id {edge_id!r}") from exc

    def node_ids(self) -> Iterable[str]:
        return sorted(self._node_ids)

    def edge_ids(self) -> Iterable[str]:
        return sorted(self._edge_ids)

    # -- rectangle queries (world GU, min-inclusive / max-exclusive) -------

    def nodes_in_rect(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> list[AlignedNode]:
        """Nodes whose aligned position lies inside the world-GU rectangle.

        Rectangle semantics are deterministic: ``[x_min, x_max) x
        [y_min, y_max)`` (minimum inclusive, maximum exclusive), identical to
        :meth:`edges_in_rect`.
        """
        if x_min > x_max or y_min > y_max:
            raise ValueError("rectangle minimum must not exceed maximum")
        return [
            node
            for node in self.nodes.values()
            if x_min <= node.position_gu[0] < x_max
            and y_min <= node.position_gu[1] < y_max
        ]

    def edges_in_rect(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> list[AlignedEdge]:
        """Edges whose aligned smooth polyline intersects the world-GU rect.

        Intersection means **true segment/rectangle intersection**: an edge
        is returned when any polyline segment has a point inside the
        half-open rectangle ``[x_min, x_max) x [y_min, y_max)`` (minimum
        inclusive, maximum exclusive), including a segment that *crosses*
        the rectangle with no sampled vertex inside.  The test is the
        deterministic Liang-Barsky clip of each segment against the closed
        rectangle followed by a half-open membership check of the clipped
        portion (see :func:`segment_intersects_rect`): a segment lying
        entirely on the exclusive boundary ``x == x_max`` or ``y == y_max``
        does not intersect, while a segment touching the inclusive
        ``(x_min, y_min)`` corner does.
        """
        if x_min > x_max or y_min > y_max:
            raise ValueError("rectangle minimum must not exceed maximum")
        return [
            edge
            for edge in self.edges.values()
            if any(
                segment_intersects_rect(a, b, x_min, y_min, x_max, y_max)
                for a, b in zip(edge.smooth_gu_polyline, edge.smooth_gu_polyline[1:])
            )
        ]

    # -- site-local frame --------------------------------------------------

    def to_site_local(
        self, world_xy: Sequence[float], origin_gu: Sequence[float]
    ) -> tuple[float, float]:
        """Convert one aligned world-GU point to the declared site-local frame.

        ``plan = world - survey_origin`` (plan-frame GU, +x east, +y north),
        the same convention T1.1 uses for plan geometry.
        """
        if len(world_xy) != 2 or len(origin_gu) != 2:
            raise ValueError("world point and origin must be [x, y] pairs")
        return (
            float(world_xy[0]) - float(origin_gu[0]),
            float(world_xy[1]) - float(origin_gu[1]),
        )

    def edge_site_chain(
        self, edge_id: str, origin_gu: Sequence[float]
    ) -> list[tuple[float, float]]:
        """Smooth aligned chain of one edge in the site-local frame."""
        return [
            self.to_site_local(point, origin_gu)
            for point in self.edge(edge_id).smooth_gu_polyline
        ]

    # -- corridor width / provenance ---------------------------------------

    def corridor_width(self, edge_id: str) -> dict[str, Any]:
        """Road corridor width plus provenance and source-vs-repair status."""
        return self.edge(edge_id).corridor_width()

    # -- nearest centerline -------------------------------------------------

    def nearest_centerline(
        self, world_xy: Sequence[float], *, chain: str = "smooth"
    ) -> NearestResult:
        """Nearest centerline point, tangent, distance, and edge id.

        ``chain`` selects ``"smooth"`` (default) or ``"raw"`` world-GU
        chains.  Tangents are unit vectors along the winning segment.
        """
        if len(world_xy) != 2:
            raise ValueError("query point must be [x, y]")
        if chain not in ("smooth", "raw"):
            raise ValueError("chain must be 'smooth' or 'raw'")
        query = (float(world_xy[0]), float(world_xy[1]))
        best: NearestResult | None = None
        for edge in self.edges.values():
            polyline = (
                edge.smooth_gu_polyline if chain == "smooth" else edge.raw_gu_chain
            )
            along = 0.0
            for index in range(len(polyline) - 1):
                a = polyline[index]
                b = polyline[index + 1]
                point, t, distance = _nearest_on_segment(query, a, b)
                if best is None or distance < best.distance_gu - 1e-9:
                    tangent = (
                        (b[0] - a[0]) / _segment_length(a, b),
                        (b[1] - a[1]) / _segment_length(a, b),
                    )
                    best = NearestResult(
                        point=point,
                        tangent=tangent,
                        distance_gu=distance,
                        edge_id=edge.id,
                        segment_index=index,
                        along_gu=along + t * _segment_length(a, b),
                    )
                along += _segment_length(a, b)
        if best is None:
            raise AlignedRoadsError("network has no edges; cannot measure nearest centerline")
        return best

    # -- corridor polygons ---------------------------------------------------

    def corridor_polygons(
        self, edge_id: str, *, margin_gu: float = 0.0, chain: str = "smooth",
        width_scale: float = 1.0,
    ) -> list[list[tuple[float, float]]]:
        """Per-segment corridor quads of one edge for plan collision checks.

        Each segment becomes a closed ring (5 vertices) offset by
        ``half_width = estimated_width_gu * width_scale / 2 + margin_gu`` on
        both sides of the segment.  Pass
        ``width_scale=SOURCE_ROAD_PRACTICAL_PATH_FRACTION`` to shrink the
        band to the practical walkable path (the VTEX-blended band
        overstates the clear path ~2.5x, user-measured 2026-08-12).  Returns one ring per
        segment; callers may union them for containment checks.  Pure
        geometry -- no numpy.
        """
        if margin_gu < 0.0:
            raise ValueError("margin must be non-negative")
        edge = self.edge(edge_id)
        polyline = (
            edge.smooth_gu_polyline if chain == "smooth" else edge.raw_gu_chain
        )
        half = edge.estimated_width_gu * float(width_scale) / 2.0 + float(margin_gu)
        rings: list[list[tuple[float, float]]] = []
        for index in range(len(polyline) - 1):
            a = polyline[index]
            b = polyline[index + 1]
            length = _segment_length(a, b)
            if length == 0.0:
                continue
            nx = -(b[1] - a[1]) / length
            ny = (b[0] - a[0]) / length
            rings.append(
                [
                    (a[0] + nx * half, a[1] + ny * half),
                    (b[0] + nx * half, b[1] + ny * half),
                    (b[0] - nx * half, b[1] - ny * half),
                    (a[0] - nx * half, a[1] - ny * half),
                    (a[0] + nx * half, a[1] + ny * half),
                ]
            )
        return rings


# ---------------------------------------------------------------------------
# Fail-closed loader
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise AlignedRoadsError(f"cannot load {label} from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AlignedRoadsError(f"{label} {path} is not a JSON object")
    return data


def _verify_coordinate_invariants(
    nodes: Sequence[dict], edges: Sequence[dict]
) -> None:
    """Pixel round-trip gate at the corrected registration.

    Every aligned world coordinate must map back to the source pixel that
    produced it: ``px = x/512 + 4055.5``, ``py = 959.5 - y/512``.  Any other
    translation (wrong sign, wrong magnitude, Y offset) breaks this
    invariant and fails closed.
    """
    for node in nodes:
        position = node.get("position_gu")
        pixel = node.get("position_px")
        if not isinstance(position, list) or len(position) != 2:
            raise AlignedRoadsError(f"node {node.get('id')} has no position_gu pair")
        if not isinstance(pixel, list) or len(pixel) != 2:
            raise AlignedRoadsError(f"node {node.get('id')} has no position_px pair")
        px, py = px_from_aligned_gu(position[0], position[1])
        if abs(px - pixel[0]) > 0.51 or abs(py - pixel[1]) > 0.51:
            raise AlignedRoadsError(
                f"node {node.get('id')} coordinate invariant failed: "
                f"position_gu {position} does not round-trip to position_px {pixel}"
            )
    for edge in edges:
        raw_gu = edge.get("raw_gu_chain")
        raw_px = edge.get("raw_pixel_chain")
        smooth_gu = edge.get("smooth_gu_polyline")
        smooth_px = edge.get("smooth_pixel_polyline")
        if not (isinstance(raw_gu, list) and isinstance(raw_px, list)):
            raise AlignedRoadsError(f"edge {edge.get('id')} has no raw chains")
        if len(raw_gu) != len(raw_px):
            raise AlignedRoadsError(
                f"edge {edge.get('id')} raw chain length {len(raw_gu)} "
                f"!= pixel chain length {len(raw_px)}"
            )
        for index, (point, pixel) in enumerate(zip(raw_gu, raw_px)):
            px, py = px_from_aligned_gu(point[0], point[1])
            if abs(px - pixel[0]) > 0.51 or abs(py - pixel[1]) > 0.51:
                raise AlignedRoadsError(
                    f"edge {edge.get('id')} raw chain point {index} invariant failed"
                )
        if not (isinstance(smooth_gu, list) and isinstance(smooth_px, list)):
            raise AlignedRoadsError(f"edge {edge.get('id')} has no smooth chains")
        if len(smooth_gu) != len(smooth_px):
            raise AlignedRoadsError(
                f"edge {edge.get('id')} smooth chain length {len(smooth_gu)} "
                f"!= pixel polyline length {len(smooth_px)}"
            )
        for index, (point, pixel) in enumerate(zip(smooth_gu, smooth_px)):
            px, py = px_from_aligned_gu(point[0], point[1])
            if abs(px - pixel[0]) > ROUND_TRIP_TOLERANCE_GU * 512 or \
                    abs(py - pixel[1]) > ROUND_TRIP_TOLERANCE_GU * 512:
                raise AlignedRoadsError(
                    f"edge {edge.get('id')} smooth chain point {index} invariant failed"
                )


def _require_pinned_counts(nodes: Sequence[dict], edges: Sequence[dict]) -> None:
    if len(nodes) != SOURCE_NODE_COUNT:
        raise AlignedRoadsError(
            f"aligned product node count {len(nodes)} != pinned {SOURCE_NODE_COUNT}"
        )
    if len(edges) != SOURCE_EDGE_COUNT:
        raise AlignedRoadsError(
            f"aligned product edge count {len(edges)} != pinned {SOURCE_EDGE_COUNT}"
        )


def network_from_product(
    product: Mapping[str, Any],
    *,
    product_dir: Path,
    product_sha256: str,
    manifest: Mapping[str, Any],
) -> AlignedNetwork:
    """Build and gate an :class:`AlignedNetwork` from a parsed product dict.

    Used by :func:`load_aligned_network` and by the build CLI to verify the
    freshly written product with exactly the same gates a consumer sees:
    alignment marker/translation, pinned source/audit/alpha/ESM hashes,
    manifest consistency, topology counts, and per-coordinate pixel
    round-trip invariants.
    """
    alignment = product.get("alignment")
    if not isinstance(alignment, dict):
        raise AlignedRoadsError(
            "product has no alignment section; source-space v1 files are "
            "refused as consumer input"
        )
    if alignment.get("alignment_version") != ALIGNMENT_VERSION:
        raise AlignedRoadsError(
            f"unsupported alignment version {alignment.get('alignment_version')!r}"
        )
    if int(alignment.get("dx_gu")) != ALIGNMENT_DX_GU or \
            int(alignment.get("dy_gu")) != ALIGNMENT_DY_GU:
        raise AlignedRoadsError(
            f"product translation ({alignment.get('dx_gu')}, {alignment.get('dy_gu')}) "
            f"!= pinned (+{ALIGNMENT_DX_GU}, +{ALIGNMENT_DY_GU})"
        )
    if alignment.get("source_canonical_sha256") != SOURCE_CANONICAL_SHA256:
        raise AlignedRoadsError(
            "product source hash differs from the pinned accepted source bundle"
        )
    if alignment.get("source_audit_sha256") != SOURCE_AUDIT_SHA256:
        raise AlignedRoadsError(
            "product source audit hash differs from the pinned accepted source audit"
        )
    if alignment.get("source_effective_alpha_sha256") != SOURCE_EFFECTIVE_ALPHA_SHA256:
        raise AlignedRoadsError(
            "product source effective-alpha hash differs from the pinned accepted value"
        )
    if manifest.get("source_canonical_sha256") != SOURCE_CANONICAL_SHA256:
        raise AlignedRoadsError(
            "manifest source hash differs from the pinned accepted source bundle"
        )
    for manifest_key, pinned, label in (
        ("source_audit_sha256", SOURCE_AUDIT_SHA256, "source audit"),
        ("source_effective_alpha_sha256", SOURCE_EFFECTIVE_ALPHA_SHA256,
         "source effective-alpha"),
    ):
        declared = manifest.get(manifest_key)
        if declared is not None and declared != pinned:
            raise AlignedRoadsError(
                f"manifest {label} hash {declared} differs from the pinned "
                f"accepted value {pinned}"
            )
    if manifest.get("tamriel_esm_sha256") != TAMRIEL_ESM_SHA256:
        raise AlignedRoadsError("manifest tamriel.esm hash differs from the pinned hash")

    nodes_raw = product.get("nodes")
    edges_raw = product.get("edges")
    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        raise AlignedRoadsError("product nodes/edges are not lists")
    _require_pinned_counts(nodes_raw, edges_raw)

    manifest_node_count = int(manifest.get("node_count", -1))
    manifest_edge_count = int(manifest.get("edge_count", -1))
    if len(nodes_raw) != manifest_node_count or len(edges_raw) != manifest_edge_count:
        raise AlignedRoadsError(
            f"product counts ({len(nodes_raw)}, {len(edges_raw)}) differ from "
            f"manifest counts ({manifest_node_count}, {manifest_edge_count})"
        )

    _verify_coordinate_invariants(nodes_raw, edges_raw)

    nodes: list[AlignedNode] = []
    for record in nodes_raw:
        nodes.append(
            AlignedNode(
                id=str(record["id"]),
                component_id=str(record["component_id"]),
                degree=int(record["degree"]),
                kind=str(record["kind"]),
                position_gu=(float(record["position_gu"][0]), float(record["position_gu"][1])),
                position_px=(int(record["position_px"][0]), int(record["position_px"][1])),
                skeleton_pixels=tuple(
                    (int(p[0]), int(p[1])) for p in record["skeleton_pixels"]
                ),
                synthetic_loop_anchor=bool(record.get("synthetic_loop_anchor", False)),
            )
        )
    edges: list[AlignedEdge] = []
    for record in edges_raw:
        edges.append(
            AlignedEdge(
                id=str(record["id"]),
                from_node=str(record["from"]),
                to_node=str(record["to"]),
                component_id=str(record["component_id"]),
                source_status=str(record["source_status"]),
                bridge_ids=tuple(str(b) for b in record["bridge_ids"]),
                estimated_width_gu=float(record["estimated_width_gu"]),
                width_gu_p10=(
                    float(record["width_gu_p10"])
                    if record.get("width_gu_p10") is not None
                    else None
                ),
                width_gu_p90=(
                    float(record["width_gu_p90"])
                    if record.get("width_gu_p90") is not None
                    else None
                ),
                raw_gu_chain=tuple(
                    (float(p[0]), float(p[1])) for p in record["raw_gu_chain"]
                ),
                smooth_gu_polyline=tuple(
                    (float(p[0]), float(p[1])) for p in record["smooth_gu_polyline"]
                ),
                raw_pixel_chain=tuple(
                    (int(p[0]), int(p[1])) for p in record["raw_pixel_chain"]
                ),
                smooth_pixel_polyline=tuple(
                    (float(p[0]), float(p[1])) for p in record["smooth_pixel_polyline"]
                ),
                raw_length_gu=float(record["raw_length_gu"]),
                length_gu=float(record["length_gu"]),
                provenance=dict(record.get("provenance", {})),
                smoothing=dict(record.get("smoothing", {})),
            )
        )
    return AlignedNetwork(
        product_dir=Path(product_dir),
        product_sha256=product_sha256,
        schema_version=int(product.get("schema_version", 0)),
        alignment=alignment,
        manifest=manifest,
        nodes=nodes,
        edges=edges,
        source_hash_verified=False,
    )


def load_aligned_network(
    path: str | Path | None = None, *, verify_source_hash: bool = True
) -> AlignedNetwork:
    """Load and gate the aligned canonical product.

    Gates (fail closed on any violation):

    1. the consumer path must be the aligned product (source-space refused);
    2. ``alignment_manifest.json`` exists and its ``product_canonical_sha256``
       equals the actual product file hash;
    3. the product declares ``alignment_v1`` with exactly ``dx=+4096``,
       ``dy=+0`` and the pinned source/ESM/audit/alpha hashes;
    4. topology counts match the manifest and the pinned 3847/4142;
    5. every node/edge coordinate round-trips to its source pixel at the
       corrected registration;
    6. with ``verify_source_hash=True`` (default) the recorded source bundle
       must **exist** and its file hash must match the pinned source hash;
       absence or drift fails closed (``source_hash_verified=True`` on
       success).

    Returns :class:`AlignedNetwork` whose nodes/edges expose the aligned
    world-GU geometry plus the alignment version and source hashes.
    """
    product_dir = resolve_product_dir(path)
    canonical_path, manifest_path = _canonical_paths(product_dir)
    if not canonical_path.is_file():
        raise AlignedRoadsError(f"aligned product file missing: {canonical_path}")
    if not manifest_path.is_file():
        raise AlignedRoadsError(f"alignment manifest missing: {manifest_path}")

    product_sha256 = _sha256_file(canonical_path)
    manifest = _load_json(manifest_path, "alignment manifest")

    expected = manifest.get("product_canonical_sha256")
    if expected != product_sha256:
        raise AlignedRoadsError(
            f"alignment manifest product hash {expected} != actual {product_sha256}"
        )
    if manifest.get("product_canonical_json") != PRODUCT_CANONICAL_NAME:
        raise AlignedRoadsError(
            f"manifest product name {manifest.get('product_canonical_json')!r} "
            f"!= {PRODUCT_CANONICAL_NAME}"
        )

    product = _load_json(canonical_path, "aligned road product")
    network = network_from_product(
        product,
        product_dir=product_dir,
        product_sha256=product_sha256,
        manifest=manifest,
    )

    if verify_source_hash:
        source_dir = Path(str(manifest.get("source_bundle_dir", "")))
        source_canonical = source_dir / "tamriel_road_centerlines_v1.json"
        if not source_canonical.is_file():
            # Fail closed: the recorded source bundle is part of the hash
            # contract; its absence is drift, not a lenient skip.
            raise AlignedRoadsError(
                f"recorded source bundle missing at {source_canonical}; "
                "verify_source_hash=True requires the source bundle to exist "
                "with the pinned hash"
            )
        actual = _sha256_file(source_canonical)
        if actual != SOURCE_CANONICAL_SHA256:
            raise AlignedRoadsError(
                f"recorded source bundle hash drift: {source_canonical} is "
                f"{actual}, expected {SOURCE_CANONICAL_SHA256}"
            )
        network.source_hash_verified = True

    return network


ROOT = Path(__file__).resolve().parents[2]
#: Default aligned product directory (the canonical consumer product).
DEFAULT_ALIGNED_PRODUCT_DIR = (
    ROOT / "output" / "mapdata" / "roads" / "tamriel_aligned_centerlines_v1"
)
#: Default source-space bundle directory (topology/provenance storage only).
DEFAULT_SOURCE_PRODUCT_DIR = (
    ROOT / "output" / "mapdata" / "roads" / "tamriel_source_centerlines_v1"
)

__all__ = [
    "ALIGNMENT_DX_GU",
    "ALIGNMENT_DY_GU",
    "ALIGNMENT_VERSION",
    "DEFAULT_ALIGNED_PRODUCT_DIR",
    "DEFAULT_SOURCE_PRODUCT_DIR",
    "FALKREATH_CANARY_NODE_IDS",
    "FALKREATH_CELL_BOUNDS",
    "MANIFEST_NAME",
    "PRODUCT_CANONICAL_NAME",
    "RAW_VTEX_ROAD",
    "SOURCE_AUDIT_SHA256",
    "SOURCE_CANONICAL_SHA256",
    "SOURCE_EDGE_COUNT",
    "SOURCE_EFFECTIVE_ALPHA_SHA256",
    "SOURCE_NODE_COUNT",
    "TAMRIEL_ESM_SHA256",
    "AlignedEdge",
    "AlignedNetwork",
    "AlignedNode",
    "AlignedRoadsError",
    "NearestResult",
    "aligned_gu_from_px",
    "edge_corridor_report",
    "esm78_tile_count",
    "is_aligned_product_path",
    "is_source_space_path",
    "load_aligned_network",
    "load_esm78_tiles",
    "nearest_road_tile_distance",
    "network_from_product",
    "px_from_aligned_gu",
    "registration_agreement",
    "registration_stats",
    "resolve_product_dir",
    "skeleton_registration_stats",
    "world_to_tile",
]
