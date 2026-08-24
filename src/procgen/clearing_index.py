"""clearing_index.py — shared exclusion queries over a settlement clearing (Stage 2).

Purpose
-------
Build a fast, shared query structure over a clearing document (produced by
``procgen.settlement_clearing.build_clearing``) so the scatter generator
(plan Stage 3) and the groundcover generator (plan Stage 4) reject candidates
inside a town with one consistent set of rules:

* ``blocks_point(x, y)``  - a candidate point is blocked if it lies inside a
  building footprint (buffered) or circulation surface polygon, or within a
  road corridor (the stored ``half_width_gu`` from the road centreline).
* ``blocks_aabb(minx, miny, maxx, maxy)`` - an axis-aligned footprint AABB
  (e.g. a rock/cliff bbox) is blocked if it intersects any building/surface
  polygon or any buffered road corridor.
* ``in_city_domain_point(x, y)`` / ``blocks_city_domain_aabb(...)`` - a
  GEOMETRIC test against the broad city-domain perimeter polygon (the area
  that defines the city, walled or not).  These are the authoritative rock-ban
  tests: zero rocks anywhere inside the city.  User ruling 2026-08-18.
* ``in_city_domain(cell_x, cell_y)`` - cell-centre variant, retained only for
  the discrete per-cell flora rule; deliberately under-covers the city and
  must NOT be used for the rock ban.

All public geometry queries accept **world / global TES3 game units** (the
native frame of the scatter and groundcover generators); the index converts to
the clearing's plan-GU frame internally via ``frame_origin_gu``.  This lets any
generated city's clearing be read out fluidly regardless of where the city is
placed.

The clearing geometry is already in absolute plan GU; road buffers are applied
at query time from the stored centreline + ``half_width_gu`` so the clearing
document keeps centreline geometry.

Pipeline position
-----------------
Approved plan: ``.opencode/runs/cityforge-scatter-groundcover-integration/
plan.md``, Stage 2.  Shared by the scatter and groundcover generators and by
the unified CLI (Stage 5).  Pure read-only queries; never writes.

Cell-grid -> plan-GU mapping
---------------------------
The city layout is authored in plan GU (origin at a settlement-specific cell,
e.g. cell (-95, -11) for the accepted Falkreath layout, i.e.
``frame_origin_gu = [-778240.0, -90112.0]``).  The scatter/groundcover
generators operate on TES3 cell grid coordinates, so ``in_city_domain`` maps
``cell -> world GU -> plan GU``:

    world_gu = (cell_x * 8192 + 4096, cell_y * 8192 + 4096)
    plan_gu  = world_gu - frame_origin_gu

When the clearing document carries no ``frame_origin_gu`` the cell grid is
assumed to already be in the plan frame (``frame_origin_gu = (0, 0)``); this
fallback is documented rather than silent.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree

CELL_SIZE_GU = 8192.0
CELL_HALF_GU = CELL_SIZE_GU / 2.0


def _to_ring_list(rings: object) -> list[list[list[float]]]:
    """Validate a ``rings`` payload (list of closed rings) into float lists."""
    if not isinstance(rings, list) or not rings:
        raise ValueError("clearing polygon 'rings' must be a non-empty list")
    out: list[list[list[float]]] = []
    for ring in rings:
        if not isinstance(ring, list) or len(ring) < 3:
            raise ValueError("clearing polygon ring must be a list of [x, y]")
        pts: list[list[float]] = []
        for pt in ring:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                raise ValueError("clearing polygon point must be [x, y]")
            pts.append([float(pt[0]), float(pt[1])])
        out.append(pts)
    return out


def _rings_to_polygon(rings: list[list[list[float]]]) -> Polygon:
    exterior = rings[0]
    holes = rings[1:]
    return Polygon(exterior, holes)


def _extract_frame_origin(doc: Mapping[str, Any]) -> tuple[float, float]:
    origin = doc.get("frame_origin_gu")
    if (
        isinstance(origin, (list, tuple))
        and len(origin) >= 2
        and all(isinstance(v, (int, float)) for v in origin[:2])
    ):
        return float(origin[0]), float(origin[1])
    return 0.0, 0.0


class ClearingIndex:
    """Shared exclusion queries over a settlement clearing document."""

    def __init__(self, doc: Mapping[str, Any]) -> None:
        self._doc = doc
        self._frame_origin_x, self._frame_origin_y = _extract_frame_origin(doc)

        self._polygons: list[Polygon] = []
        self._road_linestrings: list[LineString] = []
        self._road_half_widths: list[float] = []

        for entry in doc.get("building_exclusions", []):
            rings = _to_ring_list(entry.get("rings"))
            self._polygons.append(_rings_to_polygon(rings))
        for entry in doc.get("surface_exclusions", []):
            rings = _to_ring_list(entry.get("rings"))
            self._polygons.append(_rings_to_polygon(rings))
        for entry in doc.get("road_exclusions", []):
            points = entry.get("points")
            half = entry.get("half_width_gu")
            if not isinstance(points, list) or len(points) < 2:
                raise ValueError("clearing road 'points' must be a list of [x, y]")
            pts = [(float(p[0]), float(p[1])) for p in points]
            if half is None:
                raise ValueError("clearing road missing 'half_width_gu'")
            self._road_linestrings.append(LineString(pts))
            self._road_half_widths.append(float(half))

        self._city_domain: Polygon | None = None
        domain = doc.get("city_domain")
        if isinstance(domain, list) and domain:
            self._city_domain = Polygon(
                [(float(p[0]), float(p[1])) for p in domain]
            )

        # STRtree over exclusion polygons only (roads are checked by
        # distance so their half-width can be applied per road).
        if self._polygons:
            self._polygon_tree = STRtree(self._polygons)
        else:
            self._polygon_tree = None

    # -- public queries ----------------------------------------------------
    # All public geometry queries accept GLOBAL / world TES3 game units (the
    # native frame of the scatter and groundcover generators: ref positions
    # are ``cell * 8192 + local``).  They are converted to the clearing's plan
    # GU frame internally via ``frame_origin_gu``, so the exclusion index reads
    # out any city's clearing fluidly regardless of where it is placed.

    def _plan(self, x: float, y: float) -> tuple[float, float]:
        return (float(x) - self._frame_origin_x, float(y) - self._frame_origin_y)

    def blocks_point(self, x: float, y: float) -> bool:
        """True if the world-GU point (x, y) is inside any exclusion
        (building footprint, surface, or road corridor)."""
        px, py = self._plan(x, y)
        point = Point(px, py)
        if self._polygon_tree is not None:
            for i in self._polygon_tree.query(point):
                if self._polygons[int(i)].contains(point):
                    return True
        for road, half in zip(self._road_linestrings, self._road_half_widths):
            if road.distance(point) <= half:
                return True
        return False

    def blocks_aabb(
        self, minx: float, miny: float, maxx: float, maxy: float
    ) -> bool:
        """True if the world-GU box [minx..maxx] x [miny..maxy] intersects any
        building/surface polygon or any buffered road corridor."""
        if not (minx <= maxx and miny <= maxy):
            return False
        pminx, pminy = self._plan(minx, miny)
        pmaxx, pmaxy = self._plan(maxx, maxy)
        bbox = box(pminx, pminy, pmaxx, pmaxy)
        if self._polygon_tree is not None:
            for i in self._polygon_tree.query(bbox):
                if self._polygons[int(i)].intersects(bbox):
                    return True
        for road, half in zip(self._road_linestrings, self._road_half_widths):
            if road.distance(bbox) <= half:
                return True
        return False

    # -- city-domain (broad city perimeter) ---------------------------------
    # The city-domain rock ban is a GEOMETRIC test over the whole city
    # perimeter polygon (the broad area that defines the city), NOT a
    # cell-centre rasterization.  User ruling 2026-08-18: zero rocks anywhere
    # in the region defined as part of the city, walled or not.

    def in_city_domain_point(self, x: float, y: float) -> bool:
        """True if the world-GU point (x, y) lies inside the city-domain
        perimeter polygon.  This is the authoritative in-city test for the
        rock ban."""
        if self._city_domain is None:
            return False
        px, py = self._plan(x, y)
        return self._city_domain.contains(Point(px, py))

    def blocks_city_domain_aabb(
        self, minx: float, miny: float, maxx: float, maxy: float
    ) -> bool:
        """True if the world-GU box [minx..maxx] x [miny..maxy] intersects the
        city-domain perimeter polygon.  Used to reject a rock/cliff whose
        footprint AABB enters the city."""
        if self._city_domain is None:
            return False
        if not (minx <= maxx and miny <= maxy):
            return False
        pminx, pminy = self._plan(minx, miny)
        pmaxx, pmaxy = self._plan(maxx, maxy)
        bbox = box(pminx, pminy, pmaxx, pmaxy)
        return self._city_domain.intersects(bbox)

    def in_city_domain(self, cell_x: int, cell_y: int) -> bool:
        """True if the centre of TES3 cell (cell_x, cell_y) is inside the city
        boundary (cell-grid -> world -> plan-GU mapping).

        This cell-centre variant is retained only where a discrete per-cell
        answer is needed (the flora-allowed rule); it deliberately under-covers
        the city, so it must NOT be used for the rock ban (use
        ``in_city_domain_point`` / ``blocks_city_domain_aabb`` instead).
        """
        if self._city_domain is None:
            return False
        world_x = float(cell_x) * CELL_SIZE_GU + CELL_HALF_GU
        world_y = float(cell_y) * CELL_SIZE_GU + CELL_HALF_GU
        plan_x = world_x - self._frame_origin_x
        plan_y = world_y - self._frame_origin_y
        return self._city_domain.contains(Point(plan_x, plan_y))

    # -- introspection -----------------------------------------------------

    @property
    def polygon_count(self) -> int:
        return len(self._polygons)

    @property
    def road_count(self) -> int:
        return len(self._road_linestrings)

    @property
    def frame_origin_gu(self) -> tuple[float, float]:
        return (self._frame_origin_x, self._frame_origin_y)


class MultiClearingIndex:
    """Union of several settlement clearings (corridor-wide runs).

    Each clearing keeps its own ``frame_origin_gu``, so documents cannot be
    merged into one; this wrapper queries each member index in its own plan
    frame and ORs the results.  It intentionally mirrors the ClearingIndex
    public query API so scatter/groundcover code can take either type.
    """

    def __init__(self, docs: Sequence[Mapping[str, Any]]) -> None:
        if not docs:
            raise ValueError("MultiClearingIndex requires at least one clearing document")
        self._indexes = [ClearingIndex(doc) for doc in docs]

    def blocks_point(self, x: float, y: float) -> bool:
        return any(index.blocks_point(x, y) for index in self._indexes)

    def blocks_aabb(self, minx: float, miny: float, maxx: float, maxy: float) -> bool:
        return any(index.blocks_aabb(minx, miny, maxx, maxy) for index in self._indexes)

    def in_city_domain_point(self, x: float, y: float) -> bool:
        return any(index.in_city_domain_point(x, y) for index in self._indexes)

    def blocks_city_domain_aabb(self, minx: float, miny: float, maxx: float, maxy: float) -> bool:
        return any(index.blocks_city_domain_aabb(minx, miny, maxx, maxy) for index in self._indexes)

    def in_city_domain(self, cell_x: int, cell_y: int) -> bool:
        return any(index.in_city_domain(cell_x, cell_y) for index in self._indexes)

    @property
    def polygon_count(self) -> int:
        return sum(index.polygon_count for index in self._indexes)

    @property
    def road_count(self) -> int:
        return sum(index.road_count for index in self._indexes)

    @property
    def frame_origin_gu(self) -> tuple[float, float]:
        return self._indexes[0].frame_origin_gu if self._indexes else (0.0, 0.0)

    @property
    def frame_origins_gu(self) -> list[tuple[float, float]]:
        return [idx.frame_origin_gu for idx in self._indexes]


def build_clearing_index(doc_or_docs: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> ClearingIndex | MultiClearingIndex:
    """Build the right index for one clearing document or a list of them."""

    if isinstance(doc_or_docs, Mapping):
        return ClearingIndex(doc_or_docs)
    docs = list(doc_or_docs)
    if len(docs) == 1:
        return ClearingIndex(docs[0])
    return MultiClearingIndex(docs)


__all__ = [
    "ClearingIndex",
    "MultiClearingIndex",
    "build_clearing_index",
]
