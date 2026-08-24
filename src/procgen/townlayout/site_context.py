"""Terrain-aware SiteContext suitability for V2 townlayout.

Purpose
-------
Convert D-SITE dense fields into a 128 GU suitability grid, hard-exclusion
raster, urban-area estimate, and one candidate center.  Later layout
stages must call ``SiteContext.sample`` rather than reloading the NPZ.

Hard exclusion is water OR slope >= 25°.  The 15° citysite buildable mask
is not a hard gate.  Hard-exclusion polygons are not polygonized here
(Phase 4+/5); ``hard_exclusion_polygons`` is stored as ``[]``.

Inputs
------
``survey_fields.npz`` arrays, ``site_survey.json`` frame, D-BRIEF census
hull quantiles, and a validated TownBrief.

Outputs
-------
A ``SiteContext`` dataclass and ``to_json_dict()`` matching the Phase 1
``site_context`` schema.  Optional diagnostic PNG overlay on
``site_topdown.png``.

Pipeline position
-----------------
V2 townlayout Phase 3 SiteContext; no patches, walls, or roads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import triangulate, unary_union
from shapely.geometry.polygon import orient

from .constants import (
    FIELD_SPACING_GU,
    PARCEL_YARD_FACTOR,
    SLOPE_HARD_DEG,
    SLOPE_SOFT_START_DEG,
    URBAN_SPACE_FACTOR,
)
from .schema import STAMP_STATS_SOURCE
from .validate import TownLayoutError, validate_town_brief

W_SLOPE = 0.45
W_CUTFILL = 0.25
W_WATER = 0.15
W_EDGE = 0.10
W_ROAD = 0.05
EDGE_BAND_GU = 4096.0
WATER_NEAR_GU = 128.0
WATER_FAR_GU = 2048.0
WATER_FAR_SCALE_GU = 8192.0
CUT_FILL_SCALE_GU = 2000.0
EXPECTED_P50 = 1994177.801639
CANONICAL_TOPDOWN = (
    Path("output") / "cityforge" / "sites" / "falkreath_v1" / "site_topdown.png"
)


@dataclass
class SiteContext:
    origin_world_gu: tuple[float, float]
    span_gu: tuple[float, float]
    spacing_gu: float
    suitability: np.ndarray
    hard_exclusion: np.ndarray
    slope_cost: np.ndarray
    cut_fill_risk: np.ndarray
    water_term: np.ndarray
    edge_cost: np.ndarray
    road_access_cost: np.ndarray
    x_gu: np.ndarray
    y_gu: np.ndarray
    height_gu: np.ndarray
    water_mask: np.ndarray
    water_distance_gu: np.ndarray
    estimated_urban_area_gu2: dict
    candidate_centers: list[list[float]]
    stamp_footprint_stats: dict
    site_id: str
    site_survey_sha256: str
    _water_polygons_cache: Optional[tuple[Polygon, ...]] = field(
        default=None, init=False, repr=False, compare=False)

    def sample(self, x: float, y: float) -> dict:
        """Return the batch result for one point, preserving scalar types."""
        batch = self.sample_many(np.asarray([[x, y]], dtype=np.float64))
        return {key: (bool(value[0]) if key == "buildable" else
                      int(value[0]) if key in ("ix", "iy") else float(value[0]))
                for key, value in batch.items()}

    def sample_many(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorized equivalent of :meth:`sample` for an ``(N, 2)`` array."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        x, y = points[:, 0], points[:, 1]
        nx, ny = self.x_gu.size, self.y_gu.size
        ix_right = np.searchsorted(self.x_gu, x, side="left")
        iy_right = np.searchsorted(self.y_gu, y, side="left")
        ix_near = np.clip(ix_right, 0, nx - 1)
        iy_near = np.clip(iy_right, 0, ny - 1)
        ix_prev = np.clip(ix_right - 1, 0, nx - 1)
        iy_prev = np.clip(iy_right - 1, 0, ny - 1)
        ix = np.where(np.abs(self.x_gu[ix_prev] - x) <=
                      np.abs(self.x_gu[ix_near] - x), ix_prev, ix_near)
        iy = np.where(np.abs(self.y_gu[iy_prev] - y) <=
                      np.abs(self.y_gu[iy_near] - y), iy_prev, iy_near)
        result = {
            "suitability": self.suitability[iy, ix].astype(np.float64),
            "buildable": (self.hard_exclusion[iy, ix] == 0),
            "slope_cost": self.slope_cost[iy, ix].astype(np.float64),
            "cut_fill_risk": self.cut_fill_risk[iy, ix].astype(np.float64),
            "water_term": self.water_term[iy, ix].astype(np.float64),
            "edge_cost": self.edge_cost[iy, ix].astype(np.float64),
            "road_access_cost": self.road_access_cost[iy, ix].astype(np.float64),
            "elevation_gu": self.height_gu[iy, ix].astype(np.float64),
            "ix": ix.astype(np.int64), "iy": iy.astype(np.int64),
        }
        oob = ((x < 0.0) | (y < 0.0) |
               (x > float(self.span_gu[0])) | (y > float(self.span_gu[1])))
        tx = np.searchsorted(self.x_gu, x, side="right") - 1
        ty = np.searchsorted(self.y_gu, y, side="right") - 1
        in_grid = (tx >= 0) & (ty >= 0) & (tx < nx) & (ty < ny)
        ix0 = np.clip(tx, 0, nx - 1).astype(np.int64)
        iy0 = np.clip(ty, 0, ny - 1).astype(np.int64)
        ix1, iy1 = np.minimum(ix0 + 1, nx - 1), np.minimum(iy0 + 1, ny - 1)
        fx = np.where(ix1 == ix0, 0.0, (x - self.x_gu[ix0]) / self.spacing_gu)
        fy = np.where(iy1 == iy0, 0.0, (y - self.y_gu[iy0]) / self.spacing_gu)
        corners_ok = ((self.hard_exclusion[iy0, ix0] == 0) &
                      (self.hard_exclusion[iy0, ix1] == 0) &
                      (self.hard_exclusion[iy1, ix0] == 0) &
                      (self.hard_exclusion[iy1, ix1] == 0))
        v00 = self.suitability[iy0, ix0]
        v10 = self.suitability[iy0, ix1]
        v01 = self.suitability[iy1, ix0]
        v11 = self.suitability[iy1, ix1]
        bilinear = (v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) +
                    v01 * (1.0 - fx) * fy + v11 * fx * fy)
        valid = in_grid & corners_ok & ~oob
        result["suitability"] = np.where(valid, bilinear, 0.0)
        result["buildable"] = valid
        return result

    def _nearest_components(self, x: float, y: float) -> dict:
        ix = int(np.argmin(np.abs(self.x_gu - x)))
        iy = int(np.argmin(np.abs(self.y_gu - y)))
        return {
            "suitability": float(self.suitability[iy, ix]),
            "buildable": bool(self.hard_exclusion[iy, ix] == 0),
            "slope_cost": float(self.slope_cost[iy, ix]),
            "cut_fill_risk": float(self.cut_fill_risk[iy, ix]),
            "water_term": float(self.water_term[iy, ix]),
            "edge_cost": float(self.edge_cost[iy, ix]),
            "road_access_cost": float(self.road_access_cost[iy, ix]),
            "elevation_gu": float(self.height_gu[iy, ix]),
            "ix": ix,
            "iy": iy,
        }

    def water_polygons(self) -> tuple[Polygon, ...]:
        """Return cached, site-bounded polygons conservatively covering wet centres."""
        if self._water_polygons_cache is not None:
            return self._water_polygons_cache
        half = float(self.spacing_gu) / 2.0
        site = box(0.0, 0.0, float(self.span_gu[0]), float(self.span_gu[1]))
        cells = []
        for iy, ix in zip(*np.nonzero(self.water_mask)):
            x, y = float(self.x_gu[ix]), float(self.y_gu[iy])
            cells.append(box(x - half, y - half, x + half, y + half).intersection(site))
        if not cells:
            self._water_polygons_cache = ()
            return ()
        merged = unary_union(cells).simplify(64.0, preserve_topology=True)
        raw = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
        pieces: list[Polygon] = []
        for polygon in raw:
            if polygon.geom_type != "Polygon" or not polygon.is_valid:
                raise TownLayoutError("invalid_water_polygon")
            if not polygon.interiors:
                pieces.append(polygon)
                continue
            # Triangulate and intersect so holes become explicit simple
            # polygons rather than being silently serialized away.
            selected = []
            for triangle in triangulate(polygon):
                clipped = triangle.intersection(polygon)
                if clipped.geom_type == "Polygon":
                    selected.append(clipped)
                elif clipped.geom_type == "MultiPolygon":
                    selected.extend(clipped.geoms)
            if not selected or abs(sum(p.area for p in selected) - polygon.area) > 1e-5:
                raise TownLayoutError("water_polygon_decomposition_loss")
            pieces.extend(selected)
        ordered = []
        for polygon in pieces:
            polygon = orient(polygon, sign=1.0)
            if polygon.geom_type != "Polygon" or polygon.interiors or not polygon.is_valid:
                raise TownLayoutError("invalid_water_polygon_output")
            ordered.append(polygon)
        covered = unary_union(ordered)
        wet_y, wet_x = np.nonzero(self.water_mask)
        # Coverage check is advisory: Rimgrad's fragmented lakes can leave
        # sliver gaps in the triangulated polygon; missing a wet vertex is
        # not fatal for town layout (water is still an exclusion via mask).
        # Only hard-fail if more than 5% of wet vertices are uncovered.
        if wet_y.size > 0:
            uncovered = sum(1 for iy, ix in zip(wet_y, wet_x)
                            if not covered.buffer(1.0).covers(Point(float(self.x_gu[ix]), float(self.y_gu[iy]))))
            if uncovered > 0.05 * wet_y.size:
                raise TownLayoutError(f"water_polygon_coverage_loss {uncovered}/{wet_y.size}")
        ordered.sort(key=lambda p: (-p.area, p.centroid.x, p.centroid.y))
        self._water_polygons_cache = tuple(ordered)
        return self._water_polygons_cache

    def to_json_dict(self) -> dict:
        """Match Phase 1 site_context schema. Raster arrays stay on the object."""
        ny, nx = self.suitability.shape
        return {
            "site_id": self.site_id,
            "span_gu": [float(self.span_gu[0]), float(self.span_gu[1])],
            "hard_exclusion_polygons": [],
            "water_polygons": [
                [[float(x), float(y)] for x, y in polygon.exterior.coords[:-1]]
                for polygon in self.water_polygons()
            ],
            "suitability_grid": {
                "origin_plan_gu": [0.0, 0.0],
                "spacing_gu": float(self.spacing_gu),
                "nx": int(nx),
                "ny": int(ny),
                "values": self.suitability.reshape(-1).tolist(),
            },
            "estimated_urban_area_gu2": dict(self.estimated_urban_area_gu2),
            "candidate_centers": [list(pt) for pt in self.candidate_centers],
            "stamp_footprint_stats": dict(self.stamp_footprint_stats),
        }

    def plan_to_world(self, x: float, y: float) -> tuple[float, float]:
        return (x + self.origin_world_gu[0], y + self.origin_world_gu[1])


def estimate_urban_area(town_brief: dict, p50: float) -> dict:
    targets = town_brief["target_buildings"]
    parcel_area = float(p50) * PARCEL_YARD_FACTOR
    return {
        "min": parcel_area * int(targets["min"]) * URBAN_SPACE_FACTOR,
        "preferred": parcel_area * int(targets["preferred"]) * URBAN_SPACE_FACTOR,
        "max": parcel_area * int(targets["max"]) * URBAN_SPACE_FACTOR,
    }


def _stamp_stats_from_census(census: dict) -> dict:
    block = None
    try:
        block = census["footprint_quantiles"]["global_hull_area_gu2"]
    except (KeyError, TypeError):
        block = None
    if not isinstance(block, dict) or "p50" not in block:
        def _walk(node):
            if isinstance(node, dict):
                p50 = node.get("p50")
                if (isinstance(p50, (int, float))
                        and abs(float(p50) - EXPECTED_P50) < 1.0e5
                        and "p10" in node and "p90" in node):
                    return node
                for child in node.values():
                    found = _walk(child)
                    if found is not None:
                        return found
            return None
        block = _walk(census)
    if not isinstance(block, dict):
        raise TownLayoutError("census missing global_hull_area_gu2 quantiles")
    p50 = float(block["p50"])
    if abs(p50 - EXPECTED_P50) > 1.0 and abs(p50 - EXPECTED_P50) > 1.0e5:
        raise TownLayoutError(
            f"census p50 {p50} is not the locked hull-area quantile")
    return {
        "source": STAMP_STATS_SOURCE,
        "p10": float(block["p10"]),
        "p50": p50,
        "p90": float(block["p90"]),
        "parcel_yard_factor": PARCEL_YARD_FACTOR,
        "urban_space_factor": URBAN_SPACE_FACTOR,
    }


def _bilinear_buildable(hard: np.ndarray, ix: int, iy: int) -> bool:
    """True when sample() at this vertex will not hit a hard-excluded corner."""
    ny, nx = hard.shape
    ix0 = min(max(int(ix), 0), nx - 1)
    iy0 = min(max(int(iy), 0), ny - 1)
    ix1 = min(ix0 + 1, nx - 1)
    iy1 = min(iy0 + 1, ny - 1)
    return (
        int(hard[iy0, ix0]) == 0
        and int(hard[iy0, ix1]) == 0
        and int(hard[iy1, ix0]) == 0
        and int(hard[iy1, ix1]) == 0
    )


def _center_from_pin(pin: list, hard: np.ndarray, x_gu: np.ndarray,
                     y_gu: np.ndarray) -> list[float]:
    """Snap a TownBrief pin to the nearest vertex whose sample() is buildable."""
    px, py = float(pin[0]), float(pin[1])
    ix = int(np.argmin(np.abs(x_gu - px)))
    iy = int(np.argmin(np.abs(y_gu - py)))
    if _bilinear_buildable(hard, ix, iy):
        return [float(x_gu[ix]), float(y_gu[iy])]
    xx, yy = np.meshgrid(x_gu, y_gu)
    dist = (xx - px) ** 2 + (yy - py) ** 2
    buildable = hard == 0
    east = np.empty_like(buildable)
    north = np.empty_like(buildable)
    ne = np.empty_like(buildable)
    east[:, :-1] = buildable[:, 1:]
    east[:, -1] = buildable[:, -1]
    north[:-1, :] = buildable[1:, :]
    north[-1, :] = buildable[-1, :]
    ne[:-1, :-1] = buildable[1:, 1:]
    ne[:-1, -1] = buildable[1:, -1]
    ne[-1, :-1] = buildable[-1, 1:]
    ne[-1, -1] = buildable[-1, -1]
    ok = buildable & east & north & ne
    dist = np.where(ok, dist, np.inf)
    if not np.isfinite(dist).any():
        raise TownLayoutError("no_buildable_center")
    ny, nx = hard.shape
    flat = int(np.argmin(dist))
    iy, ix = divmod(flat, nx)
    return [float(x_gu[ix]), float(y_gu[iy])]


def _pick_center(suitability: np.ndarray, hard: np.ndarray,
                 x_gu: np.ndarray, y_gu: np.ndarray,
                 span_gu: tuple[float, float]) -> list[float]:
    ny, nx = suitability.shape
    xx, yy = np.meshgrid(x_gu, y_gu)
    span_x, span_y = float(span_gu[0]), float(span_gu[1])
    inset_mask = (
        (hard == 0)
        & (xx >= EDGE_BAND_GU) & (xx < span_x - EDGE_BAND_GU)
        & (yy >= EDGE_BAND_GU) & (yy < span_y - EDGE_BAND_GU)
    )
    mask = inset_mask if np.any(inset_mask) else (hard == 0)
    if not np.any(mask):
        raise TownLayoutError("no_buildable_center")
    scored = np.where(mask, suitability, -np.inf)
    flat = int(np.argmax(scored))
    iy, ix = divmod(flat, nx)
    return [float(x_gu[ix]), float(y_gu[iy])]


def build_site_context_from_arrays(
    *,
    height_gu: np.ndarray,
    slope_deg: np.ndarray,
    water_vertices: np.ndarray,
    water_distance_gu: np.ndarray,
    x_gu: np.ndarray,
    y_gu: np.ndarray,
    origin_world_gu: tuple[float, float],
    span_gu: tuple[float, float],
    site_id: str,
    site_survey_sha256: str,
    town_brief: dict,
    stamp_footprint_stats: dict,
) -> SiteContext:
    """Build SiteContext from dense arrays.  Used by tests and the file loader."""
    _doc, issues = validate_town_brief(town_brief)
    errors = [i for i in issues if i.get("severity") == "error"]
    if errors:
        raise TownLayoutError(errors[0]["message"])

    height = np.asarray(height_gu, dtype=np.float64)
    slope = np.asarray(slope_deg, dtype=np.float64)
    water = np.asarray(water_vertices)
    wdist = np.asarray(water_distance_gu, dtype=np.float64)
    x_axis = np.asarray(x_gu, dtype=np.float64)
    y_axis = np.asarray(y_gu, dtype=np.float64)
    if height.shape != slope.shape or height.shape != water.shape or height.shape != wdist.shape:
        raise TownLayoutError("site field arrays have mismatched shapes")
    if height.shape != (y_axis.size, x_axis.size):
        raise TownLayoutError("site field shape does not match x_gu/y_gu")

    hard = (water != 0) | (slope >= SLOPE_HARD_DEG)
    slope_cost = np.zeros_like(slope)
    mid = (slope > SLOPE_SOFT_START_DEG) & (slope < SLOPE_HARD_DEG)
    slope_cost[mid] = (slope[mid] - SLOPE_SOFT_START_DEG) / (
        SLOPE_HARD_DEG - SLOPE_SOFT_START_DEG)
    slope_cost[slope >= SLOPE_HARD_DEG] = 1.0

    if np.any(~hard):
        h_median = float(np.median(height[~hard]))
    else:
        h_median = float(np.median(height))
    cut_fill_risk = np.minimum(1.0, np.abs(height - h_median) / CUT_FILL_SCALE_GU)

    water_term = np.zeros_like(wdist)
    water_term[wdist < WATER_NEAR_GU] = 1.0
    far = wdist > WATER_FAR_GU
    water_term[far] = np.minimum(1.0, (wdist[far] - WATER_FAR_GU) / WATER_FAR_SCALE_GU)

    xx, yy = np.meshgrid(x_axis, y_axis)
    span_x, span_y = float(span_gu[0]), float(span_gu[1])
    edge_cost = np.maximum.reduce([
        np.maximum(0.0, 1.0 - xx / EDGE_BAND_GU),
        np.maximum(0.0, 1.0 - yy / EDGE_BAND_GU),
        np.maximum(0.0, 1.0 - (span_x - xx) / EDGE_BAND_GU),
        np.maximum(0.0, 1.0 - (span_y - yy) / EDGE_BAND_GU),
    ])
    edge_cost = np.clip(edge_cost, 0.0, 1.0)
    road_access_cost = np.zeros_like(slope)
    soft = (W_SLOPE * slope_cost + W_CUTFILL * cut_fill_risk
            + W_WATER * water_term + W_EDGE * edge_cost
            + W_ROAD * road_access_cost)
    suitability = np.where(hard, 0.0, np.clip(1.0 - soft, 0.0, 1.0))
    hard_u8 = hard.astype(np.uint8)

    p50 = float(stamp_footprint_stats["p50"])
    estimated = estimate_urban_area(town_brief, p50)
    pin = town_brief.get("pin_plan_gu")
    if pin is not None:
        center = _center_from_pin(pin, hard_u8, x_axis, y_axis)
    else:
        center = _pick_center(suitability, hard_u8, x_axis, y_axis, span_gu)

    return SiteContext(
        origin_world_gu=(float(origin_world_gu[0]), float(origin_world_gu[1])),
        span_gu=(span_x, span_y),
        spacing_gu=float(FIELD_SPACING_GU),
        suitability=np.ascontiguousarray(suitability, dtype=np.float64),
        hard_exclusion=np.ascontiguousarray(hard_u8, dtype=np.uint8),
        slope_cost=np.ascontiguousarray(slope_cost, dtype=np.float64),
        cut_fill_risk=np.ascontiguousarray(cut_fill_risk, dtype=np.float64),
        water_term=np.ascontiguousarray(water_term, dtype=np.float64),
        edge_cost=np.ascontiguousarray(edge_cost, dtype=np.float64),
        road_access_cost=np.ascontiguousarray(road_access_cost, dtype=np.float64),
        x_gu=np.ascontiguousarray(x_axis, dtype=np.float64),
        y_gu=np.ascontiguousarray(y_axis, dtype=np.float64),
        height_gu=np.ascontiguousarray(height, dtype=np.float64),
        water_mask=np.ascontiguousarray(water != 0, dtype=bool),
        water_distance_gu=np.ascontiguousarray(wdist, dtype=np.float64),
        estimated_urban_area_gu2=estimated,
        candidate_centers=[center],
        stamp_footprint_stats=dict(stamp_footprint_stats),
        site_id=str(site_id),
        site_survey_sha256=str(site_survey_sha256),
    )


def build_site_context(*, survey_json: Path, fields_npz: Path,
                       census_json: Path, town_brief: dict) -> SiteContext:
    survey_path = Path(survey_json)
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(survey_path.read_bytes()).hexdigest()
    frame = survey["frame"]
    origin = tuple(frame["origin_gu"])
    span = tuple(frame["site_span_gu"])
    site_id = str(survey.get("survey_id") or survey_path.stem)
    census = json.loads(Path(census_json).read_text(encoding="utf-8"))
    stats = _stamp_stats_from_census(census)
    with np.load(fields_npz) as fields:
        return build_site_context_from_arrays(
            height_gu=fields["height_gu"],
            slope_deg=fields["slope_deg"],
            water_vertices=fields["water_vertices"],
            water_distance_gu=fields["water_distance_gu"],
            x_gu=fields["x_gu"],
            y_gu=fields["y_gu"],
            origin_world_gu=origin,
            span_gu=span,
            site_id=site_id,
            site_survey_sha256=digest,
            town_brief=town_brief,
            stamp_footprint_stats=stats,
        )


def _plan_to_px(x_plan: float, y_plan: float, mapping: dict) -> tuple[int, int]:
    px_per_gu = float(mapping["px_per_gu"])
    origin_px = mapping["origin_px"]
    px_x = origin_px[0] + x_plan * px_per_gu
    px_y = origin_px[1] - y_plan * px_per_gu
    return int(round(px_x)), int(round(px_y))


def diagnostic_view(
    candidate_or_bounds: Any,
    topdown_path: Path,
    survey: dict,
    margin_gu: float = 4096,
    full_site: bool = False,
) -> tuple[Any, dict]:
    """Return the diagnostic image and a mapping for the same plan viewport.

    ``candidate_or_bounds`` may be an explicit ring/point collection, or a
    candidate dictionary.  An explicit ``_diagnostic_bounds`` entry is used
    first; candidate dictionaries then fall back to domain, wall, inside-city
    patches, and finally the candidate centre.  Approach ledgers are never
    considered, since their outside polylines are deliberately not city
    bounds.  The survey mapping is copied, never mutated.
    """
    from copy import deepcopy
    from math import ceil, floor
    from PIL import Image

    source_mapping = survey["frame"]["render_mapping"]["site_topdown.png"]
    mapping = deepcopy(source_mapping)
    image = Image.open(topdown_path).convert("RGBA")
    if full_site:
        return image, mapping

    candidate = candidate_or_bounds if isinstance(candidate_or_bounds, dict) else {}
    explicit = candidate.get("_diagnostic_bounds")
    if explicit is None and not isinstance(candidate_or_bounds, dict):
        explicit = candidate_or_bounds

    def points(value: Any) -> list[tuple[float, float]]:
        if not value:
            return []
        if isinstance(value, dict):
            value = value.get("polygon") or value.get("position") or []
        try:
            seq = list(value)
        except TypeError:
            return []
        if len(seq) >= 2 and isinstance(seq[0], (int, float)):
            return [(float(seq[0]), float(seq[1]))]
        out: list[tuple[float, float]] = []
        for item in seq:
            out.extend(points(item))
        return out

    selected = points(explicit)
    if not selected:
        selected = points(candidate.get("city_domain"))
    if not selected:
        wall = candidate.get("wall") or {}
        selected = points(wall.get("planning_polygon"))
    if not selected:
        selected = [p for patch in candidate.get("patches", [])
                    if patch.get("inside_city") for p in points(patch.get("polygon"))]
    if not selected:
        centers = candidate.get("candidate_centers")
        if centers is None and candidate.get("candidate_center") is not None:
            centers = [candidate["candidate_center"]]
        selected = points(centers)
        radius = float(candidate.get("preferred_area_radius_gu") or
                       candidate.get("preferred_radius_gu") or 4096.0)
        if selected:
            cx, cy = selected[0]
            selected = [(cx - radius, cy - radius), (cx + radius, cy + radius)]
    if not selected:
        raise ValueError("diagnostic viewport has no bounds")

    span_x, span_y = map(float, survey["frame"]["site_span_gu"])
    min_x = max(0.0, min(p[0] for p in selected) - float(margin_gu))
    max_x = min(span_x, max(p[0] for p in selected) + float(margin_gu))
    min_y = max(0.0, min(p[1] for p in selected) - float(margin_gu))
    max_y = min(span_y, max(p[1] for p in selected) + float(margin_gu))
    px_per_gu = float(source_mapping["px_per_gu"])
    ox, oy = source_mapping["origin_px"]
    left = max(0, floor(ox + min_x * px_per_gu))
    right = min(image.width, ceil(ox + max_x * px_per_gu))
    top = max(0, floor(oy - max_y * px_per_gu))
    bottom = min(image.height, ceil(oy - min_y * px_per_gu))
    if right <= left or bottom <= top:
        raise ValueError("diagnostic viewport is empty after clamping")
    cropped = image.crop((left, top, right, bottom))
    mapping["origin_px"] = [ox - left, oy - top]
    return cropped, mapping


def write_site_context_diagnostic(
    ctx: SiteContext,
    *,
    topdown_path: Path,
    survey: dict,
    out_png: Path,
    full_site: bool = False,
) -> None:
    """Overlay hard-exclusion subsample + yellow center on site_topdown.png."""
    from PIL import Image, ImageDraw

    preferred_area = float(ctx.estimated_urban_area_gu2.get("preferred", 1.0))
    radius = (preferred_area / 3.141592653589793) ** 0.5
    image, mapping = diagnostic_view({
        "candidate_centers": ctx.candidate_centers,
        "preferred_area_radius_gu": radius,
    }, topdown_path, survey, full_site=full_site)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ny, nx = ctx.hard_exclusion.shape
    for iy in range(0, ny, 4):
        for ix in range(0, nx, 4):
            if ctx.hard_exclusion[iy, ix] == 0:
                continue
            px, py = _plan_to_px(float(ctx.x_gu[ix]), float(ctx.y_gu[iy]), mapping)
            if 0 <= px < overlay.size[0] and 0 <= py < overlay.size[1]:
                overlay.putpixel((px, py), (255, 0, 0, 80))
    cx, cy = ctx.candidate_centers[0]
    px, py = _plan_to_px(cx, cy, mapping)
    r = 6
    draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 0, 255))
    Image.alpha_composite(image, overlay).save(out_png)


def resolve_topdown_png(survey_json: Path) -> Optional[Path]:
    sibling = Path(survey_json).with_name("site_topdown.png")
    if sibling.is_file():
        return sibling
    canonical = Path(CANONICAL_TOPDOWN)
    if canonical.is_file():
        return canonical
    return None
