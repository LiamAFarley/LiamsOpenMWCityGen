"""Deterministic planning-background terrain renderer for Cityforge.

Pipeline position
------------------
This is the visual-planning background stage between the accepted D-SITE
survey and the vision planner.  It consumes the read-only
``site_survey.json``/``survey_fields.npz`` bundle and exposes a requested
cell/world rectangle, exact source masks, height/slope sampling, and a cheap
Pillow map image.  The renderer is a planning aid only: it never writes LAND,
VTEX, VNML, or any other TES3 record and never converts the smoothed visual
appearance back into authoring data.

Inputs and outputs
------------------
``TerrainBundle.from_paths`` validates the accepted survey frame and dense
field shapes, decodes the exact 512-GU masks/raw VTEX grid, and records source
hashes.  ``PlanningRectangle`` maps an inclusive cell rectangle (or an
absolute world-GU rectangle) into the survey's plan-frame GU.  ``render_map``
creates a compact RGB/RGBA Pillow map with smoothly interpolated material
appearance, height-derived hillshade, exact nearest-sampled water vertices,
and optional slope/contour layers.  ``render_full_site_inset`` uses the same
path for a consistent context inset.

Invariants
----------
* Plan geometry is ``+x east, +y north`` relative to the survey frame origin;
  image Y is north-up (decreasing pixel Y).
* Height, slope, water vertices, and raw VTEX values are sampled only from the
  accepted survey bundle.  The discrete raw VTEX grid is retained exactly in
  ``TerrainBundle.raw_vtex_tiles`` and in the manifest.
* Material colours are a smoothed visual interpolation of the discrete raw
  VTEX classes.  That interpolation is never used as TES3 output semantics.
* Water is based on the exact survey ``water_vertices`` mask; its shoreline is
  rendered as the boundary of that mask, not inferred from a colour gradient.
* All operations are deterministic for identical input bytes, rectangle, and
  render options.  No random generator or Blender process is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CELL_SIZE_GU = 8192.0
FIELD_SPACING_GU = 128.0
TILE_SIZE_GU = 512.0
FIELD_SIDE = 449
TILE_SIDE = 112


class TerrainBundleError(ValueError):
    """Raised when exact site-survey terrain evidence cannot be loaded."""


def sha256_file(path: Path | str) -> str:
    """Hash a file without changing or fully buffering it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_mask(value: str, side: int, dtype: np.dtype = np.uint8) -> np.ndarray:
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:  # noqa: BLE001 - convert to input-contract error
        raise TerrainBundleError(f"invalid base64 mask: {exc}") from exc
    expected = side * side * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise TerrainBundleError(f"mask bytes {len(raw)} != expected {expected}")
    return np.frombuffer(raw, dtype=dtype).reshape((side, side)).copy()


def _finite_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TerrainBundleError(f"{label} must be a pair")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(v) for v in pair):
        raise TerrainBundleError(f"{label} is not finite")
    return pair


@dataclass(frozen=True)
class PlanningRectangle:
    """Requested and render-space bounds in survey-local plan-frame GU."""

    cell_bounds: tuple[int, int, int, int]
    requested_bounds_gu: tuple[float, float, float, float]
    render_bounds_gu: tuple[float, float, float, float]
    context_margin_gu: float
    full_site_inset: bool

    @property
    def width_gu(self) -> float:
        return self.render_bounds_gu[2] - self.render_bounds_gu[0]

    @property
    def height_gu(self) -> float:
        return self.render_bounds_gu[3] - self.render_bounds_gu[1]

    @property
    def requested_width_gu(self) -> float:
        return self.requested_bounds_gu[2] - self.requested_bounds_gu[0]

    @property
    def requested_height_gu(self) -> float:
        return self.requested_bounds_gu[3] - self.requested_bounds_gu[1]

    def contains(self, point: Sequence[float], *, requested: bool = True) -> bool:
        bounds = self.requested_bounds_gu if requested else self.render_bounds_gu
        return bounds[0] <= float(point[0]) <= bounds[2] and \
            bounds[1] <= float(point[1]) <= bounds[3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_bounds_inclusive": list(self.cell_bounds),
            "requested_bounds_plan_gu": list(self.requested_bounds_gu),
            "render_bounds_plan_gu": list(self.render_bounds_gu),
            "context_margin_gu": self.context_margin_gu,
            "full_site_inset": self.full_site_inset,
            "mapping_convention": (
                "plan frame relative to site survey SW origin; x east, y north; "
                "render bounds are min-inclusive/max-inclusive for visual mapping"
            ),
        }

    @classmethod
    def from_request(
        cls,
        survey: Mapping[str, Any],
        *,
        cell_bounds: Sequence[int] | None = None,
        world_bounds_gu: Sequence[Sequence[float]] | Sequence[float] | None = None,
        context_margin_gu: float = 1024.0,
        full_site_inset: bool = True,
    ) -> "PlanningRectangle":
        """Build a validated rectangle from cells or an absolute world-GU box."""

        frame = survey.get("frame")
        target = survey.get("target_cells")
        if not isinstance(frame, Mapping) or not isinstance(target, Mapping):
            raise TerrainBundleError("survey frame/target_cells are missing")
        origin = _finite_pair(frame.get("origin_gu"), "survey origin")
        survey_min_x = int(target["min_x"])
        survey_min_y = int(target["min_y"])
        survey_max_x = int(target["max_x"])
        survey_max_y = int(target["max_y"])
        site_width = (survey_max_x - survey_min_x + 1) * CELL_SIZE_GU
        site_height = (survey_max_y - survey_min_y + 1) * CELL_SIZE_GU
        if cell_bounds is not None and world_bounds_gu is not None:
            raise TerrainBundleError("provide cell_bounds or world_bounds_gu, not both")
        if cell_bounds is not None:
            if len(cell_bounds) != 4:
                raise TerrainBundleError("cell_bounds must be [min_x,max_x,min_y,max_y]")
            cells = tuple(int(value) for value in cell_bounds)
            if cells[0] > cells[1] or cells[2] > cells[3]:
                raise TerrainBundleError("cell rectangle minimum exceeds maximum")
            if not (survey_min_x <= cells[0] <= cells[1] <= survey_max_x and
                    survey_min_y <= cells[2] <= cells[3] <= survey_max_y):
                raise TerrainBundleError(
                    f"requested cells {cells} are outside surveyed cells "
                    f"[{survey_min_x},{survey_max_x},{survey_min_y},{survey_max_y}]")
            requested = (
                (cells[0] - survey_min_x) * CELL_SIZE_GU,
                (cells[2] - survey_min_y) * CELL_SIZE_GU,
                (cells[1] + 1 - survey_min_x) * CELL_SIZE_GU,
                (cells[3] + 1 - survey_min_y) * CELL_SIZE_GU,
            )
        elif world_bounds_gu is not None:
            if (isinstance(world_bounds_gu, Sequence) and len(world_bounds_gu) == 2 and
                    all(isinstance(v, Sequence) and len(v) == 2 for v in world_bounds_gu)):
                low = _finite_pair(world_bounds_gu[0], "world lower bound")
                high = _finite_pair(world_bounds_gu[1], "world upper bound")
            elif isinstance(world_bounds_gu, Sequence) and len(world_bounds_gu) == 4:
                low = (float(world_bounds_gu[0]), float(world_bounds_gu[1]))
                high = (float(world_bounds_gu[2]), float(world_bounds_gu[3]))
            else:
                raise TerrainBundleError(
                    "world_bounds_gu must be [[min_x,min_y],[max_x,max_y]] or four values")
            if low[0] > high[0] or low[1] > high[1]:
                raise TerrainBundleError("world rectangle minimum exceeds maximum")
            requested = (low[0] - origin[0], low[1] - origin[1],
                         high[0] - origin[0], high[1] - origin[1])
            cells = (
                survey_min_x + int(math.floor(requested[0] / CELL_SIZE_GU)),
                survey_min_x + int(math.ceil(requested[2] / CELL_SIZE_GU)) - 1,
                survey_min_y + int(math.floor(requested[1] / CELL_SIZE_GU)),
                survey_min_y + int(math.ceil(requested[3] / CELL_SIZE_GU)) - 1,
            )
            if not (0.0 <= requested[0] < requested[2] <= site_width and
                    0.0 <= requested[1] < requested[3] <= site_height):
                raise TerrainBundleError("world rectangle lies outside survey frame")
        else:
            raise TerrainBundleError("one of cell_bounds/world_bounds_gu is required")

        margin = float(context_margin_gu)
        if not math.isfinite(margin) or margin < 0.0:
            raise TerrainBundleError("context_margin_gu must be finite and non-negative")
        render = (
            max(0.0, requested[0] - margin),
            max(0.0, requested[1] - margin),
            min(site_width, requested[2] + margin),
            min(site_height, requested[3] + margin),
        )
        if render[0] >= render[2] or render[1] >= render[3]:
            raise TerrainBundleError("render rectangle has no area")
        return cls(cells, requested, render, margin, bool(full_site_inset))


def _palette_for_raw(raw: int) -> tuple[int, int, int]:
    """Stable visual material palette; raw values remain in the bundle."""

    return {
        0: (106, 99, 78),    # LAND default / unclassified ground
        1: (170, 137, 95),   # sand / lakebed
        33: (74, 105, 64),   # grass
        78: (154, 112, 68),  # protected road material
        92: (54, 87, 60),    # pine/moss
    }.get(int(raw), (94, 91, 78))


def _bilinear_grid(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample a [row(y), column(x)] grid at float indices."""

    height, width = values.shape[:2]
    x = np.clip(x, 0.0, width - 1.0)
    y = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = x - x0
    fy = y - y0
    v00 = values[y0, x0]
    v10 = values[y0, x1]
    v01 = values[y1, x0]
    v11 = values[y1, x1]
    return (v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) +
            v01 * (1.0 - fx) * fy + v11 * fx * fy)


@dataclass
class TerrainBundle:
    """Exact D-SITE terrain evidence plus cheap visual sampling helpers."""

    survey_path: Path
    fields_path: Path
    survey: dict[str, Any]
    height_gu: np.ndarray
    slope_deg: np.ndarray
    water_distance_gu: np.ndarray
    water_vertices: np.ndarray
    raw_vtex_tiles: np.ndarray
    water_mask: np.ndarray
    buildable_mask: np.ndarray
    road_mask: np.ndarray
    hashes: dict[str, str]

    @classmethod
    def from_paths(cls, survey_path: Path | str, fields_path: Path | str) -> "TerrainBundle":
        """Load and validate the accepted survey/field bundle read-only."""

        survey_file, fields_file = Path(survey_path), Path(fields_path)
        try:
            survey = json.loads(survey_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TerrainBundleError(f"cannot load site survey {survey_file}: {exc}") from exc
        if not isinstance(survey, dict):
            raise TerrainBundleError("site survey must be a JSON object")
        try:
            archive = np.load(fields_file, allow_pickle=False)
            required = ("height_gu", "slope_deg", "water_distance_gu",
                        "water_vertices", "raw_vtex_tiles")
            missing = [key for key in required if key not in archive.files]
            if missing:
                raise TerrainBundleError(f"survey fields missing arrays {missing}")
            arrays = {key: np.asarray(archive[key]) for key in required}
        except (OSError, ValueError) as exc:
            raise TerrainBundleError(f"cannot load survey fields {fields_file}: {exc}") from exc
        finally:
            try:
                archive.close()  # type: ignore[union-attr]
            except (NameError, AttributeError):
                pass
        expected = {
            "height_gu": (FIELD_SIDE, FIELD_SIDE),
            "slope_deg": (FIELD_SIDE, FIELD_SIDE),
            "water_distance_gu": (FIELD_SIDE, FIELD_SIDE),
            "water_vertices": (FIELD_SIDE, FIELD_SIDE),
            "raw_vtex_tiles": (TILE_SIDE, TILE_SIDE),
        }
        for key, shape in expected.items():
            if arrays[key].shape != shape:
                raise TerrainBundleError(f"{key} shape {arrays[key].shape} != {shape}")
            if not np.isfinite(arrays[key]).all():
                raise TerrainBundleError(f"{key} contains non-finite values")
        frame = survey.get("frame")
        if not isinstance(frame, Mapping) or frame.get("field_spacing_gu") != FIELD_SPACING_GU:
            raise TerrainBundleError("survey field spacing is not the accepted 128 GU")
        tile_grids = survey.get("tile_grids")
        if not isinstance(tile_grids, Mapping) or tile_grids.get("side") != TILE_SIDE:
            raise TerrainBundleError("survey tile grid is not the accepted 112x112 grid")
        water_mask = _decode_mask(str(tile_grids["water_mask"]), TILE_SIDE)
        buildable_mask = _decode_mask(str(tile_grids["buildable_mask"]), TILE_SIDE)
        road_mask = _decode_mask(str(tile_grids["road_mask"]), TILE_SIDE)
        if int(np.count_nonzero(road_mask)) != int(survey["stats"]["road_tiles_78"]):
            raise TerrainBundleError("survey road mask/count provenance mismatch")
        return cls(
            survey_path=survey_file,
            fields_path=fields_file,
            survey=survey,
            height_gu=np.ascontiguousarray(arrays["height_gu"], dtype=np.float64),
            slope_deg=np.ascontiguousarray(arrays["slope_deg"], dtype=np.float64),
            water_distance_gu=np.ascontiguousarray(arrays["water_distance_gu"], dtype=np.float64),
            water_vertices=np.ascontiguousarray(arrays["water_vertices"], dtype=np.uint8),
            raw_vtex_tiles=np.ascontiguousarray(arrays["raw_vtex_tiles"], dtype=np.uint16),
            water_mask=water_mask,
            buildable_mask=buildable_mask,
            road_mask=road_mask,
            hashes={"site_survey": sha256_file(survey_file),
                    "survey_fields": sha256_file(fields_file)},
        )

    @property
    def origin_gu(self) -> tuple[float, float]:
        return _finite_pair(self.survey["frame"]["origin_gu"], "survey origin")

    @property
    def target_cell_bounds(self) -> tuple[int, int, int, int]:
        target = self.survey["target_cells"]
        return (int(target["min_x"]), int(target["max_x"]),
                int(target["min_y"]), int(target["max_y"]))

    @property
    def site_span_gu(self) -> tuple[float, float]:
        return (float(self.survey["frame"]["site_span_gu"][0]),
                float(self.survey["frame"]["site_span_gu"][1]))

    def rectangle(self, *, cell_bounds: Sequence[int] | None = None,
                  world_bounds_gu: Sequence[Sequence[float]] | Sequence[float] | None = None,
                  context_margin_gu: float = 1024.0,
                  full_site_inset: bool = True) -> PlanningRectangle:
        return PlanningRectangle.from_request(
            self.survey, cell_bounds=cell_bounds, world_bounds_gu=world_bounds_gu,
            context_margin_gu=context_margin_gu, full_site_inset=full_site_inset)

    def _field_xy(self, x: float, y: float) -> tuple[float, float]:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise TerrainBundleError("terrain sample is not finite")
        if x < 0.0 or y < 0.0 or x > self.site_span_gu[0] or y > self.site_span_gu[1]:
            raise TerrainBundleError(f"terrain sample ({x}, {y}) lies outside survey")
        return x / FIELD_SPACING_GU, y / FIELD_SPACING_GU

    def sample_height(self, x: float, y: float) -> float:
        fx, fy = self._field_xy(float(x), float(y))
        return float(_bilinear_grid(self.height_gu, np.asarray(fx), np.asarray(fy)))

    def sample_slope(self, x: float, y: float) -> float:
        fx, fy = self._field_xy(float(x), float(y))
        return float(_bilinear_grid(self.slope_deg, np.asarray(fx), np.asarray(fy)))

    def water_at(self, x: float, y: float) -> bool:
        fx, fy = self._field_xy(float(x), float(y))
        ix = min(FIELD_SIDE - 1, max(0, int(round(fx))))
        iy = min(FIELD_SIDE - 1, max(0, int(round(fy))))
        return bool(self.water_vertices[iy, ix])

    def tile_at(self, x: float, y: float) -> tuple[int, int, int]:
        tx = min(TILE_SIDE - 1, max(0, int(math.floor(float(x) / TILE_SIZE_GU))))
        ty = min(TILE_SIDE - 1, max(0, int(math.floor(float(y) / TILE_SIZE_GU))))
        return tx, ty, int(self.raw_vtex_tiles[ty, tx])

    def tile_buildable(self, x: float, y: float) -> bool:
        tx, ty, _ = self.tile_at(x, y)
        return bool(self.buildable_mask[ty, tx])

    def terrain_metrics(self, polygon: Sequence[Sequence[float]]) -> dict[str, Any]:
        """Measure target height/slope/relief under a transformed footprint."""

        points = [(float(p[0]), float(p[1])) for p in polygon]
        if not points:
            raise TerrainBundleError("terrain metrics require a non-empty polygon")
        min_x, max_x = min(p[0] for p in points), max(p[0] for p in points)
        min_y, max_y = min(p[1] for p in points), max(p[1] for p in points)
        count_x = max(2, min(24, int(math.ceil((max_x - min_x) / FIELD_SPACING_GU)) + 1))
        count_y = max(2, min(24, int(math.ceil((max_y - min_y) / FIELD_SPACING_GU)) + 1))
        heights: list[float] = []
        slopes: list[float] = []
        for y in np.linspace(min_y, max_y, count_y):
            for x in np.linspace(min_x, max_x, count_x):
                try:
                    heights.append(self.sample_height(float(x), float(y)))
                    slopes.append(self.sample_slope(float(x), float(y)))
                except TerrainBundleError:
                    continue
        return {
            "height_min_gu": min(heights) if heights else None,
            "height_max_gu": max(heights) if heights else None,
            "relief_gu": (max(heights) - min(heights)) if heights else None,
            "slope_min_deg": min(slopes) if slopes else None,
            "slope_max_deg": max(slopes) if slopes else None,
            "slope_mean_deg": (sum(slopes) / len(slopes)) if slopes else None,
            "sample_count": len(heights),
            "water_vertices": int(sum(self.water_at(x, y) for x, y in points)),
        }

    def manifest(self, rectangle: PlanningRectangle) -> dict[str, Any]:
        """Machine-readable background provenance/audit payload."""

        return {
            "schema_version": 1,
            "kind": "cityforge_visual_planning_background",
            "coordinate_frame": "site_survey_plan_gu",
            "survey_id": self.survey.get("survey_id"),
            "survey_path": str(self.survey_path),
            "fields_path": str(self.fields_path),
            "input_hashes": dict(sorted(self.hashes.items())),
            "rectangle": rectangle.to_dict(),
            "arrays": {
                "height_gu": {"shape": list(self.height_gu.shape), "units": "game_units", "spacing_gu": 128},
                "slope_deg": {"shape": list(self.slope_deg.shape), "units": "degrees", "spacing_gu": 128},
                "water_vertices": {"shape": list(self.water_vertices.shape), "semantics": "exact survey vertex mask"},
                "raw_vtex_tiles": {"shape": list(self.raw_vtex_tiles.shape), "semantics": "exact discrete OpenMW-normalized raw VTEX"},
            },
            "render_semantics": {
                "material_interpolation": "visual-only bilinear interpolation of raw-VTEX class colours",
                "hillshade": "height-derived finite-difference normal from exact 128-GU field",
                "water": "nearest exact water_vertices mask with explicit shoreline boundary",
                "tes3_semantics_changed": False,
            },
        }

    def _map_arrays(self, rectangle: PlanningRectangle, size: tuple[int, int]) -> dict[str, np.ndarray]:
        width, height = size
        x0, y0, x1, y1 = rectangle.render_bounds_gu
        xs = np.linspace(x0, x1, width, endpoint=False, dtype=np.float64) + (x1 - x0) / (2.0 * width)
        ys = np.linspace(y1, y0, height, endpoint=False, dtype=np.float64) - (y1 - y0) / (2.0 * height)
        xx, yy = np.meshgrid(xs, ys)
        fx = xx / FIELD_SPACING_GU
        fy = yy / FIELD_SPACING_GU
        heights = _bilinear_grid(self.height_gu, fx, fy)
        slopes = _bilinear_grid(self.slope_deg, fx, fy)
        water_x = np.clip(np.rint(fx).astype(np.int64), 0, FIELD_SIDE - 1)
        water_y = np.clip(np.rint(fy).astype(np.int64), 0, FIELD_SIDE - 1)
        water = self.water_vertices[water_y, water_x].astype(bool)
        raw_x = np.clip(np.floor(xx / TILE_SIZE_GU).astype(np.int64), 0, TILE_SIDE - 1)
        raw_y = np.clip(np.floor(yy / TILE_SIZE_GU).astype(np.int64), 0, TILE_SIDE - 1)
        raw_nearest = self.raw_vtex_tiles[raw_y, raw_x]
        # Bilinear material interpolation.  Sampling the palette at tile
        # centres deliberately feathers class changes instead of drawing 512-GU
        # spreadsheet squares; raw_nearest remains available for exact audits.
        palette = np.asarray([_palette_for_raw(v) for v in range(0, 256)], dtype=np.float64)
        material_grid = palette[np.asarray(self.raw_vtex_tiles, dtype=np.int64)]
        grid_x = xx / TILE_SIZE_GU - 0.5
        grid_y = yy / TILE_SIZE_GU - 0.5
        material = np.stack([
            _bilinear_grid(material_grid[:, :, channel], grid_x, grid_y)
            for channel in range(3)
        ], axis=-1)
        # Exact height-derived hillshade.  Derivatives are measured in GU/GU
        # from the authoritative field and then bilinearly sampled for pixels.
        dzdy, dzdx = np.gradient(self.height_gu, FIELD_SPACING_GU, FIELD_SPACING_GU)
        light = np.asarray((-0.55, -0.35, 0.76), dtype=np.float64)
        light /= np.linalg.norm(light)
        nx, ny, nz = -dzdx, -dzdy, np.ones_like(self.height_gu)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        shade_grid = np.clip((nx * light[0] + ny * light[1] + nz * light[2]) / norm, -1.0, 1.0)
        shade = _bilinear_grid(shade_grid, fx, fy)
        return {"xx": xx, "yy": yy, "height": heights, "slope": slopes,
                "water": water, "raw": raw_nearest, "material": material,
                "shade": shade}

    @staticmethod
    def _apply_shade(material: np.ndarray, shade: np.ndarray) -> np.ndarray:
        factor = np.clip(0.70 + 0.48 * shade, 0.30, 1.18)
        return np.clip(material * factor[..., None], 0, 255).astype(np.uint8)

    def render_map(
        self,
        rectangle: PlanningRectangle,
        *,
        size: tuple[int, int] = (1080, 1080),
        hillshade: bool = True,
        slope_advisory: bool = False,
        contours: bool = False,
        contour_levels: Iterable[float] | None = None,
    ) -> Image.Image:
        """Render only the terrain body; symbols are composited by the symbol layer."""

        arrays = self._map_arrays(rectangle, size)
        material = arrays["material"]
        rgb = self._apply_shade(material, arrays["shade"]) if hillshade else np.clip(material, 0, 255).astype(np.uint8)
        image = Image.fromarray(rgb, mode="RGB").convert("RGBA")
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        water = arrays["water"]
        water_colour = np.zeros((*water.shape, 4), dtype=np.uint8)
        water_colour[..., 0] = 56
        water_colour[..., 1] = 145
        water_colour[..., 2] = 205
        water_colour[..., 3] = np.where(water, 205, 0).astype(np.uint8)
        image.alpha_composite(Image.fromarray(water_colour, mode="RGBA"))
        # Shoreline = exact mask boundary, drawn after water fill but before
        # planning symbols.  A one-pixel boundary is sufficient at this scale.
        edge = water & ((np.roll(water, 1, 0) != water) |
                        (np.roll(water, -1, 0) != water) |
                        (np.roll(water, 1, 1) != water) |
                        (np.roll(water, -1, 1) != water))
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = False
        edge_image = np.zeros((*edge.shape, 4), dtype=np.uint8)
        edge_image[..., :3] = (210, 230, 196)
        edge_image[..., 3] = np.where(edge, 225, 0).astype(np.uint8)
        image.alpha_composite(Image.fromarray(edge_image, mode="RGBA"))
        if slope_advisory:
            mask = arrays["slope"] > 15.0
            slope_layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
            slope_layer[..., :3] = (220, 103, 38)
            slope_layer[..., 3] = np.where(mask & ~water, 52, 0).astype(np.uint8)
            image.alpha_composite(Image.fromarray(slope_layer, mode="RGBA"))
        if contours:
            self.draw_contours(draw, rectangle, size,
                               levels=list(contour_levels or self._default_contours(rectangle)))
        image.alpha_composite(overlay)
        return image

    def _default_contours(self, rectangle: PlanningRectangle) -> list[float]:
        x0, y0, x1, y1 = rectangle.render_bounds_gu
        ix0, ix1 = max(0, int(math.floor(x0 / FIELD_SPACING_GU))), min(FIELD_SIDE - 1, int(math.ceil(x1 / FIELD_SPACING_GU)))
        iy0, iy1 = max(0, int(math.floor(y0 / FIELD_SPACING_GU))), min(FIELD_SIDE - 1, int(math.ceil(y1 / FIELD_SPACING_GU)))
        low = float(np.min(self.height_gu[iy0:iy1 + 1, ix0:ix1 + 1]))
        high = float(np.max(self.height_gu[iy0:iy1 + 1, ix0:ix1 + 1]))
        start = math.floor(low / 500.0) * 500.0
        return [level for level in np.arange(start, high + 500.0, 500.0)]

    def world_to_pixel(self, rectangle: PlanningRectangle, size: tuple[int, int],
                       point: Sequence[float]) -> tuple[float, float]:
        x0, y0, x1, y1 = rectangle.render_bounds_gu
        return ((float(point[0]) - x0) / (x1 - x0) * size[0],
                (y1 - float(point[1])) / (y1 - y0) * size[1])

    def pixel_to_plan(self, rectangle: PlanningRectangle, size: tuple[int, int],
                      point: Sequence[float]) -> tuple[float, float]:
        x0, y0, x1, y1 = rectangle.render_bounds_gu
        return (x0 + float(point[0]) / size[0] * (x1 - x0),
                y1 - float(point[1]) / size[1] * (y1 - y0))

    def draw_contours(self, draw: ImageDraw.ImageDraw, rectangle: PlanningRectangle,
                      size: tuple[int, int], levels: Sequence[float]) -> None:
        """Draw labelled marching-squares contours from exact field heights."""

        x0, y0, x1, y1 = rectangle.render_bounds_gu
        ix0 = max(0, int(math.floor(x0 / FIELD_SPACING_GU)))
        ix1 = min(FIELD_SIDE - 2, int(math.ceil(x1 / FIELD_SPACING_GU)))
        iy0 = max(0, int(math.floor(y0 / FIELD_SPACING_GU)))
        iy1 = min(FIELD_SIDE - 2, int(math.ceil(y1 / FIELD_SPACING_GU)))
        # Explicitly list the ambiguous marching-squares cases so the hot loop
        # stays simple and deterministic.
        table: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = {
            1: [((0, .5), (.5, 0))], 2: [((.5, 0), (1, .5))],
            3: [((0, .5), (1, .5))], 4: [((1, .5), (.5, 1))],
            5: [((0, .5), (.5, 0)), ((1, .5), (.5, 1))],
            6: [((.5, 0), (.5, 1))], 7: [((0, .5), (.5, 1))],
            8: [((.5, 1), (0, .5))], 9: [((.5, 0), (.5, 1))],
            10: [((.5, 0), (1, .5)), ((0, .5), (.5, 1))],
            11: [((1, .5), (.5, 1))], 12: [((1, .5), (0, .5))],
            13: [((.5, 0), (1, .5))], 14: [((.5, 0), (0, .5))],
        }
        font = ImageFont.load_default()
        for level in levels:
            label_position: tuple[float, float] | None = None
            for iy in range(iy0, iy1 + 1):
                for ix in range(ix0, ix1 + 1):
                    h = [float(self.height_gu[iy, ix]), float(self.height_gu[iy, ix + 1]),
                         float(self.height_gu[iy + 1, ix + 1]), float(self.height_gu[iy + 1, ix])]
                    bits = sum((1 << index) for index, value in enumerate(h) if value >= level)
                    for start, end in table.get(bits, []):
                        points: list[tuple[float, float]] = []
                        for edge_point in (start, end):
                            ex, ey = edge_point
                            if ex == 0.5 and ey == 0:
                                t = (level - h[0]) / (h[1] - h[0]) if h[1] != h[0] else .5
                                gx, gy = ix + t, iy
                            elif ex == 1 and ey == 0.5:
                                t = (level - h[1]) / (h[2] - h[1]) if h[2] != h[1] else .5
                                gx, gy = ix + 1, iy + t
                            elif ex == 0.5 and ey == 1:
                                t = (level - h[3]) / (h[2] - h[3]) if h[2] != h[3] else .5
                                gx, gy = ix + t, iy + 1
                            else:
                                t = (level - h[0]) / (h[3] - h[0]) if h[3] != h[0] else .5
                                gx, gy = ix, iy + t
                            world = (gx * FIELD_SPACING_GU, gy * FIELD_SPACING_GU)
                            points.append(self.world_to_pixel(rectangle, size, world))
                        draw.line(points, fill=(242, 225, 143, 180), width=2)
                        if label_position is None:
                            label_position = points[0]
            if label_position is not None:
                text = f"{level:.0f} GU"
                draw.text((label_position[0] + 3, label_position[1] - 10), text,
                          font=font, fill=(255, 242, 170, 230),
                          stroke_width=2, stroke_fill=(34, 44, 38, 190))

    def render_full_site_inset(self, size: tuple[int, int] = (230, 230)) -> Image.Image:
        """Render a consistent full-survey context inset."""

        rect = self.rectangle(cell_bounds=self.target_cell_bounds,
                              context_margin_gu=0.0, full_site_inset=False)
        return self.render_map(rect, size=size, hillshade=True,
                               slope_advisory=False, contours=False)


__all__ = [
    "CELL_SIZE_GU", "FIELD_SIDE", "FIELD_SPACING_GU", "PlanningRectangle",
    "TerrainBundle", "TerrainBundleError", "TILE_SIDE", "TILE_SIZE_GU",
    "sha256_file",
]
