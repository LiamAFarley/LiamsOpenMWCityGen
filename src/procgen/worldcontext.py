"""Lazy, typed access to the validated geographic input products."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from .coords import (
    CELL_POINTS,
    MAX_X,
    MAX_Y,
    MIN_X,
    MIN_Y,
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    Cell,
    cell_to_pixel,
    require_in_grid,
)

JsonObject: TypeAlias = dict[str, Any]
JsonRecords: TypeAlias = list[JsonObject]

_MAPDATA_FILES = {
    "transform": "transform.json",
    "regions_document": "regions.json",
    "region_labels_document": "labels.json",
    "settlements_document": "settlements.json",
    "settlement_footprints_document": "settlement_footprints.json",
    "masterlist_coverage": "masterlist_coverage_report.json",
    "roads_graph": "roads_graph_clean.json",
    "borders_graph": "borders_graph_clean.json",
}


def _default_workspace() -> Path:
    # worldcontext.py -> procgen -> src -> workspace
    return Path(__file__).resolve().parents[2]


@dataclass
class WorldContext:
    """Geographic interface shared by later generation stages.

    JSON products are loaded on first property access and cached for the life
    of the context.  The 460 MB composite RAW is *not* loaded eagerly: the
    first height query creates a read-only ``numpy.memmap``.  ``height_at``
    returns the stored signed height in THU (1 THU = 8 GU) at a heightmap
    sample.  A two-integer input addresses the south-west sample of a cell;
    a two-real input addresses a fractional game-grid position; a four-value
    input ``(cell_x, cell_y, offset_x, offset_y)`` addresses a pixel offset
    within a cell.
    """

    workspace: str | Path | None = None
    mapdata_dir: str | Path | None = None
    composite_raw: str | Path | None = None
    masterlist_path: str | Path | None = None
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _heightmap: np.memmap | None = field(default=None, init=False, repr=False)

    # These are deliberately explicit: terrain_cells.json covers the full
    # rectangular raster, while only these counts represent exterior records.
    EXTERIOR_CELL_COUNT = 32086
    TERRAIN_PRESENT_CELL_COUNT = 32064
    ABOVE_SEA_CELL_COUNT = 26358

    def __post_init__(self) -> None:
        root = Path(self.workspace) if self.workspace is not None else _default_workspace()
        self.workspace = root.resolve()
        self.mapdata_dir = (
            Path(self.mapdata_dir)
            if self.mapdata_dir is not None
            else self.workspace / "output" / "mapdata"
        ).resolve()
        self.composite_raw = (
            Path(self.composite_raw)
            if self.composite_raw is not None
            else Path(
                r"C:\Users\LiamF\AppData\Local\Temp\opencode\tesannwyn-composite\composite_full.raw"
            )
        ).resolve()
        self.masterlist_path = (
            Path(self.masterlist_path)
            if self.masterlist_path is not None
            else Path(
                r"C:\Users\LiamF\AppData\Local\Temp\opencode\city_masterlist.json"
            )
        ).resolve()

    def _load_json(self, cache_key: str, filename: str) -> JsonObject:
        if cache_key not in self._cache:
            path = Path(self.mapdata_dir) / filename
            if not path.is_file():
                raise FileNotFoundError(f"WorldContext input is missing: {path}")
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError(f"WorldContext input is not a JSON object: {path}")
            self._cache[cache_key] = value
        return self._cache[cache_key]

    @staticmethod
    def _records(document: JsonObject, key: str) -> JsonRecords:
        value = document.get(key)
        if not isinstance(value, list):
            raise ValueError(f"WorldContext document field {key!r} is not a list")
        return value

    @property
    def transform(self) -> JsonObject:
        """Map registration and output-pixel coordinate transform."""

        return self._load_json("transform", _MAPDATA_FILES["transform"])

    @property
    def regions(self) -> JsonRecords:
        """The 240 PTR region records, without their document envelope."""

        return self._records(
            self._load_json("regions_document", _MAPDATA_FILES["regions_document"]),
            "regions",
        )

    @property
    def region_labels(self) -> JsonRecords:
        """Map label records from ``labels.json``."""

        return self._records(
            self._load_json(
                "region_labels_document", _MAPDATA_FILES["region_labels_document"]
            ),
            "labels",
        )

    @property
    def settlements(self) -> JsonRecords:
        """Settlement marker records, retaining ``name_method`` metadata."""

        return self._records(
            self._load_json(
                "settlements_document", _MAPDATA_FILES["settlements_document"]
            ),
            "settlements",
        )

    @property
    def settlement_footprints(self) -> JsonRecords:
        """Settlement footprint records."""

        return self._records(
            self._load_json(
                "settlement_footprints_document",
                _MAPDATA_FILES["settlement_footprints_document"],
            ),
            "footprints",
        )

    @property
    def masterlist_coverage(self) -> JsonObject:
        """Masterlist-to-marker/footprint coverage diagnostics."""

        return self._load_json(
            "masterlist_coverage", _MAPDATA_FILES["masterlist_coverage"]
        )

    @property
    def roads_graph(self) -> JsonObject:
        """Clean road graph; every edge must retain ``road_class``."""

        graph = self._load_json("roads_graph", _MAPDATA_FILES["roads_graph"])
        edges = graph.get("edges")
        if not isinstance(edges, list) or any(
            not isinstance(edge, dict) or "road_class" not in edge for edge in edges
        ):
            raise ValueError("roads_graph_clean.json has an edge without road_class")
        return graph

    @property
    def borders_graph(self) -> JsonObject:
        """Clean border graph containing the three retained networks."""

        return self._load_json("borders_graph", _MAPDATA_FILES["borders_graph"])

    @property
    def terrain_cells(self) -> JsonObject:
        """Per-cell terrain summaries in GU, including explicit field names."""

        if "terrain_cells" not in self._cache:
            path = Path(self.workspace) / "output" / "terrain_cells.json"
            if not path.is_file():
                raise FileNotFoundError(f"WorldContext input is missing: {path}")
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError(f"WorldContext input is not a JSON object: {path}")
            self._cache["terrain_cells"] = value
        return self._cache["terrain_cells"]

    @property
    def terrain_population(self) -> dict[str, int]:
        """The three non-interchangeable terrain populations used by stages."""

        return {
            "exterior_records": self.EXTERIOR_CELL_COUNT,
            "terrain_present": self.TERRAIN_PRESENT_CELL_COUNT,
            "above_sea": self.ABOVE_SEA_CELL_COUNT,
        }

    def land_cells(self, above_sea_only: bool = True) -> list[Cell]:
        """Return deterministic terrain cells, optionally excluding sea/seabed.

        ``above_sea_only=True`` means ``e_max_gu > 0`` and yields 26,358 cells
        for the current input.  With it disabled, the ``land`` flag from the
        terrain product includes terrain-present seabed cells (32,064 cells).
        Neither mode means “all cells in the rectangular heightmap”.
        """

        document = self.terrain_cells
        fields = document.get("fields")
        rows = document.get("cells")
        if not isinstance(fields, list) or not isinstance(rows, list):
            raise ValueError("terrain_cells.json must contain fields and cells lists")
        try:
            x_index = fields.index("x")
            y_index = fields.index("y")
            max_index = fields.index("e_max_gu")
            land_index = fields.index("land")
        except ValueError as exc:
            raise ValueError("terrain_cells.json lacks a required field") from exc

        result: list[Cell] = []
        for row in rows:
            if not isinstance(row, list):
                continue
            is_land = bool(row[land_index])
            if is_land and (not above_sea_only or float(row[max_index]) > 0):
                result.append((int(row[x_index]), int(row[y_index])))
        return result

    @property
    def raw_available(self) -> bool:
        """Whether the configured composite RAW exists and is a regular file."""

        return Path(self.composite_raw).is_file()

    @property
    def heightmap(self) -> np.memmap:
        """Open the composite heightmap read-only as a lazy NumPy memmap."""

        if self._heightmap is None:
            path = Path(self.composite_raw)
            if not path.is_file():
                raise FileNotFoundError(
                    "Composite heightmap is missing; expected read-only input at "
                    f"{path}"
                )
            expected_bytes = PIXEL_WIDTH * PIXEL_HEIGHT * np.dtype("<i2").itemsize
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"composite RAW has {actual_bytes} bytes; expected "
                    f"{expected_bytes} for {PIXEL_WIDTH}x{PIXEL_HEIGHT} s16 samples"
                )
            self._heightmap = np.memmap(
                path,
                dtype="<i2",
                mode="r",
                shape=(PIXEL_HEIGHT, PIXEL_WIDTH),
            )
        return self._heightmap

    @staticmethod
    def _sample_pixel(value: Real, origin: int, limit: int) -> int:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"heightmap coordinate must be finite: {value!r}")
        pixel = math.floor(numeric)
        if pixel < origin or pixel >= limit:
            raise ValueError(f"heightmap coordinate {value!r} is outside [{origin},{limit})")
        return pixel

    def height_at(self, cell_or_subcell: tuple[Real, ...]) -> int:
        """Read one signed height sample in THU from the memory-mapped RAW.

        Accepted forms are ``(cell_x, cell_y)`` for the cell's south-west
        sample, ``(game_x, game_y)`` for a fractional game-grid position, and
        ``(cell_x, cell_y, pixel_x, pixel_y)`` for a pixel offset in a cell.
        The four-value form is useful when a caller needs an unambiguous
        subcell sample without converting coordinates itself.
        """

        if len(cell_or_subcell) == 2:
            x, y = cell_or_subcell
            if isinstance(x, Integral) and isinstance(y, Integral):
                px_float, py_float = cell_to_pixel((int(x), int(y)))
            else:
                game_x, game_y = float(x), float(y)
                if not (
                    MIN_X <= game_x < MAX_X + 1
                    and MIN_Y <= game_y < MAX_Y + 1
                ):
                    raise ValueError(
                        f"fractional game coordinate {(game_x, game_y)!r} is outside grid"
                    )
                px_float = (game_x - MIN_X) * CELL_POINTS
                py_float = (game_y - MIN_Y) * CELL_POINTS
            px = self._sample_pixel(px_float, 0, PIXEL_WIDTH)
            py = self._sample_pixel(py_float, 0, PIXEL_HEIGHT)
        elif len(cell_or_subcell) == 4:
            cell_x, cell_y, offset_x, offset_y = cell_or_subcell
            if not isinstance(cell_x, Integral) or not isinstance(cell_y, Integral):
                raise ValueError("four-value height query requires integer cell coordinates")
            x, y = require_in_grid((int(cell_x), int(cell_y)))
            if not isinstance(offset_x, Integral) or not isinstance(offset_y, Integral):
                raise ValueError("four-value height query requires integer pixel offsets")
            if not (0 <= int(offset_x) < CELL_POINTS and 0 <= int(offset_y) < CELL_POINTS):
                raise ValueError("subcell pixel offsets must be in [0, 64)")
            px = (x - MIN_X) * CELL_POINTS + int(offset_x)
            py = (y - MIN_Y) * CELL_POINTS + int(offset_y)
        else:
            raise ValueError(
                "height query must be (x,y) or (cell_x,cell_y,offset_x,offset_y)"
            )
        return int(self.heightmap[py, px])
