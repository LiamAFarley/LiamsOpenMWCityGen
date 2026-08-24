"""Resolve region-map polygons to authoritative exterior cell grids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def region_cells(region_map: Path | str, region_id: str, *, pixels_per_cell: float = 64.0) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    """Rasterize one indexed region polygon using the map's cell convention."""

    path = Path(region_map)
    document = json.loads(path.read_text(encoding="utf-8"))
    regions = document.get("regions") if isinstance(document, Mapping) else None
    meta = document.get("meta") if isinstance(document, Mapping) else None
    grid = meta.get("grid") if isinstance(meta, Mapping) else None
    if not isinstance(regions, list) or not isinstance(grid, Mapping):
        raise ValueError(f"region map {path} has no regions/meta.grid")
    matches = [row for row in regions if isinstance(row, Mapping) and row.get("region_id") == region_id]
    if len(matches) != 1:
        raise ValueError(f"region map {path} has {len(matches)} matches for {region_id!r}")
    row = matches[0]
    bbox = row.get("bbox_cells")
    polygon = row.get("polygon_map_px")
    declared = row.get("cell_count")
    if not isinstance(bbox, list) or len(bbox) != 4 or not isinstance(polygon, list) or not isinstance(declared, int):
        raise ValueError(f"region {region_id!r} has malformed polygon metadata")
    scale = float(pixels_per_cell)
    if scale <= 0.0:
        raise ValueError("pixels_per_cell must be positive")
    map_min_x = int(grid["min_x"])
    map_max_y = int(grid["max_y"])

    def inside(point: tuple[float, float]) -> bool:
        x, y = point
        result = False
        previous = len(polygon) - 1
        for index, current in enumerate(polygon):
            prior = polygon[previous]
            if ((current[1] > y) != (prior[1] > y)) and (
                x < (prior[0] - current[0]) * (y - current[1]) / (prior[1] - current[1]) + current[0]
            ):
                result = not result
            previous = index
        return result

    cells = {
        (cell_x, cell_y)
        for cell_x in range(int(bbox[0]), int(bbox[2]) + 1)
        for cell_y in range(int(bbox[1]), int(bbox[3]) + 1)
        if inside(((cell_x - map_min_x + 0.5) * scale, (map_max_y - cell_y + 0.5) * scale))
    }
    if len(cells) != declared:
        raise ValueError(f"region {region_id!r} rasterized to {len(cells)} cells, declared {declared}")
    return cells, {
        "region_id": region_id,
        "region_name": row.get("name"),
        "region_map": str(path.resolve()),
        "authoritative_cell_count": len(cells),
        "bbox_cells": [int(value) for value in bbox],
    }
