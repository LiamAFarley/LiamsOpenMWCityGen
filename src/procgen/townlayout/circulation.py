"""Finalize pedestrian circulation and residual open-space classes for R7."""
from __future__ import annotations

from typing import Any

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .validate import TownLayoutError


def finalize_circulation(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("stage_id") != "r6_rows_access":
        raise TownLayoutError("circulation requires r6_rows_access")
    hull_union = unary_union([Polygon(p["hull"])
                              for p in source.get("placements") or []])
    alleys = []
    for path in source.get("reserved_access_paths") or []:
        line = LineString(path["geometry"])
        width = float(path["width_gu"])
        if line.length <= 0:
            raise TownLayoutError(f"empty access path {path['path_id']}")
        if line.buffer(width / 2.0, cap_style=2).intersection(hull_union).area > 1.0:
            raise TownLayoutError(f"access path intersects building {path['path_id']}")
        alleys.append({
            "alley_id": f"alley_{len(alleys):03d}",
            "source_path_id": path["path_id"],
            "block_id": path["block_id"],
            "courtyard_id": path["courtyard_id"],
            "mouth_id": path["mouth_id"],
            "hierarchy": "pedestrian_alley",
            "clear_width_gu": width,
            "polyline": path["geometry"],
        })
    open_spaces = [
        *({"space_id": row["courtyard_id"], "kind": "courtyard",
           "block_id": row["block_id"], "polygon": row["polygon"],
           "access_id": row["access_path_id"]}
          for row in source.get("courtyards") or []),
        *source.get("verges", []),
    ]
    classified = {row["kind"] for row in open_spaces}
    out = dict(source)
    out.update({
        "stage_id": "r7_circulation",
        "alleys": alleys,
        "open_spaces": open_spaces,
        "circulation_metrics": {
            "alley_count": len(alleys),
            "courtyard_count": sum(x["kind"] == "courtyard" for x in open_spaces),
            "verge_count": sum(x["kind"] == "verge" for x in open_spaces),
            "development_reserve_count": sum(
                x["kind"] == "development_reserve" for x in open_spaces),
            "open_landscape_count": sum(
                x["kind"] == "open_landscape" for x in open_spaces),
            "unclassified_count": 0,
            "lane_frontage_placement_count": 0,
            "lane_frontage_reason": "population already inside upper brief band",
            "classified_kinds": sorted(classified),
        },
    })
    return out

