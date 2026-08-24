"""Street-demo: pre-R3 preview of arterials plus selective seam promotion.

Reads the R2 checkpoint, splits walled cells crossed by gate-ingress arterial
chains, then demotes a deterministic subset of core seams so fine cells merge
into varied blocks (1-2 stamp islands and longer runs), per the accepted
correction design
(`.opencode/runs/townlayout-phase21-recovery/2026-08-15_r2_visual_correction_design.md`
section 3a).  Guards: merged block area cap, and every cell part must stay
connected to the arterial network through promoted seams.  This module is a
review preview, not the R3 authority: no road corridors, curves, or final
block faces are produced.
"""
from __future__ import annotations

import math
from typing import Any

from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import split as shapely_split
from shapely.ops import unary_union

from .geometry import normalize_ring, polygon_from_ring

# Demo band edges in GU^2, expressed against the approximate p50 stamp hull
# (1.8M GU^2, from the v2 library stats) — block depth is measured in stamps.
STAMP_P50_GU2 = 1_800_000.0
ISLAND_MAX_GU2 = 2.0 * STAMP_P50_GU2       # 1-2 stamp island
STANDARD_MAX_GU2 = 6.5 * STAMP_P50_GU2     # ~2-3 stamps deep
MERGE_CAP_GU2 = 11.0 * STAMP_P50_GU2       # ~3 fine cells: a longer run
MIN_SHARED_BOUNDARY_GU = 96.0              # JUNCTION_MERGE_GU consistency
SLIVER_FLOOR_GU2 = 10_000.0


def _union_find_ops():
    parent: dict[Any, Any] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    return find, union


def _split_cells(walled: dict[str, Polygon], arterial_lines: list[LineString]) -> dict[tuple[str, int], Polygon]:
    """Split arterial-crossed cells; absorb slivers into the longest-shared neighbour part."""
    parts: dict[tuple[str, int], Polygon] = {}
    splitter = MultiLineString([list(line.coords) for line in arterial_lines]) if arterial_lines else None
    for pid in sorted(walled):
        poly = walled[pid]
        pieces = [poly]
        if splitter is not None and poly.intersection(splitter).length > 0:
            try:
                result = shapely_split(poly, splitter)
                pieces = [g for g in result.geoms if g.geom_type == "Polygon" and g.area > 0]
            except (ValueError, TypeError):
                pieces = [poly]
        pieces.sort(key=lambda g: -g.area)
        for index, piece in enumerate(pieces):
            parts[(pid, index)] = piece
    # Absorb slivers into the part sharing the longest boundary.
    for key in sorted([k for k, p in parts.items() if p.area < SLIVER_FLOOR_GU2]):
        if key not in parts:
            continue
        sliver = parts[key]
        best, best_len = None, MIN_SHARED_BOUNDARY_GU
        for other_key, other in parts.items():
            if other_key == key:
                continue
            shared = sliver.boundary.intersection(other.boundary).length
            if shared > best_len:
                best, best_len = other_key, shared
        if best is not None:
            merged = parts[best].union(sliver)
            if merged.geom_type == "Polygon" and not merged.interiors:
                parts[best] = merged
                del parts[key]
    return parts


def _seam_part_pairs(parts: dict[tuple[str, int], Polygon],
                     seams: list[dict]) -> dict[str, list[tuple[tuple[str, int], tuple[str, int]]]]:
    """Map each core seam to the part pairs still sharing it (>= 96 GU)."""
    by_patch: dict[str, list[tuple[str, int]]] = {}
    for key in parts:
        by_patch.setdefault(key[0], []).append(key)
    result: dict[str, list] = {}
    for seam in seams:
        pairs = []
        for ka in by_patch.get(seam["patch_left"], []):
            for kb in by_patch.get(seam["patch_right"], []):
                if parts[ka].boundary.intersection(parts[kb].boundary).length >= MIN_SHARED_BOUNDARY_GU:
                    pairs.append((ka, kb))
        if pairs:
            result[seam["edge_id"]] = sorted(pairs)
    return result


def _size_class(area: float) -> str:
    if area <= ISLAND_MAX_GU2:
        return "island"
    if area <= STANDARD_MAX_GU2:
        return "standard"
    return "run"


def build_street_demo(product: dict) -> dict[str, Any]:
    walled_ids = set(product.get("walled_patch_ids") or [])
    patches = {p["patch_id"]: polygon_from_ring(p["polygon"])
               for p in product.get("patches", []) if p.get("patch_id") in walled_ids}
    if not patches:
        from .validate import TownLayoutError
        raise TownLayoutError("street demo: no walled patches")
    arterials = [e for e in product.get("edges", []) if e.get("edge_role") == "gate_ingress"]
    arterial_lines = [LineString(e["geometry"]) for e in arterials]
    arterial_union = unary_union(arterial_lines) if arterial_lines else None
    seams = [e for e in product.get("edges", [])
             if e.get("edge_role") == "block"
             and e.get("patch_left") in walled_ids and e.get("patch_right") in walled_ids]

    parts = _split_cells(patches, arterial_lines)
    seam_pairs = _seam_part_pairs(parts, seams)
    arterial_parts = {key for key, poly in parts.items()
                      if arterial_union is not None and poly.distance(arterial_union) <= 1.0}

    # Street-network connectivity at the line level.  Node identity is not
    # reliable here (water cropping and short-edge coalescing leave
    # T-junctions), so two streets are connected exactly when their
    # geometries intersect.  Intersection adjacency is static; demotion only
    # removes vertices, so it is precomputed once.
    # The wall lane (R4 element) is part of the street network even in this
    # preview: perimeter seams dead-end on the ring, and without the lane the
    # network is fragmented.  It is never a demotion candidate.
    ring_pts = (product.get("planning_ring") or {}).get("ring") or []
    street_geoms: dict[str, LineString] = {e["edge_id"]: LineString(e["geometry"]) for e in seams}
    street_geoms.update({e["edge_id"]: LineString(e["geometry"]) for e in arterials})
    ring_line = None
    if len(ring_pts) >= 3:
        ring_line = LineString(ring_pts + [ring_pts[0]])
        street_geoms["wall_lane"] = ring_line
    street_ids = sorted(street_geoms)
    adjacency: dict[str, set[str]] = {sid: set() for sid in street_ids}
    for i, a in enumerate(street_ids):
        for b in street_ids[i + 1:]:
            if street_geoms[a].intersects(street_geoms[b]):
                adjacency[a].add(b)
                adjacency[b].add(a)

    def network_connected(excluded_seam: str | None) -> bool:
        find, union = _union_find_ops()
        active = [sid for sid in street_ids
                  if sid != excluded_seam and sid not in demoted]
        for sid in active:
            for other in adjacency[sid]:
                if other != excluded_seam and other not in demoted:
                    union(sid, other)
        return len({find(sid) for sid in active}) <= 1

    def parts_keep_frontage(seam: dict) -> bool:
        """Parts adjacent to a demoted seam must still touch another street."""
        for key, poly in ((k, parts[k]) for pair in seam_pairs.get(seam["edge_id"], []) for k in pair):
            has_street = False
            if arterial_union is not None and poly.distance(arterial_union) <= 1.0:
                has_street = True
            elif ring_line is not None and poly.distance(ring_line) <= 1.0:
                has_street = True
            else:
                for other in seams:
                    if other["edge_id"] in demoted or other["edge_id"] == seam["edge_id"]:
                        continue
                    if any(key in pair for pair in seam_pairs.get(other["edge_id"], [])):
                        has_street = True
                        break
            if not has_street:
                return False
        return True

    # Demote-from-full-grid: every seam starts promoted; repeatedly apply the
    # feasible merge with the smallest resulting block area (tie: edge id).
    demoted: set[str] = set()

    def current_blocks() -> dict[tuple[str, int], Any]:
        find, union = _union_find_ops()
        for seam in seams:
            if seam["edge_id"] in demoted:
                for ka, kb in seam_pairs.get(seam["edge_id"], []):
                    union(ka, kb)
        return {key: find(key) for key in parts}

    while True:
        blocks = current_blocks()
        block_area: dict[Any, float] = {}
        for key, root in blocks.items():
            block_area[root] = block_area.get(root, 0.0) + parts[key].area
        best = None
        for seam in seams:
            eid = seam["edge_id"]
            if eid in demoted or eid not in seam_pairs:
                continue
            roots = set()
            for ka, kb in seam_pairs[eid]:
                roots.add(blocks[ka])
                roots.add(blocks[kb])
            if len(roots) < 2:
                continue  # already merged across this seam
            merged_area = sum(block_area[r] for r in roots)
            if merged_area > MERGE_CAP_GU2:
                continue
            if not network_connected(eid) or not parts_keep_frontage(seam):
                continue
            if best is None or (merged_area, eid) < (best[0], best[1]):
                best = (merged_area, eid)
        if best is None:
            break
        demoted.add(best[1])

    blocks = current_blocks()
    groups: dict[Any, list[tuple[str, int]]] = {}
    for key, root in blocks.items():
        groups.setdefault(root, []).append(key)
    block_rows = []
    for seq, (root, members) in enumerate(sorted(groups.items(), key=lambda item: min(m[0] for m in item[1]))):
        poly = unary_union([parts[m] for m in members])
        geoms = [poly] if poly.geom_type == "Polygon" else [g for g in poly.geoms if g.geom_type == "Polygon"]
        for part_no, geom in enumerate(geoms):
            block_rows.append({
                "block_id": f"demo_block_{seq:03d}" + (f"_p{part_no}" if part_no else ""),
                "area_gu2": float(geom.area),
                "footprint_equivalents": float(geom.area / STAMP_P50_GU2),
                "size_class": _size_class(geom.area),
                "member_patch_ids": sorted({m[0] for m in members}),
                "polygon": normalize_ring([[float(x), float(y)] for x, y in geom.exterior.coords])["ring"],
            })
    promoted = [e for e in seams if e["edge_id"] not in demoted]
    areas = sorted(b["area_gu2"] for b in block_rows)
    metrics = {
        "block_count": len(block_rows),
        "island_count": sum(b["size_class"] == "island" for b in block_rows),
        "standard_count": sum(b["size_class"] == "standard" for b in block_rows),
        "run_count": sum(b["size_class"] == "run" for b in block_rows),
        "promoted_seam_count": len(promoted),
        "demoted_seam_count": len(demoted),
        "cell_part_count": len(parts),
        "arterial_part_count": len(arterial_parts),
        "block_area_p10_gu2": areas[max(0, int(len(areas) * 0.1) - 1)] if areas else 0.0,
        "block_area_p50_gu2": areas[len(areas) // 2] if areas else 0.0,
        "block_area_p90_gu2": areas[min(len(areas) - 1, int(len(areas) * 0.9))] if areas else 0.0,
    }
    return {"stage_id": "r3_street_demo",
            "blocks": block_rows,
            "promoted_seams": [{"edge_id": e["edge_id"], "geometry": e["geometry"]} for e in promoted],
            "demoted_seam_ids": sorted(demoted),
            "arterials": [{"edge_id": e["edge_id"], "geometry": e["geometry"]} for e in arterials],
            "stage_metrics": metrics,
            "reports": [{"stage": "r3_street_demo", "status": "ok",
                         "message": f"blocks={len(block_rows)} islands={metrics['island_count']} "
                                    f"runs={metrics['run_count']} promoted={len(promoted)} "
                                    f"demoted={len(demoted)}"}]}


def render_street_demo(product: dict, demo: dict, survey_path, output_path) -> None:
    """Render blocks, promoted streets, and arterials over the site topdown."""
    import json
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    from .site_context import _plan_to_px, diagnostic_view, resolve_topdown_png
    from .validate import TownLayoutError

    survey_file = Path(survey_path)
    topdown = resolve_topdown_png(survey_file)
    if topdown is None:
        raise TownLayoutError("missing_diagnostic_input: site_topdown.png")
    survey = json.loads(survey_file.read_text(encoding="utf-8"))
    ring = product.get("planning_ring", {}).get("ring") or []
    image, mapping = diagnostic_view({"_diagnostic_bounds": [ring]}, topdown, survey,
                                     full_site=False)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    def to_px(point):
        return _plan_to_px(float(point[0]), float(point[1]), mapping)

    fills = {"island": (240, 220, 120, 90), "standard": (200, 200, 200, 80),
             "run": (235, 160, 80, 90)}
    for block in demo.get("blocks", []):
        ring_pts = [to_px(p) for p in block["polygon"]]
        draw.polygon(ring_pts, fill=fills.get(block["size_class"], (200, 200, 200, 80)),
                     outline=(30, 30, 30, 220))
    for seam in demo.get("promoted_seams", []):
        chain = seam.get("geometry") or []
        if len(chain) >= 2:
            draw.line([to_px(p) for p in chain], fill=(70, 70, 70, 235), width=3)
    for edge in demo.get("arterials", []):
        chain = edge.get("geometry") or []
        if len(chain) >= 2:
            draw.line([to_px(p) for p in chain], fill=(20, 20, 20, 255), width=7)
    if len(ring) >= 3:
        draw.line([to_px(p) for p in ring + [ring[0]]], fill=(0, 220, 255, 255), width=3)
    for node in product.get("nodes", []):
        if node.get("kind") == "gate":
            px, py = to_px(node["position"])
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(255, 0, 190, 255))
        elif node.get("kind") in ("source_junction", "source_terminus"):
            px, py = to_px(node["position"])
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=(255, 230, 40, 255))
    m = demo.get("stage_metrics", {})
    legend = (f"R3 STREET DEMO  blocks={m.get('block_count', 0)}"
              f" islands={m.get('island_count', 0)} standard={m.get('standard_count', 0)}"
              f" runs={m.get('run_count', 0)} streets={m.get('promoted_seam_count', 0)}"
              f" demoted={m.get('demoted_seam_count', 0)}")
    draw.rectangle((8, 8, min(image.width - 8, 8 + len(legend) * 7), 26),
                   fill=(0, 0, 0, 190))
    draw.text((12, 12), legend, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    draw.text((12, 30), "yellow=island(1-2 stamps)  gray=standard  orange=run  "
                        "black=arterial  dark gray=street  cyan=ring",
              fill=(255, 255, 255, 255), font=ImageFont.load_default())
    Image.alpha_composite(image, overlay).save(output_path)
