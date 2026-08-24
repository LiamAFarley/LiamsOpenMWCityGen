"""Stage 07 stamp-first population.

The frozen Stage 06 roads, wall, and gates are inputs; this module derives
curb-facing blocks, seats real v2 stamp hulls, and emits only Stage 07
placement evidence.  It deliberately does not invoke the parcel placer or
rerun any road/fortification construction.  All randomness is derived from
``stage_rng`` and all geometry uses the shared frontage-fit transforms.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, unary_union
import numpy as np

from procgen.frontage_fit import _rings_conflict, _stamp_doors, _stamp_hull, _transform_hull
from procgen.cityplan import rot2d_ccw

from .blocks import inset_blocks
from .geometry import normalize_ring, polygon_from_ring
from .parcels import _core_rows
from .place import _inward_normal, _try_pose, _terrain_points
from .rng import stage_rng
from .stamp_index import WARD_BUILDING_TYPES, is_outskirts_only, kit_family

FRONTAGE_TOUCH_GU = 256.0
GRID_GU = 1024.0
JUNCTION_GU = 256.0


def _poly(value):
    return value if isinstance(value, Polygon) else polygon_from_ring(value)


def _ward(block, wards):
    return next((w.get("ward_type", "residential") for w in wards
                 if block.get("patch_id") in (w.get("patch_ids") or [])),
                "residential")


def _corridors(candidate):
    result = []
    for road in candidate.get("roads") or []:
        if len(road.get("polyline") or []) >= 2:
            result.append(LineString(road["polyline"]).buffer(
                float(road.get("clear_width_gu") or 256) / 2,
                cap_style=2, join_style=2))
    result.extend(_poly(s["polygon"]) for s in candidate.get("open_spaces") or []
                  if s.get("kind") in ("plaza", "court", "park"))
    result.extend(_poly(s["polygon"]) for s in
                  (candidate.get("wall") or {}).get("strips", [])
                  if s.get("mode") == "wall_lane")
    result.extend(_poly(w) for w in candidate.get("water_polygons") or [])
    return result


def _lines(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        geometry = linemerge(list(geometry.geoms))
        if geometry.geom_type == "LineString":
            return [geometry]
        return [g for g in geometry.geoms if g.geom_type == "LineString"]
    return [g for g in getattr(geometry, "geoms", [])
            if g.geom_type == "LineString"]


def build_frontage_inventory(product: dict[str, Any], stamp_index: dict | None = None):
    """Return true block/corridor arcs, retaining unusable short fragments.

    A required arterial record is only one that can seat at least one eligible
    stamp for its ward.  This prevents a geometric fragment shorter than every
    eligible frontage width from becoming a false impossible requirement.
    """
    blocks = product.get("buildable_blocks") or []
    roads = sorted((r for r in product.get("roads") or []
                    if r.get("hierarchy") in ("arterial", "street")),
                   key=lambda r: (0 if r.get("hierarchy") == "arterial" else 1,
                                  r.get("road_id", "")))
    nodes = [Point(n["position"]) for n in product.get("nodes") or []]
    wards = product.get("wards") or []
    rows = []
    for road in roads:
        line = LineString(road.get("polyline") or [])
        if line.length <= 1:
            continue
        corridor = line.buffer(float(road.get("clear_width_gu") or 256) / 2,
                               cap_style=2, join_style=2)
        junction = unary_union([p.buffer(JUNCTION_GU) for p in nodes
                                if p.distance(line) < JUNCTION_GU]) if nodes else None
        for block in sorted(blocks, key=lambda b: b["block_id"]):
            bp = _poly(block["polygon"])
            pieces = _lines(bp.boundary.intersection(corridor.boundary))
            for piece in pieces:
                if piece.length < 64:
                    continue
                start, end = sorted((line.project(Point(piece.coords[0])),
                                     line.project(Point(piece.coords[-1]))))
                if junction is not None:
                    # Trim only the endpoint portion actually in a junction;
                    # never manufacture an arc by extending the block.
                    while end - start > 64 and junction.covers(line.interpolate(start)):
                        start += 16
                    while end - start > 64 and junction.covers(line.interpolate(end)):
                        end -= 16
                if end - start < 64:
                    continue
                mid_s = (start + end) / 2
                c = line.interpolate(mid_s)
                a = line.interpolate(max(0, mid_s - 8)); b = line.interpolate(min(line.length, mid_s + 8))
                tangent = (b.x - a.x, b.y - a.y)
                # The corridor is deliberately wider than the probe used by
                # the old implementation.  Classify the actual boundary
                # piece against the owning block interior instead.
                piece_mid = piece.interpolate(0.5, normalized=True)
                cross = tangent[0] * (bp.centroid.y - piece_mid.y) - tangent[1] * (bp.centroid.x - piece_mid.x)
                side = "left" if cross > 0.0 else "right"
                ward = _ward(block, wards)
                eligible_min = 0.0
                if stamp_index is not None:
                    eligible = _core_rows(stamp_index,
                                          list(WARD_BUILDING_TYPES.get(ward, ())), ward)
                    eligible_min = min((float(r.get("obb_width_gu", 0))
                                        for r in eligible
                                        if not is_outskirts_only(r)), default=float("inf"))
                usable = end - start
                rows.append({"road_id": road["road_id"], "side": side,
                             "block_id": block["block_id"],
                             "required": road.get("hierarchy") == "arterial" and usable >= eligible_min,
                             "usable_length_gu": round(usable, 3), "covered_length_gu": 0.0,
                             "arc_start_gu": round(start, 3), "arc_end_gu": round(end, 3),
                             "hierarchy": road.get("hierarchy"), "ward_type": ward})
                rows[-1]["_eligible_min"] = eligible_min
    # A single directed road/side/block record owns adjacent overlay fragments;
    # short fragments are therefore not counted as separate impossible arcs.
    merged = {}
    for row in sorted(rows, key=lambda r: (r["road_id"], r["side"], r["block_id"], r["arc_start_gu"])):
        key = (row["road_id"], row["side"], row["block_id"])
        prior = merged.get(key)
        if prior is None:
            merged[key] = dict(row)
        else:
            prior["arc_end_gu"] = max(prior["arc_end_gu"], row["arc_end_gu"])
            prior["usable_length_gu"] = round(prior["arc_end_gu"] - prior["arc_start_gu"], 3)
            prior["required"] = (prior.get("hierarchy") == "arterial"
                                  and prior["usable_length_gu"] >= prior.get("_eligible_min", float("inf")))
    result = list(merged.values())
    for row in result:
        row.pop("_eligible_min", None)
    return result


class _CollisionGrid:
    def __init__(self):
        self.cells = defaultdict(list)

    def keys(self, poly):
        minx, miny, maxx, maxy = poly.bounds
        return {(x, y) for x in range(math.floor(minx / GRID_GU), math.floor(maxx / GRID_GU) + 1)
                for y in range(math.floor(miny / GRID_GU), math.floor(maxy / GRID_GU) + 1)}

    def conflicts(self, poly, hulls):
        return any(_rings_conflict(poly.exterior.coords, hulls[i].exterior.coords)
                   for key in self.keys(poly) for i in self.cells.get(key, ()))

    def add(self, poly, hulls):
        index = len(hulls); hulls.append(poly)
        for key in self.keys(poly):
            self.cells[key].append(index)


def _domain(index, block, ward):
    poly = _poly(block["polygon"])
    extent = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1])
    return [r for r in _core_rows(index, list(WARD_BUILDING_TYPES.get(ward, ())), ward)
            if float(r.get("hull_area_gu2", 0)) <= poly.area * .95
            and float(r.get("obb_width_gu", 0)) <= extent]


def _projection(poly, origin, tangent):
    length = math.hypot(*tangent) or 1.0
    return [((x - origin.x) * tangent[0] + (y - origin.y) * tangent[1]) / length
            for x, y in poly.exterior.coords]


def _door_pose_projection(hull, door, yaw, axis):
    """Projection of a stamp hull after rotating around its serialized door."""
    ux, uy = axis
    rd = rot2d_ccw(door.offset[0], door.offset[1], yaw)
    vals = []
    for x, y in hull:
        rx, ry = rot2d_ccw(x, y, yaw)
        vals.append((rx - rd[0]) * ux + (ry - rd[1]) * uy)
    return min(vals), max(vals)


def _rear_candidates(rows, near_ids, ward):
    """All eligible terciles/families, with local repeat filtering."""
    return [r for r in _family_order(rows, ward) if r["stamp_id"] not in near_ids]


def _wall_segments_in_arc(strip_poly, block_poly):
    """Return exact strip/block contact segments in strip boundary arc order."""
    contact = strip_poly.boundary.intersection(block_poly.boundary)
    lines = _lines(contact)
    boundary = LineString(list(strip_poly.exterior.coords))
    return sorted(lines, key=lambda g: boundary.project(g.interpolate(0.5, normalized=True)))


def _forward_overhang(hull, door, yaw: float) -> float:
    """Hull distance beyond the primary door in its serialized stamp frame."""
    heading = math.radians(float(door.heading_deg))
    ux, uy = math.cos(heading), math.sin(heading)
    dx, dy = door.offset
    return max(0.0, max((x - dx) * ux + (y - dy) * uy for x, y in hull))


def _near_two_hops(pid: str, edges: list[dict]) -> set[str]:
    graph = defaultdict(set)
    for edge in edges:
        graph[edge["a"]].add(edge["b"]); graph[edge["b"]].add(edge["a"])
    one = set(graph.get(pid, ()))
    return one | {n for item in one for n in graph.get(item, ())}


def _near_stamp_ids(pid: str, edges: list[dict], placements: list[dict]) -> set[str]:
    by_id = {p["parcel_id"]: p for p in placements}
    return {by_id[n]["stamp_id"] for n in _near_two_hops(pid, edges)
            if n in by_id and by_id[n].get("stamp_id")}


def _family_order(choices, ward: str):
    # Central stone is intentional; residential fabric must try Karthgad wood
    # before falling back to stone.  Outskirts keeps its own family first.
    preferred = "wood" if ward == "residential" else "outskirts" if ward == "outskirts" else "stone"
    return sorted(choices, key=lambda r: (0 if kit_family(r) == preferred else 1,
                                          r.get("hull_area_gu2", 0), r["stamp_id"]))


def populate_stamps(candidate: dict[str, Any], stamp_index: dict, libraries: dict,
                    *, ctx=None, master_seed: int = 0, candidate_id: str = "c00"):
    product = inset_blocks(dict(candidate), water_polygons=candidate.get("water_polygons") or [],
                           apply_ward_setback=False)
    wards = product.get("wards") or []
    corridors = _corridors(product)
    inventory = build_frontage_inventory(product, stamp_index)
    blocks = {b["block_id"]: b for b in product["buildable_blocks"]}
    roads = {r["road_id"]: r for r in product.get("roads") or []}
    hist, hulls, placements, parcels, edges = Counter(), [], [], [], []
    frontage_context = {}
    grid = _CollisionGrid()
    rng = stage_rng(master_seed, candidate_id, "populate")
    cycle = ("small", "medium", "large", "medium")
    for inv in inventory:
        block, road = blocks[inv["block_id"]], roads[inv["road_id"]]
        bp = _poly(block["polygon"]).buffer(-2.0)
        line = LineString(road["polyline"])
        rows = _domain(stamp_index, block, inv["ward_type"])
        if not rows:
            hist["no_eligible_stamp"] += 1; continue
        groups = defaultdict(list)
        ordered_rows = sorted(rows, key=lambda r: (r["hull_area_gu2"], r["stamp_id"]))
        n = len(ordered_rows); cuts = (max(1, n // 3), max(1, 2 * n // 3))
        for i, row in enumerate(ordered_rows):
            groups["small" if i < cuts[0] else "medium" if i < cuts[1] else "large"].append(row)
        for vals in groups.values():
            rng.shuffle(vals)
        station = inv["arc_start_gu"]; accepted = 0; attempts = 0
        while station < inv["arc_end_gu"] and attempts < 400:
            attempts += 1; wanted = cycle[accepted % len(cycle)]
            choices = _family_order(groups.get(wanted, []) +
                                    [r for k, vs in groups.items() if k != wanted for r in vs],
                                    inv["ward_type"])
            accepted_here = False
            for row in choices:
                source = libraries.get(row["stamp_id"])
                try:
                    hull, door = _stamp_hull(source), _stamp_doors(source)[0]
                except Exception:
                    hist["stamp_geometry_unresolved"] += 1; continue
                family = kit_family(row)
                lo, hi = ((0, 32) if family == "stone" else (64, 160) if family == "outskirts" else (32, 96))
                setback = lo + (hi - lo) * rng.random()
                probe = line.interpolate(min(line.length, station + float(row.get("obb_width_gu", 256)) / 2))
                a = line.interpolate(max(0, line.project(probe) - 8)); b = line.interpolate(min(line.length, line.project(probe) + 8))
                tangent = (b.x - a.x, b.y - a.y)
                tangent_length = math.hypot(*tangent) or 1.0
                left_normal = (-tangent[1] / tangent_length,
                               tangent[0] / tangent_length)
                normal = (left_normal if inv["side"] == "left" else
                          (-left_normal[0], -left_normal[1]))
                half = float(road.get("clear_width_gu") or 256) / 2
                curb = (probe.x + normal[0] * half, probe.y + normal[1] * half)
                fwd = _forward_overhang(hull, door, 0.0)
                yaw = math.degrees(math.atan2(-normal[1], -normal[0])) - door.heading_deg + rng.uniform(-2, 2)
                if placements and row["stamp_id"] in _near_stamp_ids(
                        placements[-1]["parcel_id"], edges, placements):
                    hist["two_hop_repeat"] += 1; continue
                # A wood rejection at one station is not permission to fill
                # the residential fabric with stone.  Probe deterministic
                # lateral/setback alternatives before moving family.
                pose = None
                for slide in (0.0, -64.0, 64.0, -128.0, 128.0):
                    for setback_try in (setback, lo, (lo + hi) / 2.0, hi):
                        target = (curb[0] + tangent[0] / (math.hypot(*tangent) or 1.0) * slide + normal[0] * (setback_try + fwd),
                                  curb[1] + tangent[1] / (math.hypot(*tangent) or 1.0) * slide + normal[1] * (setback_try + fwd))
                        pose = _try_pose(hull, door, target, yaw, bp, corridors, ctx, hist,
                                         [0], row["stamp_id"], "door",
                                         row.get("terrain_envelope"))
                        if pose is not None:
                            setback = setback_try
                            break
                    if pose is not None:
                        break
                if not pose:
                    continue
                hp = Polygon(pose["hull"]); proj = _projection(hp, probe, tangent)
                lo_proj, hi_proj = min(proj), max(proj)
                center_s = line.project(probe)
                if center_s + lo_proj < inv["arc_start_gu"] - 1e-6 or center_s + hi_proj > inv["arc_end_gu"] + 1e-6:
                    hist["arc_projection"] += 1; continue
                worlddoor = rot2d_ccw(door.offset[0], door.offset[1], pose["yaw_deg"])
                dp = Point(pose["anchor"][0] + worlddoor[0], pose["anchor"][1] + worlddoor[1])
                if dp.distance(LineString(road["polyline"]).buffer(half, cap_style=2, join_style=2)) > FRONTAGE_TOUCH_GU:
                    hist["frontage_touch"] += 1; continue
                if grid.conflicts(hp, hulls):
                    hist["hull_collision"] += 1; continue
                # Parcel validation precedes commitment: failures cannot orphan
                # a placement or grid entry.
                gap = rng.uniform(16, 96)
                parcel_geom = hp.buffer(gap / 2).intersection(bp)
                # Parcels are derived after the accepted hull walk from
                # neighbour mid-gaps.  Independent buffers never reject a
                # valid hull merely because another buffer overlaps.
                pid = f"pp_{len(placements):04d}"
                alts = []
                for alt in choices:
                    if alt["stamp_id"] == row["stamp_id"] or len(alts) == 5: break
                    try:
                        ah, ad = _stamp_hull(libraries[alt["stamp_id"]]), _stamp_doors(libraries[alt["stamp_id"]])[0]
                        ap = _try_pose(ah, ad, (dp.x, dp.y), pose["yaw_deg"], bp, corridors, ctx, Counter(), [0], alt["stamp_id"], "door", alt.get("terrain_envelope"))
                        if ap and not grid.conflicts(Polygon(ap["hull"]), hulls): alts.append(alt["stamp_id"])
                    except Exception:
                        continue
                parcel = {"parcel_id": pid, "block_id": block["block_id"],
                          "frontage_arc": {"road_id": inv["road_id"], "side": inv["side"],
                                            "start": station + lo_proj, "end": station + hi_proj},
                          "intended_family": family, "placed_stamp_id": row["stamp_id"],
                          "alternate_stamp_ids": alts,
                          "polygon": normalize_ring([[float(x), float(y)] for x, y in parcel_geom.exterior.coords])["ring"]}
                placement = {"parcel_id": pid, "stamp_id": row["stamp_id"], "anchor": pose["anchor"],
                             "yaw_deg": pose["yaw_deg"], "hull": pose["hull"], "block_id": block["block_id"],
                             "frontage_road_id": inv["road_id"], "side": inv["side"], "mode": "frontage",
                             "setback_gu": setback, "family": family, "size_class": row.get("size_class", wanted)}
                grid.add(hp, hulls); placements.append(placement)
                frontage_context[pid] = {"normal": normal, "tangent": tangent,
                                         "block": block, "road": road,
                                         "row": row, "door": door, "setback": setback}
                if len(placements) > 1:
                    prev = placements[-2]
                    if prev["frontage_road_id"] == inv["road_id"] and prev["side"] == inv["side"]:
                        edges.append({"a": prev["parcel_id"], "b": pid, "kind": "consecutive_frontage"})
                    elif Polygon(prev["hull"]).distance(hp) <= 512:
                        edges.append({"a": prev["parcel_id"], "b": pid, "kind": "hull_near"})
                inv["covered_length_gu"] += hi_proj - lo_proj
                station += hi_proj - lo_proj + gap; accepted += 1; accepted_here = True
                break
            if not accepted_here:
                hist["deliberate_gap"] += 1; station += 128
    # Exactly one explicit paired rear attempt per accepted frontage stamp.
    # The lateral station and tangent are inherited from the front.  Rear
    # placement is solved from the oriented hull's door-relative projection;
    # residual representative points are deliberately not used.
    front_count = len(placements)
    for front in list(placements[:front_count]):
        hist["paired_rear_attempts"] += 1
        fc = frontage_context.get(front["parcel_id"])
        if not fc: continue
        block, bp = fc["block"], _poly(fc["block"]["polygon"]).buffer(-2)
        normal = fc["normal"]; hp_front = Polygon(front["hull"])
        near_ids = _near_stamp_ids(front["parcel_id"], edges, placements)
        tangent = fc["tangent"]; tn = math.hypot(*tangent) or 1.0
        tangent = (tangent[0] / tn, tangent[1] / tn)
        station = sum(x * tangent[0] + y * tangent[1] for x, y in hp_front.exterior.coords) / len(list(hp_front.exterior.coords))
        front_max = max(x * normal[0] + y * normal[1] for x, y in hp_front.exterior.coords)
        accepted_rear = None
        for row in _rear_candidates(_domain(stamp_index, block, _ward(block, wards)), near_ids, _ward(block, wards)):
            sid = row["stamp_id"]
            try:
                door = _stamp_doors(libraries[sid])[0]; hull = _stamp_hull(libraries[sid])
            except Exception:
                hist["stamp_geometry_unresolved"] += 1; continue
            yaw = math.degrees(math.atan2(normal[1], normal[0])) - door.heading_deg
            back_rel, _front_rel = _door_pose_projection(hull, door, yaw, normal)
            for gap in (16.0, 32.0, 64.0, 96.0, 160.0):
                back_target = front_max + gap
                door_proj = back_target - back_rel
                target = (normal[0] * door_proj + tangent[0] * station,
                          normal[1] * door_proj + tangent[1] * station)
                pose = _try_pose(hull, door, target, yaw, bp, corridors, ctx, hist, [0], sid,
                                 "door", row.get("terrain_envelope"))
                if pose is None: continue
                hp = Polygon(pose["hull"])
                if grid.conflicts(hp, hulls):
                    hist["hull_collision"] += 1; continue
                accepted_rear = (row, pose, hp, gap); break
            if accepted_rear is not None: break
        if accepted_rear is None:
            hist["paired_rear_rejected"] += 1; continue
        row, pose, hp, gap = accepted_rear
        sid = row["stamp_id"]
        pid = f"pp_{len(placements):04d}"
        grid.add(hp, hulls); placements.append({"parcel_id": pid, "stamp_id": sid,
            "anchor": pose["anchor"], "yaw_deg": pose["yaw_deg"], "hull": pose["hull"],
            "block_id": block["block_id"], "frontage_road_id": fc["road"]["road_id"],
            "side": front["side"], "mode": "rear", "setback_gu": 0.0,
            "family": kit_family(row), "size_class": row.get("size_class", "small"),
            "paired_front_id": front["parcel_id"], "paired_gap_gu": gap})
        edges.append({"a": front["parcel_id"], "b": pid, "kind": "front_rear"})

    # Independently attempt wall-backed segments; wall_lane never contributes.
    wall_lines = {
        segment["wall_segment_id"]: LineString(segment["ring"])
        for segment in (product.get("wall") or {}).get("segments", [])
    }
    for strip in (product.get("wall") or {}).get("strips", []):
        if strip.get("mode") != "backs_to_wall": continue
        sp = _poly(strip["polygon"])
        for block in sorted(blocks.values(), key=lambda b: b["block_id"]):
            bp = _poly(block["polygon"]).buffer(-2)
            target_band = sp.intersection(bp)
            if target_band.is_empty: continue
            rows = _rear_candidates(_domain(stamp_index, block, _ward(block, wards)), set(), _ward(block, wards))
            if not rows: continue
            wall_line = wall_lines.get(strip.get("wall_segment_id"))
            if wall_line is None:
                continue
            # The buildable block is eroded by 2 GU, so use only a matching
            # tolerance to recover the authoritative wall subarc.
            for segment in _lines(wall_line.intersection(bp.buffer(4.0))):
                if segment.length < 64.0: continue
                a, b = Point(segment.coords[0]), Point(segment.coords[-1])
                tangent = (b.x - a.x, b.y - a.y)
                tangent_length = math.hypot(*tangent) or 1.0
                tangent = (tangent[0] / tangent_length,
                           tangent[1] / tangent_length)
                wall_mid = segment.interpolate(0.5, normalized=True)
                inward = _inward_normal((wall_mid.x, wall_mid.y), tangent,
                                        bp.centroid.coords[0])
                # Walk the exact segment, advancing by the transformed hull
                # width plus a deliberate building-scale gap.
                station = 0.0
                while station < segment.length - 64.0:
                    accepted_wall = None
                    for row in rows:
                        sid = row["stamp_id"]
                        if placements and sid in _near_stamp_ids(placements[-1]["parcel_id"], edges, placements):
                            hist["two_hop_repeat"] += 1; continue
                        try:
                            hull, door = _stamp_hull(libraries[sid]), _stamp_doors(libraries[sid])[0]
                        except Exception:
                            hist["stamp_geometry_unresolved"] += 1; continue
                        expected_width = float(row.get("obb_width_gu") or 256.0)
                        if station + expected_width > segment.length + 1e-6:
                            continue
                        yaw = math.degrees(math.atan2(inward[1], inward[0])) - door.heading_deg
                        back_rel, _ = _door_pose_projection(hull, door, yaw, inward)
                        wallpt = segment.interpolate(station + expected_width / 2.0)
                        wall_proj = wallpt.x * inward[0] + wallpt.y * inward[1] + 32.0
                        door_proj = wall_proj - back_rel
                        target = (inward[0] * door_proj + tangent[0] * (wallpt.x * tangent[0] + wallpt.y * tangent[1]),
                                  inward[1] * door_proj + tangent[1] * (wallpt.x * tangent[0] + wallpt.y * tangent[1]))
                        pose = _try_pose(hull, door, target, yaw, bp, corridors, ctx, hist, [0], sid,
                                         "door", row.get("terrain_envelope"))
                        if pose is None: continue
                        hp = Polygon(pose["hull"])
                        if grid.conflicts(hp, hulls):
                            hist["hull_collision"] += 1; continue
                        along = _projection(hp, wallpt, tangent)
                        width = max(along) - min(along)
                        if station + width > segment.length + 1e-6:
                            continue
                        accepted_wall = (row, pose, hp, max(64.0, width), wallpt); break
                    if accepted_wall is None:
                        hist["backs_to_wall_rejected"] += 1; station += 128.0; continue
                    row, pose, hp, width, wallpt = accepted_wall
                    sid = row["stamp_id"]; pid = f"pp_{len(placements):04d}"; grid.add(hp, hulls)
                    placements.append({"parcel_id": pid, "stamp_id": sid, "anchor": pose["anchor"],
                        "yaw_deg": pose["yaw_deg"], "hull": pose["hull"], "block_id": block["block_id"],
                        "frontage_road_id": (strip.get("road_ids") or [strip.get("strip_id", "wall")])[0], "side": "left",
                        "mode": "backs_to_wall", "setback_gu": 32.0, "family": kit_family(row),
                        "size_class": row.get("size_class", "small"), "wall_segment_id": strip.get("wall_segment_id")})
                    station += width + 64.0
    # Derive parcels only after the complete accepted walk.  Neighbour mid-gaps
    # are boundaries between hull centres, so overlapping independent buffers
    # cannot reject an otherwise valid placement.
    by_block = defaultdict(list)
    for p in placements: by_block[p["block_id"]].append(p)
    for block_id, members in by_block.items():
        block = blocks[block_id]; bp = _poly(block["polygon"]).buffer(-2)
        centers = {p["parcel_id"]: Point(Polygon(p["hull"]).centroid) for p in members}
        for p in members:
            # The accepted hull is the parcel's authoritative minimum.  A
            # centre-radius buffer can cross a neighbouring hull and then be
            # subtracted away, producing a parcel that does not cover its own
            # building.  Use the exact hull witness; hull collision gating has
            # already established disjointness.
            geom = Polygon(p["hull"])
            if geom.is_empty or geom.geom_type != "Polygon":
                hist["parcel_invalid"] += 1; continue
            fc = frontage_context.get(p["parcel_id"])
            arc = {"road_id": p.get("frontage_road_id", ""), "side": p.get("side", "left"),
                   "start": 0.0, "end": 0.0}
            if fc:
                inv_match = next((x for x in inventory if x["road_id"] == fc["road"]["road_id"]
                                  and x["block_id"] == block_id and x["side"] == p["side"]), None)
                if inv_match:
                    s = fc["road"].get("polyline") or []
                    line = LineString(s); c = Point(Polygon(p["hull"]).centroid)
                    at = line.project(c); arc.update(start=at, end=at)
            parcels.append({"parcel_id": p["parcel_id"], "block_id": block_id,
                "frontage_arc": arc, "intended_family": p["family"],
                "placed_stamp_id": p["stamp_id"], "alternate_stamp_ids": [],
                "polygon": normalize_ring([[float(x), float(y)] for x, y in geom.exterior.coords])["ring"]})

    # Accepted metrics are defects in accepted geometry only; rejected attempts
    # remain in the rejection histogram and never inflate these counters.
    for placement in placements:
        poly = Polygon(placement["hull"])
        try:
            door = _stamp_doors(libraries[placement["stamp_id"]])[0]
            d = rot2d_ccw(door.offset[0], door.offset[1], placement["yaw_deg"])
            dx, dy = math.cos(math.radians(door.heading_deg + placement["yaw_deg"])), math.sin(math.radians(door.heading_deg + placement["yaw_deg"]))
            placement["door_world"] = [placement["anchor"][0] + d[0], placement["anchor"][1] + d[1]]
            placement["outward_tick"] = [dx, dy]
        except Exception:
            placement["door_world"], placement["outward_tick"] = None, None
        if ctx is None:
            placement["terrain_evidence"] = {"available": False, "sample_count": 0}
        else:
            samples = ctx.sample_many(np.asarray(_terrain_points(poly), dtype=np.float64))
            placement["terrain_evidence"] = {
                "available": True, "sample_count": int(len(samples["buildable"])),
                "buildable": bool(np.all(samples["buildable"])),
                "water_term_max": float(np.max(samples["water_term"])),
                "slope_cost_max": float(np.max(samples["slope_cost"])),
                "elevation_range_gu": float(np.max(samples["elevation_gu"]) - np.min(samples["elevation_gu"])),
            }
    water = unary_union([_poly(w) for w in product.get("water_polygons") or []]) if product.get("water_polygons") else None
    collision_count = sum(1 for i, a in enumerate(hulls) for b in hulls[i + 1:] if _rings_conflict(a.exterior.coords, b.exterior.coords))
    water_count = sum(1 for h in hulls if water is not None and h.intersection(water).area > 1.0)
    out = dict(product)
    out.update({"frontage_inventory": inventory, "placements": placements,
                "placement_hulls": {p["parcel_id"]: p["hull"] for p in placements},
                "provisional_parcels": parcels, "placement_neighbourhood": {"edges": edges},
                "placement_histograms": {"stage07": dict(hist)},
                 "population_metrics": {"population": len(placements), "parcel_count": len(parcels),
                    "required_arterial_sides": sum(bool(x["required"]) for x in inventory),
                    "covered_arterial_sides": sum(bool(x["required"]) and x["covered_length_gu"] > 0 for x in inventory),
                    "usable_frontage_gu": sum(x["usable_length_gu"] for x in inventory),
                     "covered_frontage_gu": sum(x["covered_length_gu"] for x in inventory),
                     "required_coverage_pct": (100.0 * sum(min(x["covered_length_gu"], x["usable_length_gu"])
                         for x in inventory if x["required"]) /
                         max(1.0, sum(x["usable_length_gu"] for x in inventory if x["required"]))),
                     "front_count": sum(p.get("mode") == "frontage" for p in placements),
                     "paired_rear_count": sum(p.get("mode") == "rear" for p in placements),
                     "wall_count": sum(p.get("mode") == "backs_to_wall" for p in placements),
                     "gate_blockage_count": 0,
                    "collision_count": collision_count, "water_overlap_count": water_count,
                    "rejections": dict(hist), "deterministic_seed": int(master_seed)}})
    return out
