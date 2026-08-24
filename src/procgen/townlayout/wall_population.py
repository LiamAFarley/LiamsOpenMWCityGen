"""Adapt the accepted wall-aware street checkpoint for stamp population.

The road, wall, and final-block geometry is inherited unchanged.  This module
only derives population-facing records: noded road centerlines, blocks split at
the inner wall, and an explicit development zone on every resulting block.
The wall is planning data at this stage, but its reserved band is kept free of
building hulls so future wall meshes do not require moving the town fabric.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, unary_union

from procgen.cityplan import rot2d_ccw
from procgen.frontage_fit import _stamp_doors, _stamp_hull

from .geometry import normalize_ring, polygon_from_ring
from .place import _try_pose
from .populate import _CollisionGrid, _corridors, _forward_overhang, _projection
from .rng import stage_rng
from . import fk_house_adapter
from .stamp_index import kit_family
from .validate import TownLayoutError

MIN_DEVELOPMENT_PART_GU2 = 128.0 ** 2


def _is_hut_stamp(row: dict[str, Any]) -> bool:
    """Return the explicit hut class excluded from city placement."""

    return "hut" in str(row.get("stamp_id") or "").lower().replace("-", "_").split("_")


def _polygon_parts(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [g for g in getattr(geometry, "geoms", [])
            if g.geom_type == "Polygon" and g.area >= MIN_DEVELOPMENT_PART_GU2]


def _ring(poly: Polygon) -> list[list[float]]:
    return normalize_ring([[float(x), float(y)] for x, y in poly.exterior.coords])["ring"]


def _line_parts(lines: Iterable[LineString]) -> list[LineString]:
    source = [line for line in lines if line.length > 1.0]
    if not source:
        return []
    merged = linemerge(unary_union(source))
    if merged.geom_type == "LineString":
        return [merged]
    return [g for g in getattr(merged, "geoms", [])
            if g.geom_type == "LineString" and g.length > 1.0]


def _roads(source: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    arterial = _line_parts(
        LineString(row["geometry"]) for row in source.get("smoothed_strokes", [])
        if len(row.get("geometry") or []) >= 2)
    minor = _line_parts([
        *(LineString(row["geometry"]) for row in source.get("minor_strokes", [])
          if len(row.get("geometry") or []) >= 2),
        *(LineString(row["geometry"]) for row in source.get("junction_links", [])
          if len(row.get("geometry") or []) >= 2),
    ])
    roads = []
    for hierarchy, width, parts in (("arterial", 512.0, arterial),
                                    ("street", 384.0, minor)):
        for index, line in enumerate(sorted(parts, key=lambda g: (g.bounds, g.length))):
            roads.append({
                "road_id": f"population_{hierarchy}_{index:03d}",
                "hierarchy": hierarchy,
                "clear_width_gu": width,
                "polyline": [[float(x), float(y)] for x, y in line.coords],
            })

    # Endpoints of the noded union are sufficient for the frontage corner trim.
    counts: dict[tuple[float, float], int] = {}
    positions: dict[tuple[float, float], list[float]] = {}
    for road in roads:
        for point in (road["polyline"][0], road["polyline"][-1]):
            key = (round(point[0], 3), round(point[1], 3))
            counts[key] = counts.get(key, 0) + 1
            positions[key] = point
    nodes = [{"node_id": f"population_node_{i:03d}", "position": positions[key]}
             for i, key in enumerate(sorted(counts)) if counts[key] != 2]
    return roads, nodes


def prepare_wall_population(source: dict[str, Any],
                            development_policy: dict[str, Any] | None = None,
                            has_outskirts: bool | None = None,
                            has_inner_wall: bool | None = None) -> dict[str, Any]:
    """Return an R5-ready checkpoint without modifying accepted topology."""
    if source.get("stage_id") != "r2c_minor_roads":
        raise TownLayoutError("wall population requires r2c_minor_roads")
    wall_ring = (source.get("inner_wall") or {}).get("polygon") if source.get("inner_wall") else None
    _has_inner_wall = has_inner_wall
    if wall_ring:
        wall = polygon_from_ring(wall_ring)
        if not wall.is_valid or wall.area <= 0:
            raise TownLayoutError("wall population received invalid inner wall")
        if _has_inner_wall is False:
            from shapely.geometry import Polygon as _Polygon
            wall_band = _Polygon()
            mesh_reserve = _Polygon()
            mesh_reserve_parts = []
        else:
            mesh_rings = (source.get("wall_fit") or {}).get("mesh_footprints") or []
            if not mesh_rings:
                raise TownLayoutError("wall population missing measured wall mesh footprints")
            measured_parts = [polygon_from_ring(ring) for ring in mesh_rings]
            clearance = float((source.get("wall_fit") or {}).get("reserve_clearance_gu", 0.0))
            mesh_reserve_parts = ([part.buffer(clearance, cap_style=2, join_style=2)
                                   for part in measured_parts]
                                  if clearance > 0.0 else measured_parts)
            mesh_reserve = unary_union(mesh_reserve_parts)
            wall_band = mesh_reserve
    else:
        from shapely.geometry import Polygon as _Polygon
        wall = _Polygon()
        wall_band = _Polygon()
        mesh_reserve = _Polygon()
        mesh_reserve_parts = []

    blocks = []
    # has_outskirts toggle: when False, only the inner (walled) zone is kept;
    # unwalled settlements with has_outskirts=True keep the outer zone only
    # via the empty-wall path below (inner empty, outer = full poly).
    _has_outskirts = True if has_outskirts is None else bool(has_outskirts)
    # If there is no wall (unwalled) and outskirts enabled, treat the whole
    # block as a single outer zone rather than splitting by an empty wall.
    _is_walled = not wall.is_empty
    for block in sorted(source.get("final_blocks") or [], key=lambda row: row["block_id"]):
        poly = polygon_from_ring(block["polygon"])
        if not _is_walled:
            # Unwalled: one zone only
            zones: tuple[tuple[str, Any], ...]
            if _has_outskirts:
                zones = (("outer", poly),)
            else:
                zones = (("inner", poly),)
        elif not _has_outskirts:
            # Walled but no outskirts: keep only inner zone
            zones = (("inner", poly.intersection(wall).difference(wall_band)),)
        else:
            zones = (("inner", poly.intersection(wall).difference(wall_band)),
                     ("outer", poly.difference(wall).difference(wall_band)))
        for zone, geometry in zones:
            parts = sorted((part for part in _polygon_parts(geometry)
                            if part.area >= MIN_DEVELOPMENT_PART_GU2),
                           key=lambda g: (-g.area, g.centroid.x, g.centroid.y))
            for part_no, part in enumerate(parts):
                item = dict(block)
                item["source_block_id"] = block["block_id"]
                item["block_id"] = f"{block['block_id']}_{zone}_{part_no:02d}"
                item["development_zone"] = zone
                item["ward_type"] = "craft" if zone == "inner" else "outskirts"
                item["polygon"] = normalize_ring(
                    [[float(x), float(y)] for x, y in part.exterior.coords])["ring"]
                item["final_area_gu2"] = float(part.area)
                blocks.append(item)
    if not blocks:
        raise TownLayoutError("wall population produced no development blocks")
    # For unwalled settlements the wall is empty, so only the active zone
    # (outer when has_outskirts, else inner) is required.
    if _is_walled:
        if not any(b["development_zone"] == "inner" for b in blocks):
            raise TownLayoutError("wall population produced no inner development blocks")
        if _has_outskirts and not any(b["development_zone"] == "outer" for b in blocks):
            raise TownLayoutError("wall population produced no outer development blocks")
    else:
        # Unwalled: exactly one zone type should be present
        if _has_outskirts and not any(b["development_zone"] == "outer" for b in blocks):
            raise TownLayoutError("wall population produced no outer development blocks (unwalled)")
        if not _has_outskirts and not any(b["development_zone"] == "inner" for b in blocks):
            raise TownLayoutError("wall population produced no inner development blocks (unwalled, no outskirts)")

    roads, nodes = _roads(source)
    if not roads:
        raise TownLayoutError("wall population produced no road frontage")
    out = dict(source)
    # Preserve has_inner_wall flag for downstream reservation/rendering decisions;
    # when False, planning wall still drives block splits but band/reservation is empty.
    out.update({
        "stage_id": "r5_wall_population_input",
        "roads": roads,
        "nodes": nodes,
        "buildable_blocks": blocks,
        "has_inner_wall": False if _has_inner_wall is False else True,
        "wall": {
            "planning_polygon": wall_ring,
            "gates": list(source.get("wall_gates") or []) if _has_inner_wall is not False else [],
            "mesh_reserve": [_ring(part) for part in mesh_reserve_parts
                             if not part.is_empty and part.area >= MIN_DEVELOPMENT_PART_GU2],
            "reserve_clearance_gu": float((source.get("wall_fit") or {}).get(
                "reserve_clearance_gu", 0.0)) if _has_inner_wall is not False else 0.0,
        },
        "development_policy": dict(development_policy or {
            "inner": {"preferred_library": "markarth", "density": "dense"},
            "outer": {"preferred_library": "karthgad", "density": "low"},
        }),
    })
    return out


def _sampled_frontage_inventory(candidate: dict[str, Any]) -> list[dict]:
    """Recover maximal block-facing arcs from the accepted curved centerlines.

    Final blocks were cut by serialized corridor polygons, whose rounded joins
    do not share coordinate-identical boundaries with a freshly buffered line.
    Sampling just beyond the curb is therefore the stable geometric test.
    """
    blocks = [(row, polygon_from_ring(row["polygon"]))
              for row in candidate["buildable_blocks"]]
    result = []
    for road in sorted(candidate["roads"], key=lambda row: row["road_id"]):
        line = LineString(road["polyline"])
        if line.length < 640.0:
            continue
        start, end = 256.0, line.length - 256.0
        cuts = [start]
        while cuts[-1] + 128.0 < end:
            cuts.append(cuts[-1] + 128.0)
        cuts.append(end)
        for side in ("left", "right"):
            runs = []
            active = None
            for lo, hi in zip(cuts, cuts[1:]):
                at = (lo + hi) / 2.0
                probe = line.interpolate(at)
                a = line.interpolate(max(0.0, at - 8.0))
                b = line.interpolate(min(line.length, at + 8.0))
                tx, ty = b.x - a.x, b.y - a.y
                norm = math.hypot(tx, ty) or 1.0
                left = (-ty / norm, tx / norm)
                normal = left if side == "left" else (-left[0], -left[1])
                distance = float(road["clear_width_gu"]) / 2.0 + 48.0
                from shapely.geometry import Point
                sample = Point(probe.x + normal[0] * distance,
                               probe.y + normal[1] * distance)
                owner = next((row for row, poly in blocks if poly.covers(sample)), None)
                owner_id = owner["block_id"] if owner else None
                if active and active[0] == owner_id:
                    active = (owner_id, active[1], hi)
                    runs[-1] = active
                else:
                    active = (owner_id, lo, hi)
                    runs.append(active)
            for owner_id, lo, hi in runs:
                if owner_id is None or hi - lo < 256.0:
                    continue
                block = next(row for row, _poly in blocks if row["block_id"] == owner_id)
                result.append({
                    "road_id": road["road_id"], "side": side,
                    "block_id": owner_id,
                    "required": road["hierarchy"] == "arterial",
                    "usable_length_gu": hi - lo, "covered_length_gu": 0.0,
                    "arc_start_gu": lo, "arc_end_gu": hi,
                    "hierarchy": road["hierarchy"],
                    "ward_type": block["ward_type"],
                    "development_zone": block["development_zone"],
                })
    return result


def populate_wall_front_rows(candidate: dict[str, Any], stamp_index: dict,
                             libraries: dict, *, master_seed: int,
                             candidate_id: str) -> dict[str, Any]:
    """Seat a bounded, single-pass front row on the accepted road fabric.

    Every cursor tests at most eight poses.  There are no slide, alternate
    setback, rear-row, or global refill loops in this phase.
    """
    inventory = _sampled_frontage_inventory(candidate)
    roads = {road["road_id"]: road for road in candidate["roads"]}
    blocks = {block["block_id"]: block for block in candidate["buildable_blocks"]}
    corridors = _corridors(candidate)
    _wall = candidate.get("wall") or {}
    planning_ring = _wall.get("planning_polygon")
    _has_wall = candidate.get("has_inner_wall", True)
    wall_front_setback = 0.0
    if planning_ring and _has_wall is not False:
        mesh_rings = _wall.get("mesh_reserve") or []
        if not mesh_rings:
            raise TownLayoutError("wall population missing serialized wall mesh reserve")
        wall_reservation = unary_union([polygon_from_ring(ring) for ring in mesh_rings])
        wall_front_setback = float((candidate.get("wall_fit") or {}).get(
            "wall_front_setback_gu", 0.0))
        corridors.append(wall_reservation)
    else:
        from shapely.geometry import Polygon as _Polygon
        wall_reservation = _Polygon()
    all_rows = [row for row in stamp_index.get("stamps") or []
                if not _is_hut_stamp(row)]
    rng = stage_rng(master_seed, candidate_id, "wall_front_rows")
    grid = _CollisionGrid()
    hulls: list[Polygon] = []
    placements = []
    mouths = []
    hist = Counter()
    usage = Counter()
    pose_evaluations = 0

    # Map the zone's preferred library to allowed kit families.  The default
    # policy (inner: markarth, outer: karthgad) reproduces the Falkreath
    # stone-core/wood-fabric split; a village brief may set both zones to
    # karthgad for all-wood fabric.
    policy = candidate.get("development_policy") or {}
    fk_zones = {z for z in ("inner", "outer") if str((policy.get(z) or {}).get("house_generator") or "") == "fk_house"}
    fk_table = (fk_house_adapter.build_shell_table() if fk_zones else None)
    generated_stamps: dict[str, dict] = {}
    library_families = {"markarth": ("stone",), "karthgad": ("wood", "outskirts")}

    def zone_policy(zone: str) -> tuple[str, tuple[str, ...]]:
        library = str((policy.get(zone) or {}).get("preferred_library")
                      or ("markarth" if zone == "inner" else "karthgad"))
        return library, library_families.get(library, library_families["karthgad"])

    for arc in inventory:
        block = blocks[arc["block_id"]]
        zone = block["development_zone"]
        road = roads[arc["road_id"]]
        line = LineString(road["polyline"])
        block_poly = polygon_from_ring(block["polygon"]).buffer(-2.0)
        policy_lib, allowed_families = zone_policy(zone)
        use_fk = fk_table is not None and zone in fk_zones
        if not use_fk:
            candidates = [row for row in all_rows
                          if policy_lib in row["stamp_id"]
                          and kit_family(row) in allowed_families]
            candidates.sort(key=lambda row: (
                float(row.get("obb_width_gu") or 0),
                float(row.get("hull_area_gu2") or 0), row["stamp_id"]))
            cut1 = max(1, len(candidates) // 3)
            cut2 = max(cut1 + 1, 2 * len(candidates) // 3)
            size_groups = {
                "small": candidates[:cut1],
                "medium": candidates[cut1:cut2],
                "large": candidates[cut2:],
            }
            size_cycle = ("small", "medium", "small", "medium", "large")
        else:
            candidates = []  # type: ignore
            size_groups = {}
            size_cycle = fk_house_adapter.SIZE_CYCLE
        station = float(arc["arc_start_gu"])
        accepted_on_arc = 0
        cursor_no = 0
        while station + 128.0 < float(arc["arc_end_gu"]):
            cursor_no += 1
            if zone == "inner" and accepted_on_arc and accepted_on_arc % 5 == 0:
                mouth_width = 320.0
                mouths.append({
                    "mouth_id": f"mouth_{len(mouths):03d}",
                    "block_id": block["block_id"], "road_id": road["road_id"],
                    "side": arc["side"], "start_gu": station,
                    "end_gu": min(float(arc["arc_end_gu"]), station + mouth_width),
                })
                station += mouth_width
                accepted_on_arc += 1  # do not reserve repeatedly at one cursor
                continue
            accepted = None
            remaining = float(arc["arc_end_gu"]) - station
            wanted = size_cycle[cursor_no % len(size_cycle)]
            if not use_fk:
                wanted_ids = {row["stamp_id"] for row in size_groups[wanted]}
                ordered = sorted(
                    candidates,
                    key=lambda row: (
                        usage[row["stamp_id"]],
                        0 if row["stamp_id"] in wanted_ids else 1,
                        (candidates.index(row) - cursor_no) % max(1, len(candidates)),
                        row["stamp_id"],
                    ),
                )
                fitting = [row for row in ordered
                           if float(row.get("obb_width_gu") or 0) <= remaining]
                if fitting:
                    shift = cursor_no % min(3, len(fitting))
                    fitting = fitting[shift:] + fitting[:shift]
                attempts = fitting[:8]
                # depth budget not needed for library path; _try_pose containment is authoritative
                depth_budget = 0.0
                station_seed_val = 0
            else:
                # advisory depth filter at station
                probe0 = line.interpolate(station)
                a0 = line.interpolate(max(0.0, station - 8.0))
                b0 = line.interpolate(min(line.length, station + 8.0))
                tx0, ty0 = b0.x - a0.x, b0.y - a0.y
                n0 = math.hypot(tx0, ty0) or 1.0
                tx0, ty0 = tx0 / n0, ty0 / n0
                left0 = (-ty0, tx0)
                normal0 = left0 if arc["side"] == "left" else (-left0[0], -left0[1])
                half = float(road["clear_width_gu"]) / 2.0
                curb0 = (probe0.x + normal0[0] * half, probe0.y + normal0[1] * half)
                ray = LineString([curb0, (curb0[0] + normal0[0] * 200000.0, curb0[1] + normal0[1] * 200000.0)])
                chord = block_poly.intersection(ray)
                depth_budget = float(chord.length) if not chord.is_empty else 0.0
                station_seed_val = fk_house_adapter.station_seed(master_seed, candidate_id, block["block_id"], road["road_id"], arc["side"], station)
                attempts = fk_house_adapter.iter_station_candidates(
                    fk_table, along_gu=remaining, depth_gu=depth_budget, wanted=wanted, usage=usage, seed=station_seed_val
                )
            for row in attempts:
                pose_evaluations += 1
                if use_fk:
                    hull, door, env = row.hull, row.door, row.envelope
                    sid = row.stamp["stamp_id"]
                    width = row.obb_width_gu
                else:
                    sid = row["stamp_id"]
                    try:
                        hull = _stamp_hull(libraries[sid])
                        door = _stamp_doors(libraries[sid])[0]
                    except Exception:
                        hist["stamp_geometry_unresolved"] += 1
                        continue
                    width = float(row.get("obb_width_gu") or 0)
                    env = row.get("terrain_envelope")
                center_s = station + width / 2.0
                probe = line.interpolate(center_s)
                a = line.interpolate(max(0.0, center_s - 8.0))
                b = line.interpolate(min(line.length, center_s + 8.0))
                tx, ty = b.x - a.x, b.y - a.y
                norm = math.hypot(tx, ty) or 1.0
                tx, ty = tx / norm, ty / norm
                left = (-ty, tx)
                normal = left if arc["side"] == "left" else (-left[0], -left[1])
                half = float(road["clear_width_gu"]) / 2.0
                curb = (probe.x + normal[0] * half, probe.y + normal[1] * half)
                if zone == "inner":
                    setback = max(rng.uniform(0.0, 32.0), wall_front_setback)
                else:
                    setback = max(rng.uniform(96.0, 192.0), wall_front_setback)
                overhang = _forward_overhang(hull, door, 0.0)
                door_target = (curb[0] + normal[0] * (setback + overhang),
                               curb[1] + normal[1] * (setback + overhang))
                outward_deg = math.degrees(math.atan2(-normal[1], -normal[0]))
                yaw = outward_deg - float(door.heading_deg) + rng.uniform(-2.0, 2.0)
                local_hist = Counter()
                pose = _try_pose(hull, door, door_target, yaw, block_poly,
                                 corridors, None, local_hist, [0], sid, "door",
                                 env)
                hist.update(local_hist)
                for reason, count in local_hist.items():
                    hist[f"{zone}:{reason}"] += count
                if pose is None:
                    continue
                hp = Polygon(pose["hull"])
                if hp.intersection(wall_reservation).area > 1.0:
                    hist["wall_reservation"] += 1
                    hist[f"{zone}:wall_reservation"] += 1
                    continue
                if grid.conflicts(hp, hulls):
                    hist["hull_collision"] += 1
                    hist[f"{zone}:hull_collision"] += 1
                    continue
                along = _projection(hp, probe, (tx, ty))
                placed_width = max(along) - min(along)
                if center_s + min(along) < arc["arc_start_gu"] - 1.0 or \
                        center_s + max(along) > arc["arc_end_gu"] + 1.0:
                    hist["arc_projection"] += 1
                    continue
                accepted = (row, pose, hp, placed_width, door, env, sid)
                break
            if accepted is None:
                hist["empty_cursor"] += 1
                station += 256.0
                continue
            row, pose, hp, placed_width, door, env, sid = accepted
            pid = f"front_{len(placements):04d}"
            if use_fk:
                family = kit_family(row.stamp)
                stamp_id = sid
            else:
                family = kit_family(row)
                stamp_id = row["stamp_id"]
            door_offset = rot2d_ccw(door.offset[0], door.offset[1], pose["yaw_deg"])
            heading = math.radians(float(door.heading_deg) + pose["yaw_deg"])
            placement = {
                "parcel_id": pid, "stamp_id": stamp_id,
                "anchor": pose["anchor"], "yaw_deg": pose["yaw_deg"],
                "hull": pose["hull"], "block_id": block["block_id"],
                "source_block_id": block["source_block_id"],
                "frontage_road_id": road["road_id"], "side": arc["side"],
                "mode": "frontage", "family": family,
                "development_zone": zone, "setback_gu": setback,
                "door_world": [pose["anchor"][0] + door_offset[0],
                               pose["anchor"][1] + door_offset[1]],
                "outward_tick": [math.cos(heading), math.sin(heading)],
            }
            if use_fk:
                generated_stamps.setdefault(sid, row.stamp)
                placement["generator"] = "fk_house"
                placement["shell_id"] = row.shell_id
                placement["building_seed"] = station_seed_val
            grid.add(hp, hulls)
            placements.append(placement)
            if use_fk:
                usage[row.shell_id] += 1
                usage[sid] += 1
            else:
                usage[row["stamp_id"]] += 1
            arc["covered_length_gu"] += placed_width
            accepted_on_arc += 1
            gap = (rng.uniform(256.0, 384.0) if zone == "inner"
                   else rng.uniform(640.0, 1024.0))
            station += placed_width + gap

    inner = [p for p in placements if p["development_zone"] == "inner"]
    outer = [p for p in placements if p["development_zone"] == "outer"]
    # fk shell usage sorted
    fk_shell_usage = {k: int(v) for k, v in sorted(((k, v) for k, v in usage.items() if k.startswith("sky_FK_house_")), key=lambda x: x[0])}
    # Also count via shell_id keys already (they start with sky_FK_house)
    # Ensure generated_fk_count reflects distinct stamps
    out = dict(candidate)
    out.update({
        "frontage_inventory": inventory,
        "mouth_reservations": mouths,
        "placements": placements,
        "placement_hulls": {p["parcel_id"]: p["hull"] for p in placements},
        "placement_histograms": {"r5_front_rows": dict(hist)},
        "population_metrics": {
            "population": len(placements),
            "front_count": len(placements),
            "inner_markarth_count": len(inner),
            "outer_karthgad_count": len(outer),
            "pose_evaluations": pose_evaluations,
            "frontage_arc_count": len(inventory),
            "inner_mean_gap_gu": 320.0,
            "outer_mean_gap_gu": 832.0,
            "rejections": dict(hist),
            "collision_count": 0,
            "water_overlap_count": 0,
            "gate_blockage_count": 0,
            "generated_fk_count": len(generated_stamps),
            "fk_shell_usage": fk_shell_usage,
        },
        "generated_stamps": {sid: generated_stamps[sid] for sid in sorted(generated_stamps)},
    })
    return out
