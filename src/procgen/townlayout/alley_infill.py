"""Design connected, branching rear circulation before seating dense infill.

The two civic sectors retain their plaza/courtyard vocabulary. Every other
substantial inner rear pocket becomes an alley quarter with a curved through
trunk, side branches where the pocket is broad enough, and compact buildings
whose selected doors face the path that actually serves them.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, polylabel, unary_union

from procgen.cityplan import rot2d_ccw
from procgen import fk_house
from procgen.frontage_fit import _stamp_doors, _stamp_hull, _transform_hull
from .place import _try_pose
from .populate import _forward_overhang
from .constants import ROUTE_CONNECTOR_GU, ROUTE_REACH_GU
from .spatial_roles import (ALLEY_CLEAR_WIDTH_GU, BUILDING_CLEARANCE_GU,
                            MAX_NEW_INNER_PLACEMENTS, PLAZA_MOUTH_WIDTH_GU)
from .stamp_index import DEFAULT_LIBRARIES, load_stamp_libraries
from .validate import TownLayoutError
from procgen.visual_planner_eligibility import build_eligibility_policy

BACK_ALLEY_WIDTH_GU = 128.0


def _is_hut_stamp(stamp: dict[str, Any]) -> bool:
    """Return the explicit hut class excluded from city placement."""

    return "hut" in str(stamp.get("stamp_id") or "").lower().replace("-", "_").split("_")


def _line_points(line: LineString) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in line.coords]


def _chaikin(points, iterations=2):
    rows = [(float(x), float(y)) for x, y in points]
    for _ in range(iterations):
        refined = [rows[0]]
        for a, b in zip(rows, rows[1:]):
            refined.extend(((0.75*a[0] + 0.25*b[0], 0.75*a[1] + 0.25*b[1]),
                            (0.25*a[0] + 0.75*b[0], 0.25*a[1] + 0.75*b[1])))
        refined.append(rows[-1])
        rows = refined
    return rows


def _curved_arm(start: Point, target: Point, free: Polygon, width: float,
                handedness: int) -> LineString:
    dx, dy = target.x - start.x, target.y - start.y
    length = math.hypot(dx, dy)
    if length <= 1.0:
        return LineString([start, target])
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    # Mouth endpoints lie on the free polygon boundary. Validate the shaped
    # interior after moving one alley half-width inward; the short endpoint
    # connectors were already proven clear when the mouths were discovered.
    inset = min(width / 2.0 + 24.0, length * 0.16)
    inner_start = Point(start.x + ux * inset, start.y + uy * inset)
    inner_target = Point(target.x - ux * inset, target.y - uy * inset)
    mid = Point((inner_start.x + inner_target.x) / 2.0,
                (inner_start.y + inner_target.y) / 2.0)
    for bend in (min(448.0, length * 0.18), min(256.0, length * 0.11), 0.0):
        control = Point(mid.x + nx * bend * handedness,
                        mid.y + ny * bend * handedness)
        points = _chaikin([inner_start.coords[0], control.coords[0],
                           inner_target.coords[0]])
        core = LineString(points)
        # Validate the target connector as well as the bent core. The old
        # check accepted a safe shortened core and then appended the final
        # segment to target without checking it, allowing routes to cross
        # buildings at the far end.
        checked = LineString([inner_start.coords[0], *core.coords[1:],
                              target.coords[0]])
        if free.buffer(2.0).covers(checked.buffer(width / 2.0,
                                                  cap_style=2,
                                                  join_style=2)):
            full = LineString([start.coords[0], *core.coords, target.coords[0]])
            if free.buffer(2.0).covers(full):
                return full
    return _grid_route(start, target, free, width)


def _grid_route(start: Point, target: Point, free: Polygon,
                width: float) -> LineString:
    """Bounded any-direction fallback through the alley-width-eroded polygon."""
    safe = free.buffer(-(width / 2.0 + 4.0))
    parts = [safe] if safe.geom_type == "Polygon" else [
        g for g in getattr(safe, "geoms", []) if g.geom_type == "Polygon"]
    viable = [part for part in parts
              if part.distance(start) < ROUTE_REACH_GU and part.distance(target) < ROUTE_REACH_GU]
    if not viable:
        raise TownLayoutError("no alley-width channel inside sector")
    safe = min(viable, key=lambda part: part.distance(start) + part.distance(target))
    a = nearest_points(start, safe)[1]
    b = nearest_points(target, safe)[1]
    # ``free`` may be the expanded safe-lobe route polygon rather than the
    # whole sector polygon. The mouth-side connector is validated by mouth
    # discovery; this local polygon can validate the target-side connector.
    target_connector = LineString([target, b])
    if (target_connector.length > ROUTE_CONNECTOR_GU or
            not free.buffer(2.0).covers(target_connector)):
        raise TownLayoutError("no free-ground connector to alley channel")
    step = 128.0
    minx, miny, maxx, maxy = safe.bounds
    graph = nx.Graph()
    coords = {}
    ix0, iy0 = math.floor(minx / step), math.floor(miny / step)
    ix1, iy1 = math.ceil(maxx / step), math.ceil(maxy / step)
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            point = Point(ix * step, iy * step)
            if safe.covers(point):
                node = (ix, iy)
                graph.add_node(node)
                coords[node] = point
    for node, point in coords.items():
        ix, iy = node
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            other = (ix + dx, iy + dy)
            if other in coords:
                segment = LineString([point, coords[other]])
                if safe.covers(segment):
                    graph.add_edge(node, other, weight=segment.length)
    for special, point in (("start", a), ("target", b)):
        graph.add_node(special)
        nearest = sorted(coords, key=lambda node: point.distance(coords[node]))[:16]
        for node in nearest:
            segment = LineString([point, coords[node]])
            if segment.length <= 512.0 and safe.covers(segment):
                graph.add_edge(special, node, weight=segment.length)
    try:
        nodes = nx.shortest_path(graph, "start", "target", weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise TownLayoutError("no bounded alley route inside sector") from exc
    points = [a.coords[0], *(coords[node].coords[0] for node in nodes[1:-1]),
              b.coords[0]]
    # Remove grid stair-steps whenever a later node is directly visible.
    simplified = [points[0]]
    index = 0
    while index < len(points) - 1:
        next_index = len(points) - 1
        while next_index > index + 1 and not safe.covers(
                LineString([points[index], points[next_index]])):
            next_index -= 1
        simplified.append(points[next_index])
        index = next_index
    smoothed = _chaikin(simplified, iterations=1) if len(simplified) > 2 else simplified
    core = LineString(smoothed)
    if not free.buffer(2.0).covers(core.buffer(width / 2.0, cap_style=2,
                                               join_style=2)):
        core = LineString(simplified)
    full = LineString([start.coords[0], *core.coords, target.coords[0]])
    if not free.buffer(2.0).covers(full):
        raise TownLayoutError("grid route endpoint connector crosses free boundary")
    return full


def _branch_targets(free: Polygon, trunk: LineString, width: float,
                    count: int) -> list[Point]:
    """Choose separated deep-pocket points for real side branches."""
    safe = free.buffer(-(width / 2.0 + 32.0))
    parts = [safe] if safe.geom_type == "Polygon" else [
        row for row in getattr(safe, "geoms", []) if row.geom_type == "Polygon"]
    if not parts:
        return []
    candidates = []
    step = 256.0
    for part in parts:
        minx, miny, maxx, maxy = part.bounds
        x = math.ceil(minx / step) * step
        while x <= maxx:
            y = math.ceil(miny / step) * step
            while y <= maxy:
                point = Point(x, y)
                distance = point.distance(trunk)
                anchor = nearest_points(point, trunk)[1]
                end_clearance = min(anchor.distance(Point(trunk.coords[0])),
                                    anchor.distance(Point(trunk.coords[-1])))
                if (part.covers(point) and distance >= 640.0 and
                        end_clearance >= 384.0):
                    candidates.append((distance + 0.25 * point.distance(part.boundary),
                                       point))
                y += step
            x += step
    selected = []
    for _score, point in sorted(candidates, key=lambda row: (-row[0],
                                                              row[1].x,
                                                              row[1].y)):
        if all(point.distance(other) >= 1280.0 for other in selected):
            selected.append(point)
            if len(selected) >= count:
                break
    return selected


def _deep_visible_arm(start: Point, safe: Polygon, free: Polygon,
                      width: float) -> LineString:
    """Fallback to the deepest directly visible point in one safe pocket."""
    candidates = [polylabel(safe, tolerance=24.0)]
    minx, miny, maxx, maxy = safe.bounds
    step = 192.0
    x = math.ceil(minx / step) * step
    while x <= maxx:
        y = math.ceil(miny / step) * step
        while y <= maxy:
            point = Point(x, y)
            if safe.covers(point):
                candidates.append(point)
            y += step
        x += step
    for point in sorted(candidates, key=lambda row: -row.distance(start)):
        line = LineString([start, point])
        if free.buffer(2.0).covers(line.buffer(width / 2.0,
                                                cap_style=2, join_style=2)):
            return line
    raise TownLayoutError("no visible rear-alley arm")


def _sector_free_polygon(sector: dict[str, Any]) -> Polygon:
    """Restore serialized building-clearance holes in a rear pocket."""
    return Polygon(sector["free_polygon"], sector.get("free_holes") or [])


def _sector_routes(sector: dict) -> list[dict]:
    free = _sector_free_polygon(sector)
    center = Point(sector["center"])
    kernel = Polygon(sector["polygon"]) if sector.get("polygon") else Polygon()
    rows = []
    if sector["role"] == "alley_quarter":
        width = BACK_ALLEY_WIDTH_GU
        mouths = sector["mouths"]
        safe = free.buffer(-(width / 2.0 + 4.0))
        safe_parts = [safe] if safe.geom_type == "Polygon" else [
            part for part in getattr(safe, "geoms", [])
            if part.geom_type == "Polygon" and part.area >= 256.0 ** 2]
        if not safe_parts:
            raise TownLayoutError(f"{sector['sector_id']} has no rear-alley channel")
        assigned = {}
        for mouth in mouths:
            point = Point(mouth["inside_point"])
            part_index = min(range(len(safe_parts)),
                             key=lambda index: point.distance(safe_parts[index]))
            if point.distance(safe_parts[part_index]) <= 768.0:
                assigned.setdefault(part_index, []).append(mouth)
        if not assigned:
            raise TownLayoutError(f"{sector['sector_id']} has no reachable rear pocket")
        trunk_records = []
        for trunk_index, (part_index, part_mouths) in enumerate(sorted(assigned.items())):
            part = safe_parts[part_index]
            route_free = part.buffer(width / 2.0 + 6.0).intersection(free)
            route_centers = [polylabel(part, tolerance=24.0),
                             part.representative_point()]
            if part.covers(part.centroid):
                route_centers.append(part.centroid)
            try:
                def make_arm(start: Point, handedness: int) -> LineString:
                    for route_center in route_centers:
                        try:
                            return _curved_arm(start, route_center, free, width, handedness)
                        except TownLayoutError:
                            continue
                    return _deep_visible_arm(start, part, free, width)

                # A changed footprint can make one mouth's approach unusable
                # while another mouth reaches the same safe lobe.
                first = None
                arm_a = None
                for mouth in part_mouths:
                    try:
                        candidate_arm = make_arm(
                            Point(mouth["inside_point"]),
                            1 if trunk_index % 2 == 0 else -1)
                    except TownLayoutError:
                        continue
                    first, arm_a = mouth, candidate_arm
                    break
                if first is None or arm_a is None:
                    raise TownLayoutError("no reachable mouth for safe lobe")
                if len(part_mouths) > 1:
                    second = next((mouth for mouth in part_mouths if mouth is not first), None)
                    if second is None:
                        raise TownLayoutError("no second mouth")
                    inside_b = Point(second["inside_point"])
                    join = Point(arm_a.coords[-1])
                    try:
                        arm_b = _curved_arm(
                            join, inside_b, free, width,
                            -1 if trunk_index % 2 == 0 else 1)
                    except TownLayoutError:
                        arm_b = _grid_route(join, inside_b, free, width)
                    joined = LineString([*arm_a.coords,
                                         *list(arm_b.coords)[1:]])
                    if free.buffer(2.0).covers(joined.buffer(
                            width / 2.0, cap_style=2, join_style=2)):
                        interior = joined
                        full = LineString([first["road_point"], *interior.coords,
                                           second["road_point"]])
                        road_ids = [first["road_id"], second["road_id"]]
                    else:
                        interior = arm_a
                        full = LineString([first["road_point"], *interior.coords])
                        road_ids = [first["road_id"]]
                else:
                    interior = arm_a
                    full = LineString([first["road_point"], *interior.coords])
                    road_ids = [first["road_id"]]
            except TownLayoutError as exc:
                # Another eroded lobe may be separated from every road mouth
                # by an accepted frontage hull. Keep routing the other lobes;
                # the sector is rejected below only if none is reachable.
                continue
            trunk_id = f"alley:{sector['sector_id']}:trunk_{trunk_index:02d}"
            rows.append({"alley_id": trunk_id, "role": "quarter_trunk",
                         "source_role_id": sector["sector_id"],
                         "block_id": sector["block_id"], "road_ids": road_ids,
                         "parent_alley_ids": [], "clear_width_gu": width,
                         "polyline": _line_points(full),
                         "interior_polyline": _line_points(interior),
                         "surface_class": "settlement_dirt", "status": "accepted"})
            trunk_records.append((trunk_id, interior, route_free, part.area))
        if not trunk_records:
            # A changed frontage footprint can close every safe connector in
            # a rear pocket. Keep the sector empty rather than emitting a
            # route through a building; the caller records it as
            # circulation-only for later fill policy work.
            return rows
        branch_serial = 0
        total_branch_count = max(1, min(3, int(
            sector["free_area_gu2"] // 4_000_000.0)))
        for trunk_id, interior, route_free, part_area in trunk_records:
            share = max(1, round(total_branch_count * part_area /
                                 sum(row[3] for row in trunk_records)))
            for target in _branch_targets(route_free, interior, width, share):
                anchor = nearest_points(target, interior)[1]
                try:
                    branch = _curved_arm(anchor, target, route_free, width,
                                         1 if branch_serial % 2 == 0 else -1)
                except TownLayoutError:
                    continue
                rows.append({"alley_id": f"alley:{sector['sector_id']}:branch_{branch_serial:02d}",
                             "role": "quarter_branch",
                             "source_role_id": sector["sector_id"],
                             "block_id": sector["block_id"], "road_ids": [],
                             "parent_alley_ids": [trunk_id],
                             "clear_width_gu": width,
                             "polyline": _line_points(branch),
                             "interior_polyline": _line_points(branch),
                             "surface_class": "settlement_dirt", "status": "accepted"})
                branch_serial += 1
        return rows
    if sector["role"] == "mews_alley":
        if not sector["mouths"]:
            raise TownLayoutError(f"sector {sector['sector_id']} has no routable mouth")
        mouth = sector["mouths"][0]
        inside = Point(mouth["inside_point"])
        try:
            interior = _curved_arm(inside, center, free, ALLEY_CLEAR_WIDTH_GU, 1)
        except TownLayoutError as exc:
            raise TownLayoutError(f"{sector['sector_id']} {exc}") from exc
        full = LineString([mouth["road_point"], *interior.coords])
        rows.append({"alley_id": f"alley:{sector['sector_id']}:mews",
                     "role": "mews_alley", "source_role_id": sector["sector_id"],
                     "block_id": sector["block_id"], "road_ids": [mouth["road_id"]],
                     "clear_width_gu": ALLEY_CLEAR_WIDTH_GU,
                     "polyline": _line_points(full), "interior_polyline": _line_points(interior),
                     "surface_class": "settlement_dirt", "status": "accepted"})
        return rows
    if sector["role"] == "through_alley":
        a, b = sector["mouths"][:2]
        ia, ib = Point(a["inside_point"]), Point(b["inside_point"])
        try:
            arm_a = _curved_arm(ia, center, free, ALLEY_CLEAR_WIDTH_GU, 1)
            arm_b = _curved_arm(center, ib, free, ALLEY_CLEAR_WIDTH_GU, -1)
        except TownLayoutError as exc:
            raise TownLayoutError(f"{sector['sector_id']} {exc}") from exc
        interior = LineString(list(arm_a.coords) + list(arm_b.coords)[1:])
        full = LineString([a["road_point"], *interior.coords, b["road_point"]])
        rows.append({"alley_id": f"alley:{sector['sector_id']}:through",
                     "role": "through_alley", "source_role_id": sector["sector_id"],
                     "block_id": sector["block_id"],
                     "road_ids": [a["road_id"], b["road_id"]],
                     "clear_width_gu": ALLEY_CLEAR_WIDTH_GU,
                     "polyline": _line_points(full), "interior_polyline": _line_points(interior),
                     "surface_class": "settlement_dirt", "status": "accepted"})
        return rows
    for index, mouth in enumerate(sector["mouths"]):
        inside = Point(mouth["inside_point"])
        target = nearest_points(inside, kernel.boundary)[1]
        # The mouth deliberately overlaps the open-space kernel at its end;
        # validate against the sector's full free ground, not a kernel-cut hole.
        try:
            interior = _curved_arm(inside, target, free,
                                   PLAZA_MOUTH_WIDTH_GU if sector["role"] == "plaza"
                                   else ALLEY_CLEAR_WIDTH_GU,
                                   1 if index % 2 == 0 else -1)
        except TownLayoutError as exc:
            raise TownLayoutError(f"{sector['sector_id']} mouth {index} {exc}") from exc
        full = LineString([mouth["road_point"], *interior.coords])
        rows.append({"alley_id": f"alley:{sector['sector_id']}:{index:02d}",
                     "role": "plaza_mouth" if sector["role"] == "plaza"
                     else "court_alley", "source_role_id": sector["sector_id"],
                     "block_id": sector["block_id"], "road_ids": [mouth["road_id"]],
                     "clear_width_gu": PLAZA_MOUTH_WIDTH_GU if sector["role"] == "plaza"
                     else ALLEY_CLEAR_WIDTH_GU,
                     "polyline": _line_points(full), "interior_polyline": _line_points(interior),
                     "surface_class": "public_packed_earth" if sector["role"] == "plaza"
                     else "settlement_dirt", "status": "accepted"})
    return rows


def _normal(line: LineString, station: float) -> tuple[float, float]:
    a = line.interpolate(max(0.0, station - 8.0))
    b = line.interpolate(min(line.length, station + 8.0))
    dx, dy = b.x - a.x, b.y - a.y
    length = math.hypot(dx, dy) or 1.0
    return -dy / length, dx / length


def _append_doors(doors: list[dict], placement: dict, stamp: dict,
                  primary_id: str, service_alley_id: str | None = None) -> None:
    for door in _stamp_doors(stamp):
        dx, dy = rot2d_ccw(door.offset[0], door.offset[1], placement["yaw_deg"])
        row = {"door_id": f"{placement['parcel_id']}:{door.door_id}",
                      "placement_id": placement["parcel_id"],
                      "source_door_id": door.door_id,
                      "position": [placement["anchor"][0] + dx,
                                   placement["anchor"][1] + dy],
                      "outward_heading_deg": (door.heading_deg + placement["yaw_deg"]) % 360.0,
                      "role": "primary" if door.door_id == primary_id else "secondary",
                      "source": "stamp_library"}
        if row["role"] == "secondary" and service_alley_id:
            row["service_alley_id"] = service_alley_id
        doors.append(row)


def _add_generated_rear_doors(
    placements: list[dict],
    doors: list[dict],
    generated_stamps: dict[str, dict],
    alleys: list[dict],
    occupied: list[Polygon],
) -> int:
    """Attach one reachable secondary door to eligible generated front houses.

    The primary authored facade remains the frontage-facing front. A secondary
    facade is used only when its door can face an interior alley and a short
    apron reaches that alley without crossing another building.
    """
    alley_lines = [
        (row["alley_id"], LineString(row["interior_polyline"]))
        for row in alleys
    ]
    variants: dict[tuple[str, str], dict] = {}
    upgraded = 0
    for placement in placements:
        base_id = str(placement.get("stamp_id") or "")
        base_stamp = generated_stamps.get(base_id)
        if base_stamp is None:
            continue
        shell_id = str((base_stamp.get("source") or {}).get("shell_id") or "")
        if not shell_id:
            continue
        base_doors = _stamp_doors(base_stamp)
        if not base_doors:
            continue
        primary = base_doors[0]
        primary_offset = rot2d_ccw(primary.offset[0], primary.offset[1],
                                    placement["yaw_deg"])
        primary_pos = Point(placement["anchor"][0] + primary_offset[0],
                            placement["anchor"][1] + primary_offset[1])
        primary_heading = math.radians(primary.heading_deg + placement["yaw_deg"])
        primary_dir = (math.cos(primary_heading), math.sin(primary_heading))
        candidates = []
        for facade in fk_house.fk_secondary_door_facades(shell_id):
            key = (shell_id, facade)
            variant = variants.get(key)
            if variant is None:
                variant_id = f"fkgen__{shell_id}__rear_{facade}"
                source = base_stamp.get("source") or {}
                options = {
                    "window_facades": source.get("window_facades"),
                    "window_model": source.get("window_model"),
                    "door_model": source.get("door_model"),
                    "door_frame": source.get("door_frame"),
                    "porch_facades": source.get("porch_facades") or None,
                    "porch_model": source.get("porch_model") or None,
                    "stair_facades": source.get("stair_facades") or None,
                    "stair_model": source.get("stair_model") or None,
                }
                variant = fk_house.generate_fk_house(
                    shell_id,
                    generated_id=variant_id,
                    secondary_door_facades=[facade],
                    **options,
                )
                variants[key] = variant
                generated_stamps[variant_id] = variant
            secondary = next(
                (door for door in _stamp_doors(variant)
                 if door.door_id != primary.door_id),
                None,
            )
            if secondary is None:
                continue
            offset = rot2d_ccw(secondary.offset[0], secondary.offset[1],
                               placement["yaw_deg"])
            door_pos = Point(placement["anchor"][0] + offset[0],
                             placement["anchor"][1] + offset[1])
            heading = math.radians(secondary.heading_deg + placement["yaw_deg"])
            door_dir = (math.cos(heading), math.sin(heading))
            for alley_id, line in alley_lines:
                contact = line.interpolate(line.project(door_pos))
                vx, vy = contact.x - door_pos.x, contact.y - door_pos.y
                distance = math.hypot(vx, vy)
                if not 96.0 <= distance <= 900.0:
                    continue
                outward_dot = (vx * door_dir[0] + vy * door_dir[1]) / distance
                behind = ((contact.x - primary_pos.x) * primary_dir[0] +
                          (contact.y - primary_pos.y) * primary_dir[1]) < 0.0
                if outward_dot < 0.5 or not behind:
                    continue
                apron = LineString([door_pos, contact]).buffer(
                    48.0, cap_style=2, join_style=2)
                if any(apron.intersection(other).area > 1.0
                       for other in occupied
                       if other is not occupied[placements.index(placement)]):
                    continue
                candidates.append((distance, facade, alley_id, variant, door_pos,
                                   apron))
        if not candidates:
            continue
        _distance, facade, alley_id, variant, _door_pos, apron = min(
            candidates,
            key=lambda row: (row[0], row[1], row[2]),
        )
        variant_hull = Polygon(_transform_hull(
            _stamp_hull(variant), placement["anchor"], placement["yaw_deg"]))
        if any(variant_hull.intersection(other).area > 1.0
               for other in occupied
               if other is not occupied[placements.index(placement)]):
            continue
        placement_index = placements.index(placement)
        placement["stamp_id"] = variant["stamp_id"]
        placement["hull"] = [list(point) for point in variant_hull.exterior.coords]
        placement["rear_door_facade"] = facade
        placement["rear_access_alley_id"] = alley_id
        occupied[placement_index] = variant_hull
        doors[:] = [row for row in doors
                    if row.get("placement_id") != placement["parcel_id"]]
        _append_doors(doors, placement, variant, primary.door_id,
                      service_alley_id=alley_id)
        upgraded += 1
    return upgraded


def _add_generated_accessories(
    placements: list[dict],
    generated_stamps: dict[str, dict],
    occupied: list[Polygon],
    blocks: dict[str, dict],
    alley_reserve,
) -> int:
    """Upgrade a deterministic subset of fitted FK shells when space allows."""
    upgraded = 0
    for index, placement in enumerate(placements):
        base_stamp = generated_stamps.get(str(placement.get("stamp_id") or ""))
        if base_stamp is None:
            continue
        source = base_stamp.get("source") or {}
        shell_id = str(source.get("shell_id") or "")
        if not shell_id or placement.get("accessory") in {"porch", "stairs"}:
            continue
        seed = int(placement.get("building_seed") or 0)
        accessory = "porch" if seed % 5 == 0 else "stairs" if seed % 5 == 1 else "base"
        if accessory == "base":
            continue
        primary = tuple(source.get("door_facades") or
                        fk_house.SHELL_SPECS[shell_id]["default_door_facades"][:1])
        if accessory == "porch":
            options = {"porch_facades": primary,
                       "porch_model": "sky_FK_Porch_01a"}
        else:
            options = {"stair_facades": primary,
                       "stair_model": "sky_ex_mk_str_02"}
        variant_id = f"fkgen__{shell_id}__{accessory}"
        variant = generated_stamps.get(variant_id)
        if variant is None:
            variant = fk_house.generate_fk_house(
                shell_id, generated_id=variant_id, **options)
            generated_stamps[variant_id] = variant
        hull = Polygon(_transform_hull(
            _stamp_hull(variant), placement["anchor"], placement["yaw_deg"]))
        block = Polygon(blocks[placement["block_id"]]["polygon"]).buffer(-2.0)
        if hull.difference(block).area > 1.0:
            continue
        if hull.intersection(alley_reserve).area > 1.0:
            continue
        if any(hull.intersection(other).area > 1.0
               for other_index, other in enumerate(occupied)
               if other_index != index):
            continue
        placement["stamp_id"] = variant_id
        placement["hull"] = [list(point) for point in hull.exterior.coords]
        placement["accessory"] = accessory
        occupied[index] = hull
        upgraded += 1
    return upgraded


def realize_alley_infill(source: dict[str, Any],
                         development_policy: dict[str, Any] | None = None) -> dict[str, Any]:
    if source.get("stage_id") != "r10_spatial_roles":
        raise TownLayoutError("alley infill requires r10_spatial_roles")
    libraries = load_stamp_libraries(DEFAULT_LIBRARIES)
    palette = Path("output/settlement-splits/markarth-side-v2/"
                   "final-markarth-extraction-2026-08-10-library/"
                   "stamp_palette_v1/catalog.json")
    policy = build_eligibility_policy(DEFAULT_LIBRARIES, palette_path=palette)
    # Inner infill stamps come from the zone's preferred library
    # (development_policy.inner.preferred_library); default markarth preserves
    # the Falkreath stone-core behavior.
    inner_library = str(((development_policy or {}).get("inner") or {})
                        .get("preferred_library") or "markarth")
    generated_stamps = dict(source.get("generated_stamps") or {})
    for base_stamp in list(generated_stamps.values()):
        source_row = base_stamp.get("source") or {}
        shell_id = str(source_row.get("shell_id") or "")
        if not shell_id or not str(base_stamp.get("stamp_id") or "").endswith("__base"):
            continue
        primary = tuple(fk_house.SHELL_SPECS[shell_id]["default_door_facades"])
        for accessory, options in (
            ("porch", {"porch_facades": primary,
                       "porch_model": "sky_FK_Porch_01a"}),
            ("stairs", {"stair_facades": primary,
                         "stair_model": "sky_ex_mk_str_02"}),
        ):
            variant_id = f"fkgen__{shell_id}__{accessory}"
            generated_stamps.setdefault(
                variant_id,
                fk_house.generate_fk_house(shell_id,
                                           generated_id=variant_id,
                                           **options),
            )
    accepted_ids = [sid for sid in policy.accepted_stamp_ids
                    if inner_library in sid and sid in libraries]
    stamps = [libraries[sid] for sid in accepted_ids
              if not _is_hut_stamp(libraries[sid])
              and str(libraries[sid].get("building_type") or "")
               not in {"farm", "stable", "mill", "keep", "barracks"}]
    generated_candidates = [
        stamp for stamp in generated_stamps.values()
        if not _is_hut_stamp(stamp)
    ]
    stamps.extend(generated_candidates)
    stamps = list({stamp["stamp_id"]: stamp for stamp in stamps}.values())
    stamps.sort(key=lambda row: (Polygon(_stamp_hull(row)).area, row["stamp_id"]))
    if not stamps:
        raise TownLayoutError("no accepted inner stamps for infill")

    blocks = {row["block_id"]: row for row in source["buildable_blocks"]}
    road_reserve = unary_union([
        LineString(row["polyline"]).buffer(row["clear_width_gu"] / 2.0,
                                            cap_style=2, join_style=2)
        for row in source["roads"]])
    mesh_rings = source["wall"].get("mesh_reserve") or []
    if not mesh_rings:
        raise TownLayoutError("alley infill missing serialized wall mesh reserve")
    wall_band = unary_union([Polygon(ring) for ring in mesh_rings])
    water = unary_union([Polygon(ring) for ring in source.get("water_polygons") or []])
    placements = list(source["placements"])
    doors = list(source["doors"])
    occupied = [Polygon(row["hull"]) for row in placements]
    alleys = []
    route_collision_rejections = 0
    wall_collision_rejections = 0
    for sector in source["spatial_roles"]:
        routes = []
        try:
            candidate_routes = _sector_routes(sector)
        except TownLayoutError as exc:
            if (sector["role"] in ("plaza", "front_courtyard") and
                    "alley-width channel" in str(exc)):
                sector["status"] = "circulation_only"
                candidate_routes = []
            else:
                raise
        for route in candidate_routes:
            route_buffer = LineString(route["polyline"]).buffer(
                float(route["clear_width_gu"]) / 2.0,
                cap_style=1,
                join_style=1,
            )
            if route_buffer.intersection(wall_band).area > 1.0:
                wall_collision_rejections += 1
                continue
            if any(route_buffer.intersection(hull).area > 1.0
                   for hull in occupied):
                route_collision_rejections += 1
                continue
            routes.append(route)
        route_ids = {row["alley_id"] for row in routes}
        routes = [
            row for row in routes
            if all(parent in route_ids for parent in row.get("parent_alley_ids") or [])
        ]
        alleys.extend(routes)
        sector["alley_ids"] = [row["alley_id"] for row in routes]
    alley_reserve = unary_union([
        LineString(row["polyline"]).buffer(row["clear_width_gu"] / 2.0,
                                            cap_style=1, join_style=1)
        for row in alleys])
    accessory_count = _add_generated_accessories(
        placements, generated_stamps, occupied, blocks, alley_reserve)
    rear_door_count = _add_generated_rear_doors(
        placements, doors, generated_stamps, alleys, occupied)
    kernels = unary_union([Polygon(row["polygon"]) for row in source["spatial_roles"]
                           if row.get("polygon")])
    # _try_pose accepts a corridor collection and unions it during each exact
    # pose check. These exclusions are immutable for the whole R11 pass, so
    # collapse them once instead of rebuilding the same complex union for
    # every rear-house candidate.
    corridors = [unary_union([road_reserve, wall_band, water,
                              alley_reserve, kernels])]
    hist = Counter()
    usage = Counter()
    new_ids = []
    access_reserves = []

    def try_station(sector, target, outward, access_id, serial,
                    direct_door=None):
        nonlocal placements, occupied
        block_row = blocks[sector["block_id"]]
        block = Polygon(block_row["polygon"]).buffer(-2.0)
        desired = math.degrees(math.atan2(outward[1], outward[0]))
        order = sorted(
            enumerate(stamps),
            key=lambda item: (
                usage[item[1]["stamp_id"]],
                (item[0] - serial) % max(1, len(stamps)),
                item[1]["stamp_id"],
            ),
        )
        for _stamp_index, stamp in order:
            stamp_doors = _stamp_doors(stamp)
            if not stamp_doors:
                continue
            hull = _stamp_hull(stamp)
            pose = primary = door_target = None
            chosen_setback = None
            lateral_axis = (-outward[1], outward[0])
            setbacks = ((64.0, 128.0, 192.0)
                        if sector["role"] == "alley_quarter" and direct_door is None
                        else (192.0,))
            laterals = ((0.0, -128.0, 128.0)
                        if sector["role"] == "alley_quarter" and direct_door is None
                        else (0.0,))
            for candidate_door in stamp_doors:
                yaw = desired - candidate_door.heading_deg
                overhang = _forward_overhang(hull, candidate_door, 0.0)
                for setback in setbacks:
                    for lateral in laterals:
                        trial_target = ((float(direct_door.x), float(direct_door.y))
                                        if direct_door is not None else (
                            target[0] - outward[0] * (setback + overhang) +
                            lateral_axis[0] * lateral,
                            target[1] - outward[1] * (setback + overhang) +
                            lateral_axis[1] * lateral,
                        ))
                        local = Counter()
                        trial = _try_pose(
                            hull, candidate_door, trial_target, yaw, block,
                            corridors, None, local, [0], stamp["stamp_id"],
                            "door", stamp.get("terrain_envelope"))
                        hist.update(local)
                        if trial is not None:
                            pose = trial
                            primary = candidate_door
                            door_target = trial_target
                            chosen_setback = (float(direct_door.distance(
                                Point(target))) if direct_door is not None
                                else setback)
                            break
                    if pose is not None:
                        break
                if pose is not None:
                    break
            if pose is None or primary is None or door_target is None:
                continue
            hp = Polygon(pose["hull"])
            if hp.intersection(wall_band).area > 1.0:
                hist["wall_reservation"] += 1
                continue
            if any(hp.buffer(BUILDING_CLEARANCE_GU).intersects(other)
                   for other in occupied):
                hist["hull_collision"] += 1
                continue
            if hp.intersection(alley_reserve).area > 1.0:
                hist["hull_in_alley"] += 1
                continue
            candidate_apron = LineString([
                door_target, (float(target[0]), float(target[1]))
            ]).buffer(48.0, cap_style=2, join_style=2)
            if any(candidate_apron.intersection(other).area > 1.0
                   for other in occupied):
                hist["access_building_collision"] += 1
                continue
            if any(candidate_apron.intersection(apron).area > 1.0
                   for apron in access_reserves):
                hist["access_collision"] += 1
                continue
            pid = f"infill_{len(new_ids):03d}"
            placement = {"parcel_id": pid, "stamp_id": stamp["stamp_id"],
                         "anchor": pose["anchor"], "yaw_deg": pose["yaw_deg"],
                         "hull": pose["hull"], "block_id": sector["block_id"],
                         "source_block_id": block_row.get("source_block_id"),
                         "frontage_road_id": access_id, "access_target_id": access_id,
                         "side": "infill", "mode": f"{sector['role']}_infill",
                          "family": ("wood" if stamp["stamp_id"].startswith("fkgen__")
                                     else "stone"),
                          "development_zone": "inner",
                         "setback_gu": chosen_setback,
                         "door_world": [float(door_target[0]), float(door_target[1])],
                         "outward_tick": [float(outward[0]), float(outward[1])],
                          "infill_role": sector["role"]}
            if stamp["stamp_id"].startswith("fkgen__"):
                placement["generator"] = "fk_house"
                placement["shell_id"] = (stamp.get("source") or {}).get("shell_id")
            placements.append(placement)
            usage[stamp["stamp_id"]] += 1
            occupied.append(hp)
            access_reserves.append(candidate_apron)
            new_ids.append(pid)
            _append_doors(doors, placement, stamp, primary.door_id)
            return pid
        return None

    for sector in source["spatial_roles"]:
        made = []
        court_facing = []
        alley_facing = []
        if (sector["role"] in ("plaza", "front_courtyard") and
                not any(row["source_role_id"] == sector["sector_id"]
                        for row in alleys)):
            # A measured wall footprint can remove the only safe approach to
            # an open-space kernel. Do not seed buildings into an unconnected
            # court; leave it as circulation-only rather than creating
            # unreachable primary doors.
            sector["status"] = "circulation_only"
            continue
        if sector["role"] in ("plaza", "front_courtyard"):
            center = Point(sector["center"])
            kernel = Polygon(sector["polygon"])
            count = 4 if sector["role"] == "plaza" else 3
            # Probe a full perimeter but stop as soon as a coherent minimum
            # frontage exists; rejected stations never leave orphan buildings.
            for serial in range(12):
                if len(made) >= count:
                    break
                angle = 2.0 * math.pi * (serial + 0.35) / 12.0
                outward_radial = (math.cos(angle), math.sin(angle))
                edge = (center.x + outward_radial[0] * sector["kernel_radius_gu"],
                        center.y + outward_radial[1] * sector["kernel_radius_gu"])
                # The door faces inward, toward the open plaza/court.
                pid = try_station(sector, edge,
                                  (-outward_radial[0], -outward_radial[1]),
                                  sector["sector_id"], serial)
                if pid:
                    made.append(pid)
                    court_facing.append(pid)
            # The curved approach is also a real frontage. Add a restrained
            # pair of buildings where its middle has space, always facing the
            # alley rather than the open-space center.
            route = next((row for row in alleys
                          if row["source_role_id"] == sector["sector_id"]), None)
            if route is not None:
                line = LineString(route["interior_polyline"])
                for serial, fraction in enumerate((0.34, 0.58)):
                    station = line.length * fraction
                    q = line.interpolate(station)
                    nx, ny = _normal(line, station)
                    side = 1.0 if serial % 2 == 0 else -1.0
                    normal = (nx * side, ny * side)
                    target = (q.x + normal[0] * route["clear_width_gu"] / 2.0,
                              q.y + normal[1] * route["clear_width_gu"] / 2.0)
                    pid = try_station(sector, target, (-normal[0], -normal[1]),
                                      route["alley_id"], serial + 20)
                    if pid:
                        made.append(pid)
                        alley_facing.append(pid)
        else:
            routes = [row for row in alleys
                      if row["source_role_id"] == sector["sector_id"]]
            target_count = max(6, min(15,
                               int(sector["free_area_gu2"] / 600_000.0)))
            serial = 0
            # Interleave trunks and branches so one path cannot consume all
            # available stamps before the deeper limbs receive frontage.
            stations = []
            for route in routes:
                line = LineString(route["interior_polyline"])
                station = 256.0
                while station < line.length - 160.0:
                    stations.append((station / max(1.0, line.length),
                                     route, line, station))
                    station += 512.0
            stations.sort(key=lambda row: (row[0], row[1]["role"],
                                           row[1]["alley_id"]))
            for _fraction, route, line, station in stations:
                if len(made) >= target_count:
                    break
                q = line.interpolate(station)
                nx, ny = _normal(line, station)
                for side in (1.0, -1.0):
                    if len(made) >= target_count:
                        break
                    normal = (nx * side, ny * side)
                    target = (q.x + normal[0] * route["clear_width_gu"] / 2.0,
                              q.y + normal[1] * route["clear_width_gu"] / 2.0)
                    pid = try_station(sector, target,
                                      (-normal[0], -normal[1]),
                                      route["alley_id"], serial)
                    if pid:
                        made.append(pid)
                        alley_facing.append(pid)
                    serial += 1
            # Station frontage alone misses usable concave pockets. Fill the
            # remaining capacity from a deterministic rear-ground lattice;
            # every accepted door still faces and receives a direct apron to
            # its nearest designed alley.
            if len(made) < target_count and routes:
                free = _sector_free_polygon(sector)
                route_lines = [(route, LineString(route["interior_polyline"]))
                               for route in routes]
                minx, miny, maxx, maxy = free.bounds
                step = 384.0
                samples = []
                x = math.ceil(minx / step) * step
                while x <= maxx:
                    y = math.ceil(miny / step) * step
                    while y <= maxy:
                        point = Point(x, y)
                        if free.covers(point):
                            route, line = min(route_lines,
                                key=lambda row: point.distance(row[1]))
                            distance = point.distance(line)
                            if 176.0 <= distance <= 1152.0:
                                samples.append((abs(distance - 448.0), x, y,
                                                point, route, line))
                        y += step
                    x += step
                for _score, _x, _y, point, route, line in sorted(samples):
                    if len(made) >= target_count:
                        break
                    contact = nearest_points(point, line)[1]
                    length = point.distance(contact)
                    if length <= 1.0:
                        continue
                    apron = LineString([point, contact]).buffer(
                        48.0, cap_style=2, join_style=2)
                    if any(apron.intersection(hull).area > 1.0
                           for hull in occupied):
                        continue
                    outward = ((contact.x - point.x) / length,
                                (contact.y - point.y) / length)
                    pid = try_station(
                        sector, (contact.x, contact.y), outward,
                        route["alley_id"], serial, direct_door=point)
                    serial += 1
                    if pid:
                        made.append(pid)
                        alley_facing.append(pid)
        sector["accepted_building_ids"] = made
        sector["court_facing_building_ids"] = court_facing
        sector["alley_facing_building_ids"] = alley_facing
        sector["realized_building_count"] = len(made)
        if sector["role"] in ("plaza", "front_courtyard"):
            sector["status"] = ("realized" if len(court_facing) >= 3
                                else "underfilled")
        else:
            sector["status"] = "realized" if made else "circulation_only"
    if len(new_ids) > MAX_NEW_INNER_PLACEMENTS:
        raise TownLayoutError("infill safety ceiling exceeded")
    out = dict(source)
    out.update({"stage_id": "r11_alley_infill", "placements": placements,
                "placement_hulls": {row["parcel_id"]: row["hull"] for row in placements},
                "doors": sorted(doors, key=lambda row: row["door_id"]),
                "alleys": alleys, "realized_roles": source["spatial_roles"],
                "generated_stamps": generated_stamps,
                "alley_infill_metrics": {
                    "inherited_placement_count": len(source["placements"]),
                    "new_inner_placement_count": len(new_ids),
                    "alley_count": len(alleys),
                    "route_collision_rejections": route_collision_rejections,
                    "wall_collision_rejections": wall_collision_rejections,
                    "generated_rear_door_count": rear_door_count,
                    "generated_accessory_count": accessory_count,
                    "coherent_sector_count": sum(row["status"] == "realized"
                                                 for row in source["spatial_roles"]),
                    "inhabited_quarter_count": sum(
                        row["role"] == "alley_quarter" and
                        row["status"] == "realized"
                        for row in source["spatial_roles"]),
                    "circulation_only_quarter_count": sum(
                        row["role"] == "alley_quarter" and
                        row["status"] == "circulation_only"
                        for row in source["spatial_roles"]),
                    "rejections": dict(hist)}})
    return out
