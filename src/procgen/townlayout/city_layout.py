"""Derive final parcels, door aprons, and exact access reachability for R8."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import nearest_points, unary_union, voronoi_diagram

from .geometry import normalize_ring, polygon_from_ring
from .validate import TownLayoutError

DOOR_APRON_HALF_GU = 32.0


def build_final_city_layout(source: dict[str, Any]) -> dict[str, Any]:
    """R13 rebuild from final doors/surfaces; legacy R8 remains callable.

    The final graph follows the rendered circulation exactly: regional ports
    enter the arterial graph, alleys touch their source roads, plazas/courts
    touch their alleys, and each new primary door reaches its named surface
    through the apron that is actually drawn.  Inherited frontage doors retain
    their accepted direct road contact.
    """
    if source.get("stage_id") != "r12_circulation_surfaces":
        raise TownLayoutError("final city layout requires r12_circulation_surfaces")
    placements = list(source.get("placements") or [])
    roads = {r["road_id"]: r for r in source.get("roads", [])}
    road_lines = {rid: LineString(row["polyline"]) for rid, row in roads.items()}
    graph = nx.Graph()
    nodes, edges = [], []

    def add_node(node_id: str, kind: str, **data: Any) -> None:
        if node_id not in graph:
            graph.add_node(node_id, kind=kind)
            nodes.append({"node_id": node_id, "kind": kind, **data})

    def add_edge(a: str, b: str, kind: str) -> None:
        if a not in graph or b not in graph:
            raise TownLayoutError(f"R13 access edge references missing node {a} -> {b}")
        graph.add_edge(a, b, kind=kind)
        edges.append({"a": a, "b": b, "kind": kind})

    for rid in sorted(roads):
        add_node(f"road:{rid}", "road", road_id=rid)
    road_ids = sorted(roads)
    for index, a in enumerate(road_ids):
        for b in road_ids[index + 1:]:
            if road_lines[a].distance(road_lines[b]) <= 1.0:
                add_edge(f"road:{a}", f"road:{b}", "exact_road_contact")

    port_nodes = []
    arterial_ids = [rid for rid, row in roads.items()
                    if row.get("hierarchy") == "arterial"]
    if not arterial_ids:
        raise TownLayoutError("R13 has no arterial roads for regional ports")
    for port in source.get("ports") or []:
        position = Point(port["position"])
        road_id = min(arterial_ids, key=lambda rid: position.distance(road_lines[rid]))
        if position.distance(road_lines[road_id]) > 2.0:
            raise TownLayoutError(f"R13 port is not on an arterial {port['port_id']}")
        node_id = f"port:{port['port_id']}"
        add_node(node_id, "regional_port", position=port["position"])
        add_edge(node_id, f"road:{road_id}", "port_road_contact")
        port_nodes.append(node_id)
    if not port_nodes:
        raise TownLayoutError("R13 has no regional ports")

    surface_ids = {row["surface_id"] for row in source.get("surfaces", [])}
    for surface in source.get("surfaces", []):
        sid = surface["surface_id"]
        add_node(f"surface:{sid}", surface["role"], surface_id=sid)
    for surface in source.get("surfaces", []):
        sid = surface["surface_id"]
        contacts = surface.get("contacts", {})
        for road_id in contacts.get("road_ids", []):
            if road_id not in roads:
                raise TownLayoutError(f"R13 surface references missing road {sid}: {road_id}")
            add_edge(f"surface:{sid}", f"road:{road_id}", "exact_surface_road_contact")
        for alley_id in contacts.get("alley_ids", []):
            if alley_id not in surface_ids:
                raise TownLayoutError(f"R13 surface references missing alley {sid}: {alley_id}")
            add_edge(f"surface:{sid}", f"surface:{alley_id}", "exact_surface_alley_contact")

    apron_by_placement = {}
    for apron in source.get("door_aprons", []):
        pid = apron["placement_id"]
        target_id = apron["target_surface_id"]
        if target_id not in surface_ids:
            raise TownLayoutError(f"R13 apron references missing surface {pid}: {target_id}")
        node_id = f"apron:{apron['apron_id']}"
        add_node(node_id, "door_apron", apron_id=apron["apron_id"], placement_id=pid)
        add_edge(node_id, f"surface:{target_id}", "exact_apron_surface_contact")
        apron_by_placement[pid] = node_id

    doors_by_placement = {}
    primary_door_nodes = []
    for door in source.get("doors", []):
        doors_by_placement.setdefault(door["placement_id"], []).append(door)
        node_id = f"door:{door['door_id']}"
        add_node(node_id, "door", position=door["position"],
                 placement_id=door["placement_id"], role=door["role"])
        if door["role"] == "primary":
            primary_door_nodes.append(node_id)
    parcels = []
    for p in placements:
        hull = Polygon(p["hull"])
        parcels.append({"parcel_id": p["parcel_id"], "block_id": p["block_id"], "stamp_id": p["stamp_id"],
                        "development_zone": p["development_zone"], "required_occupancy": True,
                        "polygon": [[float(x), float(y)] for x, y in hull.exterior.coords],
                        "polygons": [[[float(x), float(y)] for x, y in hull.exterior.coords]],
                        "ownership_polygons": [[[float(x), float(y)] for x, y in hull.exterior.coords]],
                        "approved_overlap_with": [], "access_apron_id": None})
        pid = p["parcel_id"]
        for door in doors_by_placement.get(pid, []):
            dn = f"door:{door['door_id']}"
            if door["role"] != "primary":
                continue
            if pid in apron_by_placement:
                add_edge(dn, apron_by_placement[pid], "exact_primary_door_apron_contact")
            elif p.get("frontage_road_id") in roads:
                add_edge(dn, f"road:{p['frontage_road_id']}", "inherited_primary_door_road_contact")
            else:
                raise TownLayoutError(f"R13 primary door has no rendered access {pid}")

    unreachable = [node_id for node_id in primary_door_nodes
                   if not any(nx.has_path(graph, node_id, port) for port in port_nodes)]
    if unreachable:
        raise TownLayoutError(f"R13 unreachable primary doors {unreachable[:8]}")
    metrics = {"population": len(placements), "inherited_population": sum(not p["parcel_id"].startswith("infill_") for p in placements),
               "new_inner_population": sum(p["parcel_id"].startswith("infill_") for p in placements),
               "parcel_count": len(parcels), "door_count": len(source.get("doors", [])),
               "primary_door_count": sum(d["role"] == "primary" for d in source.get("doors", [])),
               "secondary_door_count": sum(d["role"] == "secondary" for d in source.get("doors", [])),
               "plaza_count": sum(s["role"] == "plaza" for s in source.get("surfaces", [])),
               "front_court_count": sum(s["role"] == "front_courtyard" for s in source.get("surfaces", [])),
               "back_court_count": sum(s["role"] == "back_court" for s in source.get("surfaces", [])),
               "alley_count": sum(s["role"] == "alley" for s in source.get("surfaces", [])),
               "door_apron_count": len(source.get("door_aprons", [])),
               "unreachable_primary_door_count": len(unreachable),
               "service_connected_secondary_door_count": sum(
                   d["role"] == "secondary" and bool(d.get("service_alley_id"))
                   for d in source.get("doors", []))}
    out = dict(source)
    out.update({"stage_id": "r13_city_layout", "parcels": parcels,
                "access_graph": {"nodes": nodes, "edges": edges},
                "city_layout_metrics": metrics, "final_frontage_coverage": source.get("frontage_inventory", [])})
    return out


def _polygon_parts(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [g for g in getattr(geometry, "geoms", [])
            if g.geom_type == "Polygon" and g.area > 1.0]


def _rings(geometry) -> list[list[list[float]]]:
    return [normalize_ring([[float(x), float(y)] for x, y in poly.exterior.coords])["ring"]
            for poly in _polygon_parts(geometry)]


def _interval_union(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = None
    for lo, hi in sorted(intervals):
        if end is None:
            start, end = lo, hi
        elif lo > end:
            total += end - start
            start, end = lo, hi
        else:
            end = max(end, hi)
    return total + ((end - start) if end is not None else 0.0)


def build_city_layout(source: dict[str, Any], *, minimum: int,
                      maximum: int) -> dict[str, Any]:
    if source.get("stage_id") != "r7_circulation":
        raise TownLayoutError("city layout requires r7_circulation")
    placements = list(source.get("placements") or [])
    if not minimum <= len(placements) <= maximum:
        raise TownLayoutError(
            f"R8 population outside brief {len(placements)} not in [{minimum},{maximum}]")
    roads = {row["road_id"]: row for row in source["roads"]}
    road_lines = {rid: LineString(row["polyline"]) for rid, row in roads.items()}
    hulls = {row["parcel_id"]: Polygon(row["hull"]) for row in placements}

    aprons = []
    apron_geoms = {}
    for placement in placements:
        pid = placement["parcel_id"]
        road_id = placement["frontage_road_id"]
        if road_id not in road_lines or not placement.get("door_world"):
            raise TownLayoutError(f"missing door access authority {pid}")
        door = Point(placement["door_world"])
        road_point = nearest_points(door, road_lines[road_id])[1]
        center = LineString([(door.x, door.y), (road_point.x, road_point.y)])
        apron = center.buffer(DOOR_APRON_HALF_GU, cap_style=2, join_style=2)
        other_hulls = unary_union([h for other, h in hulls.items() if other != pid])
        if apron.intersection(other_hulls).area > 1.0:
            raise TownLayoutError(f"door apron intersects another building {pid}")
        apron_geoms[pid] = apron
        aprons.append({
            "apron_id": f"apron_{pid}", "placement_id": pid,
            "road_id": road_id, "width_gu": DOOR_APRON_HALF_GU * 2.0,
            "centerline": [[float(door.x), float(door.y)],
                           [float(road_point.x), float(road_point.y)]],
            "polygons": _rings(apron),
        })

    placements_by_block = defaultdict(list)
    for placement in placements:
        placements_by_block[placement["block_id"]].append(placement)
    spaces_by_block = defaultdict(list)
    for space in source.get("open_spaces") or []:
        if space.get("block_id") and space.get("polygon"):
            spaces_by_block[space["block_id"]].append(
                polygon_from_ring(space["polygon"]))
    parcels = []
    for block in source["buildable_blocks"]:
        members = placements_by_block.get(block["block_id"], [])
        if not members:
            continue
        block_poly = polygon_from_ring(block["polygon"])
        reserved = unary_union(spaces_by_block.get(block["block_id"], [])) \
            if spaces_by_block.get(block["block_id"]) else Polygon()
        parcelable = block_poly.difference(reserved)
        centroids = [hulls[p["parcel_id"]].centroid for p in members]
        if len(centroids) == 1:
            cells = [parcelable]
        else:
            diagram = voronoi_diagram(MultiPoint(centroids), envelope=block_poly)
            raw_cells = _polygon_parts(diagram)
            cells = []
            for seed in centroids:
                cell = min(raw_cells, key=lambda poly: seed.distance(poly))
                cells.append(cell.intersection(parcelable))
        for placement, ownership in zip(members, cells):
            pid = placement["parcel_id"]
            parcel_geom = unary_union([ownership, hulls[pid], apron_geoms[pid]])
            rings = _rings(parcel_geom)
            if not rings:
                raise TownLayoutError(f"empty derived parcel {pid}")
            parcels.append({
                "parcel_id": pid, "block_id": block["block_id"],
                "development_zone": placement["development_zone"],
                "stamp_id": placement["stamp_id"],
                "required_occupancy": True,
                "polygon": rings[0], "polygons": rings,
                "ownership_polygons": _rings(ownership),
                "approved_overlap_with": [],
                "access_apron_id": f"apron_{pid}",
            })

    graph = nx.Graph()
    graph_nodes = []
    graph_edges = []

    def add_node(node_id: str, kind: str, **data) -> None:
        graph.add_node(node_id, kind=kind)
        graph_nodes.append({"node_id": node_id, "kind": kind, **data})

    def add_edge(a: str, b: str, kind: str) -> None:
        graph.add_edge(a, b, kind=kind)
        graph_edges.append({"a": a, "b": b, "kind": kind})

    for road_id in sorted(roads):
        add_node(f"road:{road_id}", "road", road_id=road_id)
    road_ids = sorted(roads)
    for i, a in enumerate(road_ids):
        for b in road_ids[i + 1:]:
            if road_lines[a].distance(road_lines[b]) <= 1.0:
                add_edge(f"road:{a}", f"road:{b}", "exact_road_contact")
    port_nodes = []
    arterial_ids = [rid for rid, row in roads.items() if row["hierarchy"] == "arterial"]
    for port in source.get("ports") or []:
        pos = Point(port["position"])
        road_id = min(arterial_ids, key=lambda rid: pos.distance(road_lines[rid]))
        if pos.distance(road_lines[road_id]) > 2.0:
            raise TownLayoutError(f"port is not on arterial {port['port_id']}")
        node_id = f"port:{port['port_id']}"
        add_node(node_id, "regional_port", position=port["position"])
        add_edge(node_id, f"road:{road_id}", "port_road_contact")
        port_nodes.append(node_id)
    mouths = {row["mouth_id"]: row for row in source.get("access_mouths") or []}
    for alley in source.get("alleys") or []:
        alley_node = f"alley:{alley['alley_id']}"
        add_node(alley_node, "pedestrian_alley", alley_id=alley["alley_id"])
        road_id = mouths[alley["mouth_id"]]["road_id"]
        add_edge(alley_node, f"road:{road_id}", "alley_road_contact")
        court_node = f"court:{alley['courtyard_id']}"
        add_node(court_node, "courtyard", courtyard_id=alley["courtyard_id"])
        add_edge(court_node, alley_node, "court_alley_contact")
    unreachable = []
    for placement in placements:
        node_id = f"door:{placement['parcel_id']}"
        add_node(node_id, "door", placement_id=placement["parcel_id"],
                 position=placement["door_world"])
        add_edge(node_id, f"road:{placement['frontage_road_id']}",
                 "door_apron_contact")
        if not any(nx.has_path(graph, node_id, port) for port in port_nodes):
            unreachable.append(placement["parcel_id"])
    if unreachable:
        raise TownLayoutError(f"R8 unreachable placements {unreachable[:8]}")

    coverage = []
    for arc in source.get("frontage_inventory") or []:
        line = road_lines[arc["road_id"]]
        intervals = []
        for placement in placements:
            if (placement["block_id"] != arc["block_id"] or
                    placement["frontage_road_id"] != arc["road_id"] or
                    placement["side"] != arc["side"]):
                continue
            values = [line.project(Point(x, y)) for x, y in hulls[placement["parcel_id"]].exterior.coords]
            lo = max(float(arc["arc_start_gu"]), min(values))
            hi = min(float(arc["arc_end_gu"]), max(values))
            if hi > lo:
                intervals.append((lo, hi))
        covered = _interval_union(intervals)
        coverage.append({**arc, "covered_length_gu": covered,
                         "coverage_pct": 100.0 * covered / max(1.0, arc["usable_length_gu"])})
    blocks_with_buildings = set(p["block_id"] for p in placements)
    inner_blocks = [b for b in source["buildable_blocks"]
                    if b["development_zone"] == "inner"]
    inner_arcs = [a for a in coverage if a["development_zone"] == "inner"]
    outer_arcs = [a for a in coverage if a["development_zone"] == "outer"]
    out = dict(source)
    out.update({
        "stage_id": "r8_city_layout",
        "door_aprons": aprons,
        "parcels": parcels,
        "access_graph": {"nodes": graph_nodes, "edges": graph_edges},
        "final_frontage_coverage": coverage,
        "city_layout_metrics": {
            "population": len(placements),
            "parcel_count": len(parcels),
            "door_apron_count": len(aprons),
            "unreachable_placement_count": 0,
            "inner_block_occupancy_pct": 100.0 * sum(
                b["block_id"] in blocks_with_buildings for b in inner_blocks) /
                max(1, len(inner_blocks)),
            "inner_frontage_coverage_pct": 100.0 * sum(a["covered_length_gu"] for a in inner_arcs) /
                max(1.0, sum(a["usable_length_gu"] for a in inner_arcs)),
            "outer_frontage_coverage_pct": 100.0 * sum(a["covered_length_gu"] for a in outer_arcs) /
                max(1.0, sum(a["usable_length_gu"] for a in outer_arcs)),
            "visible_hull_count": len(placements),
        },
    })
    return out
