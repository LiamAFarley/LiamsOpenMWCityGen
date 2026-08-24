"""Build R12 circulation polygons from realized R11 centerlines and hulls."""
from __future__ import annotations

from typing import Any

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union

from .spatial_roles import REAR_APRON_DEPTH_GU, REAR_APRON_WIDTH_GU
from .validate import TownLayoutError


def _rings(geom):
    if geom.is_empty:
        return []
    parts = [geom] if geom.geom_type == "Polygon" else [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return [[[float(x), float(y)] for x, y in p.exterior.coords] for p in parts if p.area > 1.0]


def build_access_surfaces(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("stage_id") != "r11_alley_infill":
        raise TownLayoutError("access surfaces requires r11_alley_infill")
    hulls = [Polygon(p["hull"]) for p in source.get("placements", []) if p.get("hull")]
    hull_union = unary_union(hulls) if hulls else Polygon()
    surfaces = []
    for alley in source.get("alleys", []):
        line = LineString(alley["polyline"])
        raw = line.buffer(float(alley["clear_width_gu"]) / 2.0, cap_style=1, join_style=1)
        clipped = raw.difference(hull_union)
        if clipped.is_empty:
            alley["status"] = "rejected"
            alley["rejection_reason"] = "buffer fully occupied by building hulls"
            continue
        surfaces.append({"surface_id": alley["alley_id"], "role": "alley",
                         "surface_class": alley.get("surface_class", "settlement_dirt"),
                         "polygon": _rings(clipped), "centerline": alley["polyline"],
                         "contacts": {
                             "road_ids": list(alley.get("road_ids") or []),
                             "alley_ids": list(alley.get("parent_alley_ids") or []),
                         },
                         "building_ids": []})
    for role in source.get("spatial_roles", []):
        # An underfilled plaza/court can still own accepted buildings. Keep
        # its actual residual surface so those doors retain a real access
        # target; underfilled is a density diagnostic, not permission to drop
        # the geometry referenced by the placements.
        if (role.get("status") != "realized" and
                not (role.get("role") in ("plaza", "front_courtyard") and
                     role.get("accepted_building_ids"))):
            continue
        if role.get("role") == "alley_quarter":
            # Rear quarters are served by their explicit lane surfaces; the
            # residual domain itself is not a texture or an open-space object.
            continue
        kernel = Polygon(role["polygon"]).difference(hull_union)
        if kernel.is_empty or role.get("role") == "back_court":
            # Back courts remain semantic records, but their surface is the
            # connected service paths/aprons.  Never paint a whole residual
            # block as a brown courtyard fill.
            kernel = Polygon()
        surfaces.append({"surface_id": role["candidate_id"], "role": role["role"],
                         "surface_class": "public_packed_earth" if role["role"] == "plaza" else "settlement_dirt",
                         "polygon": _rings(kernel), "centerline": None,
                         "contacts": {"alley_ids": list(role.get("alley_ids") or [])},
                         "building_ids": role.get("accepted_building_ids", [])})
    aprons = []
    for door in source.get("doors", []):
        if door.get("role") != "secondary" or not door.get("service_alley_id"):
            continue
        # R5's current eligible stamp set has no secondary doors in the real
        # Falkreath checkpoint; retain this fail-closed path for future kits.
        import math
        x, y = door["position"]
        h = math.radians(door["outward_heading_deg"])
        pad = Polygon([(x, y), (x + math.cos(h)*REAR_APRON_DEPTH_GU + math.sin(h)*REAR_APRON_WIDTH_GU/2,
                                 y + math.sin(h)*REAR_APRON_DEPTH_GU - math.cos(h)*REAR_APRON_WIDTH_GU/2),
                       (x + math.cos(h)*REAR_APRON_DEPTH_GU - math.sin(h)*REAR_APRON_WIDTH_GU/2,
                        y + math.sin(h)*REAR_APRON_DEPTH_GU + math.cos(h)*REAR_APRON_WIDTH_GU/2)])
        pad = pad.difference(hull_union)
        if not pad.is_empty:
            aprons.append({"apron_id": f"apron:{door['door_id']}", "door_id": door["door_id"], "polygon": _rings(pad)})
    # New plaza/court/alley frontage gets a short explicit path from the door
    # to the surface it faces. This prevents a visual tick or graph declaration
    # from standing in for real walkable geometry.
    surface_geoms = {row["surface_id"]: unary_union([Polygon(ring) for ring in row["polygon"]])
                     for row in surfaces if row.get("polygon")}
    hull_by_id = {row["parcel_id"]: Polygon(row["hull"])
                  for row in source.get("placements", [])}
    door_aprons = []
    for placement in source.get("placements", []):
        if not placement["parcel_id"].startswith("infill_"):
            continue
        target_id = placement.get("access_target_id")
        target = surface_geoms.get(target_id)
        if target is None or target.is_empty:
            raise TownLayoutError(f"missing infill access surface {placement['parcel_id']} {target_id}")
        door = Point(placement["door_world"])
        contact = nearest_points(door, target)[1]
        centerline = LineString([door, contact])
        apron = centerline.buffer(48.0, cap_style=2, join_style=2)
        others = unary_union([hull for pid, hull in hull_by_id.items()
                              if pid != placement["parcel_id"]])
        if apron.intersection(others).area > 1.0:
            raise TownLayoutError(f"infill apron crosses building {placement['parcel_id']}")
        door_aprons.append({"apron_id": f"front_apron:{placement['parcel_id']}",
                            "placement_id": placement["parcel_id"],
                            "target_surface_id": target_id,
                            "width_gu": 96.0,
                            "centerline": [[door.x, door.y], [contact.x, contact.y]],
                            "polygon": _rings(apron)})
    out = dict(source)
    out.update({"stage_id": "r12_circulation_surfaces", "surfaces": surfaces,
                "rear_aprons": aprons, "door_aprons": door_aprons,
                "circulation_surfaces": surfaces,
                "surface_metrics": {"surface_count": len(surfaces), "alley_surface_count": sum(s["role"] == "alley" for s in surfaces),
                                     "plaza_surface_count": sum(s["role"] == "plaza" for s in surfaces),
                                     "front_court_surface_count": sum(s["role"] == "front_courtyard" for s in surfaces),
                                     "back_court_surface_count": sum(s["role"] == "back_court" for s in surfaces),
                                     "rear_apron_count": len(aprons),
                                     "door_apron_count": len(door_aprons),
                                     "surface_area_gu2": sum(sum(Polygon(r).area for r in s["polygon"]) for s in surfaces)}})
    return out
