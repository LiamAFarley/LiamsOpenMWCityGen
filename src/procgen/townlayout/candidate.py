"""Canonical MacroLayoutCandidate contract for V2 townlayout (Phase 5).

Purpose
-------
Every morphology generator (organic first, later grid/radial/mixed) must
emit the same intermediate product: city domain, patch polygons, shared
boundary topology, and reports.  Downstream stages must not care which
morphology produced the patches.

Inputs
------
A dict with ``city_domain``, ``patches``, ``boundary_edges``, ``nodes``.

Outputs
-------
``validate_macro_layout`` returns ``(document, issues)``.  Fatal geometry
or topology defects raise ``TownLayoutError``.

Pipeline position
-----------------
V2 townlayout Phase 5 interface; no generation of its own.
"""

from __future__ import annotations

from typing import Any

from procgen.cityplan import ring_area

from .constants import MIN_PATCH_AREA_GU2, VERTEX_EPS_GU
from .geometry import polygon_from_ring
from .schema import ITEM_SPEC, check_structure, issue, json_path
from .validate import TownLayoutError

OVERLAP_AREA_EPS = 1.0

_PATCH_SPEC = {"type": "object", "keys": ITEM_SPEC["patch"]}
_EDGE_SPEC = {"type": "object", "keys": ITEM_SPEC["boundary_edge"]}
_NODE_SPEC = {"type": "object", "keys": ITEM_SPEC["road_node"]}


def validate_macro_layout(document: dict) -> tuple[dict, list[dict]]:
    """Structural + planar checks for a MacroLayoutCandidate."""
    if not isinstance(document, dict):
        raise TownLayoutError("wrong_type: MacroLayoutCandidate must be an object")
    issues: list[dict] = []
    for key in ("candidate_id", "city_domain", "patches",
                "boundary_edges", "nodes"):
        if key not in document:
            raise TownLayoutError(f"missing_key: {key}")

    domain = document["city_domain"]
    try:
        polygon_from_ring(domain)
    except TownLayoutError as exc:
        raise TownLayoutError(f"invalid_polygon: city_domain {exc}") from exc

    patches = document["patches"]
    edges = document["boundary_edges"]
    nodes = document["nodes"]
    if not isinstance(patches, list) or len(patches) < 3:
        raise TownLayoutError("invalid_polygon: need at least 3 patches")

    patch_ids: list[str] = []
    polys = []
    for i, patch in enumerate(patches):
        path = json_path("patches", f"[{i}]")
        check_structure(patch, _PATCH_SPEC, path, issues)
        if issues:
            raise TownLayoutError(issues[0]["message"])
        pid = str(patch["patch_id"])
        if pid in patch_ids:
            raise TownLayoutError(f"duplicate_id: {pid}")
        patch_ids.append(pid)
        poly = polygon_from_ring(patch["polygon"])
        if poly.area + 1e-6 < MIN_PATCH_AREA_GU2:
            raise TownLayoutError(
                f"invalid_polygon: {pid} area {poly.area} below MIN_PATCH_AREA_GU2")
        if abs(ring_area(patch["polygon"])) <= VERTEX_EPS_GU:
            raise TownLayoutError(f"invalid_polygon: {pid} zero area")
        polys.append(poly)

    for i, a in enumerate(polys):
        for j, b in enumerate(polys):
            if j <= i:
                continue
            overlap = a.intersection(b).area
            if overlap > OVERLAP_AREA_EPS:
                raise TownLayoutError(
                    f"invalid_polygon: overlap {overlap} between "
                    f"{patch_ids[i]} and {patch_ids[j]}")

    node_ids = []
    for i, node in enumerate(nodes):
        path = json_path("nodes", f"[{i}]")
        check_structure(node, _NODE_SPEC, path, issues)
        if issues:
            raise TownLayoutError(issues[0]["message"])
        nid = str(node["node_id"])
        if nid in node_ids:
            raise TownLayoutError(f"duplicate_id: {nid}")
        node_ids.append(nid)
    node_set = set(node_ids)

    adj: dict[str, set[str]] = {pid: set() for pid in patch_ids}
    incident: dict[str, int] = {pid: 0 for pid in patch_ids}
    edge_ids: list[str] = []
    for i, edge in enumerate(edges):
        path = json_path("boundary_edges", f"[{i}]")
        check_structure(edge, _EDGE_SPEC, path, issues)
        if issues:
            raise TownLayoutError(issues[0]["message"])
        eid = str(edge["edge_id"])
        if eid in edge_ids:
            raise TownLayoutError(f"duplicate_id: {eid}")
        edge_ids.append(eid)
        if edge["a_node"] not in node_set or edge["b_node"] not in node_set:
            raise TownLayoutError(f"missing_ref: {eid} node")
        left, right = edge.get("patch_left"), edge.get("patch_right")
        if left:
            if left not in adj:
                raise TownLayoutError(f"missing_ref: {eid} patch_left {left}")
            incident[left] += 1
        if right:
            if right not in adj:
                raise TownLayoutError(f"missing_ref: {eid} patch_right {right}")
            incident[right] += 1
        if left and right:
            adj[left].add(right)
            adj[right].add(left)

    for patch in patches:
        pid = patch["patch_id"]
        declared = set(patch["neighbour_patch_ids"])
        if declared != adj[pid]:
            raise TownLayoutError(
                f"topology: {pid} neighbour_patch_ids != boundary_edges")
        if incident[pid] < 1:
            raise TownLayoutError(f"topology: {pid} has no usable boundary edge")

    return document, issues


def require_macro_layout(document: dict) -> dict:
    """Validate and return the candidate, or raise TownLayoutError."""
    document, _issues = validate_macro_layout(document)
    return document
