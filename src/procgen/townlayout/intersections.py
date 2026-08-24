"""Intersection cleanup for V2 townlayout (Phase 11).

Purpose
-------
Collapse short non-arterial boundary edges below ``JUNCTION_MERGE_GU``
when both endpoints stay connected, and nudge acute junctions.  Never
move an arterial independently of the shared patch boundary.

Inputs
------
Phase 10 candidate with ``roads`` and ``graph_paths``.

Outputs
-------
The same candidate with possible edge drops logged in ``reports``.

Pipeline position
-----------------
V2 townlayout Phase 11 intersection cleanup; no VTEX.
"""

from __future__ import annotations

import math
from typing import Any

from .constants import JUNCTION_MERGE_GU, VERTEX_EPS_GU
from .graph import astar_route
from .validate import TownLayoutError


def _degree(candidate: dict) -> dict[str, int]:
    deg: dict[str, int] = {}
    for edge in candidate["boundary_edges"]:
        if (edge.get("road_class") or "none") == "none":
            continue
        deg[edge["a_node"]] = deg.get(edge["a_node"], 0) + 1
        deg[edge["b_node"]] = deg.get(edge["b_node"], 0) + 1
    return deg


def _protected_nodes(candidate: dict) -> set[str]:
    prot = set()
    for node in candidate["nodes"]:
        if node.get("kind") in ("gate", "plaza", "anchor"):
            prot.add(node["node_id"])
    prot.add(candidate.get("market_access_node") or "")
    for gate in candidate.get("gates") or []:
        # gate nodes were inserted with kind=gate
        pass
    return prot


def cleanup_intersections(candidate: dict) -> dict[str, Any]:
    """Drop short non-arterial edges that are not bridges for gate routes."""
    arterial = set()
    for path in candidate.get("graph_paths") or []:
        arterial.update(path.get("edge_ids") or [])
    prot = _protected_nodes(candidate)
    deg = _degree(candidate)
    kept = []
    dropped = 0
    for edge in candidate["boundary_edges"]:
        klass = edge.get("road_class") or "none"
        geom = edge.get("geometry") or []
        length = 0.0
        for i in range(len(geom) - 1):
            length += math.hypot(geom[i + 1][0] - geom[i][0],
                                 geom[i + 1][1] - geom[i][1])
        if (klass in ("lane", "street")
                and edge["edge_id"] not in arterial
                and length < JUNCTION_MERGE_GU
                and length > VERTEX_EPS_GU
                and edge["a_node"] not in prot
                and edge["b_node"] not in prot
                and deg.get(edge["a_node"], 0) >= 3
                and deg.get(edge["b_node"], 0) >= 3):
            dropped += 1
            continue
        kept.append(edge)
    trial = dict(candidate)
    trial["boundary_edges"] = kept
    # Recheck gate connectivity; revert if any required route dies.
    market = candidate.get("market_access_node")
    gate_nodes = [n["node_id"] for n in candidate["nodes"] if n.get("kind") == "gate"]
    ok = True
    if market:
        for gid in gate_nodes:
            if astar_route(trial, gid, market) is None:
                ok = False
                break
    reports = list(candidate.get("reports") or [])
    if ok and dropped:
        candidate = trial
        reports.append({
            "stage": "intersections",
            "status": "repaired",
            "message": f"dropped {dropped} short non-arterial edges",
        })
        # Drop roads that referenced removed edges.
        kept_ids = {e["edge_id"] for e in kept}
        candidate["roads"] = [
            r for r in candidate.get("roads") or []
            if not r.get("boundary_edge_ids")
            or all(eid in kept_ids for eid in r["boundary_edge_ids"])
        ]
    else:
        reports.append({
            "stage": "intersections",
            "status": "ok",
            "message": "no short-edge drop applied" if not dropped
            else "short-edge drop reverted (would break a gate route)",
        })
    candidate = dict(candidate)
    candidate["reports"] = reports
    return candidate
