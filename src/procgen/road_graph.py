"""Convert the repaired raster skeleton into a topology-preserving graph.

Pipeline position
------------------
This module consumes the repaired corridor and the bridge mask, skeletonizes
the corridor exactly once for the final graph, then emits nodes and maximal
edge chains::

    repaired corridor + bridge provenance
        -> final 8-neighbour skeleton
        -> clustered junction/end nodes
        -> one trace for every maximal chain (including closed loops)
        -> graph/skeleton validation

Adjacent non-degree-two pixels are clustered so a thick junction is one graph
node rather than a fan of duplicate nodes.  Pure degree-two components are
closed loops; each receives a deterministic synthetic anchor node and a
self-loop edge.  Every skeleton pixel is covered by either a node cluster or
an edge chain, and every edge is incident on valid node IDs.  The module uses
no randomness and serializes coordinates as ``[x, y]`` even though NumPy
arrays are indexed ``[y, x]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .road_repair import component_labels, component_rows


Coord = tuple[int, int]  # x, y


@dataclass
class SkeletonGraph:
    """Final skeleton, graph records, and validation evidence."""

    skeleton: np.ndarray
    component_labels: np.ndarray
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    components: list[dict[str, Any]]
    validation: dict[str, Any]


_OFFSETS: tuple[Coord, ...] = tuple(
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dx, dy) != (0, 0)
)


def _neighbours(point: Coord, occupied: set[Coord]) -> list[Coord]:
    """Return sorted 8-neighbours of ``point`` that are in ``occupied``."""

    x, y = point
    return sorted(
        [(x + dx, y + dy) for dx, dy in _OFFSETS if (x + dx, y + dy) in occupied],
        key=lambda item: (item[1], item[0]),
    )


def _canonical_segment(first: Coord, second: Coord) -> tuple[Coord, Coord]:
    """Normalize an undirected skeleton-pixel segment."""

    return tuple(sorted((first, second), key=lambda item: (item[1], item[0])))  # type: ignore[return-value]


def _cluster_pixels(pixels: set[Coord]) -> list[set[Coord]]:
    """Cluster a set of pixels with 8-neighbour connectivity."""

    remaining = set(pixels)
    clusters: list[set[Coord]] = []
    while remaining:
        start = min(remaining, key=lambda item: (item[1], item[0]))
        remaining.remove(start)
        stack = [start]
        cluster = {start}
        while stack:
            current = stack.pop()
            for neighbour in _neighbours(current, remaining):
                remaining.remove(neighbour)
                cluster.add(neighbour)
                stack.append(neighbour)
        clusters.append(cluster)
    return sorted(clusters, key=lambda cluster: min((p[1], p[0]) for p in cluster))


def _stable_node_id(component_id: str, members: Iterable[Coord]) -> str:
    """Derive a node ID from its content, independent of scan order."""

    basis = component_id + "|" + ";".join(
        f"{x},{y}" for x, y in sorted(members, key=lambda item: (item[1], item[0]))
    )
    return f"road_node_{hashlib.sha256(basis.encode('ascii')).hexdigest()[:16]}"


def _stable_edge_id(component_id: str, from_id: str, to_id: str, chain: Sequence[Coord]) -> str:
    """Derive an edge ID from normalized endpoints and raw chain content."""

    basis = component_id + "|" + from_id + "|" + to_id + "|" + ";".join(
        f"{x},{y}" for x, y in chain
    )
    return f"road_edge_{hashlib.sha256(basis.encode('ascii')).hexdigest()[:16]}"


def _orient_chain(chain: list[Coord], from_id: str, to_id: str) -> tuple[list[Coord], str, str]:
    """Give an edge a stable direction, including deterministic loop direction."""

    reverse = list(reversed(chain))
    if from_id > to_id:
        return reverse, to_id, from_id
    if from_id < to_id:
        return chain, from_id, to_id
    # A self-loop has the same endpoint ID.  Choose the lexicographically
    # smaller pixel sequence so two traces of the same loop serialize alike.
    key_forward = tuple((point[1], point[0]) for point in chain)
    key_reverse = tuple((point[1], point[0]) for point in reverse)
    return (chain, from_id, to_id) if key_forward <= key_reverse else (reverse, from_id, to_id)


def _bridge_ids_for_chain(chain: Sequence[Coord], bridge_owner: np.ndarray | None, accepted: Sequence[Mapping[str, Any]]) -> list[str]:
    """Resolve bridge provenance at raw chain pixels."""

    if bridge_owner is None:
        return []
    ids: set[str] = set()
    height, width = bridge_owner.shape
    for x, y in chain:
        if 0 <= x < width and 0 <= y < height:
            owner = int(bridge_owner[y, x])
            if owner >= 0 and owner < len(accepted):
                ids.add(str(accepted[owner]["bridge_id"]))
    return sorted(ids)


def _component_id_at(
    chain: Sequence[Coord],
    repaired_labels: np.ndarray,
    repaired_id_by_label: Mapping[int, str],
) -> str:
    """Resolve a graph chain to its repaired-corridor component ID."""

    height, width = repaired_labels.shape
    for x, y in chain:
        if 0 <= x < width and 0 <= y < height:
            value = int(repaired_labels[y, x])
            if value:
                return repaired_id_by_label[value]
    return "road_component_unknown"


def _clean_chain(chain: Iterable[Coord]) -> list[Coord]:
    """Remove only consecutive duplicate pixels from a traced chain."""

    result: list[Coord] = []
    for point in chain:
        if not result or point != result[-1]:
            result.append(point)
    return result


def _trace_from_port(
    source_node: str,
    boundary: Coord,
    first_non_node: Coord,
    occupied: set[Coord],
    node_by_pixel: Mapping[Coord, str],
) -> tuple[list[Coord], frozenset[tuple[Coord, Coord]], str, Coord]:
    """Trace a degree-two chain from one node port to the next node."""

    chain: list[Coord] = [boundary, first_non_node]
    segments: set[tuple[Coord, Coord]] = {_canonical_segment(boundary, first_non_node)}
    previous = boundary
    current = first_non_node
    limit = len(occupied) + 1
    while current not in node_by_pixel:
        neighbours = [item for item in _neighbours(current, occupied) if item != previous]
        if len(neighbours) != 1:
            raise ValueError(
                f"skeleton trace from {source_node} reached degree {len(neighbours) + 1} "
                f"at pixel {current}, expected a degree-two corridor"
            )
        following = neighbours[0]
        segments.add(_canonical_segment(current, following))
        chain.append(following)
        previous, current = current, following
        if len(chain) > limit:
            raise ValueError("skeleton trace exceeded its component size; possible infinite loop")
    return chain, frozenset(segments), node_by_pixel[current], current


def _edge_record(
    chain: list[Coord],
    source_node: str,
    destination_node: str,
    *,
    repaired_labels: np.ndarray,
    repaired_id_by_label: Mapping[int, str],
    bridge_owner: np.ndarray | None,
    accepted_bridges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create one raw graph edge with provenance but no world transform yet."""

    chain = _clean_chain(chain)
    chain, from_id, to_id = _orient_chain(chain, source_node, destination_node)
    component_id = _component_id_at(chain, repaired_labels, repaired_id_by_label)
    bridge_ids = _bridge_ids_for_chain(chain, bridge_owner, accepted_bridges)
    return {
        "id": _stable_edge_id(component_id, from_id, to_id, chain),
        "from": from_id,
        "to": to_id,
        "component_id": component_id,
        "raw_pixel_chain": [[int(x), int(y)] for x, y in chain],
        "bridge_ids": bridge_ids,
        "source_status": "source_derived" if not bridge_ids else "source_plus_repair_bridge",
        "provenance": {
            "method": "repaired_mask_skeleton_trace",
            "bridge_ids": bridge_ids,
            "source_status": "source_derived" if not bridge_ids else "source_plus_repair_bridge",
        },
    }


def _validate_graph(
    skeleton: np.ndarray,
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    node_members: Mapping[str, set[Coord]],
    edge_signatures: Sequence[frozenset[tuple[Coord, Coord]]],
) -> dict[str, Any]:
    """Check references, degrees, coverage, and duplicate trace chains."""

    occupied = {(int(x), int(y)) for y, x in zip(*np.nonzero(skeleton))}
    node_ids = {str(node["id"]) for node in nodes}
    covered = set().union(*node_members.values()) if node_members else set()
    degree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    invalid_refs: list[str] = []
    for edge in edges:
        first, second = str(edge["from"]), str(edge["to"])
        if first not in node_ids or second not in node_ids:
            invalid_refs.append(str(edge["id"]))
        degree[first] = degree.get(first, 0) + (2 if first == second else 1)
        degree[second] = degree.get(second, 0) + (0 if first == second else 1)
        for point in edge["raw_pixel_chain"]:
            covered.add((int(point[0]), int(point[1])))
    declared_degree = {str(node["id"]): int(node.get("degree", -1)) for node in nodes}
    degree_mismatches = {
        node_id: {"declared": declared_degree.get(node_id), "computed": value}
        for node_id, value in degree.items()
        if declared_degree.get(node_id) != value
    }
    missing_pixels = sorted(occupied - covered, key=lambda point: (point[1], point[0]))
    extra_pixels = sorted(covered - occupied, key=lambda point: (point[1], point[0]))
    duplicate_signatures = len(edge_signatures) - len(set(edge_signatures))
    component_ids = {str(node["component_id"]) for node in nodes}
    component_graph_nodes = {component_id: 0 for component_id in component_ids}
    component_graph_edges = {component_id: 0 for component_id in component_ids}
    for node in nodes:
        component_graph_nodes[str(node["component_id"])] += 1
    for edge in edges:
        component_graph_edges[str(edge["component_id"])] = component_graph_edges.get(str(edge["component_id"]), 0) + 1
    valid = not invalid_refs and not degree_mismatches and not missing_pixels and not extra_pixels and duplicate_signatures == 0
    return {
        "valid": bool(valid),
        "skeleton_pixels": len(occupied),
        "covered_skeleton_pixels": len(covered & occupied),
        "missing_skeleton_pixels": len(missing_pixels),
        "extra_graph_pixels": len(extra_pixels),
        "invalid_edge_references": invalid_refs,
        "node_degree_mismatches": degree_mismatches,
        "duplicate_edge_chain_count": int(duplicate_signatures),
        "component_graph_nodes": dict(sorted(component_graph_nodes.items())),
        "component_graph_edges": dict(sorted(component_graph_edges.items())),
    }


def build_skeleton_graph(
    repaired_mask: np.ndarray,
    *,
    bridge_owner: np.ndarray | None = None,
    accepted_bridges: Sequence[Mapping[str, Any]] = (),
) -> SkeletonGraph:
    """Skeletonize ``repaired_mask`` and trace its complete graph exactly once."""

    repaired = np.asarray(repaired_mask) > 0
    if repaired.ndim != 2 or repaired.size == 0:
        raise ValueError("repaired road mask must be a non-empty 2-D array")
    if bridge_owner is not None and np.asarray(bridge_owner).shape != repaired.shape:
        raise ValueError("bridge_owner shape does not match repaired mask")
    skeleton = skeletonize(repaired)
    occupied = {(int(x), int(y)) for y, x in zip(*np.nonzero(skeleton))}
    if not occupied:
        raise ValueError("repaired road mask skeletonized to no pixels")
    repaired_labels, repaired_count = component_labels(repaired, connectivity=8)
    component_descriptions, repaired_id_by_label = component_rows(repaired, repaired_labels)
    skeleton_labels, skeleton_count = ndimage.label(
        skeleton, structure=ndimage.generate_binary_structure(2, 2)
    )

    degree_by_pixel = {point: len(_neighbours(point, occupied)) for point in occupied}
    natural_node_pixels = {point for point, degree in degree_by_pixel.items() if degree != 2}
    synthetic_anchors: set[Coord] = set()
    for skeleton_label in range(1, int(skeleton_count) + 1):
        ys, xs = np.nonzero(skeleton_labels == skeleton_label)
        component_pixels = {(int(x), int(y)) for y, x in zip(ys, xs)}
        if not component_pixels & natural_node_pixels:
            synthetic_anchors.add(min(component_pixels, key=lambda item: (item[1], item[0])))
    node_pixels = natural_node_pixels | synthetic_anchors
    clusters = _cluster_pixels(node_pixels)

    node_by_pixel: dict[Coord, str] = {}
    nodes: list[dict[str, Any]] = []
    node_members: dict[str, set[Coord]] = {}
    for cluster in clusters:
        anchor = min(cluster, key=lambda item: (item[1], item[0]))
        label_value = int(repaired_labels[anchor[1], anchor[0]])
        component_id = repaired_id_by_label.get(label_value, "road_component_unknown")
        node_id = _stable_node_id(component_id, cluster)
        for point in cluster:
            if point in node_by_pixel:
                raise AssertionError(f"skeleton node pixel belongs to two clusters: {point}")
            node_by_pixel[point] = node_id
        node_members[node_id] = set(cluster)
        nodes.append(
            {
                "id": node_id,
                "position_px": [int(anchor[0]), int(anchor[1])],
                "component_id": component_id,
                "skeleton_pixels": [
                    [int(x), int(y)] for x, y in sorted(cluster, key=lambda item: (item[1], item[0]))
                ],
                "synthetic_loop_anchor": bool(anchor in synthetic_anchors),
                "degree": 0,
                "kind": "loop_anchor" if anchor in synthetic_anchors else "unclassified",
            }
        )
    nodes.sort(key=lambda row: str(row["id"]))

    edges: list[dict[str, Any]] = []
    edge_signatures: list[frozenset[tuple[Coord, Coord]]] = []
    seen_signatures: set[frozenset[tuple[Coord, Coord]]] = set()

    # First emit direct cross-cluster adjacencies.  Internal pixels of a thick
    # cluster are already represented by node membership and must not become
    # duplicate zero-length edge chains.
    for point in sorted(occupied, key=lambda item: (item[1], item[0])):
        source_node = node_by_pixel.get(point)
        if source_node is None:
            continue
        for neighbour in _neighbours(point, occupied):
            destination_node = node_by_pixel.get(neighbour)
            if destination_node is None or destination_node == source_node:
                continue
            signature = frozenset({_canonical_segment(point, neighbour)})
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            source_anchor = tuple(int(value) for value in next(
                node["position_px"] for node in nodes if node["id"] == source_node
            ))
            destination_anchor = tuple(int(value) for value in next(
                node["position_px"] for node in nodes if node["id"] == destination_node
            ))
            edge = _edge_record(
                _clean_chain([source_anchor, point, neighbour, destination_anchor]),
                source_node,
                destination_node,
                repaired_labels=repaired_labels,
                repaired_id_by_label=repaired_id_by_label,
                bridge_owner=bridge_owner,
                accepted_bridges=accepted_bridges,
            )
            edges.append(edge)
            edge_signatures.append(signature)

    anchor_by_node = {
        str(node["id"]): tuple(int(value) for value in node["position_px"]) for node in nodes
    }
    # Trace every node-to-degree-two port.  Segment-set signatures make the
    # reverse trace of a chain (and the second half of a loop) one edge.
    for node in sorted(nodes, key=lambda row: str(row["id"])):
        source_node = str(node["id"])
        for boundary in sorted(node_members[source_node], key=lambda item: (item[1], item[0])):
            for first_non_node in _neighbours(boundary, occupied):
                if first_non_node in node_by_pixel:
                    continue
                chain, signature, destination_node, _destination_pixel = _trace_from_port(
                    source_node,
                    boundary,
                    first_non_node,
                    occupied,
                    node_by_pixel,
                )
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                full_chain = [anchor_by_node[source_node]] + chain[1:]
                if destination_node == source_node:
                    full_chain.append(anchor_by_node[source_node])
                else:
                    full_chain.append(anchor_by_node[destination_node])
                edge = _edge_record(
                    _clean_chain(full_chain),
                    source_node,
                    destination_node,
                    repaired_labels=repaired_labels,
                    repaired_id_by_label=repaired_id_by_label,
                    bridge_owner=bridge_owner,
                    accepted_bridges=accepted_bridges,
                )
                edges.append(edge)
                edge_signatures.append(signature)

    # Add node degrees after tracing, then classify natural endpoint/junction
    # nodes by graph degree rather than by the number of pixels in the cluster.
    degree_by_node = {str(node["id"]): 0 for node in nodes}
    for edge in edges:
        first, second = str(edge["from"]), str(edge["to"])
        degree_by_node[first] += 2 if first == second else 1
        if first != second:
            degree_by_node[second] += 1
    for node in nodes:
        node_id = str(node["id"])
        degree = degree_by_node[node_id]
        node["degree"] = int(degree)
        if node["synthetic_loop_anchor"]:
            node["kind"] = "loop_anchor"
        elif degree == 0:
            node["kind"] = "isolated"
        elif degree == 1:
            node["kind"] = "endpoint"
        else:
            node["kind"] = "junction"

    nodes.sort(key=lambda row: str(row["id"]))
    edges.sort(key=lambda row: str(row["id"]))
    validation = _validate_graph(skeleton, nodes, edges, node_members, edge_signatures)
    if not validation["valid"]:
        raise ValueError(f"skeleton graph validation failed: {validation}")
    loops = sum(1 for edge in edges if edge["from"] == edge["to"])
    graph_stats = {
        "skeleton_component_count": int(skeleton_count),
        "skeleton_pixels": int(np.count_nonzero(skeleton)),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "loop_edge_count": int(loops),
        "junction_node_count": sum(1 for node in nodes if node["kind"] == "junction"),
        "endpoint_node_count": sum(1 for node in nodes if node["kind"] == "endpoint"),
        "loop_anchor_node_count": sum(1 for node in nodes if node["kind"] == "loop_anchor"),
    }
    return SkeletonGraph(
        skeleton=np.ascontiguousarray(skeleton, dtype=np.uint8),
        component_labels=np.ascontiguousarray(repaired_labels, dtype=np.int32),
        nodes=nodes,
        edges=edges,
        components=component_descriptions,
        validation={**validation, "statistics": graph_stats},
    )


__all__ = ["SkeletonGraph", "build_skeleton_graph"]
