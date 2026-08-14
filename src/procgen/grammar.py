"""Deterministic settlement grammar extraction helpers.

This module is intentionally an analysis layer.  It consumes the compact CELL
summaries produced by :mod:`procgen.espscan`, clusters exterior construction
references around TES3 door links, and serializes recipe-like statistics.  It
does not author TES3 records or make OpenMW API claims.

The clustering is deliberately conservative:

* a DOOR reference with ``DODT`` and non-empty ``DNAM`` is the primary building
  anchor (the settlement-anatomy rule for an exterior door into a dwelling);
* doors sharing the same destination cell are one building, which preserves
  guard towers and large buildings with multiple exterior doors;
* nearby exterior/interior construction pieces are assigned to the nearest
  anchored building in the horizontal footprint (X/Y); unanchored pieces become
  deterministic spatial fallback components so ruins and public shells are not
  silently discarded.  Z is deliberately ignored for this attachment test so
  elevated roof pieces are not lost merely because they sit above the door.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from .espscan import CellReference, CellSummary


PIECE_RADIUS = 1800.0
COMPONENT_RADIUS = 1400.0


@dataclass(frozen=True)
class GrammarCell:
    """A selected cell with only the refs needed by grammar extraction.

    ``references`` may be filtered to construction and NPC refs, while the
    original total counts are retained for useful diagnostics.  Keeping this
    compact object rather than whole source files is the streaming boundary for
    the town driver.
    """

    cell: CellSummary
    source_name: str
    pathgrid_present: bool
    total_ref_count: int
    npc_ref_count: int


@dataclass(frozen=True)
class _Candidate:
    ref: CellReference
    ordinal: int


def _norm(value: str | None) -> str:
    return " ".join((value or "").casefold().strip().split())


def _owner_key(value: str | None) -> str:
    return _norm(value)


def _is_door(ref: CellReference) -> bool:
    return ref.record_type == "DOOR" or ref.category == "door"


def is_npc_reference(ref: CellReference) -> bool:
    """Cheap NPC ref predicate; no NPC record body is parsed."""

    return ref.record_type == "NPC_" or (ref.object_id or "").casefold().startswith("npc_")


def _distance_sq(a: CellReference, b: CellReference) -> float:
    if a.position is None or b.position is None:
        return float("inf")
    return sum((a.position[index] - b.position[index]) ** 2 for index in range(3))


def _distance_position_sq(position: tuple[float, float, float], ref: CellReference) -> float:
    if ref.position is None:
        return float("inf")
    return sum((position[index] - ref.position[index]) ** 2 for index in range(3))


def _linked_destination(ref: CellReference) -> str | None:
    if not ref.door_to_interior:
        return None
    return _norm(ref.destination_cell) or None


def _spatial_components(candidates: Sequence[_Candidate], radius: float) -> list[list[_Candidate]]:
    """Build deterministic spatial fallback components with a coarse index."""

    if not candidates:
        return []
    radius_sq = radius * radius
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    positions: list[tuple[float, float, float] | None] = []
    for index, candidate in enumerate(candidates):
        position = candidate.ref.position
        positions.append(position)
        if position is not None:
            bins[(math.floor(position[0] / radius), math.floor(position[1] / radius))].append(index)

    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, position in enumerate(positions):
        if position is None:
            continue
        bx = math.floor(position[0] / radius)
        by = math.floor(position[1] / radius)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in bins.get((bx + dx, by + dy), ()):
                    if other >= index:
                        continue
                    other_position = positions[other]
                    if other_position is None:
                        continue
                    distance_sq = sum(
                        (position[axis] - other_position[axis]) ** 2 for axis in range(3)
                    )
                    if distance_sq <= radius_sq:
                        union(index, other)

    groups: dict[int, list[_Candidate]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[find(index)].append(candidate)
    return [groups[key] for key in sorted(groups)]


def cluster_buildings(
    cells: Iterable[GrammarCell],
    *,
    piece_radius: float = PIECE_RADIUS,
    component_radius: float = COMPONENT_RADIUS,
) -> list[list[CellReference]]:
    """Cluster selected construction refs into deterministic building groups."""

    candidates: list[_Candidate] = []
    for cell in cells:
        # A town grammar counts exterior building instances.  Interior cells
        # remain in the selected stream for NPC/pathgrid statistics, but their
        # shell/furniture refs must not become additional exterior buildings.
        if cell.cell.is_interior:
            continue
        for ref in cell.cell.references:
            if (ref.building or _is_door(ref)) and ref.position is not None:
                candidates.append(_Candidate(ref=ref, ordinal=len(candidates)))
    if not candidates:
        return []

    # Group linked doors by DNAM.  A multi-door tower or hall therefore remains
    # one building even when the two door positions are farther apart than the
    # shell-piece radius.
    linked: dict[str, list[_Candidate]] = defaultdict(list)
    unlinked_doors: list[_Candidate] = []
    for candidate in candidates:
        if not _is_door(candidate.ref):
            continue
        destination = _linked_destination(candidate.ref)
        if destination is None:
            unlinked_doors.append(candidate)
        else:
            linked[destination].append(candidate)

    groups: list[list[_Candidate]] = []
    anchor_positions: list[list[tuple[float, float, float]]] = []
    for destination in sorted(linked):
        group = sorted(linked[destination], key=lambda item: item.ordinal)
        groups.append(group)
        anchor_positions.append([item.ref.position for item in group if item.ref.position is not None])
    for candidate in sorted(unlinked_doors, key=lambda item: item.ordinal):
        groups.append([candidate])
        anchor_positions.append([candidate.ref.position] if candidate.ref.position is not None else [])

    assigned: set[int] = {candidate.ordinal for group in groups for candidate in group}
    radius_sq = piece_radius * piece_radius
    for candidate in candidates:
        if candidate.ordinal in assigned:
            continue
        distances = [
            min(
                sum((candidate.ref.position[axis] - position[axis]) ** 2 for axis in range(2))
                for position in positions
            )
            if positions and candidate.ref.position is not None
            else float("inf")
            for positions in anchor_positions
        ]
        if distances and min(distances) <= radius_sq:
            group_index = min(range(len(distances)), key=lambda index: (distances[index], index))
            groups[group_index].append(candidate)
            assigned.add(candidate.ordinal)

    # Anything outside an anchored shell is retained as a fallback structure.
    # It is common for ruins or civic shells to have no interior-linked door.
    remainder = [candidate for candidate in candidates if candidate.ordinal not in assigned]
    groups.extend(_spatial_components(remainder, component_radius))

    result = []
    for group in groups:
        ordered = sorted(group, key=lambda item: item.ordinal)
        result.append([item.ref for item in ordered])
    return result


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def size_class(area: float) -> str:
    """Stable footprint-area classes used in signatures and recipes."""

    if area <= 100_000:
        return "tiny"
    if area <= 500_000:
        return "small"
    if area <= 1_500_000:
        return "medium"
    if area <= 4_000_000:
        return "large"
    return "huge"


def density_bucket(value: float) -> str:
    if value <= 0:
        return "empty"
    if value <= 3:
        return "sparse"
    if value <= 10:
        return "village"
    if value <= 30:
        return "town"
    if value <= 75:
        return "city"
    return "dense"


def _dominant(counter: Mapping[str, int], default: str = "none") -> str:
    if not counter:
        return default
    return sorted(counter, key=lambda key: (-int(counter[key]), str(key)))[0]


def _position_values(refs: Sequence[CellReference]) -> list[tuple[float, float, float]]:
    return [ref.position for ref in refs if ref.position is not None]


def _piece_dict(refs: Sequence[CellReference]) -> tuple[list[dict[str, object]], Counter[str]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    details: dict[tuple[str, str, str, str], CellReference] = {}
    for ref in refs:
        key = (
            (ref.model or "").casefold(),
            (ref.object_id or "").casefold(),
            ref.record_type or "",
            ref.kit,
        )
        counts[key] += 1
        details.setdefault(key, ref)
    pieces: list[dict[str, object]] = []
    vocabulary: Counter[str] = Counter()
    for key in sorted(counts):
        ref = details[key]
        model = ref.model or ""
        vocabulary[model.casefold() if model else (ref.object_id or "").casefold()] += counts[key]
        pieces.append(
            {
                "object_id": ref.object_id,
                "model": ref.model,
                "record_type": ref.record_type,
                "kit": ref.kit,
                "category": ref.category,
                "count": counts[key],
            }
        )
    return pieces, vocabulary


def _building_dict(refs: Sequence[CellReference], index: int) -> dict[str, object]:
    positions = _position_values(refs)
    if positions:
        minimum = [min(position[axis] for position in positions) for axis in range(3)]
        maximum = [max(position[axis] for position in positions) for axis in range(3)]
    else:
        minimum = [0.0, 0.0, 0.0]
        maximum = [0.0, 0.0, 0.0]
    width = max(0.0, maximum[0] - minimum[0])
    depth = max(0.0, maximum[1] - minimum[1])
    area = max(1.0, width) * max(1.0, depth)
    pieces, vocabulary = _piece_dict(refs)
    kit_counts = Counter(ref.kit for ref in refs if ref.kit and ref.kit != "unknown")
    doors = [ref for ref in refs if _is_door(ref)]
    linked_doors = [ref for ref in doors if _linked_destination(ref) is not None]
    owners = Counter(_owner_key(ref.owner) for ref in refs if _owner_key(ref.owner))
    owner_display: dict[str, str] = {}
    for ref in refs:
        key = _owner_key(ref.owner)
        if key:
            owner_display.setdefault(key, str(ref.owner))
    owner_key = _dominant(owners, default="")
    owner = owner_display.get(owner_key)
    destination_display: dict[str, str] = {}
    for ref in linked_doors:
        key = _norm(ref.destination_cell)
        if key:
            destination_display.setdefault(key, str(ref.destination_cell))
    destination_cells = [
        destination_display[key]
        for key in sorted(destination_display)
    ]
    door_rows = []
    for ref in doors:
        door_rows.append(
            {
                "object_id": ref.object_id,
                "model": ref.model,
                "position": [_round(value) for value in ref.position] if ref.position else None,
                "owner": ref.owner,
                "destination_cell": ref.destination_cell,
                "has_dodt": ref.has_dodt,
                "door_to_interior": ref.door_to_interior,
            }
        )
    return {
        "building_id": f"b{index:04d}",
        "footprint_bbox": {
            "min": [_round(value) for value in minimum],
            "max": [_round(value) for value in maximum],
            "width": _round(width),
            "depth": _round(depth),
            "area": _round(area),
            "metric": "x-y reference bbox area in game units squared",
        },
        "size_class": size_class(area),
        "pieces": pieces,
        "piece_count": len(refs),
        "piece_vocabulary": sorted(key for key in vocabulary if key),
        "kit": _dominant(kit_counts),
        "kit_histogram": dict(sorted(kit_counts.items())),
        "doors": door_rows,
        "door_count": len(doors),
        "linked_door_count": len(linked_doors),
        "interior_destinations": destination_cells,
        "owner": owner,
        "owner_ids": sorted(owner_display.values(), key=lambda value: (value.casefold(), value)),
        "cluster_method": "door DNAM groups plus nearest shell pieces by horizontal X/Y footprint; spatial fallback for unlinked shells",
    }


def build_town_grammar(
    town_key: str,
    town_name: str,
    cells: Iterable[GrammarCell],
    *,
    metadata: Mapping[str, object] | None = None,
    piece_radius: float = PIECE_RADIUS,
    component_radius: float = COMPONENT_RADIUS,
) -> dict[str, object]:
    """Build one JSON-compatible town grammar from compact selected cells."""

    selected_cells = list(cells)
    buildings = [
        _building_dict(refs, index)
        for index, refs in enumerate(
            cluster_buildings(
                selected_cells,
                piece_radius=piece_radius,
                component_radius=component_radius,
            )
        )
    ]
    buildings.sort(key=lambda item: str(item["building_id"]))
    ext_cells = sum(1 for item in selected_cells if not item.cell.is_interior)
    int_cells = sum(1 for item in selected_cells if item.cell.is_interior)
    pathgrid_cells = sum(1 for item in selected_cells if item.pathgrid_present)
    npc_refs = sum(item.npc_ref_count for item in selected_cells)
    all_refs = [ref for item in selected_cells for ref in item.cell.references]
    building_refs = [ref for ref in all_refs if ref.building or _is_door(ref)]
    sizes = [float(item["footprint_bbox"]["area"]) for item in buildings]  # type: ignore[index]
    size_histogram = Counter(str(item["size_class"]) for item in buildings)
    kit_histogram = Counter(
        ref.kit for ref in building_refs if ref.kit and ref.kit != "unknown"
    )
    vocabulary = sorted(
        {
            (ref.model or ref.object_id or "").casefold()
            for ref in building_refs
            if (ref.model or ref.object_id)
        }
    )
    door_count = sum(int(item["door_count"]) for item in buildings)
    linked_door_count = sum(int(item["linked_door_count"]) for item in buildings)
    density = len(buildings) / ext_cells if ext_cells else 0.0
    modal_size = _dominant(size_histogram)
    stats = {
        "building_count": len(buildings),
        "building_ref_count": len(building_refs),
        "selected_cell_count": len(selected_cells),
        "exterior_cell_count": ext_cells,
        "interior_cell_count": int_cells,
        "density_per_cell": _round(density, 6),
        "density_metric": "clustered buildings per selected exterior CELL",
        "size_p10": _round(_percentile(sizes, 10)),
        "size_p50": _round(_percentile(sizes, 50)),
        "size_p90": _round(_percentile(sizes, 90)),
        "size_metric": "footprint_bbox area in x-y game units squared",
        "size_class_histogram": dict(sorted(size_histogram.items())),
        "piece_vocabulary_size": len(vocabulary),
        "kit_histogram": dict(sorted(kit_histogram.items())),
        "dominant_kit": _dominant(kit_histogram),
        "doors": door_count,
        "linked_doors": linked_door_count,
        "unlinked_doors": max(0, door_count - linked_door_count),
        "door_link_rate": _round(linked_door_count / door_count, 6) if door_count else 0.0,
        "npc_ref_count": npc_refs,
        "pathgrid_cells": pathgrid_cells,
        "pathgrid_presence_rate": _round(pathgrid_cells / len(selected_cells), 6)
        if selected_cells
        else 0.0,
        "source_files": sorted({item.source_name for item in selected_cells if item.source_name}),
    }
    signature = {
        "dominant_kit": stats["dominant_kit"],
        "density_bucket": density_bucket(density),
        "size_class_histogram": dict(sorted(size_histogram.items())),
        "modal_size_class": modal_size,
        "pathgrid_bucket": (
            "none" if pathgrid_cells == 0 else "all" if pathgrid_cells == len(selected_cells) else "partial"
        ),
    }
    row: dict[str, object] = {
        "settlement_key": town_key,
        "name": town_name,
        "buildings": buildings,
        "stats": stats,
        "kit_piece_vocabulary": vocabulary,
        "grammar_signature": signature,
    }
    if metadata:
        for key, value in metadata.items():
            if key not in row:
                row[key] = value
    return row


def _cluster_key(town: Mapping[str, object], include_size: bool = True) -> tuple[str, ...]:
    signature = town.get("grammar_signature", {})
    if not isinstance(signature, Mapping):
        signature = {}
    base = (
        str(signature.get("dominant_kit", "none")),
        str(signature.get("density_bucket", "empty")),
    )
    if include_size:
        return base + (str(signature.get("modal_size_class", "tiny")),)
    return base


def _centroid(members: Sequence[Mapping[str, object]]) -> dict[str, object]:
    stats = [item.get("stats", {}) for item in members]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in stats if isinstance(row, Mapping) and key in row]

    return {
        "building_count": _round(sum(values("building_count")) / len(values("building_count")), 3)
        if values("building_count")
        else 0.0,
        "density_per_cell": _round(sum(values("density_per_cell")) / len(values("density_per_cell")), 6)
        if values("density_per_cell")
        else 0.0,
        "size_p50": _round(sum(values("size_p50")) / len(values("size_p50")), 3)
        if values("size_p50")
        else 0.0,
        "doors": _round(sum(values("doors")) / len(values("doors")), 3) if values("doors") else 0.0,
        "npc_ref_count": _round(sum(values("npc_ref_count")) / len(values("npc_ref_count")), 3)
        if values("npc_ref_count")
        else 0.0,
        "pathgrid_presence_rate": _round(
            sum(values("pathgrid_presence_rate")) / len(values("pathgrid_presence_rate")), 6
        )
        if values("pathgrid_presence_rate")
        else 0.0,
    }


def cluster_archetypes(towns: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Cluster towns by stable recipe buckets without ML or randomness.

    The first pass includes dominant kit, density bucket, and modal size class.
    If that would exceed the accepted 30-archetype ceiling, the size component
    is dropped deterministically; kit × six density buckets is at most 30.
    """

    include_size = True
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for town in towns:
        grouped[_cluster_key(town, include_size)].append(town)
    if len(grouped) > 30:
        include_size = False
        grouped = defaultdict(list)
        for town in towns:
            grouped[_cluster_key(town, include_size)].append(town)

    ordered_keys = sorted(grouped)
    archetypes: list[dict[str, object]] = []
    assignments: dict[str, str] = {}
    for number, key in enumerate(ordered_keys, 1):
        members = sorted(
            grouped[key],
            key=lambda town: (str(town.get("settlement_key", "")), str(town.get("name", ""))),
        )
        archetype_id = f"recipe_{number:02d}"
        kit = key[0] if key else "none"
        density = key[1] if len(key) > 1 else "empty"
        size = key[2] if len(key) > 2 else "mixed"
        name = f"{kit}-{density}-{size}" if kit != "none" else f"unobserved-{density}-{size}"
        member_keys = [str(town.get("settlement_key", "")) for town in members]
        for member_key in member_keys:
            assignments[member_key] = archetype_id
        size_histogram: Counter[str] = Counter()
        for town in members:
            signature = town.get("grammar_signature", {})
            if isinstance(signature, Mapping):
                raw = signature.get("size_class_histogram", {})
                if isinstance(raw, Mapping):
                    for size_name, count in raw.items():
                        size_histogram[str(size_name)] += int(count)
        archetypes.append(
            {
                "archetype_id": archetype_id,
                "name": name,
                "members": member_keys,
                "member_count": len(member_keys),
                "signature": {
                    "dominant_kit": kit,
                    "density_bucket": density,
                    "modal_size_class": size,
                    "size_class_histogram": dict(sorted(size_histogram.items())),
                    "clustering_dimensions": ["dominant_kit", "density_bucket"]
                    + (["modal_size_class"] if include_size else []),
                },
                "centroid": _centroid(members),
            }
        )
    return archetypes, assignments


__all__ = [
    "COMPONENT_RADIUS",
    "GrammarCell",
    "PIECE_RADIUS",
    "build_town_grammar",
    "cluster_archetypes",
    "cluster_buildings",
    "density_bucket",
    "is_npc_reference",
    "size_class",
]
