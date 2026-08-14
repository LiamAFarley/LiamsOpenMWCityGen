"""Contact/proximity building chunking for TES3 exterior reference analysis.

The original :mod:`procgen.grammar` assignment is intentionally left alone.  A
v2 chunk is a geometry decision instead: resolved reference pieces carry a
world-aligned bounding box in game units and enter a contact graph when their
boxes overlap or have a gap no larger than ``epsilon``.  A door is a seed/core,
and breadth-first growth follows only those contact links.  ``proximity_bound``
limits the distance of a seed from the component under consideration; it is a
diagnostic guard rather than a reason to cut a contact-connected mesh apart.

That last distinction is important for authored rows, walls, and long roofs:
continuous geometry is never auto-split.  If a contact-connected component
extends past the seed bound it is retained and labelled ``building_cluster``
with a diagnostic flag.  Separate houses cannot be joined merely because they
are nearby; they need an actual bbox contact edge.

The module has no Blender or TES3 file dependency.  Blender-produced mesh
bboxes are supplied by callers, which keeps the analysis/generation/output
pipeline boundaries explicit and makes the synthetic tests fast.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
import re
from typing import Iterable, Mapping, Sequence


DEFAULT_EPSILON_GU = 10.0
DEFAULT_PROXIMITY_BOUND_GU = 1800.0
DEFAULT_CITY_WALL_RUN_GU = 600.0
# These bounds are deliberately conservative for a single exterior house or
# hut.  They are analysis thresholds, not TES3 limits: a contact-connected
# over-bound component is retained and typed as a cluster rather than split.
DEFAULT_BUILDING_MIN_PIECES = 4
DEFAULT_BUILDING_MAX_PIECES = 80
DEFAULT_BUILDING_MIN_VOLUME_GU3 = 10_000.0
DEFAULT_BUILDING_MAX_VOLUME_GU3 = 5_000_000_000.0

COMPONENT_TYPES = frozenset(
    {
        "building",
        "building_cluster",
        "gate_fragment",
        "cave_entrance",
        "castle_complex",
        "barrow",
        "city_wall",
        "yard_object",
    }
)

# The vocabulary is intentionally mesh/object based rather than province-kit
# based.  A building needs at least one non-door piece carrying one of these
# shell markers; doors, railings, fences and decorative clutter cannot satisfy
# the requirement by themselves.
SHELL_VOCABULARY = frozenset(
    {
        "base",
        "beam",
        "bottom",
        "building",
        "chimney",
        "doorframe",
        "floor",
        "house",
        "hut",
        "pillar",
        "roof",
        "scrapwood",
        "shelter",
        "sidewall",
        "wall",
        "window",
    }
)

GATE_VOCABULARY = frozenset(
    {"gate", "fencegate", "fortgate", "palisade", "railing", "fence", "post"}
)
CAVE_VOCABULARY = frozenset({"cave", "rock", "tunnel", "mine", "mineshaft"})
BARROW_VOCABULARY = frozenset({"barrow", "dngbarrow", "barrowf"})
CASTLE_VOCABULARY = frozenset({"castle", "palace", "arena", "citadel"})


def _finite_triplet(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class WorldBBox:
    """A world-aligned bounding box in TES3 game units (GU)."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = _finite_triplet(self.minimum, "bbox.minimum")
        maximum = _finite_triplet(self.maximum, "bbox.maximum")
        if any(minimum[index] > maximum[index] for index in range(3)):
            raise ValueError("bbox minimum must not exceed maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def span(self) -> tuple[float, float, float]:
        return tuple(self.maximum[index] - self.minimum[index] for index in range(3))  # type: ignore[return-value]

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((self.minimum[index] + self.maximum[index]) / 2.0 for index in range(3))  # type: ignore[return-value]

    def union(self, other: "WorldBBox") -> "WorldBBox":
        return WorldBBox(
            tuple(min(self.minimum[index], other.minimum[index]) for index in range(3)),  # type: ignore[arg-type]
            tuple(max(self.maximum[index], other.maximum[index]) for index in range(3)),  # type: ignore[arg-type]
        )

    def contact_gaps(self, other: "WorldBBox") -> tuple[float, float, float]:
        """Return non-negative gaps per axis; overlap is represented by zero."""

        return tuple(
            max(self.minimum[index] - other.maximum[index], other.minimum[index] - self.maximum[index], 0.0)
            for index in range(3)
        )  # type: ignore[return-value]

    def touches(self, other: "WorldBBox", epsilon: float = DEFAULT_EPSILON_GU) -> bool:
        if epsilon < 0.0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be a finite non-negative number")
        return max(self.contact_gaps(other)) <= epsilon

    def distance_xy(self, other: "WorldBBox") -> float:
        gaps = self.contact_gaps(other)
        return math.hypot(gaps[0], gaps[1])


@dataclass(frozen=True)
class ChunkPiece:
    """Geometry and TES3 metadata required by the v2 chunker."""

    piece_id: str
    model: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    bbox: WorldBBox = field(default_factory=lambda: WorldBBox((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    record_type: str = "STAT"
    category: str = "exterior"
    object_id: str | None = None
    is_door: bool = False
    destination_cell: str | None = None
    structural: bool = True
    source_cell: tuple[int, int] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "piece_id", str(self.piece_id))
        object.__setattr__(self, "model", str(self.model))
        object.__setattr__(self, "position", _finite_triplet(self.position, "piece.position"))
        object.__setattr__(self, "rotation", _finite_triplet(self.rotation, "piece.rotation"))
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError("piece.scale must be a finite positive number")
        if self.source_cell is not None and len(self.source_cell) != 2:
            raise ValueError("piece.source_cell must contain two grid integers")

    @property
    def seed_key(self) -> str:
        """Stable door-core key; unlinked doors are intentionally independent."""

        if not self.is_door:
            return ""
        destination = " ".join((self.destination_cell or "").casefold().split())
        return f"destination:{destination}" if destination else f"door:{self.piece_id}"


@dataclass(frozen=True)
class ValidationFlag:
    component_id: str
    code: str
    piece_ids: tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "code": self.code,
            "piece_ids": list(self.piece_ids),
            "message": self.message,
        }


@dataclass(frozen=True)
class SeveredUnit:
    """One door-seeded region produced from a multi-door contact component.

    ``piece_ids`` is the render-complete membership for the unit.  When
    ``tie_policy`` is ``duplicate``, a shared wall/foundation can therefore
    occur in more than one unit and is listed in ``duplicated_piece_ids``.
    ``primary_piece_owner`` is not exposed as a public field: it is an
    internal deterministic partition used to decide which graph edges were cut
    and to avoid ambiguous cut-edge ownership when a tie is duplicated.
    """

    seed_door_id: str
    piece_ids: tuple[str, ...]
    duplicated_piece_ids: tuple[str, ...] = ()
    cut_edges: tuple[tuple[str, str], ...] = ()
    distances: Mapping[str, int] = field(default_factory=dict)
    tie_policy: str = "duplicate"
    removed_sliver_piece_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "seed_door_id": self.seed_door_id,
            "piece_ids": list(self.piece_ids),
            "piece_count": len(self.piece_ids),
            "duplicated_piece_ids": list(self.duplicated_piece_ids),
            "duplicated_piece_count": len(self.duplicated_piece_ids),
            "cut_edges": [list(edge) for edge in self.cut_edges],
            "cut_edge_count": len(self.cut_edges),
            "contact_path_distance": dict(sorted(self.distances.items())),
            "tie_policy": self.tie_policy,
            "removed_sliver_piece_ids": list(self.removed_sliver_piece_ids),
            "removed_sliver_piece_count": len(self.removed_sliver_piece_ids),
        }


@dataclass(frozen=True)
class ChunkComponent:
    """One v2 geometry component."""

    component_id: str
    component_type: str
    piece_ids: tuple[str, ...]
    door_ids: tuple[str, ...]
    seed_keys: tuple[str, ...]
    contact_edges: tuple[tuple[str, str], ...]
    bbox: WorldBBox
    flags: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def has_door(self) -> bool:
        return bool(self.door_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "type": self.component_type,
            "piece_ids": list(self.piece_ids),
            "door_ids": list(self.door_ids),
            "seed_keys": list(self.seed_keys),
            "contact_edges": [list(edge) for edge in self.contact_edges],
            "bbox": {"min": list(self.bbox.minimum), "max": list(self.bbox.maximum)},
            "flags": list(self.flags),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class ComponentTyping:
    """Deterministic semantic typing decision for one geometry group.

    ``reasons`` is part of the analysis contract.  It is written into the
    typed-component report so a later visual curation pass can distinguish a
    rejected shell from a deliberately retained cluster or special structure.
    """

    component_type: str
    reasons: tuple[str, ...]
    shell_piece_ids: tuple[str, ...]
    piece_count: int
    door_count: int
    destination_count: int
    volume_gu3: float

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.component_type,
            "reasons": list(self.reasons),
            "shell_piece_ids": list(self.shell_piece_ids),
            "shell_piece_count": len(self.shell_piece_ids),
            "piece_count": self.piece_count,
            "door_count": self.door_count,
            "destination_count": self.destination_count,
            "bbox_volume_gu3": round(self.volume_gu3, 6),
            "piece_band": [DEFAULT_BUILDING_MIN_PIECES, DEFAULT_BUILDING_MAX_PIECES],
            "volume_band_gu3": [DEFAULT_BUILDING_MIN_VOLUME_GU3, DEFAULT_BUILDING_MAX_VOLUME_GU3],
        }


def _normalise(value: str | None) -> str:
    return " ".join((value or "").casefold().replace("/", " ").replace("\\", " ").split())


def _has_vocabulary(text: str, vocabulary: Iterable[str]) -> bool:
    """Return true when a normalized model/object string contains a marker."""

    return any(marker in text for marker in vocabulary)


def _piece_vocabulary_text(piece: ChunkPiece) -> str:
    return _normalise(f"{piece.model} {piece.object_id or ''}")


def _shell_piece_ids(pieces: Sequence[ChunkPiece]) -> tuple[str, ...]:
    return tuple(
        sorted(
            piece.piece_id
            for piece in pieces
            if not piece.is_door and _has_vocabulary(_piece_vocabulary_text(piece), SHELL_VOCABULARY)
        )
    )


def classify_component(
    component: ChunkComponent,
    pieces: Mapping[str, ChunkPiece],
    *,
    existing_type: str | None = None,
) -> ComponentTyping:
    """Classify a chunk using shell vocabulary and documented size bands.

    The ordering is intentional: special structures are typed before the
    generic building test, and a contact-connected component that is too large
    remains a ``building_cluster``.  This function does not remove pieces.
    Visual membership changes belong to the live curation stage.
    """

    members = [pieces[piece_id] for piece_id in component.piece_ids]
    doors = [piece for piece in members if piece.is_door]
    texts = [_piece_vocabulary_text(piece) for piece in members]
    combined = " ".join(texts + [_normalise(destination) for destination in component.seed_keys])
    shell_ids = _shell_piece_ids(members)
    span = component.bbox.span
    volume = float(span[0] * span[1] * span[2])
    destinations = {
        _normalise(piece.destination_cell)
        for piece in doors
        if _normalise(piece.destination_cell)
    }
    reasons: list[str] = []

    if existing_type == "yard_object":
        return ComponentTyping("yard_object", ("non-structural component",), shell_ids,
                               len(members), len(doors), len(destinations), volume)
    if existing_type == "city_wall":
        return ComponentTyping("city_wall", ("fortification vocabulary/long-run wall heuristic",), shell_ids,
                               len(members), len(doors), len(destinations), volume)
    if _has_vocabulary(combined, BARROW_VOCABULARY):
        reasons.append("barrow vocabulary in model/object/destination")
        if len(members) > DEFAULT_BUILDING_MAX_PIECES:
            reasons.append(f"piece count {len(members)} exceeds single-building maximum {DEFAULT_BUILDING_MAX_PIECES}")
        return ComponentTyping("barrow", tuple(reasons), shell_ids, len(members), len(doors), len(destinations), volume)
    if _has_vocabulary(combined, CASTLE_VOCABULARY):
        reasons.append("castle/arena/palace vocabulary in model/object/destination")
        if len(members) > DEFAULT_BUILDING_MAX_PIECES:
            reasons.append(f"piece count {len(members)} exceeds single-building maximum {DEFAULT_BUILDING_MAX_PIECES}")
        if volume > DEFAULT_BUILDING_MAX_VOLUME_GU3:
            reasons.append(f"bbox volume {volume:.3f} exceeds {DEFAULT_BUILDING_MAX_VOLUME_GU3:.3f} GU³")
        return ComponentTyping("castle_complex", tuple(reasons), shell_ids, len(members), len(doors), len(destinations), volume)
    cave_door_or_destination = any(
        _has_vocabulary(_piece_vocabulary_text(piece), CAVE_VOCABULARY)
        or _has_vocabulary(_normalise(piece.destination_cell), CAVE_VOCABULARY)
        for piece in doors
    )
    if cave_door_or_destination or (not shell_ids and _has_vocabulary(combined, CAVE_VOCABULARY)):
        return ComponentTyping(
            "cave_entrance",
            ("cave/rock/tunnel/mine vocabulary on a door/destination or shell-less component",),
            shell_ids,
            len(members),
            len(doors),
            len(destinations),
            volume,
        )

    if not doors:
        return ComponentTyping(
            "building_cluster",
            ("no door seed; retained as unassigned structural geometry",),
            shell_ids,
            len(members),
            0,
            0,
            volume,
        )
    if not shell_ids:
        return ComponentTyping(
            "gate_fragment",
            ("door-bearing component has no shell-vocabulary piece",),
            shell_ids,
            len(members),
            len(doors),
            len(destinations),
            volume,
        )
    if _has_vocabulary(combined, GATE_VOCABULARY) and len(shell_ids) <= 1:
        return ComponentTyping(
            "gate_fragment",
            ("gate/fence vocabulary with no substantive shell",),
            shell_ids,
            len(members),
            len(doors),
            len(destinations),
            volume,
        )
    if len(members) < DEFAULT_BUILDING_MIN_PIECES:
        return ComponentTyping(
            "gate_fragment",
            (f"piece count {len(members)} below single-building minimum {DEFAULT_BUILDING_MIN_PIECES}",),
            shell_ids,
            len(members),
            len(doors),
            len(destinations),
            volume,
        )
    if len(members) > DEFAULT_BUILDING_MAX_PIECES:
        return ComponentTyping(
            "building_cluster",
            (f"piece count {len(members)} exceeds single-building maximum {DEFAULT_BUILDING_MAX_PIECES}",),
            shell_ids,
            len(members),
            len(doors),
            len(destinations),
            volume,
        )
    if volume < DEFAULT_BUILDING_MIN_VOLUME_GU3:
        return ComponentTyping(
            "gate_fragment",
            (f"bbox volume {volume:.3f} below {DEFAULT_BUILDING_MIN_VOLUME_GU3:.3f} GU³",),
            shell_ids,
            len(members),
            len(doors),
            len(destinations),
            volume,
        )
    if volume > DEFAULT_BUILDING_MAX_VOLUME_GU3:
        return ComponentTyping(
            "building_cluster",
            (f"bbox volume {volume:.3f} exceeds {DEFAULT_BUILDING_MAX_VOLUME_GU3:.3f} GU³",),
            shell_ids,
            len(members),
            len(doors),
            len(destinations),
            volume,
        )
    if len(destinations) > 1 or existing_type == "building_cluster":
        if len(destinations) > 1:
            reasons.append(f"{len(destinations)} distinct door destinations in one contact component")
        if existing_type == "building_cluster":
            reasons.append("contact/proximity chunker marked continuous or merged geometry as a cluster")
        return ComponentTyping("building_cluster", tuple(reasons), shell_ids, len(members), len(doors), len(destinations), volume)

    reasons.extend(
        (
            "door-bearing component",
            f"shell vocabulary present ({len(shell_ids)} pieces)",
            f"piece count {len(members)} within {DEFAULT_BUILDING_MIN_PIECES}..{DEFAULT_BUILDING_MAX_PIECES}",
            f"bbox volume {volume:.3f} GU³ within configured band",
        )
    )
    return ComponentTyping("building", tuple(reasons), shell_ids, len(members), len(doors), len(destinations), volume)


FORTIFICATION_VOCABULARY = frozenset(
    {
        "battlement",
        "castlewall",
        "citywall",
        "curtainwall",
        "fortwall",
        "fortification",
        "keepwall",
        "palisade",
        "rampart",
        "stonewall",
        "wallsegment",
        "wall section",
    }
)


def _has_fortification_vocabulary(piece: ChunkPiece) -> bool:
    text = _normalise(f"{piece.model} {piece.object_id or ''}")
    tokens = set(re.split(r"[^a-z0-9]+", text))
    return any(token and (token in text if " " in token else token in tokens) for token in FORTIFICATION_VOCABULARY)


def is_city_wall_piece(piece: ChunkPiece, *, minimum_run_gu: float = DEFAULT_CITY_WALL_RUN_GU) -> bool:
    """Classify exterior fortification pieces without consuming house walls.

    The documented heuristic is deliberately two-part: a fortification term in
    the MODL/object vocabulary, plus a long run (largest horizontal span at
    least ``minimum_run_gu`` or at least three times the smaller horizontal
    span).  Explicitly named palisade/rampart/curtain/city/fort walls are also
    accepted when their model is a short segment because those kits commonly
    tile short pieces.  ``house``/``hut``/``ruin`` markers override a generic
    ``wall`` token.
    """

    if not piece.structural or piece.is_door:
        return False
    text = _normalise(f"{piece.model} {piece.object_id or ''}")
    if any(marker in text for marker in ("house wall", "housewall", "hut wall", "ruin wall", "ruinwall")):
        return False
    if not _has_fortification_vocabulary(piece):
        return False
    span_x, span_y, _span_z = piece.bbox.span
    long_run = max(span_x, span_y) >= float(minimum_run_gu) or max(span_x, span_y) >= max(1.0, 3.0 * min(span_x, span_y))
    explicit = any(marker in text for marker in ("palisade", "rampart", "curtain", "city wall", "citywall", "fortwall"))
    return long_run or explicit


def build_contact_graph(
    pieces: Iterable[ChunkPiece],
    *,
    epsilon: float = DEFAULT_EPSILON_GU,
    minimum_run_gu: float = DEFAULT_CITY_WALL_RUN_GU,
    minimum_contact_area_gu2: float = 0.0,
) -> tuple[dict[str, ChunkPiece], dict[str, set[str]], set[str]]:
    """Build deterministic contact adjacency and identify city-wall pieces.

    ``minimum_contact_area_gu2`` is an optional graph-level guard against
    bbox-grazing slivers. The historical default is zero, so existing callers
    retain the epsilon-only contact behavior.
    """

    ordered = sorted(pieces, key=lambda item: item.piece_id)
    if minimum_contact_area_gu2 < 0.0 or not math.isfinite(minimum_contact_area_gu2):
        raise ValueError("minimum_contact_area_gu2 must be a finite non-negative number")
    by_id: dict[str, ChunkPiece] = {}
    for piece in ordered:
        if piece.piece_id in by_id:
            raise ValueError(f"duplicate piece id: {piece.piece_id}")
        by_id[piece.piece_id] = piece
    graph: dict[str, set[str]] = {piece_id: set() for piece_id in by_id}
    city_walls = {piece_id for piece_id, piece in by_id.items() if is_city_wall_piece(piece, minimum_run_gu=minimum_run_gu)}
    candidates = [piece for piece in ordered if piece.structural and piece.piece_id not in city_walls]
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if (
                left.bbox.touches(right.bbox, epsilon)
                and bbox_contact_area(left.bbox, right.bbox, epsilon=epsilon)
                >= minimum_contact_area_gu2
            ):
                graph[left.piece_id].add(right.piece_id)
                graph[right.piece_id].add(left.piece_id)
    return by_id, graph, city_walls


def _connected_components(graph: Mapping[str, set[str]], allowed: set[str]) -> list[tuple[str, ...]]:
    remaining = set(allowed)
    result: list[tuple[str, ...]] = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        remaining.remove(start)
        members: list[str] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in sorted(graph.get(current, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        result.append(tuple(sorted(members)))
    return sorted(result)


def _door_seeded_bfs(graph: Mapping[str, set[str]], core_ids: Sequence[str], allowed: set[str]) -> set[str]:
    """Walk contact links from door cores in stable order."""

    reached: set[str] = set()
    queue: deque[str] = deque(sorted(set(core_ids) & allowed))
    reached.update(queue)
    while queue:
        current = queue.popleft()
        for neighbor in sorted(graph.get(current, ())):
            if neighbor in allowed and neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def _union_bbox(pieces: Mapping[str, ChunkPiece], piece_ids: Sequence[str]) -> WorldBBox:
    if not piece_ids:
        return WorldBBox((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    result = pieces[piece_ids[0]].bbox
    for piece_id in piece_ids[1:]:
        result = result.union(pieces[piece_id].bbox)
    return result


def _component_edges(graph: Mapping[str, set[str]], piece_ids: Sequence[str]) -> tuple[tuple[str, str], ...]:
    allowed = set(piece_ids)
    return tuple(
        (piece_id, neighbor)
        for piece_id in sorted(allowed)
        for neighbor in sorted(graph.get(piece_id, ()) if graph.get(piece_id, set()) else ())
        if neighbor in allowed and piece_id < neighbor
    )


def chunk_pieces(
    pieces: Iterable[ChunkPiece],
    *,
    epsilon: float = DEFAULT_EPSILON_GU,
    proximity_bound: float = DEFAULT_PROXIMITY_BOUND_GU,
    minimum_run_gu: float = DEFAULT_CITY_WALL_RUN_GU,
    minimum_contact_area_gu2: float = 0.0,
) -> list[ChunkComponent]:
    """Return deterministic contact/proximity components.

    Door-seeded BFS is performed over the contact graph.  A component becomes a
    ``building`` only when it has a door, shell vocabulary, and the documented
    piece/volume band.  Multiple destinations, over-bound geometry, or missing
    doors remain ``building_cluster``.  Special vocabulary produces
    ``gate_fragment``, ``cave_entrance``, ``castle_complex``, or ``barrow``.
    It is never split merely to satisfy the bound.  Non-structural pieces become
    ``yard_object`` components and city exterior walls become independent
    ``city_wall`` components.
    """

    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a finite non-negative number")
    if proximity_bound < 0.0 or not math.isfinite(proximity_bound):
        raise ValueError("proximity_bound must be a finite non-negative number")
    if minimum_contact_area_gu2 < 0.0 or not math.isfinite(minimum_contact_area_gu2):
        raise ValueError("minimum_contact_area_gu2 must be a finite non-negative number")
    by_id, graph, city_walls = build_contact_graph(
        pieces,
        epsilon=epsilon,
        minimum_run_gu=minimum_run_gu,
        minimum_contact_area_gu2=minimum_contact_area_gu2,
    )
    structural_ids = {piece_id for piece_id, piece in by_id.items() if piece.structural and piece_id not in city_walls}
    structural_components = _connected_components(graph, structural_ids)
    output: list[ChunkComponent] = []

    for member_ids in structural_components:
        members = [by_id[piece_id] for piece_id in member_ids]
        doors = sorted((piece for piece in members if piece.is_door), key=lambda item: item.piece_id)
        seed_keys = tuple(sorted({piece.seed_key for piece in doors}))
        core_ids = tuple(piece.piece_id for piece in doors)
        seeded_reached = _door_seeded_bfs(graph, core_ids, set(member_ids)) if core_ids else set()
        # Contact-connected geometry is authoritative.  The assertion-like
        # union below makes the no-auto-split rule explicit even if a future
        # graph implementation adds a proximity pre-filter.
        bfs_piece_ids = tuple(sorted(set(member_ids) | seeded_reached))
        component_bbox = _union_bbox(by_id, bfs_piece_ids)
        core_bbox = _union_bbox(by_id, core_ids) if core_ids else component_bbox
        # Measure the bounded BFS against the nearest individual door core,
        # rather than the union of all doors.  A long continuous row with doors
        # at both ends must still expose that its middle is beyond the local
        # seed bound; it is nevertheless retained intact below.
        core_boxes = [by_id[piece_id].bbox for piece_id in core_ids]
        proximity_exceeded = bool(
            core_boxes
            and any(
                min(by_id[piece_id].bbox.distance_xy(core_box) for core_box in core_boxes) > proximity_bound
                for piece_id in member_ids
            )
        )
        flags: list[str] = []
        if not doors:
            flags.append("no_door")
        if proximity_exceeded:
            flags.append("proximity_bound_exceeded_continuous_geometry")
        entrance_only = bool(doors and all(piece.is_door for piece in members))
        if entrance_only:
            component_type = "gate_fragment"
        elif len(seed_keys) > 1 or not doors or proximity_exceeded:
            component_type = "building_cluster"
        else:
            component_type = "building"
        # A single isolated structural piece without an anchor is useful
        # evidence, but the explicit floating flag tells later generation not
        # to assume it has a doorway or ground attachment.
        if not doors:
            flags.append("floating_piece")
        provisional = ChunkComponent(
            component_id="",
            component_type=component_type,
            piece_ids=bfs_piece_ids,
            door_ids=tuple(piece.piece_id for piece in doors),
            seed_keys=seed_keys,
            contact_edges=_component_edges(graph, bfs_piece_ids),
            bbox=component_bbox,
            flags=tuple(sorted(set(flags))),
            diagnostics={
                "epsilon_gu": epsilon,
                "proximity_bound_gu": proximity_bound,
                "minimum_contact_area_gu2": minimum_contact_area_gu2,
                "core_piece_ids": list(core_ids),
                "contact_linked_piece_count": len(bfs_piece_ids),
                "door_seeded_bfs_reached_count": len(seeded_reached),
                "contact_graph_edge_count": len(_component_edges(graph, bfs_piece_ids)),
                "continuous_geometry_preserved": True,
            },
        )
        typing = classify_component(provisional, by_id, existing_type=component_type)
        output.append(
            ChunkComponent(
                component_id="",
                component_type=typing.component_type,
                piece_ids=bfs_piece_ids,
                door_ids=tuple(piece.piece_id for piece in doors),
                seed_keys=seed_keys,
                contact_edges=_component_edges(graph, bfs_piece_ids),
                bbox=component_bbox,
                flags=tuple(sorted(set(flags))),
                diagnostics={
                    "epsilon_gu": epsilon,
                    "proximity_bound_gu": proximity_bound,
                    "minimum_contact_area_gu2": minimum_contact_area_gu2,
                    "core_piece_ids": list(core_ids),
                    "contact_linked_piece_count": len(bfs_piece_ids),
                    "door_seeded_bfs_reached_count": len(seeded_reached),
                    "contact_graph_edge_count": len(_component_edges(graph, bfs_piece_ids)),
                    "continuous_geometry_preserved": True,
                    "typing_reasons": list(typing.reasons),
                    "shell_piece_ids": list(typing.shell_piece_ids),
                    "bbox_volume_gu3": typing.volume_gu3,
                },
            )
        )

    # City walls are intentionally not allowed to bridge any structural group,
    # but touching wall segments are retained as one wall run/component.
    wall_graph: dict[str, set[str]] = {piece_id: set() for piece_id in city_walls}
    wall_pieces = [by_id[piece_id] for piece_id in sorted(city_walls)]
    for index, left in enumerate(wall_pieces):
        for right in wall_pieces[index + 1 :]:
            if (
                left.bbox.touches(right.bbox, epsilon)
                and bbox_contact_area(left.bbox, right.bbox, epsilon=epsilon)
                >= minimum_contact_area_gu2
            ):
                wall_graph[left.piece_id].add(right.piece_id)
                wall_graph[right.piece_id].add(left.piece_id)
    for member_ids in _connected_components(wall_graph, city_walls):
        wall_bbox = _union_bbox(by_id, member_ids)
        output.append(
            ChunkComponent(
                component_id="",
                component_type="city_wall",
                piece_ids=member_ids,
                door_ids=(),
                seed_keys=(),
                contact_edges=_component_edges(wall_graph, member_ids),
                bbox=wall_bbox,
                flags=("no_door",),
                diagnostics={
                    "heuristic": "fortification vocabulary plus long horizontal run; explicit palisade/rampart/curtain/fortwall vocabulary accepted for short tiles",
                    "epsilon_gu": epsilon,
                    "proximity_bound_gu": proximity_bound,
                    "minimum_contact_area_gu2": minimum_contact_area_gu2,
                    "wall_run_piece_count": len(member_ids),
                },
            )
        )

    yard_ids = {piece_id for piece_id, piece in by_id.items() if not piece.structural and piece_id not in city_walls}
    for member_ids in _connected_components(graph, yard_ids):
        output.append(
            ChunkComponent(
                component_id="",
                component_type="yard_object",
                piece_ids=member_ids,
                door_ids=(),
                seed_keys=(),
                contact_edges=_component_edges(graph, member_ids),
                bbox=_union_bbox(by_id, member_ids),
                flags=("no_door",),
                diagnostics={
                    "epsilon_gu": epsilon,
                    "proximity_bound_gu": proximity_bound,
                    "minimum_contact_area_gu2": minimum_contact_area_gu2,
                },
            )
        )

    def order_key(component: ChunkComponent) -> tuple[float, float, str, tuple[str, ...]]:
        return (component.bbox.minimum[0], component.bbox.minimum[1], component.component_type, component.piece_ids)

    output.sort(key=order_key)
    numbered: list[ChunkComponent] = []
    for index, component in enumerate(output, 1):
        numbered.append(
            ChunkComponent(
                component_id=f"c{index:04d}",
                component_type=component.component_type,
                piece_ids=component.piece_ids,
                door_ids=component.door_ids,
                seed_keys=component.seed_keys,
                contact_edges=component.contact_edges,
                bbox=component.bbox,
                flags=component.flags,
                diagnostics=component.diagnostics,
            )
        )
    return numbered


def validate_components(components: Iterable[ChunkComponent]) -> list[ValidationFlag]:
    """Return explicit floating/no-door validator flags for report output."""

    flags: list[ValidationFlag] = []
    for component in sorted(components, key=lambda item: item.component_id):
        if "no_door" in component.flags:
            flags.append(
                ValidationFlag(
                    component.component_id,
                    "no_door",
                    component.piece_ids,
                    "component has no door-seeded core; retain as cluster/wall/yard evidence",
                )
            )
        if "floating_piece" in component.flags:
            flags.append(
                ValidationFlag(
                    component.component_id,
                    "floating_piece",
                    component.piece_ids,
                    "isolated structural piece has no contact path to a door core",
                )
            )
        if "proximity_bound_exceeded_continuous_geometry" in component.flags:
            flags.append(
                ValidationFlag(
                    component.component_id,
                    "proximity_bound_exceeded_continuous_geometry",
                    component.piece_ids,
                    "contact-connected geometry was kept intact rather than auto-split",
                )
            )
    return flags


def sever_contact_component(
    component: ChunkComponent,
    pieces: Mapping[str, ChunkPiece],
    *,
    tie_policy: str = "duplicate",
) -> list[SeveredUnit]:
    """Split a multi-door contact component into door-seeded regions.

    Distances are shortest-path lengths over the component's contact graph,
    not Euclidean distances.  Every piece is assigned to the closest door. If
    two or more door seeds are tied, the default ``duplicate`` policy puts the
    piece in every tied unit so a shared foundation or wall remains complete in
    each render.  ``assign_nearest`` keeps only the lexicographically first
    tied door and is provided for experiments that need disjoint units.

    The returned ``cut_edges`` are the contact edges whose deterministic
    primary owners differ.  Each affected unit receives the edge in its own
    record, while duplicated pieces remain internal to every unit that owns
    them.  The function deliberately performs no size/type decision; callers
    decide which components meet their experiment's over-size criterion.
    """

    if tie_policy not in {"duplicate", "assign_nearest"}:
        raise ValueError("tie_policy must be 'duplicate' or 'assign_nearest'")
    member_ids = tuple(sorted(set(component.piece_ids)))
    if len(member_ids) != len(component.piece_ids):
        raise ValueError(f"component {component.component_id} contains duplicate piece ids")
    missing = [piece_id for piece_id in member_ids if piece_id not in pieces]
    if missing:
        raise ValueError(f"component {component.component_id} has unknown pieces: {missing}")
    door_ids = tuple(sorted(set(component.door_ids)))
    if len(door_ids) < 2:
        raise ValueError("sever_contact_component requires at least two door seeds")
    missing_doors = [door_id for door_id in door_ids if door_id not in pieces]
    if missing_doors:
        raise ValueError(f"component {component.component_id} has unknown doors: {missing_doors}")
    if any(not pieces[door_id].is_door for door_id in door_ids):
        raise ValueError("component door_ids must identify door pieces")

    allowed = set(member_ids)
    adjacency: dict[str, set[str]] = {piece_id: set() for piece_id in member_ids}
    for left, right in component.contact_edges:
        if left not in allowed or right not in allowed:
            raise ValueError(f"component contact edge references a non-member: {(left, right)}")
        if left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)

    # One deterministic BFS per seed keeps the path-distance semantics explicit
    # and avoids relying on insertion order of a dictionary/set.
    distances_by_door: dict[str, dict[str, int]] = {}
    for seed in door_ids:
        distances: dict[str, int] = {seed: 0}
        queue: deque[str] = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency[current]):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        distances_by_door[seed] = distances

    unreachable = [
        piece_id
        for piece_id in member_ids
        if not any(piece_id in distances_by_door[door_id] for door_id in door_ids)
    ]
    if unreachable:
        raise ValueError(
            f"component {component.component_id} has pieces unreachable from every door: {unreachable}"
        )

    primary_owner: dict[str, str] = {}
    all_owners: dict[str, tuple[str, ...]] = {}
    per_unit_pieces: dict[str, set[str]] = {door_id: set() for door_id in door_ids}
    per_unit_distances: dict[str, dict[str, int]] = {door_id: {} for door_id in door_ids}
    duplicated: set[str] = set()
    for piece_id in member_ids:
        candidates = tuple(
            door_id
            for door_id in door_ids
            if piece_id in distances_by_door[door_id]
        )
        best_distance = min(distances_by_door[door_id][piece_id] for door_id in candidates)
        tied = tuple(
            door_id
            for door_id in candidates
            if distances_by_door[door_id][piece_id] == best_distance
        )
        primary = tied[0]
        primary_owner[piece_id] = primary
        owners = tied if tie_policy == "duplicate" else (primary,)
        all_owners[piece_id] = owners
        if len(owners) > 1:
            duplicated.add(piece_id)
        for owner in owners:
            per_unit_pieces[owner].add(piece_id)
            per_unit_distances[owner][piece_id] = best_distance

    cut_edges_by_unit: dict[str, set[tuple[str, str]]] = {door_id: set() for door_id in door_ids}
    normalized_edges = {
        tuple(sorted((left, right)))
        for left, right in component.contact_edges
        if left != right
    }
    for left, right in sorted(normalized_edges):
        left_owner = primary_owner[left]
        right_owner = primary_owner[right]
        if left_owner == right_owner:
            continue
        edge = (left, right)
        cut_edges_by_unit[left_owner].add(edge)
        cut_edges_by_unit[right_owner].add(edge)

    units: list[SeveredUnit] = []
    for door_id in door_ids:
        units.append(
            SeveredUnit(
                seed_door_id=door_id,
                piece_ids=tuple(sorted(per_unit_pieces[door_id])),
                duplicated_piece_ids=tuple(sorted(piece_id for piece_id in duplicated if door_id in all_owners[piece_id])),
                cut_edges=tuple(sorted(cut_edges_by_unit[door_id])),
                distances=per_unit_distances[door_id],
                tie_policy=tie_policy,
            )
        )
    return units


def is_over_size_cluster(component: ChunkComponent, *, max_pieces: int = 40) -> bool:
    """Return the Karthwasten experiment's exact over-size predicate."""

    if max_pieces < 0:
        raise ValueError("max_pieces must be non-negative")
    return (
        component.component_type == "building_cluster"
        and len(component.door_ids) > 1
        and len(component.piece_ids) > max_pieces
    )


def bbox_contact_area(
    left: WorldBBox,
    right: WorldBBox,
    *,
    epsilon: float = DEFAULT_EPSILON_GU,
) -> float:
    """Return the largest AABB contact-face area in GU².

    The contact graph intentionally permits a small gap.  A bbox that only
    grazes another bbox at a point, line, or tiny sliver therefore has a zero
    or very small contact area even though ``WorldBBox.touches`` is true.  The
    largest face projection is used so a normal end-to-end pole contact is
    measured by its cross-section rather than by an arbitrary axis order.
    """

    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a finite non-negative number")
    gaps = left.contact_gaps(right)
    if max(gaps) > epsilon:
        return 0.0
    overlaps = tuple(
        max(min(left.maximum[index], right.maximum[index]) - max(left.minimum[index], right.minimum[index]), 0.0)
        for index in range(3)
    )
    return max(
        overlaps[(axis + 1) % 3] * overlaps[(axis + 2) % 3]
        for axis in range(3)
        if gaps[axis] <= epsilon
    )


def filter_severed_unit_slivers(
    component: ChunkComponent,
    unit: SeveredUnit,
    pieces: Mapping[str, ChunkPiece],
    *,
    min_contact_area_gu2: float = 100.0,
    epsilon: float = DEFAULT_EPSILON_GU,
) -> SeveredUnit:
    """Drop non-door unit pieces whose only graph contacts are marginal.

    This is deliberately a post-severing filter.  It does not rewrite the
    authoritative v2 component or severing graph; it removes a piece only when
    every contact edge retained inside this door unit has AABB contact area
    below ``min_contact_area_gu2``.  Doors are always preserved so the filter
    cannot silently erase a seed.  The removed IDs are carried in the returned
    unit for audit and correction logs.
    """

    threshold = float(min_contact_area_gu2)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("min_contact_area_gu2 must be a finite non-negative number")
    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a finite non-negative number")
    member_ids = set(unit.piece_ids)
    if unit.seed_door_id not in member_ids:
        raise ValueError("severed unit seed door is not in its piece membership")
    if unit.seed_door_id not in pieces or not pieces[unit.seed_door_id].is_door:
        raise ValueError("severed unit seed must identify a door piece")
    adjacency: dict[str, list[tuple[str, float]]] = {piece_id: [] for piece_id in member_ids}
    for left, right in component.contact_edges:
        if left not in member_ids or right not in member_ids or left == right:
            continue
        contact_area = bbox_contact_area(pieces[left].bbox, pieces[right].bbox, epsilon=epsilon)
        adjacency[left].append((right, contact_area))
        adjacency[right].append((left, contact_area))

    removed: set[str] = set()
    for piece_id in sorted(member_ids):
        if piece_id == unit.seed_door_id or pieces[piece_id].is_door:
            continue
        contacts = adjacency.get(piece_id, [])
        if not contacts or max(area for _neighbor, area in contacts) < threshold:
            removed.add(piece_id)

    kept = member_ids - removed
    cut_edges = tuple(
        edge
        for edge in unit.cut_edges
        if edge[0] in kept and edge[1] in kept
    )
    return SeveredUnit(
        seed_door_id=unit.seed_door_id,
        piece_ids=tuple(sorted(kept)),
        duplicated_piece_ids=tuple(sorted(set(unit.duplicated_piece_ids) & kept)),
        cut_edges=tuple(sorted(cut_edges)),
        distances={piece_id: distance for piece_id, distance in unit.distances.items() if piece_id in kept},
        tie_policy=unit.tie_policy,
        removed_sliver_piece_ids=tuple(sorted(set(unit.removed_sliver_piece_ids) | removed)),
    )


def filter_severed_units_slivers(
    component: ChunkComponent,
    units: Iterable[SeveredUnit],
    pieces: Mapping[str, ChunkPiece],
    *,
    min_contact_area_gu2: float = 100.0,
    epsilon: float = DEFAULT_EPSILON_GU,
) -> list[SeveredUnit]:
    """Apply :func:`filter_severed_unit_slivers` in stable seed order."""

    return [
        filter_severed_unit_slivers(
            component,
            unit,
            pieces,
            min_contact_area_gu2=min_contact_area_gu2,
            epsilon=epsilon,
        )
        for unit in sorted(units, key=lambda value: value.seed_door_id)
    ]


def piece_from_mapping(value: Mapping[str, object]) -> ChunkPiece:
    """Construct a piece from a JSON-friendly mapping used by tools/tests."""

    bbox_value = value.get("bbox")
    if not isinstance(bbox_value, Mapping):
        raise ValueError("piece mapping requires bbox with min/max")
    minimum = bbox_value.get("min")
    maximum = bbox_value.get("max")
    if not isinstance(minimum, Sequence) or isinstance(minimum, (str, bytes)) or not isinstance(maximum, Sequence) or isinstance(maximum, (str, bytes)):
        raise ValueError("piece bbox min/max must be sequences")
    source_cell_value = value.get("source_cell")
    source_cell = tuple(int(item) for item in source_cell_value) if isinstance(source_cell_value, Sequence) and len(source_cell_value) == 2 else None
    return ChunkPiece(
        piece_id=str(value.get("piece_id", value.get("id", ""))),
        model=str(value.get("model", value.get("modl", ""))),
        position=_finite_triplet(value.get("position", (0.0, 0.0, 0.0)), "piece.position"),  # type: ignore[arg-type]
        rotation=_finite_triplet(value.get("rotation", (0.0, 0.0, 0.0)), "piece.rotation"),  # type: ignore[arg-type]
        scale=float(value.get("scale", 1.0)),
        bbox=WorldBBox(tuple(float(item) for item in minimum), tuple(float(item) for item in maximum)),  # type: ignore[arg-type]
        record_type=str(value.get("record_type", "STAT")),
        category=str(value.get("category", "exterior")),
        object_id=str(value["object_id"]) if value.get("object_id") is not None else None,
        is_door=bool(value.get("is_door", value.get("record_type") == "DOOR")),
        destination_cell=str(value["destination_cell"]) if value.get("destination_cell") is not None else None,
        structural=bool(value.get("structural", True)),
        source_cell=source_cell,  # type: ignore[arg-type]
        metadata=value,
    )


__all__ = [
    "BARROW_VOCABULARY",
    "CAVE_VOCABULARY",
    "CASTLE_VOCABULARY",
    "COMPONENT_TYPES",
    "ComponentTyping",
    "ChunkComponent",
    "ChunkPiece",
    "bbox_contact_area",
    "SeveredUnit",
    "DEFAULT_BUILDING_MAX_PIECES",
    "DEFAULT_BUILDING_MAX_VOLUME_GU3",
    "DEFAULT_BUILDING_MIN_PIECES",
    "DEFAULT_BUILDING_MIN_VOLUME_GU3",
    "DEFAULT_CITY_WALL_RUN_GU",
    "DEFAULT_EPSILON_GU",
    "DEFAULT_PROXIMITY_BOUND_GU",
    "FORTIFICATION_VOCABULARY",
    "GATE_VOCABULARY",
    "SHELL_VOCABULARY",
    "ValidationFlag",
    "WorldBBox",
    "build_contact_graph",
    "classify_component",
    "chunk_pieces",
    "is_city_wall_piece",
    "is_over_size_cluster",
    "filter_severed_unit_slivers",
    "filter_severed_units_slivers",
    "piece_from_mapping",
    "sever_contact_component",
    "validate_components",
]
