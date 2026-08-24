"""Deterministic Cityforge frontage-fit v1 geometry and composition solver.

Pipeline position
------------------
This pure host-side stage sits between an authored world-GU intent sketch and
``tools/cityforge/plan_sketch.py``.  It loads manifest-pinned, full-precision
stamp hulls/doors, named authored or source targets, and a terrain-mask
protocol; it emits the canonical intent copy, resolved centroid/yaw sketch,
and fit report.  It does not render images, read images, run Blender or
subprocesses, write TES3 records, select stamps semantically, or invent roads,
spaces, districts, roles, targets, or geometry.

The strict intent vocabulary keeps legacy ``kind`` while optionally adding
road ``purpose`` and an urban/service-road
``max_unsupported_frontage_gu`` bound, lot ``intentional_outlier``, and
non-overlapping ``lot_groups``.  Group characters, target/order/sector
declarations, and span, gap, same-side, and non-outlier bounds are authored;
the fitter never infers them.  The optional lot ``frontage_side`` remains a
polyline-only, segment-normal gate, while absent sides retain marker-derived
behavior.  Explicit door targets are always measured against their assigned
target; no nearest-target fallback is used.

For composition-enabled intents, all unary-feasible candidates are retained
for proof rather than treating the rank-best capped prefix as the whole
domain.  Complete MRV/forward-checking passes widen the default 64-candidate
prefix through 128, 256, 512, and the full retained domain (redundant widths
are skipped), with one global node budget.  Complete collision-valid leaves
are evaluated by :mod:`composition_eval`; all nine authored relationship
finding codes are hard gates and rejected leaves continue exhaustive search.
The resulting terminal code distinguishes unary, collision, relationship, and
budget outcomes; budget exhaustion is inconclusive, never an impossibility
proof.

When feasibility succeeds for a composition intent, a separate bounded
improvement walk starts from the hard-valid incumbent.  Its default budget is
50,000 nodes and zero disables the walk.  It searches only the exact domain of
the successful feasibility pass and compares a fixed lexicographic objective;
disabled, exhausted, or faulted improvement retains the solved incumbent and
reports the improvement evidence.  Intents without any composition
declaration keep the legacy one capped pass and do not run improvement.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

from .cityplan import (
    point_in_ring,
    point_seg_distance,
    polygon_centroid,
    ring_area,
    ring_min_distance,
    rings_overlap_exact,
    rot2d_ccw,
)
from .composition_eval import composition_preference_components, evaluate_composition
from .frontage_targets import (
    corridor_rings,
    reach_distance,
    target_nearest_point,
)


FIT_SCHEMA_VERSION = 1
FIT_PRODUCT = "cityforge_frontage_fit_v1"
AMBIGUITY_EPSILON_GU = 1e-6
CONTACT_EPSILON_GU = 1e-9

# These evaluator findings are the authored hard constraints in Section 3B.
# Other evaluator metrics remain observational; availability from
# composition_eval does not silently turn a non-finding into a search gate.
HARD_RELATIONSHIP_FINDINGS = frozenset({
    "road_unsupported_frontage_exceeded",
    "group_shared_target_mismatch",
    "along_order_violation",
    "plaza_sector_unoccupied",
    "group_span_exceeded",
    "group_gap_exceeded",
    "group_gap_unmeasurable",
    "group_same_side_run_exceeded",
    "group_non_outlier_distance_exceeded",
})


class TerrainMask(Protocol):
    """Minimal terrain dependency used by the fitter and synthetic tests."""

    rectangle_gu: Sequence[float]

    def water_at(self, x: float, y: float) -> bool:
        """Return whether a world-GU sample is water."""

    def buildable_at(self, x: float, y: float) -> bool:
        """Return whether a world-GU sample is buildable."""


@dataclass(frozen=True)
class FitConfig:
    """Frozen v1 search constants; serialized verbatim in ``fit_report.json``."""

    along_offsets_gu: tuple[float, ...] = (
        0.0,
        128.0,
        -128.0,
        256.0,
        -256.0,
        384.0,
        -384.0,
        512.0,
        -512.0,
        640.0,
        -640.0,
        768.0,
        -768.0,
        896.0,
        -896.0,
        1024.0,
        -1024.0,
        1152.0,
        -1152.0,
        1280.0,
        -1280.0,
        1408.0,
        -1408.0,
        1536.0,
        -1536.0,
        1664.0,
        -1664.0,
        1792.0,
        -1792.0,
        1920.0,
        -1920.0,
        2048.0,
        -2048.0,
    )
    door_gaps_gu: tuple[float, ...] = (128.0, 256.0, 384.0, 512.0, 640.0)
    yaw_perturbations_deg: tuple[float, ...] = (0.0, 7.5, -7.5, 15.0, -15.0)
    max_candidates_per_lot: int = 64
    raw_reach_limit_gu: float = 768.0
    safety_reach_limit_gu: float = 640.0
    facing_limit_deg: float = 60.0
    reach_safety_margin_gu: float = 128.0
    # Plan §3.1: 0 means unlimited; a positive value caps calls to the
    # recursive search state function.  Serialized verbatim in to_dict.
    search_node_budget: int = 1_000_000
    # Section 4B: a separate bounded traversal after feasibility succeeds.
    # Unlike search_node_budget, zero disables this phase rather than meaning
    # unlimited traversal.  It is not consumed by the feasibility proof.
    improvement_node_budget: int = 50_000

    def __post_init__(self) -> None:
        # Reject bool, negative, and non-integer values with the named error;
        # the acceptance predicate is exactly isinstance(value, int) and
        # not isinstance(value, bool) and value >= 0.
        if not (isinstance(self.search_node_budget, int) and not isinstance(self.search_node_budget, bool)
                and self.search_node_budget >= 0):
            raise FrontageFitError("search_node_budget must be a non-negative integer")
        if not (isinstance(self.improvement_node_budget, int) and not isinstance(self.improvement_node_budget, bool)
                and self.improvement_node_budget >= 0):
            raise FrontageFitError("improvement_node_budget must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "along_offsets_gu": list(self.along_offsets_gu),
            "door_gaps_gu": list(self.door_gaps_gu),
            "yaw_perturbations_deg": list(self.yaw_perturbations_deg),
            "max_candidates_per_lot": self.max_candidates_per_lot,
            "raw_reach_limit_gu": self.raw_reach_limit_gu,
            "safety_reach_limit_gu": self.safety_reach_limit_gu,
            "facing_limit_deg": self.facing_limit_deg,
            "reach_safety_margin_gu": self.reach_safety_margin_gu,
            "search_node_budget": self.search_node_budget,
            "improvement_node_budget": self.improvement_node_budget,
        }


@dataclass(frozen=True)
class _Segment:
    target_id: str
    target: Mapping[str, Any]
    segment_index: int
    start: tuple[float, float]
    end: tuple[float, float]
    cumulative_start: float
    cumulative_end: float
    closed: bool
    ring_area_value: float = 0.0

    @property
    def length(self) -> float:
        return self.cumulative_end - self.cumulative_start


@dataclass(frozen=True)
class DoorGeometry:
    door_id: str
    offset: tuple[float, float]
    heading_deg: float


@dataclass(frozen=True)
class Candidate:
    ordinal: int
    centroid: tuple[float, float]
    anchor: tuple[float, float]
    yaw_deg: float
    hull: tuple[tuple[float, float], ...]
    primary_door_position: tuple[float, float]
    along_offset_gu: float
    door_gap_gu: float
    yaw_perturbation_deg: float
    primary_target_id: str
    target_arc_gu: float
    target_length_gu: float
    frontage_side: str | None
    plaza_angle_deg: float | None
    door_reports: tuple[dict[str, Any], ...]

    @property
    def rank(self) -> tuple[Any, ...]:
        return (
            self._marker_displacement_sq,
            abs(self.along_offset_gu),
            abs(self.door_gap_gu - 256.0),
            abs(self.yaw_perturbation_deg),
            round(self.centroid[0], 1),
            round(self.centroid[1], 1),
            round(self.yaw_deg, 1),
        )

    _marker_displacement_sq: float = field(default=0.0, compare=False)


def _candidate_composition_fact(
    lot: Mapping[str, Any], candidate: Candidate
) -> dict[str, Any]:
    """Adapt one selected candidate to the evaluator's exact fact contract.

    Intent owns the two authored flags; all measured geometry/projection values
    come from the immutable candidate.  Keeping this adapter separate avoids
    making the observational evaluator depend on the fitter's domain types.
    """
    return {
        "lot_id": str(lot["id"]),
        "centroid": [float(candidate.centroid[0]), float(candidate.centroid[1])],
        "intentional_outlier": lot.get("intentional_outlier", False),
        "primary_target_id": candidate.primary_target_id,
        "target_arc_gu": candidate.target_arc_gu,
        "target_length_gu": candidate.target_length_gu,
        "frontage_side": lot.get("frontage_side"),
        "plaza_angle_deg": candidate.plaza_angle_deg,
    }


def _evaluate_selected_composition(
    intent: Mapping[str, Any],
    ordered: Sequence[_LotCandidates],
    chosen: Sequence[tuple[str, Candidate]] | Mapping[str, Candidate],
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    """Evaluate one collision-valid complete assignment.

    The callback is deliberately at the fit/search boundary: candidate facts
    are authored by the fitter, while relationship meaning remains owned by
    :mod:`composition_eval`.  Only the Section 3B hard finding codes are
    returned to the search as rejection evidence; other evaluator findings are
    observational and never become implicit solver constraints.
    """
    if isinstance(chosen, Mapping):
        chosen_by_id = {str(lot_id): candidate for lot_id, candidate in chosen.items()}
    else:
        chosen_by_id = {str(lot_id): candidate for lot_id, candidate in chosen}
    facts: list[dict[str, Any]] = []
    for item in sorted(ordered, key=lambda row: row.lot["id"]):
        lot_id = str(item.lot["id"])
        candidate = chosen_by_id.get(lot_id)
        if candidate is None:
            raise FrontageFitError(
                f"complete composition assignment is missing lot {lot_id!r}")
        facts.append(_candidate_composition_fact(item.lot, candidate))
    composition = evaluate_composition(intent, facts)
    hard_findings = tuple(
        finding for finding in composition["findings"]
        if finding.get("code") in HARD_RELATIONSHIP_FINDINGS
    )
    return composition, hard_findings


def _assignment_preference(
    intent: Mapping[str, Any],
    composition: Mapping[str, Any],
    selected: Sequence[tuple[str, Candidate]] | Mapping[str, Candidate],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return the exact Section 4B objective and JSON-ready components.

    The first four fields are the fixed observational components owned by
    :mod:`composition_eval`.  Their lists are already descending profiles, so
    ordinary tuple minimization prefers the smaller worst value first.  The
    The final two fields belong to frontage-fit: marker displacement is rounded
    once at the objective boundary, and the canonical lot-id/ordinal signature
    makes a complete tie deterministic without a weighted score.
    """
    preference = composition_preference_components(intent, composition)
    if isinstance(selected, Mapping):
        rows = [(str(lot_id), candidate) for lot_id, candidate in selected.items()]
    else:
        rows = [(str(lot_id), candidate) for lot_id, candidate in selected]
    rows.sort(key=lambda row: row[0])
    signature = tuple((lot_id, int(candidate.ordinal)) for lot_id, candidate in rows)
    marker_displacement_sq = round(
        sum(float(candidate._marker_displacement_sq) for _, candidate in rows), 6)
    components: dict[str, Any] = {
        "urban_unsupported_profile_gu": list(
            preference["urban_unsupported_profile_gu"]),
        "compact_span_profile_gu": list(preference["compact_span_profile_gu"]),
        "compact_gap_profile_gu": list(preference["compact_gap_profile_gu"]),
        "irregular_repeated_gap_pairs": int(
            preference["irregular_repeated_gap_pairs"]),
        "marker_displacement_sq": marker_displacement_sq,
        "assignment_signature": [[lot_id, ordinal] for lot_id, ordinal in signature],
    }
    objective = (
        tuple(components["urban_unsupported_profile_gu"]),
        tuple(components["compact_span_profile_gu"]),
        tuple(components["compact_gap_profile_gu"]),
        components["irregular_repeated_gap_pairs"],
        components["marker_displacement_sq"],
        signature,
    )
    return objective, components


@dataclass(frozen=True)
class _ConstructedCandidate:
    """Transform-only candidate draft, produced before unary evaluation.

    Splitting construction from evaluation lets the generator deduplicate by
    the plan §6.3 rounded ``(centroid_x, centroid_y, yaw)`` key *before* the
    unary gates run, so a repeated key is never re-evaluated and the report's
    ``deduplicated`` count means "unique constructed keys" (plan §7 wording),
    not "unique feasible keys".
    """

    centroid: tuple[float, float]
    anchor: tuple[float, float]
    yaw_deg: float
    hull: tuple[tuple[float, float], ...]
    primary_door_position: tuple[float, float]
    along_offset_gu: float
    door_gap_gu: float
    yaw_perturbation_deg: float
    primary_target_id: str
    target_arc_gu: float
    target_length_gu: float
    frontage_side: str | None
    plaza_angle_deg: float | None
    marker_displacement_sq: float


@dataclass
class _LotCandidates:
    lot: dict[str, Any]
    doors: list[DoorGeometry]
    primary: DoorGeometry | None
    generated: int = 0
    deduplicated: int = 0
    unary_feasible: list[Candidate] = field(default_factory=list)
    all_unary_feasible: list[Candidate] = field(default_factory=list)
    rejections: Counter[str] = field(default_factory=Counter)
    frontage_error: str | None = None
    marker_projection: dict[str, Any] | None = None


@dataclass(frozen=True)
class _SearchStats:
    """Deterministic counters for one complete-search run (plan §3.2)."""

    nodes: int
    extensions: int
    compatibility_checks: int
    backtracks: int
    budget_exhausted: bool
    complete_assignments_checked: int = 0
    relationship_rejections: Mapping[str, int] = field(default_factory=dict)
    collision_valid_assignments: int = 0


@dataclass(frozen=True)
class _SearchResult:
    """Complete-search outcome (plan §3.2).

    ``selected is None`` means either exhaustive infeasibility or budget
    exhaustion; ``stats.budget_exhausted`` distinguishes them.  Selected rows
    are in canonical lot order (plan §3.3), never recursion-assignment order.
    """

    selected: tuple[tuple[str, Candidate], ...] | None
    stats: _SearchStats
    composition: dict[str, Any] | None = None


@dataclass(frozen=True)
class _ImprovementStats:
    """Frozen counters for one bounded post-feasibility traversal.

    ``domain_sizes`` is a canonical ``(lot_id, count)`` tuple rather than a
    mutable mapping.  The matrix is supplied by the successful hard-search
    pass, so compatibility-pair construction is deliberately not counted by
    this phase.
    """

    nodes: int
    extensions: int
    collision_valid_assignments: int
    hard_valid_assignments: int
    relationship_rejections: Mapping[str, int]
    budget_exhausted: bool
    incumbent_improved: bool
    domain_sizes: tuple[tuple[str, int], ...]

    @property
    def collision_valid_leaves(self) -> int:
        """Compatibility alias using the plan's leaf terminology."""
        return self.collision_valid_assignments

    @property
    def complete_assignments_checked(self) -> int:
        """All collision-valid leaves are composition-evaluation leaves."""
        return self.collision_valid_assignments

    @property
    def hard_valid_leaves(self) -> int:
        """Compatibility alias using the plan's leaf terminology."""
        return self.hard_valid_assignments


@dataclass(frozen=True)
class _ImprovementResult:
    """Best assignment found by the bounded improvement traversal."""

    selected: tuple[tuple[str, Candidate], ...]
    composition: dict[str, Any] | None
    objective: tuple[Any, ...]
    objective_components: dict[str, Any]
    stats: _ImprovementStats

    @property
    def components(self) -> dict[str, Any]:
        """Short alias for callers that use the plan's "components" wording."""
        return self.objective_components


class _SearchBudgetExhausted(Exception):
    """Private control flow: the node budget ran out mid-search (plan §5).

    Raised from the recursive state function at state entry and caught exactly
    once at the search boundary; never inside a branch.  The caller must treat
    it as inconclusive, not as proof of unsatisfiability.
    """


class _ImprovementBudgetExhausted(Exception):
    """Private control flow for the independent bounded improvement walk."""


class FrontageFitError(ValueError):
    """Schema/geometry input error; the CLI prints it as a frontage_fit failure."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _pair(value: Any, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2 or not all(_finite(item) for item in value):
        raise FrontageFitError(f"{path} must be a finite [x, y] pair")
    return [float(value[0]), float(value[1])]


def _points(value: Any, minimum: int, path: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < minimum:
        raise FrontageFitError(f"{path} must contain at least {minimum} points")
    return [_pair(point, f"{path}[{index}]") for index, point in enumerate(value)]


def _require_keys(record: Mapping[str, Any], allowed: set[str], required: set[str], path: str) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise FrontageFitError(f"{path} has unknown keys {unknown}")
    missing = sorted(required - set(record))
    if missing:
        raise FrontageFitError(f"{path} is missing required keys {missing}")


def _validate_road(value: Any, index: int, seen: set[str]) -> dict[str, Any]:
    path = f"$.roads[{index}]"
    if not isinstance(value, dict):
        raise FrontageFitError(f"{path} must be an object")
    _require_keys(value, {"id", "kind", "width_gu", "points", "purpose", "max_unsupported_frontage_gu"},
                  {"id", "kind", "width_gu", "points"}, path)
    road_id = value["id"]
    if not isinstance(road_id, str) or not road_id:
        raise FrontageFitError(f"{path}.id must be a non-empty string")
    if road_id in seen:
        raise FrontageFitError(f"duplicate road id {road_id!r}")
    seen.add(road_id)
    kind = value["kind"]
    if kind not in ("street", "alley"):
        raise FrontageFitError(f"{path}.kind must be 'street' or 'alley'")
    width = value["width_gu"]
    low, high = (256.0, 1024.0) if kind == "street" else (128.0, 512.0)
    if not _finite(width) or not low <= float(width) <= high:
        raise FrontageFitError(f"{path}.width_gu must be {low:g}-{high:g} GU")
    normalized: dict[str, Any] = {
        "id": road_id,
        "kind": kind,
        "width_gu": float(width),
        "points": _points(value["points"], 2, f"{path}.points"),
    }
    if "purpose" in value:
        purpose = value["purpose"]
        if purpose not in ("urban_street", "service_lane", "connector"):
            raise FrontageFitError(
                f"{path}.purpose must be 'urban_street', 'service_lane', or 'connector'")
        normalized["purpose"] = purpose
    if "max_unsupported_frontage_gu" in value:
        if value.get("purpose") not in ("urban_street", "service_lane"):
            raise FrontageFitError(
                f"{path}.max_unsupported_frontage_gu requires purpose "
                "'urban_street' or 'service_lane'")
        maximum = value["max_unsupported_frontage_gu"]
        if not _finite(maximum) or float(maximum) < 0.0:
            raise FrontageFitError(
                f"{path}.max_unsupported_frontage_gu must be a finite non-negative number")
        normalized["max_unsupported_frontage_gu"] = float(maximum)
    return normalized


def _validate_space(value: Any, index: int, seen: set[str]) -> dict[str, Any]:
    path = f"$.spaces[{index}]"
    if not isinstance(value, dict):
        raise FrontageFitError(f"{path} must be an object")
    _require_keys(value, {"id", "kind", "polygon"}, {"id", "kind", "polygon"}, path)
    space_id = value["id"]
    if not isinstance(space_id, str) or not space_id:
        raise FrontageFitError(f"{path}.id must be a non-empty string")
    if space_id in seen:
        raise FrontageFitError(f"duplicate space id {space_id!r}")
    seen.add(space_id)
    if value["kind"] not in ("plaza", "court"):
        raise FrontageFitError(f"{path}.kind must be 'plaza' or 'court'")
    polygon = _points(value["polygon"], 3, f"{path}.polygon")
    if abs(ring_area(polygon)) <= AMBIGUITY_EPSILON_GU:
        raise FrontageFitError(f"{path}.polygon has zero area")
    return {"id": space_id, "kind": value["kind"], "polygon": polygon}


def _validate_district(value: Any, index: int, seen: set[str]) -> dict[str, Any]:
    path = f"$.districts[{index}]"
    if not isinstance(value, dict):
        raise FrontageFitError(f"{path} must be an object")
    _require_keys(value, {"id", "polygon", "character"}, {"id", "polygon", "character"}, path)
    district_id = value["id"]
    if not isinstance(district_id, str) or not district_id:
        raise FrontageFitError(f"{path}.id must be a non-empty string")
    if district_id in seen:
        raise FrontageFitError(f"duplicate district id {district_id!r}")
    seen.add(district_id)
    if not isinstance(value["character"], str):
        raise FrontageFitError(f"{path}.character must be a string")
    polygon = _points(value["polygon"], 3, f"{path}.polygon")
    if abs(ring_area(polygon)) <= AMBIGUITY_EPSILON_GU:
        raise FrontageFitError(f"{path}.polygon has zero area")
    return {"id": district_id, "polygon": polygon, "character": value["character"]}


def _validate_lot_shape(
    value: Any,
    index: int,
    seen: set[str],
    stamp_ids: set[str],
    districts: Mapping[str, Mapping[str, Any]],
    site_rect: Sequence[float],
) -> dict[str, Any]:
    path = f"$.lots[{index}]"
    if not isinstance(value, dict):
        raise FrontageFitError(f"{path} must be an object")
    _require_keys(value, {"id", "district", "role", "stamp", "marker", "frontages", "note", "frontage_side",
                           "intentional_outlier"},
                  {"id", "district", "role", "stamp", "marker", "frontages"}, path)
    lot_id = value["id"]
    if not isinstance(lot_id, str) or not lot_id:
        raise FrontageFitError(f"{path}.id must be a non-empty string")
    if lot_id in seen:
        raise FrontageFitError(f"duplicate lot id {lot_id!r}")
    seen.add(lot_id)
    frontage_side = value.get("frontage_side")
    if frontage_side is not None and frontage_side not in ("left", "right"):
        raise FrontageFitError(f"{path}.frontage_side must be 'left' or 'right'")
    district = value["district"]
    if not isinstance(district, str) or district not in districts:
        raise FrontageFitError(f"{path}.district {district!r} is not declared")
    if not isinstance(value["role"], str):
        raise FrontageFitError(f"{path}.role must be a string")
    stamp = value["stamp"]
    if not isinstance(stamp, str) or stamp not in stamp_ids:
        raise FrontageFitError(f"{path}.stamp {stamp!r} is not eligible in bundle stamps.json")
    marker = _pair(value["marker"], f"{path}.marker")
    if not (float(site_rect[0]) <= marker[0] <= float(site_rect[2]) and
            float(site_rect[1]) <= marker[1] <= float(site_rect[3])):
        raise FrontageFitError(f"{path}.marker lies outside site rectangle_gu")
    if not point_in_ring((marker[0], marker[1]), districts[district]["polygon"]):
        raise FrontageFitError(f"{path}.marker lies outside district {district!r}")
    frontages = value["frontages"]
    if not isinstance(frontages, list) or not frontages:
        raise FrontageFitError(f"{path}.frontages must be a non-empty array")
    primary_count = 0
    seen_doors: set[str] = set()
    normalized_frontages: list[dict[str, Any]] = []
    for f_index, frontage in enumerate(frontages):
        fpath = f"{path}.frontages[{f_index}]"
        if not isinstance(frontage, dict):
            raise FrontageFitError(f"{fpath} must be an object")
        _require_keys(frontage, {"door_id", "target_id", "intent", "primary"},
                      {"door_id", "target_id", "intent", "primary"}, fpath)
        door_id = frontage["door_id"]
        target_id = frontage["target_id"]
        if not isinstance(door_id, str) or not door_id:
            raise FrontageFitError(f"{fpath}.door_id must be a non-empty string")
        if door_id in seen_doors:
            raise FrontageFitError(f"{path} declares door {door_id!r} more than once")
        seen_doors.add(door_id)
        if not isinstance(target_id, str) or not target_id:
            raise FrontageFitError(f"{fpath}.target_id must be a non-empty string")
        if frontage["intent"] not in ("public", "service"):
            raise FrontageFitError(f"{fpath}.intent must be 'public' or 'service'")
        if not isinstance(frontage["primary"], bool):
            raise FrontageFitError(f"{fpath}.primary must be boolean")
        primary_count += int(frontage["primary"])
        normalized_frontages.append({"door_id": door_id, "target_id": target_id,
                                     "intent": frontage["intent"],
                                     "primary": frontage["primary"]})
    if primary_count != 1:
        raise FrontageFitError(f"{path}.frontages must contain exactly one primary door")
    normalized: dict[str, Any] = {
        "id": lot_id,
        "district": district,
        "role": value["role"],
        "stamp": stamp,
        "marker": marker,
        "frontages": normalized_frontages,
    }
    if frontage_side is not None:
        normalized["frontage_side"] = frontage_side
    if "note" in value:
        if not isinstance(value["note"], str):
            raise FrontageFitError(f"{path}.note must be a string")
        normalized["note"] = value["note"]
    if "intentional_outlier" in value:
        if not isinstance(value["intentional_outlier"], bool):
            raise FrontageFitError(f"{path}.intentional_outlier must be boolean")
        normalized["intentional_outlier"] = value["intentional_outlier"]
    return normalized


def _validate_lot_group(
    value: Any,
    index: int,
    declared_lot_ids: set[str],
    seen_group_ids: set[str],
    assigned_lot_ids: set[str],
) -> dict[str, Any]:
    path = f"$.lot_groups[{index}]"
    if not isinstance(value, dict):
        raise FrontageFitError(f"{path} must be an object")
    _require_keys(
        value,
        {"id", "character", "lot_ids", "shared_target_id", "max_span_gu",
          "max_consecutive_gap_gu", "max_non_outlier_distance_gu",
          "max_consecutive_same_side", "along_order", "plaza_sectors"},
        {"id", "character", "lot_ids"},
        path,
    )
    group_id = value["id"]
    if not isinstance(group_id, str) or not group_id:
        raise FrontageFitError(f"{path}.id must be a non-empty string")
    if group_id in seen_group_ids:
        raise FrontageFitError(f"duplicate lot group id {group_id!r}")
    seen_group_ids.add(group_id)

    character = value["character"]
    characters = ("compact_cluster", "irregular_two_sided", "formal_square",
                  "gateway_cluster", "sparse_outskirts")
    if character not in characters:
        raise FrontageFitError(
            f"{path}.character must be 'compact_cluster', 'irregular_two_sided', "
            "'formal_square', 'gateway_cluster', or 'sparse_outskirts'")

    raw_lot_ids = value["lot_ids"]
    if not isinstance(raw_lot_ids, list) or not raw_lot_ids:
        raise FrontageFitError(f"{path}.lot_ids must be a non-empty array")
    lot_ids: list[str] = []
    local_lot_ids: set[str] = set()
    for lot_id in raw_lot_ids:
        if not isinstance(lot_id, str) or not lot_id:
            raise FrontageFitError(f"{path}.lot_ids must contain non-empty strings")
        if lot_id in local_lot_ids:
            raise FrontageFitError(f"{path}.lot_ids contains duplicate lot id {lot_id!r}")
        if lot_id not in declared_lot_ids:
            raise FrontageFitError(f"{path}.lot_ids references undeclared lot {lot_id!r}")
        local_lot_ids.add(lot_id)
        lot_ids.append(lot_id)
    overlap = sorted(local_lot_ids & assigned_lot_ids)
    if overlap:
        raise FrontageFitError(f"{path}.lot_ids overlaps another group on lots {overlap}")
    assigned_lot_ids.update(local_lot_ids)

    normalized: dict[str, Any] = {
        "id": group_id,
        "character": character,
        "lot_ids": sorted(lot_ids),
    }
    if "shared_target_id" in value:
        shared_target_id = value["shared_target_id"]
        if not isinstance(shared_target_id, str) or not shared_target_id:
            raise FrontageFitError(f"{path}.shared_target_id must be a non-empty string")
        normalized["shared_target_id"] = shared_target_id

    for key in ("max_span_gu", "max_consecutive_gap_gu", "max_non_outlier_distance_gu"):
        if key not in value:
            continue
        bound = value[key]
        if not _finite(bound) or float(bound) < 0.0:
            raise FrontageFitError(f"{path}.{key} must be a finite non-negative number")
        normalized[key] = float(bound)

    if "max_consecutive_same_side" in value:
        side_run = value["max_consecutive_same_side"]
        if not (isinstance(side_run, int) and not isinstance(side_run, bool) and side_run >= 1):
            raise FrontageFitError(
                f"{path}.max_consecutive_same_side must be a positive integer")
        normalized["max_consecutive_same_side"] = side_run

    if "along_order" in value:
        along_order = value["along_order"]
        if not isinstance(along_order, list) or not along_order:
            raise FrontageFitError(f"{path}.along_order must be a non-empty array")
        seen_order: set[str] = set()
        for lot_id in along_order:
            if not isinstance(lot_id, str) or not lot_id:
                raise FrontageFitError(f"{path}.along_order must contain non-empty strings")
            if lot_id in seen_order:
                raise FrontageFitError(f"{path}.along_order contains duplicate lot id {lot_id!r}")
            seen_order.add(lot_id)
        if seen_order != local_lot_ids:
            missing = sorted(local_lot_ids - seen_order)
            extra = sorted(seen_order - local_lot_ids)
            raise FrontageFitError(
                f"{path}.along_order must be a permutation of lot_ids "
                f"(missing {missing}, extra {extra})")
        normalized["along_order"] = list(along_order)

    if "plaza_sectors" in value:
        if character != "formal_square":
            raise FrontageFitError(f"{path}.plaza_sectors requires character 'formal_square'")
        if "shared_target_id" not in value:
            raise FrontageFitError(f"{path}.plaza_sectors requires shared_target_id")
        raw_sectors = value["plaza_sectors"]
        if not isinstance(raw_sectors, list) or not raw_sectors:
            raise FrontageFitError(f"{path}.plaza_sectors must be a non-empty array")
        sector_seen: set[str] = set()
        sectors: list[dict[str, Any]] = []
        for sector_index, sector in enumerate(raw_sectors):
            sector_path = f"{path}.plaza_sectors[{sector_index}]"
            if not isinstance(sector, dict):
                raise FrontageFitError(f"{sector_path} must be an object")
            _require_keys(sector, {"id", "start_deg", "end_deg"},
                          {"id", "start_deg", "end_deg"}, sector_path)
            sector_id = sector["id"]
            if not isinstance(sector_id, str) or not sector_id:
                raise FrontageFitError(f"{sector_path}.id must be a non-empty string")
            if sector_id in sector_seen:
                raise FrontageFitError(f"{path}.plaza_sectors contains duplicate id {sector_id!r}")
            sector_seen.add(sector_id)
            start_deg = sector["start_deg"]
            end_deg = sector["end_deg"]
            if not _finite(start_deg) or not _finite(end_deg):
                raise FrontageFitError(
                    f"{sector_path}.start_deg and end_deg must be finite numbers")
            start = float(start_deg)
            end = float(end_deg)
            if not (0.0 <= start < end <= 360.0):
                raise FrontageFitError(
                    f"{sector_path} must satisfy 0 <= start_deg < end_deg <= 360")
            sectors.append({"id": sector_id, "start_deg": start, "end_deg": end})
        sectors.sort(key=lambda sector: sector["id"])
        normalized["plaza_sectors"] = sectors
    return normalized


def validate_intent(
    intent: Any,
    *,
    site_name: str,
    stamp_ids: set[str],
    site_rect: Sequence[float],
) -> dict[str, Any]:
    """Strictly validate and normalize authored intent shape.

    Target existence and door identity need the loaded bundle geometry/target
    map and are therefore checked by :func:`fit_intent` after this structural
    gate.  This function intentionally does not invent a target fallback.
    """

    if not isinstance(intent, dict):
        raise FrontageFitError("intent must be a JSON object")
    _require_keys(intent, {"schema_version", "site", "districts", "roads", "spaces", "lots", "notes", "lot_groups"},
                  {"schema_version", "site", "districts", "roads", "spaces", "lots", "notes"}, "$")
    if intent["schema_version"] != FIT_SCHEMA_VERSION or isinstance(intent["schema_version"], bool):
        raise FrontageFitError("$.schema_version must be integer 1")
    if intent["site"] != site_name:
        raise FrontageFitError(f"$.site {intent['site']!r} does not match bundle site {site_name!r}")
    for key in ("districts", "roads", "spaces", "lots"):
        if not isinstance(intent[key], list):
            raise FrontageFitError(f"$.{key} must be an array")
    if not isinstance(intent["notes"], str):
        raise FrontageFitError("$.notes must be a string")
    district_seen: set[str] = set()
    districts = [_validate_district(row, index, district_seen) for index, row in enumerate(intent["districts"])]
    district_map = {row["id"]: row for row in districts}
    road_seen: set[str] = set()
    space_seen: set[str] = set()
    roads = [_validate_road(row, index, road_seen) for index, row in enumerate(intent["roads"])]
    spaces = [_validate_space(row, index, space_seen) for index, row in enumerate(intent["spaces"])]
    if not intent["lots"]:
        raise FrontageFitError("$.lots must be non-empty")
    lot_seen: set[str] = set()
    lots = [_validate_lot_shape(row, index, lot_seen, stamp_ids, district_map, site_rect)
            for index, row in enumerate(intent["lots"])]
    # Canonical lot order makes reversing authored lot order byte-identical.
    lots.sort(key=lambda row: row["id"])
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "site": site_name,
        "districts": districts,
        "roads": roads,
        "spaces": spaces,
        "lots": lots,
        "notes": intent["notes"],
    }
    if "lot_groups" in intent:
        if not isinstance(intent["lot_groups"], list):
            raise FrontageFitError("$.lot_groups must be an array")
        group_seen: set[str] = set()
        assigned_lot_ids: set[str] = set()
        groups = [
            _validate_lot_group(row, index, lot_seen, group_seen, assigned_lot_ids)
            for index, row in enumerate(intent["lot_groups"])
        ]
        groups.sort(key=lambda row: row["id"])
        normalized["lot_groups"] = groups
    return normalized


def _stamp_hull(stamp: Mapping[str, Any]) -> list[tuple[float, float]]:
    raw = stamp.get("footprint", {}).get("hull_xy_rel")
    if raw is None:
        raw = stamp.get("hull")
    if not isinstance(raw, list) or len(raw) < 3:
        raise FrontageFitError("selected stamp has no usable full-precision hull")
    return [(float(point[0]), float(point[1])) for point in raw]


def _stamp_doors(stamp: Mapping[str, Any]) -> list[DoorGeometry]:
    rows: list[DoorGeometry] = []
    members = stamp.get("members")
    if isinstance(members, list):
        for member in members:
            if not isinstance(member, Mapping) or not member.get("is_door"):
                continue
            door_id = member.get("source_id")
            offset = member.get("offset_gu", [0.0, 0.0, 0.0])
            heading = member.get("outward_heading_deg")
            if heading is None:
                heading = member.get("heading_deg")
            if heading is None:
                rotation = member.get("rotation", [0.0, 0.0, 0.0])
                heading = math.degrees(float(rotation[2])) if len(rotation) >= 3 else None
            if not isinstance(door_id, str) or not door_id or not isinstance(offset, Sequence) or len(offset) < 2 or not _finite(heading):
                raise FrontageFitError("selected stamp has an unresolved door member")
            rows.append(DoorGeometry(door_id, (float(offset[0]), float(offset[1])), float(heading) % 360.0))
    else:
        for member in stamp.get("doors", []):
            if not isinstance(member, Mapping):
                raise FrontageFitError("selected stamp has an invalid compact door row")
            door_id = member.get("door_id")
            if not isinstance(door_id, str) or not door_id:
                raise FrontageFitError("selected stamp has a door without stable door_id")
            rows.append(DoorGeometry(door_id, (float(member.get("dx_gu", 0.0)), float(member.get("dy_gu", 0.0))),
                                     float(member.get("heading_deg", 0.0)) % 360.0))
    rows.sort(key=lambda door: (door.offset[0], door.offset[1], door.door_id))
    if not rows:
        raise FrontageFitError("selected stamp has no resolved doors")
    return rows


def _target_segments(target_id: str, target: Mapping[str, Any]) -> list[_Segment]:
    polyline = target.get("polyline")
    if isinstance(polyline, list) and len(polyline) >= 2:
        closed = False
        points = [(float(point[0]), float(point[1])) for point in polyline]
        ring_value = 0.0
    else:
        polygon = target.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            return []
        points = [(float(point[0]), float(point[1])) for point in polygon]
        closed = True
        ring_value = ring_area([list(point) for point in points])
    segments: list[_Segment] = []
    cumulative = 0.0
    count = len(points) if closed else len(points) - 1
    for index in range(count):
        start, end = points[index], points[(index + 1) % len(points)]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length <= AMBIGUITY_EPSILON_GU:
            continue
        segments.append(_Segment(target_id, target, index, start, end, cumulative,
                                 cumulative + length, closed, ring_value))
        cumulative += length
    return segments


def _project_marker(marker: Sequence[float], target_id: str, target: Mapping[str, Any]) -> tuple[_Segment, float, tuple[float, float]] | None:
    best: tuple[float, int, _Segment, float, tuple[float, float]] | None = None
    for segment in _target_segments(target_id, target):
        ax, ay = segment.start
        bx, by = segment.end
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((float(marker[0]) - ax) * dx + (float(marker[1]) - ay) * dy) / length_sq))
        projection = (ax + t * dx, ay + t * dy)
        distance = math.hypot(float(marker[0]) - projection[0], float(marker[1]) - projection[1])
        candidate = (distance, segment.segment_index, segment, segment.cumulative_start + t * segment.length, projection)
        if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
            best = candidate
    if best is None:
        return None
    return best[2], best[3], best[4]


def _sample_arc(
    segments: Sequence[_Segment], arc: float
) -> tuple[_Segment, tuple[float, float], tuple[float, float], float]:
    total = segments[-1].cumulative_end
    closed = segments[0].closed
    if closed:
        canonical_arc = arc % total
    else:
        canonical_arc = max(0.0, min(total, arc))
    selected = segments[-1]
    for segment in segments:
        if canonical_arc < segment.cumulative_end or (
                canonical_arc == segment.cumulative_end and segment is segments[-1]):
            selected = segment
            break
    t = max(0.0, min(1.0, (canonical_arc - selected.cumulative_start) / selected.length))
    point = (selected.start[0] + (selected.end[0] - selected.start[0]) * t,
             selected.start[1] + (selected.end[1] - selected.start[1]) * t)
    tangent_length = selected.length
    tangent = ((selected.end[0] - selected.start[0]) / tangent_length,
               (selected.end[1] - selected.start[1]) / tangent_length)
    return selected, point, tangent, canonical_arc


def _point_on_polygon_boundary(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    closed = list(polygon) + [list(polygon[0])]
    return any(point_seg_distance(point, first, second) <= AMBIGUITY_EPSILON_GU
               for first, second in zip(closed, closed[1:]))


def _angle_deviation(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _transform_hull(hull: Sequence[Sequence[float]], anchor: Sequence[float], yaw: float) -> tuple[tuple[float, float], ...]:
    return tuple((float(anchor[0]) + rotated[0], float(anchor[1]) + rotated[1])
                 for rotated in (rot2d_ccw(float(point[0]), float(point[1]), yaw) for point in hull))


def _rings_conflict(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> bool:
    return rings_overlap_exact([list(point) for point in first], [list(point) for point in second]) or \
        ring_min_distance([list(point) for point in first], [list(point) for point in second]) <= CONTACT_EPSILON_GU


def _ring_contains_or_conflicts(
    hull: Sequence[Sequence[float]], polygon: Sequence[Sequence[float]]
) -> bool:
    if _rings_conflict(hull, polygon):
        return True
    return (any(point_in_ring((float(point[0]), float(point[1])), list(polygon)) for point in hull) or
            any(point_in_ring((float(point[0]), float(point[1])), list(hull)) for point in polygon))


def _world_target_rows(targets: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(key): dict(value) for key, value in targets.items()}


def _candidate_door_reports(
    lot: Mapping[str, Any],
    doors: Sequence[DoorGeometry],
    frontage_by_door: Mapping[str, Mapping[str, Any]],
    anchor: Sequence[float],
    yaw: float,
    targets: Mapping[str, Mapping[str, Any]],
    config: FitConfig,
) -> tuple[tuple[dict[str, Any], ...], str | None]:
    reports: list[dict[str, Any]] = []
    for door in doors:
        frontage = frontage_by_door.get(door.door_id)
        if frontage is None:
            continue
        offset = rot2d_ccw(door.offset[0], door.offset[1], yaw)
        point = (float(anchor[0]) + offset[0], float(anchor[1]) + offset[1])
        target_id = str(frontage["target_id"])
        target = targets.get(target_id)
        if target is None:
            return (), "target_unresolved"
        raw_reach = reach_distance(point, target)
        safety_distance, _ = target_nearest_point(point, target, path_edge=True)
        _, target_point = target_nearest_point(point, target)
        heading = (door.heading_deg + yaw) % 360.0
        facing = None
        if target_point is not None:
            if safety_distance <= AMBIGUITY_EPSILON_GU:
                facing = 0.0
            else:
                desired = math.degrees(math.atan2(target_point[1] - point[1], target_point[0] - point[0]))
                facing = _angle_deviation(heading, desired)
        row = {
            "door_id": door.door_id,
            "target_id": target_id,
            "target_kind": str(target.get("kind", "")),
            "intent": str(frontage["intent"]),
            "raw_reach_gu": round(raw_reach, 6),
            "path_edge_distance_gu": round(safety_distance, 6),
            "safety_limit_gu": config.safety_reach_limit_gu,
            "heading_deg": round(heading % 360.0, 6),
            "facing_deviation_deg": None if facing is None else round(facing, 6),
            "raw_reach_limit_gu": config.raw_reach_limit_gu,
        }
        # The limits are "at or below" bounds (plan §6.2), but the measured
        # values carry ~1e-10 GU float noise from hypot/nearest-point against
        # the polyline, so a door constructed exactly at the 640-GU gap can
        # measure 640.0000000001.  Tolerate up to the 1e-6 ambiguity epsilon
        # above each limit while still rejecting any value materially over.
        if not math.isfinite(raw_reach) or raw_reach > config.raw_reach_limit_gu + AMBIGUITY_EPSILON_GU:
            return tuple(reports + [row]), "assigned_door_raw_reach"
        if not math.isfinite(safety_distance) or safety_distance > config.safety_reach_limit_gu + AMBIGUITY_EPSILON_GU:
            return tuple(reports + [row]), "assigned_door_safety_reach"
        if facing is None:
            return tuple(reports + [row]), "assigned_target_projection_unresolved"
        if facing > config.facing_limit_deg + AMBIGUITY_EPSILON_GU:
            return tuple(reports + [row]), "assigned_door_facing"
        reports.append(row)
    return tuple(reports), None


def _validate_lot_geometry(
    lot: Mapping[str, Any],
    stamp: Mapping[str, Any],
    doors: Sequence[DoorGeometry],
    hull: Sequence[Sequence[float]],
    marker: Sequence[float],
    centroid: Sequence[float],
    transformed_hull: Sequence[Sequence[float]],
    anchor: Sequence[float],
    yaw: float,
    targets: Mapping[str, Mapping[str, Any]],
    terrain: TerrainMask,
    site_rect: Sequence[float],
    districts: Mapping[str, Mapping[str, Any]],
    frontage_by_door: Mapping[str, Mapping[str, Any]],
    config: FitConfig,
) -> tuple[str | None, tuple[dict[str, Any], ...]]:
    x0, y0, x1, y1 = (float(value) for value in site_rect)
    if any(not (x0 <= float(point[0]) <= x1 and y0 <= float(point[1]) <= y1) for point in transformed_hull):
        return "hull_outside_scope", ()
    door_reports, door_error = _candidate_door_reports(
        lot, doors, frontage_by_door, anchor, yaw, targets, config)
    if door_error == "target_unresolved":
        return door_error, door_reports
    if door_error is not None:
        return door_error, door_reports
    for door in doors:
        if door.door_id not in frontage_by_door:
            continue
        offset = rot2d_ccw(door.offset[0], door.offset[1], yaw)
        point = (float(anchor[0]) + offset[0], float(anchor[1]) + offset[1])
        if not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1):
            return "door_outside_scope", door_reports
    samples: list[tuple[float, float]] = [tuple(float(value) for value in point) for point in transformed_hull]
    samples.append((float(centroid[0]), float(centroid[1])))
    min_x = min(float(point[0]) for point in transformed_hull)
    max_x = max(float(point[0]) for point in transformed_hull)
    min_y = min(float(point[1]) for point in transformed_hull)
    max_y = max(float(point[1]) for point in transformed_hull)
    # TerrainBundle exposes tile-level buildability through point queries.  Add
    # deterministic 512-GU tile-centre samples and edge midpoints so a hull
    # whose vertices happen to straddle a blocked tile cannot pass by sampling
    # only its corners/centroid.
    for y in range(math.floor(min_y / 512.0), math.ceil(max_y / 512.0)):
        for x in range(math.floor(min_x / 512.0), math.ceil(max_x / 512.0)):
            centre = (x * 512.0 + 256.0, y * 512.0 + 256.0)
            if point_in_ring(centre, list(transformed_hull)):
                samples.append(centre)
    for first, second in zip(transformed_hull, (*transformed_hull[1:], transformed_hull[0])):
        samples.append(((float(first[0]) + float(second[0])) / 2.0,
                        (float(first[1]) + float(second[1])) / 2.0))
    for door in doors:
        if door.door_id not in frontage_by_door:
            continue
        offset = rot2d_ccw(door.offset[0], door.offset[1], yaw)
        samples.append((float(anchor[0]) + offset[0], float(anchor[1]) + offset[1]))
    try:
        if any(bool(terrain.water_at(x, y)) for x, y in samples):
            return "water", door_reports
        if any(not bool(terrain.buildable_at(x, y)) for x, y in samples):
            return "unbuildable", door_reports
    except Exception:  # noqa: BLE001 - a mask failure is a failed candidate, not a fallback
        return "terrain_sample_unresolved", door_reports
    for target_id, target in sorted(targets.items()):
        for corridor in corridor_rings(target):
            if _rings_conflict(transformed_hull, corridor):
                return "road_corridor_conflict", door_reports
    for target_id, target in sorted(targets.items()):
        if target.get("kind") in ("road_surface_polygon", "shared_court", "plaza", "court"):
            polygon = target.get("polygon")
            if isinstance(polygon, list) and _ring_contains_or_conflicts(transformed_hull, polygon):
                return "space_interior_conflict", door_reports
    district = districts.get(str(lot["district"]))
    if district is None or not point_in_ring((float(centroid[0]), float(centroid[1])), district["polygon"]):
        return "district_centroid_outside", door_reports
    return None, door_reports


def _frontage_setup(
    lot: Mapping[str, Any],
    doors: Sequence[DoorGeometry],
    targets: Mapping[str, Mapping[str, Any]],
    frontage_side: str | None = None,
) -> tuple[DoorGeometry, dict[str, Mapping[str, Any]], tuple[_Segment, float, tuple[float, float]] | None, str | None]:
    """Resolve the primary door, its target, and the marker projection.

    With an explicit ``frontage_side`` the side is authored, so the marker
    position cannot make the side ambiguous and the polyline marker-on-
    centreline check is skipped (the polygon branch is unreachable for
    explicit sides: :func:`fit_intent` rejects polygon primary targets first).
    """
    frontage_by_door = {str(row["door_id"]): row for row in lot["frontages"]}
    door_by_id = {door.door_id: door for door in doors}
    unknown_doors = sorted(set(frontage_by_door) - set(door_by_id))
    if unknown_doors:
        return doors[0], frontage_by_door, None, "door_unresolved"
    primary_row = next(row for row in lot["frontages"] if row["primary"])
    primary = door_by_id[primary_row["door_id"]]
    target = targets.get(str(primary_row["target_id"]))
    if target is None:
        return primary, frontage_by_door, None, "target_unresolved"
    marker = lot["marker"]
    projection = _project_marker(marker, str(primary_row["target_id"]), target)
    if projection is None:
        return primary, frontage_by_door, None, "target_projection_unresolved"
    segment, _, projection_point = projection
    target_kind = str(target.get("kind", ""))
    if isinstance(target.get("polygon"), list):
        polygon = target["polygon"]
        if _point_on_polygon_boundary(marker, polygon):
            return primary, frontage_by_door, projection, "frontage_side_ambiguous"
        if point_in_ring((float(marker[0]), float(marker[1])), polygon):
            return primary, frontage_by_door, projection, "frontage_marker_inside_polygon"
    else:
        if frontage_side is None and math.hypot(float(marker[0]) - projection_point[0], float(marker[1]) - projection_point[1]) <= AMBIGUITY_EPSILON_GU:
            return primary, frontage_by_door, projection, "frontage_side_ambiguous"
    return primary, frontage_by_door, projection, None


def _construct_candidate(
    lot: Mapping[str, Any],
    primary: DoorGeometry,
    hull: Sequence[Sequence[float]],
    target: Mapping[str, Any],
    segment: _Segment,
    sample_point: Sequence[float],
    tangent: Sequence[float],
    target_arc: float,
    target_length: float,
    along_offset: float,
    gap: float,
    perturbation: float,
    frontage_side: str | None = None,
) -> tuple[_ConstructedCandidate | None, str | None]:
    """Compute the pivot-fixed transform for one sampled frontage point.

    Pure construction: primary door at the path edge plus gap, yaw
    perturbations about that fixed door point, then the transformed hull and
    footprint centroid.  Unary evaluation is intentionally NOT run here; the
    generator deduplicates the returned draft by rounded ``(centroid, yaw)``
    before evaluation (plan §6.3).  The only rejections this stage can emit
    are ``frontage_side_ambiguous`` (the sampled segment's marker side is
    undefined within the ambiguity epsilon) and, with an explicit
    ``frontage_side``, ``frontage_centroid_wrong_side`` (the constructed
    footprint centroid is not strictly on the authored side of the sampled
    segment by more than the ambiguity epsilon).
    """
    tx, ty = float(tangent[0]), float(tangent[1])
    target_kind = str(target.get("kind", ""))
    if frontage_side is not None:
        # Authored side relative to increasing point order of the polyline:
        # left = (-ty, tx), right = (ty, -tx).  The marker plays no role and
        # the sign is never inferred from the marker or lot id.
        normal = (-ty, tx) if frontage_side == "left" else (ty, -tx)
    elif isinstance(target.get("polygon"), list):
        area = segment.ring_area_value
        # For a CCW ring interior is left, so outward is right; CW reverses it.
        outward = (ty, -tx) if area > 0.0 else (-ty, tx)
        normal = outward
    else:
        # The authored side is the side of the *sampled* segment containing
        # the marker (plan §6.1).  A signed offset that walks past a road
        # bend lands on a later segment where the projection-segment sign
        # would place the door on the marker-opposite side, so the sign is
        # recomputed here against this sample's segment instead of being
        # reused blindly.  A marker on the sampled centreline within the
        # ambiguity epsilon stays fail-closed rather than choosing a side.
        marker_x, marker_y = float(lot["marker"][0]), float(lot["marker"][1])
        sampled_cross = tx * (marker_y - float(sample_point[1])) - ty * (marker_x - float(sample_point[0]))
        if abs(sampled_cross) <= AMBIGUITY_EPSILON_GU:
            return None, "frontage_side_ambiguous"
        sampled_sign = 1.0 if sampled_cross > 0.0 else -1.0
        left = (-ty, tx)
        normal = (left[0] * sampled_sign, left[1] * sampled_sign)
    half_width = 0.0 if isinstance(target.get("polygon"), list) else float(target.get("width_gu", 0.0)) / 2.0
    if target_kind == "existing_source_road":
        from .aligned_roads import SOURCE_ROAD_PRACTICAL_PATH_FRACTION
        half_width = float(target.get("width_gu", 0.0)) * SOURCE_ROAD_PRACTICAL_PATH_FRACTION / 2.0
    path_edge = (float(sample_point[0]) + normal[0] * half_width,
                 float(sample_point[1]) + normal[1] * half_width)
    primary_point = (path_edge[0] + normal[0] * gap, path_edge[1] + normal[1] * gap)
    desired_heading = math.degrees(math.atan2(path_edge[1] - primary_point[1],
                                               path_edge[0] - primary_point[0]))
    base_yaw = desired_heading - primary.heading_deg
    yaw = base_yaw + perturbation
    rotated_door = rot2d_ccw(primary.offset[0], primary.offset[1], yaw)
    anchor = (primary_point[0] - rotated_door[0], primary_point[1] - rotated_door[1])
    transformed_hull = _transform_hull(hull, anchor, yaw)
    centroid = polygon_centroid([list(point) for point in transformed_hull])
    if frontage_side is not None:
        # The resolved footprint centroid must remain strictly on the authored
        # side of THIS sampled segment by more than the ambiguity epsilon; a
        # candidate whose (possibly asymmetric) footprint hangs across the
        # road is rejected here, before deduplication or unary evaluation.
        centroid_cross = tx * (float(centroid[1]) - float(sample_point[1])) - \
            ty * (float(centroid[0]) - float(sample_point[0]))
        if (frontage_side == "left" and centroid_cross <= AMBIGUITY_EPSILON_GU) or \
           (frontage_side == "right" and centroid_cross >= -AMBIGUITY_EPSILON_GU):
            return None, "frontage_centroid_wrong_side"
    displacement_sq = (float(centroid[0]) - float(lot["marker"][0])) ** 2 + \
        (float(centroid[1]) - float(lot["marker"][1])) ** 2
    plaza_angle: float | None = None
    polygon = target.get("polygon")
    if isinstance(polygon, list):
        target_centroid = polygon_centroid(polygon)
        plaza_angle = math.degrees(math.atan2(
            float(centroid[1]) - target_centroid[1],
            float(centroid[0]) - target_centroid[0],
        )) % 360.0
    return _ConstructedCandidate(
        centroid=(float(centroid[0]), float(centroid[1])),
        anchor=(float(anchor[0]), float(anchor[1])),
        yaw_deg=float(yaw),
        hull=tuple((float(point[0]), float(point[1])) for point in transformed_hull),
        primary_door_position=(float(primary_point[0]), float(primary_point[1])),
        along_offset_gu=float(along_offset),
        door_gap_gu=float(gap),
        yaw_perturbation_deg=float(perturbation),
        primary_target_id=str(next(row["target_id"] for row in lot["frontages"] if row["primary"])),
        target_arc_gu=float(target_arc),
        target_length_gu=float(target_length),
        frontage_side=frontage_side,
        plaza_angle_deg=plaza_angle,
        marker_displacement_sq=displacement_sq,
    ), None


def _generate_lot_candidates(
    lot: dict[str, Any],
    stamp: Mapping[str, Any] | None,
    targets: Mapping[str, Mapping[str, Any]],
    terrain: TerrainMask,
    site_rect: Sequence[float],
    districts: Mapping[str, Mapping[str, Any]],
    config: FitConfig,
) -> _LotCandidates:
    if stamp is None:
        # Plan §6.4: "selected stamp/hull/door geometry is unresolved" is a
        # named unary rejection, not a hard abort: the fit stays fail-closed
        # but reports an unsatisfied lot histogram instead of a CLI crash.
        result = _LotCandidates(lot=lot, doors=[], primary=None)
        result.frontage_error = "stamp_geometry_unresolved"
        result.rejections["stamp_geometry_unresolved"] += 1
        return result
    try:
        hull = _stamp_hull(stamp)
        doors = _stamp_doors(stamp)
    except FrontageFitError:
        result = _LotCandidates(lot=lot, doors=[], primary=None)
        result.frontage_error = "stamp_geometry_unresolved"
        result.rejections["stamp_geometry_unresolved"] += 1
        return result
    result = _LotCandidates(lot=lot, doors=doors, primary=None)
    frontage_side = lot.get("frontage_side")
    primary, frontage_by_door, projection, setup_error = _frontage_setup(lot, doors, targets, frontage_side)
    result.primary = primary
    result.marker_projection = None if projection is None else {
        "segment_index": projection[0].segment_index,
        "projection_gu": list(projection[2]),
        "distance_gu": math.hypot(lot["marker"][0] - projection[2][0], lot["marker"][1] - projection[2][1]),
    }
    if setup_error is not None:
        result.frontage_error = setup_error
        result.rejections[setup_error] += 1
        return result
    assert projection is not None
    segment, arc_zero, projection_point = projection
    target = targets[str(next(row["target_id"] for row in lot["frontages"] if row["primary"]))]
    segments = _target_segments(segment.target_id, target)
    # Fail fast when the marker lies on the projection segment's centreline:
    # the side is undefined and the lot is reported as frontage_error instead
    # of a per-sample rejection histogram.  For samples past a bend the side
    # is recomputed per sampled segment inside _construct_candidate.  An
    # explicit frontage_side is authored, so this marker-derived ambiguity
    # check does not apply.
    if frontage_side is None:
        tangent_dx = segment.end[0] - segment.start[0]
        tangent_dy = segment.end[1] - segment.start[1]
        cross = tangent_dx * (lot["marker"][1] - projection_point[1]) - tangent_dy * (lot["marker"][0] - projection_point[0])
        if abs(cross) <= AMBIGUITY_EPSILON_GU:
            result.frontage_error = "frontage_side_ambiguous"
            result.rejections["frontage_side_ambiguous"] += 1
            return result
    seen_keys: set[tuple[float, float, float]] = set()
    candidate_ordinal = 0
    for along_offset in config.along_offsets_gu:
        sample_segment, sample_point, tangent, target_arc = _sample_arc(
            segments, arc_zero + along_offset)
        for gap in config.door_gaps_gu:
            for perturbation in config.yaw_perturbations_deg:
                result.generated += 1
                constructed, rejection = _construct_candidate(
                    lot, primary, hull, target,
                    sample_segment, sample_point, tangent, target_arc,
                    segments[-1].cumulative_end, along_offset,
                    gap, perturbation, frontage_side)
                if rejection is not None:
                    result.rejections[rejection] += 1
                    candidate_ordinal += 1
                    continue
                assert constructed is not None
                # Plan §6.3: deduplicate by rounded (centroid, yaw) BEFORE
                # unary evaluation, so a repeated key (e.g. multiple offsets
                # clamping to the same polyline endpoint) is never
                # re-evaluated.  Duplicates are therefore a count, not a
                # rejection code, and `deduplicated` means "unique constructed
                # keys" (generated minus construction-stage rejections minus
                # duplicates), matching plan §7.
                key = (round(constructed.centroid[0], 1),
                       round(constructed.centroid[1], 1),
                       round(constructed.yaw_deg, 1))
                if key in seen_keys:
                    candidate_ordinal += 1
                    continue
                seen_keys.add(key)
                rejection, door_reports = _validate_lot_geometry(
                    lot, stamp, doors, hull, lot["marker"], constructed.centroid,
                    constructed.hull, constructed.anchor, constructed.yaw_deg,
                    targets, terrain, site_rect, districts, frontage_by_door, config)
                if rejection is not None:
                    result.rejections[rejection] += 1
                    candidate_ordinal += 1
                    continue
                result.all_unary_feasible.append(Candidate(
                    ordinal=candidate_ordinal,
                    centroid=constructed.centroid,
                    anchor=constructed.anchor,
                    yaw_deg=constructed.yaw_deg,
                    hull=constructed.hull,
                    primary_door_position=constructed.primary_door_position,
                    along_offset_gu=constructed.along_offset_gu,
                    door_gap_gu=constructed.door_gap_gu,
                    yaw_perturbation_deg=constructed.yaw_perturbation_deg,
                    primary_target_id=constructed.primary_target_id,
                    target_arc_gu=constructed.target_arc_gu,
                    target_length_gu=constructed.target_length_gu,
                    frontage_side=constructed.frontage_side,
                    plaza_angle_deg=constructed.plaza_angle_deg,
                    door_reports=door_reports,
                    _marker_displacement_sq=constructed.marker_displacement_sq))
                candidate_ordinal += 1
    result.deduplicated = len(seen_keys)
    result.all_unary_feasible.sort(key=lambda candidate: (candidate.rank, candidate.ordinal))
    result.unary_feasible = result.all_unary_feasible[:config.max_candidates_per_lot]
    return result


def _build_compatibility(
    ordered: Sequence[_LotCandidates],
    candidate_domains: Mapping[str, Sequence[Candidate]] | None = None,
) -> tuple[tuple[tuple[tuple[int, ...] | None, ...], ...], int]:
    """Build the directed candidate compatibility bitsets (plan §4).

    Only unordered lot pairs ``i < j`` are visited; every candidate pair is
    evaluated exactly once with :func:`_rings_conflict`, and both directed
    rows are constructed during that same loop: ``rows_i[c_i]`` gets bit
    ``c_j`` when compatible and ``rows_j[c_j]`` gets bit ``c_i``.  The
    returned ``compatibility_checks`` is therefore exactly
    ``sum(len(candidates_i) * len(candidates_j) for every i < j)``.

    Candidate index is the position in the supplied rank-sorted domain.  The
    legacy domain is ``unary_feasible``; progressive relationship passes supply
    a wider prefix of ``all_unary_feasible``.  ``Candidate.ordinal`` is never
    used as a bit index.  Diagonal entries stay ``None`` and must never be
    queried.
    All lists are frozen to tuples before returning; iteration is by index
    only, never by set/dict order.
    """
    lot_count = len(ordered)
    rows: list[list[tuple[int, ...] | None]] = [[None] * lot_count for _ in range(lot_count)]
    checks = 0
    for i in range(lot_count):
        candidates_i = (
            candidate_domains.get(ordered[i].lot["id"], ordered[i].unary_feasible)
            if candidate_domains is not None else ordered[i].unary_feasible
        )
        for j in range(i + 1, lot_count):
            candidates_j = (
                candidate_domains.get(ordered[j].lot["id"], ordered[j].unary_feasible)
                if candidate_domains is not None else ordered[j].unary_feasible
            )
            forward = [0] * len(candidates_i)
            reverse = [0] * len(candidates_j)
            for c_i, candidate_i in enumerate(candidates_i):
                for c_j, candidate_j in enumerate(candidates_j):
                    checks += 1
                    if not _rings_conflict(candidate_i.hull, candidate_j.hull):
                        forward[c_i] |= 1 << c_j
                        reverse[c_j] |= 1 << c_i
            rows[i][j] = tuple(forward)
            rows[j][i] = tuple(reverse)
    return tuple(tuple(row) for row in rows), checks


def _search_compatibility(
    ordered: Sequence[_LotCandidates],
    compat: tuple[tuple[tuple[int, ...] | None, ...], ...],
    compatibility_checks: int,
    config: FitConfig,
    complete_assignment: Callable[
        [Sequence[_LotCandidates], tuple[tuple[str, Candidate], ...]],
        tuple[dict[str, Any], Sequence[Mapping[str, Any]]],
    ] | None = None,
    candidate_domains: Mapping[str, Sequence[Candidate]] | None = None,
    budget_state: list[int] | None = None,
) -> _SearchResult:
    """Complete MRV/forward-checking search over the compatibility matrix.

    Domains are immutable integer bitsets over candidate positions. The
    legacy domain is ``unary_feasible``; ``candidate_domains`` can provide a
    wider retained prefix for a progressive hard-relationship pass, avoiding
    mutable-domain journal bugs. The unassigned lot with the smallest domain is chosen by
    dynamic MRV tied on canonical lot index ``(bit_count, i)``; candidate
    values are tried in ascending bit (ascending candidate index) order, which
    exactly preserves the existing ``(candidate.rank, candidate.ordinal)``
    preference.  Branch domains are copied into a list, the chosen lot is
    pinned, every unassigned neighbour domain is intersected with the directed
    compatibility row, and a zeroed domain prunes the branch before recursion.

    At every state entry ``nodes`` increments and the budget is checked
    immediately: a positive ``search_node_budget`` that is exceeded raises
    :class:`_SearchBudgetExhausted`, which is caught exactly once here, at the
    search boundary, never inside a branch.  Exhaustive failure returns
    ``selected=None`` with ``budget_exhausted=False``; budget exhaustion
    returns ``selected=None`` with ``budget_exhausted=True`` (inconclusive,
    plan §6).  On success every ``chosen`` index is asserted non-negative and
    the selection is emitted in canonical lot order ``0..N-1``.  One
    feasibility pass returns its first complete assignment (or first
    hard-valid assignment when ``complete_assignment`` is supplied); that
    pass-local witness is rank-preferring locally, not a claim of global
    rank-optimality.  ``fit_intent`` may widen composition passes and may then
    run the separate bounded improvement phase.  Collision-valid leaves are
    evaluated and hard-finding leaves continue backtracking.
    """
    lot_count = len(ordered)
    candidate_lists = tuple(
        tuple(
            candidate_domains.get(ordered[i].lot["id"], ordered[i].unary_feasible)
            if candidate_domains is not None else ordered[i].unary_feasible
        )
        for i in range(lot_count)
    )
    domains = tuple((1 << len(candidate_lists[i])) - 1 for i in range(lot_count))
    chosen: list[int] = [-1] * lot_count
    nodes = 0
    extensions = 0
    backtracks = 0
    complete_assignments_checked = 0
    collision_valid_assignments = 0
    relationship_rejections: Counter[str] = Counter()
    last_composition: dict[str, Any] | None = None

    def search(current_domains: tuple[int, ...], assigned_mask: int) -> bool:
        nonlocal nodes, extensions, backtracks, complete_assignments_checked
        nonlocal collision_valid_assignments, last_composition
        nodes += 1
        budget_used = nodes
        if budget_state is not None:
            budget_state[0] += 1
            budget_used = budget_state[0]
        if config.search_node_budget > 0 and budget_used > config.search_node_budget:
            raise _SearchBudgetExhausted()
        if assigned_mask == (1 << lot_count) - 1:
            collision_valid_assignments += 1
            selected_at_leaf = tuple(
                (ordered[i].lot["id"], candidate_lists[i][chosen[i]])
                for i in range(lot_count)
            )
            if complete_assignment is not None:
                complete_assignments_checked += 1
                composition, hard_findings = complete_assignment(ordered, selected_at_leaf)
                last_composition = composition
                for finding in hard_findings:
                    code = (str(finding.get("code", ""))
                            if isinstance(finding, Mapping) else str(finding))
                    if code:
                        relationship_rejections[code] += 1
                if hard_findings:
                    return False
            return True
        lot_i = min(
            (i for i in range(lot_count) if not (assigned_mask >> i) & 1),
            key=lambda i: (current_domains[i].bit_count(), i),
        )
        remaining = current_domains[lot_i]
        while remaining:
            low_bit = remaining & (-remaining)
            candidate_i = low_bit.bit_length() - 1
            remaining ^= low_bit
            extensions += 1
            next_domains = list(current_domains)
            next_domains[lot_i] = 1 << candidate_i
            branch_feasible = True
            for other in range(lot_count):
                if other == lot_i or (assigned_mask >> other) & 1:
                    continue
                row = compat[lot_i][other]
                assert row is not None, "off-diagonal compatibility row must exist"
                next_domains[other] &= row[candidate_i]
                if next_domains[other] == 0:
                    branch_feasible = False
                    break
            if not branch_feasible:
                continue
            chosen[lot_i] = candidate_i
            if search(tuple(next_domains), assigned_mask | (1 << lot_i)):
                return True
            chosen[lot_i] = -1
        backtracks += 1
        return False

    budget_exhausted = False
    selected: tuple[tuple[str, Candidate], ...] | None = None
    try:
        solved = search(domains, 0)
    except _SearchBudgetExhausted:
        budget_exhausted = True
    else:
        if solved:
            assert all(index >= 0 for index in chosen)
            selected = tuple(
                (ordered[i].lot["id"], candidate_lists[i][chosen[i]])
                for i in range(lot_count)
            )
    return _SearchResult(
        selected=selected,
        stats=_SearchStats(
            nodes=nodes,
            extensions=extensions,
            compatibility_checks=compatibility_checks,
            backtracks=backtracks,
            budget_exhausted=budget_exhausted,
            complete_assignments_checked=complete_assignments_checked,
            relationship_rejections=dict(sorted(relationship_rejections.items())),
            collision_valid_assignments=collision_valid_assignments,
        ),
        composition=last_composition,
    )


def _complete_select(
    lot_candidates: Sequence[_LotCandidates], config: FitConfig,
    *,
    complete_assignment: Callable[
        [Sequence[_LotCandidates], tuple[tuple[str, Candidate], ...]],
        tuple[dict[str, Any], Sequence[Mapping[str, Any]]],
    ] | None = None,
    candidate_domains: Mapping[str, Sequence[Candidate]] | None = None,
    budget_state: list[int] | None = None,
) -> _SearchResult:
    """Complete-search wrapper: canonical ordering + build + search (plan §3.3).

    Canonical lot order is ``(domain size, lot id)``. Without an explicit
    domain this is ``(len(unary_feasible), lot id)`` exactly as in the legacy
    search. Candidate index is the position in the selected domain.
    """
    ordered, compat, checks = _ordered_search_inputs(lot_candidates, candidate_domains)
    return _search_compatibility(
        ordered, compat, checks, config,
        complete_assignment=complete_assignment,
        candidate_domains=candidate_domains,
        budget_state=budget_state,
    )


def _ordered_search_inputs(
    lot_candidates: Sequence[_LotCandidates],
    candidate_domains: Mapping[str, Sequence[Candidate]] | None = None,
) -> tuple[
    tuple[_LotCandidates, ...],
    tuple[tuple[tuple[int, ...] | None, ...], ...],
    int,
]:
    """Rebuild one pass's canonical order, domains, and compatibility matrix.

    ``_complete_select`` and the post-feasibility improvement phase must use
    the same positional matrix.  Keeping this small wrapper as the one place
    that orders a pass makes rebuilding the successful pass deterministic
    without changing the first-witness semantics of the feasibility search.
    """
    def domain_size(item: _LotCandidates) -> int:
        if candidate_domains is None:
            return len(item.unary_feasible)
        return len(candidate_domains.get(item.lot["id"], item.unary_feasible))

    ordered = sorted(
        lot_candidates,
        key=lambda item: (domain_size(item), item.lot["id"]),
    )
    compat, checks = _build_compatibility(ordered, candidate_domains)
    return tuple(ordered), compat, checks


def _improve_assignment(
    intent: Mapping[str, Any],
    ordered: Sequence[_LotCandidates],
    candidate_domains: Mapping[str, Sequence[Candidate]] | None = None,
    compatibility: tuple[tuple[tuple[int, ...] | None, ...], ...] | Mapping[Any, Any] | None = None,
    incumbent: Sequence[tuple[str, Candidate]] | Mapping[str, Candidate] | None = None,
    config: FitConfig | None = None,
    *,
    compat: tuple[tuple[tuple[int, ...] | None, ...], ...] | Mapping[Any, Any] | None = None,
) -> _ImprovementResult:
    """Enumerate the successful feasibility domain for a better assignment.

    This is the bounded post-feasibility engine used by ``fit_intent``.  It
    receives the exact domains and directed compatibility rows from the
    successful hard-search pass; it never widens or rebuilds them.  It returns
    a replacement assignment that the caller may adopt, but never mutates
    ``fit_intent`` state itself.  The incumbent is evaluated first and is held
    as the initial best.  A positive budget uses the same state-entry semantics
    as feasibility search (the state that exceeds the budget is counted, then
    the walk stops); zero means no traversal at all.  Any setup, evaluator, or
    traversal exception returns the incumbent unchanged.

    ``compat`` is accepted as a keyword alias for the longer
    ``compatibility`` spelling so direct callers can mirror
    :func:`_search_compatibility`.  Matrix rows are positional domain rows;
    a mapping keyed by ``(from_lot_id, to_lot_id)`` is also accepted for small
    literal tests.  In either form rows are remapped to canonical
    ``(domain-size, lot-id)`` order before traversal, making reversed input
    order byte-deterministic without reconstructing compatibility.
    """
    config = config or FitConfig()
    if compatibility is None:
        compatibility = compat
    raw_incumbent: tuple[tuple[str, Candidate], ...]
    if isinstance(incumbent, Mapping):
        raw_incumbent = tuple(
            (str(lot_id), candidate) for lot_id, candidate in incumbent.items())
    else:
        raw_incumbent = tuple(incumbent or ())

    def empty_result(
        selected: tuple[tuple[str, Candidate], ...],
        *,
        composition: dict[str, Any] | None = None,
        objective: tuple[Any, ...] = (),
        components: dict[str, Any] | None = None,
        nodes: int = 0,
        extensions: int = 0,
        collision_valid_leaves: int = 0,
        hard_valid_leaves: int = 0,
        relationship_rejections: Mapping[str, int] | None = None,
        budget_exhausted: bool = False,
        incumbent_improved: bool = False,
        domain_sizes: tuple[tuple[str, int], ...] = (),
    ) -> _ImprovementResult:
        return _ImprovementResult(
            selected=selected,
            composition=composition,
            objective=objective,
            objective_components=dict(components or {}),
            stats=_ImprovementStats(
                nodes=nodes,
                extensions=extensions,
                collision_valid_assignments=collision_valid_leaves,
                hard_valid_assignments=hard_valid_leaves,
                relationship_rejections=dict(sorted((relationship_rejections or {}).items())),
                budget_exhausted=budget_exhausted,
                incumbent_improved=incumbent_improved,
                domain_sizes=domain_sizes,
            ),
        )

    # The accepted incumbent is already a complete assignment.  Preserve its
    # exact tuple in all early-return paths; canonicalization below is only for
    # traversal and deterministic signatures.
    if not raw_incumbent:
        return empty_result(raw_incumbent)

    try:
        if candidate_domains is None:
            candidate_domains = {
                str(item.lot["id"]): tuple(item.unary_feasible)
                for item in ordered
            }
        original_items = tuple(ordered)
        original_ids = tuple(str(item.lot["id"]) for item in original_items)
        if len(set(original_ids)) != len(original_ids):
            raise FrontageFitError("improvement domain contains duplicate lot ids")
        domains_by_id = {
            lot_id: tuple(candidate_domains[lot_id]) for lot_id in original_ids
        }
        canonical_indices = tuple(sorted(
            range(len(original_items)),
            key=lambda index: (len(domains_by_id[original_ids[index]]), original_ids[index]),
        ))
        canonical_ordered = tuple(original_items[index] for index in canonical_indices)
        canonical_ids = tuple(str(item.lot["id"]) for item in canonical_ordered)
        candidate_lists = tuple(domains_by_id[lot_id] for lot_id in canonical_ids)
        domain_sizes = tuple(sorted(
            ((lot_id, len(domains_by_id[lot_id])) for lot_id in canonical_ids),
            key=lambda row: row[0],
        ))

        incumbent_by_id = {str(lot_id): candidate for lot_id, candidate in raw_incumbent}
        if set(incumbent_by_id) != set(canonical_ids):
            raise FrontageFitError("improvement incumbent does not cover the candidate domain")
        incumbent_selected = tuple(
            (lot_id, incumbent_by_id[lot_id]) for lot_id in canonical_ids)

        # Evaluate the incumbent before any node is entered.  An accepted
        # incumbent is hard-valid by contract, but retain it even if a malformed
        # direct fixture makes this evaluation fail.
        incumbent_composition, incumbent_hard = _evaluate_selected_composition(
            intent, canonical_ordered, incumbent_selected)
        if incumbent_hard:
            raise FrontageFitError("improvement incumbent is not hard-valid")
        incumbent_objective, incumbent_components = _assignment_preference(
            intent, incumbent_composition, incumbent_selected)
    except Exception:  # noqa: BLE001 - an improvement fault must not lose the incumbent
        fallback_sizes: tuple[tuple[str, int], ...] = ()
        if candidate_domains is not None:
            fallback_sizes = tuple(sorted(
                ((str(item.lot["id"]),
                  len(candidate_domains.get(str(item.lot["id"]), ())))
                 for item in ordered),
                key=lambda row: row[0],
            ))
        return empty_result(raw_incumbent, domain_sizes=fallback_sizes)

    # An accepted feasibility incumbent is hard-valid by contract.  A direct
    # caller that violates that contract has already been handled by the
    # fail-closed outer boundary; the original tuple remains its fallback.
    best_selected = incumbent_selected
    best_composition = incumbent_composition
    best_objective = incumbent_objective
    best_components = incumbent_components

    if config.improvement_node_budget == 0:
        return empty_result(
            best_selected,
            composition=best_composition,
            objective=best_objective,
            components=best_components,
            domain_sizes=domain_sizes,
        )

    try:
        if compatibility is None:
            raise FrontageFitError("improvement compatibility matrix is missing")

        def matrix_row(from_id: str, to_id: str, from_index: int, to_index: int) -> tuple[int, ...]:
            if isinstance(compatibility, Mapping):
                row = compatibility[(from_id, to_id)]
            else:
                row = compatibility[from_index][to_index]
            if row is None:
                raise FrontageFitError(
                    f"improvement compatibility row missing for {from_id!r} -> {to_id!r}")
            return tuple(int(value) for value in row)

        # Re-index the supplied matrix along with the canonical lot order.  No
        # geometric compatibility is recomputed in this phase.
        original_index_by_id = {lot_id: index for index, lot_id in enumerate(original_ids)}
        compat_rows: list[list[tuple[int, ...] | None]] = [
            [None] * len(canonical_ids) for _ in canonical_ids
        ]
        for i, from_id in enumerate(canonical_ids):
            original_i = original_index_by_id[from_id]
            for j, to_id in enumerate(canonical_ids):
                if i == j:
                    continue
                original_j = original_index_by_id[to_id]
                row = matrix_row(from_id, to_id, original_i, original_j)
                if len(row) != len(candidate_lists[i]):
                    raise FrontageFitError(
                        f"improvement compatibility row {from_id!r}->{to_id!r} has "
                        f"{len(row)} entries, expected {len(candidate_lists[i])}")
                compat_rows[i][j] = row
        compat_matrix = tuple(tuple(row) for row in compat_rows)
    except Exception:  # noqa: BLE001 - a matrix fault must not lose the incumbent
        return empty_result(
            best_selected,
            composition=best_composition,
            objective=best_objective,
            components=best_components,
            domain_sizes=domain_sizes,
        )

    lot_count = len(canonical_ordered)
    domains = tuple((1 << len(candidates)) - 1 for candidates in candidate_lists)
    chosen: list[int] = [-1] * lot_count
    nodes = 0
    extensions = 0
    collision_valid_leaves = 0
    hard_valid_leaves = 0
    relationship_rejections: Counter[str] = Counter()
    budget_exhausted = False
    incumbent_improved = False

    def search(current_domains: tuple[int, ...], assigned_mask: int) -> None:
        nonlocal nodes, extensions, collision_valid_leaves, hard_valid_leaves
        nonlocal best_selected, best_composition, best_objective, best_components
        nonlocal incumbent_improved
        nodes += 1
        if (config.improvement_node_budget > 0 and
                nodes > config.improvement_node_budget):
            raise _ImprovementBudgetExhausted()
        if assigned_mask == (1 << lot_count) - 1:
            collision_valid_leaves += 1
            selected_at_leaf = tuple(
                (canonical_ids[i], candidate_lists[i][chosen[i]])
                for i in range(lot_count)
            )
            composition, hard_findings = _evaluate_selected_composition(
                intent, canonical_ordered, selected_at_leaf)
            for finding in hard_findings:
                code = (str(finding.get("code", ""))
                        if isinstance(finding, Mapping) else str(finding))
                if code:
                    relationship_rejections[code] += 1
            if hard_findings:
                return
            hard_valid_leaves += 1
            objective, components = _assignment_preference(
                intent, composition, selected_at_leaf)
            if objective < best_objective:
                best_selected = selected_at_leaf
                best_composition = composition
                best_objective = objective
                best_components = components
                incumbent_improved = True
            return

        lot_i = min(
            (i for i in range(lot_count) if not (assigned_mask >> i) & 1),
            key=lambda i: (current_domains[i].bit_count(), i),
        )
        remaining = current_domains[lot_i]
        while remaining:
            low_bit = remaining & (-remaining)
            candidate_i = low_bit.bit_length() - 1
            remaining ^= low_bit
            extensions += 1
            next_domains = list(current_domains)
            next_domains[lot_i] = 1 << candidate_i
            branch_feasible = True
            for other in range(lot_count):
                if other == lot_i or (assigned_mask >> other) & 1:
                    continue
                row = compat_matrix[lot_i][other]
                assert row is not None, "off-diagonal improvement row must exist"
                next_domains[other] &= row[candidate_i]
                if next_domains[other] == 0:
                    branch_feasible = False
                    break
            if not branch_feasible:
                continue
            chosen[lot_i] = candidate_i
            search(tuple(next_domains), assigned_mask | (1 << lot_i))
            chosen[lot_i] = -1

    try:
        search(domains, 0)
    except _ImprovementBudgetExhausted:
        budget_exhausted = True
    except Exception:  # noqa: BLE001 - preserve best-so-far on evaluator/fixture faults
        pass

    return empty_result(
        best_selected,
        composition=best_composition,
        objective=best_objective,
        components=best_components,
        nodes=nodes,
        extensions=extensions,
        collision_valid_leaves=collision_valid_leaves,
        hard_valid_leaves=hard_valid_leaves,
        relationship_rejections=relationship_rejections,
        budget_exhausted=budget_exhausted,
        incumbent_improved=incumbent_improved,
        domain_sizes=domain_sizes,
    )


def _composition_domains(
    lot_results: Sequence[_LotCandidates], width: int,
) -> tuple[dict[str, tuple[Candidate, ...]], tuple[tuple[str, int], ...]]:
    """Return one deterministic retained-domain vector for a search pass."""
    domains: dict[str, tuple[Candidate, ...]] = {}
    vector: list[tuple[str, int]] = []
    for item in sorted(lot_results, key=lambda row: row.lot["id"]):
        candidates = tuple(item.all_unary_feasible[:width])
        lot_id = str(item.lot["id"])
        domains[lot_id] = candidates
        vector.append((lot_id, len(candidates)))
    return domains, tuple(vector)


def _report_lot(
    result: _LotCandidates,
    selected: Candidate | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "lot_id": result.lot["id"],
        "stamp": result.lot["stamp"],
        "district": result.lot["district"],
        "role": result.lot["role"],
        "marker": list(result.lot["marker"]),
        "frontage_side": result.lot.get("frontage_side"),
        "declared_door_ids": sorted(row["door_id"] for row in result.lot["frontages"]),
        "undeclared_doors": sorted(door.door_id for door in result.doors
                                    if door.door_id not in {row["door_id"] for row in result.lot["frontages"]}),
        "candidate_count_generated": result.generated,
        "candidate_count_deduplicated": result.deduplicated,
        "candidate_count_unary_feasible": len(result.unary_feasible),
        "rejection_histogram": dict(sorted(result.rejections.items())),
    }
    if result.marker_projection is not None:
        row["marker_projection"] = result.marker_projection
    if result.frontage_error is not None:
        row["frontage_error"] = result.frontage_error
    if selected is not None:
        row["selected"] = {
            "candidate_ordinal": selected.ordinal,
            "centroid": list(selected.centroid),
            "anchor": list(selected.anchor),
            "yaw_deg": selected.yaw_deg,
            "marker_displacement_gu": math.sqrt(selected._marker_displacement_sq),
            "along_frontage_offset_gu": selected.along_offset_gu,
            "door_gap_gu": selected.door_gap_gu,
            "yaw_perturbation_deg": selected.yaw_perturbation_deg,
            "primary_target_id": selected.primary_target_id,
        }
        row["declared_door_facts"] = [dict(fact) for fact in selected.door_reports]
    return row


def _resolved_sketch(intent: Mapping[str, Any], selected: Mapping[str, Candidate]) -> dict[str, Any]:
    lots: list[dict[str, Any]] = []
    for lot in sorted(intent["lots"], key=lambda row: row["id"]):
        candidate = selected[lot["id"]]
        row: dict[str, Any] = {
            "id": lot["id"],
            "stamp": lot["stamp"],
            "x": candidate.centroid[0],
            "y": candidate.centroid[1],
            "yaw_deg": candidate.yaw_deg,
            "door_targets": [
                {"door_id": frontage["door_id"], "target_id": frontage["target_id"],
                 "intent": frontage["intent"]}
                for frontage in sorted(lot["frontages"], key=lambda row: row["door_id"])
            ],
        }
        if "note" in lot:
            row["note"] = lot["note"]
        lots.append(row)
    # The resolved sketch is consumed by the renderer's legacy road schema.
    # Composition declarations remain in the normalized intent/report input,
    # but must not cross this boundary as unknown renderer keys.
    roads = [
        {
            "id": road["id"],
            "kind": road["kind"],
            "width_gu": road["width_gu"],
            "points": deepcopy(road["points"]),
        }
        for road in intent["roads"]
    ]
    return {
        "site": intent["site"],
        "roads": roads,
        "spaces": intent["spaces"],
        "lots": lots,
        "notes": intent["notes"],
    }


def fit_intent(
    intent: Any,
    *,
    site_name: str,
    site_rect: Sequence[float],
    stamp_ids: set[str],
    stamp_geometry: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, Any]],
    terrain: TerrainMask,
    config: FitConfig | None = None,
    input_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit one intent and return ``(intent_copy, resolved_sketch, report)``.

    A well-formed but impossible intent returns an ``unsatisfied`` report and
    an empty-lot resolved sketch.  Structural/quarantine/identity violations
    raise :class:`FrontageFitError` and are never replaced with a nearest-door
    or nearest-target substitute.
    """

    config = config or FitConfig()
    intent_copy = validate_intent(intent, site_name=site_name, stamp_ids=stamp_ids, site_rect=site_rect)
    targets = _world_target_rows(targets)
    target_ids = set(targets)
    districts = {row["id"]: row for row in intent_copy["districts"]}
    door_ids_by_stamp: dict[str, set[str]] = {}
    for stamp_id in stamp_ids:
        stamp = stamp_geometry.get(stamp_id)
        if stamp is None:
            continue
        try:
            doors = _stamp_doors(stamp)
        except FrontageFitError:
            # Unresolved selected-stamp geometry is reported per lot as the
            # plan §6.4 named rejection at candidate generation; it is not a
            # validation error for stamps no lot selects.
            continue
        door_ids_by_stamp[stamp_id] = {door.door_id for door in doors}
    for lot in intent_copy["lots"]:
        for frontage in lot["frontages"]:
            if frontage["target_id"] not in target_ids:
                raise FrontageFitError(
                    f"lot {lot['id']!r} target {frontage['target_id']!r} is unavailable")
            stamp_doors = door_ids_by_stamp.get(lot["stamp"])
            if stamp_doors is not None and frontage["door_id"] not in stamp_doors:
                raise FrontageFitError(
                    f"lot {lot['id']!r} door {frontage['door_id']!r} is not on stamp {lot['stamp']!r}")
    # Group target identity is checked at the same loaded-target boundary as
    # frontage targets.  Plaza sectors additionally require the shared target
    # to be one of the target kinds used for authored spaces; normalization
    # cannot establish that fact without the loaded target map.
    for group in intent_copy.get("lot_groups", []):
        shared_target_id = group.get("shared_target_id")
        if shared_target_id is None:
            continue
        target = targets.get(shared_target_id)
        if target is None:
            raise FrontageFitError(
                f"lot group {group['id']!r} shared target {shared_target_id!r} is unavailable")
        if "plaza_sectors" in group and target.get("kind") not in (
                "plaza", "court", "shared_court", "road_surface_polygon"):
            raise FrontageFitError(
                f"lot group {group['id']!r} shared target {shared_target_id!r} "
                "must be a plaza/space target")
    # frontage_side is polyline-only: the authored side is defined relative
    # to increasing point order of the primary target's polyline.  A polygon
    # (plaza/court) primary target fails closed here, before any candidate
    # generation, rather than being silently reinterpreted.
    for lot in intent_copy["lots"]:
        if "frontage_side" not in lot:
            continue
        primary_row = next(row for row in lot["frontages"] if row["primary"])
        target = targets.get(str(primary_row["target_id"]))
        polyline = target.get("polyline") if isinstance(target, Mapping) else None
        if (not isinstance(polyline, list) or len(polyline) < 2 or
                isinstance(target.get("polygon"), list)):
            raise FrontageFitError(
                f"lot {lot['id']!r} frontage_side requires a road/source-road "
                f"polyline primary target; target {primary_row['target_id']!r} "
                f"is a polygon/space target")
    lot_results: list[_LotCandidates] = []
    for lot in intent_copy["lots"]:
        lot_results.append(_generate_lot_candidates(
            lot, stamp_geometry.get(lot["stamp"]), targets, terrain, site_rect, districts, config))
    counts = {
        "candidate_generated": sum(item.generated for item in lot_results),
        "candidate_deduplicated": sum(item.deduplicated for item in lot_results),
        "candidate_unary_feasible": sum(len(item.unary_feasible) for item in lot_results),
        "candidate_unary_feasible_all": sum(len(item.all_unary_feasible) for item in lot_results),
        "search_nodes": 0,
        "search_extensions": 0,
        "compatibility_checks": 0,
        "search_backtracks": 0,
        "search_budget_exhausted": False,
        "complete_assignments_checked": 0,
        "relationship_rejections": {},
        "collision_valid_assignments": 0,
    }
    selected_by_id: dict[str, Candidate] = {}
    terminal_failure: str | None = None
    composition_enabled = (
        "lot_groups" in intent_copy or
        any("purpose" in road for road in intent_copy["roads"]) or
        any("intentional_outlier" in lot for lot in intent_copy["lots"])
    )
    # A hard-relationship search must not mistake an active cap of zero (or
    # another truncated prefix) for a unary proof.  Only the complete retained
    # list can establish that a composition lot has no unary candidate at all.
    unresolved = [
        item.lot["id"] for item in lot_results
        if not (item.all_unary_feasible if composition_enabled else item.unary_feasible)
    ]
    composition_report: dict[str, Any] = {"roads": [], "groups": [], "findings": []}
    search_passes: list[dict[str, Any]] = []
    successful_domain_width: int | None = None
    successful_domain_sizes: tuple[tuple[str, int], ...] = ()
    successful_domains: Mapping[str, Sequence[Candidate]] | None = None
    successful_ordered: tuple[_LotCandidates, ...] | None = None
    successful_compatibility: tuple[tuple[tuple[int, ...] | None, ...], ...] | None = None
    feasibility_incumbent: tuple[tuple[str, Candidate], ...] | None = None
    feasibility_composition: dict[str, Any] | None = None
    improvement_report: dict[str, Any] | None = None
    if unresolved:
        # Unary short-circuit (plan §8): a lot with no unary-feasible
        # candidates is unsatisfied without any global search, and every
        # search metric stays at its zeroed initial value.
        terminal_failure = "unary_unsatisfied"
    else:
        if not composition_enabled:
            # Keep the old intent path exactly one capped search, with the
            # original call shape and first-feasible semantics.
            result = _complete_select(lot_results, config)
            search_passes.append({
                "domain_width": config.max_candidates_per_lot,
                "candidate_domain_sizes": [
                    {"lot_id": item.lot["id"], "count": len(item.unary_feasible)}
                    for item in sorted(lot_results, key=lambda row: row.lot["id"])
                ],
                "search_nodes": result.stats.nodes,
                "search_extensions": result.stats.extensions,
                "compatibility_checks": result.stats.compatibility_checks,
                "search_backtracks": result.stats.backtracks,
                "search_budget_exhausted": result.stats.budget_exhausted,
                "complete_assignments_checked": 0,
                "relationship_rejections": {},
            })
            counts["search_nodes"] = result.stats.nodes
            counts["search_extensions"] = result.stats.extensions
            counts["compatibility_checks"] = result.stats.compatibility_checks
            counts["search_backtracks"] = result.stats.backtracks
            counts["search_budget_exhausted"] = result.stats.budget_exhausted
            if result.selected is not None:
                selected_by_id = {lot_id: candidate for lot_id, candidate in result.selected}
        else:
            # Relationship proof must search beyond the preference cap.  The
            # first pass is the existing cap; later passes are deterministic
            # widening prefixes, and the final pass is the complete retained
            # unary-feasible domain.  One shared counter makes the node budget
            # global across all passes.
            max_retained = max((len(item.all_unary_feasible) for item in lot_results), default=0)
            requested_widths = [config.max_candidates_per_lot]
            requested_widths.extend(
                width for width in (128, 256, 512, max_retained)
                if width > config.max_candidates_per_lot
            )
            seen_vectors: set[tuple[tuple[str, int], ...]] = set()
            budget_state = [0]
            cumulative_rejections: Counter[str] = Counter()
            cumulative_collision_valid = 0
            result: _SearchResult | None = None
            for width in requested_widths:
                domains, vector = _composition_domains(lot_results, width)
                if vector in seen_vectors:
                    continue
                seen_vectors.add(vector)
                pass_result = _complete_select(
                    lot_results,
                    config,
                    complete_assignment=lambda ordered, chosen: _evaluate_selected_composition(
                        intent_copy, ordered, chosen),
                    candidate_domains=domains,
                    budget_state=budget_state,
                )
                cumulative_rejections.update(pass_result.stats.relationship_rejections)
                cumulative_collision_valid += pass_result.stats.collision_valid_assignments
                search_passes.append({
                    "domain_width": width,
                    "candidate_domain_sizes": [
                        {"lot_id": lot_id, "count": count}
                        for lot_id, count in vector
                    ],
                    "search_nodes": pass_result.stats.nodes,
                    "search_extensions": pass_result.stats.extensions,
                    "compatibility_checks": pass_result.stats.compatibility_checks,
                    "search_backtracks": pass_result.stats.backtracks,
                    "search_budget_exhausted": pass_result.stats.budget_exhausted,
                    "complete_assignments_checked": pass_result.stats.complete_assignments_checked,
                    "relationship_rejections": dict(pass_result.stats.relationship_rejections),
                })
                for key, stat_name in (
                    ("search_nodes", "nodes"),
                    ("search_extensions", "extensions"),
                    ("compatibility_checks", "compatibility_checks"),
                    ("search_backtracks", "backtracks"),
                ):
                    counts[key] += getattr(pass_result.stats, stat_name)
                counts["complete_assignments_checked"] += pass_result.stats.complete_assignments_checked
                counts["collision_valid_assignments"] += pass_result.stats.collision_valid_assignments
                counts["search_budget_exhausted"] = pass_result.stats.budget_exhausted
                if pass_result.composition is not None:
                    composition_report = pass_result.composition
                result = pass_result
                if pass_result.selected is not None or pass_result.stats.budget_exhausted:
                    break
            counts["relationship_rejections"] = dict(sorted(cumulative_rejections.items()))
            if result is not None and result.selected is not None:
                selected_by_id = {lot_id: candidate for lot_id, candidate in result.selected}
                # Rebuild the exact successful pass inputs through the same
                # canonical wrapper used by feasibility search.  The
                # improvement walk must never widen or reconstruct a later
                # domain, and it must consume the same positional matrix.
                successful_domain_width = search_passes[-1]["domain_width"]
                if successful_domain_width is None:
                    raise FrontageFitError(
                        "composition improvement invariant: successful domain width is missing")
                successful_domains, successful_domain_sizes = _composition_domains(
                    lot_results, successful_domain_width)
                if successful_domains is None:
                    raise FrontageFitError(
                        "composition improvement invariant: successful candidate domains are missing")
                successful_ordered, successful_compatibility, _ = _ordered_search_inputs(
                    lot_results, successful_domains)
                if successful_ordered is None or successful_compatibility is None:
                    raise FrontageFitError(
                        "composition improvement invariant: ordered results or compatibility matrix is missing")
                feasibility_incumbent = result.selected
                feasibility_composition = result.composition
            elif result is not None and result.stats.budget_exhausted:
                unresolved = sorted(item.lot["id"] for item in lot_results)
                terminal_failure = "search_budget_exhausted"
            elif cumulative_collision_valid > 0:
                unresolved = sorted(item.lot["id"] for item in lot_results)
                terminal_failure = "global_relationship_unsatisfied"
            else:
                unresolved = sorted(item.lot["id"] for item in lot_results)
                terminal_failure = "global_collision_unsatisfied"
            # The loop's final result contains only the last pass's local
            # stats; the report uses the cumulative counters above.
            if result is not None and result.stats.budget_exhausted:
                counts["search_budget_exhausted"] = True
        if not composition_enabled and result.selected is None:
            # Plan §8: exhaustive proof of no complete assignment
            # (global_collision_unsatisfied) and inconclusive node-budget
            # exhaustion (search_budget_exhausted) both emit no resolved
            # assignment, so every lot id is unresolved; status and terminal
            # code distinguish proof from inconclusion.
            unresolved = sorted(item.lot["id"] for item in lot_results)
            terminal_failure = ("search_budget_exhausted" if result.stats.budget_exhausted
                                else "global_collision_unsatisfied")

    if composition_enabled and terminal_failure is None:
        # A solved composition pass always has a hard-valid incumbent and the
        # exact retained domain/matrix that produced it.  The incumbent is
        # also evaluated here so disabled/fault-safe reporting has the same
        # objective component shape as a successful improvement result.
        if (
            successful_domain_width is None
            or successful_domains is None
            or successful_ordered is None
            or successful_compatibility is None
            or feasibility_incumbent is None
            or feasibility_composition is None
        ):
            raise FrontageFitError(
                "composition improvement invariant: successful feasibility inputs are incomplete")
        _, incumbent_components = _assignment_preference(
            intent_copy, feasibility_composition, feasibility_incumbent)
        selected_for_report = feasibility_incumbent
        selected_composition = feasibility_composition
        selected_components = incumbent_components
        improvement_enabled = config.improvement_node_budget > 0
        improvement_nodes = 0
        improvement_extensions = 0
        improvement_collision_valid = 0
        improvement_hard_valid = 0
        improvement_relationship_rejections: Mapping[str, int] = {}
        improvement_budget_exhausted = False
        improvement_incumbent_improved = False
        improvement_faulted = False
        improvement_fault_code: str | None = None
        improvement_domain_sizes = successful_domain_sizes

        if improvement_enabled:
            # _improve_assignment is internally fault-safe.  Keep this outer
            # guard as well: an unexpected integration fault must not turn an
            # already solved feasibility result into an unsolved fit.
            try:
                improvement = _improve_assignment(
                    intent_copy,
                    successful_ordered,
                    successful_domains,
                    successful_compatibility,
                    feasibility_incumbent,
                    config,
                )
            except Exception:  # noqa: BLE001 - preserve the hard-valid incumbent
                improvement = None
                improvement_faulted = True
                improvement_fault_code = "improvement_exception"
            if improvement is not None:
                improvement_nodes = improvement.stats.nodes
                improvement_extensions = improvement.stats.extensions
                improvement_collision_valid = improvement.stats.collision_valid_assignments
                improvement_hard_valid = improvement.stats.hard_valid_assignments
                improvement_relationship_rejections = improvement.stats.relationship_rejections
                improvement_budget_exhausted = improvement.stats.budget_exhausted
                improvement_incumbent_improved = improvement.stats.incumbent_improved
                improvement_domain_sizes = improvement.stats.domain_sizes
                # A normal and fault-safe engine result always carries the
                # composition for its selected hard-valid assignment.  If an
                # unexpected result omits it, retain the feasibility pair.
                if improvement.composition is not None and improvement.objective_components:
                    selected_for_report = improvement.selected
                    selected_composition = improvement.composition
                    selected_components = improvement.objective_components

        selected_by_id = {lot_id: candidate for lot_id, candidate in selected_for_report}
        composition_report = selected_composition
        improvement_report = {
            "enabled": improvement_enabled,
            "faulted": improvement_faulted,
            "fault_code": improvement_fault_code,
            "domain_width": successful_domain_width,
            "domain_sizes": [
                {"lot_id": lot_id, "count": count}
                for lot_id, count in improvement_domain_sizes
            ],
            "node_budget": config.improvement_node_budget,
            "nodes": improvement_nodes,
            "extensions": improvement_extensions,
            "collision_valid_assignments": improvement_collision_valid,
            "hard_valid_assignments": improvement_hard_valid,
            "relationship_rejections": dict(sorted(improvement_relationship_rejections.items())),
            "budget_exhausted": improvement_budget_exhausted,
            "incumbent_improved": improvement_incumbent_improved,
            "incumbent_objective": incumbent_components,
            "selected_objective": selected_components,
        }
    if terminal_failure is None:
        status = "solved"
    elif terminal_failure == "search_budget_exhausted":
        status = "inconclusive"
    else:
        status = "unsatisfied"
    resolved = _resolved_sketch(intent_copy, selected_by_id) if status == "solved" else {
        "site": intent_copy["site"], "roads": intent_copy["roads"],
        "spaces": intent_copy["spaces"], "lots": [], "notes": intent_copy["notes"],
    }
    assignment_limitation = (
        "feasibility produces a first deterministic hard-valid incumbent; optional "
        "improvement is a bounded lexicographic search over the successful "
        "feasibility-pass domain, not all retained candidates or a global "
        "aesthetic optimum"
        if composition_enabled else
        "complete MRV/forward-check search returns the first deterministic feasible "
        "assignment, not a globally rank-optimal assignment"
    )
    legacy_count_keys = (
        "candidate_generated", "candidate_deduplicated", "candidate_unary_feasible",
        "search_nodes", "search_extensions", "compatibility_checks",
        "search_backtracks", "search_budget_exhausted",
    )
    report = {
        "schema_version": FIT_SCHEMA_VERSION,
        "product": FIT_PRODUCT,
        "site": site_name,
        "input_sha256": input_sha256,
        "config": config.to_dict(),
        "status": status,
        # Keep the old-intent report shape byte-semantic: Section 3B counters
        # are exposed only when authored composition declarations activate the
        # relationship path.
        "search_counts": ({key: counts[key] for key in legacy_count_keys}
                           if not composition_enabled else counts),
        "candidate_generated": counts["candidate_generated"],
        "candidate_deduplicated": counts["candidate_deduplicated"],
        "candidate_unary_feasible": counts["candidate_unary_feasible"],
        "search_nodes": counts["search_nodes"],
        "search_extensions": counts["search_extensions"],
        "compatibility_checks": counts["compatibility_checks"],
        "search_backtracks": counts["search_backtracks"],
        "search_budget_exhausted": counts["search_budget_exhausted"],
        "lots": [_report_lot(item, selected_by_id.get(item.lot["id"]))
                 for item in sorted(lot_results, key=lambda item: item.lot["id"])],
        "unresolved_lot_ids": sorted(unresolved),
        "terminal_failure_code": terminal_failure,
        "limitations": [
            "conservative exact 2D inter-building collision/contact rejection; plan-stage subterranean overlap excusal is not applied",
            "no semantic stamp selection",
            "no terrain pad or final ESP seating",
            "no road, space, district, stamp, role, or target invention",
            "plaza/court interior and buildability gates are intentionally stricter than current sketch advisories",
            "reach/safety/facing limit comparisons tolerate 1e-6 GU boundary float noise; values materially above the limit still fail",
            "--auto-face must not be applied to resolved fitter output",
            assignment_limitation,
            "search_budget_exhausted is inconclusive and never proves geometric unsatisfiability",
            "complete search is recursive with one Python frame per assigned lot; exceptionally large intents approaching the interpreter's recursion limit require partitioning (the limit is platform-dependent)",
        ],
    }
    if composition_enabled and status == "solved":
        report["composition"] = composition_report
        report["improvement"] = improvement_report
    if composition_enabled:
        report["candidate_unary_feasible_all"] = counts["candidate_unary_feasible_all"]
        report["search_passes"] = search_passes
        report["complete_assignments_checked"] = counts["complete_assignments_checked"]
        report["relationship_rejections"] = counts["relationship_rejections"]
        report["collision_valid_assignments"] = counts["collision_valid_assignments"]
    return intent_copy, resolved, report


def input_identity(data: bytes) -> str:
    """Return the one permitted intent identity hash for fit reports."""

    return hashlib.sha256(data).hexdigest()


__all__ = [
    "FIT_PRODUCT",
    "FitConfig",
    "FrontageFitError",
    "TerrainMask",
    "fit_intent",
    "input_identity",
    "validate_intent",
]
