"""Generate deterministic, measured wilderness scatter for the Falkreath block.

Pipeline position: this is the generation-core stage between the read-only
Vorndgad analysis products/direct ``tamriel.esm`` LAND records and later TES3
JSON authoring.  It deliberately does not author CELL/FRMR records.  A
placement is accepted only after its mesh quota, measured terrain envelopes,
transform samples, direct-LAND surface checks, and category-specific gates
pass.

The v6 contract retains the three narrow gen4 QA gates and adds one footprint
road gate to the core, while applying the accepted cleanup policy:
candidate centers capture the OpenMW-normalized raw VTEX value and reject the
configured road values for every scatter category; rock and cliff rotations
must leave the transformed local-up axis with positive world Z; and cliffs
must show the configured slope relief across their transformed world-AABB
footprint.  Rock and cliff transformed world-AABBs now also inspect every
globally aligned LAND 512-GU tile they intersect, rejecting configured road
VTEX values and missing LAND/VTEX data fail-closed.  All four gates are
deterministic and are represented in the output audits.  Existing
clearing/low-bank flora behavior, lighting, rock patch rules, stackers, and
cliff face chaining remain unchanged.  One-sided L_04 cliff shells and four
cluster rocks remain quarantined before quota allocation (open undersides need
pitch/roll seating yaw alone cannot provide).  Two rock meshes have a
15-degree local-up tilt ceiling; trees are emitted with sampled Z yaw and zero
X/Y tilt; and accepted large rock/cliff transformed AABBs reserve a 128-GU
margin around tree centers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import random
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .espland import (
    CELL_SIZE_GAME_UNITS,
    LAND_TEXTURE_SIDE,
    THU_TO_GU,
    LandRecord,
    height_at_game_position,
)
from .clearing_index import ClearingIndex, MultiClearingIndex, build_clearing_index
from .cliff_seating import CliffSeatingRuntime
from .scatter_analysis import (
    WaterDistanceIndex,
    bbox_info,
    normalize_mesh_key,
    rotate_tes3_reference_point,
    transformed_bbox,
)
from .seeds import derive_seed


TARGET_BOUNDS = (-95, -89, -11, -5)
# V6 deliberately retains the v3 random namespace and candidate IDs so its
# narrow QA gates do not reshuffle every unaffected placement.
GENERATION_NAMESPACE = "scatter-falkreath-v3"
OCCUPANCY_BIN_GU = 64.0
DEFAULT_CANDIDATE_SPACING_GU = 512.0
DEFAULT_JITTER_GU = 128.0
DEFAULT_WATER_SAMPLE_SPACING_GU = 128.0
DEFAULT_WATER_MARGIN_CELLS = 2
DEFAULT_SLOPE_SPACING_GU = 128.0
DEFAULT_CLIFF_MIN_SLOPE_DEG = 8.0
DEFAULT_MIN_DISTANCES_GU = {
    "flora": 128.0,
    "rocks": 512.0,
    "cliff": 3_072.0,
}
DEFAULT_CLEARING_GRID_GU = 4_096.0
DEFAULT_CLEARING_THRESHOLD = 0.74
DEFAULT_CLEARING_TREE_FACTOR = 0.20
DEFAULT_ROCK_DENSITY_FACTOR = 0.70
DEFAULT_ROCK_PATCH_GRID_GU = 3_072.0
DEFAULT_ROCK_PATCH_THRESHOLD = 0.34
DEFAULT_FLAT_ROCK_CAP_FACTOR = 0.35
DEFAULT_LOW_ROCK_CAP_FACTOR = 0.55
DEFAULT_LOW_ROCK_SLOPE_MAX_DEG = 16.0
DEFAULT_OPEN_FACE_MIN_ALIGNMENT = 0.35
DEFAULT_OPEN_FACE_ORIENTATION_ATTEMPTS = 18
DEFAULT_OPEN_FACE_NEIGHBOR_RADIUS_GU = 4_096.0
DEFAULT_OPEN_FACE_MIN_EMBED_GU = 4.0
DEFAULT_OPEN_SIDE_CLIFF_MIN_SLOPE_DEG = 24.0
DEFAULT_MIN_EMBED_GU = 1.0
DEFAULT_CLIFF_MIN_EMBED_GU = 64.0
DEFAULT_CLIFF_MIN_VISIBLE_GU = 32.0
LAND_TEXTURE_TILE_SIZE_GU = CELL_SIZE_GAME_UNITS / LAND_TEXTURE_SIDE

# These are explicit policy paths rather than filename heuristics.  The
# normalized keys are used at every comparison boundary so source spelling,
# slash direction, and case cannot bypass the cleanup rules.  L_04 shells are
# bottom-open front/top hulls; pose-first yaw does not seat them, so they stay
# quarantined until pitch/roll or terrain-shape burial exists.
QUARANTINED_MESH_PATHS = (
    # One-sided L_04 cliff shells: open bottom (+ often open side); upright yaw
    # placement exposes flat undersides on ordinary Falkreath slopes.
    r"Sky\f\Sky_TerrCliff_L_04_A1.nif",
    r"Sky\f\Sky_TerrCliff_L_04_A2.nif",
    r"Sky\f\Sky_TerrCliff_L_04_B1.nif",
    r"Sky\f\Sky_TerrCliff_L_04_B3.nif",
    r"Sky\f\Sky_TerrCliff_L_04_B4.nif",
    r"Sky\f\Sky_TerrCliff_L_04_D1.nif",
    r"Sky\f\Sky_TerrCliff_L_04_D2.nif",
    r"Sky\f\Sky_TerrCliff_L_04_E1.nif",
    r"Sky\f\Sky_TerrCliff_L_04_E2.nif",
    r"Sky\f\Sky_TerrCliff_L_04_F1.nif",
    r"Sky\f\Sky_TerrCliff_L_04_F2.nif",
    r"Sky\f\Sky_TerrCliff_L_04_G1.nif",
    r"Sky\f\Sky_TerrCliff_L_04_G2.nif",
    r"Sky\f\Sky_TerrCliff_L_04_H2.nif",
    r"Sky\f\Sky_TerrCliff_L_04_I1.nif",
    r"Sky\f\Sky_TerrCliff_L_04_I2.nif",
    r"Sky\f\Sky_TerrCliff_L_04_L1.nif",
    r"Sky\f\Sky_TerrCliff_L_04_L2.nif",
    r"Sky\f\Sky_TerrCliff_L_04_M1.nif",
    r"Sky\f\Sky_TerrCliff_L_04_M2.nif",
    # Rocks that expose open undersides or need cluster/slope support the
    # independent placer still cannot prove.
    r"Sky\f\Sky_TerrRock_04_H05.nif",
    r"Sky\f\Sky_TerrRock_04_017.nif",
    r"Sky\f\Sky_TerrRock_04_008.nif",
    r"Sky\f\Sky_TerrRock_04_084.nif",
)
QUARANTINED_MESH_KEYS = frozenset(normalize_mesh_key(path) for path in QUARANTINED_MESH_PATHS)
MAX_LOCAL_UP_TILT_DEGREES_BY_MESH = {
    normalize_mesh_key(r"Sky\f\Sky_TerrRock_LV_04_21.nif"): 15.0,
    normalize_mesh_key(r"Sky\f\Sky_TerrRock_04_027.nif"): 15.0,
}
TREE_CLEARANCE_MIN_HORIZONTAL_SPAN_GU = 1_024.0
TREE_CLEARANCE_MARGIN_GU = 128.0

WEIGHT_FUNCTION_DESCRIPTION = (
    "For each mesh and terrain dimension d, the measured [min,p10,p50,p90,max] "
    "envelope is a hard outside gate with cubic tail falloff and a Gaussian "
    "center factor; the candidate score is measured mesh frequency times the "
    "three mesh-specific memberships. Frequency quotas are allocated before "
    "candidate selection, so suitability cannot turn rare meshes into common ones. "
    "Road-designated raw VTEX candidate centers are filtered before weighting."
)


def _round(value: float | int | None, digits: int = 3) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    rounded = round(float(value), digits)
    return 0.0 if abs(rounded) < 0.5 * 10 ** (-digits) else rounded


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _distribution_from_values(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "max": None,
            "mean": None,
        }

    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = (len(ordered) - 1) * q
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {
        "count": len(ordered),
        "min": _round(ordered[0], 6),
        "p10": _round(quantile(0.10), 6),
        "p50": _round(quantile(0.50), 6),
        "p90": _round(quantile(0.90), 6),
        "max": _round(ordered[-1], 6),
        "mean": _round(sum(ordered) / len(ordered), 6),
    }


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _smoothstep_array(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _envelope_finite_params(
    envelope: Mapping[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    """Return ``(low, high, p10, p90, center, sigma)`` or ``None`` when empty."""

    values = {name: _finite(envelope.get(name)) for name in ("min", "p10", "p50", "p90", "max")}
    center = values["p50"]
    if center is None:
        return None
    low = values["min"] if values["min"] is not None else center
    high = values["max"] if values["max"] is not None else center
    if high < low:
        low, high = high, low
    if abs(high - low) <= 1e-9:
        # Degenerate envelope: callers treat this as a point match on center.
        return (low, high, low, high, center, 1.0)
    p10 = values["p10"] if values["p10"] is not None else low
    p90 = values["p90"] if values["p90"] is not None else high
    p10 = max(low, min(high, p10))
    p90 = max(low, min(high, p90))
    if p10 > p90:
        p10, p90 = p90, p10
    sigma = max((p90 - p10) / 2.0, (high - low) / 8.0, 1.0)
    return (float(low), float(high), float(p10), float(p90), float(center), float(sigma))


def envelope_weight(value: float, envelope: Mapping[str, Any]) -> float:
    """Return measured-envelope membership for one terrain dimension."""

    numeric = _finite(value)
    if numeric is None:
        return 0.0
    params = _envelope_finite_params(envelope)
    if params is None:
        return 0.0
    low, high, p10, p90, center, sigma = params
    if abs(high - low) <= 1e-9:
        return 1.0 if abs(numeric - center) <= 1e-6 else 0.0
    if numeric < low or numeric > high:
        return 0.0
    if numeric < p10 and p10 > low:
        edge = _smoothstep((numeric - low) / (p10 - low))
    elif numeric > p90 and high > p90:
        edge = 1.0 - _smoothstep((numeric - p90) / (high - p90))
    else:
        edge = 1.0
    core = math.exp(-0.5 * ((numeric - center) / sigma) ** 2)
    return max(0.0, min(1.0, edge * core))


def envelope_weight_array(values: np.ndarray, envelope: Mapping[str, Any]) -> np.ndarray:
    """Vectorized ``envelope_weight`` over a float64 value array."""

    output = np.zeros(values.shape, dtype=np.float64)
    if values.size == 0:
        return output
    params = _envelope_finite_params(envelope)
    if params is None:
        return output
    low, high, p10, p90, center, sigma = params
    numeric = values.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    if abs(high - low) <= 1e-9:
        output[finite & (np.abs(numeric - center) <= 1e-6)] = 1.0
        return output
    inside = finite & (numeric >= low) & (numeric <= high)
    if not np.any(inside):
        return output
    edge = np.ones(values.shape, dtype=np.float64)
    low_tail = inside & (numeric < p10) & (p10 > low)
    high_tail = inside & (numeric > p90) & (high > p90)
    if np.any(low_tail):
        edge[low_tail] = _smoothstep_array((numeric[low_tail] - low) / (p10 - low))
    if np.any(high_tail):
        edge[high_tail] = 1.0 - _smoothstep_array((numeric[high_tail] - p90) / (high - p90))
    core = np.exp(-0.5 * ((numeric - center) / sigma) ** 2)
    weighted = edge * core
    output[inside] = np.clip(weighted[inside], 0.0, 1.0)
    return output


@dataclass(frozen=True)
class _CandidateArrays:
    """Numeric columns for batch eligibility over one candidate list."""

    slope_deg: np.ndarray
    elevation_gu: np.ndarray
    water_distance_gu: np.ndarray
    terrain_z_thu: np.ndarray
    clearing_value: np.ndarray
    rock_patch_value: np.ndarray
    is_road: np.ndarray


def _build_candidate_arrays(
    candidates: Sequence[_Candidate], config: GenerationConfig
) -> _CandidateArrays:
    count = len(candidates)
    slope = np.empty(count, dtype=np.float64)
    elevation = np.empty(count, dtype=np.float64)
    water = np.empty(count, dtype=np.float64)
    terrain_z = np.empty(count, dtype=np.float64)
    clearing = np.empty(count, dtype=np.float64)
    rock_patch = np.empty(count, dtype=np.float64)
    is_road = np.zeros(count, dtype=bool)
    road_values = set(config.road_raw_vtex_values)
    for index, candidate in enumerate(candidates):
        slope[index] = candidate.slope_deg
        elevation[index] = candidate.terrain_z_gu
        water[index] = candidate.water_distance_gu
        terrain_z[index] = candidate.terrain_z_thu
        clearing[index] = candidate.clearing_value
        rock_patch[index] = candidate.rock_patch_value
        is_road[index] = candidate.raw_vtex is not None and candidate.raw_vtex in road_values
    return _CandidateArrays(
        slope_deg=slope,
        elevation_gu=elevation,
        water_distance_gu=water,
        terrain_z_thu=terrain_z,
        clearing_value=clearing,
        rock_patch_value=rock_patch,
        is_road=is_road,
    )


@dataclass(frozen=True)
class GenerationConfig:
    """Bounded deterministic configuration for one scatter block."""

    bounds: tuple[int, int, int, int] = TARGET_BOUNDS
    target_cells: frozenset[tuple[int, int]] | None = None
    scope_region: str = ""
    master_seed: int = 20260801
    candidate_spacing_gu: float = DEFAULT_CANDIDATE_SPACING_GU
    jitter_gu: float = DEFAULT_JITTER_GU
    water_sample_spacing_gu: float = DEFAULT_WATER_SAMPLE_SPACING_GU
    water_margin_cells: int = DEFAULT_WATER_MARGIN_CELLS
    cliff_min_slope_deg: float = DEFAULT_CLIFF_MIN_SLOPE_DEG
    # None is intentional: water distance is a measured per-mesh envelope,
    # never a global near-water rule.  The optional field keeps a caller's
    # old config loadable while the default remains data-driven.
    cliff_max_water_distance_gu: float | None = None
    min_distances_gu: Mapping[str, float] | None = None
    target_flora_per_cell: float | None = None
    target_rocks_per_cell: float | None = None
    target_cliffs_per_cell: float | None = None
    clearing_grid_gu: float = DEFAULT_CLEARING_GRID_GU
    clearing_threshold: float = DEFAULT_CLEARING_THRESHOLD
    clearing_tree_factor: float = DEFAULT_CLEARING_TREE_FACTOR
    rock_density_factor: float = DEFAULT_ROCK_DENSITY_FACTOR
    rock_patch_grid_gu: float = DEFAULT_ROCK_PATCH_GRID_GU
    rock_patch_threshold: float = DEFAULT_ROCK_PATCH_THRESHOLD
    flat_rock_cap_factor: float = DEFAULT_FLAT_ROCK_CAP_FACTOR
    low_rock_cap_factor: float = DEFAULT_LOW_ROCK_CAP_FACTOR
    low_rock_slope_max_deg: float = DEFAULT_LOW_ROCK_SLOPE_MAX_DEG
    open_face_min_alignment: float = DEFAULT_OPEN_FACE_MIN_ALIGNMENT
    open_face_orientation_attempts: int = DEFAULT_OPEN_FACE_ORIENTATION_ATTEMPTS
    open_face_neighbor_radius_gu: float = DEFAULT_OPEN_FACE_NEIGHBOR_RADIUS_GU
    open_face_min_embed_gu: float = DEFAULT_OPEN_FACE_MIN_EMBED_GU
    open_side_cliff_min_slope_deg: float = DEFAULT_OPEN_SIDE_CLIFF_MIN_SLOPE_DEG
    min_embed_gu: float = DEFAULT_MIN_EMBED_GU
    cliff_min_embed_gu: float = DEFAULT_CLIFF_MIN_EMBED_GU
    cliff_min_visible_gu: float = DEFAULT_CLIFF_MIN_VISIBLE_GU
    # Raw OpenMW VTEX values declared by the active region remap.  An empty
    # tuple is valid for a wilderness run with no road classes; there is no
    # global road raw value.
    road_raw_vtex_values: tuple[int, ...] = ()
    # Cleanup policy is stored as normalized mesh keys so callers cannot
    # accidentally make the rules case- or slash-direction-sensitive.
    quarantined_mesh_keys: frozenset[str] = QUARANTINED_MESH_KEYS
    max_local_up_tilt_degrees_by_mesh: Mapping[str, float] = field(
        default_factory=lambda: dict(MAX_LOCAL_UP_TILT_DEGREES_BY_MESH)
    )
    tree_clearance_min_horizontal_span_gu: float = TREE_CLEARANCE_MIN_HORIZONTAL_SPAN_GU
    tree_clearance_margin_gu: float = TREE_CLEARANCE_MARGIN_GU

    def __post_init__(self) -> None:
        min_x, max_x, min_y, max_y = self.bounds
        if min_x > max_x or min_y > max_y:
            raise ValueError("scatter bounds minimum must not exceed maximum")
        if self.candidate_spacing_gu <= 0 or self.jitter_gu < 0:
            raise ValueError("candidate spacing must be positive and jitter non-negative")
        if self.jitter_gu > self.candidate_spacing_gu / 2.0:
            raise ValueError("jitter must not exceed half the candidate spacing")
        if self.water_sample_spacing_gu <= 0 or self.water_margin_cells < 0:
            raise ValueError("water sampling configuration is invalid")
        if self.clearing_grid_gu <= 0 or not 0.0 <= self.clearing_threshold <= 1.0:
            raise ValueError("clearing mask configuration is invalid")
        if not 0.0 < self.clearing_tree_factor <= 1.0:
            raise ValueError("clearing_tree_factor must be in (0,1]")
        if not 0.0 < self.rock_density_factor <= 1.0:
            raise ValueError("rock_density_factor must be in (0,1]")
        if self.rock_patch_grid_gu <= 0 or not 0.0 <= self.rock_patch_threshold <= 1.0:
            raise ValueError("rock patch mask configuration is invalid")
        if not 0.0 < self.flat_rock_cap_factor <= 1.0 or not 0.0 < self.low_rock_cap_factor <= 1.0:
            raise ValueError("rock cap factors must be in (0,1]")
        if self.low_rock_slope_max_deg <= 0:
            raise ValueError("low_rock_slope_max_deg must be positive")
        if not -1.0 <= self.open_face_min_alignment <= 1.0:
            raise ValueError("open_face_min_alignment must be in [-1,1]")
        if self.open_face_orientation_attempts < 1 or self.open_face_neighbor_radius_gu <= 0:
            raise ValueError("open-face orientation configuration is invalid")
        if self.open_face_min_embed_gu < 0:
            raise ValueError("open_face_min_embed_gu must be non-negative")
        if (
            not math.isfinite(float(self.open_side_cliff_min_slope_deg))
            or float(self.open_side_cliff_min_slope_deg) < 0.0
            or float(self.open_side_cliff_min_slope_deg) >= 90.0
        ):
            raise ValueError("open_side_cliff_min_slope_deg must be finite degrees in [0,90)")
        if self.min_embed_gu < 0 or self.cliff_min_embed_gu < 0 or self.cliff_min_visible_gu < 0:
            raise ValueError("embedding thresholds must be non-negative")
        if self.tree_clearance_min_horizontal_span_gu <= 0 or self.tree_clearance_margin_gu < 0:
            raise ValueError("tree clearance thresholds are invalid")
        road_values = tuple(int(value) for value in self.road_raw_vtex_values)
        if any(value < 0 or value > 65535 for value in road_values):
            raise ValueError("road_raw_vtex_values must contain unsigned 16-bit values")
        object.__setattr__(self, "road_raw_vtex_values", road_values)
        quarantine_keys = frozenset(normalize_mesh_key(str(value)) for value in self.quarantined_mesh_keys)
        if any(not value for value in quarantine_keys):
            raise ValueError("quarantined mesh keys must be non-empty")
        object.__setattr__(self, "quarantined_mesh_keys", quarantine_keys)
        tilt_limits = {
            normalize_mesh_key(str(key)): float(value)
            for key, value in self.max_local_up_tilt_degrees_by_mesh.items()
        }
        if any(not key or not math.isfinite(value) or value < 0.0 or value >= 90.0 for key, value in tilt_limits.items()):
            raise ValueError("local-up tilt limits must be finite degrees in [0,90)")
        object.__setattr__(self, "max_local_up_tilt_degrees_by_mesh", tilt_limits)
        merged = dict(DEFAULT_MIN_DISTANCES_GU)
        if self.min_distances_gu is not None:
            merged.update({str(key): float(value) for key, value in self.min_distances_gu.items()})
        object.__setattr__(self, "min_distances_gu", merged)


@dataclass(frozen=True)
class _Species:
    key: str
    mesh: str
    category: str
    frequency: int
    conditions: Mapping[str, Mapping[str, Any]]
    scale_distribution: Mapping[str, Any]
    rotation_distribution: Mapping[str, Any]
    rotation_slope_dependency: tuple[Mapping[str, Any], ...]
    z_offset_distribution: Mapping[str, Any]
    stacker_rate: float
    shallow_water: bool
    measured: Mapping[str, Any]
    flora_role: str = "not_flora"
    open_face_profile: Mapping[str, Any] = None  # type: ignore[assignment]

    @property
    def z_offset_p50(self) -> float:
        return float(self.z_offset_distribution.get("p50") or 0.0)


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    cell: tuple[int, int]
    x_gu: float
    y_gu: float
    terrain_z_thu: float
    slope_deg: float
    water_distance_gu: float
    clearing_value: float = 0.0
    rock_patch_value: float = 1.0
    downhill_direction_xy: tuple[float, float] = (0.0, 0.0)
    # LandRecord.texture_indices is already in OpenMW row-major order.  None
    # is retained for synthetic/legacy LAND fixtures without a VTEX payload.
    raw_vtex: int | None = None

    @property
    def terrain_z_gu(self) -> float:
        return self.terrain_z_thu * THU_TO_GU


class OccupancyIndex:
    """Explicit 64-GU occupancy bitmap plus category distance checks."""

    def __init__(self, bin_size_gu: float = OCCUPANCY_BIN_GU) -> None:
        if bin_size_gu <= 0:
            raise ValueError("occupancy bin size must be positive")
        self.bin_size_gu = float(bin_size_gu)
        self._buckets: dict[str, dict[tuple[int, int], list[tuple[float, float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def _bin(self, value: float) -> int:
        return math.floor(float(value) / self.bin_size_gu)

    def can_place(self, category: str, point: Sequence[float], minimum_distance_gu: float) -> bool:
        if len(point) < 2 or minimum_distance_gu < 0:
            raise ValueError("occupancy checks need x/y and a non-negative distance")
        x, y = float(point[0]), float(point[1])
        bx, by = self._bin(x), self._bin(y)
        occupied = self._buckets[category]
        if (bx, by) in occupied:
            return False
        radius = int(math.ceil(float(minimum_distance_gu) / self.bin_size_gu)) + 1
        # When few bins are occupied (cliff pass, early rock pass), scanning
        # occupied bins is cheaper than walking a large empty Chebyshev square.
        # Flora's small radius stays on the square path.
        square_size = (2 * radius + 1) ** 2
        if len(occupied) < square_size:
            for (ix, iy), entries in occupied.items():
                if abs(ix - bx) > radius or abs(iy - by) > radius:
                    continue
                for other_x, other_y, other_distance in entries:
                    required = max(float(minimum_distance_gu), other_distance)
                    if math.hypot(x - other_x, y - other_y) < required:
                        return False
            return True
        for ix in range(bx - radius, bx + radius + 1):
            for iy in range(by - radius, by + radius + 1):
                for other_x, other_y, other_distance in occupied.get((ix, iy), ()):
                    required = max(float(minimum_distance_gu), other_distance)
                    if math.hypot(x - other_x, y - other_y) < required:
                        return False
        return True

    def add(self, category: str, point: Sequence[float], minimum_distance_gu: float) -> None:
        if len(point) < 2:
            raise ValueError("occupancy entries need x/y")
        x, y = float(point[0]), float(point[1])
        self._buckets[category][(self._bin(x), self._bin(y))].append(
            (x, y, float(minimum_distance_gu))
        )

    def count(self, category: str) -> int:
        return sum(len(rows) for rows in self._buckets[category].values())


def _species_role(mesh: str, category: str) -> str:
    if category != "flora":
        return "not_flora"
    # Sky's measured naming convention distinguishes trees/pines from bush and
    # large-grass/undergrowth families.  Unknown flora remains undergrowth so a
    # clearing never silently removes a measured foliage class.
    key = normalize_mesh_key(mesh).replace("/", "_").replace("\\", "_")
    if "_flora_tr_" in key or "_flora_p_" in key:
        return "tree"
    if any(token in key for token in ("_flora_bs_", "_flora_lg_", "undergrowth", "shrub", "bush", "grass")):
        return "undergrowth"
    return "undergrowth"


_DIRECTION_VECTORS_XY: dict[str, tuple[float, float]] = {
    "E": (1.0, 0.0),
    "NE": (math.sqrt(0.5), math.sqrt(0.5)),
    "N": (0.0, 1.0),
    "NW": (-math.sqrt(0.5), math.sqrt(0.5)),
    "W": (-1.0, 0.0),
    "SW": (-math.sqrt(0.5), -math.sqrt(0.5)),
    "S": (0.0, -1.0),
    "SE": (math.sqrt(0.5), -math.sqrt(0.5)),
}


def _default_open_face_profile() -> dict[str, Any]:
    """Backwards-compatible closed profile for small synthetic unit fixtures."""

    return {
        "status": "not_supplied",
        "open_directions": [],
        "closed_directions": list(_DIRECTION_VECTORS_XY),
        "has_open_geometry": False,
        "open_axes": [],
        "vertical_faces": {
            "up": {"open": False, "triangle_area_fraction": 1.0, "boundary_edge_length_gu": 0.0},
            "down": {"open": False, "triangle_area_fraction": 1.0, "boundary_edge_length_gu": 0.0},
        },
        "bottom_open": False,
        "top_open": False,
    }


def _open_face_profile_for_mesh(
    profiles_document: Mapping[str, Any] | None, mesh: str
) -> Mapping[str, Any]:
    if profiles_document is None:
        return _default_open_face_profile()
    rows = profiles_document.get("profiles", profiles_document)
    if not isinstance(rows, Mapping):
        raise ValueError("open-face profile sidecar has no profiles mapping")
    key = normalize_mesh_key(mesh)
    for stored, row in rows.items():
        if normalize_mesh_key(str(stored)) == key:
            if not isinstance(row, Mapping) or row.get("status") != "ok":
                raise ValueError(f"open-face profile is not a real measurement: {mesh}")
            open_directions = row.get("open_directions", [])
            if not isinstance(open_directions, list) or any(str(value) not in _DIRECTION_VECTORS_XY for value in open_directions):
                raise ValueError(f"open-face profile has invalid directions: {mesh}")
            vertical_faces = row.get("vertical_faces", {})
            if vertical_faces and (
                not isinstance(vertical_faces, Mapping)
                or any(axis not in vertical_faces for axis in ("up", "down"))
                or any(not isinstance(vertical_faces[axis], Mapping) for axis in ("up", "down"))
            ):
                raise ValueError(f"open-face profile has invalid vertical faces: {mesh}")
            return row
    raise ValueError(f"open-face profile sidecar is missing: {mesh}")


def _rotation_dependency_rows(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = row.get("rotation_slope_dependency", [])
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping) and int(item.get("count", 0)) > 0)


def _species_from_row(
    row: Mapping[str, Any],
    *,
    category: str | None = None,
    open_face_profile: Mapping[str, Any] | None = None,
) -> _Species:
    mesh = str(row.get("mesh", ""))
    actual_category = category or str(row.get("category", ""))
    if not mesh or actual_category not in {"flora", "rocks", "cliff"}:
        raise ValueError(f"invalid measured scatter species: {mesh!r}/{actual_category!r}")
    frequency = int(row.get("count", 0))
    if frequency <= 0:
        raise ValueError(f"scatter species has non-positive measured frequency: {mesh}")
    envelopes = row.get("condition_envelopes", {})
    if not isinstance(envelopes, Mapping):
        raise ValueError(f"scatter species has no condition envelopes: {mesh}")
    required = {"slope_deg", "elevation_gu", "water_distance_gu"}
    if not required.issubset(envelopes):
        raise ValueError(f"scatter species is missing terrain envelopes: {mesh}")
    scale = row.get("scale_distribution", {"min": 1.0, "p10": 1.0, "p50": 1.0, "p90": 1.0, "max": 1.0})
    rotations = row.get("rotation_distribution", {})
    z_offset = row.get("z_offset_distribution_gu", {"p50": 0.0})
    if not all(isinstance(value, Mapping) for value in (scale, rotations, z_offset)):
        raise ValueError(f"scatter species has malformed measured distributions: {mesh}")
    elevation = envelopes.get("elevation_gu", {})
    minimum_elevation = _finite(elevation.get("min")) if isinstance(elevation, Mapping) else None
    return _Species(
        key=normalize_mesh_key(mesh),
        mesh=mesh,
        category=actual_category,
        frequency=frequency,
        conditions={str(key): value for key, value in envelopes.items() if isinstance(value, Mapping)},
        scale_distribution=scale,
        rotation_distribution=rotations,
        rotation_slope_dependency=_rotation_dependency_rows(row),
        z_offset_distribution=z_offset,
        stacker_rate=max(0.0, min(1.0, float(row.get("stacked_count", 0)) / frequency)),
        shallow_water=minimum_elevation is not None and minimum_elevation <= 0.0,
        measured=row,
        flora_role=_species_role(mesh, actual_category),
        open_face_profile=open_face_profile or _default_open_face_profile(),
    )


def _rotation_from_cliff_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rotations = [row.get("rotation_radians", [0.0, 0.0, 0.0]) for row in rows]
    valid = [tuple(float(value) for value in rotation) for rotation in rotations if isinstance(rotation, Sequence) and len(rotation) == 3]
    modes = Counter("z_only" if abs(rotation[0]) <= 1e-6 and abs(rotation[1]) <= 1e-6 else "full" for rotation in valid)
    return {
        "mode_counts": dict(sorted(modes.items())),
        "x_radians": _distribution_from_values(rotation[0] for rotation in valid),
        "y_radians": _distribution_from_values(rotation[1] for rotation in valid),
        "z_radians": _distribution_from_values(rotation[2] for rotation in valid),
        "z_only_fraction": _round(modes.get("z_only", 0) / len(valid), 6) if valid else None,
    }


def _build_species(
    scatter_document: Mapping[str, Any],
    cliff_document: Mapping[str, Any],
    open_face_profiles: Mapping[str, Any] | None = None,
) -> tuple[list[_Species], list[_Species], set[str]]:
    species_rows = scatter_document.get("species_stats")
    cliff_summaries = cliff_document.get("giant_mesh_summary")
    if not isinstance(species_rows, Mapping) or not isinstance(cliff_summaries, Mapping):
        raise ValueError("analysis products are missing species_stats or giant_mesh_summary")
    giant_keys = {
        normalize_mesh_key(str(value.get("mesh", key)))
        for key, value in cliff_summaries.items()
        if isinstance(value, Mapping) and value.get("mesh")
    }
    main: list[_Species] = []
    for row in species_rows.values():
        if not isinstance(row, Mapping) or str(row.get("category", "")) not in {"flora", "rocks"}:
            continue
        profile = _species_from_row(
            row,
            open_face_profile=_open_face_profile_for_mesh(open_face_profiles, str(row.get("mesh", "")))
            if str(row.get("category", "")) == "rocks"
            else None,
        )
        if profile.key not in giant_keys:
            main.append(profile)

    giant_rows = cliff_document.get("giants", [])
    if not isinstance(giant_rows, list):
        raise ValueError("cliff analysis giants is not a list")
    rows_by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for value in giant_rows:
        if isinstance(value, Mapping) and value.get("mesh"):
            rows_by_key[normalize_mesh_key(str(value["mesh"]))].append(value)

    cliffs: list[_Species] = []
    for key in sorted(giant_keys):
        summary = cliff_summaries.get(key)
        if not isinstance(summary, Mapping):
            summary = next(
                (
                    value
                    for value in cliff_summaries.values()
                    if isinstance(value, Mapping) and normalize_mesh_key(str(value.get("mesh", ""))) == key
                ),
                None,
            )
        if not isinstance(summary, Mapping) or not summary.get("mesh"):
            continue
        mesh = str(summary["mesh"])
        rows = rows_by_key.get(key, [])
        rotation_distribution = summary.get("rotation_distribution")
        if not isinstance(rotation_distribution, Mapping):
            rotation_distribution = _rotation_from_cliff_rows(rows)
        cliff_row = {
            "mesh": mesh,
            "category": "cliff",
            "count": int(summary.get("count", len(rows))),
            "condition_envelopes": {
                "slope_deg": summary.get("slope_deg", {}),
                "elevation_gu": summary.get("elevation_gu", {}),
                "water_distance_gu": summary.get("water_distance_gu", {}),
            },
            "z_offset_distribution_gu": summary.get("z_offset_gu", {"p50": 0.0}),
            "scale_distribution": _distribution_from_values(
                float(value.get("scale", 1.0)) for value in rows if _finite(value.get("scale", 1.0)) is not None
            ) or {"min": 1.0, "p10": 1.0, "p50": 1.0, "p90": 1.0, "max": 1.0},
            "rotation_distribution": rotation_distribution,
            "rotation_slope_dependency": summary.get("rotation_slope_dependency", []),
            "stacked_count": 0,
        }
        cliffs.append(
            _species_from_row(
                cliff_row,
                category="cliff",
                open_face_profile=_open_face_profile_for_mesh(open_face_profiles, mesh),
            )
        )
    if not main:
        raise ValueError("analysis produced no non-giant scatter species")
    if not cliffs:
        raise ValueError("cliff analysis produced no giant palette")
    return (
        sorted(main, key=lambda item: (item.category, item.key)),
        sorted(cliffs, key=lambda item: item.key),
        giant_keys,
    )


def _sample_quantile(distribution: Mapping[str, Any], rng: random.Random, *, fallback: float = 1.0) -> float:
    names = ("min", "p10", "p50", "p90", "max")
    raw = [_finite(distribution.get(name)) for name in names]
    center = _finite(distribution.get("p50"))
    if center is None:
        center = fallback
    values = [center if value is None else value for value in raw]
    for index in range(1, len(values)):
        if values[index] < values[index - 1]:
            values[index] = values[index - 1]
    u = rng.random()
    knots = (0.0, 0.10, 0.50, 0.90, 1.0)
    for index in range(len(knots) - 1):
        if u <= knots[index + 1] or index == len(knots) - 2:
            fraction = (u - knots[index]) / (knots[index + 1] - knots[index])
            return values[index] + (values[index + 1] - values[index]) * fraction
    return values[-1]


def _rotation_bin(profile: _Species, slope: float) -> Mapping[str, Any] | None:
    for row in profile.rotation_slope_dependency:
        lower = _finite(row.get("slope_min_deg"))
        upper = _finite(row.get("slope_max_deg"))
        if lower is None or slope < lower:
            continue
        if upper is None or slope < upper:
            return row
    return None


def _sample_rotation(
    profile: _Species, rng: random.Random, slope_deg: float
) -> tuple[tuple[float, float, float], str, str]:
    conditioned = _rotation_bin(profile, float(slope_deg))
    distribution = conditioned if conditioned and int(conditioned.get("count", 0)) > 0 else profile.rotation_distribution
    mode_counts = distribution.get("mode_counts", {}) if isinstance(distribution, Mapping) else {}
    if not isinstance(mode_counts, Mapping):
        mode_counts = {}
    z_count = int(mode_counts.get("z_only", 0))
    full_count = int(mode_counts.get("full", 0))
    if full_count <= 0:
        mode = "z_only"
    elif z_count <= 0:
        mode = "full"
    else:
        mode = "full" if rng.random() * (z_count + full_count) >= z_count else "z_only"
    z = _sample_quantile(
        distribution.get("z_radians", {}) if isinstance(distribution, Mapping) else {},
        rng,
        fallback=0.0,
    )
    source = "measured_slope_bin" if conditioned and distribution is conditioned else "measured_mesh_distribution"
    if profile.flora_role == "tree":
        # Keep the measured yaw draw (and consume the measured XY draws when
        # the source mode was full) so the per-profile RNG stream remains
        # stable, but never allow a living tree to inherit source tilt.
        if mode == "full":
            _sample_quantile(distribution.get("x_radians", {}), rng, fallback=0.0)
            _sample_quantile(distribution.get("y_radians", {}), rng, fallback=0.0)
        return (0.0, 0.0, _round(z, 6)), "z_only", f"{source}+tree_upright_rule"
    if mode == "z_only":
        return (0.0, 0.0, _round(z, 6)), mode, source
    x = _sample_quantile(distribution.get("x_radians", {}), rng, fallback=0.0)
    y = _sample_quantile(distribution.get("y_radians", {}), rng, fallback=0.0)
    return (_round(x, 6), _round(y, 6), _round(z, 6)), mode, source  # type: ignore[return-value]


def _smooth_mask_value(
    x_gu: float,
    y_gu: float,
    master_seed: int,
    grid_gu: float,
    label: str,
    *,
    node_cache: dict[tuple[str, int, int], float] | None = None,
) -> float:
    if grid_gu <= 0:
        raise ValueError("smooth mask grid must be positive")
    gx = float(x_gu) / grid_gu
    gy = float(y_gu) / grid_gu
    ix, iy = math.floor(gx), math.floor(gy)
    tx, ty = _smoothstep(gx - ix), _smoothstep(gy - iy)

    def node(nx: int, ny: int) -> float:
        if node_cache is not None:
            key = (label, nx, ny)
            cached = node_cache.get(key)
            if cached is not None:
                return cached
            value = random.Random(derive_seed(master_seed, GENERATION_NAMESPACE, label, nx, ny)).random()
            node_cache[key] = value
            return value
        return random.Random(derive_seed(master_seed, GENERATION_NAMESPACE, label, nx, ny)).random()

    a = node(ix, iy)
    b = node(ix + 1, iy)
    c = node(ix, iy + 1)
    d = node(ix + 1, iy + 1)
    return (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty


def clearing_mask_value(
    x_gu: float,
    y_gu: float,
    master_seed: int,
    grid_gu: float = DEFAULT_CLEARING_GRID_GU,
    *,
    node_cache: dict[tuple[str, int, int], float] | None = None,
) -> float:
    """Return a smooth deterministic low-frequency clearing value in [0,1]."""

    return _smooth_mask_value(
        x_gu, y_gu, master_seed, grid_gu, "clearing", node_cache=node_cache
    )


def rock_patch_mask_value(
    x_gu: float,
    y_gu: float,
    master_seed: int,
    grid_gu: float = DEFAULT_ROCK_PATCH_GRID_GU,
    *,
    node_cache: dict[tuple[str, int, int], float] | None = None,
) -> float:
    """Return the deterministic smooth mask used to break up rock carpets."""

    return _smooth_mask_value(
        x_gu, y_gu, master_seed, grid_gu, "rock_patch", node_cache=node_cache
    )


_SLOPE_NEIGHBOR_DIRS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)


def downhill_direction_xy(
    land_records: Mapping[tuple[int, int], LandRecord],
    position: Sequence[float],
    *,
    spacing_game_units: float = DEFAULT_SLOPE_SPACING_GU,
) -> tuple[float, float]:
    """Return a normalized vector toward the steepest measured ESM-LAND drop."""

    _, _, downhill = _direct_land_slope_and_downhill(
        land_records,
        position,
        spacing_game_units=spacing_game_units,
    )
    return downhill


def _direct_land_slope_and_downhill(
    land_records: Mapping[tuple[int, int], LandRecord],
    position: Sequence[float],
    *,
    spacing_game_units: float = DEFAULT_SLOPE_SPACING_GU,
) -> tuple[float | None, float | None, tuple[float, float]]:
    """Sample center + eight neighbors once; return center THU, slope, downhill.

    Matches ``terrain_slope_deg`` and ``downhill_direction_xy`` independently so
    candidate build can avoid sampling the same stencil twice.
    """

    if spacing_game_units <= 0 or not math.isfinite(float(spacing_game_units)):
        raise ValueError("slope spacing must be finite and positive")
    center_thu = height_at_game_position(land_records, position[:2])
    if center_thu is None:
        return None, None, (0.0, 0.0)
    center_gu = float(center_thu) * THU_TO_GU
    spacing = float(spacing_game_units)
    maximum_slope: float | None = None
    best_downhill: tuple[float, float, float] | None = None
    for dx, dy in _SLOPE_NEIGHBOR_DIRS:
        neighbor = (
            float(position[0]) + dx * spacing,
            float(position[1]) + dy * spacing,
        )
        neighbor_thu = height_at_game_position(land_records, neighbor)
        if neighbor_thu is None:
            continue
        neighbor_gu = float(neighbor_thu) * THU_TO_GU
        rise_gu = abs(neighbor_gu - center_gu)
        run_gu = math.hypot(dx * spacing, dy * spacing)
        slope = math.degrees(math.atan2(rise_gu, run_gu))
        maximum_slope = slope if maximum_slope is None else max(maximum_slope, slope)
        drop = center_gu - neighbor_gu
        if drop <= 0.0:
            continue
        length = math.hypot(dx, dy)
        ranked = (drop / length, float(dx) / length, float(dy) / length)
        if best_downhill is None or ranked[0] > best_downhill[0]:
            best_downhill = ranked
    downhill = (
        (_round(best_downhill[1], 6), _round(best_downhill[2], 6))
        if best_downhill is not None
        else (0.0, 0.0)
    )
    return float(center_thu), maximum_slope, downhill  # type: ignore[return-value]


def _build_water_index(
    land_records: Mapping[tuple[int, int], LandRecord], config: GenerationConfig
) -> tuple[WaterDistanceIndex, dict[str, Any]]:
    min_x, max_x, min_y, max_y = config.bounds
    window = (
        min_x - config.water_margin_cells,
        max_x + config.water_margin_cells,
        min_y - config.water_margin_cells,
        max_y + config.water_margin_cells,
    )
    spacing = float(config.water_sample_spacing_gu)
    points: list[tuple[float, float]] = []
    for cell_y in range(window[2], window[3] + 1):
        for cell_x in range(window[0], window[1] + 1):
            if (cell_x, cell_y) not in land_records:
                continue
            for local_y in range(int(spacing / 2.0), int(CELL_SIZE_GAME_UNITS), int(spacing)):
                for local_x in range(int(spacing / 2.0), int(CELL_SIZE_GAME_UNITS), int(spacing)):
                    position = (cell_x * CELL_SIZE_GAME_UNITS + local_x, cell_y * CELL_SIZE_GAME_UNITS + local_y)
                    height = height_at_game_position(land_records, position)
                    if height is not None and height <= 0.0:
                        points.append(position)
    if not points:
        raise ValueError("expanded tamriel.esm LAND window contains no water samples")
    index = WaterDistanceIndex(
        points,
        sample_spacing_gu=spacing,
        threshold_thu=0.0,
        window_cells=list(window),
        source="tamriel.esm LAND via procgen.espland",
        require_tree=False,
    )
    return index, {
        "threshold_thu": 0.0,
        "sample_spacing_gu": spacing,
        "sample_count": len(points),
        "window_cells": list(window),
        "source": "tamriel.esm LAND via procgen.espland",
        "distance_metric": "euclidean game-unit distance to nearest direct-LAND sample with terrain <= 0 THU",
    }


def _build_candidates(
    land_records: Mapping[tuple[int, int], LandRecord],
    water_index: WaterDistanceIndex,
    config: GenerationConfig,
) -> tuple[dict[tuple[int, int], list[_Candidate]], dict[str, Any]]:
    """Build jittered candidate centers and capture their raw LAND VTEX values.

    ``LandRecord.texture_indices`` has already gone through the OpenMW
    serialized-VTEX transpose.  The local candidate coordinates therefore map
    directly to ``tile_y * 16 + tile_x`` through ``LandRecord.texture_index``;
    no LTEX lookup or zero-based conversion belongs in this stage.  A missing
    VTEX payload is retained as ``None`` for fixture compatibility and is
    counted in the candidate audit rather than silently treated as road.
    """
    min_x, max_x, min_y, max_y = config.bounds
    candidates: dict[tuple[int, int], list[_Candidate]] = {}
    spacing = float(config.candidate_spacing_gu)
    columns = int(CELL_SIZE_GAME_UNITS // spacing)
    total = 0
    above = 0
    below = 0
    slopes: list[float] = []
    raw_vtex_counts: Counter[str] = Counter()
    road_candidate_count = 0
    missing_texture_count = 0
    mask_node_cache: dict[tuple[str, int, int], float] = {}
    for cell_y in range(min_y, max_y + 1):
        for cell_x in range(min_x, max_x + 1):
            cell = (cell_x, cell_y)
            if config.target_cells is not None and cell not in config.target_cells:
                continue
            record = land_records.get(cell)
            if record is None or not record.has_heights:
                raise ValueError(f"target cell {cell} has no VHGT in tamriel.esm LAND")
            rng = random.Random(derive_seed(config.master_seed, GENERATION_NAMESPACE, "candidate", cell_x, cell_y))
            rows: list[_Candidate] = []
            ordinal = 0
            for row in range(columns):
                for column in range(columns):
                    local_x = (column + 0.5) * spacing + rng.uniform(-config.jitter_gu, config.jitter_gu)
                    local_y = (row + 0.5) * spacing + rng.uniform(-config.jitter_gu, config.jitter_gu)
                    local_x = max(1.0, min(CELL_SIZE_GAME_UNITS - 1.0, local_x))
                    local_y = max(1.0, min(CELL_SIZE_GAME_UNITS - 1.0, local_y))
                    x_gu = cell_x * CELL_SIZE_GAME_UNITS + local_x
                    y_gu = cell_y * CELL_SIZE_GAME_UNITS + local_y
                    terrain_z, slope, downhill = _direct_land_slope_and_downhill(
                        land_records,
                        (x_gu, y_gu),
                        spacing_game_units=DEFAULT_SLOPE_SPACING_GU,
                    )
                    if terrain_z is None or slope is None:
                        continue
                    tile_x = min(
                        LAND_TEXTURE_SIDE - 1,
                        max(0, int(local_x / (CELL_SIZE_GAME_UNITS / LAND_TEXTURE_SIDE))),
                    )
                    tile_y = min(
                        LAND_TEXTURE_SIDE - 1,
                        max(0, int(local_y / (CELL_SIZE_GAME_UNITS / LAND_TEXTURE_SIDE))),
                    )
                    raw_vtex = record.texture_index(tile_x, tile_y) if record.has_textures else None
                    raw_vtex_counts[str(raw_vtex) if raw_vtex is not None else "missing"] += 1
                    if raw_vtex is None:
                        missing_texture_count += 1
                    elif raw_vtex in config.road_raw_vtex_values:
                        road_candidate_count += 1
                    candidate = _Candidate(
                        candidate_id=f"c3_{cell_x}_{cell_y}_{ordinal:04d}",
                        cell=cell,
                        x_gu=_round(x_gu),  # type: ignore[arg-type]
                        y_gu=_round(y_gu),  # type: ignore[arg-type]
                        terrain_z_thu=_round(terrain_z),  # type: ignore[arg-type]
                        slope_deg=_round(slope),  # type: ignore[arg-type]
                        water_distance_gu=_round(water_index.distance_gu((x_gu, y_gu))),  # type: ignore[arg-type]
                        clearing_value=clearing_mask_value(
                            x_gu,
                            y_gu,
                            config.master_seed,
                            config.clearing_grid_gu,
                            node_cache=mask_node_cache,
                        ),
                        rock_patch_value=rock_patch_mask_value(
                            x_gu,
                            y_gu,
                            config.master_seed,
                            config.rock_patch_grid_gu,
                            node_cache=mask_node_cache,
                        ),
                        downhill_direction_xy=downhill,
                        raw_vtex=raw_vtex,
                    )
                    rows.append(candidate)
                    ordinal += 1
                    total += 1
                    slopes.append(float(slope))
                    if terrain_z > 0:
                        above += 1
                    else:
                        below += 1
            candidates[cell] = rows
    return candidates, {
        "candidate_spacing_gu": spacing,
        "jitter_gu": float(config.jitter_gu),
        "candidate_count": total,
        "above_water_candidate_count": above,
        "at_or_below_water_candidate_count": below,
        "slope_distribution_deg": _distribution_from_values(slopes),
        "raw_vtex": {
            "source": "tamriel.esm LAND via procgen.espland LandRecord.texture_index",
            "ordering": "OpenMW-normalized row-major VTEX (tile_y * 16 + tile_x)",
            "candidate_center_tile_rule": "floor(local_game_coordinate / (8192 / 16)), clamped to 0..15",
            "captured_candidate_count": total - missing_texture_count,
            "missing_texture_count": missing_texture_count,
            "value_counts": dict(sorted(raw_vtex_counts.items(), key=lambda item: item[0])),
            "road_raw_vtex_values": list(config.road_raw_vtex_values),
            "road_candidate_count": road_candidate_count,
        },
    }


def _candidate_has_road_raw_vtex(candidate: _Candidate, config: GenerationConfig) -> bool:
    """Return whether a candidate center is on a configured raw-VTEX road tile."""

    return candidate.raw_vtex is not None and candidate.raw_vtex in config.road_raw_vtex_values


def _candidate_condition_weight(candidate: _Candidate, profile: _Species) -> float:
    if candidate.terrain_z_thu <= 0.0 and not profile.shallow_water:
        return 0.0
    return profile.frequency * math.prod(
        envelope_weight(value, profile.conditions[name])
        for name, value in (
            ("slope_deg", candidate.slope_deg),
            ("elevation_gu", candidate.terrain_z_gu),
            ("water_distance_gu", candidate.water_distance_gu),
        )
    )


def _bbox_row(cache: Mapping[str, Any], mesh: str) -> Mapping[str, Any]:
    rows = cache.get("meshes")
    if not isinstance(rows, Mapping):
        raise ValueError("bbox cache has no meshes mapping")
    key = normalize_mesh_key(mesh)
    for stored, row in rows.items():
        if normalize_mesh_key(str(stored)) == key:
            if not isinstance(row, Mapping) or row.get("status") != "ok" or row.get("fallback"):
                raise ValueError(f"bbox cache entry is not a real measurement: {mesh}")
            return row
    raise ValueError(f"bbox cache is missing a real measurement for: {mesh}")


def _quota_counts(total: int, profiles: Sequence[_Species], *, eligible: set[str]) -> dict[str, int]:
    if total <= 0 or not profiles:
        return {profile.key: 0 for profile in profiles}
    active = [profile for profile in profiles if profile.key in eligible]
    if not active:
        return {profile.key: 0 for profile in profiles}
    frequency_total = sum(profile.frequency for profile in active)
    raw = {profile.key: total * profile.frequency / frequency_total for profile in active}
    counts = {profile.key: int(math.floor(value)) for profile, value in ((p, raw[p.key]) for p in active)}
    remaining = total - sum(counts.values())
    order = sorted(
        active,
        key=lambda profile: (-(raw[profile.key] - counts[profile.key]), profile.key),
    )
    for profile in order[:remaining]:
        counts[profile.key] += 1
    for profile in profiles:
        counts.setdefault(profile.key, 0)
    return counts


def _flora_quota_counts(
    baseline_total: int,
    profiles: Sequence[_Species],
    *,
    eligible: set[str],
    undergrowth_multiplier: float = 1.25,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Allocate tree and undergrowth quotas from measured source frequency.

    The baseline is split by the retained measured tree/undergrowth frequency
    before any integer allocation.  The tree target is copied unchanged; only
    the undergrowth floating target is multiplied.  Each role then receives a
    deterministic largest-remainder per-mesh allocation, so ineligible meshes
    redistribute quota only within their own flora role.
    """

    if undergrowth_multiplier <= 0.0 or not math.isfinite(float(undergrowth_multiplier)):
        raise ValueError("undergrowth multiplier must be finite and positive")
    role_profiles = {
        "tree": [profile for profile in profiles if profile.flora_role == "tree"],
        "undergrowth": [profile for profile in profiles if profile.flora_role != "tree"],
    }
    frequencies = {
        role: sum(profile.frequency for profile in rows)
        for role, rows in role_profiles.items()
    }
    frequency_total = sum(frequencies.values())
    baseline_targets = {
        role: float(baseline_total) * frequencies[role] / frequency_total if frequency_total else 0.0
        for role in role_profiles
    }
    adjusted_targets = {
        "tree": baseline_targets["tree"],
        "undergrowth": baseline_targets["undergrowth"] * float(undergrowth_multiplier),
    }
    baseline_integer_targets = {
        role: int(round(value)) for role, value in baseline_targets.items()
    }
    integer_targets = {
        "tree": baseline_integer_targets["tree"],
        "undergrowth": int(round(adjusted_targets["undergrowth"])),
    }
    quotas: dict[str, int] = {}
    for role, rows in role_profiles.items():
        role_quotas = _quota_counts(
            integer_targets[role],
            rows,
            eligible={profile.key for profile in rows if profile.key in eligible},
        )
        quotas.update(role_quotas)
    return quotas, {
        "measured_source_frequency": {
            "tree": frequencies["tree"],
            "undergrowth": frequencies["undergrowth"],
            "total": frequency_total,
            "tree_share": _round(frequencies["tree"] / frequency_total, 9) if frequency_total else 0.0,
            "undergrowth_share": _round(frequencies["undergrowth"] / frequency_total, 9)
            if frequency_total
            else 0.0,
        },
        "baseline_targets_before_integer_allocation": {
            "tree_refs": _round(baseline_targets["tree"], 9),
            "undergrowth_refs": _round(baseline_targets["undergrowth"], 9),
            "flora_refs": int(baseline_total),
        },
        "adjusted_targets_before_integer_allocation": {
            "tree_refs": _round(adjusted_targets["tree"], 9),
            "undergrowth_refs": _round(adjusted_targets["undergrowth"], 9),
            "flora_refs": _round(sum(adjusted_targets.values()), 9),
        },
        "integer_targets": {
            "baseline_tree_refs": baseline_integer_targets["tree"],
            "baseline_undergrowth_refs": baseline_integer_targets["undergrowth"],
            "tree_refs": integer_targets["tree"],
            "undergrowth_refs": integer_targets["undergrowth"],
            "flora_refs": sum(integer_targets.values()),
        },
        "undergrowth_multiplier": float(undergrowth_multiplier),
        "tree_target_unchanged": adjusted_targets["tree"] == baseline_targets["tree"]
        and integer_targets["tree"] == baseline_integer_targets["tree"],
        "undergrowth_adjustment_exact": adjusted_targets["undergrowth"]
        == baseline_targets["undergrowth"] * float(undergrowth_multiplier),
    }


def _cell_targets(total: int, cells: Sequence[tuple[int, int]], seed: int, label: str) -> dict[tuple[int, int], int]:
    if not cells:
        return {}
    base, remainder = divmod(int(total), len(cells))
    order = sorted(cells, key=lambda cell: derive_seed(seed, "scatter-falkreath-v2", label, cell[0], cell[1]))
    return {cell: base + (1 if index < remainder else 0) for index, cell in enumerate(order)}


def _rock_density_caps(
    scatter_analysis: Mapping[str, Any], config: GenerationConfig | None = None
) -> dict[str, Any]:
    config = config or GenerationConfig()
    density = scatter_analysis.get("density", {})
    raw = density.get("rock_density_by_slope", {}) if isinstance(density, Mapping) else {}
    bins = raw.get("bins", []) if isinstance(raw, Mapping) else []
    parsed: list[dict[str, Any]] = []
    if isinstance(bins, list):
        for row in bins:
            if not isinstance(row, Mapping):
                continue
            lower = _finite(row.get("slope_min_deg"))
            upper = _finite(row.get("slope_max_deg"))
            cap = row.get("hard_cap_refs_per_cell")
            if lower is None or cap is None:
                continue
            parsed.append(
                {
                    "slope_min_deg": lower,
                    "slope_max_deg": upper,
                    "hard_cap_refs_per_cell": max(0, int(cap)),
                }
            )
    parsed.sort(key=lambda row: float(row["slope_min_deg"]))
    flat_source = next(
        (row["hard_cap_refs_per_cell"] for row in parsed if float(row["slope_min_deg"]) < 8.0),
        None,
    )
    low_source = next(
        (row["hard_cap_refs_per_cell"] for row in parsed if float(row["slope_min_deg"]) >= 8.0),
        None,
    )
    flat_cap = (
        max(1, int(round(float(flat_source) * config.flat_rock_cap_factor)))
        if flat_source is not None
        else None
    )
    low_cap = (
        max(1, int(round(float(low_source) * config.low_rock_cap_factor)))
        if low_source is not None
        else None
    )
    return {
        "bins": parsed,
        "source": "output/vorndgad_scatter_analysis.json density.rock_density_by_slope",
        "flat_slope_max_deg": 8.0,
        "low_slope_max_deg": config.low_rock_slope_max_deg,
        "flat_cap_factor": config.flat_rock_cap_factor,
        "low_cap_factor": config.low_rock_cap_factor,
        "measured_flat_cap_refs_per_cell": flat_source,
        "measured_low_cap_refs_per_cell": low_source,
        "flat_cap_refs_per_cell": flat_cap,
        "low_cap_refs_per_cell": low_cap,
        "rule": "on a target cell, all rocks below 8 degrees share the reduced flat cap; rocks below the low-slope threshold share the reduced low cap; steeper bands retain their measured p90 hard cap",
    }


def _rock_cap_for_slope(slope_deg: float, caps: Mapping[str, Any]) -> int | None:
    if float(slope_deg) < float(caps.get("flat_slope_max_deg", 8.0)):
        flat_cap = caps.get("flat_cap_refs_per_cell")
        if flat_cap is not None:
            return max(0, int(flat_cap))
    elif float(slope_deg) < float(caps.get("low_slope_max_deg", 16.0)):
        low_cap = caps.get("low_cap_refs_per_cell")
        if low_cap is not None:
            return max(0, int(low_cap))
    bins = caps.get("bins", [])
    if not isinstance(bins, list):
        return None
    for row in bins:
        if not isinstance(row, Mapping):
            continue
        lower = _finite(row.get("slope_min_deg"))
        upper = _finite(row.get("slope_max_deg"))
        if lower is None or float(slope_deg) < lower:
            continue
        if upper is None or float(slope_deg) < upper:
            return max(0, int(row.get("hard_cap_refs_per_cell", 0)))
    return None


def _surface_intersects_terrain(
    world_bbox: Mapping[str, Sequence[float]], terrain_z_gu: float, *, minimum_embed: float
) -> tuple[bool, float, float]:
    minimum = world_bbox.get("min")
    maximum = world_bbox.get("max")
    if not isinstance(minimum, Sequence) or not isinstance(maximum, Sequence):
        return False, 0.0, 0.0
    bottom_embed = float(terrain_z_gu) - float(minimum[2])
    top_above = float(maximum[2]) - float(terrain_z_gu)
    return (
        bottom_embed >= float(minimum_embed) and top_above >= 0.0,
        bottom_embed,
        top_above,
    )


def transformed_local_up_world_z(rotation: Sequence[float]) -> float:
    """Return local ``+Z``'s world-Z component under the TES3 ref transform.

    The authoritative static-reference matrix is
    ``Rx(-rx) @ Ry(-ry) @ Rz(-rz)``.  Applying it to ``(0, 0, 1)`` gives
    ``cos(rotation_x) * cos(rotation_y)``; the rightmost Z yaw cannot change
    the component.  Keeping this as a pure helper makes the flipped-rock gate
    independently testable and prevents a blanket ban on ordinary authored
    X/Y tilt.
    """

    if len(rotation) != 3:
        raise ValueError("rotation must contain exactly three Euler components")
    rx, ry = float(rotation[0]), float(rotation[1])
    return math.cos(rx) * math.cos(ry)


def transformed_local_up_tilt_degrees(rotation: Sequence[float]) -> float:
    """Return the angle between transformed local ``+Z`` and world ``+Z``.

    The value is derived from the same local-up component used by the existing
    positive-up gate.  Clamping protects the audit from a one-ulp acos domain
    error while keeping the configured 15-degree limit a strict degree limit.
    """

    local_up_z = max(-1.0, min(1.0, transformed_local_up_world_z(rotation)))
    return math.degrees(math.acos(local_up_z))


def _cliff_perimeter_points(world_bbox: Mapping[str, Sequence[float]]) -> list[tuple[float, float]]:
    """Return four world-AABB corners followed by four edge midpoints."""

    minimum = world_bbox.get("min")
    maximum = world_bbox.get("max")
    if (
        not isinstance(minimum, Sequence)
        or not isinstance(maximum, Sequence)
        or len(minimum) < 2
        or len(maximum) < 2
    ):
        return []
    min_x, min_y = float(minimum[0]), float(minimum[1])
    max_x, max_y = float(maximum[0]), float(maximum[1])
    mid_x, mid_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    return [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (mid_x, min_y),
        (max_x, mid_y),
        (mid_x, max_y),
        (min_x, mid_y),
    ]


def cliff_footprint_relief_audit(
    land_records: Mapping[tuple[int, int], LandRecord],
    world_bbox: Mapping[str, Sequence[float]],
    cliff_min_slope_deg: float,
) -> dict[str, Any]:
    """Measure the eight-point direct-LAND relief across one cliff AABB.

    The result is deliberately self-contained so accepted cliff refs can carry
    their exact sample coordinates, heights, span, threshold, and margin.  A
    missing LAND sample, non-positive footprint span, non-finite threshold, or
    insufficient relief is a failed audit; callers decide whether to retry the
    sampled orientation.  Heights are converted from authoritative THU to GU
    only for the reported relief calculation.
    """

    minimum = world_bbox.get("min")
    maximum = world_bbox.get("max")
    points = _cliff_perimeter_points(world_bbox)
    audit: dict[str, Any] = {
        "terrain_source": "tamriel.esm LAND via procgen.espland",
        "sample_layout": "4 world-AABB corners followed by 4 edge midpoints",
        "sample_count": len(points),
        "samples": [],
        "world_aabb_gu": {
            "min": list(minimum) if isinstance(minimum, Sequence) else None,
            "max": list(maximum) if isinstance(maximum, Sequence) else None,
        },
        "footprint_width_gu": None,
        "footprint_depth_gu": None,
        "footprint_span_gu": None,
        "cliff_min_slope_deg": _round(float(cliff_min_slope_deg), 6)
        if _finite(cliff_min_slope_deg) is not None
        else None,
        "required_relief_gu": None,
        "observed_relief_gu": None,
        "relief_margin_gu": None,
        "passed": False,
        "failure_reason": None,
    }
    if len(points) != 8:
        audit["failure_reason"] = "degenerate_or_invalid_aabb"
        return audit

    min_x, min_y = float(minimum[0]), float(minimum[1])  # type: ignore[index]
    max_x, max_y = float(maximum[0]), float(maximum[1])  # type: ignore[index]
    width = max_x - min_x
    depth = max_y - min_y
    span = min(width, depth)
    audit["footprint_width_gu"] = _round(width)
    audit["footprint_depth_gu"] = _round(depth)
    audit["footprint_span_gu"] = _round(span)
    slope = _finite(cliff_min_slope_deg)
    if not all(math.isfinite(value) for value in (width, depth, span)) or span <= 0.0:
        audit["failure_reason"] = "degenerate_footprint"
        return audit
    if slope is None or slope < 0.0 or slope >= 90.0:
        audit["failure_reason"] = "invalid_relief_threshold"
        return audit
    required = math.tan(math.radians(slope)) * span
    audit["required_relief_gu"] = _round(required)

    heights_gu: list[float] = []
    for x_gu, y_gu in points:
        height_thu = height_at_game_position(land_records, (x_gu, y_gu))
        numeric_thu = _finite(height_thu)
        sample: dict[str, Any] = {
            "position_gu": [_round(x_gu), _round(y_gu)],
            "terrain_z_thu": _round(numeric_thu),
            "terrain_z_gu": _round(numeric_thu * THU_TO_GU) if numeric_thu is not None else None,
        }
        audit["samples"].append(sample)
        if numeric_thu is None:
            audit["failure_reason"] = "missing_land_sample"
            return audit
        heights_gu.append(numeric_thu * THU_TO_GU)

    observed = max(heights_gu) - min(heights_gu)
    margin = observed - required
    audit["observed_relief_gu"] = _round(observed)
    audit["relief_margin_gu"] = _round(margin)
    audit["passed"] = observed >= required
    if not audit["passed"]:
        audit["failure_reason"] = "insufficient_relief"
    return audit


def _aabb_xy_bounds(world_bbox: Mapping[str, Sequence[float]]) -> tuple[float, float, float, float]:
    """Return finite positive-area ``(min_x, max_x, min_y, max_y)`` bounds."""

    minimum = world_bbox.get("min")
    maximum = world_bbox.get("max")
    if (
        not isinstance(minimum, Sequence)
        or not isinstance(maximum, Sequence)
        or len(minimum) < 2
        or len(maximum) < 2
    ):
        raise ValueError("world bbox must contain min/max x/y arrays")
    min_x, max_x = float(minimum[0]), float(maximum[0])
    min_y, max_y = float(minimum[1]), float(maximum[1])
    if not all(math.isfinite(value) for value in (min_x, max_x, min_y, max_y)):
        raise ValueError("world bbox x/y bounds must be finite")
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("world bbox must have positive x/y area")
    return min_x, max_x, min_y, max_y


def large_rock_tree_clearance_violation(
    point_xy: Sequence[float],
    accepted_rock_cliffs: Sequence[Mapping[str, Any]],
    *,
    minimum_horizontal_span_gu: float = TREE_CLEARANCE_MIN_HORIZONTAL_SPAN_GU,
    margin_gu: float = TREE_CLEARANCE_MARGIN_GU,
) -> Mapping[str, Any] | None:
    """Return the first accepted large-rock AABB containing a tree center.

    Only already accepted ``rocks``/``cliff`` placements are considered.  A
    transformed horizontal span is measured independently on X and Y; the
    larger span must meet the 1024-GU threshold.  The comparison then expands
    that transformed world AABB by the configured margin on every side.  The
    Accepted placements are already emitted in deterministic placement order;
    the first matching row is therefore stable even when AABBs overlap.
    """

    if len(point_xy) < 2:
        raise ValueError("tree clearance checks need an x/y point")
    px, py = float(point_xy[0]), float(point_xy[1])
    threshold = float(minimum_horizontal_span_gu)
    margin = float(margin_gu)
    if not math.isfinite(px) or not math.isfinite(py) or threshold <= 0.0 or margin < 0.0:
        raise ValueError("tree clearance inputs are invalid")

    for row in accepted_rock_cliffs:
        if str(row.get("category")) not in {"rocks", "cliff"}:
            continue
        bbox = row.get("bbox")
        world_aabb = bbox.get("world_aabb_gu") if isinstance(bbox, Mapping) else None
        if not isinstance(world_aabb, Mapping):
            continue
        try:
            min_x, max_x, min_y, max_y = _aabb_xy_bounds(world_aabb)
        except (TypeError, ValueError, OverflowError):
            continue
        if max(max_x - min_x, max_y - min_y) < threshold:
            continue
        if min_x - margin <= px <= max_x + margin and min_y - margin <= py <= max_y + margin:
            return row
    return None


def _large_rock_tree_clearance_aabbs(
    accepted_rock_cliffs: Sequence[Mapping[str, Any]],
    *,
    minimum_horizontal_span_gu: float = TREE_CLEARANCE_MIN_HORIZONTAL_SPAN_GU,
    margin_gu: float = TREE_CLEARANCE_MARGIN_GU,
) -> list[tuple[float, float, float, float]]:
    """Return expanded XY AABBs that participate in tree clearance.

    Same filter order and inclusive expansion as
    ``large_rock_tree_clearance_violation``; only large-span boxes are kept.
    """

    threshold = float(minimum_horizontal_span_gu)
    margin = float(margin_gu)
    if threshold <= 0.0 or margin < 0.0:
        raise ValueError("tree clearance inputs are invalid")
    boxes: list[tuple[float, float, float, float]] = []
    for row in accepted_rock_cliffs:
        if str(row.get("category")) not in {"rocks", "cliff"}:
            continue
        bbox = row.get("bbox")
        world_aabb = bbox.get("world_aabb_gu") if isinstance(bbox, Mapping) else None
        if not isinstance(world_aabb, Mapping):
            continue
        try:
            min_x, max_x, min_y, max_y = _aabb_xy_bounds(world_aabb)
        except (TypeError, ValueError, OverflowError):
            continue
        if max(max_x - min_x, max_y - min_y) < threshold:
            continue
        boxes.append((min_x - margin, max_x + margin, min_y - margin, max_y + margin))
    return boxes


def _points_inside_any_aabb(
    points_xy: np.ndarray,
    expanded_aabbs: Sequence[tuple[float, float, float, float]],
) -> np.ndarray:
    """Return a boolean mask of points inside any inclusive expanded AABB."""

    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must be an (N, 2) array")
    blocked = np.zeros(points_xy.shape[0], dtype=bool)
    if points_xy.shape[0] == 0 or not expanded_aabbs:
        return blocked
    xs = points_xy[:, 0]
    ys = points_xy[:, 1]
    for min_x, max_x, min_y, max_y in expanded_aabbs:
        blocked |= (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y)
    return blocked


def tree_clearance_blocked_candidate_ids(
    candidates: Sequence[_Candidate],
    accepted_rock_cliffs: Sequence[Mapping[str, Any]],
    *,
    minimum_horizontal_span_gu: float = TREE_CLEARANCE_MIN_HORIZONTAL_SPAN_GU,
    margin_gu: float = TREE_CLEARANCE_MARGIN_GU,
) -> tuple[set[str], list[tuple[float, float, float, float]]]:
    """Build the tree-clearance blocked ID set via batch AABB tests.

    Equivalent to calling ``large_rock_tree_clearance_violation`` on every
    candidate and collecting any-hit results; only large AABBs are tested.
    """

    boxes = _large_rock_tree_clearance_aabbs(
        accepted_rock_cliffs,
        minimum_horizontal_span_gu=minimum_horizontal_span_gu,
        margin_gu=margin_gu,
    )
    if not candidates:
        return set(), boxes
    points = np.asarray([(c.x_gu, c.y_gu) for c in candidates], dtype=np.float64)
    mask = _points_inside_any_aabb(points, boxes)
    blocked = {candidates[index].candidate_id for index in np.flatnonzero(mask)}
    return blocked, boxes


def enumerate_land_texture_tiles(
    world_bbox: Mapping[str, Sequence[float]],
) -> list[tuple[int, int]]:
    """Enumerate globally aligned 512-GU LAND tiles intersecting a world AABB.

    The AABB is treated as half-open in x/y: ``[min, max)``.  Consequently a
    maximum edge exactly on a tile boundary does not include the tile on the
    far side of that boundary.  Tile indices are global, rather than reset at
    a CELL border, so floor division remains correct for negative Tamriel
    coordinates.  Results are deterministic in row-major ``y, x`` order.
    """

    min_x, max_x, min_y, max_y = _aabb_xy_bounds(world_bbox)
    min_tile_x = math.floor(min_x / LAND_TEXTURE_TILE_SIZE_GU)
    max_tile_x_exclusive = math.ceil(max_x / LAND_TEXTURE_TILE_SIZE_GU)
    min_tile_y = math.floor(min_y / LAND_TEXTURE_TILE_SIZE_GU)
    max_tile_y_exclusive = math.ceil(max_y / LAND_TEXTURE_TILE_SIZE_GU)
    return [
        (tile_x, tile_y)
        for tile_y in range(min_tile_y, max_tile_y_exclusive)
        for tile_x in range(min_tile_x, max_tile_x_exclusive)
    ]


def _land_texture_tile_cell_local(
    global_tile_x: int,
    global_tile_y: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert global texture-tile indices to a CELL and local 0..15 tile.

    ``math.floor`` is intentional.  Python's integer ``//`` would also be
    floor-safe here, but spelling the rule out keeps the negative-coordinate
    contract visible at this world-to-CELL boundary (for example, global tile
    ``-1`` belongs to CELL ``-1``'s local tile ``15``).
    """

    cell_x = math.floor(int(global_tile_x) / LAND_TEXTURE_SIDE)
    cell_y = math.floor(int(global_tile_y) / LAND_TEXTURE_SIDE)
    local_tile_x = int(global_tile_x) - cell_x * LAND_TEXTURE_SIDE
    local_tile_y = int(global_tile_y) - cell_y * LAND_TEXTURE_SIDE
    return (cell_x, cell_y), (local_tile_x, local_tile_y)


# Sentinels for optional road-footprint VTEX tile cache values.
_ROAD_VTEX_MISSING_LAND = object()
_ROAD_VTEX_MISSING_VTEX = object()


def road_footprint_audit(
    land_records: Mapping[tuple[int, int], LandRecord],
    world_bbox: Mapping[str, Sequence[float]],
    road_raw_vtex_values: Sequence[int],
    *,
    detail: bool = True,
    vtex_cache: dict[tuple[int, int], Any] | None = None,
) -> dict[str, Any]:
    """Audit every LAND VTEX tile under one transformed world-space AABB.

    This helper is deliberately independent of candidate centers.  It maps a
    globally aligned 512-GU tile to its owning CELL and local 16x16 VTEX tile,
    reads the normalized raw VTEX value, and records road hits.  Missing LAND
    or VTEX is a hard failure: a rock/cliff footprint is never allowed to pass
    merely because the authoritative texture data is unavailable.

    ``detail=False`` keeps the same pass/fail and tile counts without allocating
    per-tile dictionaries.  Placement uses that path; unit tests keep detail.
    ``vtex_cache`` optionally memoizes resolved global-tile lookups across many
    footprint audits in one generation run.
    """

    road_set = {int(value) for value in road_raw_vtex_values}
    if any(value < 0 or value > 65535 for value in road_set):
        raise ValueError("road_raw_vtex_values must contain unsigned 16-bit values")
    road_values = tuple(sorted(road_set))

    minimum = world_bbox.get("min")
    maximum = world_bbox.get("max")
    world_aabb = {
        "min": [
            _round(float(value)) for value in minimum
        ] if isinstance(minimum, Sequence) else None,
        "max": [
            _round(float(value)) for value in maximum
        ] if isinstance(maximum, Sequence) else None,
    }
    audit: dict[str, Any] = {
        "terrain_source": "tamriel.esm LAND via procgen.espland",
        "world_aabb_gu": world_aabb,
        "tile_size_gu": LAND_TEXTURE_TILE_SIZE_GU,
        "tile_alignment": "global world-origin 512-GU grid",
        "aabb_intersection_rule": "half-open [min,max) in x/y; max boundary tile is excluded",
        "road_raw_vtex_values": list(road_values),
        "tile_index_bounds": None,
        "intersected_tile_count": 0,
        "checked_tile_count": 0,
        "resolved_tile_count": 0,
        "checked_tiles": [],
        "road_hits": [],
        "road_hit_count": 0,
        "missing_land_cells": [],
        "missing_vtex_cells": [],
        "missing_land_tile_count": 0,
        "missing_vtex_tile_count": 0,
        "passed": False,
        "failure_reason": None,
    }
    try:
        min_x, max_x, min_y, max_y = _aabb_xy_bounds(world_bbox)
    except (TypeError, ValueError, OverflowError) as exc:
        audit["failure_reason"] = "invalid_or_degenerate_aabb"
        audit["failure_detail"] = str(exc)
        return audit

    min_tile_x = math.floor(min_x / LAND_TEXTURE_TILE_SIZE_GU)
    max_tile_x_exclusive = math.ceil(max_x / LAND_TEXTURE_TILE_SIZE_GU)
    min_tile_y = math.floor(min_y / LAND_TEXTURE_TILE_SIZE_GU)
    max_tile_y_exclusive = math.ceil(max_y / LAND_TEXTURE_TILE_SIZE_GU)
    audit["tile_index_bounds"] = [
        min_tile_x,
        max_tile_x_exclusive,
        min_tile_y,
        max_tile_y_exclusive,
    ]
    if max_tile_x_exclusive <= min_tile_x or max_tile_y_exclusive <= min_tile_y:
        audit["failure_reason"] = "invalid_or_degenerate_aabb"
        return audit
    audit["intersected_tile_count"] = (max_tile_x_exclusive - min_tile_x) * (
        max_tile_y_exclusive - min_tile_y
    )

    missing_land_cells: set[tuple[int, int]] = set()
    missing_vtex_cells: set[tuple[int, int]] = set()
    checked_tiles: list[dict[str, Any]] = []
    road_hits: list[dict[str, Any]] = []
    road_hit_count = 0
    resolved_tile_count = 0
    checked_tile_count = 0
    missing_land_tile_count = 0
    missing_vtex_tile_count = 0
    for global_tile_y in range(min_tile_y, max_tile_y_exclusive):
        for global_tile_x in range(min_tile_x, max_tile_x_exclusive):
            cell, local_tile = _land_texture_tile_cell_local(global_tile_x, global_tile_y)
            checked_tile_count += 1
            cache_key = (global_tile_x, global_tile_y)
            cached = vtex_cache.get(cache_key) if vtex_cache is not None else None
            if cached is _ROAD_VTEX_MISSING_LAND:
                missing_land_cells.add(cell)
                missing_land_tile_count += 1
                if detail:
                    checked_tiles.append(
                        {
                            "global_tile": [global_tile_x, global_tile_y],
                            "cell": [cell[0], cell[1]],
                            "local_tile": [local_tile[0], local_tile[1]],
                            "raw_vtex": None,
                            "status": "missing_land",
                        }
                    )
                continue
            if cached is _ROAD_VTEX_MISSING_VTEX:
                missing_vtex_cells.add(cell)
                missing_vtex_tile_count += 1
                if detail:
                    checked_tiles.append(
                        {
                            "global_tile": [global_tile_x, global_tile_y],
                            "cell": [cell[0], cell[1]],
                            "local_tile": [local_tile[0], local_tile[1]],
                            "raw_vtex": None,
                            "status": "missing_vtex",
                        }
                    )
                continue
            if isinstance(cached, int):
                raw_vtex = cached
            else:
                record = land_records.get(cell)
                if record is None:
                    if vtex_cache is not None:
                        vtex_cache[cache_key] = _ROAD_VTEX_MISSING_LAND
                    missing_land_cells.add(cell)
                    missing_land_tile_count += 1
                    if detail:
                        checked_tiles.append(
                            {
                                "global_tile": [global_tile_x, global_tile_y],
                                "cell": [cell[0], cell[1]],
                                "local_tile": [local_tile[0], local_tile[1]],
                                "raw_vtex": None,
                                "status": "missing_land",
                            }
                        )
                    continue
                if not record.has_textures:
                    if vtex_cache is not None:
                        vtex_cache[cache_key] = _ROAD_VTEX_MISSING_VTEX
                    missing_vtex_cells.add(cell)
                    missing_vtex_tile_count += 1
                    if detail:
                        checked_tiles.append(
                            {
                                "global_tile": [global_tile_x, global_tile_y],
                                "cell": [cell[0], cell[1]],
                                "local_tile": [local_tile[0], local_tile[1]],
                                "raw_vtex": None,
                                "status": "missing_vtex",
                            }
                        )
                    continue
                try:
                    raw_vtex = int(record.texture_index(local_tile[0], local_tile[1]))
                except (IndexError, TypeError, ValueError, AttributeError):
                    if vtex_cache is not None:
                        vtex_cache[cache_key] = _ROAD_VTEX_MISSING_VTEX
                    missing_vtex_cells.add(cell)
                    missing_vtex_tile_count += 1
                    if detail:
                        checked_tiles.append(
                            {
                                "global_tile": [global_tile_x, global_tile_y],
                                "cell": [cell[0], cell[1]],
                                "local_tile": [local_tile[0], local_tile[1]],
                                "raw_vtex": None,
                                "status": "missing_vtex",
                            }
                        )
                    continue
                if vtex_cache is not None:
                    vtex_cache[cache_key] = raw_vtex
            resolved_tile_count += 1
            is_road = raw_vtex in road_set
            if is_road:
                road_hit_count += 1
            if detail:
                tile = {
                    "global_tile": [global_tile_x, global_tile_y],
                    "cell": [cell[0], cell[1]],
                    "local_tile": [local_tile[0], local_tile[1]],
                    "raw_vtex": raw_vtex,
                    "status": "road" if is_road else "ok",
                }
                checked_tiles.append(tile)
                if is_road:
                    road_hits.append(tile)

    audit["checked_tiles"] = checked_tiles
    audit["checked_tile_count"] = checked_tile_count
    audit["resolved_tile_count"] = resolved_tile_count
    audit["road_hits"] = road_hits
    audit["road_hit_count"] = road_hit_count
    audit["missing_land_tile_count"] = missing_land_tile_count
    audit["missing_vtex_tile_count"] = missing_vtex_tile_count
    audit["missing_land_cells"] = [
        [cell[0], cell[1]] for cell in sorted(missing_land_cells, key=lambda value: (value[1], value[0]))
    ]
    audit["missing_vtex_cells"] = [
        [cell[0], cell[1]] for cell in sorted(missing_vtex_cells, key=lambda value: (value[1], value[0]))
    ]
    if missing_land_cells:
        audit["failure_reason"] = "missing_land"
    elif missing_vtex_cells:
        audit["failure_reason"] = "missing_vtex"
    elif road_hit_count:
        audit["failure_reason"] = "road_overlap"
    else:
        audit["passed"] = True
    return audit


def _compact_road_footprint_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the passing footprint facts needed on each accepted ref."""

    road_hits = audit.get("road_hits", [])
    road_hit_count = int(audit.get("road_hit_count", len(road_hits) if isinstance(road_hits, list) else 0))
    return {
        "tile_size_gu": audit.get("tile_size_gu"),
        "tile_alignment": audit.get("tile_alignment"),
        "aabb_intersection_rule": audit.get("aabb_intersection_rule"),
        "tile_index_bounds": list(audit.get("tile_index_bounds") or []),
        "intersected_tile_count": int(audit.get("intersected_tile_count", 0)),
        "checked_tile_count": int(audit.get("checked_tile_count", 0)),
        "resolved_tile_count": int(audit.get("resolved_tile_count", 0)),
        "road_raw_vtex_values": list(audit.get("road_raw_vtex_values", [])),
        "road_hit_count": road_hit_count,
        "road_hit_raw_vtex_values": sorted(
            {
                int(row["raw_vtex"])
                for row in road_hits
                if isinstance(row, Mapping) and row.get("raw_vtex") is not None
            }
        ),
        "missing_land_cell_count": len(audit.get("missing_land_cells", [])),
        "missing_vtex_cell_count": len(audit.get("missing_vtex_cells", [])),
        "passed": bool(audit.get("passed")),
        "failure_reason": audit.get("failure_reason"),
    }


class RockNeighborIndex:
    """Uniform grid over accepted rock/cliff centers for open-face queries."""

    def __init__(self, bin_size_gu: float = 512.0) -> None:
        if bin_size_gu <= 0:
            raise ValueError("rock neighbor bin size must be positive")
        self.bin_size_gu = float(bin_size_gu)
        self._bins: dict[tuple[int, int], list[tuple[str, float, float, float]]] = defaultdict(list)

    def _bin(self, value: float) -> int:
        return math.floor(float(value) / self.bin_size_gu)

    def add(self, ref_id: str, x_gu: float, y_gu: float, radius_gu: float) -> None:
        self._bins[(self._bin(x_gu), self._bin(y_gu))].append(
            (str(ref_id), float(x_gu), float(y_gu), float(radius_gu))
        )

    def query(
        self,
        x_gu: float,
        y_gu: float,
        radius_gu: float,
        profile_radius_gu: float,
    ) -> list[tuple[str, tuple[float, float], float]]:
        bx, by = self._bin(x_gu), self._bin(y_gu)
        reach = int(math.ceil(float(radius_gu) / self.bin_size_gu)) + 1
        targets: list[tuple[str, tuple[float, float], float]] = []
        for ix in range(bx - reach, bx + reach + 1):
            for iy in range(by - reach, by + reach + 1):
                for ref_id, other_x, other_y, target_radius in self._bins.get((ix, iy), ()):
                    dx = other_x - x_gu
                    dy = other_y - y_gu
                    distance = math.hypot(dx, dy)
                    if distance <= 1e-9 or distance > radius_gu:
                        continue
                    occlusion_limit = max(512.0, 1.75 * (float(profile_radius_gu) + target_radius))
                    if distance > min(float(radius_gu), occlusion_limit):
                        continue
                    targets.append((ref_id, (dx / distance, dy / distance), distance))
        targets.sort(key=lambda row: (row[0]))
        return targets


def _rotate_xy_direction(direction: Sequence[float], rotation: Sequence[float]) -> tuple[float, float]:
    """Project a local horizontal profile through the raw TES3 ref matrix."""

    x, y = float(direction[0]), float(direction[1])
    # Use the same authoritative raw-reference helper as transformed_bbox, then
    # project the rotated horizontal profile back onto world XY for alignment.
    x_z, y_z, _z_z = rotate_tes3_reference_point((x, y, 0.0), rotation)
    length = math.hypot(x_z, y_z)
    if length <= 1e-9:
        return (0.0, 0.0)
    return (x_z / length, y_z / length)


def _pose_first_open_face_yaw(
    local_direction: Sequence[float],
    target_direction: Sequence[float],
    *,
    rotation_x: float = 0.0,
    rotation_y: float = 0.0,
) -> float:
    """Return TES3 rotz that aims ``local_direction`` at ``target_direction``.

    For upright yaw-only poses the engine uses ``Rz(-rotz)``, so
    ``rotz = atan2(local_y, local_x) - atan2(target_y, target_x)``.  Non-zero
    measured X/Y tilt falls back to a dense 1-D search that maximizes the same
    world-XY dot product ``_open_face_orientation_decision`` uses.
    """

    lx, ly = float(local_direction[0]), float(local_direction[1])
    tx, ty = float(target_direction[0]), float(target_direction[1])
    local_len = math.hypot(lx, ly)
    target_len = math.hypot(tx, ty)
    if local_len <= 1e-9 or target_len <= 1e-9:
        return 0.0
    lx, ly = lx / local_len, ly / local_len
    tx, ty = tx / target_len, ty / target_len
    rx = float(rotation_x)
    ry = float(rotation_y)
    if abs(rx) <= 1e-12 and abs(ry) <= 1e-12:
        return math.atan2(ly, lx) - math.atan2(ty, tx)
    best_rotz = 0.0
    best_dot = -math.inf
    steps = 360
    for index in range(steps):
        rotz = (2.0 * math.pi * index) / steps
        world = _rotate_xy_direction((lx, ly), (rx, ry, rotz))
        dot = world[0] * tx + world[1] * ty
        if dot > best_dot:
            best_dot = dot
            best_rotz = rotz
    return best_rotz


def _open_face_pose_targets(
    profile: _Species,
    candidate: _Candidate,
    placements: Sequence[Mapping[str, Any]],
    config: GenerationConfig,
    profile_radius_gu: float,
    neighbor_index: RockNeighborIndex | None = None,
) -> list[tuple[str, str, tuple[float, float], float | None]]:
    """Return downhill-then-neighbor targets for pose-first open-face yaw."""

    targets: list[tuple[str, str, tuple[float, float], float | None]] = []
    downhill = candidate.downhill_direction_xy
    if math.hypot(*downhill) > 1e-9:
        targets.append(("terrain", "terrain", downhill, None))
    targets.extend(
        ("adjacent_accepted_rock", ref_id, direction, distance)
        for ref_id, direction, distance in _rock_neighbor_targets(
            candidate,
            placements,
            config.open_face_neighbor_radius_gu,
            profile_radius_gu,
            neighbor_index=neighbor_index,
        )
    )
    return targets


def _sample_pose_first_tilt(
    profile: _Species, rng: random.Random, slope_deg: float
) -> tuple[float, float, str, str]:
    """Sample measured X/Y once for pose-first; yaw is solved separately."""

    conditioned = _rotation_bin(profile, float(slope_deg))
    distribution = conditioned if conditioned and int(conditioned.get("count", 0)) > 0 else profile.rotation_distribution
    mode_counts = distribution.get("mode_counts", {}) if isinstance(distribution, Mapping) else {}
    if not isinstance(mode_counts, Mapping):
        mode_counts = {}
    z_count = int(mode_counts.get("z_only", 0))
    full_count = int(mode_counts.get("full", 0))
    if full_count <= 0:
        mode = "z_only"
    elif z_count <= 0:
        mode = "full"
    else:
        mode = "full" if rng.random() * (z_count + full_count) >= z_count else "z_only"
    source = "measured_slope_bin" if conditioned and distribution is conditioned else "measured_mesh_distribution"
    # Consume the measured Z draw so the profile RNG stream stays aligned with
    # measured sampling even though pose-first replaces Z with a solved yaw.
    _sample_quantile(
        distribution.get("z_radians", {}) if isinstance(distribution, Mapping) else {},
        rng,
        fallback=0.0,
    )
    if mode == "z_only":
        return 0.0, 0.0, mode, f"{source}+pose_first"
    x = _sample_quantile(distribution.get("x_radians", {}), rng, fallback=0.0)
    y = _sample_quantile(distribution.get("y_radians", {}), rng, fallback=0.0)
    return float(x), float(y), mode, f"{source}+pose_first"


def _rock_neighbor_targets(
    candidate: _Candidate,
    placements: Sequence[Mapping[str, Any]],
    radius_gu: float,
    profile_radius_gu: float,
    neighbor_index: RockNeighborIndex | None = None,
) -> list[tuple[str, tuple[float, float], float]]:
    if neighbor_index is not None:
        return neighbor_index.query(
            candidate.x_gu, candidate.y_gu, radius_gu, profile_radius_gu
        )
    targets: list[tuple[str, tuple[float, float], float]] = []
    for row in placements:
        if str(row.get("category")) not in {"rocks", "cliff"}:
            continue
        position = row.get("position_gu")
        if not isinstance(position, Sequence) or len(position) < 2:
            continue
        dx = float(position[0]) - candidate.x_gu
        dy = float(position[1]) - candidate.y_gu
        distance = math.hypot(dx, dy)
        if distance <= 1e-9 or distance > radius_gu:
            continue
        target_bbox = row.get("bbox", {})
        target_aabb = target_bbox.get("world_aabb_gu", {}) if isinstance(target_bbox, Mapping) else {}
        target_min = target_aabb.get("min") if isinstance(target_aabb, Mapping) else None
        target_max = target_aabb.get("max") if isinstance(target_aabb, Mapping) else None
        if isinstance(target_min, Sequence) and isinstance(target_max, Sequence) and len(target_min) >= 2 and len(target_max) >= 2:
            target_radius = 0.5 * math.hypot(
                float(target_max[0]) - float(target_min[0]),
                float(target_max[1]) - float(target_min[1]),
            )
        else:
            target_radius = 0.0
        # A target is an occluding rock only when its measured world-space
        # footprint is close enough to cover the open side. The previous
        # radius-only rule could point an opening at a distant rock.
        occlusion_limit = max(512.0, 1.75 * (float(profile_radius_gu) + target_radius))
        if distance > min(float(radius_gu), occlusion_limit):
            continue
        targets.append((str(row.get("ref_id", "adjacent_rock")), (dx / distance, dy / distance), distance))
    targets.sort(key=lambda row: (row[0]))
    return targets


def _profile_horizontal_radius(bbox: Mapping[str, Any]) -> float:
    dimensions = bbox.get("dimensions_gu")
    if isinstance(dimensions, Sequence) and len(dimensions) >= 2:
        return 0.5 * math.hypot(float(dimensions[0]), float(dimensions[1]))
    return 0.0


def _open_face_orientation_decision(
    profile: _Species,
    candidate: _Candidate,
    rotation: Sequence[float],
    placements: Sequence[Mapping[str, Any]],
    config: GenerationConfig,
    profile_radius_gu: float,
    neighbor_index: RockNeighborIndex | None = None,
) -> dict[str, Any]:
    open_directions = [
        str(value)
        for value in profile.open_face_profile.get("open_directions", [])
        if str(value) in _DIRECTION_VECTORS_XY
    ]
    bottom_open = bool(profile.open_face_profile.get("bottom_open", False))
    top_open = bool(profile.open_face_profile.get("top_open", False))
    open_axes = [str(value) for value in profile.open_face_profile.get("open_axes", [])]
    base: dict[str, Any] = {
        "profile_status": str(profile.open_face_profile.get("status", "not_supplied")),
        "open_directions_local": open_directions,
        "orientation_attempts": 1,
        "action": "closed_profile",
        "target_ref_id": None,
        "alignment_dot": None,
        "target_direction_xy": None,
        "target_distance_gu": None,
        "bottom_open": bottom_open,
        "top_open": top_open,
        "open_axes": open_axes,
        "bottom_seating": None,
    }
    if bottom_open:
        # A bottom cavity may be tilted, but it must remain a downward-facing
        # resting surface.  The later terrain gate supplies the actual burial.
        up_world_z = transformed_local_up_world_z(rotation)
        seated = up_world_z >= math.cos(math.radians(45.0))
        base["bottom_seating"] = {
            "local_up_world_z": _round(up_world_z, 6),
            "minimum_local_up_world_z": _round(math.cos(math.radians(45.0)), 6),
            "passed": seated,
        }
        if not seated:
            base["action"] = "unsafe_orientation"
            return base
    if not open_directions:
        return base
    targets: list[tuple[str, str, tuple[float, float], float | None]] = []
    downhill = candidate.downhill_direction_xy
    if math.hypot(*downhill) > 1e-9:
        targets.append(("terrain", "terrain", downhill, None))
    targets.extend(
        ("adjacent_accepted_rock", ref_id, direction, distance)
        for ref_id, direction, distance in _rock_neighbor_targets(
            candidate,
            placements,
            config.open_face_neighbor_radius_gu,
            profile_radius_gu,
            neighbor_index=neighbor_index,
        )
    )
    if not targets:
        base["action"] = "no_target_skip"
        return base
    best: tuple[float, str, str, tuple[float, float], float | None] | None = None
    for direction_name in open_directions:
        local = _DIRECTION_VECTORS_XY[direction_name]
        world = _rotate_xy_direction(local, rotation)
        for target_kind, target_ref, target_direction, target_distance in targets:
            alignment = world[0] * target_direction[0] + world[1] * target_direction[1]
            candidate_best = (alignment, target_kind, target_ref, target_direction, target_distance)
            if best is None or candidate_best[0] > best[0]:
                best = candidate_best
    if best is None:
        base["action"] = "no_target_skip"
        return base
    alignment, target_kind, target_ref, target_direction, target_distance = best
    base.update(
        {
            "target_ref_id": None if target_kind == "terrain" else target_ref,
            "alignment_dot": _round(alignment, 6),
            "target_direction_xy": [_round(target_direction[0], 6), _round(target_direction[1], 6)],
            "target_distance_gu": _round(target_distance),
        }
    )
    if alignment >= config.open_face_min_alignment:
        base["action"] = "terrain_downslope" if target_kind == "terrain" else "adjacent_rock"
        return base
    base["action"] = "unsafe_orientation"
    return base


def _new_placement(
    *,
    ref_id: str,
    candidate: _Candidate,
    profile: _Species,
    pass_name: str,
    bbox: Mapping[str, Any],
    rotation: Sequence[float],
    rotation_mode_value: str,
    rotation_source: str,
    scale: float,
    z_offset_gu: float,
    world_bbox: Mapping[str, Any],
    suitability_weight: float,
    clearing_value: float,
    config: GenerationConfig,
    open_face_decision: Mapping[str, Any] | None = None,
    face_link: Mapping[str, Any] | None = None,
    cliff_footprint_audit: Mapping[str, Any] | None = None,
    road_footprint_audit: Mapping[str, Any] | None = None,
    cliff_seating_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    placement_z = candidate.terrain_z_gu + z_offset_gu
    placement: dict[str, Any] = {
        "ref_id": ref_id,
        "cell": [candidate.cell[0], candidate.cell[1]],
        "mesh": profile.mesh,
        "category": "cliff" if profile.category == "cliff" else profile.category,
        "pass": pass_name,
        "position_gu": [_round(candidate.x_gu), _round(candidate.y_gu), _round(placement_z)],
        "rotation_radians": [_round(float(value), 6) for value in rotation],
        "rotation_mode": rotation_mode_value,
        "rotation_source": rotation_source,
        "scale": _round(scale),
        "flora_role": profile.flora_role,
        "clearing": {
            "mask_value": _round(clearing_value, 6),
            "is_clearing": clearing_value >= config.clearing_threshold,
            "tree_weight_factor": config.clearing_tree_factor if profile.flora_role == "tree" else 1.0,
            "rule": "low-frequency smooth mask; tree candidates downweighted, undergrowth unchanged",
        },
        "terrain": {
            "terrain_source": "tamriel.esm LAND via procgen.espland",
            "terrain_z_thu": _round(candidate.terrain_z_thu),
            "terrain_z_gu": _round(candidate.terrain_z_gu),
            "raw_vtex": candidate.raw_vtex,
            "road_raw_vtex": _candidate_has_road_raw_vtex(candidate, config),
            "slope_deg": _round(candidate.slope_deg),
            "elevation_gu": _round(candidate.terrain_z_gu),
            "distance_to_water_gu": _round(candidate.water_distance_gu),
            "downhill_direction_xy": [
                _round(candidate.downhill_direction_xy[0], 6),
                _round(candidate.downhill_direction_xy[1], 6),
            ],
            "water_threshold_thu": 0.0,
            "water_state": "shallow_water_candidate" if candidate.terrain_z_thu <= 0 else "above_water",
            "z_offset_gu": _round(z_offset_gu),
            "embedding_depth_gu": _round(float(candidate.terrain_z_gu) - float(world_bbox["min"][2])),
            "surface_intersection": True,
        },
        "water_rule": {
            "allowed_below_or_at_zero": bool(candidate.terrain_z_thu <= 0.0 and profile.shallow_water),
            "shallow_water_species": profile.shallow_water,
        },
        "suitability_weight": _round(suitability_weight, 9),
        "rock_patch": {
            "mask_value": _round(candidate.rock_patch_value, 6),
            "threshold": _round(config.rock_patch_threshold, 6),
            "accepted": candidate.rock_patch_value >= config.rock_patch_threshold,
            "rule": "normal rocks are rejected below the smooth patch threshold; cliffs are exempt",
        },
        "measured_species_frequency": profile.frequency,
        "bbox": {**dict(bbox), "world_aabb_gu": dict(world_bbox)},
        "stacking": {
            "enabled": False,
            "on_cliff": False,
            "parent_ref_id": None,
            "support_top_z_gu": None,
        },
    }
    if profile.category in {"rocks", "cliff"}:
        local_up_world_z = transformed_local_up_world_z(rotation)
        local_up_tilt = transformed_local_up_tilt_degrees(rotation)
        max_local_up_tilt = config.max_local_up_tilt_degrees_by_mesh.get(profile.key)
        placement["orientation_audit"] = {
            "euler_convention": "pipeline XYZ: X then Y then Z",
            "local_up_world_z": _round(local_up_world_z, 9),
            "local_up_tilt_degrees": _round(local_up_tilt, 9),
            "upright_gate": "local_up_world_z > 0",
            "max_local_up_tilt_degrees": _round(max_local_up_tilt, 9),
            "max_tilt_gate": "tilt <= configured mesh limit" if max_local_up_tilt is not None else None,
            "passed": local_up_world_z > 0.0
            and (max_local_up_tilt is None or local_up_tilt <= max_local_up_tilt),
        }
    if profile.category == "cliff" and cliff_seating_evidence is None:
        visible_height_gu = float(world_bbox["max"][2]) - float(candidate.terrain_z_gu)
        placement["cliff_surface_gate"] = {
            "sampled_z_offset_gu": _round(z_offset_gu),
            "embedding_depth_gu": _round(float(candidate.terrain_z_gu) - float(world_bbox["min"][2])),
            "minimum_embed_gu": _round(config.cliff_min_embed_gu),
            "visible_height_gu": _round(visible_height_gu),
            "minimum_visible_gu": _round(config.cliff_min_visible_gu),
            "passed": (
                float(candidate.terrain_z_gu) - float(world_bbox["min"][2]) >= config.cliff_min_embed_gu
                and visible_height_gu >= config.cliff_min_visible_gu
            ),
        }
    if open_face_decision is not None:
        placement["open_face"] = dict(open_face_decision)
    if face_link is not None:
        placement["cliff_face_link"] = dict(face_link)
    if cliff_footprint_audit is not None:
        placement["cliff_footprint_relief"] = dict(cliff_footprint_audit)
    if road_footprint_audit is not None:
        placement["road_footprint_audit"] = dict(road_footprint_audit)
    if cliff_seating_evidence is not None:
        evidence = dict(cliff_seating_evidence)
        evidence.pop("world_bbox", None)
        placement["cliff_seating"] = evidence
    return placement


def _profile_candidate_rows(
    all_candidates: Sequence[_Candidate],
    profile: _Species,
    config: GenerationConfig,
    arrays: _CandidateArrays | None = None,
) -> list[tuple[_Candidate, float, float]]:
    if not all_candidates:
        return []
    view = arrays if arrays is not None else _build_candidate_arrays(all_candidates, config)
    if len(view.slope_deg) != len(all_candidates):
        raise ValueError("candidate arrays length must match candidate list")

    eligible = ~view.is_road
    if profile.category == "rocks":
        eligible &= view.rock_patch_value >= config.rock_patch_threshold
    if not profile.shallow_water:
        eligible &= view.terrain_z_thu > 0.0
    if not np.any(eligible):
        return []

    weights = np.zeros(len(all_candidates), dtype=np.float64)
    # Evaluate envelopes only on candidates that survive road/patch/water gates.
    active = np.flatnonzero(eligible)
    slope_w = envelope_weight_array(view.slope_deg[active], profile.conditions["slope_deg"])
    elev_w = envelope_weight_array(view.elevation_gu[active], profile.conditions["elevation_gu"])
    water_w = envelope_weight_array(
        view.water_distance_gu[active], profile.conditions["water_distance_gu"]
    )
    active_weights = float(profile.frequency) * slope_w * elev_w * water_w
    if profile.flora_role == "tree":
        clearing = view.clearing_value[active]
        active_weights = np.where(
            clearing >= config.clearing_threshold,
            active_weights * config.clearing_tree_factor,
            active_weights,
        )
    positive = active_weights > 0.0
    if not np.any(positive):
        return []
    selected = active[positive]
    weights[selected] = active_weights[positive]
    return [
        (all_candidates[index], float(weights[index]), float(view.clearing_value[index]))
        for index in selected
    ]


def _nearest_face_link(candidate: _Candidate, cliff_points: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not cliff_points:
        return None
    closest = min(
        cliff_points,
        key=lambda row: (
            (candidate.x_gu - float(row["position_gu"][0])) ** 2
            + (candidate.y_gu - float(row["position_gu"][1])) ** 2,
            str(row["ref_id"]),
        ),
    )
    distance = math.hypot(
        candidate.x_gu - float(closest["position_gu"][0]),
        candidate.y_gu - float(closest["position_gu"][1]),
    )
    return {"ref_id": closest["ref_id"], "distance_gu": _round(distance)}


def _place_profile(
    profile: _Species,
    desired: int,
    *,
    all_candidates: Sequence[_Candidate],
    land_records: Mapping[tuple[int, int], LandRecord],
    bbox: Mapping[str, Any],
    config: GenerationConfig,
    occupancy: OccupancyIndex,
    used_ids: set[str],
    placements: list[dict[str, Any]],
    pass_name: str,
    minimum_distance: float,
    cliff: bool,
    cliff_points: list[dict[str, Any]],
    ordinal_start: int,
    rock_counts_by_cell: Counter[tuple[int, int]],
    rock_density_caps: Mapping[str, Any],
    road_footprint_stats: Counter[str],
    road_footprint_rejections: Counter[str],
    tree_clearance_blocked_ids: set[str] | None = None,
    constrained_tilt_rejections: Counter[str] | None = None,
    clearing_index: ClearingIndex | None = None,
    candidate_rows: list[tuple[_Candidate, float, float]] | None = None,
    neighbor_index: RockNeighborIndex | None = None,
    road_vtex_cache: dict[tuple[int, int], Any] | None = None,
    cliff_seating: CliffSeatingRuntime | None = None,
    seating_stats: dict[str, Any] | None = None,
) -> tuple[int, Counter[str]]:
    if desired <= 0:
        return 0, Counter()
    rows = (
        candidate_rows
        if candidate_rows is not None
        else _profile_candidate_rows(all_candidates, profile, config)
    )
    failures: Counter[str] = Counter()
    if not rows:
        failures["no_measured_suitable_candidates"] = desired
        return 0, failures
    rng = random.Random(derive_seed(config.master_seed, GENERATION_NAMESPACE, pass_name, profile.key))
    blocked: set[str] = set()
    placed = 0
    # One weighted random permutation per mesh gives frequency/suitability
    # ordering without rebuilding a 12k-candidate eligible list for every
    # placement.  Each candidate is still accepted only after the live
    # occupancy, water, and transformed-bbox checks below.
    ordinary_order = sorted(
        range(len(rows)),
        key=lambda index: -math.log(max(rng.random(), 1e-15)) / max(float(rows[index][1]), 1e-12),
    )
    ordinary_cursor = 0
    profile_radius = _profile_horizontal_radius(bbox)
    if cliff and cliff_points:
        # Prefer candidates near an already accepted cliff so the measured
        # giant quota forms faces when the LAND slope/geometry permits it.
        face_distances: dict[str, float] = {}
        for candidate, _weight, _mask_value in rows:
            face_distances[candidate.candidate_id] = min(
                math.hypot(
                    candidate.x_gu - float(parent["position_gu"][0]),
                    candidate.y_gu - float(parent["position_gu"][1]),
                )
                for parent in cliff_points
            )
        ordinary_order.sort(
            key=lambda index: (
                0 if face_distances[rows[index][0].candidate_id] <= 8192.0 else 1,
                face_distances[rows[index][0].candidate_id],
            )
        )
    attempt_limit = len(rows)
    for ordinal in range(desired):
        success = False
        for _attempt in range(attempt_limit):
            if ordinary_cursor >= len(ordinary_order):
                failures["candidate_order_exhausted"] += 1
                break
            index = ordinary_order[ordinary_cursor]
            ordinary_cursor += 1
            candidate, weight, mask_value = rows[index]
            if candidate.candidate_id in blocked or candidate.candidate_id in used_ids:
                continue
            if profile.flora_role == "tree" and tree_clearance_blocked_ids is not None:
                if candidate.candidate_id in tree_clearance_blocked_ids:
                    failures["tree_large_rock_clearance"] += 1
                    blocked.add(candidate.candidate_id)
                    continue
            if profile.category in {"rocks", "cliff"}:
                rock_cap = _rock_cap_for_slope(candidate.slope_deg, rock_density_caps)
                if rock_cap is not None and rock_counts_by_cell[candidate.cell] >= rock_cap:
                    continue
            if cliff:
                slope_floor = float(config.cliff_min_slope_deg)
                seated_cliff = (
                    cliff_seating is not None and cliff_seating.has_profile(profile.key)
                )
                if not seated_cliff:
                    open_side_directions = [
                        str(value)
                        for value in profile.open_face_profile.get("open_directions", [])
                        if str(value) in _DIRECTION_VECTORS_XY
                    ]
                    if open_side_directions:
                        slope_floor = max(slope_floor, float(config.open_side_cliff_min_slope_deg))
                if candidate.terrain_z_thu <= 0.0:
                    continue
                if candidate.slope_deg < slope_floor:
                    if (
                        not seated_cliff
                        and open_side_directions
                        and candidate.slope_deg >= float(config.cliff_min_slope_deg)
                    ):
                        failures["open_side_cliff_slope"] += 1
                    continue
                if config.cliff_max_water_distance_gu is not None and candidate.water_distance_gu > config.cliff_max_water_distance_gu:
                    continue
                if not occupancy.can_place("rocks", (candidate.x_gu, candidate.y_gu), minimum_distance):
                    continue
            elif not occupancy.can_place(
                "rocks" if profile.category == "rocks" else "flora",
                (candidate.x_gu, candidate.y_gu),
                minimum_distance,
            ):
                continue
            if clearing_index is not None and profile.category == "flora":
                if clearing_index.blocks_point(candidate.x_gu, candidate.y_gu):
                    failures["clearing_blocked"] += 1
                    blocked.add(candidate.candidate_id)
                    continue
            if clearing_index is not None and profile.category in {"rocks", "cliff"}:
                if clearing_index.in_city_domain_point(candidate.x_gu, candidate.y_gu):
                    failures["city_domain_rocks_banned"] += 1
                    blocked.add(candidate.candidate_id)
                    continue
            seated_cliff = (
                cliff
                and cliff_seating is not None
                and cliff_seating.has_profile(profile.key)
            )
            accepted_values: tuple[
                tuple[float, float, float],
                str,
                str,
                float,
                float,
                dict[str, Any],
                dict[str, Any] | None,
                float,
                float,
                float,
                dict[str, Any] | None,
                dict[str, Any],
                dict[str, Any] | None,
            ] | None = None
            last_open_face_decision: dict[str, Any] | None = None
            geometry_rejected = False
            if seated_cliff:
                # Observed-pose seating replaces the sampled-rotation paths:
                # one recorded member, fixed sample set, one Z solve, then the
                # existing relief and road audits.
                attempt_started = time.perf_counter()
                member = cliff_seating.select_member(profile.key, candidate.slope_deg, rng)
                outcome, payload = cliff_seating.evaluate_attempt(
                    profile_key=profile.key,
                    member=member,
                    candidate_x=candidate.x_gu,
                    candidate_y=candidate.y_gu,
                    candidate_terrain_z_gu=candidate.terrain_z_gu,
                    candidate_slope_deg=candidate.slope_deg,
                    candidate_downhill_xy=candidate.downhill_direction_xy,
                    land_records=land_records,
                    bbox=bbox,
                    clearing_index=clearing_index,
                )
                if seating_stats is not None:
                    seating_stats["attempts"] += 1
                    seating_stats["evaluator_seconds"] += time.perf_counter() - attempt_started
                if outcome == "reject":
                    reason = str(payload.get("reason", "unknown"))
                    if seating_stats is not None:
                        seating_stats["rejections"][reason] += 1
                    failures[f"cliff_seating_{reason}"] += 1
                    blocked.add(candidate.candidate_id)
                    continue
                rotation = tuple(payload.pop("rotation_radians"))
                world_bbox = payload.pop("world_bbox")
                scale = float(payload["recorded_scale"])
                z_offset = float(payload["solved_z_gu"]) - candidate.terrain_z_gu
                mode = "observed_pose_member"
                rotation_source = (
                    f"cliff_seating:{payload['profile_id']}:"
                    f"{payload['mode_id']}:{payload['member_ref_id']}"
                )
                open_face_decision = None
                bottom_embed = float(candidate.terrain_z_gu) - float(world_bbox["min"][2])
                top_above = float(world_bbox["max"][2]) - float(candidate.terrain_z_gu)
                cliff_relief_audit = cliff_footprint_relief_audit(
                    land_records,
                    world_bbox,
                    config.cliff_min_slope_deg,
                )
                if not cliff_relief_audit["passed"]:
                    relief_reason = str(
                        cliff_relief_audit.get("failure_reason") or "insufficient_relief"
                    )
                    failures[f"cliff_footprint_{relief_reason}"] += 1
                    blocked.add(candidate.candidate_id)
                    continue
                footprint_audit = road_footprint_audit(
                    land_records,
                    world_bbox,
                    config.road_raw_vtex_values,
                    detail=False,
                    vtex_cache=road_vtex_cache,
                )
                road_hit_count = int(
                    footprint_audit.get(
                        "road_hit_count",
                        len(footprint_audit.get("road_hits", [])),
                    )
                )
                road_footprint_stats["checked_attempts"] += 1
                road_footprint_stats["checked_tiles"] += int(footprint_audit["checked_tile_count"])
                road_footprint_stats["resolved_tiles"] += int(footprint_audit["resolved_tile_count"])
                road_footprint_stats["road_hit_tiles"] += road_hit_count
                road_footprint_stats["missing_land_tiles"] += int(footprint_audit["missing_land_tile_count"])
                road_footprint_stats["missing_vtex_tiles"] += int(footprint_audit["missing_vtex_tile_count"])
                if road_hit_count:
                    road_footprint_stats["road_hit_attempts"] += 1
                if footprint_audit["missing_land_cells"]:
                    road_footprint_stats["missing_land_attempts"] += 1
                if footprint_audit["missing_vtex_cells"]:
                    road_footprint_stats["missing_vtex_attempts"] += 1
                if not footprint_audit["passed"]:
                    road_reason = str(footprint_audit.get("failure_reason") or "unknown")
                    road_footprint_stats["rejected_attempts"] += 1
                    road_footprint_rejections[road_reason] += 1
                    failures[f"road_footprint_{road_reason}"] += 1
                    blocked.add(candidate.candidate_id)
                    continue
                road_footprint_stats["passed_attempts"] += 1
                if seating_stats is not None:
                    seating_stats["accepted"] += 1
                    seating_stats["member_refs_used"].add(member.ref_id)
                    seating_stats["modes_used"].add(member.mode_id)
                accepted_values = (
                    rotation,
                    mode,
                    rotation_source,
                    scale,
                    z_offset,
                    world_bbox,
                    None,
                    bottom_embed,
                    top_above,
                    float(transformed_local_up_world_z(rotation)),
                    cliff_relief_audit,
                    _compact_road_footprint_audit(footprint_audit),
                    payload,
                )
            elif profile.category in {"rocks", "cliff"} and bool(
                profile.open_face_profile.get("open_directions")
            ):
                open_side_directions = [
                    str(value)
                    for value in profile.open_face_profile.get("open_directions", [])
                    if str(value) in _DIRECTION_VECTORS_XY
                ]
                use_pose_first = True
                pose_targets = _open_face_pose_targets(
                    profile,
                    candidate,
                    placements,
                    config,
                    profile_radius,
                    neighbor_index=neighbor_index,
                )
                if not pose_targets:
                    open_face_decision = {
                        "profile_status": str(profile.open_face_profile.get("status", "not_supplied")),
                        "open_directions_local": open_side_directions,
                        "orientation_attempts": 0,
                        "orientation_source": "pose_first",
                        "action": "no_target_skip",
                        "target_ref_id": None,
                        "alignment_dot": None,
                        "target_direction_xy": None,
                        "target_distance_gu": None,
                        "bottom_open": bool(profile.open_face_profile.get("bottom_open", False)),
                        "top_open": bool(profile.open_face_profile.get("top_open", False)),
                        "open_axes": [
                            str(value) for value in profile.open_face_profile.get("open_axes", [])
                        ],
                        "bottom_seating": None,
                        "chosen_open_direction": None,
                    }
                    last_open_face_decision = open_face_decision
                    failures["open_face_no_safe_orientation"] += 1
                    blocked.add(candidate.candidate_id)
                    continue
                target_kind, target_ref, target_direction, target_distance = pose_targets[0]
                del target_kind, target_ref, target_distance
                tilt_x, tilt_y, mode, rotation_source = _sample_pose_first_tilt(
                    profile, rng, candidate.slope_deg
                )
                orientation_attempts = 0
                for direction_name in open_side_directions:
                    orientation_attempts += 1
                    local = _DIRECTION_VECTORS_XY[direction_name]
                    rotz = _pose_first_open_face_yaw(
                        local,
                        target_direction,
                        rotation_x=tilt_x,
                        rotation_y=tilt_y,
                    )
                    rotation = (_round(tilt_x, 6), _round(tilt_y, 6), _round(rotz, 6))
                    local_up_world_z = transformed_local_up_world_z(rotation)
                    if local_up_world_z <= 0.0:
                        failures["flipped_orientation"] += 1
                        continue
                    max_local_up_tilt = config.max_local_up_tilt_degrees_by_mesh.get(profile.key)
                    if max_local_up_tilt is not None:
                        local_up_tilt = transformed_local_up_tilt_degrees(rotation)
                        if local_up_tilt > max_local_up_tilt:
                            failures["constrained_tilt_exceeded"] += 1
                            if constrained_tilt_rejections is not None:
                                constrained_tilt_rejections[profile.key] += 1
                            continue
                    open_face_decision = _open_face_orientation_decision(
                        profile,
                        candidate,
                        rotation,
                        placements,
                        config,
                        profile_radius,
                        neighbor_index=neighbor_index,
                    )
                    open_face_decision["orientation_attempts"] = orientation_attempts
                    open_face_decision["orientation_source"] = "pose_first"
                    open_face_decision["chosen_open_direction"] = direction_name
                    last_open_face_decision = open_face_decision
                    if open_face_decision.get("action") not in {
                        "terrain_downslope",
                        "adjacent_rock",
                    }:
                        continue
                    scale = _sample_quantile(profile.scale_distribution, rng, fallback=1.0)
                    if not math.isfinite(scale) or scale <= 0.0:
                        failures["invalid_measured_scale"] += 1
                        break
                    z_offset = _sample_quantile(profile.z_offset_distribution, rng, fallback=0.0)
                    position = [candidate.x_gu, candidate.y_gu, candidate.terrain_z_gu + z_offset]
                    world_bbox = transformed_bbox(bbox, position, rotation, scale)
                    if clearing_index is not None and profile.category in {"rocks", "cliff"}:
                        min_x, max_x, min_y, max_y = _aabb_xy_bounds(world_bbox)
                        if clearing_index.blocks_aabb(min_x, min_y, max_x, max_y):
                            failures["rock_clearing_blocked"] += 1
                            continue
                    footprint_audit = road_footprint_audit(
                        land_records,
                        world_bbox,
                        config.road_raw_vtex_values,
                        detail=False,
                        vtex_cache=road_vtex_cache,
                    )
                    road_hit_count = int(
                        footprint_audit.get(
                            "road_hit_count",
                            len(footprint_audit.get("road_hits", [])),
                        )
                    )
                    road_footprint_stats["checked_attempts"] += 1
                    road_footprint_stats["checked_tiles"] += int(footprint_audit["checked_tile_count"])
                    road_footprint_stats["resolved_tiles"] += int(footprint_audit["resolved_tile_count"])
                    road_footprint_stats["road_hit_tiles"] += road_hit_count
                    road_footprint_stats["missing_land_tiles"] += int(footprint_audit["missing_land_tile_count"])
                    road_footprint_stats["missing_vtex_tiles"] += int(footprint_audit["missing_vtex_tile_count"])
                    if road_hit_count:
                        road_footprint_stats["road_hit_attempts"] += 1
                    if footprint_audit["missing_land_cells"]:
                        road_footprint_stats["missing_land_attempts"] += 1
                    if footprint_audit["missing_vtex_cells"]:
                        road_footprint_stats["missing_vtex_attempts"] += 1
                    if not footprint_audit["passed"]:
                        failure_reason = str(footprint_audit.get("failure_reason") or "unknown")
                        road_footprint_stats["rejected_attempts"] += 1
                        road_footprint_rejections[failure_reason] += 1
                        failures[f"road_footprint_{failure_reason}"] += 1
                        continue
                    road_footprint_stats["passed_attempts"] += 1
                    minimum_embed = config.cliff_min_embed_gu if cliff else config.min_embed_gu
                    if open_face_decision.get("action") in {"terrain_downslope", "adjacent_rock"} or bool(
                        profile.open_face_profile.get("bottom_open")
                    ):
                        minimum_embed = max(minimum_embed, config.open_face_min_embed_gu)
                    intersects, bottom_embed, top_above = _surface_intersects_terrain(
                        world_bbox,
                        candidate.terrain_z_gu,
                        minimum_embed=minimum_embed,
                    )
                    if not intersects:
                        geometry_rejected = True
                        if bool(profile.open_face_profile.get("bottom_open")):
                            failures["bottom_open_not_embedded"] += 1
                        continue
                    if cliff and top_above < config.cliff_min_visible_gu:
                        failures["cliff_fully_buried"] += 1
                        geometry_rejected = True
                        continue
                    if (
                        not cliff
                        and profile.category == "rocks"
                        and bbox.get("volume_class") == "small"
                        and bottom_embed < config.min_embed_gu
                    ):
                        failures["small_rock_not_embedded"] += 1
                        geometry_rejected = True
                        continue
                    cliff_relief_audit: dict[str, Any] | None = None
                    if cliff:
                        cliff_relief_audit = cliff_footprint_relief_audit(
                            land_records,
                            world_bbox,
                            config.cliff_min_slope_deg,
                        )
                        if not cliff_relief_audit["passed"]:
                            failure_reason = str(
                                cliff_relief_audit.get("failure_reason") or "insufficient_relief"
                            )
                            failures[f"cliff_footprint_{failure_reason}"] += 1
                            continue
                    accepted_values = (
                        rotation,  # type: ignore[assignment]
                        mode,
                        rotation_source,
                        scale,
                        z_offset,
                        world_bbox,
                        open_face_decision,
                        bottom_embed,
                        top_above,
                        float(local_up_world_z),
                        cliff_relief_audit,
                        _compact_road_footprint_audit(footprint_audit),
                        None,
                    )
                    break
            else:
                orientation_attempt_limit = 1
                for orientation_index in range(orientation_attempt_limit):
                    rotation, mode, rotation_source = _sample_rotation(profile, rng, candidate.slope_deg)
                    local_up_world_z = (
                        transformed_local_up_world_z(rotation)
                        if profile.category in {"rocks", "cliff"}
                        else None
                    )
                    if local_up_world_z is not None and local_up_world_z <= 0.0:
                        failures["flipped_orientation"] += 1
                        continue
                    max_local_up_tilt = config.max_local_up_tilt_degrees_by_mesh.get(profile.key)
                    if max_local_up_tilt is not None:
                        local_up_tilt = transformed_local_up_tilt_degrees(rotation)
                        if local_up_tilt > max_local_up_tilt:
                            failures["constrained_tilt_exceeded"] += 1
                            if constrained_tilt_rejections is not None:
                                constrained_tilt_rejections[profile.key] += 1
                            continue
                    open_face_decision = (
                        _open_face_orientation_decision(
                            profile,
                            candidate,
                            rotation,
                            placements,
                            config,
                            profile_radius,
                            neighbor_index=neighbor_index,
                        )
                        if profile.category in {"rocks", "cliff"}
                        else {
                            "action": "not_a_rock",
                            "profile_status": "not_applicable",
                            "open_directions_local": [],
                        }
                    )
                    open_face_decision["orientation_attempts"] = orientation_index + 1
                    open_face_decision["orientation_source"] = "measured_sample"
                    last_open_face_decision = open_face_decision
                    if profile.category in {"rocks", "cliff"} and open_face_decision.get("action") not in {
                        "closed_profile",
                        "terrain_downslope",
                        "adjacent_rock",
                    }:
                        continue
                    scale = _sample_quantile(profile.scale_distribution, rng, fallback=1.0)
                    if not math.isfinite(scale) or scale <= 0.0:
                        failures["invalid_measured_scale"] += 1
                        break
                    z_offset = _sample_quantile(profile.z_offset_distribution, rng, fallback=0.0)
                    position = [candidate.x_gu, candidate.y_gu, candidate.terrain_z_gu + z_offset]
                    world_bbox = transformed_bbox(bbox, position, rotation, scale)
                    if clearing_index is not None and profile.category in {"rocks", "cliff"}:
                        min_x, max_x, min_y, max_y = _aabb_xy_bounds(world_bbox)
                        if clearing_index.blocks_aabb(min_x, min_y, max_x, max_y):
                            failures["rock_clearing_blocked"] += 1
                            continue
                    footprint_audit: dict[str, Any] | None = None
                    if profile.category in {"rocks", "cliff"}:
                        footprint_audit = road_footprint_audit(
                            land_records,
                            world_bbox,
                            config.road_raw_vtex_values,
                            detail=False,
                            vtex_cache=road_vtex_cache,
                        )
                        road_hit_count = int(
                            footprint_audit.get(
                                "road_hit_count",
                                len(footprint_audit.get("road_hits", [])),
                            )
                        )
                        road_footprint_stats["checked_attempts"] += 1
                        road_footprint_stats["checked_tiles"] += int(footprint_audit["checked_tile_count"])
                        road_footprint_stats["resolved_tiles"] += int(footprint_audit["resolved_tile_count"])
                        road_footprint_stats["road_hit_tiles"] += road_hit_count
                        road_footprint_stats["missing_land_tiles"] += int(footprint_audit["missing_land_tile_count"])
                        road_footprint_stats["missing_vtex_tiles"] += int(footprint_audit["missing_vtex_tile_count"])
                        if road_hit_count:
                            road_footprint_stats["road_hit_attempts"] += 1
                        if footprint_audit["missing_land_cells"]:
                            road_footprint_stats["missing_land_attempts"] += 1
                        if footprint_audit["missing_vtex_cells"]:
                            road_footprint_stats["missing_vtex_attempts"] += 1
                        if not footprint_audit["passed"]:
                            failure_reason = str(footprint_audit.get("failure_reason") or "unknown")
                            road_footprint_stats["rejected_attempts"] += 1
                            road_footprint_rejections[failure_reason] += 1
                            failures[f"road_footprint_{failure_reason}"] += 1
                            continue
                        road_footprint_stats["passed_attempts"] += 1
                    minimum_embed = config.cliff_min_embed_gu if cliff else config.min_embed_gu
                    if open_face_decision.get("action") in {"terrain_downslope", "adjacent_rock"} or bool(
                        profile.open_face_profile.get("bottom_open")
                    ):
                        minimum_embed = max(minimum_embed, config.open_face_min_embed_gu)
                    intersects, bottom_embed, top_above = _surface_intersects_terrain(
                        world_bbox,
                        candidate.terrain_z_gu,
                        minimum_embed=minimum_embed,
                    )
                    if not intersects:
                        geometry_rejected = True
                        if bool(profile.open_face_profile.get("bottom_open")):
                            failures["bottom_open_not_embedded"] += 1
                        continue
                    if cliff and top_above < config.cliff_min_visible_gu:
                        failures["cliff_fully_buried"] += 1
                        geometry_rejected = True
                        continue
                    if (
                        not cliff
                        and profile.category == "rocks"
                        and bbox.get("volume_class") == "small"
                        and bottom_embed < config.min_embed_gu
                    ):
                        failures["small_rock_not_embedded"] += 1
                        geometry_rejected = True
                        continue
                    cliff_relief_audit = None
                    if cliff:
                        cliff_relief_audit = cliff_footprint_relief_audit(
                            land_records,
                            world_bbox,
                            config.cliff_min_slope_deg,
                        )
                        if not cliff_relief_audit["passed"]:
                            failure_reason = str(
                                cliff_relief_audit.get("failure_reason") or "insufficient_relief"
                            )
                            failures[f"cliff_footprint_{failure_reason}"] += 1
                            continue
                    accepted_values = (
                        rotation,
                        mode,
                        rotation_source,
                        scale,
                        z_offset,
                        world_bbox,
                        open_face_decision,
                        bottom_embed,
                        top_above,
                        float(local_up_world_z) if local_up_world_z is not None else 0.0,
                        cliff_relief_audit,
                        _compact_road_footprint_audit(footprint_audit)
                        if footprint_audit is not None
                        else {},
                        None,
                    )
                    break
            if accepted_values is None:
                if profile.category in {"rocks", "cliff"} and last_open_face_decision is not None:
                    if last_open_face_decision.get("action") not in {
                        "closed_profile",
                        "terrain_downslope",
                        "adjacent_rock",
                    }:
                        failures["open_face_no_safe_orientation"] += 1
                if geometry_rejected:
                    failures["surface_geometry_rejection"] += 1
                blocked.add(candidate.candidate_id)
                continue
            (
                rotation,
                mode,
                rotation_source,
                scale,
                z_offset,
                world_bbox,
                open_face_decision,
                _bottom_embed,
                _top_above,
                _local_up_world_z,
                cliff_relief_audit,
                compact_road_footprint_audit,
                seating_evidence,
            ) = accepted_values
            if (
                open_face_decision is not None
                and profile.category in {"rocks", "cliff"}
                and open_face_decision.get("action") in {
                    "terrain_downslope",
                    "adjacent_rock",
                }
            ):
                rotation_source = f"{rotation_source}+open_face_rule"
            ref_id = f"{pass_name.lower()}_{profile.category}_{ordinal_start + placed:05d}"
            face_link = _nearest_face_link(candidate, cliff_points) if cliff else None
            placement = _new_placement(
                ref_id=ref_id,
                candidate=candidate,
                profile=profile,
                pass_name=pass_name,
                bbox=bbox,
                rotation=rotation,
                rotation_mode_value=mode,
                rotation_source=rotation_source,
                scale=scale,
                z_offset_gu=z_offset,
                world_bbox=world_bbox,
                suitability_weight=weight,
                clearing_value=mask_value,
                config=config,
                open_face_decision=open_face_decision if profile.category in {"rocks", "cliff"} else None,
                face_link=face_link,
                cliff_footprint_audit=cliff_relief_audit,
                road_footprint_audit=compact_road_footprint_audit
                if profile.category in {"rocks", "cliff"}
                else None,
                cliff_seating_evidence=seating_evidence,
            )
            if seating_stats is not None and seating_evidence is not None:
                margin = float(seating_evidence["stability_margin_gu"])
                previous = seating_stats["worst_margin_by_mesh"].get(profile.key)
                if previous is None or margin < previous[0]:
                    seating_stats["worst_margin_by_mesh"][profile.key] = (margin, ref_id)
            placements.append(placement)
            used_ids.add(candidate.candidate_id)
            occupancy.add("rocks" if cliff or profile.category == "rocks" else "flora", (candidate.x_gu, candidate.y_gu), minimum_distance)
            if cliff:
                cliff_points.append(placement)
            if profile.category in {"rocks", "cliff"}:
                rock_counts_by_cell[candidate.cell] += 1
                if neighbor_index is not None:
                    world_min = world_bbox.get("min")
                    world_max = world_bbox.get("max")
                    if (
                        isinstance(world_min, Sequence)
                        and isinstance(world_max, Sequence)
                        and len(world_min) >= 2
                        and len(world_max) >= 2
                    ):
                        target_radius = 0.5 * math.hypot(
                            float(world_max[0]) - float(world_min[0]),
                            float(world_max[1]) - float(world_min[1]),
                        )
                    else:
                        target_radius = profile_radius
                    neighbor_index.add(
                        ref_id, candidate.x_gu, candidate.y_gu, target_radius
                    )
            placed += 1
            success = True
            break
        if not success:
            failures["quota_shortfall"] += desired - placed
            break
    return placed, failures


def _profile_adherence(profile: _Species, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def values(path: Sequence[str]) -> list[float]:
        output: list[float] = []
        for row in rows:
            value: Any = row
            for key in path:
                if not isinstance(value, Mapping):
                    value = None
                    break
                value = value.get(key)
            numeric = _finite(value)
            if numeric is not None:
                output.append(numeric)
        return output

    placed_z = values(("terrain", "z_offset_gu"))
    placed_slope = values(("terrain", "slope_deg"))
    placed_elevation = values(("terrain", "elevation_gu"))
    placed_water = values(("terrain", "distance_to_water_gu"))
    measured_rotation = profile.rotation_distribution
    placed_rotations = [row.get("rotation_radians", [0.0, 0.0, 0.0]) for row in rows]
    placed_rotations = [tuple(float(value) for value in rotation) for rotation in placed_rotations if isinstance(rotation, Sequence) and len(rotation) == 3]
    placed_modes = Counter(str(row.get("rotation_mode", "z_only")) for row in rows)
    measured_modes = measured_rotation.get("mode_counts", {}) if isinstance(measured_rotation, Mapping) else {}

    def envelope_result(placed: Sequence[float], measured: Mapping[str, Any]) -> dict[str, Any]:
        low = _finite(measured.get("min"))
        high = _finite(measured.get("max"))
        inside = sum(1 for value in placed if (low is None or value >= low) and (high is None or value <= high))
        return {
            "measured": dict(measured),
            "placed": _distribution_from_values(placed),
            "within_measured_min_max_count": inside,
            "within_measured_min_max_fraction": _round(inside / len(placed), 6) if placed else None,
        }

    measured_conditions = profile.conditions
    return {
        "mesh": profile.mesh,
        "category": profile.category,
        "flora_role": profile.flora_role,
        "measured_frequency": profile.frequency,
        "placed_count": len(rows),
        "z_offset": envelope_result(placed_z, profile.z_offset_distribution),
        "slope": envelope_result(placed_slope, measured_conditions["slope_deg"]),
        "elevation": envelope_result(placed_elevation, measured_conditions["elevation_gu"]),
        "water_distance": envelope_result(placed_water, measured_conditions["water_distance_gu"]),
        "orientation": {
            "measured": dict(measured_rotation),
            "placed": {
                "mode_counts": dict(sorted(placed_modes.items())),
                "x_radians": _distribution_from_values(rotation[0] for rotation in placed_rotations),
                "y_radians": _distribution_from_values(rotation[1] for rotation in placed_rotations),
                "z_radians": _distribution_from_values(rotation[2] for rotation in placed_rotations),
                "z_only_fraction": _round(placed_modes.get("z_only", 0) / len(rows), 6) if rows else None,
            },
            "measured_mode_counts": dict(measured_modes) if isinstance(measured_modes, Mapping) else {},
            "forbidden_xy_tilt_count": sum(
                1
                for row in rows
                if int(measured_modes.get("full", 0)) <= 0
                and str(row.get("rotation_mode", "z_only")) == "full"
            ),
        },
    }


def generate_scatter_document(
    land_records: Mapping[tuple[int, int], LandRecord],
    scatter_analysis: Mapping[str, Any],
    cliff_analysis: Mapping[str, Any],
    bbox_cache: Mapping[str, Any],
    *,
    config: GenerationConfig = GenerationConfig(),
    terrain_source: str = "tamriel.esm",
    open_face_profiles: Mapping[str, Any] | None = None,
    open_face_source: str = "output/rock_openface_profiles.json",
    baseline_document: Mapping[str, Any] | None = None,
    clearing: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    edited_land_source: str | None = None,
    cliff_seating_config: Mapping[str, Any] | None = None,
    cliff_seating_profiles: Mapping[str, Any] | None = None,
    timing_logger: Callable[[str, float, float, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Generate a deterministic settlement block from direct LAND and measured data."""

    overall_started = time.perf_counter()

    if (cliff_seating_config is None) != (cliff_seating_profiles is None):
        raise ValueError(
            "cliff seating config and profiles must be supplied together"
        )

    def phase_done(phase: str, started: float, **details: Any) -> None:
        if timing_logger is None:
            return
        now = time.perf_counter()
        timing_logger(phase, now - started, now - overall_started, details)

    def timed_rows(phase: str, fn: Callable[[], Any], **details: Any) -> Any:
        started = time.perf_counter()
        result = fn()
        phase_done(phase, started, **details)
        return result

    clearing_index: ClearingIndex | MultiClearingIndex | None = None
    if clearing is not None:
        clearing_index = build_clearing_index(clearing)

    main_profiles, cliff_profiles, giant_keys = timed_rows(
        "build_species",
        lambda: _build_species(scatter_analysis, cliff_analysis, open_face_profiles),
    )
    profiles_before_quarantine = list(main_profiles) + list(cliff_profiles)
    quarantined_profiles = [
        profile for profile in profiles_before_quarantine if profile.key in config.quarantined_mesh_keys
    ]
    main_profiles = [profile for profile in main_profiles if profile.key not in config.quarantined_mesh_keys]
    cliff_profiles = [profile for profile in cliff_profiles if profile.key not in config.quarantined_mesh_keys]

    # Cliff seating preflight runs before candidate construction: profile and
    # analysis provenance must match, and quarantined / zero-eligible-mode
    # meshes leave quota allocation before any candidate is built.  The
    # measured-frequency ratio that sets the cliff TARGET uses the
    # pre-exclusion totals: quota redistributes across the remaining eligible
    # meshes, the target itself does not shrink.
    measured_cliff_total_unfiltered = sum(profile.frequency for profile in cliff_profiles)
    seating_runtime: CliffSeatingRuntime | None = None
    seating_preflight_started = time.perf_counter()
    if cliff_seating_config is not None:
        seating_runtime = CliffSeatingRuntime(
            cliff_seating_config,
            cliff_seating_profiles,
            cliff_analysis,
            quarantined_keys=config.quarantined_mesh_keys,
        )
        profile_excluded = {
            key for key in seating_runtime.excluded_meshes
        }
        removed_seating = [
            profile for profile in cliff_profiles if profile.key in profile_excluded
        ]
        cliff_profiles = [
            profile for profile in cliff_profiles if profile.key not in profile_excluded
        ]
        if not cliff_profiles:
            raise ValueError("cliff seating left no eligible cliff profiles")
    else:
        removed_seating = []
    seating_preflight = None
    if seating_runtime is not None:
        seating_preflight = {
            "profile_id": seating_runtime.profile_id,
            "matched_mesh_count": len(seating_runtime.mesh_states) + len(seating_runtime.excluded_meshes),
            "eligible_meshes": sorted(seating_runtime.mesh_states),
            "profile_excluded_meshes": dict(sorted(seating_runtime.excluded_meshes.items())),
            "manually_quarantined_meshes": dict(sorted(seating_runtime.quarantined_meshes.items())),
            "removed_measured_frequency": {
                "profile_excluded": sum(seating_runtime.excluded_meshes.values()),
                "manually_quarantined": sum(seating_runtime.quarantined_meshes.values()),
            },
            "eligible_mode_count": sum(
                len(state.members) for state in seating_runtime.mesh_states.values()
            ),
        }
    phase_done(
        "preflight.cliff_seating",
        seating_preflight_started,
        enabled=seating_runtime is not None,
        removed_profiles=len(removed_seating),
    )
    water_index, water_summary = timed_rows(
        "build_water_index", lambda: _build_water_index(land_records, config),
        land_cells=len(land_records),
    )
    candidates_by_cell, candidate_summary = timed_rows(
        "build_candidates", lambda: _build_candidates(land_records, water_index, config),
        candidate_spacing_gu=config.candidate_spacing_gu,
    )
    target_started = time.perf_counter()
    target_cells = sorted(
        config.target_cells
        if config.target_cells is not None
        else {
            (x, y)
            for y in range(config.bounds[2], config.bounds[3] + 1)
            for x in range(config.bounds[0], config.bounds[1] + 1)
        },
        key=lambda cell: (cell[1], cell[0]),
    )
    all_candidates = [candidate for cell in target_cells for candidate in candidates_by_cell[cell]]
    phase_done(
        "select_target_cells",
        target_started,
        target_cells=len(target_cells),
        candidates=len(all_candidates),
    )
    if not all_candidates:
        raise ValueError("target block has no valid direct-LAND candidates")
    arrays_started = time.perf_counter()
    candidate_arrays = _build_candidate_arrays(all_candidates, config)
    phase_done("build_candidate_arrays", arrays_started, candidates=len(all_candidates))

    density = scatter_analysis.get("density", {})
    wilderness = density.get("wilderness", {}) if isinstance(density, Mapping) else {}
    wilderness_mean = wilderness.get("mean_per_cell", {}) if isinstance(wilderness, Mapping) else {}
    target_flora_per_cell = float(config.target_flora_per_cell if config.target_flora_per_cell is not None else wilderness_mean.get("flora_refs", 76.836364))
    measured_rock_mean = float(wilderness_mean.get("rock_refs", 94.945455))
    target_rocks_per_cell = float(
        config.target_rocks_per_cell
        if config.target_rocks_per_cell is not None
        else measured_rock_mean * config.rock_density_factor
    )
    baseline_flora_total = int(round(target_flora_per_cell * len(target_cells)))
    target_rocks_total = int(round(target_rocks_per_cell * len(target_cells)))
    measured_rock_total = sum(profile.frequency for profile in main_profiles if profile.category == "rocks") + sum(profile.frequency for profile in cliff_profiles)
    if seating_runtime is not None:
        measured_cliff_total = measured_cliff_total_unfiltered
        measured_rock_total += sum(profile.frequency for profile in removed_seating)
    else:
        measured_cliff_total = sum(profile.frequency for profile in cliff_profiles)
    target_cliffs = int(round(target_rocks_total * measured_cliff_total / max(1, measured_rock_total)))
    if config.target_cliffs_per_cell is not None:
        target_cliffs = int(round(float(config.target_cliffs_per_cell) * len(target_cells)))

    quarantine_by_key = {
        profile.key: {
            "mesh": profile.mesh,
            "category": profile.category,
            "measured_frequency": profile.frequency,
        }
        for profile in sorted(quarantined_profiles, key=lambda item: (item.key, item.category))
    }
    quarantine_audit: dict[str, Any] = {
        "policy_paths": list(QUARANTINED_MESH_PATHS),
        "normalized_policy_paths": sorted(config.quarantined_mesh_keys),
        "rule": "remove exactly these normalized mesh profiles before eligibility and quota allocation",
        "profiles_before_quota_allocation": len(profiles_before_quarantine),
        "removed_profile_count": len(quarantined_profiles),
        "removed_profiles": list(quarantine_by_key.values()),
        "removed_measured_frequency_by_category": dict(
            sorted(
                Counter(profile.category for profile in quarantined_profiles).items()
            )
        ),
        "missing_policy_paths": [
            path
            for path in QUARANTINED_MESH_PATHS
            if normalize_mesh_key(path) not in {profile.key for profile in quarantined_profiles}
        ],
    }

    occupancy = OccupancyIndex()
    rock_neighbor_index = RockNeighborIndex()
    used_flora: set[str] = set()
    used_rocks: set[str] = set()
    rock_counts_by_cell: Counter[tuple[int, int]] = Counter()
    rock_density_caps = _rock_density_caps(scatter_analysis, config)
    placements: list[dict[str, Any]] = []
    cliffs: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    road_footprint_stats: Counter[str] = Counter()
    road_footprint_rejections: Counter[str] = Counter()
    constrained_tilt_rejections: Counter[str] = Counter()
    road_vtex_cache: dict[tuple[int, int], Any] = {}
    bbox_cache_by_key = {normalize_mesh_key(str(key)): value for key, value in (bbox_cache.get("meshes", {}) if isinstance(bbox_cache.get("meshes", {}), Mapping) else {}).items()}

    def profile_bbox(profile: _Species) -> dict[str, Any]:
        row = bbox_cache_by_key.get(profile.key)
        if not isinstance(row, Mapping):
            row = _bbox_row(bbox_cache, profile.mesh)
        return bbox_info(row)

    profile_row_cache: dict[str, list[tuple[_Candidate, float, float]]] = {}

    def profile_rows(profile: _Species, phase: str) -> list[tuple[_Candidate, float, float]]:
        cached = profile_row_cache.get(profile.key)
        if cached is not None:
            return cached
        rows = timed_rows(
            phase,
            lambda: _profile_candidate_rows(
                all_candidates, profile, config, arrays=candidate_arrays
            ),
            mesh=profile.mesh,
            category=profile.category,
        )
        profile_row_cache[profile.key] = rows
        return rows

    # Cliffs are allocated by the measured share of all rock refs, then placed
    # first.  The face affinity is only a continuity preference; every actual
    # acceptance still comes from that mesh's own envelopes and geometry.
    cliff_eligible: set[str] = set()
    cliff_eligibility_started = time.perf_counter()
    for profile in cliff_profiles:
        if profile_rows(profile, "eligibility.cliff_profile"):
            cliff_eligible.add(profile.key)
    phase_done(
        "eligibility.cliff_profiles",
        cliff_eligibility_started,
        profiles=len(cliff_profiles),
        eligible=len(cliff_eligible),
    )
    cliff_quotas = _quota_counts(target_cliffs, cliff_profiles, eligible=cliff_eligible)
    cliff_order = sorted(cliff_profiles, key=lambda profile: (cliff_quotas[profile.key], profile.key))
    cliff_ordinals = 0
    cliff_placement_seconds = 0.0
    seating_stats: dict[str, Any] | None = (
        {
            "attempts": 0,
            "rejections": Counter(),
            "accepted": 0,
            "evaluator_seconds": 0.0,
            "worst_margin_by_mesh": {},
            "member_refs_used": set(),
            "modes_used": set(),
        }
        if seating_runtime is not None
        else None
    )
    for profile in cliff_order:
        placement_started = time.perf_counter()
        placed, profile_failures = _place_profile(
            profile,
            cliff_quotas[profile.key],
            all_candidates=all_candidates,
            land_records=land_records,
            bbox=profile_bbox(profile),
            config=config,
            occupancy=occupancy,
            used_ids=used_rocks,
            placements=placements,
            pass_name="A",
            minimum_distance=float(config.min_distances_gu["cliff"]),
            cliff=True,
            cliff_points=cliffs,
            ordinal_start=cliff_ordinals,
            rock_counts_by_cell=rock_counts_by_cell,
            rock_density_caps=rock_density_caps,
            road_footprint_stats=road_footprint_stats,
            road_footprint_rejections=road_footprint_rejections,
            constrained_tilt_rejections=constrained_tilt_rejections,
            clearing_index=clearing_index,
            candidate_rows=profile_row_cache.get(profile.key),
            neighbor_index=rock_neighbor_index,
            road_vtex_cache=road_vtex_cache,
            cliff_seating=seating_runtime,
            seating_stats=seating_stats,
        )
        phase_done(
            "placement.cliff_profile",
            placement_started,
            mesh=profile.mesh,
            quota=cliff_quotas[profile.key],
            placed=placed,
        )
        cliff_placement_seconds += time.perf_counter() - placement_started
        cliff_ordinals += placed
        failures.update({f"cliff:{profile.key}:{key}": value for key, value in profile_failures.items()})
    cliffs = [row for row in placements if row["pass"] == "A"]
    cliff_count = len(cliffs)

    cliff_seating_audit = None
    if seating_stats is not None:
        worst_margin_by_mesh = {
            key: {"stability_margin_gu": _round(value[0], 3), "ref_id": value[1]}
            for key, value in sorted(seating_stats["worst_margin_by_mesh"].items())
        }
        z_adjustments = [
            float(row["cliff_seating"]["z_adjustment_gu"])
            for row in cliffs
            if isinstance(row.get("cliff_seating"), Mapping)
        ]
        embed_mins = [
            float(row["cliff_seating"]["support_embed_min_gu"])
            for row in cliffs
            if isinstance(row.get("cliff_seating"), Mapping)
        ]
        embed_maxs = [
            float(row["cliff_seating"]["support_embed_max_gu"])
            for row in cliffs
            if isinstance(row.get("cliff_seating"), Mapping)
        ]
        cover_margins = [
            float(row["cliff_seating"]["lateral_cover_margin_gu"])
            for row in cliffs
            if isinstance(row.get("cliff_seating"), Mapping)
            and row["cliff_seating"].get("lateral_cover_margin_gu") is not None
        ]
        front_margins = [
            float(row["cliff_seating"]["visible_front_margin_gu"])
            for row in cliffs
            if isinstance(row.get("cliff_seating"), Mapping)
        ]
        residuals = [
            float(row["cliff_seating"]["rotation_roundtrip_residual"])
            for row in cliffs
            if isinstance(row.get("cliff_seating"), Mapping)
        ]
        alignment_dots = [
            float(row["cliff_seating"]["lateral_alignment_dot"])
            for row in cliffs
            if isinstance(row.get("cliff_seating"), Mapping)
            and row["cliff_seating"].get("lateral_alignment_dot") is not None
        ]
        cliff_seating_audit = {
            **(seating_preflight or {}),
            "target_refs": target_cliffs,
            "placed_refs": cliff_count,
            "attempt_count": seating_stats["attempts"],
            "accepted_count": seating_stats["accepted"],
            "modes_used_count": len(seating_stats["modes_used"]),
            "source_members_used_count": len(seating_stats["member_refs_used"]),
            "rejections": dict(sorted(seating_stats["rejections"].items())),
            "z_adjustment_gu": _distribution_from_values(z_adjustments),
            "support_embed_min_gu": _distribution_from_values(embed_mins),
            "support_embed_max_gu": _distribution_from_values(embed_maxs),
            "lateral_cover_margin_gu": _distribution_from_values(cover_margins),
            "visible_front_margin_gu": _distribution_from_values(front_margins),
            "rotation_roundtrip_residual": _distribution_from_values(residuals),
            "lateral_alignment_dot": _distribution_from_values(alignment_dots),
            "worst_margin_ref_by_mesh": worst_margin_by_mesh,
            "seating_evaluator_seconds": _round(seating_stats["evaluator_seconds"], 6),
            "cliff_placement_seconds_total": _round(cliff_placement_seconds, 6),
        }

    # All remaining rock density is allocated to ordinary, non-giant meshes.
    normal_profiles = [profile for profile in main_profiles if profile.category == "rocks"]
    normal_rock_target = max(0, target_rocks_total - cliff_count)
    normal_eligibility_started = time.perf_counter()
    normal_eligible = {
        profile.key for profile in normal_profiles
        if profile_rows(profile, "eligibility.rock_profile")
    }
    phase_done(
        "eligibility.rock_profiles",
        normal_eligibility_started,
        profiles=len(normal_profiles),
        eligible=len(normal_eligible),
    )
    normal_quotas = _quota_counts(normal_rock_target, normal_profiles, eligible=normal_eligible)
    normal_order = sorted(normal_profiles, key=lambda profile: (normal_quotas[profile.key], profile.key))
    rock_ordinals = 0
    for profile in normal_order:
        placement_started = time.perf_counter()
        placed, profile_failures = _place_profile(
            profile,
            normal_quotas[profile.key],
            all_candidates=all_candidates,
            land_records=land_records,
            bbox=profile_bbox(profile),
            config=config,
            occupancy=occupancy,
            used_ids=used_rocks,
            placements=placements,
            pass_name="B",
            minimum_distance=float(config.min_distances_gu["rocks"]),
            cliff=False,
            cliff_points=cliffs,
            ordinal_start=rock_ordinals,
            rock_counts_by_cell=rock_counts_by_cell,
            rock_density_caps=rock_density_caps,
            road_footprint_stats=road_footprint_stats,
            road_footprint_rejections=road_footprint_rejections,
            constrained_tilt_rejections=constrained_tilt_rejections,
            clearing_index=clearing_index,
            candidate_rows=profile_row_cache.get(profile.key),
            neighbor_index=rock_neighbor_index,
            road_vtex_cache=road_vtex_cache,
        )
        phase_done(
            "placement.rock_profile",
            placement_started,
            mesh=profile.mesh,
            quota=normal_quotas[profile.key],
            placed=placed,
        )
        rock_ordinals += placed
        failures.update({f"rocks:{profile.key}:{key}": value for key, value in profile_failures.items()})

    # Rocks and cliffs are complete before flora begins.  Build the clearance
    # blocked set once from those accepted transformed AABBs so every tree
    # profile sees the same geometric exclusion and the rejection count is not
    # dependent on profile iteration order.
    accepted_rock_cliff_rows = [
        row for row in placements if row.get("category") in {"rocks", "cliff"}
    ]
    tree_clearance_started = time.perf_counter()
    tree_clearance_blocked_ids, tree_clearance_aabbs = tree_clearance_blocked_candidate_ids(
        all_candidates,
        accepted_rock_cliff_rows,
        minimum_horizontal_span_gu=config.tree_clearance_min_horizontal_span_gu,
        margin_gu=config.tree_clearance_margin_gu,
    )
    phase_done(
        "tree_clearance_index",
        tree_clearance_started,
        candidates=len(all_candidates),
        blocked=len(tree_clearance_blocked_ids),
        accepted_rock_cliff_rows=len(accepted_rock_cliff_rows),
        large_aabb_count=len(tree_clearance_aabbs),
    )

    flora_profiles = [profile for profile in main_profiles if profile.category == "flora"]
    flora_eligibility_started = time.perf_counter()
    flora_eligible = {
        profile.key
        for profile in flora_profiles
        if profile_rows(profile, "eligibility.flora_profile")
    }
    phase_done(
        "eligibility.flora_profiles",
        flora_eligibility_started,
        profiles=len(flora_profiles),
        eligible=len(flora_eligible),
    )
    flora_quotas, flora_density_split = _flora_quota_counts(
        baseline_flora_total,
        flora_profiles,
        eligible=flora_eligible,
    )
    adjusted_flora_total = int(flora_density_split["integer_targets"]["flora_refs"])
    flora_order = sorted(flora_profiles, key=lambda profile: (flora_quotas[profile.key], profile.key))
    flora_ordinals = 0
    for profile in flora_order:
        placement_started = time.perf_counter()
        placed, profile_failures = _place_profile(
            profile,
            flora_quotas[profile.key],
            all_candidates=all_candidates,
            land_records=land_records,
            bbox=profile_bbox(profile),
            config=config,
            occupancy=occupancy,
            used_ids=used_flora,
            placements=placements,
            pass_name="B",
            minimum_distance=float(config.min_distances_gu["flora"]),
            cliff=False,
            cliff_points=cliffs,
            ordinal_start=flora_ordinals,
            rock_counts_by_cell=rock_counts_by_cell,
            rock_density_caps=rock_density_caps,
            road_footprint_stats=road_footprint_stats,
            road_footprint_rejections=road_footprint_rejections,
            tree_clearance_blocked_ids=tree_clearance_blocked_ids,
            constrained_tilt_rejections=constrained_tilt_rejections,
            clearing_index=clearing_index,
            candidate_rows=profile_row_cache.get(profile.key),
        )
        phase_done(
            "placement.flora_profile",
            placement_started,
            mesh=profile.mesh,
            quota=flora_quotas[profile.key],
            placed=placed,
        )
        flora_ordinals += placed
        failures.update({f"flora:{profile.key}:{key}": value for key, value in profile_failures.items()})

    assembly_started = time.perf_counter()
    placements.sort(key=lambda row: (row["cell"][1], row["cell"][0], 0 if row["pass"] == "A" else 1, row["ref_id"]))
    cells: list[dict[str, Any]] = []
    for cell in target_cells:
        refs = [row for row in placements if tuple(row["cell"]) == cell]
        cells.append(
            {
                "grid": [cell[0], cell[1]],
                "refs": refs,
                "stats": {
                    "ref_count": len(refs),
                    "cliff_refs": sum(1 for row in refs if row["pass"] == "A"),
                    "flora_refs": sum(1 for row in refs if row["category"] == "flora"),
                    "tree_refs": sum(
                        1 for row in refs if row["category"] == "flora" and row.get("flora_role") == "tree"
                    ),
                    "undergrowth_refs": sum(
                        1 for row in refs if row["category"] == "flora" and row.get("flora_role") != "tree"
                    ),
                    "rock_refs": sum(1 for row in refs if row["category"] in {"rocks", "cliff"}),
                    "main_rock_refs": sum(1 for row in refs if row["category"] == "rocks"),
                    "stacker_refs": 0,
                    "clearing_refs": sum(1 for row in refs if row["clearing"]["is_clearing"]),
                },
            }
        )
    actual_stats = [row["stats"] for row in cells]
    phase_done(
        "assemble_cells",
        assembly_started,
        cells=len(cells),
        placements=len(placements),
    )
    actual_means = {
        "flora_refs": sum(row["flora_refs"] for row in actual_stats) / len(actual_stats),
        "tree_refs": sum(row["tree_refs"] for row in actual_stats) / len(actual_stats),
        "undergrowth_refs": sum(row["undergrowth_refs"] for row in actual_stats) / len(actual_stats),
        "rock_refs": sum(row["rock_refs"] for row in actual_stats) / len(actual_stats),
        "scatter_refs": sum(row["ref_count"] for row in actual_stats) / len(actual_stats),
        "cliff_refs": sum(row["cliff_refs"] for row in actual_stats) / len(actual_stats),
        "stacker_refs": 0.0,
    }
    density_comparison = {
        "reference": "output/vorndgad_scatter_analysis.json density.wilderness.mean_per_cell; settlements excluded",
        "target_mean_per_cell": {
            "flora_refs": _round(adjusted_flora_total / len(target_cells), 6),
            "tree_refs": _round(flora_density_split["integer_targets"]["tree_refs"] / len(target_cells), 6),
            "undergrowth_refs": _round(
                flora_density_split["integer_targets"]["undergrowth_refs"] / len(target_cells), 6
            ),
            "rock_refs": _round(target_rocks_per_cell, 6),
        },
        "baseline_target_mean_per_cell": {
            "flora_refs": _round(target_flora_per_cell, 6),
            "tree_refs": _round(
                flora_density_split["baseline_targets_before_integer_allocation"]["tree_refs"]
                / len(target_cells),
                6,
            ),
            "undergrowth_refs": _round(
                flora_density_split["baseline_targets_before_integer_allocation"]["undergrowth_refs"]
                / len(target_cells),
                6,
            ),
        },
        "actual_mean_per_cell": {key: _round(value, 6) for key, value in actual_means.items()},
        "relative_error": {
            "flora": _round(
                (actual_means["flora_refs"] - adjusted_flora_total / len(target_cells))
                / (adjusted_flora_total / len(target_cells)),
                6,
            )
            if adjusted_flora_total
            else None,
            "flora_vs_baseline": _round(
                (actual_means["flora_refs"] - target_flora_per_cell) / target_flora_per_cell, 6
            )
            if target_flora_per_cell
            else None,
            "tree": _round(
                (
                    actual_means["tree_refs"]
                    - flora_density_split["integer_targets"]["tree_refs"] / len(target_cells)
                )
                / (flora_density_split["integer_targets"]["tree_refs"] / len(target_cells)),
                6,
            )
            if flora_density_split["integer_targets"]["tree_refs"]
            else None,
            "undergrowth": _round(
                (
                    actual_means["undergrowth_refs"]
                    - flora_density_split["integer_targets"]["undergrowth_refs"] / len(target_cells)
                )
                / (flora_density_split["integer_targets"]["undergrowth_refs"] / len(target_cells)),
                6,
            )
            if flora_density_split["integer_targets"]["undergrowth_refs"]
            else None,
            "rocks": _round((actual_means["rock_refs"] - target_rocks_per_cell) / target_rocks_per_cell, 6) if target_rocks_per_cell else None,
        },
        "within_20_percent": abs(actual_means["flora_refs"] - adjusted_flora_total / len(target_cells))
        <= (adjusted_flora_total / len(target_cells)) * 0.2
        and abs(actual_means["rock_refs"] - target_rocks_per_cell) <= target_rocks_per_cell * 0.2,
    }

    audit_started = time.perf_counter()
    profiles = main_profiles + cliff_profiles
    rows_by_profile: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in placements:
        rows_by_profile[normalize_mesh_key(str(row["mesh"]))].append(row)
    measured_total_by_category = {
        "flora": sum(profile.frequency for profile in flora_profiles),
        "rocks": sum(profile.frequency for profile in normal_profiles) + sum(profile.frequency for profile in cliff_profiles),
        "cliff": sum(profile.frequency for profile in cliff_profiles),
    }
    generated_stats: dict[str, Any] = {}
    adherence_by_mesh: dict[str, Any] = {}
    for profile in sorted(profiles, key=lambda item: (item.category, item.key)):
        rows = rows_by_profile.get(profile.key, [])
        generated_count = len(rows)
        measured_category_total = measured_total_by_category["rocks"] if profile.category == "cliff" else measured_total_by_category[profile.category]
        if profile.category in {"rocks", "cliff"}:
            generated_category_total = sum(1 for row in placements if row["category"] in {"rocks", "cliff"})
        else:
            generated_category_total = sum(1 for row in placements if row["category"] == profile.category)
        stats = _profile_adherence(profile, rows)
        stats.update(
            {
                "measured_share": _round(profile.frequency / max(1, measured_category_total), 9),
                "generated_share": _round(generated_count / max(1, generated_category_total), 9),
                "share_error": _round(generated_count / max(1, generated_category_total) - profile.frequency / max(1, measured_category_total), 9),
                "quota_count": (cliff_quotas if profile.category == "cliff" else normal_quotas if profile.category == "rocks" else flora_quotas).get(profile.key, 0),
                "generated_count": generated_count,
                "bbox_volume_class": profile_bbox(profile).get("volume_class"),
                "shallow_water_species": profile.shallow_water,
                "open_face_profile": {
                    "status": profile.open_face_profile.get("status"),
                    "open_directions": list(profile.open_face_profile.get("open_directions", [])),
                },
            }
        )
        adherence_by_mesh[profile.key] = stats
        generated_stats[profile.key] = stats
    phase_done(
        "build_profile_adherence",
        audit_started,
        profiles=len(profiles),
        placements=len(placements),
    )
    post_audit_started = time.perf_counter()

    tree_by_clear = Counter()
    undergrowth_by_clear = Counter()
    for row in placements:
        if row["category"] != "flora":
            continue
        bucket = "clearing" if row["clearing"]["is_clearing"] else "non_clearing"
        if row["flora_role"] == "tree":
            tree_by_clear[bucket] += 1
        else:
            undergrowth_by_clear[bucket] += 1
    below_water = [row for row in placements if float(row["terrain"]["terrain_z_thu"]) <= 0.0]
    disallowed_below = [row for row in below_water if not row["water_rule"]["allowed_below_or_at_zero"]]
    cliffs = [row for row in placements if row["pass"] == "A"]
    cliff_slope_values = [float(row["terrain"]["slope_deg"]) for row in cliffs]
    cliff_embedding_values = [float(row["terrain"]["embedding_depth_gu"]) for row in cliffs]
    all_mesh_names = {str(row["mesh"]) for row in placements}
    rock_rows = [row for row in placements if row["category"] in {"rocks", "cliff"}]
    normal_rock_rows = [row for row in placements if row["category"] == "rocks"]
    rock_patch_counts = Counter(
        "patch" if float(row["rock_patch"]["mask_value"]) >= config.rock_patch_threshold else "gap"
        for row in normal_rock_rows
    )
    open_face_actions = Counter(
        str(row.get("open_face", {}).get("action", "not_recorded"))
        for row in rock_rows
    )
    open_face_orientation_sources = Counter(
        str(row.get("open_face", {}).get("orientation_source", "not_recorded"))
        for row in rock_rows
    )
    open_face_chosen_octants = Counter(
        str(row.get("open_face", {}).get("chosen_open_direction"))
        for row in rock_rows
        if row.get("open_face", {}).get("chosen_open_direction")
    )
    open_face_profile_count = sum(
        1
        for profile in main_profiles + cliff_profiles
        if profile.open_face_profile.get("status") == "ok"
    )
    open_face_mesh_count = sum(
        1
        for profile in main_profiles + cliff_profiles
        if profile.open_face_profile.get("open_directions")
    )
    open_face_no_safe_count = sum(
        value for key, value in failures.items() if str(key).endswith(":open_face_no_safe_orientation")
    )
    open_side_cliff_slope_rejects = sum(
        value for key, value in failures.items() if str(key).endswith(":open_side_cliff_slope")
    )
    flat_slope_max = float(rock_density_caps.get("flat_slope_max_deg", 8.0))
    low_slope_max = float(rock_density_caps.get("low_slope_max_deg", config.low_rock_slope_max_deg))
    flat_candidate_cells = {
        candidate.cell for candidate in all_candidates if float(candidate.slope_deg) < flat_slope_max
    }
    low_candidate_cells = {
        candidate.cell for candidate in all_candidates if float(candidate.slope_deg) < low_slope_max
    }
    flat_counts_by_cell = Counter(
        tuple(row["cell"])
        for row in rock_rows
        if float(row["terrain"]["slope_deg"]) < flat_slope_max
    )
    low_counts_by_cell = Counter(
        tuple(row["cell"])
        for row in rock_rows
        if float(row["terrain"]["slope_deg"]) < low_slope_max
    )
    flat_counts = [flat_counts_by_cell[cell] for cell in sorted(flat_candidate_cells, key=lambda value: (value[1], value[0]))]
    low_counts = [low_counts_by_cell[cell] for cell in sorted(low_candidate_cells, key=lambda value: (value[1], value[0]))]
    shore_threshold_gu = 4_096.0
    shore_candidates = [candidate for candidate in all_candidates if candidate.water_distance_gu <= shore_threshold_gu]
    shore_refs = [row for row in placements if float(row["terrain"]["distance_to_water_gu"]) <= shore_threshold_gu]
    shore_tree_refs = [row for row in shore_refs if row["category"] == "flora" and row["flora_role"] == "tree"]
    shore_undergrowth_refs = [row for row in shore_refs if row["category"] == "flora" and row["flora_role"] != "tree"]
    tree_profiles = [profile for profile in main_profiles if profile.category == "flora" and profile.flora_role == "tree"]
    shore_hits_started = time.perf_counter()
    # Cached tree-profile rows already exclude road centers; count shore strip hits.
    shore_tree_profile_candidates = sum(
        1
        for profile in tree_profiles
        for candidate, _weight, _mask in profile_row_cache.get(profile.key, ())
        if candidate.water_distance_gu <= shore_threshold_gu
    )
    phase_done(
        "audit.shore_tree_profile_hits",
        shore_hits_started,
        tree_profiles=len(tree_profiles),
        shore_candidates=len(shore_candidates),
        hits=shore_tree_profile_candidates,
    )
    shore_tree_slopes = [float(row["terrain"]["slope_deg"]) for row in shore_tree_refs]
    shore_tree_water = [float(row["terrain"]["distance_to_water_gu"]) for row in shore_tree_refs]
    shore_tree_analysis = {
        "strip_definition": "generated refs and direct-LAND candidates at distance_to_water_gu <= 4096 GU",
        "strip_distance_gu": shore_threshold_gu,
        "candidate_count": len(shore_candidates),
        "candidate_flora_tree_profile_hits": shore_tree_profile_candidates,
        "placed_refs": len(shore_refs),
        "placed_category_counts": dict(sorted(Counter(row["category"] for row in shore_refs).items())),
        "placed_flora_tree_count": len(shore_tree_refs),
        "placed_flora_undergrowth_count": len(shore_undergrowth_refs),
        "tree_fraction_of_shore_refs": _round(len(shore_tree_refs) / len(shore_refs), 6) if shore_refs else 0.0,
        "tree_slope_distribution_deg": _distribution_from_values(shore_tree_slopes),
        "tree_water_distance_distribution_gu": _distribution_from_values(shore_tree_water),
        "answer": "tree refs are available inside the measured shore strip; the gen2 missing-tree appearance was a render-selection artifact (cliffs were selected first and dominated the capped vignette), not a flora water/slope envelope rejection",
        "rule_artifact": True,
        "render_fix": "gen4 shore scene reserves deterministic tree and undergrowth slots before filling remaining near-water refs",
    }
    normal_rock_gap_candidates = [
        candidate for candidate in all_candidates if candidate.rock_patch_value < config.rock_patch_threshold
    ]
    road_candidates = [
        candidate for candidate in all_candidates if _candidate_has_road_raw_vtex(candidate, config)
    ]
    road_reject_started = time.perf_counter()
    # Equivalent to the previous nested profile x candidate road comprehension:
    # each profile rejects every road-centered candidate before weighting.
    road_profile_rejections_by_category = {
        category: len(road_candidates) * len(category_profiles)
        for category, category_profiles in (
            ("flora", flora_profiles),
            ("rocks", normal_profiles),
            ("cliff", cliff_profiles),
        )
    }
    phase_done(
        "audit.road_profile_rejections",
        road_reject_started,
        road_candidates=len(road_candidates),
    )
    road_placed_refs = [
        row for row in placements if row.get("terrain", {}).get("raw_vtex") in config.road_raw_vtex_values
    ]
    road_placed_by_category = Counter(str(row.get("category")) for row in road_placed_refs)
    road_exclusion_audit = {
        "raw_vtex_values": list(config.road_raw_vtex_values),
        "raw_vtex_semantics": "raw OpenMW VTEX values; positive raw N resolves to owning-plugin LTEX INTV N-1",
        "candidate_capture": candidate_summary.get("raw_vtex", {}),
        "candidate_count_centered_on_road_raw_vtex": len(road_candidates),
        "rejection_audit": {
            "filter_stage": "all placement profiles, before measured suitability weighting",
            "profile_candidate_rows_rejected_by_category": road_profile_rejections_by_category,
            "profile_candidate_rows_rejected_total": sum(road_profile_rejections_by_category.values()),
        },
        "placement_audit": {
            "accepted_refs": len(placements),
            "accepted_refs_by_category": dict(sorted(Counter(str(row.get("category")) for row in placements).items())),
            "placed_refs_centered_on_road_raw_vtex": len(road_placed_refs),
            "placed_refs_centered_on_road_raw_vtex_by_category": dict(sorted(road_placed_by_category.items())),
            "zero_road_placement_pass": not road_placed_refs,
            "accepted_refs_missing_raw_vtex": sum(
                1 for row in placements if row.get("terrain", {}).get("raw_vtex") is None
            ),
        },
        "rule": "every flora, normal-rock, and cliff candidate centered on a configured raw-VTEX road tile is rejected",
    }
    rock_cliff_rows = [
        row for row in placements if row.get("category") in {"rocks", "cliff"}
    ]
    quarantined_placement_counts = {
        path: sum(1 for row in placements if normalize_mesh_key(str(row.get("mesh", ""))) == normalize_mesh_key(path))
        for path in QUARANTINED_MESH_PATHS
    }
    quarantine_audit.update(
        {
            "accepted_placement_count_by_path": quarantined_placement_counts,
            "accepted_quarantined_placement_count": sum(quarantined_placement_counts.values()),
            "zero_quarantined_placements": sum(quarantined_placement_counts.values()) == 0,
            "quota_families_after_filter": {
                "normal_rock_profiles": len(normal_profiles),
                "cliff_profiles": len(cliff_profiles),
                "flora_profiles": len(flora_profiles),
            },
        }
    )
    tree_rows = [
        row for row in placements if row.get("category") == "flora" and row.get("flora_role") == "tree"
    ]
    tree_nonzero_x = [row for row in tree_rows if float(row["rotation_radians"][0]) != 0.0]
    tree_nonzero_y = [row for row in tree_rows if float(row["rotation_radians"][1]) != 0.0]
    tree_upright_audit = {
        "rule": "flora_role == tree preserves sampled measured Z yaw and forces rotation X=Y=0",
        "accepted_tree_count": len(tree_rows),
        "nonzero_x_count": len(tree_nonzero_x),
        "nonzero_y_count": len(tree_nonzero_y),
        "zero_xy_pass": not tree_nonzero_x and not tree_nonzero_y,
        "z_yaw_distribution": _distribution_from_values(
            float(row["rotation_radians"][2]) for row in tree_rows
        ),
        "rotation_source_counts": dict(
            sorted(Counter(str(row.get("rotation_source", "")) for row in tree_rows).items())
        ),
        "sampled_z_yaw_preserved": all(
            "tree_upright_rule" in str(row.get("rotation_source", "")) for row in tree_rows
        ),
    }
    tree_clearance_audit_started = time.perf_counter()
    if tree_rows:
        tree_points = np.asarray(
            [(float(row["position_gu"][0]), float(row["position_gu"][1])) for row in tree_rows],
            dtype=np.float64,
        )
        tree_violation_mask = _points_inside_any_aabb(tree_points, tree_clearance_aabbs)
        tree_clearance_violations = [
            row for index, row in enumerate(tree_rows) if bool(tree_violation_mask[index])
        ]
    else:
        tree_clearance_violations = []
    large_rock_cliff_rows = []
    for row in rock_cliff_rows:
        bbox = row.get("bbox")
        world_aabb = bbox.get("world_aabb_gu") if isinstance(bbox, Mapping) else None
        if not isinstance(world_aabb, Mapping):
            continue
        try:
            min_x, max_x, min_y, max_y = _aabb_xy_bounds(world_aabb)
        except (TypeError, ValueError, OverflowError):
            continue
        if max(max_x - min_x, max_y - min_y) >= config.tree_clearance_min_horizontal_span_gu:
            large_rock_cliff_rows.append(row)
    phase_done(
        "audit.tree_clearance",
        tree_clearance_audit_started,
        accepted_trees=len(tree_rows),
        violations=len(tree_clearance_violations),
        large_aabbs=len(large_rock_cliff_rows),
    )
    tree_profile_counts_started = time.perf_counter()
    tree_profile_candidate_rows_before_clearance = {
        profile.key: len(profile_row_cache.get(profile.key, ()))
        for profile in tree_profiles
    }
    tree_profile_candidate_rows_rejected_by_clearance = {
        profile.key: sum(
            1
            for candidate, _weight, _mask in profile_row_cache.get(profile.key, ())
            if candidate.candidate_id in tree_clearance_blocked_ids
        )
        for profile in tree_profiles
    }
    tree_clearance_rejection_count = sum(tree_profile_candidate_rows_rejected_by_clearance.values())
    phase_done(
        "audit.tree_profile_clearance_counts",
        tree_profile_counts_started,
        tree_profiles=len(tree_profiles),
        rejection_count=tree_clearance_rejection_count,
    )
    tree_clearance_audit = {
        "applies_to": ["flora_role == tree"],
        "rock_cliff_source": "accepted transformed rock/cliff world XY AABBs placed before flora",
        "minimum_horizontal_span_gu": config.tree_clearance_min_horizontal_span_gu,
        "margin_gu": config.tree_clearance_margin_gu,
        "large_rock_cliff_aabb_count": len(large_rock_cliff_rows),
        "candidate_count_checked": len(all_candidates),
        "candidate_count_inside_large_footprints": len(tree_clearance_blocked_ids),
        "tree_profile_candidate_rows_before_clearance": tree_profile_candidate_rows_before_clearance,
        "tree_profile_candidate_rows_rejected_by_clearance": tree_profile_candidate_rows_rejected_by_clearance,
        "rejection_count": tree_clearance_rejection_count,
        "accepted_tree_count": len(tree_rows),
        "accepted_tree_clearance_violation_count": len(tree_clearance_violations),
        "violating_tree_ref_ids": [str(row["ref_id"]) for row in tree_clearance_violations],
        "zero_accepted_tree_clearance_violations": not tree_clearance_violations,
        "rule": "reject a tree center inside any accepted rock/cliff transformed XY AABB with max(width,depth) >= 1024 GU, expanded by 128 GU on every side",
    }
    accepted_footprint_rows = [
        row
        for row in rock_cliff_rows
        if isinstance(row.get("road_footprint_audit"), Mapping)
    ]
    accepted_footprint_audits = [
        row["road_footprint_audit"]
        for row in accepted_footprint_rows
        if isinstance(row.get("road_footprint_audit"), Mapping)
    ]
    accepted_footprint_road_hits = sum(
        int(audit.get("road_hit_count", 0)) for audit in accepted_footprint_audits
    )
    accepted_footprint_missing_land = sum(
        int(audit.get("missing_land_cell_count", 0)) for audit in accepted_footprint_audits
    )
    accepted_footprint_missing_vtex = sum(
        int(audit.get("missing_vtex_cell_count", 0)) for audit in accepted_footprint_audits
    )
    accepted_footprint_road_overlap_count = sum(
        1 for audit in accepted_footprint_audits
        if any(value in config.road_raw_vtex_values for value in audit.get("road_hit_raw_vtex_values", []))
    ) if config.road_raw_vtex_values else None
    road_footprint_output_audit = {
        "applies_to": ["rocks", "cliff"],
        "terrain_source": "tamriel.esm LAND via procgen.espland",
        "tile_size_gu": LAND_TEXTURE_TILE_SIZE_GU,
        "tile_alignment": "global world-origin 512-GU grid",
        "aabb_intersection_rule": "half-open [min,max) in x/y; max boundary tile is excluded",
        "raw_vtex_values": list(config.road_raw_vtex_values),
        "checked_attempts": int(road_footprint_stats.get("checked_attempts", 0)),
        "passed_attempts": int(road_footprint_stats.get("passed_attempts", 0)),
        "rejected_attempts": int(road_footprint_stats.get("rejected_attempts", 0)),
        "checked_tiles": int(road_footprint_stats.get("checked_tiles", 0)),
        "resolved_tiles": int(road_footprint_stats.get("resolved_tiles", 0)),
        "road_hit_attempts": int(road_footprint_stats.get("road_hit_attempts", 0)),
        "road_hit_tiles": int(road_footprint_stats.get("road_hit_tiles", 0)),
        "missing_land_attempts": int(road_footprint_stats.get("missing_land_attempts", 0)),
        "missing_vtex_attempts": int(road_footprint_stats.get("missing_vtex_attempts", 0)),
        "missing_land_tiles": int(road_footprint_stats.get("missing_land_tiles", 0)),
        "missing_vtex_tiles": int(road_footprint_stats.get("missing_vtex_tiles", 0)),
        "rejected_attempts_by_reason": dict(sorted(road_footprint_rejections.items())),
        "accepted_rock_cliff_refs": len(rock_cliff_rows),
        "accepted_refs_with_passing_audit": sum(
            1 for audit in accepted_footprint_audits if bool(audit.get("passed"))
        ),
        "accepted_refs_with_road_hits": accepted_footprint_road_hits,
        "accepted_refs_missing_land": accepted_footprint_missing_land,
        "accepted_refs_missing_vtex": accepted_footprint_missing_vtex,
        "accepted_footprints_overlapping_configured_road_vtex": accepted_footprint_road_overlap_count,
        "zero_accepted_rock_cliff_footprints_overlapping_configured_road_vtex": (
            accepted_footprint_road_overlap_count == 0
            if accepted_footprint_road_overlap_count is not None
            else None
        ),
        "all_accepted_refs_have_passing_audit": len(accepted_footprint_audits) == len(rock_cliff_rows)
        and all(bool(audit.get("passed")) for audit in accepted_footprint_audits),
        "rule": "after transformed rotation/scale/world-AABB construction, every intersected global 512-GU LAND tile is checked; road raw VTEX or missing LAND/VTEX rejects the rock/cliff orientation",
    }
    rock_orientation_rows = [
        row for row in placements if row.get("category") in {"rocks", "cliff"}
    ]
    nonpositive_orientation_placements = [
        row
        for row in rock_orientation_rows
        if float(row.get("orientation_audit", {}).get("local_up_world_z", 0.0)) <= 0.0
    ]
    flipped_orientation_rejections = sum(
        int(value)
        for key, value in failures.items()
        if str(key).endswith(":flipped_orientation")
    )
    zero_flipped_placement_audit = {
        "applies_to": ["rocks", "cliff"],
        "euler_convention": "pipeline XYZ: X then Y then Z",
        "component": "cos(rotation_radians[0]) * cos(rotation_radians[1])",
        "rule": "reject sampled orientations when transformed local-up world Z is <= 0; retry within the existing bounded attempt loop",
        "rock_cliff_placement_count": len(rock_orientation_rows),
        "flipped_orientation_rejection_attempts": flipped_orientation_rejections,
        "placed_refs_with_nonpositive_local_up_world_z": len(nonpositive_orientation_placements),
        "minimum_accepted_local_up_world_z": min(
            (
                float(row["orientation_audit"]["local_up_world_z"])
                for row in rock_orientation_rows
                if isinstance(row.get("orientation_audit"), Mapping)
            ),
            default=None,
        ),
        "zero_flipped_placement_pass": not nonpositive_orientation_placements,
    }
    constrained_tilt_mesh_audit: dict[str, Any] = {}
    for mesh_key, max_tilt in sorted(config.max_local_up_tilt_degrees_by_mesh.items()):
        mesh_rows = [
            row for row in rock_orientation_rows if normalize_mesh_key(str(row.get("mesh", ""))) == mesh_key
        ]
        accepted_tilts = [
            float(row.get("orientation_audit", {}).get("local_up_tilt_degrees"))
            for row in mesh_rows
            if _finite(row.get("orientation_audit", {}).get("local_up_tilt_degrees")) is not None
        ]
        constrained_tilt_mesh_audit[mesh_key] = {
            "max_local_up_tilt_degrees": _round(max_tilt, 9),
            "accepted_count": len(mesh_rows),
            "max_accepted_local_up_tilt_degrees": _round(max(accepted_tilts), 9) if accepted_tilts else None,
            "rejected_orientation_attempts": int(constrained_tilt_rejections.get(mesh_key, 0)),
            "accepted_over_limit_count": sum(tilt > max_tilt for tilt in accepted_tilts),
            "pass": all(tilt <= max_tilt for tilt in accepted_tilts),
        }
    constrained_tilt_audit = {
        "rule": "constrained meshes reject sampled orientations whose transformed local-up tilt exceeds 15 degrees",
        "mesh_limits": dict(sorted(config.max_local_up_tilt_degrees_by_mesh.items())),
        "by_mesh": constrained_tilt_mesh_audit,
        "rejection_count": sum(constrained_tilt_rejections.values()),
        "accepted_over_limit_count": sum(
            int(row["accepted_over_limit_count"]) for row in constrained_tilt_mesh_audit.values()
        ),
        "zero_accepted_over_limit": all(
            bool(row["pass"]) for row in constrained_tilt_mesh_audit.values()
        ),
        "max_accepted_local_up_tilt_degrees": max(
            (
                float(row["max_accepted_local_up_tilt_degrees"])
                for row in constrained_tilt_mesh_audit.values()
                if row["max_accepted_local_up_tilt_degrees"] is not None
            ),
            default=None,
        ),
    }
    cliff_relief_rows = [
        row.get("cliff_footprint_relief")
        for row in cliffs
        if isinstance(row.get("cliff_footprint_relief"), Mapping)
    ]
    cliff_relief_margins = [
        float(row["relief_margin_gu"])
        for row in cliff_relief_rows
        if _finite(row.get("relief_margin_gu")) is not None
    ]
    cliff_observed_relief = [
        float(row["observed_relief_gu"])
        for row in cliff_relief_rows
        if _finite(row.get("observed_relief_gu")) is not None
    ]
    cliff_required_relief = [
        float(row["required_relief_gu"])
        for row in cliff_relief_rows
        if _finite(row.get("required_relief_gu")) is not None
    ]
    cliff_relief_rejections: Counter[str] = Counter()
    for key, value in failures.items():
        if ":cliff_footprint_" in str(key):
            reason = str(key).split(":")[-1].removeprefix("cliff_footprint_")
            cliff_relief_rejections[reason] += int(value)
    cliff_footprint_audit = {
        "sample_layout": "4 world-AABB corners + 4 edge midpoints",
        "sample_count_per_accepted_cliff": 8 if cliff_relief_rows else 0,
        "accepted_cliff_count": len(cliffs),
        "accepted_audits": len(cliff_relief_rows),
        "all_accepted_audits_pass": len(cliff_relief_rows) == len(cliffs)
        and all(bool(row.get("passed")) and len(row.get("samples", [])) == 8 for row in cliff_relief_rows),
        "observed_relief_gu": _distribution_from_values(cliff_observed_relief),
        "required_relief_gu": _distribution_from_values(cliff_required_relief),
        "relief_margin_gu": _distribution_from_values(cliff_relief_margins),
        "rejected_attempts_by_reason": dict(sorted(cliff_relief_rejections.items())),
        "threshold_rule": "observed max(sample terrain z) - min(sample terrain z) >= tan(cliff_min_slope_deg) * min(AABB X width, AABB Y depth)",
        "terrain_source": "tamriel.esm LAND via procgen.espland.height_at_game_position",
        "center_point_slope_prefilter_retained": True,
        "existing_surface_checks_retained": {
            "cliff_min_embed_gu": config.cliff_min_embed_gu,
            "cliff_min_visible_gu": config.cliff_min_visible_gu,
            "sampled_z_offset": True,
        },
    }
    baseline_actual = None
    if isinstance(baseline_document, Mapping):
        baseline_density = baseline_document.get("density", {})
        if isinstance(baseline_density, Mapping):
            value = baseline_density.get("actual")
            if isinstance(value, Mapping):
                baseline_actual = {
                    str(key): _round(float(number), 6)
                    for key, number in value.items()
                    if _finite(number) is not None
                }
    density_reduction = {
        "baseline": baseline_actual,
        "baseline_label": "scatter_falkreath_v2.json actual density" if baseline_actual else None,
        "after": {key: _round(value, 6) for key, value in actual_means.items()},
        "rock_mean_delta": _round(actual_means["rock_refs"] - float(baseline_actual["rock_refs"]), 6)
        if baseline_actual and _finite(baseline_actual.get("rock_refs")) is not None
        else None,
        "rock_mean_ratio_to_gen2": _round(actual_means["rock_refs"] / float(baseline_actual["rock_refs"]), 6)
        if baseline_actual and float(baseline_actual.get("rock_refs", 0.0))
        else None,
    }
    phase_done(
        "post_placement_audits",
        post_audit_started,
        placements=len(placements),
    )

    return {
        "schema_version": 6,
        "tool": "procgen.scatter_generate",
        "tool_version": "6.0",
        "seed": int(config.master_seed),
        "determinism": "derive_seed(master_seed, scatter-falkreath-v3, candidate/profile scope); sorted JSON; no timestamps",
        "scope": {
            "region": config.scope_region or "Falkreath near-water proving block",
            "anchor_cell": [-92, -10],
            "bounds_cells": [config.bounds[0], config.bounds[1], config.bounds[2], config.bounds[3]],
            "cell_count": len(target_cells),
        },
        "units": {
            "position": "TES3 game units",
            "terrain": "THU",
            "game_units_per_thu": THU_TO_GU,
            "rotation": "radians",
            "scale": "uniform measured NIF XSCL distribution",
        },
        "terrain": {
            "source": terrain_source,
            "edited_land_source": edited_land_source,
            "reader": "procgen.espland.load_land + procgen.tes3json.land_records_from_json override",
            "composite_heightmap_used": False,
            "water_threshold_thu": 0.0,
            "water_definition": "direct tamriel.esm LAND terrain <= 0 THU",
        },
        "inputs": {
            "scatter_analysis": "output/vorndgad_scatter_analysis.json",
            "cliff_analysis": "output/vorndgad_cliff_analysis.json",
            "bbox_cache": "output/mesh_bbox_cache.json",
            "open_face_profiles": open_face_source,
            "analysis_policy": "per-mesh frequency, per-mesh terrain envelopes, per-mesh z-offset, scale, and rotation/slope data",
        },
        "sampling_design": {
            "frequency": "largest-remainder per-mesh quota from measured count shares; rare meshes are not uniformized",
            "suitability": "per-mesh measured slope/elevation/water envelope membership; profile frequency is retained in candidate score",
            "z_offset": "sampled per mesh from measured [min,p10,p50,p90,max] quantiles; no universal offset",
            "rotation": "sample measured per-mesh z/full mode and X/Y/Z distributions, preferring measured slope-conditioned bin; every flora tree preserves sampled Z yaw and forces X/Y to zero",
            "scale": "sample measured per-mesh quantile distribution without category-wide scale rule",
            "water": "only measured shallow-water species can use direct-LAND terrain <= 0 THU",
            "small_rocks": "require transformed local bbox to intersect terrain with measured z-offset sample and at least configured embedding depth; no perched small rocks",
            "cliff_giants": "measured giant share of rock quota, per-mesh steep-slope/envelope gates, direct terrain intersection, minimum embedding/visibility, face-affinity preference",
            "road_exclusion": "capture OpenMW-normalized raw VTEX at every candidate center and reject configured road values before profile weighting for flora, rocks, and cliffs",
            "road_footprint_exclusion": "after transformed rock/cliff world-AABB construction, enumerate globally aligned 512-GU LAND tiles with floor-safe CELL/local-tile conversion and half-open maximum edges; reject configured road VTEX and missing LAND/VTEX data fail-closed",
            "upright_rock_gate": "for rocks and cliffs, reject sampled rotations with transformed local-up world Z <= 0 and retry within the existing bounded orientation loop",
            "quarantine": "remove explicit L_04 cliff shells and four cluster-rock mesh paths before eligibility and quota allocation; L_04 stay banned until pitch/roll or terrain-shape burial seats open bottoms",
            "constrained_tilt": "Sky_TerrRock_LV_04_21.nif and Sky_TerrRock_04_027.nif require transformed local-up tilt <= 15 degrees",
            "tree_clearance": "after rocks/cliffs are accepted, reject tree centers inside large transformed rock/cliff XY AABBs expanded by 128 GU",
            "flora_density_split": "measured tree-share target is unchanged; measured undergrowth-share target is multiplied by 1.25 before deterministic integer per-mesh allocation",
            "cliff_footprint_relief": "after rotation, scale, and world-AABB construction, sample direct LAND at four corners and four edge midpoints; require relief >= tan(cliff_min_slope_deg) * min(AABB X width, Y depth), while retaining center slope, z-offset, embedding, and visibility gates",
            "stackers": "OFF; no terrain-surface support or parent refs are generated",
            "clearing_mask": "smooth deterministic 4096-GU node interpolation; clearing threshold downweights tree candidates to 0.20, undergrowth candidates remain at 1.0",
            "rock_density": "target total rock density is 70% of the measured wilderness mean; normal rocks also use wider spacing, a smooth 3072-GU patch-gap mask, and reduced flat/low-slope per-cell caps",
            "rock_patch_mask": "smooth deterministic 3072-GU mask; normal rocks require mask >= 0.34, cliffs are exempt so landmarks remain available",
            "open_faces": "Blender-measured direction-octant sidecar; profiles with open_directions solve yaw toward ESM-LAND downslope or an adjacent rock (pose-first, O(open octants)); closed profiles keep one measured sample; side-open cliffs also require open_side_cliff_min_slope_deg; target-facing placements require deeper configured embedding",
        },
        "water_rules": {
            "threshold_thu": 0.0,
            "below_or_at_zero_refs": len(below_water),
            "disallowed_below_or_at_zero_refs": len(disallowed_below),
            "all_below_water_refs_are_documented": not disallowed_below,
        },
        "spacing": {
            "occupancy_bin_gu": OCCUPANCY_BIN_GU,
            "candidate_spacing_gu": config.candidate_spacing_gu,
            "jitter_gu": config.jitter_gu,
            "minimum_distances_gu": dict(config.min_distances_gu),
            "cliff_and_normal_rocks_share_occupancy": True,
        },
        "water_index": water_summary,
        "candidate_summary": candidate_summary,
        "quarantine_audit": quarantine_audit,
        "road_exclusion": road_exclusion_audit,
        "road_footprint_audit": road_footprint_output_audit,
        "zero_flipped_placement_audit": zero_flipped_placement_audit,
        "constrained_tilt_audit": constrained_tilt_audit,
        "tree_upright_audit": tree_upright_audit,
        "tree_clearance_audit": tree_clearance_audit,
        "cliff_footprint_relief_audit": cliff_footprint_audit,
        "clearing": {
            "grid_gu": config.clearing_grid_gu,
            "threshold": config.clearing_threshold,
            "tree_factor": config.clearing_tree_factor,
            "tree_counts": dict(sorted(tree_by_clear.items())),
            "undergrowth_counts": dict(sorted(undergrowth_by_clear.items())),
            "tree_reduction_observable": tree_by_clear.get("clearing", 0) <= tree_by_clear.get("non_clearing", 0),
        },
        "city_clearing": {
            "enabled": clearing_index is not None,
            "frame_origin_gu": (
                [list(o) for o in clearing_index.frame_origins_gu] if hasattr(clearing_index, "frame_origins_gu")
                else list(clearing_index.frame_origin_gu)
            ) if clearing_index is not None else None,
            "flora_clearing_blocked": int(failures.get("clearing_blocked", 0)),
            "rock_clearing_blocked": int(failures.get("rock_clearing_blocked", 0)),
            "city_domain_rocks_banned": int(failures.get("city_domain_rocks_banned", 0)),
            "accepted_rock_cliff_in_city": (
                sum(
                    1
                    for row in placements
                    if row.get("category") in {"rocks", "cliff"}
                    and clearing_index is not None
                    and clearing_index.in_city_domain_point(
                        float(row["position_gu"][0]), float(row["position_gu"][1])
                    )
                )
                if clearing_index is not None
                else None
            ),
            "rule": (
                "flora: reject candidate inside building/surface/road via blocks_point; "
                "rocks/cliff: reject anchor inside city_domain (city_domain_rocks_banned) "
                "and reject transformed footprint AABB intersecting building/surface/road "
                "(rock_clearing_blocked); terrain source is the edited-LAND plugin"
            ),
        },
        "rock_density_cap": {
            **dict(rock_density_caps),
            "final_rock_counts_by_cell": {
                f"{cell[0]},{cell[1]}": count
                for cell, count in sorted(rock_counts_by_cell.items(), key=lambda item: (item[0][1], item[0][0]))
            },
            "flat_cell_max_count": max(
                (
                    count for count in flat_counts
                ),
                default=0,
            ),
            "flat_land": {
                "candidate_cell_count": len(flat_candidate_cells),
                "placed_rock_counts_by_cell": {
                    f"{cell[0]},{cell[1]}": flat_counts_by_cell[cell]
                    for cell in sorted(flat_candidate_cells, key=lambda value: (value[1], value[0]))
                },
                "placed_mean_refs_per_flat_candidate_cell": _round(sum(flat_counts) / len(flat_counts), 6) if flat_counts else 0.0,
                "placed_max_refs_per_flat_candidate_cell": max(flat_counts, default=0),
                "cap_pass": max(flat_counts, default=0) <= int(rock_density_caps.get("flat_cap_refs_per_cell") or 0) if flat_counts else True,
                "rule": "all rock refs whose ESM-LAND slope is below 8 degrees share the reduced flat cap; this deliberately allows a mean below measured wilderness density on flats",
            },
            "low_slope": {
                "candidate_cell_count": len(low_candidate_cells),
                "placed_mean_refs_per_low_candidate_cell": _round(sum(low_counts) / len(low_counts), 6) if low_counts else 0.0,
                "placed_max_refs_per_low_candidate_cell": max(low_counts, default=0),
            },
            "patch_mask": {
                "grid_gu": config.rock_patch_grid_gu,
                "threshold": config.rock_patch_threshold,
                "placed_refs_by_bucket": dict(sorted(rock_patch_counts.items())),
                "normal_rock_gap_refs": sum(1 for row in normal_rock_rows if float(row["rock_patch"]["mask_value"]) < config.rock_patch_threshold),
                "normal_rock_gap_candidate_count": len(normal_rock_gap_candidates),
                "normal_rock_gap_candidate_cell_count": len({candidate.cell for candidate in normal_rock_gap_candidates}),
                "cliff_refs_ignore_patch_mask": True,
            },
        },
        "rock_density_reduction": density_reduction,
        "shore_tree_analysis": shore_tree_analysis,
            "open_face_usage": {
            "profile_source": open_face_source,
            "profile_count": open_face_profile_count,
            "meshes_with_open_geometry": open_face_mesh_count,
            "placed_rock_refs": len(rock_rows),
            "placed_refs_with_profile": sum(1 for row in rock_rows if row.get("open_face", {}).get("profile_status") == "ok"),
            "actions": dict(sorted(open_face_actions.items())),
            "orientation_sources": dict(sorted(open_face_orientation_sources.items())),
            "chosen_open_directions": dict(sorted(open_face_chosen_octants.items())),
            "pose_first_placed_count": open_face_orientation_sources.get("pose_first", 0),
            "safe_target_actions": {
                "esm_land_downslope": open_face_actions.get("terrain_downslope", 0),
                "adjacent_accepted_rock": open_face_actions.get("adjacent_rock", 0),
            },
            "no_safe_orientation_count": open_face_no_safe_count,
            "open_side_cliff_slope_rejects": open_side_cliff_slope_rejects,
            "open_side_cliff_min_slope_deg": config.open_side_cliff_min_slope_deg,
            "min_embed_gu_for_target_facing_refs": config.open_face_min_embed_gu,
            "rule": "open side uses pose-first yaw toward ESM-LAND downslope or an adjacent accepted rock; closed profiles keep one measured sample; side-open cliffs also require open_side_cliff_min_slope_deg",
        },
        "rock_profiles": {
            "profile_count": len(main_profiles) + len(cliff_profiles),
            "meshes_with_open_directions": sorted(
                profile.mesh for profile in main_profiles + cliff_profiles if profile.open_face_profile.get("open_directions")
            ),
        },
        "density": {
            "target": {
                "flora_refs": adjusted_flora_total / len(target_cells),
                "flora_baseline_refs": target_flora_per_cell,
                "flora_tree_refs": flora_density_split["integer_targets"]["tree_refs"] / len(target_cells),
                "flora_undergrowth_refs": flora_density_split["integer_targets"]["undergrowth_refs"] / len(target_cells),
                "rock_refs": target_rocks_per_cell,
                "flora_total": adjusted_flora_total,
                "flora_baseline_total": baseline_flora_total,
                "flora_tree_total": flora_density_split["integer_targets"]["tree_refs"],
                "flora_undergrowth_total": flora_density_split["integer_targets"]["undergrowth_refs"],
                "rock_total": target_rocks_total,
            },
            "actual": actual_means,
            "comparison": density_comparison,
            "flora_split": {
                **flora_density_split,
                "actuals": {
                    "tree_refs": sum(1 for row in placements if row["category"] == "flora" and row["flora_role"] == "tree"),
                    "undergrowth_refs": sum(1 for row in placements if row["category"] == "flora" and row["flora_role"] != "tree"),
                    "flora_refs": sum(1 for row in placements if row["category"] == "flora"),
                },
            },
            "reference_group": "Vorndgad wilderness cells only (55 cells; named settlements excluded)",
            "cells": cells,
        },
        "pass_a_cliff": {
            "description": "measured giant quota and per-mesh steep/envelope/geometry sampling; cliff face affinity and open-face orientation link terrain-compatible giants",
            "target_refs": target_cliffs,
            "placed_refs": cliff_count,
            "mesh_counts": dict(sorted(Counter(row["mesh"] for row in cliffs).items(), key=lambda item: (item[0].casefold(), item[0]))),
            "measured_giant_ref_total": measured_cliff_total,
            "measured_rock_ref_total": measured_rock_total,
            "slope_min_deg": config.cliff_min_slope_deg,
            "open_side_cliff_min_slope_deg": config.open_side_cliff_min_slope_deg,
            "water_rule": "per-mesh measured water envelope; no global water-distance cutoff",
            "embedding_depth_gu": _distribution_from_values(cliff_embedding_values),
            "slope_distribution_deg": _distribution_from_values(cliff_slope_values),
            "face_link_count": sum(1 for row in cliffs if row.get("cliff_face_link")),
            "footprint_relief_audit": cliff_footprint_audit,
        },
        "cliff_seating_audit": cliff_seating_audit,
        "pass_b_scatter": {
            "description": "frequency-quota flora and non-giant rocks using each mesh's measured suitability and transform distributions",
            "placed_refs": sum(1 for row in placements if row["pass"] == "B"),
            "category_counts": dict(sorted(Counter(row["category"] for row in placements if row["pass"] == "B").items())),
            "stackers": 0,
            "stacker_rule": "disabled by accepted plan",
        },
        "adherence": {
            "per_mesh": adherence_by_mesh,
            "tables": {
                "frequency": [
                    {
                        "mesh": key,
                        "category": value["category"],
                        "measured_share": value["measured_share"],
                        "generated_share": value["generated_share"],
                        "share_error": value["share_error"],
                        "measured_frequency": value["measured_frequency"],
                        "placed_count": value["placed_count"],
                    }
                    for key, value in adherence_by_mesh.items()
                ],
                "z_offset": [
                    {"mesh": key, "measured": value["z_offset"]["measured"], "placed": value["z_offset"]["placed"], "within_fraction": value["z_offset"]["within_measured_min_max_fraction"]}
                    for key, value in adherence_by_mesh.items()
                ],
                "slope": [
                    {"mesh": key, "measured": value["slope"]["measured"], "placed": value["slope"]["placed"], "within_fraction": value["slope"]["within_measured_min_max_fraction"]}
                    for key, value in adherence_by_mesh.items()
                ],
                "orientation": [
                    {"mesh": key, "measured": value["orientation"]["measured"], "placed": value["orientation"]["placed"], "forbidden_xy_tilt_count": value["orientation"]["forbidden_xy_tilt_count"]}
                    for key, value in adherence_by_mesh.items()
                ],
            },
        },
        "stacking": {
            "enabled": False,
            "placement_count": 0,
            "terrain_surfaces": [],
            "reason": "stackers are OFF until mesh-geometry placement exists",
        },
        "generation_failures": dict(sorted(failures.items())),
        "species_stats": generated_stats,
        "placement_stats": {
            "total_refs": len(placements),
            "target_flora_refs": adjusted_flora_total,
            "target_flora_baseline_refs": baseline_flora_total,
            "target_tree_refs": flora_density_split["integer_targets"]["tree_refs"],
            "target_undergrowth_refs": flora_density_split["integer_targets"]["undergrowth_refs"],
            "target_rock_refs": target_rocks_total,
            "target_cliff_refs": target_cliffs,
            "by_pass": {"A": sum(1 for row in placements if row["pass"] == "A"), "B": sum(1 for row in placements if row["pass"] == "B")},
            "by_category": dict(sorted(Counter(row["category"] for row in placements).items())),
            "stackers": 0,
            "unique_meshes": len(all_mesh_names),
            "giant_palette_meshes": len(giant_keys),
            "zero_flipped_placement_count": len(nonpositive_orientation_placements),
            "tree_upright_violation_count": tree_upright_audit["nonzero_x_count"] + tree_upright_audit["nonzero_y_count"],
            "tree_clearance_rejection_count": tree_clearance_audit["rejection_count"],
            "tree_clearance_violation_count": tree_clearance_audit["accepted_tree_clearance_violation_count"],
            "constrained_tilt_rejection_count": constrained_tilt_audit["rejection_count"],
            "road_raw_vtex_placement_count": len(road_placed_refs),
            "road_footprint_checked_attempt_count": int(road_footprint_stats.get("checked_attempts", 0)),
            "road_footprint_rejected_attempt_count": int(road_footprint_stats.get("rejected_attempts", 0)),
            "road_footprint_road_hit_attempt_count": int(road_footprint_stats.get("road_hit_attempts", 0)),
        },
    }


__all__ = [
    "DEFAULT_CANDIDATE_SPACING_GU",
    "DEFAULT_CLEARING_GRID_GU",
    "DEFAULT_CLEARING_THRESHOLD",
    "DEFAULT_JITTER_GU",
    "DEFAULT_MIN_DISTANCES_GU",
    "GENERATION_NAMESPACE",
    "GenerationConfig",
    "LAND_TEXTURE_TILE_SIZE_GU",
    "MAX_LOCAL_UP_TILT_DEGREES_BY_MESH",
    "OCCUPANCY_BIN_GU",
    "OccupancyIndex",
    "QUARANTINED_MESH_KEYS",
    "QUARANTINED_MESH_PATHS",
    "TARGET_BOUNDS",
    "TREE_CLEARANCE_MARGIN_GU",
    "TREE_CLEARANCE_MIN_HORIZONTAL_SPAN_GU",
    "WEIGHT_FUNCTION_DESCRIPTION",
    "clearing_mask_value",
    "cliff_footprint_relief_audit",
    "enumerate_land_texture_tiles",
    "envelope_weight",
    "envelope_weight_array",
    "generate_scatter_document",
    "large_rock_tree_clearance_violation",
    "road_footprint_audit",
    "transformed_local_up_tilt_degrees",
    "transformed_local_up_world_z",
    "tree_clearance_blocked_candidate_ids",
    "_pose_first_open_face_yaw",
    "_rotate_xy_direction",
]

