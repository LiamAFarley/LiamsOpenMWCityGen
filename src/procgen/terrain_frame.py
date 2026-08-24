"""terrain_frame.py — shared terrain-relative orthonormal frames for cliffs.

Purpose
-------
One deterministic construction of a terrain-relative orthonormal frame from
the existing center-plus-eight-neighbor direct-LAND stencil, plus the exact
pose-transfer convention used by cliff seating:

* offline: ``R_relative = T_source.T @ M_source`` for every extracted giant;
* runtime: ``M_candidate = T_candidate @ R_relative``.

Pipeline position: consumed by the cliff-seating profile builder
(``tools/scatter/build_cliff_seating_profiles.py``) and the runtime evaluator
(``src/procgen/cliff_seating.py`` via ``scatter_generate``).  It never reads
files and never calls Blender.

Convention (binding, from
``.opencode/runs/2026-08-24-r18-cliff-seating-analysis/
2026-08-24_r18_cliff_seating_full_implementation_plan.md``)
--------------------------------------------------------------------
1. Sample the center and eight neighbors at the configured spacing.
2. The maximum absolute slope is retained separately for candidate
   eligibility; the FRAME uses the neighbor with the greatest positive
   ``drop/run`` scanned in the same stable direction order as candidate
   construction (``scatter_generate._SLOPE_NEIGHBOR_DIRS``).  A real lower
   neighbor is required.
3. With ``d=(dx,dy)`` the normalized world-XY downhill and ``a`` its positive
   downhill angle ``atan2(drop, run)``::

       x = (cos(a)*dx, cos(a)*dy, -sin(a))   downhill tangent
       z = (sin(a)*dx, sin(a)*dy,  cos(a))   up-facing terrain normal
       y = z × x                             cross-slope tangent

4. Columns are normalized and verified right-handed orthonormal within the
   configured residual tolerance (analytically exact up to float rounding).
5. Source observation matrix: ``M_source = tes3_euler_to_matrix(rotation)``.
6. Terrain-relative pose: ``R_relative = T_source.T @ M_source``.
7. Runtime transfer: ``M_candidate = T_candidate @ R_relative``.
8. Euler conversion goes only through
   ``engine_transform.matrix_to_tes3_euler`` and the resulting TES3 matrix
   must reproduce the candidate matrix within tolerance.

Invariants
----------
* Column vectors; matrices are row-major NumPy float64 acting on column
  vectors; all heights come from ``espland.height_at_game_position`` (THU,
  converted to GU at one boundary).
* Fail-closed: a missing LAND sample at any stencil point, or the absence of
  any real lower neighbor, raises :class:`TerrainFrameError`.
* Deterministic: pure function of the land records and the position.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .engine_transform import (
    matrix_to_tes3_euler,
    tes3_euler_to_matrix,
)
from .espland import LandRecord, THU_TO_GU, height_at_game_position


# Must remain identical to scatter_generate._SLOPE_NEIGHBOR_DIRS so frame
# downhill selection cannot diverge from candidate downhill selection.
_SLOPE_NEIGHBOR_DIRS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)


class TerrainFrameError(ValueError):
    """A terrain frame could not be constructed from the supplied LAND."""


@dataclass(frozen=True)
class TerrainFrame:
    """One terrain-relative orthonormal frame plus its stencil evidence."""

    matrix: np.ndarray  # columns [x downhill, y cross-slope, z up-normal]
    downhill_xy: tuple[float, float]
    downhill_angle_rad: float
    maximum_slope_deg: float | None
    center_z_gu: float

    @property
    def upslope_xy(self) -> tuple[float, float]:
        return (-self.downhill_xy[0], -self.downhill_xy[1])


def build_terrain_frame(
    land_records: Mapping[tuple[int, int], LandRecord],
    position: Sequence[float],
    *,
    sample_spacing_gu: float,
    matrix_residual_tolerance: float = 1e-8,
) -> TerrainFrame:
    """Construct the terrain frame at one world position (game units).

    Raises :class:`TerrainFrameError` when the center or any sampled height is
    unavailable or when no neighbor is strictly lower than the center.
    """

    spacing = float(sample_spacing_gu)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("sample_spacing_gu must be finite and positive")
    tolerance = float(matrix_residual_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("matrix_residual_tolerance must be finite and non-negative")

    center_thu = height_at_game_position(land_records, position[:2])
    if center_thu is None:
        raise TerrainFrameError(f"no LAND height at frame center {tuple(position[:2])}")
    center_gu = float(center_thu) * THU_TO_GU

    maximum_slope: float | None = None
    # (drop/length rank, dx_unit, dy_unit, horizontal run GU, drop GU)
    best: tuple[float, float, float, float, float] | None = None
    for dx, dy in _SLOPE_NEIGHBOR_DIRS:
        neighbor = (
            float(position[0]) + dx * spacing,
            float(position[1]) + dy * spacing,
        )
        neighbor_thu = height_at_game_position(land_records, neighbor)
        if neighbor_thu is None:
            raise TerrainFrameError(
                f"no LAND height at frame stencil neighbor "
                f"({neighbor[0]:.3f}, {neighbor[1]:.3f})"
            )
        neighbor_gu = float(neighbor_thu) * THU_TO_GU
        rise_gu = abs(neighbor_gu - center_gu)
        run_gu = math.hypot(dx * spacing, dy * spacing)
        slope = math.degrees(math.atan2(rise_gu, run_gu))
        maximum_slope = slope if maximum_slope is None else max(maximum_slope, slope)
        drop = center_gu - neighbor_gu
        if drop <= 0.0:
            continue
        length = math.hypot(dx, dy)
        ranked = drop / length
        # Strict comparison keeps the first direction on ties, matching the
        # candidate-construction scan order exactly.
        if best is None or ranked > best[0]:
            best = (ranked, float(dx) / length, float(dy) / length, run_gu, drop)
    if best is None:
        raise TerrainFrameError(
            f"no real lower neighbor in the terrain stencil at {tuple(position[:2])}"
        )

    _ranked, dx_unit, dy_unit, run_gu, drop_gu = best
    angle = math.atan2(drop_gu, run_gu)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x_col = np.array([cos_a * dx_unit, cos_a * dy_unit, -sin_a], dtype=np.float64)
    z_col = np.array([sin_a * dx_unit, sin_a * dy_unit, cos_a], dtype=np.float64)
    y_col = np.cross(z_col, x_col)
    frame = np.column_stack((x_col, y_col, z_col))

    identity = np.eye(3, dtype=np.float64)
    if not np.all(np.isfinite(frame)):
        raise TerrainFrameError("terrain frame contains non-finite values")
    residual = float(np.max(np.abs(frame.T @ frame - identity)))
    if residual > tolerance:
        raise TerrainFrameError(
            f"terrain frame is not orthonormal within {tolerance:g} (residual {residual:.3g})"
        )
    determinant = float(np.linalg.det(frame))
    if abs(determinant - 1.0) > tolerance:
        raise TerrainFrameError(
            f"terrain frame is not right-handed within {tolerance:g} (det {determinant:.9f})"
        )
    return TerrainFrame(
        matrix=frame,
        downhill_xy=(round(dx_unit, 6), round(dy_unit, 6)),
        downhill_angle_rad=angle,
        maximum_slope_deg=None if maximum_slope is None else round(maximum_slope, 6),
        center_z_gu=center_gu,
    )


def relative_pose_matrix(
    terrain_frame_matrix: np.ndarray, source_rotation_radians: Sequence[float]
) -> np.ndarray:
    """Return ``R_relative = T_source.T @ M_source`` for one observation."""

    source = tes3_euler_to_matrix(source_rotation_radians)
    basis = np.asarray(terrain_frame_matrix, dtype=np.float64)
    if basis.shape != (3, 3):
        raise ValueError("terrain frame matrix must be 3x3")
    return basis.T @ source


def transfer_pose_matrix(
    terrain_frame_matrix: np.ndarray, relative_matrix: np.ndarray
) -> np.ndarray:
    """Return ``M_candidate = T_candidate @ R_relative``."""

    basis = np.asarray(terrain_frame_matrix, dtype=np.float64)
    relative = np.asarray(relative_matrix, dtype=np.float64)
    if basis.shape != (3, 3) or relative.shape != (3, 3):
        raise ValueError("terrain frame and relative matrices must be 3x3")
    return basis @ relative


def euler_round_trip(
    matrix: np.ndarray,
) -> tuple[tuple[float, float, float], float]:
    """Decompose to TES3 Euler and report the recomposition residual.

    Returns ``(euler_radians, max_abs_element_residual)`` where the residual
    compares ``tes3_euler_to_matrix(euler)`` against the input matrix.
    """

    candidate = np.asarray(matrix, dtype=np.float64)
    if candidate.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    euler = matrix_to_tes3_euler(candidate)
    recomposed = tes3_euler_to_matrix(euler)
    residual = float(np.max(np.abs(recomposed - candidate)))
    return euler, residual


def rotation_geodesic_distance_radians(
    first: np.ndarray, second: np.ndarray
) -> float:
    """Geodesic SO(3) distance ``acos((trace(A.T B) - 1)/2)``, clamped."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    trace = float(np.trace(a.T @ b))
    value = (trace - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, value)))


def slope_bin_index(slope_deg: float, bin_edges_deg: Sequence[float]) -> int:
    """Return the index of ``slope_deg`` among ascending bin edges.

    Edges define consecutive half-open bins ``[edge[i], edge[i+1])`` with the
    final bin unbounded above; values below the first edge fall into bin 0.
    """

    edges = [float(value) for value in bin_edges_deg]
    if len(edges) < 2 or any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError("slope_bin_edges_deg must be strictly ascending")
    value = float(slope_deg)
    for index in range(len(edges) - 1):
        if value < edges[index + 1]:
            return index
    return len(edges) - 1


__all__ = [
    "TerrainFrame",
    "TerrainFrameError",
    "build_terrain_frame",
    "euler_round_trip",
    "relative_pose_matrix",
    "rotation_geodesic_distance_radians",
    "slope_bin_index",
    "transfer_pose_matrix",
]
