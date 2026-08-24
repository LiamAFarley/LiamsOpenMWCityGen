"""Frame conversion helpers for Phase 1 building evidence.

All rotation and decomposition work delegates to ``engine_transform``.  The
helpers operate in GU and preserve authored scale as a separate value; no
scene-unit placement matrix is used here.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..engine_transform import matrix_to_tes3_euler, tes3_euler_to_matrix


def _vector(value: Sequence[float], field: str) -> np.ndarray:
    if len(value) != 3:
        raise ValueError(f"{field} must contain three values")
    result = np.asarray([float(item) for item in value], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field} must be finite")
    return result


def to_template_local(
    offset_gu: Sequence[float],
    rotation_rad: Sequence[float],
    p0: Sequence[float],
    R0: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    """Convert one source-world member into the seed-door template frame."""

    position = _vector(offset_gu, "offset_gu")
    origin = _vector(p0, "p0")
    frame = np.asarray(R0, dtype=np.float64)
    if frame.shape != (3, 3) or not np.all(np.isfinite(frame)):
        raise ValueError("R0 must be a finite 3x3 matrix")
    local_offset = frame.T @ (position - origin)
    local_rotation = frame.T @ tes3_euler_to_matrix(rotation_rad)
    return local_offset.tolist(), list(matrix_to_tes3_euler(local_rotation))


def to_source_world(
    offset_local_gu: Sequence[float],
    rotation_local_rad: Sequence[float],
    p0: Sequence[float],
    R0: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    """Reconstruct source-world position and TES3 Euler rotation."""

    local_offset = _vector(offset_local_gu, "offset_local_gu")
    origin = _vector(p0, "p0")
    frame = np.asarray(R0, dtype=np.float64)
    if frame.shape != (3, 3) or not np.all(np.isfinite(frame)):
        raise ValueError("R0 must be a finite 3x3 matrix")
    position = origin + frame @ local_offset
    rotation = matrix_to_tes3_euler(frame @ tes3_euler_to_matrix(rotation_local_rad))
    return position.tolist(), list(rotation)


def rebase_connection(
    position_a_gu: Sequence[float],
    rotation_a_rad: Sequence[float],
    position_b_gu: Sequence[float],
    rotation_b_rad: Sequence[float],
) -> dict[str, Any]:
    """Build an ordered A-local relation from two source-world transforms."""

    p_a = _vector(position_a_gu, "position_a_gu")
    p_b = _vector(position_b_gu, "position_b_gu")
    R_a = tes3_euler_to_matrix(rotation_a_rad)
    R_b = tes3_euler_to_matrix(rotation_b_rad)
    return {
        "offset_b_in_a_frame_gu": (R_a.T @ (p_b - p_a)).tolist(),
        "relative_engine_matrix_3x3": (R_a.T @ R_b).tolist(),
    }


def matrix_max_error(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    """Return the maximum absolute element residual for a rotation matrix."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"matrix shapes differ: {a.shape} != {b.shape}")
    return float(np.max(np.abs(a - b)))


def vector_max_error(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the maximum absolute element residual for a GU vector."""

    a = _vector(left, "left")
    b = _vector(right, "right")
    return float(np.max(np.abs(a - b)))


def canonicalize(value: Any) -> Any:
    """Round emitted JSON floats to six places and normalize negative zero."""

    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        result = round(value, 6)
        return 0.0 if result == 0 else result
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [canonicalize(item) for item in value]
    if isinstance(value, Mapping):
        return {key: canonicalize(item) for key, item in value.items()}
    return value
