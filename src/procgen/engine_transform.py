"""engine_transform.py — ONE central OpenMW/TES3 rotation helper (cityforge T0.1).

Purpose
-------
Consolidates the three independent host-side rotation implementations that
historically duplicated the same engine convention into a single NumPy-based
module with one documented contract:

* ``tools/karthgad_rebuild_geometry.py`` — ``rotation_xyz_matrix(_raw)``,
  ``blender_xyz_euler_from_matrix``, ``blender_xyz_euler_for_tes3_rotation``,
  ``placement_scene_matrix`` (the de-facto settlement/B1 authority).
* ``src/procgen/scatter_analysis.py`` — ``rotate_tes3_reference_point``.
* ``src/procgen/groundcover_generate.py`` — ``_engine_tilt`` / ``_engine_up``.

This module does NOT rewire any existing call site; the old helpers stay
authoritative for their current consumers.  It exists so new cityforge
consumers (``cityplace.py``, ``cityscape.py``, ``render_city.py``,
``cityauthor.py`` in later phases) import ONE tested API instead of a third
parallel implementation (master plan §3.3, §4.1 T0.1).

Binding convention (validated 2026-08-04, see
``.opencode/runs/karthgad-corrected-connectivity/2026-08-04_osg_quaternion_order_implementation_report.md``)
------------------------------------------------------------------------------
OpenMW 0.51 ``components/misc/convert.hpp`` lines 50-54 (``makeOsgQuat``)
constructs ``Qz(-rz) * Qy(-ry) * Qx(-rx)``.  OpenSceneGraph's
``osg::Quat::operator*`` uses reversed Hamilton operand semantics, so the
active column-vector rotation matrix is

    M = Rx(-rx) @ Ry(-ry) @ Rz(-rz)     (rightmost Z rotation applies FIRST)

Blender's default ``XYZ`` mode recomposes a scene Euler triple as
``Rz(z) @ Ry(y) @ Rx(x)``, so Blender scene values are ALWAYS matrix-decomposed
here, never copied from the raw negated TES3 triple (that was the historical
Stage-7 bug; ``[-rx, -ry, -rz]`` recomposes the wrong product).

Inputs / outputs
----------------
All rotations are raw TES3 reference Euler triples ``(rx, ry, rz)`` in radians
(``None`` means identity).  Points are 3-element float sequences in game
units.  Matrices are row-major NumPy float64 arrays acting on column vectors:
3x3 rotation-only, or 4x4 affine where noted.  Scene units are 0.01x game
units (``SCENE_UNITS_PER_GU``), matching the B1 placement manifests.

Public API
----------
* ``tes3_euler_to_matrix(rotation)`` — full-precision ``Rx(-rx) @ Ry(-ry) @ Rz(-rz)``.
* ``tes3_euler_to_matrix_rounded(rotation)`` — same, nine-digit-rounded with
  ``-0.0`` normalized to ``0.0`` (byte-compatible with
  ``karthgad_rebuild_geometry.rotation_xyz_matrix``).
* ``matrix_to_tes3_euler(matrix)`` — inverse decomposition (deterministic near
  gimbal: fixes ``rz = 0`` and solves ``rx`` from the remaining matrix).
* ``blender_xyz_euler_from_matrix(matrix)`` — closed-form Blender XYZ
  decomposition with deterministic near-gimbal candidate selection.
* ``blender_xyz_euler_for_tes3_rotation(rotation)`` — compatibility-first
  serialization: decomposes the rounded matrix; only if its nine-digit
  serialization fails the gate (``BLENDER_SERIALIZED_EULER_MATRIX_TOLERANCE``)
  re-decomposes the full-precision matrix; fails closed otherwise.
* ``rotate_reference_point(point, rotation)`` — ``M @ point`` (scatter family).
* ``engine_up_vector(rotation)`` — ``M @ (0, 0, 1)``; closed form
  ``(-sin ry, sin rx * cos ry, cos rx * cos ry)``.
* ``engine_tilt_for_normal(normal, yaw)`` — solves ``(rx, ry)`` so the engine
  up vector equals the given (unit, up-facing) terrain normal for any yaw
  (groundcover family).  Returns ``(rx, ry)``.
* ``placement_scene_matrix(position, rotation, scale)`` — 4x4 scene-unit
  affine ``T(0.01*pos) @ M @ S(scale)``, nine-digit-rounded like B1.

Invariants
----------
* Column vectors; composition ``Rx(-rx) @ Ry(-ry) @ Rz(-rz)``; Blender XYZ
  recomposition ``Rz(z) @ Ry(y) @ Rx(x)``.
* Scalar transcendentals use ``math.*`` (identical libm calls to the existing
  pure-Python family) so candidate selections and serialized values agree
  bit-for-bit with ``karthgad_rebuild_geometry``; NumPy provides array/matrix
  math (matmul, shape/finiteness validation).  No Blender/scipy dependency.
* Determinism: every function is a pure function of its inputs; singular
  branches fix a degree of freedom (documented per function) instead of
  depending on iteration or environment state.
* Fail-closed: ``blender_xyz_euler_for_tes3_rotation`` raises ``ValueError``
  if no Blender XYZ representation reproduces the authoritative rounded
  matrix within the gate after nine-digit serialization.
* Input validation: non-finite values, wrong-length vectors, non-numeric
  values, and non-3x3/non-4x4 matrices raise ``ValueError``.  Booleans are
  rejected as numbers.

Pipeline position
-----------------
Phase-0 foundation of the cityforge arc (master plan §6 T0.1).  Consumed by
all later placement/landscape/authoring/render stages; the old call sites keep
their own implementations (thin delegation is a later, separate task).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Documented convention constants
# ---------------------------------------------------------------------------

VECTOR_CONVENTION = "column_vectors"
EULER_ORDER = "XYZ"
ROTATION_COMPOSITION = "Rx(-rx) @ Ry(-ry) @ Rz(-rz)"
BLENDER_SCENE_EULER_ORDER = "XYZ"

# B1/scene serialization discipline (identical values to
# karthgad_rebuild_geometry): scene JSON writes nine-digit-rounded Euler
# triples, and Stage 7 gates the recomposition at 1e-7.
BLENDER_EULER_SERIALIZATION_DIGITS = 9
BLENDER_GIMBAL_CANDIDATE_THRESHOLD = 1.0e-6
BLENDER_SERIALIZED_EULER_MATRIX_TOLERANCE = 1.0e-7

# Deterministic singular-branch thresholds.
# - ``matrix_to_tes3_euler``: when |cos(b)| (b = -ry of the engine matrix)
#   falls below this, the decomposition fixes rz = 0 and solves rx from the
#   remaining matrix (m21, m11).
TES3_EULER_INVERSE_GIMBAL_THRESHOLD = 1.0e-12
# - ``blender_xyz_euler_from_matrix``: mirrors karthgad's non-singular/gimbal
#   candidate split (same 1e-12 / 1e-6 values) so selections agree.
BLENDER_NONSINGULAR_COS_Y_FLOOR = 1.0e-12
# - ``engine_tilt_for_normal``: at |nx| ~ 1 the y-axis solve degenerates;
#   return the deterministic (0, ry) branch (groundcover parity).
TILT_COS_RY_FLOOR = 1.0e-9

# Round-trip comparison tolerance used by the focused tests for matrix-space
# comparisons (documented in tests/test_engine_transform.py).  Full-precision
# libm differences between this module and the pure-Python family are ~1e-16;
# nine-digit serialization dominates everything else.
MATRIX_SPACE_ROUND_TRIP_TOLERANCE = 1.0e-9

SCENE_UNITS_PER_GU = 0.01
GU_PER_SCENE_UNIT = 100.0

TRANSFORM_CONVENTION: dict[str, object] = {
    "vector_convention": VECTOR_CONVENTION,
    "euler_order": EULER_ORDER,
    "angle_conversion": "negate TES3 reference Euler angles",
    "rotation_composition": ROTATION_COMPOSITION,
    "placement_scene_matrix": "T(0.01 * position) @ Rx(-rx) @ Ry(-ry) @ Rz(-rz) @ S(scale)",
    "blender_scene_euler_order": BLENDER_SCENE_EULER_ORDER,
    "scene_rotation_encoding": (
        "matrix-decomposed Blender XYZ Euler; rounded recomposition validated"
    ),
    "blender_scene_euler_digits": BLENDER_EULER_SERIALIZATION_DIGITS,
    "blender_scene_rotation_gate": (
        "max element error <= 1e-7 after 9-digit serialization"
    ),
    "composition_note": (
        "OpenMW 0.51 components/misc/convert.hpp lines 50-54 (makeOsgQuat) "
        "composes osg::Quat(rz, -Z) * osg::Quat(ry, -Y) * osg::Quat(rx, -X); "
        "OpenSceneGraph osg::Quat::operator* uses reversed Hamilton operand "
        "semantics, so the active column-vector matrix is Rx(-rx) @ Ry(-ry) "
        "@ Rz(-rz) (the rightmost Z rotation applies first)."
    ),
}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _reject(value: object, label: str) -> None:
    raise ValueError(f"{label} must be a finite three-element sequence of numbers")


def _finite_triplet(values: object, label: str) -> tuple[float, float, float]:
    if values is None:
        return (0.0, 0.0, 0.0)
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        _reject(values, label)
    items = list(values)
    if len(items) != 3:
        _reject(values, label)
    result: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
            _reject(values, label)
        value = float(item)
        if not math.isfinite(value):
            _reject(values, label)
        result.append(value)
    return (result[0], result[1], result[2])


def _rotation_angles(rotation: object) -> tuple[float, float, float]:
    """Validate a TES3 Euler triple (radians); None means identity."""
    return _finite_triplet(rotation, "rotation")


def _point_vector(point: object, label: str = "point") -> tuple[float, float, float]:
    """Validate a 3-element point/vector triple; None is NOT accepted here."""
    if point is None:
        _reject(point, label)
    return _finite_triplet(point, label)


def _rotation_matrix_array(matrix: object, label: str = "rotation matrix") -> np.ndarray:
    """Validate and return a 3x3 float64 rotation matrix (3x3 or affine 4x4)."""
    try:
        array = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric 3x3 or 4x4 matrix") from exc
    if array.ndim != 2:
        _reject(matrix, label)
    if array.shape == (4, 4):
        array = array[:3, :3]
    if array.shape != (3, 3):
        _reject(matrix, label)
    if not np.all(np.isfinite(array)):
        _reject(matrix, label)
    return array


def _round_serialized(value: float, digits: int = BLENDER_EULER_SERIALIZATION_DIGITS) -> float:
    """Nine-digit serialization rounding; ``-0.0`` normalizes to ``0.0``.

    Mirrors ``karthgad_rebuild_geometry._round_float`` so serialized values
    match the B1 byte contract (JSON never writes ``-0.0`` from this module's
    own serialization paths).
    """
    result = round(float(value), digits)
    return 0.0 if result == 0 else result


# ---------------------------------------------------------------------------
# TES3 Euler -> matrix
# ---------------------------------------------------------------------------


def tes3_euler_to_matrix(rotation: Sequence[float] | None) -> np.ndarray:
    """Full-precision ``Rx(-rx) @ Ry(-ry) @ Rz(-rz)`` as a 3x3 float64 array.

    The rightmost Z rotation applies first to column vectors.  Scalar
    transcendentals use ``math.*`` (identical to the pure-Python family);
    only the final matrix product is NumPy.
    """
    rx, ry, rz = _rotation_angles(rotation)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # Rx(-rx): standard right-handed X rotation about the negated angle.
    rx_neg = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]], dtype=np.float64
    )
    # Ry(-ry).
    ry_neg = np.array(
        [[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]], dtype=np.float64
    )
    # Rz(-rz).
    rz_neg = np.array(
        [[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return rx_neg @ (ry_neg @ rz_neg)


def tes3_euler_to_matrix_rounded(rotation: Sequence[float] | None) -> np.ndarray:
    """Nine-digit-rounded authoritative matrix (B1 ``rotation_xyz_matrix`` parity).

    Serialized consumers (scene JSON, manifests) round every matrix element
    to nine digits with ``-0.0`` normalized; this is the matrix the
    compatibility-first Blender Euler path treats as authoritative.
    """
    raw = tes3_euler_to_matrix(rotation)
    rounded = np.array(
        [[_round_serialized(value) for value in row] for row in raw],
        dtype=np.float64,
    )
    return rounded


# ---------------------------------------------------------------------------
# Matrix -> TES3 Euler (inverse decomposition)
# ---------------------------------------------------------------------------


def matrix_to_tes3_euler(matrix: Sequence[Sequence[float]] | np.ndarray) -> tuple[float, float, float]:
    """Decompose ``M = Rx(-rx) @ Ry(-ry) @ Rz(-rz)`` back to ``(rx, ry, rz)``.

    With ``a = -rx, b = -ry, c = -rz`` the product
    ``Rx(a) @ Ry(b) @ Rz(c)`` (standard right-handed matrices, so
    ``Rz(c)[0][1] = -sin c``) has elements

        m00 = cos b * cos c          m01 = -cos b * sin c         m02 = sin b
        m10 = sin a sin b cos c + cos a sin c
        m11 = -sin a sin b sin c + cos a cos c
        m12 = -sin a cos b
        m20 = -cos a sin b cos c + sin a sin c
        m21 = cos a sin b sin c + sin a cos c
        m22 = cos a cos b

    so the non-singular solve is ``b = atan2(m02, hypot(m00, m01))`` (always
    in [-pi/2, pi/2] with cos b >= 0), ``a = atan2(-m12, m22)`` and
    ``c = atan2(-m01, m00)``.

    Near gimbal (|cos b| < ``TES3_EULER_INVERSE_GIMBAL_THRESHOLD``) the
    decomposition fixes ``c = 0`` deterministically: with ``Rz(0) = I`` the
    matrix is ``Rx(a) @ Ry(b)`` with ``m21 = sin a`` and ``m11 = cos a``, so
    ``a = atan2(m21, m11)`` and ``b = asin(m02)``.  This branch is exact at
    the singularity (any ``c`` can be absorbed into ``a``) and deterministic.

    Returns the triple ``(-a, -b, -c)`` in principal ranges:
    rx in [-pi, pi], ry in [-pi/2, pi/2], rz in [-pi, pi].
    """
    m = _rotation_matrix_array(matrix)
    m00, m01, m02 = m[0, 0], m[0, 1], m[0, 2]
    m11, m12 = m[1, 1], m[1, 2]
    m21, m22 = m[2, 1], m[2, 2]

    sin_b = m02
    cos_b = math.hypot(m00, m01)
    if cos_b >= TES3_EULER_INVERSE_GIMBAL_THRESHOLD:
        b = math.atan2(sin_b, cos_b)
        a = math.atan2(-m12, m22)
        c = math.atan2(-m01, m00)
    else:
        # Deterministic singular branch: absorb any z rotation into x.
        b = math.asin(max(-1.0, min(1.0, sin_b)))
        a = math.atan2(m21, m11)
        c = 0.0
    return (-a, -b, -c)


# ---------------------------------------------------------------------------
# Matrix -> Blender XYZ Euler (with deterministic near-gimbal handling)
# ---------------------------------------------------------------------------


def _blender_xyz_matrix(euler: Sequence[float]) -> np.ndarray:
    """Recompose Blender's column-vector XYZ rotation ``Rz(z) @ Ry(y) @ Rx(x)``."""
    x, y, z = (float(value) for value in euler)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ (ry @ rx)


def _matrix_max_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Maximum absolute element difference (the Stage-7 gate metric)."""
    return float(np.max(np.abs(actual - expected)))


def _serialized_blender_euler_error(euler: Sequence[float], matrix: np.ndarray) -> float:
    """Gate-style error of an Euler triple after nine-digit serialization.

    Mirrors Stage 7's ``prepare_render_scenes._validate_serialized_euler``
    exactly: plain ``round(value, 9)`` (no ``-0.0`` normalization, matching
    JSON serialization semantics), Blender-order recomposition, max element
    error against the authoritative matrix.
    """
    serialized = tuple(
        round(float(value), BLENDER_EULER_SERIALIZATION_DIGITS) for value in euler
    )
    return _matrix_max_error(_blender_xyz_matrix(serialized), matrix)


def blender_xyz_euler_from_matrix(
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[float, float, float]:
    """Decompose a rotation matrix into stable Blender XYZ Euler values.

    Blender's default ``XYZ`` mode recomposes a column-vector matrix as
    ``Rz(z) @ Ry(y) @ Rx(x)``.  The non-singular closed form is

        cos_y = hypot(m00, m10)
        x = atan2(m21, m22);  y = atan2(-m20, cos_y);  z = atan2(m10, m00)

    Near gimbal (cos_y <= ``BLENDER_GIMBAL_CANDIDATE_THRESHOLD``) two
    deterministic z = 0 representations at y = +/-pi/2 are added:

        y = +pi/2:  x = atan2(m01, m02)
        y = -pi/2:  x = atan2(-m01, -m02)

    Every candidate is scored by (serialized error, full-precision error,
    priority, euler tuple) and the minimum is returned, so repeated builds
    are deterministic and the same candidate math as
    ``karthgad_rebuild_geometry.blender_xyz_euler_from_matrix`` is used.
    Accepts a 3x3 rotation matrix or an affine 4x4 (translation ignored).
    """
    m = _rotation_matrix_array(matrix)
    m00, m01, m02 = m[0, 0], m[0, 1], m[0, 2]
    m10 = m[1, 0]
    m20, m21, m22 = m[2, 0], m[2, 1], m[2, 2]

    cos_y = math.hypot(m00, m10)
    candidates: list[tuple[tuple[float, float, float], float, float, int]] = []

    def add_candidate(euler: tuple[float, float, float], priority: int) -> None:
        full_error = _matrix_max_error(_blender_xyz_matrix(euler), m)
        serialized = tuple(
            _round_serialized(value, BLENDER_EULER_SERIALIZATION_DIGITS)
            for value in euler
        )
        serialized_error = _matrix_max_error(_blender_xyz_matrix(serialized), m)
        if not (math.isfinite(full_error) and math.isfinite(serialized_error)):
            raise ValueError(
                "Blender XYZ candidate recomposition is not finite; "
                "matrix is not a valid rotation"
            )
        candidates.append((euler, serialized_error, full_error, priority))

    if cos_y > BLENDER_NONSINGULAR_COS_Y_FLOOR:
        add_candidate(
            (math.atan2(m21, m22), math.atan2(-m20, cos_y), math.atan2(m10, m00)),
            0,
        )
    else:
        # Exact singularity: keep the branch deterministic by forcing the
        # gimbal candidate path below (same split as the B1 helper).
        cos_y = 0.0

    if cos_y <= BLENDER_GIMBAL_CANDIDATE_THRESHOLD:
        add_candidate((math.atan2(m01, m02), math.pi / 2.0, 0.0), 1)
        add_candidate((math.atan2(-m01, -m02), -math.pi / 2.0, 0.0), 2)

    if not candidates:
        raise ValueError("Blender XYZ decomposition produced no valid candidate")
    # Serialized error is the primary score (scene JSON writes the rounded
    # triple); full-precision error and candidate priority break ties.
    selected = min(candidates, key=lambda item: (item[1], item[2], item[3], item[0]))
    return selected[0]


def blender_xyz_euler_for_tes3_rotation(
    rotation: Sequence[float] | None,
) -> tuple[float, float, float]:
    """Blender XYZ encoding of a raw TES3 rotation, compatibility-first.

    Primary path: decompose the nine-digit-rounded authoritative matrix and
    keep that representation whenever its nine-digit serialization passes the
    ``BLENDER_SERIALIZED_EULER_MATRIX_TOLERANCE`` gate against the SAME
    rounded matrix (ordinary refs stay byte-identical to prior builds).

    Fallback path (only when the primary fails, e.g. near-gimbal refs like
    Heldorn ``-125_10_ref_029742``): re-decompose the full-precision matrix,
    whose near-gimbal terms were not lost to nine-digit rounding, and score
    the fallback after serialization against the SAME authoritative rounded
    matrix.  If neither representation passes, the build fails closed
    (``ValueError``) — no degraded Euler is ever emitted.
    """
    rounded = tes3_euler_to_matrix_rounded(rotation)
    raw = tes3_euler_to_matrix(rotation)

    primary = blender_xyz_euler_from_matrix(rounded)
    primary_error = _serialized_blender_euler_error(primary, rounded)
    if primary_error <= BLENDER_SERIALIZED_EULER_MATRIX_TOLERANCE:
        return primary

    fallback = blender_xyz_euler_from_matrix(raw)
    fallback_error = _serialized_blender_euler_error(fallback, rounded)
    if fallback_error > BLENDER_SERIALIZED_EULER_MATRIX_TOLERANCE:
        raise ValueError(
            "no Blender XYZ Euler representation reproduces the authoritative "
            f"rotation matrix within {BLENDER_SERIALIZED_EULER_MATRIX_TOLERANCE:.1e} "
            f"after {BLENDER_EULER_SERIALIZATION_DIGITS}-digit serialization "
            f"(TES3 rotation {list(rotation) if rotation is not None else None!r}); "
            f"rounded-matrix decomposition error {primary_error:.6g}, "
            f"full-precision fallback error {fallback_error:.6g}"
        )
    return fallback


# ---------------------------------------------------------------------------
# Reference-point rotation / engine basis vectors
# ---------------------------------------------------------------------------


def rotate_reference_point(
    point: Sequence[float], rotation: Sequence[float] | None
) -> tuple[float, float, float]:
    """Apply the authoritative OpenMW matrix to one raw TES3 point.

    ``result = M @ point`` with ``M = Rx(-rx) @ Ry(-ry) @ Rz(-rz)`` for column
    vectors (the rightmost Z rotation applies first).  Equivalent to
    ``scatter_analysis.rotate_tes3_reference_point``; kept as one line of
    matrix math so this module has a single source of truth.
    """
    px, py, pz = _point_vector(point)
    matrix = tes3_euler_to_matrix(rotation)
    result = matrix @ np.array([px, py, pz], dtype=np.float64)
    return (float(result[0]), float(result[1]), float(result[2]))


def engine_up_vector(rotation: Sequence[float] | None) -> tuple[float, float, float]:
    """Map the local up vector (0, 0, 1) through the engine rotation.

    ``M @ (0, 0, 1)`` reduces to the closed form
    ``(-sin ry, sin rx * cos ry, cos rx * cos ry)`` because the rightmost
    Z yaw acts on the local quad before the two tilt axes and leaves the up
    vector unchanged.  Matches ``groundcover_generate._engine_up``.
    """
    rx, ry, _ = _rotation_angles(rotation)
    return (
        -math.sin(ry),
        math.sin(rx) * math.cos(ry),
        math.cos(rx) * math.cos(ry),
    )


def engine_tilt_for_normal(
    normal: Sequence[float], yaw: float
) -> tuple[float, float]:
    """Solve ``(rx, ry)`` so the engine up vector equals a terrain normal.

    Precondition: ``normal`` is a unit vector with ``nz > 0`` (an up-facing
    terrain normal, as produced by terrain samplers).  Solving
    ``up = (-sin ry, sin rx cos ry, cos rx cos ry) = normal`` gives
    ``ry = asin(-nx)`` and ``rx = asin(ny / cos ry)``; the solve is exact for
    any ``yaw`` (which stays in the authored rotation triple but does not
    enter the normal solve — the rightmost Z rotation is applied before the
    tilt axes and cannot lift the up vector).

    At ``|nx| ~ 1`` (vertical slope) ``cos ry`` degenerates; the deterministic
    branch returns ``(0.0, ry)``, mirroring ``groundcover_generate._engine_tilt``.
    """
    nx, ny, _ = _point_vector(normal, "normal")
    _ = float(yaw)  # retained in the API; see the composition note above
    ry = math.asin(max(-1.0, min(1.0, -nx)))
    cos_ry = math.sqrt(max(0.0, 1.0 - math.sin(ry) ** 2))
    if cos_ry < TILT_COS_RY_FLOOR:
        return (0.0, ry)
    rx = math.asin(max(-1.0, min(1.0, ny / cos_ry)))
    return (rx, ry)


# ---------------------------------------------------------------------------
# Placement matrix (scene units)
# ---------------------------------------------------------------------------


def placement_scene_matrix(
    position: Sequence[float],
    rotation: Sequence[float] | None,
    scale: float | None,
) -> np.ndarray:
    """Compose the TES3 ref transform in scene units as a 4x4 float64 array.

    ``M = T(0.01 * position) @ Rx(-rx) @ Ry(-ry) @ Rz(-rz) @ S(scale)``,
    nine-digit-rounded with ``-0.0`` normalized (byte-compatible with
    ``karthgad_rebuild_geometry.placement_scene_matrix``: the ROTATION matrix
    is rounded before the scale is applied, exactly as B1 does).  ``scale =
    None`` means 1.0.
    """
    px, py, pz = _point_vector(position, "position")
    normalized_scale = 1.0 if scale is None else float(scale)
    if isinstance(scale, bool) or not math.isfinite(normalized_scale):
        raise ValueError("scale must be a finite number or None")

    # B1 parity: the linear block is the nine-digit-rounded rotation matrix
    # scaled, then the whole 4x4 is rounded again.
    rotation_matrix = tes3_euler_to_matrix_rounded(rotation)
    linear = rotation_matrix * normalized_scale

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = linear
    result[0, 3] = SCENE_UNITS_PER_GU * px
    result[1, 3] = SCENE_UNITS_PER_GU * py
    result[2, 3] = SCENE_UNITS_PER_GU * pz
    rounded = np.array(
        [[_round_serialized(value) for value in row] for row in result],
        dtype=np.float64,
    )
    return rounded
