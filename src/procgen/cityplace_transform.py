"""Matrix-space stamp seating and replay/oracle gates for Cityforge T1.2.

Pipeline position
------------------
This module is the only T1.2 code that composes a plan yaw with a D-STAMP
member rotation.  It sits after T1.1 deterministic stamp resolution and before
terrain/collision checks.  All authoritative rotations are delegated to
``procgen.engine_transform``; this file never performs direct Euler yaw
addition and never uses Blender Euler values for TES3 authoring.

The plan yaw is a conventional positive-CCW world yaw.  To express that world
matrix through the engine's TES3 convention, the helper asks
``engine_transform.tes3_euler_to_matrix`` for ``(0, 0, -yaw_rad)``.  That
negative raw TES3 value is an encoding conversion, not Euler arithmetic.  The
same matrix rotates member offsets and left-multiplies each member's source
matrix.  The output raw Euler is decomposed from the composed matrix, except
at exact yaw zero where the source-authored triple is retained byte-for-byte
for the source replay gate.

Inputs and outputs
------------------
``place_stamp_members`` consumes one D-STAMP stamp, a world anchor, and an
exact plan yaw.  It returns JSON-ready member transforms containing world GU
positions, raw TES3 Euler radians, authoritative matrix evidence, scales, and
mathematical cell buckets.  ``replay_source_libraries`` independently reads
the hash-pinned source component manifests embedded in both libraries and
proves every eligible source member.  ``yaw37_oracle`` uses a deliberately
separate standard-matrix implementation for all multi-axis members.

Invariants
----------
* Member-relative offsets, source scales, and record types are never altered.
* Source yaw zero reproduces the D-STAMP source records; nonzero yaw is matrix
  composition followed by deterministic TES3 decomposition.
* World cell buckets use ``floor(world_gu / 8192)`` for negative coordinates.
* Replay, oracle, and transform mismatches raise ``TransformContractError``;
  callers must use the stage failure protocol rather than degrade fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import engine_transform
from .cityplace_contracts import CityPlaceInputError, sha256_file


WORLD_CELL_SIZE_GU = 8192.0
POSITION_TOLERANCE_GU = 0.001
LINEAR_TOLERANCE = 0.0001
MATRIX_TRANSLATION_TOLERANCE_GU = 0.01
ORACLE_ANGLE_DEG = 37.0


class TransformContractError(CityPlaceInputError):
    """Fatal source-replay or matrix-composition mismatch."""


def _finite_triplet(value: Sequence[Any], label: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise TransformContractError(f"{label} must contain three numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise TransformContractError(f"{label} contains a non-finite number")
    return result  # type: ignore[return-value]


def _matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=np.float64)]


def _vec_list(value: Sequence[float]) -> list[float]:
    return [float(item) for item in value]


def mathematical_cell_bucket(world_x_gu: float, world_y_gu: float) -> list[int]:
    """TES3 exterior cell grid using mathematical floor on both axes."""

    return [
        int(math.floor(float(world_x_gu) / WORLD_CELL_SIZE_GU)),
        int(math.floor(float(world_y_gu) / WORLD_CELL_SIZE_GU)),
    ]


def world_yaw_matrix(yaw_deg: float) -> np.ndarray:
    """Return the standard positive-CCW world yaw through engine_transform."""

    if not math.isfinite(float(yaw_deg)):
        raise TransformContractError("plan yaw must be finite")
    # engine_transform encodes raw TES3 rz with a negative sign.  Passing the
    # negated raw angle yields the requested standard +CCW world matrix.
    return engine_transform.tes3_euler_to_matrix((0.0, 0.0, -math.radians(float(yaw_deg))))


def _compose_rotation(
    source_rotation: Sequence[float], yaw_deg: float
) -> tuple[np.ndarray, tuple[float, float, float]]:
    source = _finite_triplet(source_rotation, "source member rotation")
    source_matrix = engine_transform.tes3_euler_to_matrix(source)
    yaw_matrix = world_yaw_matrix(yaw_deg)
    composed = np.asarray(yaw_matrix @ source_matrix, dtype=np.float64)
    # Retaining the authored triple at yaw zero proves source replay and avoids
    # replacing a legitimate 2*pi-equivalent source representation.
    if float(yaw_deg) == 0.0:
        raw = source
    else:
        raw = tuple(float(item) for item in engine_transform.matrix_to_tes3_euler(composed))
    rebuilt = engine_transform.tes3_euler_to_matrix(raw)
    error = float(np.max(np.abs(rebuilt - composed)))
    if error > 1.0e-9:
        raise TransformContractError(
            f"TES3 Euler decomposition changed composed matrix by {error:.3g}"
        )
    return composed, raw


@dataclass(frozen=True)
class PlacedMember:
    """One fully seated member, retaining both raw authoring and matrix proof."""

    source_id: str
    object_id: str | None
    record_type: str
    model_key: str | None
    structural_role: str | None
    is_door: bool
    scale: float
    source_offset_gu: tuple[float, float, float]
    source_rotation: tuple[float, float, float]
    world_position_gu: tuple[float, float, float]
    raw_tes3_rotation_rad: tuple[float, float, float]
    rotation_matrix: np.ndarray
    source_rotation_matrix: np.ndarray

    def to_dict(self, *, include_render_euler: bool = True) -> dict[str, Any]:
        """Serialize the member without conflating render and authoring Euler."""

        world = self.world_position_gu
        result: dict[str, Any] = {
            "source_id": self.source_id,
            "object_id": self.object_id,
            "record_type": self.record_type,
            "model_key": self.model_key,
            "structural_role": self.structural_role,
            "is_door": self.is_door,
            "scale": self.scale,
            "source_offset_gu": _vec_list(self.source_offset_gu),
            "source_rotation_tes3_rad": _vec_list(self.source_rotation),
            "world_position_gu": _vec_list(world),
            "world_cell": mathematical_cell_bucket(world[0], world[1]),
            "raw_tes3_rotation_rad": _vec_list(self.raw_tes3_rotation_rad),
            "rotation_matrix_3x3": _matrix_list(self.rotation_matrix),
            "matrix_evidence": {
                "composition": "Rz_world(yaw) @ engine_transform.tes3_euler_to_matrix(source_rotation)",
                "source_matrix_3x3": _matrix_list(self.source_rotation_matrix),
                "composed_matrix_3x3": _matrix_list(self.rotation_matrix),
            },
        }
        if include_render_euler:
            # Explicitly labelled render-only data; TES3 output must consume the
            # raw field above, never this Blender compatibility representation.
            result["render_data"] = {
                "blender_xyz_euler_rad": list(
                    engine_transform.blender_xyz_euler_for_tes3_rotation(
                        self.raw_tes3_rotation_rad
                    )
                ),
                "label": "optional_preconverted_render_euler; not TES3 authoring data",
            }
        return result


def place_stamp_members(
    stamp: Mapping[str, Any],
    *,
    anchor_world_gu: Sequence[float],
    yaw_deg: float,
    include_render_euler: bool = True,
) -> list[PlacedMember]:
    """Transform every D-STAMP member around the seed-door world anchor."""

    anchor = np.asarray(_finite_triplet(anchor_world_gu, "stamp anchor"), dtype=np.float64)
    yaw_matrix = world_yaw_matrix(float(yaw_deg))
    members = stamp.get("members")
    if not isinstance(members, list) or not members:
        raise TransformContractError(f"stamp {stamp.get('stamp_id')} has no members")
    placed: list[PlacedMember] = []
    for index, member in enumerate(members):
        if not isinstance(member, Mapping):
            raise TransformContractError(f"stamp member {index} is not an object")
        source_id = member.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise TransformContractError(f"stamp member {index} has no source_id")
        offset = np.asarray(_finite_triplet(member.get("offset_gu"), "member offset"), dtype=np.float64)
        source_rotation = _finite_triplet(member.get("rotation"), "member rotation")
        composed, raw = _compose_rotation(source_rotation, float(yaw_deg))
        # Position and orientation use the same world-yaw matrix.  The source
        # offset's z component is deliberately preserved by this Z rotation.
        world_position = tuple(float(v) for v in anchor + yaw_matrix @ offset)
        scale_value = member.get("scale", 1.0)
        scale = 1.0 if scale_value is None else float(scale_value)
        if not math.isfinite(scale) or scale <= 0.0:
            raise TransformContractError(f"member {source_id} has invalid scale {scale_value!r}")
        placed.append(
            PlacedMember(
                source_id=source_id,
                object_id=member.get("object_id") if isinstance(member.get("object_id"), str) else None,
                record_type=str(member.get("record_type", "")),
                model_key=member.get("model_key") if isinstance(member.get("model_key"), str) else None,
                structural_role=(
                    member.get("structural_role")
                    if isinstance(member.get("structural_role"), str)
                    else None
                ),
                is_door=bool(member.get("is_door", False)),
                scale=scale,
                source_offset_gu=tuple(float(v) for v in offset),
                source_rotation=source_rotation,
                world_position_gu=world_position,
                raw_tes3_rotation_rad=raw,
                rotation_matrix=composed,
                source_rotation_matrix=engine_transform.tes3_euler_to_matrix(source_rotation),
            )
        )
    return placed


def _independent_standard_matrix(rotation: Sequence[float]) -> np.ndarray:
    """Independent oracle for ``Rx(-rx) @ Ry(-ry) @ Rz(-rz)``.

    This intentionally does not import or call any production transform helper;
    it is used by the 37-degree gate to catch a shared-helper mistake.
    """

    rx, ry, rz = (float(v) for v in rotation)
    ax, ay, az = -rx, -ry, -rz
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rx_m @ ry_m @ rz_m


def _independent_world_yaw(yaw_deg: float) -> np.ndarray:
    angle = math.radians(float(yaw_deg))
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw37_oracle(
    stamps: Iterable[Mapping[str, Any]], *, yaw_deg: float = ORACLE_ANGLE_DEG
) -> dict[str, Any]:
    """Compare production transforms to an independent 37-degree matrix oracle."""

    checked = 0
    multi_axis = 0
    mismatches: list[dict[str, Any]] = []
    canaries: list[str] = []
    max_linear_seen = 0.0
    max_position_seen = 0.0
    for stamp in sorted(stamps, key=lambda item: str(item.get("stamp_id"))):
        anchor = stamp.get("anchor", {}).get("source_position_gu")
        if not isinstance(anchor, list):
            raise TransformContractError(f"stamp {stamp.get('stamp_id')} has no source anchor")
        for member in stamp.get("members", []):
            rotation = member.get("rotation")
            if not isinstance(rotation, list) or len(rotation) != 3:
                raise TransformContractError(f"member {member.get('source_id')} has bad rotation")
            if abs(float(rotation[0])) <= 1.0e-12 and abs(float(rotation[1])) <= 1.0e-12:
                continue
            multi_axis += 1
            placed = place_stamp_members(
                {"stamp_id": stamp.get("stamp_id"), "members": [member]},
                anchor_world_gu=anchor,
                yaw_deg=yaw_deg,
                include_render_euler=False,
            )[0]
            oracle_yaw = _independent_world_yaw(yaw_deg)
            oracle_source = _independent_standard_matrix(rotation)
            oracle_matrix = oracle_yaw @ oracle_source
            offset = np.asarray(member["offset_gu"], dtype=np.float64)
            oracle_position = np.asarray(anchor, dtype=np.float64) + oracle_yaw @ offset
            linear_error = float(np.max(np.abs(placed.rotation_matrix - oracle_matrix)))
            position_error = float(np.max(np.abs(np.asarray(placed.world_position_gu) - oracle_position)))
            max_linear_seen = max(max_linear_seen, linear_error)
            max_position_seen = max(max_position_seen, position_error)
            checked += 1
            if linear_error > 1.0e-9 or position_error > POSITION_TOLERANCE_GU:
                mismatches.append(
                    {
                        "stamp_id": stamp.get("stamp_id"),
                        "source_id": member.get("source_id"),
                        "linear_error": linear_error,
                        "position_error_gu": position_error,
                    }
                )
            if member.get("source_id") == "-102_11_ref_095307":
                canaries.append(member["source_id"])
    return {
        "yaw_deg": float(yaw_deg),
        "multi_axis_members": multi_axis,
        "checked_members": checked,
        "mismatches": len(mismatches),
        "max_linear_error": max_linear_seen,
        "max_position_error_gu": max_position_seen,
        "known_canaries": sorted(set(canaries)),
        "details": mismatches,
        "tolerances": {
            "linear_matrix_element": "1e-9",
            "position_game_units": POSITION_TOLERANCE_GU,
        },
    }


def _resolve_workspace_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else root / path


def _source_members_by_id(
    library: Mapping[str, Any], *, workspace_root: Path
) -> tuple[dict[str, Mapping[str, Any]], int]:
    """Load every source component manifest listed in one library's inputs."""

    by_id: dict[str, Mapping[str, Any]] = {}
    manifest_count = 0
    inputs = library.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TransformContractError(f"library {library.get('library_id')} has no inputs")
    for relative, expected_hash in sorted(inputs.items()):
        if not isinstance(relative, str) or not relative.endswith("/manifest.json"):
            continue
        path = _resolve_workspace_path(workspace_root, relative)
        if not path.is_file():
            raise TransformContractError(f"source manifest is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise TransformContractError(
                f"source manifest hash drift for {relative}: {actual_hash} != {expected_hash}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TransformContractError(f"cannot read source manifest {path}: {exc}") from exc
        members = payload.get("members") if isinstance(payload, Mapping) else None
        if not isinstance(members, list):
            raise TransformContractError(f"source manifest {path} has no members list")
        manifest_count += 1
        for member in members:
            if not isinstance(member, Mapping) or not isinstance(member.get("source_id"), str):
                raise TransformContractError(f"source manifest {path} has malformed member")
            source_id = str(member["source_id"])
            if source_id in by_id:
                raise TransformContractError(f"duplicate source member id in library inputs: {source_id}")
            by_id[source_id] = member
    return by_id, manifest_count


def replay_source_libraries(
    libraries: Iterable[Mapping[str, Any]],
    eligible_stamp_ids: set[str],
    *,
    workspace_root: Path | str,
) -> dict[str, Any]:
    """Replay every eligible source member against source manifests.

    The source library's embedded replay block is evidence, not authority for
    this gate.  Positions, raw rotations, scales, and placement matrices are
    reconstructed independently from the hash-pinned component manifests.
    """

    root = Path(workspace_root)
    checked_stamps = 0
    checked_members = 0
    multi_axis_members = 0
    position_mismatches = 0
    rotation_mismatches = 0
    scale_mismatches = 0
    mesh_mismatches = 0
    matrix_mismatches = 0
    max_position_error = 0.0
    max_linear_error = 0.0
    max_translation_error = 0.0
    library_rows: list[dict[str, Any]] = []
    for library in sorted(libraries, key=lambda item: str(item.get("library_id"))):
        source_by_id, manifest_count = _source_members_by_id(library, workspace_root=root)
        lib_stamps = [
            stamp for stamp in library.get("stamps", [])
            if isinstance(stamp, Mapping) and stamp.get("stamp_id") in eligible_stamp_ids
        ]
        lib_counts = {"stamps": 0, "members": 0, "position_mismatches": 0,
                      "rotation_mismatches": 0, "scale_mismatches": 0,
                      "mesh_mismatches": 0, "matrix_mismatches": 0}
        for stamp in sorted(lib_stamps, key=lambda item: str(item.get("stamp_id"))):
            stamp_id = str(stamp["stamp_id"])
            anchor = _finite_triplet(stamp.get("anchor", {}).get("source_position_gu"),
                                     f"{stamp_id} source anchor")
            checked_stamps += 1
            lib_counts["stamps"] += 1
            for member in stamp.get("members", []):
                if not isinstance(member, Mapping):
                    raise TransformContractError(f"{stamp_id} contains malformed member")
                source_id = str(member.get("source_id"))
                source = source_by_id.get(source_id)
                if source is None:
                    raise TransformContractError(f"{stamp_id} source member is absent: {source_id}")
                for field_name in ("model_key", "record_type", "object_id"):
                    if field_name in source and source.get(field_name) != member.get(field_name):
                        mesh_mismatches += 1
                        lib_counts["mesh_mismatches"] += 1
                        raise TransformContractError(
                            f"{stamp_id}/{source_id} {field_name} differs from source manifest"
                        )
                source_transform = source.get("source_transform")
                if not isinstance(source_transform, Mapping):
                    raise TransformContractError(f"{source_id} source_transform is missing")
                source_position = _finite_triplet(source_transform.get("position_gu"), source_id)
                offset = _finite_triplet(member.get("offset_gu"), source_id + " offset")
                reconstructed = tuple(anchor[i] + offset[i] for i in range(3))
                position_error = max(abs(reconstructed[i] - source_position[i]) for i in range(3))
                max_position_error = max(max_position_error, position_error)
                checked_members += 1
                lib_counts["members"] += 1
                if position_error > POSITION_TOLERANCE_GU:
                    position_mismatches += 1
                    lib_counts["position_mismatches"] += 1
                expected_rotation = _finite_triplet(source_transform.get("rotation"), source_id)
                actual_rotation = _finite_triplet(member.get("rotation"), source_id + " D-STAMP rotation")
                rotation_error = max(abs(actual_rotation[i] - expected_rotation[i]) for i in range(3))
                if rotation_error > 1.0e-9:
                    rotation_mismatches += 1
                    lib_counts["rotation_mismatches"] += 1
                expected_scale = 1.0 if source_transform.get("scale") is None else float(source_transform["scale"])
                actual_scale = float(member.get("scale", 1.0))
                if abs(actual_scale - expected_scale) > 1.0e-9:
                    scale_mismatches += 1
                    lib_counts["scale_mismatches"] += 1
                source_matrix = engine_transform.tes3_euler_to_matrix(expected_rotation)
                library_matrix = engine_transform.tes3_euler_to_matrix(actual_rotation)
                linear_error = float(np.max(np.abs(source_matrix - library_matrix)))
                max_linear_error = max(max_linear_error, linear_error)
                source_scene = source.get("placement_scene_matrix")
                if not isinstance(source_scene, list):
                    raise TransformContractError(f"{source_id} has no placement_scene_matrix")
                expected_scene = np.asarray(source_scene, dtype=np.float64)
                actual_scene = engine_transform.placement_scene_matrix(
                    source_position, actual_rotation, actual_scale
                )
                if expected_scene.shape != (4, 4):
                    raise TransformContractError(f"{source_id} placement matrix is not 4x4")
                scene_error = float(np.max(np.abs(expected_scene - actual_scene)))
                translation_error = float(np.max(np.abs(expected_scene[:3, 3] - actual_scene[:3, 3]))) / 0.01
                max_translation_error = max(max_translation_error, translation_error)
                if scene_error > LINEAR_TOLERANCE or translation_error > MATRIX_TRANSLATION_TOLERANCE_GU:
                    matrix_mismatches += 1
                    lib_counts["matrix_mismatches"] += 1
                if abs(expected_rotation[0]) > 1.0e-12 or abs(expected_rotation[1]) > 1.0e-12:
                    multi_axis_members += 1
        library_rows.append({
            "library_id": library.get("library_id"),
            "source_manifests_checked": manifest_count,
            **lib_counts,
        })
    result = {
        "libraries": library_rows,
        "eligible_stamps_checked": checked_stamps,
        "members_checked": checked_members,
        "multi_axis_members": multi_axis_members,
        "position_mismatches": position_mismatches,
        "rotation_mismatches": rotation_mismatches,
        "scale_mismatches": scale_mismatches,
        "mesh_mismatches": mesh_mismatches,
        "matrix_mismatches": matrix_mismatches,
        "max_position_error_gu": max_position_error,
        "max_linear_matrix_error": max_linear_error,
        "max_translation_error_gu": max_translation_error,
        "tolerances": {
            "position_gu": POSITION_TOLERANCE_GU,
            "linear_matrix_element": LINEAR_TOLERANCE,
            "translation_gu": MATRIX_TRANSLATION_TOLERANCE_GU,
            "rotation_rad": "1e-9",
            "scale": "1e-9",
        },
    }
    if any(result[key] for key in (
        "position_mismatches", "rotation_mismatches", "scale_mismatches",
        "mesh_mismatches", "matrix_mismatches"
    )):
        raise TransformContractError(f"source replay mismatch: {result}")
    return result
