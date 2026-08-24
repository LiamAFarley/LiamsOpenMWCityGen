"""Cityforge T1.5 terrain-backed review-render contract and audit core.

Purpose
-------
This module is the ordinary-Python host side of the Cityforge review renderer.
It consumes the *accepted* T1.1 plan/validation, the T1.2 final-reseat
``city_placement.json``, and the T1.3 final terrain/LAND products.  It emits a
deterministic render-scene contract for the Blender worker and independently
audits the worker's imports, transforms, terrain, clipped water, cameras, and
PNG outputs.

Pipeline position
-----------------

``T1.1 validation -> T1.2 final placement -> T1.3 final terrain/LAND
-> THIS host contract/audit -> Blender NIF/material worker -> review PNGs``

The renderer is deliberately downstream-only.  It never authors a production
ESP, edits a source plugin, reinterprets a plan lot, or creates a fallback box
for a missing asset.  The Blender worker receives every placed member as an
individual scene entry, with its raw TES3 rotation converted through
``procgen.engine_transform`` and its expected relative placement matrix
recorded beside it.

Inputs
======
* T1.1 plan and zero-error validation result;
* T1.2 final-reseat placement product (not the planned/provisional product);
* T1.3 ``final_terrain_field.npz`` and metadata;
* T1.3 ``land_records.json`` and its local LTEX records;
* the read-only mesh/texture roots from ``configs/procgen.json``.

Outputs
-------
``build_render_scene`` returns a JSON-ready scene document and manifest.  The
CLI writes those documents, the Blender worker writes a worker audit, and
``finalize_render_audit`` adds dimensions, hashes, luminance/nonblank metrics,
and the final acceptance-facing count block.

Hard invariants
---------------
* all 49 T1.3 LAND cells are present and decode to the quantized final field;
* all emitted positive VTEX values have exactly one local LTEX record and each
  declared LTEX file resolves under the configured read-only roots;
* every placed member has a resolved NIF and a non-empty Blender import group;
* Blender XYZ Euler serialization reconstructs the authoritative rounded
  engine matrix with maximum element error ``<= 1e-7``;
* translations use one declared game-unit render origin and ``0.01`` scene
  units per game unit; normalization/grounding is false;
* water is made only from triangles clipped to the final field at scene ``z=0``;
  a rectangular or external water plane is never a success path;
* every street and focused single-lot detail camera is selected from an ordered,
  deterministic door/road candidate set against the exact final height field; every
  selected camera has at least ``0.30`` scene-unit terrain line-of-sight
  clearance and ``12.0`` scene-unit finite-edge clearance;
* focused detail cameras must keep their actual imported lot bounds in frame at
  a non-microscopic span, while the eleven base views retain their fixed view
  identities and count;
* any failed essential gate raises ``RenderContractError``.  Callers must
  expose that as ``FAILURE: render_city <reason>`` rather than degrading.

The worker imports the existing ``blender_flat_render`` NIF resolver/importer
and its image-backed material conventions through a wrapper; that existing
tool is not modified by this stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import engine_transform, espland, tes3json
from .meshcheck import AssetResolver, configured_data_roots


SCHEMA_VERSION = 2
STAGE = "cityforge_t1_5_render"
SCENE_UNITS_PER_GAME_UNIT = engine_transform.SCENE_UNITS_PER_GU
MATRIX_TOLERANCE = engine_transform.BLENDER_SERIALIZED_EULER_MATRIX_TOLERANCE
INPUT_MATRIX_EVIDENCE_TOLERANCE = 1.0e-6
FIELD_SIDE = 449
LAND_SIDE = 65
CELL_SIZE_GU = 8192.0
FIELD_SPACING_GU = 128.0
TILE_SIDE = 16
TILE_FACES_PER_SIDE = 4
SYNTHETIC_BANNER = "SYNTHETIC ENGINE FIXTURE — NOT A FALKREATH DESIGN"

# The eleven original views remain the required base review set.  The seven
# additional views are deliberately focused-detail views for the sparse
# synthetic fixture; they use the same scene/build and are still hard-gated
# once declared in the scene contract.
VIEW_CONTRACT_VERSION = 2
REQUIRED_BASE_VIEW_COUNT = 11
FOCUSED_DETAIL_VIEW_COUNT = 7

# Camera values are scene units (100 game units = 1 scene unit).  These are
# review-camera parameters only: no placement, terrain, or source asset is
# changed to make a camera pass.
STREET_EYE_HEIGHT_SCENE = 2.35
STREET_TARGET_OFFSETS_SCENE = (0.15, 0.90, 1.55)
STREET_COMPOSITION_TARGET_RAISE_SCENE = 1.15
STREET_DOOR_LOS_FLOOR_OFFSET_SCENE = 0.70
STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE = 1.00
STREET_GROUND_INTERFACE_LOS_CLEARANCE_SCENE = 0.75
STREET_MAX_DOOR_BAND_OCCLUSION_FRACTION = 0.05
STREET_EDGE_MARGIN_SCENE = 12.0
STREET_LOS_SAMPLE_COUNT = 96
STREET_FIT_MARGIN = 0.92
STREET_CAMERA_CANDIDATE_COUNT = 41
STREET_LOS_TARGET_NAMES = (
    "building_ground_interface",
    "door_lower_threshold",
    "door_center",
    "facade_center",
)
TERRAIN_EDGE_SUBJECT_MARGIN_NDC = 0.04

# Final image gates are intentionally stronger than the old non-black check.
# They are statistics on the rendered pixels, not semantic recolouring or a
# substitute for the required manual inspection of every PNG.
MIN_FOREGROUND_LUMINANCE_MEAN = 0.115
MIN_FOREGROUND_LUMINANCE_P25 = 0.082
MAX_DARK_FOREGROUND_FRACTION = 0.55
MIN_FOCUSED_CONTENT_SPAN = 0.16


class RenderContractError(RuntimeError):
    """A required T1.5 input, transform, asset, or audit gate failed."""


@dataclass(frozen=True)
class RenderInputPaths:
    """Pinned source products consumed by one render build.

    The dataclass is intentionally path-only.  The large decoded LAND payload
    stays in the T1.3 JSON/binary products and is not duplicated in the scene
    contract.
    """

    workspace_root: Path
    plan: Path
    validation: Path
    placement: Path
    placement_manifest: Path | None
    t1_3_manifest: Path | None
    t1_3_validation: Path | None
    land_records: Path
    final_field: Path
    final_field_metadata: Path
    procgen_config: Path


def default_render_input_paths(workspace_root: Path | str) -> RenderInputPaths:
    """Return the canonical synthetic T1.5 input bundle.

    These defaults point to accepted synthetic products only.  A future real
    city run must pass its own explicit downstream products; this helper never
    creates or discovers a Falkreath plan.
    """

    root = Path(workspace_root).resolve()
    phase = root / "output" / "cityforge" / "phase1"
    t12 = phase / "t1_2_placement_fixture"
    t13 = phase / "t1_3_cityscape_fixture"
    final = t13 / "t1_2_final_reseat"
    return RenderInputPaths(
        workspace_root=root,
        plan=t12 / "synthetic_not_a_falkreath_design.city_plan.json",
        validation=t12 / "synthetic_not_a_falkreath_design.validation.json",
        placement=final / "city_placement.json",
        placement_manifest=final / "manifest.json",
        t1_3_manifest=t13 / "manifest.json",
        t1_3_validation=t13 / "validation.json",
        land_records=t13 / "land_records.json",
        final_field=t13 / "final_terrain_field.npz",
        final_field_metadata=t13 / "final_terrain_field.metadata.json",
        procgen_config=root / "configs" / "procgen.json",
    )


def sha256_file(path: Path | str) -> str:
    """Hash one file without changing it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    """Return the stable JSON identity encoding used by T1.5 manifests."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RenderContractError(f"value is not canonical JSON: {exc}") from exc


def write_json(path: Path | str, value: Any) -> None:
    """Write deterministic, LF-terminated JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )


def _read_json(path: Path, label: str) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderContractError(f"cannot read {label} {path}: {exc}") from exc
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RenderContractError(message)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RenderContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RenderContractError(f"{label} must be finite")
    return result


def _triplet(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RenderContractError(f"{label} must contain three numbers")
    return tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _normalize_zero(value: float) -> float:
    return 0.0 if float(value) == 0.0 else float(value)


def _round_euler(values: Sequence[float]) -> list[float]:
    return [_normalize_zero(round(float(value), engine_transform.BLENDER_EULER_SERIALIZATION_DIGITS)) for value in values]


def _blender_xyz_matrix(euler: Sequence[float]) -> np.ndarray:
    """Recompose Blender's column-vector XYZ scene Euler convention.

    ``engine_transform`` owns the decomposition.  Keeping this tiny
    recomposition here lets the ordinary-Python host perform the same
    serialized 1e-7 gate before Blender is started.
    """

    if len(euler) != 3:
        raise RenderContractError("Blender XYZ Euler must contain three values")
    x, y, z = (float(value) for value in euler)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), dtype=np.float64)
    ry = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=np.float64)
    rz = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    return rz @ (ry @ rx)


def matrix_max_error(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    """Return the maximum absolute element difference for two 3x3/4x4 values."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape not in {(3, 3), (4, 4)}:
        raise RenderContractError(f"matrix shapes do not agree: {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b)))


def _world_to_relative_matrix(
    position_world_gu: Sequence[float],
    rotation: Sequence[float],
    scale: float,
    render_origin_gu: Sequence[float],
) -> np.ndarray:
    """Build the authoritative placement matrix in the local render frame."""

    absolute = np.asarray(
        engine_transform.placement_scene_matrix(position_world_gu, rotation, scale),
        dtype=np.float64,
    )
    origin = np.asarray(_triplet((*render_origin_gu, 0.0), "render origin"), dtype=np.float64) if len(render_origin_gu) == 2 else np.asarray(_triplet(render_origin_gu, "render origin"), dtype=np.float64)
    absolute[:3, 3] -= SCENE_UNITS_PER_GAME_UNIT * origin[:3]
    return absolute


def _decode_vtex(record: Mapping[str, Any]) -> np.ndarray:
    blob = record.get("texture_indices")
    if not isinstance(blob, Mapping):
        raise RenderContractError(f"LAND {record.get('grid')} has no texture_indices")
    try:
        payload = tes3json.decode_blob(blob.get("data"))
        values = struct.unpack("<256H", payload)
        normalized = espland.transpose_vtex_serialized_to_openmw(values)
    except (TypeError, ValueError, struct.error) as exc:
        raise RenderContractError(f"LAND {record.get('grid')} VTEX decode failed: {exc}") from exc
    return np.asarray(normalized, dtype=np.uint16).reshape((TILE_SIDE, TILE_SIDE))


def _land_record_index(document: Sequence[Mapping[str, Any]]) -> tuple[dict[tuple[int, int], Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    headers = [row for row in document if row.get("type") == "Header"]
    ltex = [row for row in document if row.get("type") == "LandscapeTexture"]
    lands = [row for row in document if row.get("type") == "Landscape"]
    _require(len(headers) == 1, f"T1.3 LAND product must contain one Header, got {len(headers)}")
    _require(headers[0].get("masters") == [], "T1.3 render terrain must remain masterless (masters: [])")
    _require(len(ltex) == 7, f"T1.3 final LAND product must contain 7 local LTEX records, got {len(ltex)}")
    _require(len(lands) == 49, f"T1.3 final LAND product must contain 49 LAND records, got {len(lands)}")
    by_cell: dict[tuple[int, int], Mapping[str, Any]] = {}
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in ltex:
        index = int(row.get("index", -1))
        _require(index not in by_index, f"duplicate T1.3 local LTEX index {index}")
        by_index[index] = row
    for row in lands:
        grid = row.get("grid")
        _require(isinstance(grid, list) and len(grid) == 2, "T1.3 LAND grid is malformed")
        cell = (int(grid[0]), int(grid[1]))
        _require(cell not in by_cell, f"duplicate T1.3 LAND cell {cell}")
        by_cell[cell] = row
    return by_cell, by_index


def _load_field(path: Path, metadata_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    _require(path.is_file(), f"final T1.3 terrain field is missing: {path}")
    _require(metadata_path.is_file(), f"final T1.3 terrain metadata is missing: {metadata_path}")
    metadata = _read_json(metadata_path, "T1.3 final terrain metadata")
    _require(isinstance(metadata, dict), "T1.3 terrain metadata must be an object")
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require("height_gu" in archive, "T1.3 final terrain NPZ has no height_gu array")
            values = np.asarray(archive["height_gu"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise RenderContractError(f"cannot load T1.3 final terrain field {path}: {exc}") from exc
    declared_shape = metadata.get("shape")
    if isinstance(declared_shape, list) and len(declared_shape) == 2:
        _require(
            values.shape == (int(declared_shape[0]), int(declared_shape[1])),
            f"T1.3 final terrain shape is {values.shape}, metadata declares {declared_shape}",
        )
    _require(values.ndim == 2, f"T1.3 final terrain field must be 2D, got {values.shape}")
    _require(np.isfinite(values).all(), "T1.3 final terrain contains non-finite heights")
    return values, metadata


def terrain_field_content_hash(values_gu: np.ndarray) -> str:
    """Hash the canonical float64 field values exactly as T1.3 does."""

    return hashlib.sha256(np.ascontiguousarray(np.asarray(values_gu, dtype="<f8")).tobytes(order="C")).hexdigest()


def _validate_terrain_products(
    paths: RenderInputPaths,
    *,
    scratch_plugin: Path | None,
) -> tuple[dict[str, Any], dict[tuple[int, int], Mapping[str, Any]], dict[int, Mapping[str, Any]], np.ndarray]:
    """Validate T1.3 field/LAND/LTEX products and optionally their scratch ESP."""

    field, metadata = _load_field(paths.final_field, paths.final_field_metadata)
    field_hash = terrain_field_content_hash(field)
    _require(metadata.get("pass") == "final", "T1.5 must consume the T1.3 final terrain pass")
    _require(metadata.get("shape") == [int(v) for v in field.shape], "T1.3 final terrain metadata shape disagrees with the field array")
    _require(metadata.get("spacing_gu") == [FIELD_SPACING_GU, FIELD_SPACING_GU], "T1.3 final terrain spacing is not 128 GU")
    _require(metadata.get("terrain_field_sha256") == field_hash, "T1.3 final field content hash disagrees with metadata")
    cells_meta = metadata.get("cells")
    _require(isinstance(cells_meta, list) and cells_meta, "T1.3 final terrain metadata has no cell list")
    cells = tuple(sorted((int(row[0]), int(row[1])) for row in cells_meta if isinstance(row, list) and len(row) == 2))
    _require(len(cells) == len(cells_meta), "T1.3 final terrain metadata contains malformed cells")
    _require(int(metadata.get("cell_count", -1)) == len(cells), "T1.3 final terrain metadata cell_count disagrees with its cell list")
    xs = sorted({cell[0] for cell in cells})
    ys = sorted({cell[1] for cell in cells})
    _require(
        xs == list(range(xs[0], xs[-1] + 1)) and ys == list(range(ys[0], ys[-1] + 1))
        and len(set(cells)) == len(xs) * len(ys),
        "T1.3 final terrain cells are not a contiguous rectangular block",
    )
    expected_field_shape = (len(ys) * 64 + 1, len(xs) * 64 + 1)
    _require(
        tuple(int(v) for v in field.shape) == expected_field_shape,
        f"T1.3 final terrain shape {field.shape} disagrees with the {len(xs)}x{len(ys)} cell block",
    )

    document = _read_json(paths.land_records, "T1.3 land_records.json")
    _require(isinstance(document, list), "T1.3 land_records.json must be a top-level array")
    issues = tes3json.validate(document)
    _require(not issues, "T1.3 land_records.json fails tes3json.validate: " + "; ".join(map(str, issues[:5])))
    land_by_cell, ltex_by_index = _land_record_index(document)
    _require(set(land_by_cell) == set(cells), "T1.3 LAND grid set does not equal final field metadata")
    _require(sorted(ltex_by_index) == [0, 32, 77, 91, 141, 143, 240], "T1.3 local LTEX indices drifted")

    record_height_mismatches = 0
    record_height_max_error = 0.0
    record_normal_bytes = 0
    raw_values: set[int] = set()
    expected_quantized_field = np.rint(field / 8.0) * 8.0
    for cell in cells:
        row = land_by_cell[cell]
        decoded = np.asarray(tes3json.decode_land_heights(row, game_units=True), dtype=np.float64)
        x0 = (cell[0] - xs[0]) * 64
        y0 = (cell[1] - ys[0]) * 64
        expected = expected_quantized_field[y0 : y0 + LAND_SIDE, x0 : x0 + LAND_SIDE]
        _require(decoded.shape == (LAND_SIDE, LAND_SIDE), f"T1.3 LAND {cell} decoded to {decoded.shape}")
        delta = np.abs(decoded - expected)
        record_height_max_error = max(record_height_max_error, float(delta.max()))
        record_height_mismatches += int(np.count_nonzero(delta != 0.0))
        normals = row.get("vertex_normals")
        _require(isinstance(normals, Mapping), f"T1.3 LAND {cell} has no VNML payload")
        normal_bytes = tes3json.decode_blob(normals.get("data"))
        _require(len(normal_bytes) == LAND_SIDE * LAND_SIDE * 3, f"T1.3 LAND {cell} has incomplete VNML")
        record_normal_bytes += len(normal_bytes)
        raw_values.update(int(value) for value in _decode_vtex(row).reshape(-1) if int(value) > 0)
    _require(record_height_mismatches == 0, f"T1.3 LAND/field height mismatch count is {record_height_mismatches}")
    _require(raw_values and all(raw - 1 in ltex_by_index for raw in raw_values), "T1.3 LAND has a positive VTEX value without local LTEX closure")

    binary_land_count = None
    binary_ltex_count = None
    binary_height_mismatches = None
    if scratch_plugin is not None:
        _require(scratch_plugin.is_file(), f"render-only scratch terrain plugin is missing: {scratch_plugin}")
        binary_land = espland.load_land(scratch_plugin, max_seconds=180.0)
        binary_ltex = espland.load_ltex(scratch_plugin, max_seconds=180.0)
        binary_land_count = len(binary_land)
        binary_ltex_count = len(binary_ltex)
        _require(binary_land_count == len(cells), f"scratch terrain plugin decoded {binary_land_count} LAND cells, expected {len(cells)}")
        _require(binary_ltex_count == len(ltex_by_index), f"scratch terrain plugin decoded {binary_ltex_count} LTEX records, expected {len(ltex_by_index)}")
        binary_height_mismatches = 0
        for cell in cells:
            record = binary_land.get(cell)
            _require(record is not None and record.heights_thu is not None, f"scratch terrain plugin lacks LAND {cell}")
            expected_thu = np.rint(expected_quantized_field[(cell[1] - ys[0]) * 64 : (cell[1] - ys[0]) * 64 + LAND_SIDE, (cell[0] - xs[0]) * 64 : (cell[0] - xs[0]) * 64 + LAND_SIDE] / 8.0).astype(np.int64)
            binary_values = np.asarray(record.heights_thu, dtype=np.int64)
            binary_height_mismatches += int(np.count_nonzero(binary_values != expected_thu))
        _require(binary_height_mismatches == 0, f"scratch terrain plugin LAND height mismatch count is {binary_height_mismatches}")
        _require(sorted(binary_ltex) == sorted(ltex_by_index), "scratch terrain plugin local LTEX indices disagree with T1.3 JSON")

    audit = {
        "required": True,
        "render_mode": "opaque_exact_t1_3_final_field",
        "cells_expected": len(cells),
        "cells_emitted": len(land_by_cell),
        "cells": [list(cell) for cell in cells],
        "field_shape": [int(v) for v in field.shape],
        "field_spacing_gu": [FIELD_SPACING_GU, FIELD_SPACING_GU],
        "field_hash": field_hash,
        "field_npz_sha256": sha256_file(paths.final_field),
        "field_metadata_sha256": sha256_file(paths.final_field_metadata),
        "land_records_sha256": sha256_file(paths.land_records),
        "local_ltex_count": len(ltex_by_index),
        "local_ltex_indices": sorted(ltex_by_index),
        "positive_raw_values": sorted(raw_values),
        "record_height_mismatch_count": record_height_mismatches,
        "record_height_max_error_gu": record_height_max_error,
        "record_vnml_payload_bytes": record_normal_bytes,
        "binary_land_count": binary_land_count,
        "binary_ltex_count": binary_ltex_count,
        "binary_height_mismatch_count": binary_height_mismatches,
        "normal_source": "T1.3 final VNML payload provenance plus final field geometry",
        "water_policy": "clip final-field terrain triangles at z=0; no plane or skirt",
    }
    return audit, land_by_cell, ltex_by_index, field


def _validate_t1_3_manifest(paths: RenderInputPaths, terrain_audit: Mapping[str, Any]) -> None:
    if paths.t1_3_manifest is None or not paths.t1_3_manifest.is_file():
        return
    manifest = _read_json(paths.t1_3_manifest, "T1.3 manifest")
    _require(isinstance(manifest, Mapping), "T1.3 manifest must be an object")
    output_hashes = manifest.get("output_hashes")
    _require(isinstance(output_hashes, Mapping), "T1.3 manifest has no output_hashes")
    for name, actual in (
        ("final_terrain_field.npz", terrain_audit["field_npz_sha256"]),
        ("final_terrain_field.metadata.json", terrain_audit["field_metadata_sha256"]),
        ("land_records.json", terrain_audit["land_records_sha256"]),
    ):
        _require(output_hashes.get(name) == actual, f"T1.3 manifest hash mismatch for {name}")
    source_hashes = manifest.get("source_hashes")
    _require(isinstance(source_hashes, Mapping), "T1.3 manifest has no source_hashes")
    _require(source_hashes.get("tamriel_esm") is not None, "T1.3 manifest does not pin tamriel.esm provenance")


def _validate_t1_1_and_placement(
    paths: RenderInputPaths,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], tuple[float, float], str]:
    plan = _read_json(paths.plan, "T1.1 city plan")
    validation = _read_json(paths.validation, "T1.1 validation")
    placement = _read_json(paths.placement, "T1.2 final placement")
    _require(isinstance(plan, Mapping), "T1.1 plan must be an object")
    _require(isinstance(validation, Mapping), "T1.1 validation must be an object")
    _require(isinstance(placement, Mapping), "T1.2 placement must be an object")
    plan_hash = sha256_file(paths.plan)
    _require(validation.get("valid") is True and int(validation.get("error_count", -1)) == 0, "T1.5 requires a zero-error T1.1 validation")
    if validation.get("plan_file_sha256") is not None:
        _require(validation.get("plan_file_sha256") == plan_hash, "T1.1 validation plan_file_sha256 disagrees with the plan")
    _require(placement.get("product") == "cityforge_t1_2_houses_only_placement", "input is not a T1.2 placement product")
    _require(placement.get("plan_sha256") == plan_hash, "T1.2 placement plan hash disagrees with the consumed T1.1 plan")
    plan_id = str(placement.get("plan_id", ""))
    _require(plan_id and plan.get("plan_id") == plan_id, "T1.1/T1.2 plan ids disagree")
    counts = placement.get("counts")
    _require(isinstance(counts, Mapping), "T1.2 placement has no counts")
    _require(int(counts.get("rejected", -1)) == 0 and int(counts.get("provisional", -1)) == 0, "T1.5 cannot render rejected or provisional T1.2 lots")
    frame = placement.get("frame")
    _require(isinstance(frame, Mapping), "T1.2 placement has no frame")
    origin = frame.get("origin_gu")
    _require(isinstance(origin, list) and len(origin) == 2, "T1.2 frame origin_gu is malformed")
    render_origin = (_finite(origin[0], "frame.origin_gu[0]"), _finite(origin[1], "frame.origin_gu[1]"))
    raw_lots = placement.get("placements")
    _require(isinstance(raw_lots, list) and raw_lots, "T1.2 placement has no accepted lots")
    lots = sorted((dict(lot) for lot in raw_lots if isinstance(lot, Mapping)), key=lambda lot: str(lot.get("lot_id")))
    _require(len(lots) == len(raw_lots), "T1.2 placement contains a malformed lot")
    return dict(plan), dict(validation), lots, render_origin, plan_hash


def _road_length(polyline: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(tuple(map(float, polyline[index])), tuple(map(float, polyline[index + 1]))) for index in range(len(polyline) - 1))


def _road_midpoint(polyline: Sequence[Sequence[float]]) -> tuple[float, float]:
    first = tuple(float(value) for value in polyline[0])
    last = tuple(float(value) for value in polyline[-1])
    return ((first[0] + last[0]) / 2.0, (first[1] + last[1]) / 2.0)


def terrain_height_scene(field: np.ndarray, x_scene: float, y_scene: float) -> float | None:
    """Bilinearly sample the exact final T1.3 field in render scene units.

    The Blender worker uses the same sampler for its independent terrain LOS
    gate.  ``x_scene``/``y_scene`` are local render coordinates, so the field
    origin is ``(0, 0)`` and adjacent final-field vertices are 1.28 scene
    units apart.  Returning ``None`` instead of clamping is important: a
    candidate outside the finite 49-cell field must be rejected, never made
    to look valid by sampling the nearest border value.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise RenderContractError(f"terrain LOS field must be 2D, got {values.shape}")
    field_h, field_w = values.shape
    spacing_scene = FIELD_SPACING_GU * SCENE_UNITS_PER_GAME_UNIT
    fx = float(x_scene) / spacing_scene
    fy = float(y_scene) / spacing_scene
    edge_epsilon = 1.0e-9
    if fx < -edge_epsilon or fy < -edge_epsilon or fx > field_w - 1 + edge_epsilon or fy > field_h - 1 + edge_epsilon:
        return None
    # The finite perimeter is a valid sample.  Clamp only floating-point
    # round-off at that exact boundary; genuine out-of-field points still
    # return None above rather than silently sampling a border value.
    fx = min(field_w - 1.0, max(0.0, fx))
    fy = min(field_h - 1.0, max(0.0, fy))
    x0 = min(field_w - 2, max(0, int(math.floor(fx))))
    y0 = min(field_h - 2, max(0, int(math.floor(fy))))
    tx = fx - x0
    ty = fy - y0
    x1 = x0 + 1
    y1 = y0 + 1
    lower = float(values[y0, x0]) * (1.0 - tx) + float(values[y0, x1]) * tx
    upper = float(values[y1, x0]) * (1.0 - tx) + float(values[y1, x1]) * tx
    return (lower * (1.0 - ty) + upper * ty) * SCENE_UNITS_PER_GAME_UNIT


def terrain_edge_clearance_scene(
    x_scene: float,
    y_scene: float,
    *,
    span_x_scene: float | None = None,
    span_y_scene: float | None = None,
) -> float:
    """Return horizontal clearance from the finite final-field rectangle.

    Defaults to the Falkreath 7x7 span for backwards compatibility; callers
    rendering a different site pass its actual spans (scene units).
    """

    default_span = (FIELD_SIDE - 1) * FIELD_SPACING_GU * SCENE_UNITS_PER_GAME_UNIT
    span_x = default_span if span_x_scene is None else float(span_x_scene)
    span_y = default_span if span_y_scene is None else float(span_y_scene)
    return min(float(x_scene), float(y_scene), span_x - float(x_scene), span_y - float(y_scene))


def terrain_line_of_sight(
    field: np.ndarray,
    camera: Sequence[float],
    target: Sequence[float],
    *,
    samples: int = STREET_LOS_SAMPLE_COUNT,
    minimum_clearance: float = STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE,
) -> dict[str, Any]:
    """Measure terrain clearance along one camera-to-subject segment.

    This is a height-field ray test against the exact T1.3 final field.  The
    endpoints are included so a target sunk into or outside the finite field
    cannot pass merely because every interior sample is clear.  The result is
    evidence only; callers must enforce ``passed`` as a hard gate.
    """

    start = tuple(float(value) for value in camera)
    end = tuple(float(value) for value in target)
    _require(len(start) == 3 and len(end) == 3, "terrain LOS endpoints must be three-dimensional")
    count = max(8, int(samples))
    minimum_value = float("inf")
    minimum_t = 0.0
    outside_count = 0
    for index in range(count + 1):
        fraction = index / float(count)
        point = tuple(start[axis] + (end[axis] - start[axis]) * fraction for axis in range(3))
        height = terrain_height_scene(field, point[0], point[1])
        if height is None:
            outside_count += 1
            continue
        clearance = point[2] - height
        if clearance < minimum_value:
            minimum_value = clearance
            minimum_t = fraction
    if outside_count or not math.isfinite(minimum_value):
        return {
            "passed": False,
            "occluded": False,
            "outside_field_count": outside_count,
            "minimum_clearance_scene": None,
            "minimum_clearance_t": None,
            "sample_count": count + 1,
            "reason": "segment_leaves_finite_terrain_field",
        }
    passed = minimum_value >= float(minimum_clearance)
    return {
        "passed": passed,
        "occluded": not passed,
        "outside_field_count": 0,
        "minimum_clearance_scene": round(minimum_value, 9),
        "minimum_clearance_t": round(minimum_t, 9),
        "sample_count": count + 1,
        "required_clearance_scene": float(minimum_clearance),
        "reason": None if passed else "terrain_intersects_camera_to_subject_segment",
    }


def _rotate_xy(vector: tuple[float, float], angle_rad: float) -> tuple[float, float]:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return (vector[0] * cosine - vector[1] * sine, vector[0] * sine + vector[1] * cosine)


def _street_camera_candidate_specs(
    door: tuple[float, float, float],
    road: tuple[float, float, float],
    tangent: tuple[float, float],
) -> list[dict[str, Any]]:
    """Return ordered deterministic XY candidates anchored to the road.

    The first family stays at the measured face-road closest point and walks
    along the measured road tangent.  The second family samples the access
    ray and fixed angular offsets around it.  No world-axis camera is added:
    every candidate remains derived from the accepted door/road geometry.
    """

    access_x = road[0] - door[0]
    access_y = road[1] - door[1]
    access_length = math.hypot(access_x, access_y)
    _require(access_length > 1.0e-6, "street camera access heading is zero")
    access = (access_x / access_length, access_y / access_length)
    tangent_length = math.hypot(tangent[0], tangent[1])
    _require(tangent_length > 1.0e-6, "street camera road tangent is zero")
    tangent_unit = (tangent[0] / tangent_length, tangent[1] / tangent_length)
    specifications: list[dict[str, Any]] = []
    for offset in (0.0, -8.0, 8.0, -16.0, 16.0, -24.0, 24.0):
        specifications.append(
            {
                "candidate_id": f"road_anchor_tangent_{offset:+05.1f}",
                "family": "face_road_tangent",
                "x_scene": road[0] + tangent_unit[0] * offset,
                "y_scene": road[1] + tangent_unit[1] * offset,
                "road_tangent_offset_scene": offset,
                "access_angle_offset_deg": 0.0,
            }
        )
    # Keep the candidate cardinality stable for the audit contract while
    # covering the access approach, both flanks, and rear-quarter terrain
    # alternatives.  The old forward-only cone could leave every accepted
    # door view behind the same foreground ridge.  The ordering is part of the
    # deterministic tie-break evidence.
    for factor in (0.65, 0.85, 1.0, 1.45, 2.0):
        for angle_deg in (0.0, -30.0, 30.0, -60.0, 60.0):
            direction = _rotate_xy(access, math.radians(angle_deg))
            specifications.append(
                {
                    "candidate_id": f"access_{factor:.2f}_{angle_deg:+04.0f}",
                    "family": "access_heading_offset",
                    "x_scene": door[0] + direction[0] * access_length * factor,
                    "y_scene": door[1] + direction[1] * access_length * factor,
                    "road_tangent_offset_scene": 0.0,
                    "access_angle_offset_deg": angle_deg,
                }
            )
    # Preserve the original near-road family and add a deterministic set of
    # farther escape candidates.  A rear-quarter view can place the opaque
    # terrain surface behind a below-grade foundation instead of projecting its
    # underside against the sky, while the original fit-safe candidates remain
    # available as deterministic fallbacks.
    for factor in (2.0, 3.0, 4.0):
        for angle_deg in (-90.0, 90.0, 180.0):
            direction = _rotate_xy(access, math.radians(angle_deg))
            specifications.append(
                {
                    "candidate_id": f"rear_access_{factor:.2f}_{angle_deg:+04.0f}",
                    "family": "far_side_access_escape",
                    "x_scene": door[0] + direction[0] * access_length * factor,
                    "y_scene": door[1] + direction[1] * access_length * factor,
                    "road_tangent_offset_scene": 0.0,
                    "access_angle_offset_deg": angle_deg,
                }
            )
    return specifications


def _select_street_camera_candidate(
    field: np.ndarray,
    *,
    door: tuple[float, float, float],
    road: tuple[float, float, float],
    tangent: tuple[float, float],
) -> dict[str, Any]:
    """Select a terrain-clear, finite-field street camera deterministically.

    Host selection is an input-side hard gate and evidence producer.  The
    Blender worker repeats the same terrain test after importing the actual
    NIF group, adding the measured facade bounds before it accepts the same
    candidate for rendering.
    """

    door_terrain = terrain_height_scene(field, door[0], door[1])
    _require(door_terrain is not None, "street door anchor is outside the finite terrain field")
    targets = {
        "building_ground_interface": (
            door[0],
            door[1],
            max(door[2] + STREET_TARGET_OFFSETS_SCENE[0], float(door_terrain) + STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE + 0.05),
        ),
        "door_lower_threshold": (door[0], door[1], door[2] + STREET_DOOR_LOS_FLOOR_OFFSET_SCENE),
        "door_center": (door[0], door[1], door[2] + STREET_TARGET_OFFSETS_SCENE[1]),
        "facade_center": (door[0], door[1], door[2] + STREET_TARGET_OFFSETS_SCENE[2]),
    }
    candidate_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for order, specification in enumerate(_street_camera_candidate_specs(door, road, tangent)):
        x_scene = float(specification["x_scene"])
        y_scene = float(specification["y_scene"])
        terrain_height = terrain_height_scene(field, x_scene, y_scene)
        reasons: list[str] = []
        edge_clearance = min(
            terrain_edge_clearance_scene(x_scene, y_scene),
            terrain_edge_clearance_scene(door[0], door[1]),
        )
        if terrain_height is None:
            reasons.append("camera_outside_finite_terrain_field")
        if edge_clearance < STREET_EDGE_MARGIN_SCENE:
            reasons.append("camera_or_subject_too_close_to_terrain_edge")
        los_rows: dict[str, Any] = {}
        minimum_clearance = float("inf")
        if terrain_height is not None:
            camera = (x_scene, y_scene, terrain_height + STREET_EYE_HEIGHT_SCENE)
            for name, target in targets.items():
                required_clearance = (
                    STREET_GROUND_INTERFACE_LOS_CLEARANCE_SCENE
                    if name == "building_ground_interface"
                    else STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE
                )
                los = terrain_line_of_sight(field, camera, target, minimum_clearance=required_clearance)
                los_rows[name] = los
                value = los.get("minimum_clearance_scene")
                if isinstance(value, (int, float)):
                    minimum_clearance = min(minimum_clearance, float(value))
                if not bool(los.get("passed")):
                    reasons.append(f"terrain_occludes_{name}")
        else:
            camera = (x_scene, y_scene, None)
        passed = not reasons
        row = {
            **specification,
            "order": order,
            "camera_position_scene": [round(x_scene, 9), round(y_scene, 9), None if camera[2] is None else round(float(camera[2]), 9)],
            "terrain_height_scene": None if terrain_height is None else round(float(terrain_height), 9),
            "terrain_edge_clearance_scene": round(float(edge_clearance), 9),
            "targets_scene": {name: [round(float(value), 9) for value in target] for name, target in targets.items()},
            "terrain_los": los_rows,
            "minimum_terrain_los_clearance_scene": None if not math.isfinite(minimum_clearance) else round(minimum_clearance, 9),
            "passed": passed,
            "rejection_reasons": sorted(set(reasons)),
        }
        candidate_rows.append(row)
        if passed:
            accepted.append(row)
    _require(accepted, "no deterministic street camera candidate has clear terrain LOS and finite-edge clearance")
    _require(
        len(candidate_rows) == STREET_CAMERA_CANDIDATE_COUNT,
        f"street camera candidate family drifted: {len(candidate_rows)} != {STREET_CAMERA_CANDIDATE_COUNT}",
    )
    selected = max(
        accepted,
        key=lambda row: (
            float(row["terrain_height_scene"]),
            float(row["minimum_terrain_los_clearance_scene"]),
            float(row["terrain_edge_clearance_scene"]),
            -math.dist((float(row["x_scene"]), float(row["y_scene"])), (road[0], road[1])),
            -int(row["order"]),
        ),
    )
    return {
        "version": VIEW_CONTRACT_VERSION,
        "eye_height_scene": STREET_EYE_HEIGHT_SCENE,
        "required_minimum_clearance_scene": STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE,
        "required_ground_interface_clearance_scene": STREET_GROUND_INTERFACE_LOS_CLEARANCE_SCENE,
        "required_edge_margin_scene": STREET_EDGE_MARGIN_SCENE,
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_minimum_terrain_los_clearance_scene": selected["minimum_terrain_los_clearance_scene"],
        "candidate_count": len(candidate_rows),
        "rejected_candidate_count": sum(1 for row in candidate_rows if not row["passed"]),
        "candidates": candidate_rows,
    }


def _build_street_views(
    plan: Mapping[str, Any],
    lots: Sequence[Mapping[str, Any]],
    ref_rows: Sequence[Mapping[str, Any]],
    *,
    render_origin_gu: tuple[float, float],
    field: np.ndarray,
    all_lots: bool = False,
) -> list[dict[str, Any]]:
    """Build deterministic street-camera contracts from actual door/road anchors.

    The synthetic plan has six streets sharing one measured external exit
    reference rather than an internal crossing.  That shared external reference
    is recorded as the fixture's graph junction; no arbitrary world-axis camera
    is invented.  The normal result is the six selected base street views;
    ``all_lots=True`` returns every lot contract for focused detail views.
    """

    roads = [dict(row) for row in plan.get("roads", []) if isinstance(row, Mapping)]
    roads_by_id = {str(row.get("road_id")): row for row in roads}
    ref_by_key = {str(row["ref_key"]): row for row in ref_rows}
    degree: dict[str, int] = {}
    for road in roads:
        for external in road.get("connects", []):
            degree[str(external)] = degree.get(str(external), 0) + 1
    lot_rows: list[dict[str, Any]] = []
    for lot in lots:
        lot_id = str(lot.get("lot_id"))
        members = lot.get("members")
        _require(isinstance(members, list), f"lot {lot_id} has no members")
        doors = [member for member in members if isinstance(member, Mapping) and member.get("is_door") is True]
        _require(len(doors) == 1, f"lot {lot_id} must have exactly one renderable door anchor")
        door = doors[0]
        source_id = str(door.get("source_id"))
        ref_key = next((str(row["ref_key"]) for row in ref_rows if row.get("lot_id") == lot_id and row.get("source_id") == source_id), None)
        _require(ref_key is not None, f"lot {lot_id} door {source_id} is absent from render refs")
        access = lot.get("checks", {}).get("road_access", {}) if isinstance(lot.get("checks"), Mapping) else {}
        road_id = str(access.get("road_id", ""))
        road = roads_by_id.get(road_id)
        _require(road is not None, f"lot {lot_id} door has no declared road anchor {road_id!r}")
        closest = access.get("closest_point_plan_gu")
        _require(isinstance(closest, list) and len(closest) == 2, f"lot {lot_id} road access has no closest road anchor")
        door_world = tuple(float(value) for value in door.get("world_position_gu", ()))
        _require(len(door_world) == 3, f"lot {lot_id} door world position is malformed")
        road_anchor_world = (
            render_origin_gu[0] + float(closest[0]),
            render_origin_gu[1] + float(closest[1]),
            door_world[2],
        )
        polyline = road.get("polyline", [])
        _require(isinstance(polyline, list) and len(polyline) >= 2, f"road {road_id} has no usable polyline tangent")
        tangent = (
            float(polyline[-1][0]) - float(polyline[0][0]),
            float(polyline[-1][1]) - float(polyline[0][1]),
        )
        tangent_length = math.hypot(*tangent)
        _require(tangent_length > 1.0e-6, f"road {road_id} has a zero-length tangent")
        tangent = (tangent[0] / tangent_length, tangent[1] / tangent_length)
        door_scene = (
            (door_world[0] - render_origin_gu[0]) * SCENE_UNITS_PER_GAME_UNIT,
            (door_world[1] - render_origin_gu[1]) * SCENE_UNITS_PER_GAME_UNIT,
            door_world[2] * SCENE_UNITS_PER_GAME_UNIT,
        )
        road_scene = (
            float(closest[0]) * SCENE_UNITS_PER_GAME_UNIT,
            float(closest[1]) * SCENE_UNITS_PER_GAME_UNIT,
            door_scene[2],
        )
        camera_contract = _select_street_camera_candidate(
            field,
            door=door_scene,
            road=road_scene,
            tangent=tangent,
        )
        midpoint = _road_midpoint(road.get("polyline", []))
        neighbour_count = sum(
            1
            for other in roads
            if other is not road
            and math.dist(midpoint, _road_midpoint(other.get("polyline", []))) <= 4096.0
        )
        external_degree = max((degree.get(str(item), 0) for item in road.get("connects", [])), default=0)
        lot_rows.append(
            {
                "lot_id": lot_id,
                "door_ref_key": ref_key,
                "member_ref_keys": [str(row["ref_key"]) for row in ref_rows if row.get("lot_id") == lot_id],
                "road_id": road_id,
                "road_anchor_world_gu": list(road_anchor_world),
                "door_world_gu": list(door_world),
                "door_to_road_distance_gu": float(access.get("door_to_road_distance_gu", 0.0)),
                "road_tangent_scene": [round(float(tangent[0]), 9), round(float(tangent[1]), 9)],
                "road_external_degree": external_degree,
                "road_neighbour_count": neighbour_count,
                "road_length_gu": _road_length(road.get("polyline", [])),
                "kit_style": "karthgad" if str(lot.get("stamp_id", "")).startswith("karthgad") else "markarth",
                "junction_refs": sorted(str(item) for item in road.get("connects", []) if degree.get(str(item), 0) >= 3),
                "street_camera_contract": camera_contract,
            }
        )

    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def choose(rows: Iterable[Mapping[str, Any]], reason: str) -> None:
        for row in sorted(rows, key=lambda item: str(item["lot_id"])):
            key = str(row["door_ref_key"])
            if key in used:
                continue
            used.add(key)
            selected.append({**dict(row), "selection_reason": reason})
            return
        raise RenderContractError(f"unable to select a unique street/door anchor for {reason}")

    choose(
        sorted(
            lot_rows,
            key=lambda row: (
                -int(row["road_external_degree"]),
                -int(row["road_neighbour_count"]),
                float(row["door_to_road_distance_gu"]),
                str(row["lot_id"]),
            ),
        ),
        "densest_street",
    )
    choose(
        sorted(
            (row for row in lot_rows if row["junction_refs"]),
            key=lambda row: (
                -max((degree.get(item, 0) for item in row["junction_refs"]), default=0),
                str(row["lot_id"]),
            ),
        ),
        "junction_shared_external_exit",
    )
    choose((row for row in lot_rows if row["kit_style"] == "markarth"), "kit_style_markarth")
    choose((row for row in lot_rows if row["kit_style"] == "karthgad"), "kit_style_karthgad")
    choose(sorted(lot_rows, key=lambda row: (float(row["door_to_road_distance_gu"]), str(row["lot_id"]))), "additional_door_anchor_1")
    choose(sorted(lot_rows, key=lambda row: (str(row["lot_id"])),), "additional_door_anchor_2")
    _require(len(selected) >= 6, "T1.5 requires at least six street/door-height views")
    if all_lots:
        return sorted(lot_rows, key=lambda row: str(row["lot_id"]))
    return selected[:6]


def _build_views(
    plan: Mapping[str, Any],
    lots: Sequence[Mapping[str, Any]],
    ref_rows: Sequence[Mapping[str, Any]],
    *,
    render_origin_gu: tuple[float, float],
    field: np.ndarray,
    build_hash: str,
) -> list[dict[str, Any]]:
    """Return the base review set plus clearly named sparse-fixture details."""

    short = build_hash[:16]
    views: list[dict[str, Any]] = [
        {
            "view_id": "overview",
            "kind": "overview",
            "camera_mode": "ORTHO",
            "resolution": [3072, 3072],
            "file": f"overview__{short}.png",
            "framing": "focused all-content placement bounds plus immediate approach terrain",
            "focus": "all_city_content",
            "focused": True,
        },
    ]
    for view_id, label, signs in (
        ("oblique_sw_ne", "SW→NE", (1.0, 1.0)),
        ("oblique_se_nw", "SE→NW", (-1.0, 1.0)),
        ("oblique_nw_se", "NW→SE", (1.0, -1.0)),
        ("oblique_ne_sw", "NE→SW", (-1.0, -1.0)),
    ):
        views.append(
            {
                "view_id": view_id,
                "kind": "oblique",
                "label": label,
                "camera_mode": "ORTHO",
                "horizontal_view_direction": [signs[0], signs[1]],
                "resolution": [2048, 1536],
                "file": f"{view_id}__{short}.png",
                "framing": "focused all-content placement bounds plus immediate approach terrain",
                "focus": "all_city_content",
                "focused": True,
            }
        )
    street_rows = _build_street_views(plan, lots, ref_rows, render_origin_gu=render_origin_gu, field=field)
    for index, row in enumerate(street_rows, 1):
        views.append(
            {
                **row,
                "view_id": f"street_{index:02d}_{row['selection_reason']}",
                "kind": "street",
                "camera_mode": "PERSP",
                "resolution": [1600, 1000],
                "file": f"street_{index:02d}_{row['selection_reason']}__{short}.png",
                "framing": "door-height road-facing local city content",
                "focus": "selected_lot",
                "focused": False,
            }
        )

    # Each synthetic lot gets its own focused door-height view.  The lots are
    # deliberately spread over the finite fixture, so grouping them into one
    # orthographic frame made the buildings microscopic or terrain-occluded.
    # These rows reuse the same accepted door/road/tangent candidate contract
    # as the base street views; only the content scope differs.
    all_lot_rows = _build_street_views(
        plan,
        lots,
        ref_rows,
        render_origin_gu=render_origin_gu,
        field=field,
        all_lots=True,
    )
    _require(len(all_lot_rows) == FOCUSED_DETAIL_VIEW_COUNT, f"synthetic focused detail lot count drifted: expected {FOCUSED_DETAIL_VIEW_COUNT}, got {len(all_lot_rows)}")
    for row in all_lot_rows:
        view_id = f"detail_{row['lot_id']}_street"
        views.append(
            {
                **row,
                "view_id": view_id,
                "kind": "detail",
                "camera_mode": "PERSP",
                "resolution": [1600, 1000],
                "file": f"{view_id}__{short}.png",
                "framing": "focused door-height local placement and terrain",
                "focus": "single_lot",
                "focus_lot_ids": [str(row["lot_id"])],
                "focused": True,
            }
        )
    _require(len(views) == REQUIRED_BASE_VIEW_COUNT + len(all_lot_rows), f"T1.5 render set must contain {REQUIRED_BASE_VIEW_COUNT} base views plus one detail per lot, got {len(views)}")
    _require(sum(1 for row in views if row.get("kind") != "detail") == REQUIRED_BASE_VIEW_COUNT, "T1.5 base view count drifted")
    return views


def build_render_scene(
    paths: RenderInputPaths,
    *,
    scratch_plugin: Path,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Validate accepted inputs and build the deterministic Blender scene contract."""

    for path in (
        paths.plan,
        paths.validation,
        paths.placement,
        paths.land_records,
        paths.final_field,
        paths.final_field_metadata,
        paths.procgen_config,
    ):
        _require(path.is_file(), f"required T1.5 input is missing: {path}")
    plan, validation, lots, render_origin_gu, plan_hash = _validate_t1_1_and_placement(paths)
    plan_id = str(plan.get("plan_id", ""))
    terrain_audit, land_by_cell, ltex_by_index, field = _validate_terrain_products(paths, scratch_plugin=scratch_plugin)
    _validate_t1_3_manifest(paths, terrain_audit)
    if paths.placement_manifest is not None and paths.placement_manifest.is_file():
        placement_manifest = _read_json(paths.placement_manifest, "T1.2 final placement manifest")
        expected = placement_manifest.get("output_hashes", {}) if isinstance(placement_manifest, Mapping) else {}
        _require(expected.get("city_placement.json") == sha256_file(paths.placement), "T1.2 final manifest does not pin the consumed city_placement.json")
        _require(placement_manifest.get("terrain_pass") == "final", "T1.2 placement manifest is not the final pass")

    frame_origin_z = 0.0
    ref_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    multi_axis: list[str] = []
    near_gimbal: list[str] = []
    seen_ref_keys: set[str] = set()
    for lot in lots:
        lot_id = str(lot.get("lot_id", ""))
        _require(lot_id, "T1.2 lot has no lot_id")
        _require(str(lot.get("status", "")).startswith("accepted"), f"T1.2 lot {lot_id} is not accepted_final")
        _require(lot.get("anchor", {}).get("field_pass") == "final", f"T1.2 lot {lot_id} is not seated on the final terrain pass")
        members = lot.get("members")
        _require(isinstance(members, list) and members, f"T1.2 lot {lot_id} has no placed members")
        for member_index, member in enumerate(members):
            _require(isinstance(member, Mapping), f"T1.2 lot {lot_id} member {member_index} is malformed")
            source_id = str(member.get("source_id", ""))
            model_key = member.get("model_key")
            record_type = member.get("record_type")
            _require(source_id and isinstance(model_key, str) and model_key.strip(), f"T1.2 lot {lot_id} member {member_index} lacks source_id/model_key")
            _require(record_type in {"STAT", "DOOR", "ACTI"}, f"T1.2 member {source_id} has unsupported record type {record_type!r}")
            position_world = _triplet(member.get("world_position_gu"), f"{lot_id}/{source_id}.world_position_gu")
            raw_rotation = _triplet(member.get("raw_tes3_rotation_rad"), f"{lot_id}/{source_id}.raw_tes3_rotation_rad")
            scale = _finite(member.get("scale", 1.0), f"{lot_id}/{source_id}.scale")
            _require(scale > 0.0, f"{lot_id}/{source_id}.scale must be positive")
            authoritative_rotation = engine_transform.tes3_euler_to_matrix_rounded(raw_rotation)
            blender_euler = engine_transform.blender_xyz_euler_for_tes3_rotation(raw_rotation)
            serialized_euler = _round_euler(blender_euler)
            reconstructed = _blender_xyz_matrix(serialized_euler)
            euler_error = matrix_max_error(reconstructed, authoritative_rotation)
            _require(euler_error <= MATRIX_TOLERANCE, f"{lot_id}/{source_id}: serialized Blender matrix error {euler_error:.3e} exceeds {MATRIX_TOLERANCE:.1e}")
            evidence = member.get("rotation_matrix_3x3")
            evidence_error = None
            if isinstance(evidence, list):
                evidence_error = matrix_max_error(authoritative_rotation, evidence)
                _require(evidence_error <= INPUT_MATRIX_EVIDENCE_TOLERANCE, f"{lot_id}/{source_id}: T1.2 rotation evidence drift {evidence_error:.3e}")
            ref_key = f"{lot_id}__member_{member_index:03d}__{source_id}"
            _require(ref_key not in seen_ref_keys, f"duplicate deterministic render ref key {ref_key}")
            seen_ref_keys.add(ref_key)
            render_origin_3 = (render_origin_gu[0], render_origin_gu[1], frame_origin_z)
            relative_position = [
                _normalize_zero(round((position_world[axis] - render_origin_3[axis]) * SCENE_UNITS_PER_GAME_UNIT, 9))
                for axis in range(3)
            ]
            expected_matrix = _world_to_relative_matrix(
                position_world,
                raw_rotation,
                scale,
                render_origin_3,
            )
            raw_matrix = engine_transform.tes3_euler_to_matrix(raw_rotation)
            if abs(float(raw_rotation[0])) > 1.0e-12 or abs(float(raw_rotation[1])) > 1.0e-12:
                multi_axis.append(ref_key)
            if math.hypot(float(authoritative_rotation[0, 0]), float(authoritative_rotation[1, 0])) <= engine_transform.BLENDER_GIMBAL_CANDIDATE_THRESHOLD:
                near_gimbal.append(ref_key)
            row = {
                "id": ref_key,
                "ref_key": ref_key,
                "lot_id": lot_id,
                "member_index": member_index,
                "source_id": source_id,
                "record_type": record_type,
                "is_door": bool(member.get("is_door", False)),
                "structural_role": member.get("structural_role"),
                "object_id": member.get("object_id"),
                "model_key": model_key,
                "position": relative_position,
                "position_world_gu": list(position_world),
                "rotation": serialized_euler,
                "raw_tes3_rotation_rad": list(raw_rotation),
                "scale": scale,
                "expected_relative_matrix": expected_matrix.tolist(),
                "source_matrix_3x3": raw_matrix.tolist(),
                "matrix_evidence": {
                    "serialized_blender_matrix_error": euler_error,
                    "t1_2_rotation_evidence_error": evidence_error,
                    "authoritative_matrix": authoritative_rotation.tolist(),
                },
                "normalize_to_position": False,
                "scene_units_per_game_unit": SCENE_UNITS_PER_GAME_UNIT,
            }
            ref_rows.append(row)
            matrix_rows.append(
                {
                    "ref_key": ref_key,
                    "lot_id": lot_id,
                    "source_id": source_id,
                    "error": euler_error,
                }
            )

    ref_rows.sort(key=lambda row: str(row["ref_key"]))
    model_keys = sorted({str(row["model_key"]) for row in ref_rows}, key=str.casefold)
    roots = configured_data_roots(paths.procgen_config)
    resolver = AssetResolver(roots=roots)
    resolved_models: dict[str, str] = {}
    for model_key in model_keys:
        resolved = resolver.resolve(model_key, "mesh")
        _require(resolved is not None and Path(resolved).is_file(), f"placed model cannot resolve under configured roots: {model_key}")
        resolved_models[model_key] = str(Path(resolved).resolve())

    resolved_textures: list[dict[str, Any]] = []
    for index in sorted(ltex_by_index):
        record = ltex_by_index[index]
        file_name = str(record.get("file_name", ""))
        _require(file_name, f"T1.3 local LTEX {index} has no file_name")
        resolved = resolver.resolve(file_name, "texture")
        _require(resolved is not None and Path(resolved).is_file(), f"T1.3 local LTEX texture cannot resolve: {file_name}")
        resolved_textures.append(
            {
                "index": index,
                "id": str(record.get("id", "")),
                "file_name": file_name,
                "resolved_path": str(Path(resolved).resolve()),
            }
        )

    input_hashes = {
        "city_plan": sha256_file(paths.plan),
        "t1_1_validation": sha256_file(paths.validation),
        "t1_2_final_placement": sha256_file(paths.placement),
        "t1_3_land_records": terrain_audit["land_records_sha256"],
        "t1_3_final_field": terrain_audit["field_npz_sha256"],
        "t1_3_final_field_metadata": terrain_audit["field_metadata_sha256"],
        "procgen_config": sha256_file(paths.procgen_config),
    }
    if paths.placement_manifest is not None and paths.placement_manifest.is_file():
        input_hashes["t1_2_final_manifest"] = sha256_file(paths.placement_manifest)
    if paths.t1_3_manifest is not None and paths.t1_3_manifest.is_file():
        input_hashes["t1_3_manifest"] = sha256_file(paths.t1_3_manifest)
    if paths.t1_3_validation is not None and paths.t1_3_validation.is_file():
        input_hashes["t1_3_validation"] = sha256_file(paths.t1_3_validation)
    input_hashes["scratch_terrain_plugin"] = sha256_file(scratch_plugin)
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "view_contract_version": VIEW_CONTRACT_VERSION,
        "required_base_view_count": REQUIRED_BASE_VIEW_COUNT,
        "focused_detail_view_count": FOCUSED_DETAIL_VIEW_COUNT,
        "camera_contract": {
            "street_eye_height_scene": STREET_EYE_HEIGHT_SCENE,
            "street_target_offsets_scene": STREET_TARGET_OFFSETS_SCENE,
            "street_door_los_floor_offset_scene": STREET_DOOR_LOS_FLOOR_OFFSET_SCENE,
            "street_composition_target_raise_scene": STREET_COMPOSITION_TARGET_RAISE_SCENE,
            "minimum_terrain_los_clearance_scene": STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE,
            "street_edge_margin_scene": STREET_EDGE_MARGIN_SCENE,
            "street_los_sample_count": STREET_LOS_SAMPLE_COUNT,
            "street_fit_margin": STREET_FIT_MARGIN,
            "terrain_edge_subject_margin_ndc": TERRAIN_EDGE_SUBJECT_MARGIN_NDC,
            "street_candidate_count": STREET_CAMERA_CANDIDATE_COUNT,
        },
        "render_settings": {
            "engine": "BLENDER_EEVEE_NEXT",
            "world_color": [0.07, 0.09, 0.12, 1.0],
            "world_strength": 1.05,
            "view_look": "AgX - Medium High Contrast",
            "exposure": 1.75,
        },
        "plan_id": plan_id,
        "input_hashes": input_hashes,
        "render_origin_gu": [render_origin_gu[0], render_origin_gu[1], frame_origin_z],
        "scene_units_per_game_unit": SCENE_UNITS_PER_GAME_UNIT,
        "refs": ref_rows,
        "terrain_field_hash": terrain_audit["field_hash"],
        "resolved_models": resolved_models,
        "resolved_textures": resolved_textures,
        "canary_tolerance": MATRIX_TOLERANCE,
    }
    build_hash = hashlib.sha256(canonical_bytes(identity_payload)).hexdigest()
    views = _build_views(
        plan,
        lots,
        ref_rows,
        render_origin_gu=render_origin_gu,
        field=field,
        build_hash=build_hash,
    )
    synthetic_scope = "synthetic_not_a_falkreath_design" if synthetic else "production_render_input"
    banner = SYNTHETIC_BANNER if synthetic else None
    terrain_spec = {
        **dict(terrain_audit),
        "plugin": str(scratch_plugin.resolve()),
        "texture_plugin": str(scratch_plugin.resolve()),
        "land_records_json": str(paths.land_records.resolve()),
        "final_field_npz": str(paths.final_field.resolve()),
        "final_field_metadata": str(paths.final_field_metadata.resolve()),
        "frame_origin_gu": [render_origin_gu[0], render_origin_gu[1], frame_origin_z],
        "cells": [list(cell) for cell in sorted(land_by_cell)],
        "ltex": resolved_textures,
        "require_real_textures": True,
        "opaque": True,
        "height_scale": SCENE_UNITS_PER_GAME_UNIT,
        "scene_units_per_game_unit": SCENE_UNITS_PER_GAME_UNIT,
        "normal_payload_policy": "T1.3 final VNML is audited; smooth normals follow the exact final field geometry",
    }
    scene = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "scene_name": f"Cityforge_T15_{build_hash[:16]}",
        "diagnostic_scope": synthetic_scope,
        "synthetic_banner": banner,
        "plan_id": plan_id,
        "plan_sha256": plan_hash,
        "build_hash": build_hash,
        "view_contract": {
            "version": VIEW_CONTRACT_VERSION,
            "required_base_view_count": REQUIRED_BASE_VIEW_COUNT,
            "focused_detail_view_count": FOCUSED_DETAIL_VIEW_COUNT,
            "required_view_ids": [str(row["view_id"]) for row in views],
            "terrain_los_required": True,
            "terrain_los_target_names": list(STREET_LOS_TARGET_NAMES),
            "terrain_edge_margin_scene": STREET_EDGE_MARGIN_SCENE,
            "minimum_terrain_los_clearance_scene": STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE,
        },
        "render_origin_gu": [render_origin_gu[0], render_origin_gu[1], frame_origin_z],
        "scene_units_per_game_unit": SCENE_UNITS_PER_GAME_UNIT,
        "normalize_to_position": False,
        "input_hashes": dict(sorted(input_hashes.items())),
        "source_paths": {
            "city_plan": str(paths.plan.resolve()),
            "t1_1_validation": str(paths.validation.resolve()),
            "t1_2_final_placement": str(paths.placement.resolve()),
            "t1_3_land_records": str(paths.land_records.resolve()),
            "t1_3_final_field": str(paths.final_field.resolve()),
            "t1_3_final_field_metadata": str(paths.final_field_metadata.resolve()),
            "scratch_terrain_plugin": str(scratch_plugin.resolve()),
            "procgen_config": str(paths.procgen_config.resolve()),
        },
        "import": {
            "scale_correction": SCENE_UNITS_PER_GAME_UNIT,
            "normalize_to_position": False,
            "use_existing_materials": True,
            "ignore_collision_nodes": True,
            "ignore_animations": True,
            "reuse_meshes": True,
            "vertex_precision": 0.001,
        },
        "render": {
            "engine": "BLENDER_EEVEE_NEXT",
            "film_transparent": False,
            "world_color": [0.07, 0.09, 0.12, 1.0],
            "world_strength": 1.05,
            "view_look": "AgX - Medium High Contrast",
            "exposure": 1.75,
            "building_material_policy": "actual image-backed base color, matte roughness 0.88, no specular glare",
            "terrain_material_policy": "actual T1.3 local LTEX images, opaque, no fallback",
            "readability_policy": "neutral exposure and broad matte fill; no semantic color or proxy emission",
        },
        "terrain": terrain_spec,
        "water": {
            "enabled": True,
            "z_scene_units": 0.0,
            "source": "final T1.3 field triangle clipping",
            "clip_policy": "Sutherland-Hodgman clip each final-field terrain triangle to z<=0, then flatten clipped vertices to z=0",
            "rectangular_plane_forbidden": True,
            "external_skirt_forbidden": True,
        },
        "matrix_canaries": {
            "multi_axis_required": True,
            "near_gimbal_required": True,
            "multi_axis_refs": sorted(multi_axis),
            "near_gimbal_refs": sorted(near_gimbal),
            "note": "The renderer preserves and gates these categories when present in the accepted placement product; no unplaced source member is added to satisfy the gate.",
        },
        "counts": {
            "expected_ref_count": len(ref_rows),
            "expected_unique_model_count": len(model_keys),
            "expected_lot_count": len(lots),
            "expected_terrain_cell_count": 49,
            "expected_view_count": len(views),
            "required_base_view_count": REQUIRED_BASE_VIEW_COUNT,
            "focused_detail_view_count": FOCUSED_DETAIL_VIEW_COUNT,
        },
        "resolved_models": resolved_models,
        "resolved_textures": resolved_textures,
        "refs": ref_rows,
        "views": views,
        "identity_payload_sha256": hashlib.sha256(canonical_bytes(identity_payload)).hexdigest(),
    }
    return scene


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def audit_png(
    path: Path,
    expected_size: Sequence[int],
    *,
    view_kind: str | None = None,
    focused: bool = False,
) -> dict[str, Any]:
    """Read one final PNG and enforce nonblank/readability evidence.

    The foreground statistics deliberately compare against the estimated sky
    or neutral corner background.  A bright background alone therefore cannot
    hide a black terrain/building subject, which was the defect in the prior
    T1.5 audit.
    """

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RenderContractError(f"Pillow is required for render image audit: {exc}") from exc
    _require(path.is_file(), f"required render PNG is missing: {path}")
    _require(path.stat().st_size > 1024, f"required render PNG is unexpectedly small: {path}")
    try:
        with Image.open(path) as image:
            actual = tuple(int(value) for value in image.size)
            _require(actual == tuple(int(value) for value in expected_size), f"{path.name} dimensions are {actual}, expected {tuple(expected_size)}")
            sampled = image.convert("RGB")
            sampled.thumbnail((512, 512), Image.Resampling.BOX)
            pixels = list(sampled.getdata())
    except (OSError, ValueError) as exc:
        raise RenderContractError(f"cannot audit render PNG {path}: {exc}") from exc
    _require(pixels, f"render PNG has no pixels: {path}")
    rgb = [tuple(float(channel) / 255.0 for channel in pixel) for pixel in pixels]
    luminance = [0.2126 * row[0] + 0.7152 * row[1] + 0.0722 * row[2] for row in rgb]
    corner_width = max(1, sampled.width // 12)
    corner_indexes: list[int] = []
    for y0 in (0, max(0, sampled.height - corner_width)):
        for x0 in (0, max(0, sampled.width - corner_width)):
            for y in range(y0, min(sampled.height, y0 + corner_width)):
                for x in range(x0, min(sampled.width, x0 + corner_width)):
                    corner_indexes.append(y * sampled.width + x)
    background = [
        _percentile([rgb[index][channel] for index in corner_indexes], 0.5)
        for channel in range(3)
    ]
    foreground = [
        index
        for index, row in enumerate(rgb)
        if max(abs(row[channel] - background[channel]) for channel in range(3)) > 0.018
    ]
    foreground_luminance = [luminance[index] for index in foreground]
    clip_count = sum(1 for row in rgb if max(row) >= 0.996)
    unique_colors = len({tuple(int(channel) for channel in pixel) for pixel in pixels})
    rgb_min = min(min(row) for row in rgb)
    rgb_max = max(max(row) for row in rgb)
    _require(rgb_max - rgb_min > 0.015, f"render PNG has no color variation: {path.name}")
    _require(len(foreground) >= max(32, len(rgb) // 1000), f"render PNG has no visible foreground/terrain content: {path.name}")
    _require(max(luminance) > 0.03, f"render PNG is effectively black: {path.name}")
    foreground_mean = sum(foreground_luminance) / len(foreground_luminance)
    foreground_p25 = _percentile(foreground_luminance, 0.25)
    dark_foreground_fraction = sum(value < 0.08 for value in foreground_luminance) / len(foreground_luminance)
    if view_kind is not None:
        _require(
            foreground_mean >= MIN_FOREGROUND_LUMINANCE_MEAN,
            f"{path.name} foreground is too dark: mean {foreground_mean:.4f} < {MIN_FOREGROUND_LUMINANCE_MEAN:.4f}",
        )
        _require(
            foreground_p25 >= MIN_FOREGROUND_LUMINANCE_P25,
            f"{path.name} foreground readability p25 {foreground_p25:.4f} < {MIN_FOREGROUND_LUMINANCE_P25:.4f}",
        )
        _require(
            dark_foreground_fraction <= MAX_DARK_FOREGROUND_FRACTION,
            f"{path.name} dark foreground fraction {dark_foreground_fraction:.4f} > {MAX_DARK_FOREGROUND_FRACTION:.4f}",
        )
    return {
        "file": path.name,
        "relative_path": path.name,
        "width": actual[0],
        "height": actual[1],
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "sample_width": sampled.width,
        "sample_height": sampled.height,
        "sample_count": len(rgb),
        "unique_sample_colors": unique_colors,
        "nonblank_fraction": round(len(foreground) / len(rgb), 6),
        "sample_rgb_min": round(rgb_min, 6),
        "sample_rgb_max": round(rgb_max, 6),
        "sample_rgb_mean": round(sum(sum(row) for row in rgb) / (len(rgb) * 3), 6),
        "sample_luminance_mean": round(sum(luminance) / len(luminance), 6),
        "sample_luminance_percentiles": {
            key: round(_percentile(luminance, fraction), 6)
            for key, fraction in (("p01", 0.01), ("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p99", 0.99))
        },
        "sample_clip_fraction": round(clip_count / len(rgb), 6),
        "background_rgb_estimate": [round(value, 6) for value in background],
        "foreground_sample_count": len(foreground),
        "foreground_luminance_mean": round(foreground_mean, 6),
        "foreground_luminance_p25": round(foreground_p25, 6),
        "dark_foreground_fraction": round(dark_foreground_fraction, 6),
        "readability_gate": {
            "view_kind": view_kind,
            "focused": bool(focused),
            "minimum_foreground_luminance_mean": MIN_FOREGROUND_LUMINANCE_MEAN,
            "minimum_foreground_luminance_p25": MIN_FOREGROUND_LUMINANCE_P25,
            "maximum_dark_foreground_fraction": MAX_DARK_FOREGROUND_FRACTION,
            "passed": view_kind is None or (
                foreground_mean >= MIN_FOREGROUND_LUMINANCE_MEAN
                and foreground_p25 >= MIN_FOREGROUND_LUMINANCE_P25
                and dark_foreground_fraction <= MAX_DARK_FOREGROUND_FRACTION
            ),
        },
        "tonal_flags": [],
    }


def validate_focused_content_span(content: Mapping[str, Any], view_id: str = "focused view") -> None:
    """Fail closed when a focused camera makes its subject microscopic.

    The Blender worker supplies normalized NDC bounds for the imported lot.
    Keeping this gate as a small host helper makes the acceptance rule directly
    testable without requiring Blender, while ``finalize_render_audit`` still
    consumes the worker's actual post-import measurement.
    """

    _require(isinstance(content, Mapping), f"{view_id}: focused view has no content bounds")
    _require(
        float(content.get("span_width", 0.0)) >= MIN_FOCUSED_CONTENT_SPAN
        and float(content.get("span_height", 0.0)) >= MIN_FOCUSED_CONTENT_SPAN,
        f"{view_id}: focused content coverage is microscopic",
    )


def normalize_png_bytes(path: Path) -> None:
    """Rewrite one Blender PNG without volatile encoder metadata.

    Eevee's pixels are repeatable on the same machine, but Blender writes a
    run-specific PNG chunk in otherwise identical files.  Re-encoding through
    Pillow removes that metadata while preserving the RGB pixels and dimensions
    before the host computes the acceptance hashes.  This is a byte-level
    determinism normalization only; it does not alter rendered content.
    """

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RenderContractError(f"Pillow is required for deterministic render PNG normalization: {exc}") from exc
    _require(path.is_file(), f"cannot normalize missing render PNG: {path}")
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            encoded = io.BytesIO()
            rgb.save(encoded, format="PNG", optimize=False, compress_level=9)
        path.write_bytes(encoded.getvalue())
    except (OSError, ValueError) as exc:
        raise RenderContractError(f"cannot normalize render PNG {path}: {exc}") from exc


def finalize_render_audit(scene: Mapping[str, Any], output_dir: Path, worker_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Independently audit worker output and write the acceptance-facing audit."""

    _require(worker_audit.get("stage") == STAGE, "Blender worker audit has the wrong stage")
    _require(worker_audit.get("build_hash") == scene.get("build_hash"), "Blender worker audit build hash disagrees with host scene")
    expected_views = scene.get("views")
    _require(
        isinstance(expected_views, list)
        and len(expected_views) == REQUIRED_BASE_VIEW_COUNT + FOCUSED_DETAIL_VIEW_COUNT,
        f"scene contract does not contain the required {REQUIRED_BASE_VIEW_COUNT} base views plus {FOCUSED_DETAIL_VIEW_COUNT} detail views",
    )
    image_rows: list[dict[str, Any]] = []
    expected_by_id: dict[str, Mapping[str, Any]] = {}
    for view in expected_views:
        _require(isinstance(view, Mapping), "scene view row is malformed")
        view_id = str(view["view_id"])
        _require(view_id not in expected_by_id, f"duplicate scene view id {view_id}")
        expected_by_id[view_id] = view
        image = audit_png(
            output_dir / str(view["file"]),
            view["resolution"],
            view_kind=str(view["kind"]),
            focused=bool(view.get("focused", False)),
        )
        image["view_id"] = view_id
        image_rows.append(image)
    worker_counts = worker_audit.get("counts")
    _require(isinstance(worker_counts, Mapping), "Blender worker audit has no counts")
    worker_views = worker_audit.get("views")
    _require(isinstance(worker_views, list) and len(worker_views) == len(expected_views), "Blender worker view audit count disagrees with the scene contract")
    worker_by_id = {str(row.get("view_id")): row for row in worker_views if isinstance(row, Mapping)}
    _require(set(worker_by_id) == set(expected_by_id), "Blender worker view ids do not equal the scene contract")
    for view_id, view in expected_by_id.items():
        worker_view = worker_by_id[view_id]
        camera = worker_view.get("camera")
        _require(isinstance(camera, Mapping), f"{view_id}: worker view has no camera audit")
        if str(view["kind"]) in {"street", "detail"} and isinstance(view.get("street_camera_contract"), Mapping):
            los = worker_view.get("terrain_los")
            _require(isinstance(los, Mapping), f"{view_id}: worker omitted terrain LOS audit")
            _require(bool(los.get("passed")) is True, f"{view_id}: terrain LOS audit failed")
            _require(
                set(str(name) for name in los.get("target_names", [])) == set(STREET_LOS_TARGET_NAMES),
                f"{view_id}: terrain LOS target set drifted",
            )
            _require(int(los.get("terrain_occluded_target_count", -1)) == 0, f"{view_id}: terrain occludes one or more street targets")
            door_band = los.get("terrain_door_band")
            _require(isinstance(door_band, Mapping), f"{view_id}: worker omitted door-band terrain occlusion audit")
            _require(bool(door_band.get("passed")), f"{view_id}: door-band terrain occlusion audit failed")
            _require(
                float(door_band.get("terrain_occluded_fraction", float("inf"))) <= STREET_MAX_DOOR_BAND_OCCLUSION_FRACTION,
                f"{view_id}: terrain occludes the readable door/facade band",
            )
            _require(
                float(los.get("minimum_door_clearance_scene", float("-inf"))) >= STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE,
                f"{view_id}: minimum terrain LOS clearance is below the hard gate",
            )
            _require(
                float(los.get("minimum_edge_clearance_scene", float("-inf"))) >= STREET_EDGE_MARGIN_SCENE,
                f"{view_id}: street camera/subject is too close to the finite terrain edge",
            )
            edge_intrusion = los.get("terrain_edge_intrusion")
            _require(isinstance(edge_intrusion, Mapping), f"{view_id}: worker omitted finite-edge intrusion audit")
            _require(bool(edge_intrusion.get("passed")), f"{view_id}: finite terrain edge intrudes near the focused subject")
            _require(int(edge_intrusion.get("intrusion_count", -1)) == 0, f"{view_id}: finite terrain edge intrusion count is nonzero")
        if str(view["kind"]) == "detail" and bool(view.get("focused", False)):
            content = camera.get("content_in_frame")
            validate_focused_content_span(content, view_id)
    for name, expected in (
        ("expected_ref_count", scene["counts"]["expected_ref_count"]),
        ("imported_ref_count", scene["counts"]["expected_ref_count"]),
        ("expected_terrain_cell_count", 49),
        ("terrain_cell_count", 49),
        ("expected_view_count", len(expected_views)),
        ("rendered_view_count", len(expected_views)),
    ):
        _require(int(worker_counts.get(name, -1)) == int(expected), f"worker audit count {name} is {worker_counts.get(name)!r}, expected {expected}")
    _require(int(worker_counts.get("empty_import_count", -1)) == 0, "worker reported an empty NIF import")
    _require(int(worker_counts.get("unresolved_model_count", -1)) == 0, "worker reported an unresolved model")
    _require(int(worker_counts.get("placeholder_material_count", -1)) == 0, "worker reported a placeholder building material")
    _require(int(worker_counts.get("unresolved_texture_count", -1)) == 0, "worker reported an unresolved texture")
    _require(int(worker_counts.get("matrix_mismatch_count", -1)) == 0, "worker reported a placement matrix mismatch")
    _require(float(worker_counts.get("max_matrix_error", float("inf"))) <= MATRIX_TOLERANCE, "worker placement rotation matrix error exceeds the T1.5 gate")
    _require(float(worker_counts.get("max_translation_storage_error", float("inf"))) <= 2.0e-5, "worker placement translation storage error exceeds the Blender float gate")
    _require(int(worker_counts.get("terrain_field_record_mismatch_count", -1)) == 0, "worker reported a T1.3 field/LAND mismatch")
    _require(int(worker_counts.get("terrain_texture_missing_count", -1)) == 0, "worker reported a missing terrain texture")
    _require(int(worker_counts.get("terrain_edge_intrusion_failed_view_count", -1)) == 0, "worker reported finite terrain edge intrusion near a subject")
    expected_los_view_count = sum(1 for view in expected_views if str(view.get("kind")) in {"street", "detail"})
    _require(
        int(worker_counts.get("terrain_los_view_count", -1)) == expected_los_view_count,
        "worker terrain LOS view count does not cover every street/detail view",
    )
    _require(int(worker_counts.get("terrain_los_failed_view_count", -1)) == 0, "worker reported a failed terrain LOS view")
    _require(int(worker_counts.get("terrain_door_band_failed_view_count", -1)) == 0, "worker reported terrain occlusion across a readable door/facade band")
    _require(int(worker_counts.get("proxy_geometry_count", -1)) == 0, "worker reported proxy geometry")
    _require(int(worker_counts.get("flat_terrain_fallback_count", -1)) == 0, "worker reported flat terrain fallback")
    water = worker_audit.get("water")
    _require(isinstance(water, Mapping), "worker audit has no water block")
    _require(int(water.get("triangle_count", -1)) > 0, "worker emitted no clipped water triangles")
    _require(int(water.get("z_mismatch_count", -1)) == 0, "worker water is not exactly z=0")
    _require(int(water.get("outside_terrain_footprint_count", -1)) == 0, "worker water escaped the terrain footprint")
    _require(bool(water.get("rectangular_plane_used")) is False, "worker used a forbidden rectangular water plane")
    terrain = worker_audit.get("terrain")
    _require(isinstance(terrain, Mapping), "worker audit has no terrain block")
    _require(terrain.get("field_hash") == scene["terrain"]["field_hash"], "worker terrain field hash disagrees with T1.3")
    _require(bool(terrain.get("opaque")) is True, "worker terrain is not opaque in the final review views")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "diagnostic_scope": scene.get("diagnostic_scope"),
        "synthetic_banner": scene.get("synthetic_banner"),
        "plan_id": scene.get("plan_id"),
        "build_hash": scene.get("build_hash"),
        "view_contract": dict(scene.get("view_contract", {})),
        "scene_manifest_sha256": sha256_file(output_dir / "scene_manifest.json"),
        "render_scene_sha256": sha256_file(output_dir / "render_scene.json"),
        "input_hashes": dict(scene.get("input_hashes", {})),
        "source_paths": dict(scene.get("source_paths", {})),
        "counts": {
            "expected_ref_count": int(scene["counts"]["expected_ref_count"]),
            "emitted_ref_count": int(worker_counts["emitted_ref_count"]),
            "imported_ref_count": int(worker_counts["imported_ref_count"]),
            "unique_model_count": int(worker_counts["unique_model_count"]),
            "expected_terrain_cell_count": 49,
            "terrain_cell_count": int(worker_counts["terrain_cell_count"]),
            "terrain_ltex_count": int(terrain.get("ltex_count", -1)),
            "terrain_texture_resolved_count": int(terrain.get("texture_resolved_count", -1)),
            "terrain_texture_missing_count": int(worker_counts["terrain_texture_missing_count"]),
            "building_texture_resolved_count": int(worker_counts["building_texture_resolved_count"]),
            "building_texture_missing_count": int(worker_counts["building_texture_missing_count"]),
            "matrix_checked_count": int(worker_counts["matrix_checked_count"]),
            "matrix_mismatch_count": int(worker_counts["matrix_mismatch_count"]),
            "max_translation_storage_error": float(worker_counts.get("max_translation_storage_error", -1.0)),
            "multi_axis_canary_count": int(worker_counts.get("multi_axis_canary_count", 0)),
            "near_gimbal_canary_count": int(worker_counts.get("near_gimbal_canary_count", 0)),
            "water_triangle_count": int(water["triangle_count"]),
            "required_base_view_count": REQUIRED_BASE_VIEW_COUNT,
            "focused_detail_view_count": FOCUSED_DETAIL_VIEW_COUNT,
            "rendered_view_count": int(worker_counts["rendered_view_count"]),
            "terrain_los_view_count": int(worker_counts.get("terrain_los_view_count", -1)),
            "terrain_los_failed_view_count": int(worker_counts.get("terrain_los_failed_view_count", -1)),
            "terrain_door_band_failed_view_count": int(worker_counts.get("terrain_door_band_failed_view_count", -1)),
            "terrain_edge_intrusion_failed_view_count": int(worker_counts.get("terrain_edge_intrusion_failed_view_count", -1)),
            "terrain_edge_intrusion_sample_count": int(worker_counts.get("terrain_edge_intrusion_sample_count", -1)),
            "proxy_geometry_count": int(worker_counts["proxy_geometry_count"]),
            "flat_terrain_fallback_count": int(worker_counts["flat_terrain_fallback_count"]),
        },
        "import_audit": worker_audit.get("imports"),
        "texture_audit": worker_audit.get("textures"),
        "matrix_audit": worker_audit.get("matrices"),
        "terrain_audit": terrain,
        "water_audit": water,
        "views": worker_audit.get("views"),
        "images": image_rows,
        "determinism_contract": {
            "scene_units_per_game_unit": SCENE_UNITS_PER_GAME_UNIT,
            "normalize_to_position": False,
            "matrix_tolerance": MATRIX_TOLERANCE,
            "render_pixels_same_machine_only": True,
        },
    }
    write_json(output_dir / "render_audit.json", audit)
    return audit


__all__ = [
    "FIELD_SIDE",
    "FOCUSED_DETAIL_VIEW_COUNT",
    "MATRIX_TOLERANCE",
    "MAX_DARK_FOREGROUND_FRACTION",
    "MIN_FOCUSED_CONTENT_SPAN",
    "MIN_FOREGROUND_LUMINANCE_MEAN",
    "MIN_FOREGROUND_LUMINANCE_P25",
    "REQUIRED_BASE_VIEW_COUNT",
    "RenderContractError",
    "RenderInputPaths",
    "SCENE_UNITS_PER_GAME_UNIT",
    "SCHEMA_VERSION",
    "STAGE",
    "SYNTHETIC_BANNER",
    "STREET_EDGE_MARGIN_SCENE",
    "STREET_CAMERA_CANDIDATE_COUNT",
    "STREET_COMPOSITION_TARGET_RAISE_SCENE",
    "STREET_DOOR_LOS_FLOOR_OFFSET_SCENE",
    "STREET_LOS_TARGET_NAMES",
    "STREET_MIN_TERRAIN_LOS_CLEARANCE_SCENE",
    "STREET_TARGET_OFFSETS_SCENE",
    "STREET_EYE_HEIGHT_SCENE",
    "TERRAIN_EDGE_SUBJECT_MARGIN_NDC",
    "VIEW_CONTRACT_VERSION",
    "audit_png",
    "build_render_scene",
    "canonical_bytes",
    "default_render_input_paths",
    "finalize_render_audit",
    "matrix_max_error",
    "sha256_file",
    "terrain_field_content_hash",
    "terrain_edge_clearance_scene",
    "terrain_height_scene",
    "terrain_line_of_sight",
    "validate_focused_content_span",
    "validate_terrain_building_occlusion",
    "write_json",
]
