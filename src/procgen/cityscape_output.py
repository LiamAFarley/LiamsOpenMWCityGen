"""T1.3 LAND/LTEX record assembly and post-assembly audits.

Pipeline position
------------------
This module is the boundary from the host-side landscape engine to T1.4.  It
receives the final quantized joint field, effective/painted VTEX grids, source
payloads, and recomputed VNML bytes.  It emits a masterless tes3conv JSON
record document containing one local LTEX record for every positive raw VTEX
value and one LAND record per target cell.  T1.4 may consume that exact
document without reconstructing terrain decisions.

Invariants
----------
* ``tamriel.esm`` is provenance only and never appears in ``masters``.
* Heights are handed to :func:`procgen.tes3json.build_land` as signed THU
  values already quantized by T1.3; no second rounding occurs here.
* VCLR, WNAM, LAND DATA flags/unknown bits, and source VNML on unedited cells
  are copied byte-for-byte.  Height-edited cells differ only in VHGT/VNML;
  VTEX differences are declared in the paint ledger.
* The JSON document passes ``tes3json.validate`` and its decoded LAND payloads
  are checked against the intended final field before it is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from . import espland, tes3json
from .cityscape_field import TargetBlock, split_field
from .cityscape_vtex import PaintResult


class CityscapeOutputError(ValueError):
    """Hard LAND/LTEX record or decoded-output audit failure."""


def _landscape_flags_string(flags: int) -> str:
    bits = ((0x1, "USES_VERTEX_HEIGHTS_AND_NORMALS"), (0x2, "USES_VERTEX_COLORS"), (0x4, "USES_TEXTURES"))
    remaining = int(flags)
    if remaining < 0 or remaining > 0xFFFFFFFF:
        raise CityscapeOutputError(f"LAND DATA flags outside u32: {flags}")
    result: list[str] = []
    for bit, name in bits:
        if flags & bit:
            result.append(name)
            remaining &= ~bit
    bit = 1
    while remaining:
        if remaining & bit:
            result.append(f"0x{bit:x}")
            remaining &= ~bit
        bit <<= 1
    return " | ".join(result)


def _decode_vtex_record(record: Mapping[str, Any]) -> np.ndarray:
    blob = record.get("texture_indices")
    if not isinstance(blob, Mapping):
        raise CityscapeOutputError("LAND record has no texture_indices blob")
    raw = tes3json.decode_blob(blob.get("data"))
    if len(raw) != 512:
        raise CityscapeOutputError(f"decoded VTEX has {len(raw)} bytes, expected 512")
    values = struct.unpack("<256H", raw)
    return np.asarray(espland.transpose_vtex_serialized_to_openmw(values), dtype=np.uint16).reshape(16, 16)


def _record_by_grid(document: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in document:
        if record.get("type") != "Landscape":
            continue
        grid = record.get("grid")
        if not isinstance(grid, list) or len(grid) != 2:
            raise CityscapeOutputError("Landscape record has malformed grid")
        key = (int(grid[0]), int(grid[1]))
        if key in result:
            raise CityscapeOutputError(f"duplicate output LAND grid {key}")
        result[key] = record
    return result


def _payload_equal_except(
    source: espland.LandRecord,
    record: Mapping[str, Any],
    *,
    expected_height_thu: np.ndarray,
    expected_vtex: np.ndarray,
    expected_normals: bytes,
) -> dict[str, bool]:
    heights = np.asarray(tes3json.decode_land_heights(record, game_units=False), dtype=np.int64)
    normals = tes3json.decode_blob(record["vertex_normals"]["data"])
    colors = tes3json.decode_blob(record["vertex_colors"]["data"])
    world = tes3json.decode_blob(record["world_map_data"]["data"])
    return {
        "heights_exact": bool(np.array_equal(heights, expected_height_thu)),
        "normals_exact": normals == expected_normals,
        "colors_exact": source.vertex_colors == colors,
        "world_map_exact": source.world_map_data == world,
        "flags_exact": int(source.flags) == _flags_from_string(str(record.get("landscape_flags", ""))),
        "vtex_exact": bool(np.array_equal(_decode_vtex_record(record), expected_vtex)),
    }


def _flags_from_string(value: str) -> int:
    names = {
        "USES_VERTEX_HEIGHTS_AND_NORMALS": 0x1,
        "USES_VERTEX_COLORS": 0x2,
        "USES_TEXTURES": 0x4,
    }
    result = 0
    for item in value.split("|"):
        name = item.strip()
        if not name:
            continue
        if name.startswith("0x"):
            result |= int(name[2:], 16)
        elif name in names:
            result |= names[name]
        else:
            raise CityscapeOutputError(f"unknown LAND flag in assembled record: {name}")
    return result


@dataclass(frozen=True)
class LandAssembly:
    """Exact tes3conv document and decoded audit evidence."""

    document: list[dict[str, Any]]
    land_records: Mapping[tuple[int, int], Mapping[str, Any]]
    ltex_records: tuple[Mapping[str, Any], ...]
    audit: Mapping[str, Any]


def assemble_land_records(
    *,
    block: TargetBlock,
    final_values_gu: np.ndarray,
    painted: PaintResult,
    normal_payloads: Mapping[tuple[int, int], bytes],
    height_edited_cells: set[tuple[int, int]],
    plan_id: str,
) -> LandAssembly:
    """Build and decode-check a masterless LAND/LTEX document."""

    heights_by_cell = split_field(np.asarray(final_values_gu, dtype=np.float64), block.cells)
    ltex_rows = tuple(sorted((dict(row) for row in painted.local_ltex), key=lambda row: int(row["index"])))
    if not ltex_rows:
        raise CityscapeOutputError("paint output has no local LTEX records")
    ltex_indices = [int(row["index"]) for row in ltex_rows]
    if len(ltex_indices) != len(set(ltex_indices)):
        raise CityscapeOutputError("assembled local LTEX indices are duplicated")
    document = tes3json.new_plugin({
        "author": "Procedural Tamriel",
        "description": f"Cityforge T1.3 synthetic landscape field {plan_id}",
        "masters": [],
        "num_objects": len(ltex_rows) + len(block.cells),
    })
    document.extend(
        tes3json.build_ltex(
            str(row["record_id"]),
            int(row["index"]),
            str(row["file_name"]),
        )
        for row in ltex_rows
    )
    for cell in sorted(block.cells, key=lambda grid: (grid[1], grid[0])):
        source = block.source_land[cell]
        if source.vertex_normals is None or source.vertex_colors is None or source.world_map_data is None:
            raise CityscapeOutputError(f"source LAND {cell} has incomplete payload")
        grid = np.asarray(painted.grids[cell], dtype=np.uint16)
        if grid.shape != (16, 16):
            raise CityscapeOutputError(f"painted VTEX {cell} shape is {grid.shape}")
        raw_values = {int(value) for value in grid.reshape(-1)}
        missing = [raw for raw in sorted(raw_values) if raw > 0 and raw - 1 not in ltex_indices]
        if missing:
            raise CityscapeOutputError(f"LAND {cell} references raw values without local LTEX: {missing}")
        normals = normal_payloads[cell] if cell in height_edited_cells else bytes(source.vertex_normals)
        if len(normals) != 65 * 65 * 3:
            raise CityscapeOutputError(f"assembled VNML {cell} has invalid length")
        serialized_vtex = espland.transpose_vtex_openmw_to_serialized(tuple(int(value) for value in grid.reshape(-1)))
        document.append(tes3json.build_land(
            cell,
            np.rint(np.asarray(heights_by_cell[cell], dtype=np.float64) / 8.0).astype(np.int64),
            heights_in_thu=True,
            landscape_flags=_landscape_flags_string(source.flags),
            vertex_normals=normals,
            world_map_data=bytes(source.world_map_data),
            vertex_colors=bytes(source.vertex_colors),
            texture_indices=serialized_vtex,
        ))
    issues = tes3json.validate(document)
    if issues:
        raise CityscapeOutputError("tes3json.validate rejected LAND document: " + "; ".join(map(str, issues[:8])))
    decoded = _record_by_grid(document)
    if set(decoded) != set(block.cells):
        raise CityscapeOutputError("assembled LAND grid set does not equal target 49 cells")
    record_audit: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    for cell in sorted(block.cells, key=lambda grid: (grid[1], grid[0])):
        expected_thu = np.rint(heights_by_cell[cell] / 8.0).astype(np.int64)
        expected_vtex = np.asarray(painted.grids[cell], dtype=np.uint16)
        expected_normals = normal_payloads[cell] if cell in height_edited_cells else bytes(block.source_land[cell].vertex_normals or b"")
        checks = _payload_equal_except(
            block.source_land[cell], decoded[cell], expected_height_thu=expected_thu,
            expected_vtex=expected_vtex, expected_normals=expected_normals,
        )
        edited_fields = ["VHGT", "VNML"] if cell in height_edited_cells else []
        if np.any(expected_vtex != painted.source_grids[cell]):
            edited_fields.append("VTEX")
        row = {
            "cell": [cell[0], cell[1]],
            "height_edited": cell in height_edited_cells,
            "declared_changed_fields": edited_fields,
            "checks": checks,
            "source_payload_digest": _source_digest_for_audit(block.source_land[cell]),
        }
        record_audit.append(row)
        required = {"heights_exact", "normals_exact", "colors_exact", "world_map_exact", "flags_exact", "vtex_exact"}
        if not all(checks[field] for field in required):
            differences.append(row)
    if differences:
        raise CityscapeOutputError("decoded LAND records differ from intended fields: " + json.dumps(differences[:3], sort_keys=True))
    decoded_raw_values = sorted({
        int(value)
        for record in decoded.values()
        for value in _decode_vtex_record(record).reshape(-1)
        if int(value) > 0
    })
    decoded_missing_ltex = [raw for raw in decoded_raw_values if raw - 1 not in ltex_indices]
    if decoded_missing_ltex:
        raise CityscapeOutputError(
            f"decoded output LAND references raw values without local LTEX: {decoded_missing_ltex}"
        )
    audit = {
        "product": "cityforge_t1_3_land_records",
        "plan_id": plan_id,
        "plugin_scope": {"file_type": "Esp", "masters": []},
        "record_counts": {"header": 1, "ltex": len(ltex_rows), "land": len(block.cells), "total_objects": len(ltex_rows) + len(block.cells)},
        "ltex_indices": ltex_indices,
        "height_edited_cells": [list(cell) for cell in sorted(height_edited_cells)],
        "decoded_payload_audit": record_audit,
        "all_decoded_records_exact": not differences,
        "tes3json_issue_count": 0,
        "decoded_positive_raw_values": decoded_raw_values,
        "decoded_local_ltex_complete": not decoded_missing_ltex,
    }
    return LandAssembly(document, decoded, ltex_rows, audit)


def _source_digest_for_audit(record: espland.LandRecord) -> str:
    from .cityscape_field import payload_digest
    return payload_digest(record)


def build_land_edits_document(
    *,
    plan_id: str,
    terrain_field_sha256: str,
    height_edits: Sequence[Mapping[str, Any]],
    final_encoding: Mapping[str, Any],
    source_unchanged: Mapping[str, Any],
    painted: PaintResult,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the T1.3 host-side ledger handed alongside LAND records."""

    vtex: dict[str, Any] = {}
    for cell in sorted(painted.grids, key=lambda grid: (grid[1], grid[0])):
        vtex[f"{cell[0]}_{cell[1]}"] = {
            "class_grid": painted.grids[cell].astype(int).tolist(),
            "source_grid_sha256": painted.paint_ledger["cells"][f"{cell[0]},{cell[1]}"]["source_grid_sha256"],
            "painted_grid_sha256": painted.paint_ledger["cells"][f"{cell[0]},{cell[1]}"]["painted_grid_sha256"],
            "support_tile_count": painted.paint_ledger["cells"][f"{cell[0]},{cell[1]}"]["support_tile_count"],
        }
    return {
        "schema_version": 1,
        "product": "cityforge_t1_3_land_edits",
        "diagnostic_scope": "synthetic_not_a_falkreath_design",
        "plan_id": plan_id,
        "height_edits": [dict(row) for row in height_edits],
        "height_encoding": dict(final_encoding),
        "vertex_provenance": dict(source_unchanged),
        "vtex_paint": vtex,
        "vtex_audit": dict(painted.paint_ledger),
        "ltex_table": [dict(row) for row in painted.local_ltex],
        "terrain_field_sha256": terrain_field_sha256,
        "inputs": dict(inputs),
    }


__all__ = [
    "CityscapeOutputError",
    "LandAssembly",
    "assemble_land_records",
    "build_land_edits_document",
]
