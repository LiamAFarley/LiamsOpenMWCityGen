"""Cumulative xFa layer evidence and synthetic-source helpers.

The xFa ESPs are cumulative WIP patches, while the generic settlement stages
accept one source plugin.  This module performs only the deterministic
per-cell/refr-index merge required to present the effective layer view to
those existing stages.  It does not decide house membership or cluster by
position; those decisions stay with contact/components and unlinked_units.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

from .espscan import CellReference, CellSummary, ScanResult


@dataclass(frozen=True)
class EffectiveCells:
    cells: tuple[CellSummary, ...]
    source_layers: Mapping[tuple[int, int, int], str]


def merge_layer_cells(
    layer_results: Sequence[tuple[str, ScanResult]],
) -> EffectiveCells:
    """Merge exterior cells in configured load order, later refs winning."""

    by_grid: dict[tuple[int, int], dict[int, CellReference]] = {}
    cell_meta: dict[tuple[int, int], CellSummary] = {}
    source_layers: dict[tuple[int, int, int], str] = {}
    for layer_name, result in layer_results:
        for cell in result.cells:
            if cell.is_interior or cell.grid is None:
                continue
            grid = cell.grid
            refs = by_grid.setdefault(grid, {})
            for reference in cell.references:
                refs[int(reference.refr_index)] = reference
                source_layers[(grid[0], grid[1], int(reference.refr_index))] = layer_name
            previous = cell_meta.get(grid)
            cell_meta[grid] = CellSummary(
                name=cell.name or (previous.name if previous else None),
                is_interior=False,
                grid=grid,
                region=cell.region or (previous.region if previous else None),
                flags=int(cell.flags) | int(previous.flags if previous else 0),
                references=tuple(),
                offset=cell.offset,
            )
    cells = []
    for grid in sorted(by_grid):
        meta = cell_meta[grid]
        cells.append(
            CellSummary(
                name=meta.name,
                is_interior=False,
                grid=grid,
                region=meta.region,
                flags=meta.flags,
                references=tuple(by_grid[grid][key] for key in sorted(by_grid[grid])),
                offset=meta.offset,
            )
        )
    return EffectiveCells(tuple(cells), source_layers)


def _subrecord(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack("<I", len(payload)) + payload


def _text(value: str | None) -> bytes:
    return (value or "").encode("cp1252", errors="replace") + b"\0"


def _cell_body(cell: CellSummary) -> bytes:
    body = bytearray()
    body += _subrecord(b"NAME", _text(cell.name)) if cell.name else b""
    body += _subrecord(b"DATA", struct.pack("<Iii", int(cell.flags), *cell.grid))
    if cell.region:
        body += _subrecord(b"RGNN", _text(cell.region))
    for reference in cell.references:
        raw = ((int(reference.mast_index) & 0xFF) << 24) | (int(reference.refr_index) & 0xFFFFFF)
        ref_body = bytearray(struct.pack("<I", raw))
        ref_body += _subrecord(b"NAME", _text(reference.object_id))
        position = tuple(float(value) for value in (reference.position or (0.0, 0.0, 0.0)))
        rotation = tuple(float(value) for value in (reference.rotation or (0.0, 0.0, 0.0)))
        ref_body += _subrecord(b"DATA", struct.pack("<6f", *(position + rotation)))
        if reference.scale is not None:
            ref_body += _subrecord(b"XSCL", struct.pack("<f", float(reference.scale)))
        if reference.owner:
            ref_body += _subrecord(b"ANAM", _text(reference.owner))
        if reference.has_dodt and reference.destination_position is not None and reference.destination_rotation is not None:
            destination = tuple(float(value) for value in reference.destination_position)
            destination_rotation = tuple(float(value) for value in reference.destination_rotation)
            ref_body += _subrecord(b"DODT", struct.pack("<6f", *(destination + destination_rotation)))
        if reference.destination_cell:
            ref_body += _subrecord(b"DNAM", _text(reference.destination_cell))
        body += _subrecord(b"FRMR", bytes(ref_body[:4])) + bytes(ref_body[4:])
    return bytes(body)


def write_synthetic_source(path: Path, cells: Sequence[CellSummary]) -> None:
    """Write the minimal CELL-only TES3 source consumed by generic A1."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    for cell in sorted(cells, key=lambda row: row.grid or (0, 0)):
        body = _cell_body(cell)
        payload += b"CELL" + struct.pack("<III", len(body), 0, 0) + body
    path.write_bytes(bytes(payload))


def effective_door_rows(
    cells: Sequence[CellSummary],
    source_layers: Mapping[tuple[int, int, int], str],
) -> list[dict[str, Any]]:
    """Return audit rows for every effective DOOR, including its winning layer."""

    rows: list[dict[str, Any]] = []
    for cell in cells:
        for reference in cell.references:
            if reference.record_type != "DOOR":
                continue
            grid = cell.grid or (0, 0)
            rows.append(
                {
                    "grid": [grid[0], grid[1]],
                    "cell_name": cell.name or "",
                    "refr_index": int(reference.refr_index),
                    "source_layer": source_layers.get((grid[0], grid[1], int(reference.refr_index))),
                    "object_id": reference.object_id,
                    "model": reference.model,
                    "door_to_interior": bool(reference.door_to_interior),
                    "resolved": bool(reference.model) and not reference.unresolved,
                    "position": list(reference.position or ()),
                }
            )
    return sorted(rows, key=lambda row: (tuple(row["grid"]), int(row["refr_index"])))


def layer_summary(result: ScanResult) -> dict[str, Any]:
    return {
        "path": result.path,
        "sha256": result.sha256,
        "size_bytes": int(result.size_bytes),
        "exterior_cells": int(result.exterior_cells),
        "reference_count": int(result.reference_count),
        "door_count": sum(1 for cell in result.cells for ref in cell.references if ref.record_type == "DOOR"),
    }
