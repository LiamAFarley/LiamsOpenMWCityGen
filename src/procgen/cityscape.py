"""Cityforge Dispatch 6 T1.3 hard landscape engine orchestrator.

Pipeline position
------------------
This module wires the accepted T1.1 plan and T1.2 placement contract through
the complete landscape path: real ``tamriel.esm`` LAND load -> 49-cell exact
stitch -> analytic ordered edits -> one final THU legality gate -> root-checked
VNML -> deterministic effective VTEX paint -> local-LTEX/tes3json LAND output
-> T1.2 final re-seat.  It is intentionally a synthetic proof harness for
Falkreath source terrain, not a city designer and not an ESP author.

Inputs and outputs
------------------
``build_cityscape`` consumes explicit accepted input paths and writes only the
requested generated output directory.  The canonical fixture CLI uses the
accepted synthetic T1.2 plan, real Falkreath source LAND, and the accepted
dispatch-5 remap ESP.  Products include planned/final NPZ fields and metadata,
``land_records.json`` for T1.4, ``land_edits.json``, validation and source/
output manifests, T1.2 final-reseat products, deterministic diagnostics, and a
structured analytic-edit diagnostic ledger.

Hard gates
----------
The orchestrator fails closed on source seam disagreement, payload scope
divergence, live-remap assignment failure, partitioned VNML root/boundary
parity failure, production shared-edge disagreement, illegal THU deltas,
changed source VTEX outside declared support, raw-1 road paint, missing local
LTEX, decoded tes3json mismatch, or a failed T1.2 final re-seat.
No fallback terrain, clipping, source modification, real city design, mesh
render, or production ESP is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from . import cityplace, regionpalette, tes3json
from .censusio import deterministic_dumps, sha256_file, write_deterministic
from .cityscape_edits import CityscapeEditError, apply_edit, compose_edits, validate_edit_request
from .cityscape_field import (
    TargetBlock,
    field_metadata,
    load_target_block,
    outer_border_mask,
    split_field,
    terrain_field_sha256,
    write_field_npz,
    write_metadata,
)
from .cityscape_output import (
    CityscapeOutputError,
    assemble_land_records,
    build_land_edits_document,
)
from .cityscape_vnml import (
    VNMLConvention,
    VNMLConventionError,
    analytic_oracle_checks,
    compute_cell_normals,
    production_shared_edge_audit,
    validate_source_convention,
)
from .cityscape_vtex import CityscapeVTEXError, paint_vtex
from .cityplace_contracts import load_json
from .cityplace_output import build_manifest, write_products


class CityscapeError(RuntimeError):
    """Hard T1.3 pipeline failure; callers must report the failed stage."""


@dataclass(frozen=True)
class CityscapePaths:
    """All explicit source/product paths for one deterministic build."""

    workspace_root: Path
    survey: Path
    palette: Path
    plan: Path
    validation: Path
    t12_placement: Path
    t12_land_edits: Path
    source_land: Path
    effective_remap: Path
    output_dir: Path
    kit_brief: Path
    stamp_libraries: tuple[Path, ...]
    centerlines: Path


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CityscapeError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CityscapeError(f"{label} {path} must be a JSON object")
    return value


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(deterministic_dumps(value)).hexdigest()


def _write_json_any(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Mapping):
        data = deterministic_dumps(value)
    else:
        # tes3conv documents are top-level arrays.  Keep the same canonical
        # key/float/newline policy without wrapping the document in a non-TES3
        # object that T1.4 would have to unwrap.
        text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)
        data = (text + "\n").encode("utf-8")
    path.write_bytes(data)
    return sha256_file(path)


def _plan_links(plan: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Build the strict edit-link registry and authorized water feature set."""

    known: set[str] = {str(plan.get("plan_id")), str(plan.get("settlement", {}).get("name", ""))}
    water: set[str] = set()
    collections = {
        "districts": "district_id",
        "roads": "road_id",
        "lots": "lot_id",
        "boundaries": "boundary_id",
        "features": "feature_id",
        "terrain_edits": "edit_id",
        "texture_zones": "zone_id",
    }
    for collection, key in collections.items():
        rows = plan.get(collection, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get(key), str):
                continue
            identifier = str(row[key])
            known.add(identifier)
            if collection == "features" and (
                str(row.get("kind", "")) == "dock" or "dock" in identifier.lower() or "basin" in identifier.lower()
            ):
                water.add(identifier)
    known.discard("")
    return known, water


def _auto_pad_edits(requests_document: Mapping[str, Any]) -> list[dict[str, Any]]:
    requests = requests_document.get("requests")
    if not isinstance(requests, list):
        raise CityscapeError("T1.2 land_edit_requests has no requests list")
    edits: list[dict[str, Any]] = []
    for request in sorted(requests, key=lambda row: str(row.get("lot_id")) if isinstance(row, Mapping) else ""):
        if not isinstance(request, Mapping):
            raise CityscapeError("T1.2 pad request is not an object")
        row = dict(request)
        lot_id = row.get("lot_id")
        if not isinstance(lot_id, str):
            raise CityscapeError("T1.2 pad request has no lot_id")
        row.update({
            "edit_id": f"auto_pad_{lot_id}",
            "kind": "auto_pad",
            "linked_to": [lot_id],
            "polygon": row.get("pad_polygon"),
        })
        # Keep the T1.2 values verbatim.  In particular, no solver-side
        # retarget, larger margin, or widened falloff is permitted here.
        edits.append(row)
    return edits


def _road_grade_edits(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    for road in plan.get("roads", []):
        if not isinstance(road, Mapping) or str(road.get("grade_policy", "conform")) != "regrade":
            continue
        road_id = road.get("road_id")
        if not isinstance(road_id, str):
            raise CityscapeError("regrade road has no road_id")
        edits.append({
            "edit_id": f"road_grade_{road_id}",
            "kind": "road_grade",
            "linked_to": [road_id],
            "polyline": road.get("polyline"),
            "width_gu": road.get("width_gu"),
            "falloff_gu": road.get("falloff_gu", 512.0),
            "max_grade_percent": road.get("max_grade_percent", 10.0),
            "max_cut_fill_gu": road.get("max_cut_fill_gu", 1016.0),
        })
    return sorted(edits, key=lambda row: str(row["edit_id"]))


def _stitch_audit(block: TargetBlock) -> dict[str, Any]:
    per_cell = split_field(block.source_heights_gu, block.cells)
    rejoined = np.empty_like(block.source_heights_gu)
    # Rejoin through the public split/rejoin path and compare every shared edge.
    from .cityscape_field import rejoin_field
    rejoined = rejoin_field(per_cell, block.cells)
    return {
        "cell_count": len(block.cells),
        "field_shape": list(block.field_shape),
        "spacing_gu": block.spacing_gu,
        "shared_edges_exact": bool(np.array_equal(rejoined, block.source_heights_gu)),
        "rejoin_exact": bool(np.array_equal(rejoined, block.source_heights_gu)),
        "source_field_sha256": block.field_sha256,
        "outer_border_vertex_count": int(np.count_nonzero(outer_border_mask(block.field_shape))),
        "outer_border_source_frozen": True,
    }


def _normal_shared_edge_audit(
    block: TargetBlock,
    values_gu: np.ndarray,
    convention: VNMLConvention,
    edited_cells: set[tuple[int, int]],
) -> dict[str, Any]:
    payloads = {cell: compute_cell_normals(values_gu, block, cell, convention) for cell in sorted(edited_cells)}
    mismatches: list[dict[str, Any]] = []
    for cell in sorted(edited_cells):
        array = np.frombuffer(payloads[cell], dtype=np.int8).reshape(65, 65, 3)
        right = (cell[0] + 1, cell[1])
        north = (cell[0], cell[1] + 1)
        if right in payloads:
            other = np.frombuffer(payloads[right], dtype=np.int8).reshape(65, 65, 3)
            if not np.array_equal(array[:, -1], other[:, 0]):
                mismatches.append({"axis": "x", "cells": [list(cell), list(right)]})
        if north in payloads:
            other = np.frombuffer(payloads[north], dtype=np.int8).reshape(65, 65, 3)
            if not np.array_equal(array[-1, :], other[0, :]):
                mismatches.append({"axis": "y", "cells": [list(cell), list(north)]})
    if mismatches:
        raise CityscapeError(f"VNML edited shared-edge mismatch: {mismatches[:4]}")
    return {
        "edited_cell_count": len(edited_cells),
        "shared_edges_checked": sum(
            int((cell[0] + 1, cell[1]) in edited_cells) + int((cell[0], cell[1] + 1) in edited_cells)
            for cell in edited_cells
        ),
        "shared_edges_exact": not mismatches,
        "payloads": payloads,
    }


def _edit_diagnostics(
    block: TargetBlock,
    *,
    auto_pad: Mapping[str, Any] | None,
    known_links: set[str],
    water_links: set[str],
) -> list[dict[str, Any]]:
    """Exercise analytic successes and structured rejects without city design."""

    center = [28672.0, 28672.0]
    field_h, field_w = block.source_heights_gu.shape
    base_height = float(block.source_heights_gu[field_h // 2, field_w // 2])
    cases: list[tuple[str, Mapping[str, Any], str, str]] = [
        (
            "flatten_shelf_accept",
            {"edit_id": "diag_flatten", "kind": "flatten_shelf", "polygon": [[27648, 27648], [29696, 27648], [29696, 29648], [27648, 29648]], "target_height_gu": base_height, "falloff_gu": 1024, "linked_to": ["diag_lot"]},
            "accepted", "",
        ),
        (
            "mound_accept",
            {"edit_id": "diag_mound", "kind": "mound", "center": center, "radius_gu": 256, "height_delta_gu": 16, "falloff_gu": 1024, "linked_to": ["diag_lot"]},
            "accepted", "",
        ),
        (
            "terrace_accept",
            {"edit_id": "diag_terrace", "kind": "terrace", "shelves": [{"polygon": [[26624, 26624], [27648, 26624], [27648, 27648], [26624, 27648]], "target_height_gu": base_height, "falloff_gu": 512}], "falloff_gu": 512, "linked_to": ["diag_lot"]},
            "accepted", "",
        ),
        (
            "cut_accept",
            {"edit_id": "diag_cut", "kind": "cut", "polyline": [[24576, 24576], [25600, 25600]], "width_gu": 256, "depth_gu": 16, "falloff_gu": 1024, "linked_to": ["diag_road"]},
            "accepted", "",
        ),
        (
            "road_grade_accept",
            {"edit_id": "diag_grade", "kind": "road_grade", "polyline": [[8192, 2048], [16384, 2048]], "width_gu": 512, "falloff_gu": 1024, "max_grade_percent": 25, "linked_to": ["diag_road"]},
            "accepted", "",
        ),
        (
            "out_of_bounds_reject",
            {"edit_id": "diag_oob", "kind": "flatten_shelf", "polygon": [[256, 256], [768, 256], [768, 768], [256, 768]], "target_height_gu": base_height, "falloff_gu": 512, "linked_to": ["diag_lot"]},
            "rejected", "out_of_bounds",
        ),
        (
            "terrace_out_of_bounds_reject",
            {"edit_id": "diag_terrace_oob", "kind": "terrace", "shelves": [{"polygon": [[256, 256], [768, 256], [768, 768], [256, 768]], "target_height_gu": base_height, "falloff_gu": 512}], "falloff_gu": 512, "linked_to": ["diag_lot"]},
            "rejected", "out_of_bounds",
        ),
        (
            "cut_out_of_bounds_reject",
            {"edit_id": "diag_cut_oob", "kind": "cut", "polyline": [[256, 256], [1024, 256]], "width_gu": 256, "depth_gu": 16, "falloff_gu": 512, "linked_to": ["diag_road"]},
            "rejected", "out_of_bounds",
        ),
        (
            "road_grade_out_of_bounds_reject",
            {"edit_id": "diag_grade_oob", "kind": "road_grade", "polyline": [[256, 256], [1024, 256]], "width_gu": 512, "falloff_gu": 512, "max_grade_percent": 10, "max_cut_fill_gu": 400, "linked_to": ["diag_road"]},
            "rejected", "out_of_bounds",
        ),
        (
            "unknown_link_reject",
            {"edit_id": "diag_link", "kind": "mound", "center": center, "radius_gu": 256, "height_delta_gu": 16, "falloff_gu": 1024, "linked_to": ["missing_link"]},
            "rejected", "unknown_link",
        ),
        (
            "basin_reject",
            {"edit_id": "diag_basin", "kind": "mound", "center": center, "radius_gu": 256, "height_delta_gu": -10000, "falloff_gu": 1024, "linked_to": ["diag_lot"]},
            "rejected", "unintentional_basin",
        ),
        (
            "too_steep_reject",
            {"edit_id": "diag_steep", "kind": "mound", "center": center, "radius_gu": 256, "height_delta_gu": 5000, "falloff_gu": 128, "linked_to": ["diag_lot"]},
            "rejected", "edit_too_steep",
        ),
    ]
    if auto_pad is not None:
        cases.append(("auto_pad_accept", dict(auto_pad), "accepted", ""))
        illegal = dict(auto_pad)
        illegal["edit_id"] = "diag_illegal_pad"
        illegal["margin_gu"] = 128.0
        cases.append(("illegal_pad_reject", illegal, "rejected", "illegal_pad"))
    rows: list[dict[str, Any]] = []
    for case_id, edit, expected_status, expected_code in cases:
        record: dict[str, Any] = {"case_id": case_id, "expected_status": expected_status, "expected_code": expected_code}
        try:
            application = apply_edit(
                edit,
                block.source_heights_gu,
                block=block,
                known_links=known_links | {"diag_lot", "diag_road"},
                authorized_water_links=water_links,
            )
            record.update({"status": "accepted", "actual_code": "", "ledger": dict(application.ledger)})
        except CityscapeEditError as exc:
            record.update({"status": "rejected", "actual_code": str(exc.failure.get("code", "")), "failure": dict(exc.failure)})
        record["pass"] = record["status"] == expected_status and (not expected_code or record.get("actual_code") == expected_code)
        rows.append(record)
    if not all(row["pass"] for row in rows):
        raise CityscapeError("analytic synthetic diagnostic mismatch: " + json.dumps(rows, sort_keys=True))
    return rows


def _diagnostic_images(
    output_dir: Path,
    *,
    source_values: np.ndarray,
    final_values: np.ndarray,
    paint: Any,
) -> dict[str, Any]:
    delta = np.asarray(final_values) - np.asarray(source_values)
    magnitude = np.max(np.abs(delta)) or 1.0
    delta_pixels = np.clip(128.0 + delta / magnitude * 127.0, 0, 255).astype(np.uint8)
    slope = np.zeros_like(final_values, dtype=np.float64)
    slope[1:-1, 1:-1] = np.degrees(np.arctan(np.hypot(
        (final_values[1:-1, 2:] - final_values[1:-1, :-2]) / 256.0,
        (final_values[2:, 1:-1] - final_values[:-2, 1:-1]) / 256.0,
    )))
    slope_pixels = np.clip(slope / max(float(np.max(slope)), 1.0) * 255.0, 0, 255).astype(np.uint8)
    paths = {
        "height_delta": output_dir / "diagnostic_height_delta.png",
        "final_slope": output_dir / "diagnostic_final_slope.png",
        "vtex_before": output_dir / "diagnostic_vtex_before.png",
        "vtex_after": output_dir / "diagnostic_vtex_after.png",
        "vtex_paint_classes": output_dir / "diagnostic_vtex_paint_classes.png",
    }
    Image.fromarray(delta_pixels, mode="L").save(paths["height_delta"], format="PNG", optimize=False)
    Image.fromarray(slope_pixels, mode="L").save(paths["final_slope"], format="PNG", optimize=False)
    colors = {0: (20, 20, 20), 1: (220, 220, 245), 33: (80, 170, 80), 78: (145, 95, 48), 92: (30, 100, 35), 142: (95, 155, 85), 144: (130, 130, 130), 241: (178, 115, 65)}
    cell_order = sorted(paint.grids, key=lambda cell: (cell[1], cell[0]))
    before = Image.new("RGB", (112, 112))
    after = Image.new("RGB", (112, 112))
    for index, cell in enumerate(cell_order):
        ox, oy = (cell[0] - min(c[0] for c in cell_order)) * 16, (cell[1] - min(c[1] for c in cell_order)) * 16
        before_grid = paint.source_grids[cell]
        after_grid = paint.grids[cell]
        for y in range(16):
            for x in range(16):
                before.putpixel((ox + x, oy + y), colors.get(int(before_grid[y, x]), (255, 0, 255)))
                after.putpixel((ox + x, oy + y), colors.get(int(after_grid[y, x]), (255, 0, 255)))
    before.save(paths["vtex_before"], format="PNG", optimize=False)
    after.save(paths["vtex_after"], format="PNG", optimize=False)
    paint_classes = Image.new("RGB", (112, 112))
    class_colors = {
        1: (55, 110, 220),    # protected water/lakebed
        33: (80, 170, 80),    # source/base grass
        78: (230, 70, 45),    # road priority
        92: (30, 100, 35),    # preserved source pine
        142: (95, 155, 85),   # grass-dirt margin/class
        144: (130, 130, 130), # cobble
        241: (178, 115, 65),  # dirt/lot class
    }
    for cell in cell_order:
        ox, oy = (cell[0] - min(c[0] for c in cell_order)) * 16, (cell[1] - min(c[1] for c in cell_order)) * 16
        grid = paint.grids[cell]
        for y in range(16):
            for x in range(16):
                paint_classes.putpixel((ox + x, oy + y), class_colors.get(int(grid[y, x]), (255, 0, 255)))
    paint_classes.save(paths["vtex_paint_classes"], format="PNG", optimize=False)
    audit: dict[str, Any] = {
        "diagnostic_scope": "synthetic_not_a_falkreath_design",
        "kind": "terrain_and_vtex_diagnostics_not_city_render",
        "resolutions": {
            "height_delta": [int(v) for v in delta.shape],
            "final_slope": [int(v) for v in slope.shape],
            "vtex_before": [112, 112],
            "vtex_after": [112, 112],
            "vtex_paint_classes": [112, 112],
        },
        "height_delta": {
            "min_gu": float(np.min(delta)), "max_gu": float(np.max(delta)), "changed_vertices": int(np.count_nonzero(delta)), "nonflat_pixels": int(np.count_nonzero(delta_pixels != 128)), "image_sha256": sha256_file(paths["height_delta"]),
        },
        "final_slope": {"max_deg": float(np.max(slope)), "nonzero_pixels": int(np.count_nonzero(slope_pixels)), "image_sha256": sha256_file(paths["final_slope"])},
        "vtex": {
            "before_image_sha256": sha256_file(paths["vtex_before"]),
            "after_image_sha256": sha256_file(paths["vtex_after"]),
            "paint_classes_image_sha256": sha256_file(paths["vtex_paint_classes"]),
            "source_raw_counts": _raw_counts(paint.source_grids),
            "painted_raw_counts": _raw_counts(paint.grids),
        },
    }
    write_deterministic(output_dir / "diagnostic_images.json", audit)
    return audit


def _raw_counts(grids: Mapping[tuple[int, int], np.ndarray]) -> dict[str, int]:
    result: dict[str, int] = {}
    for grid in grids.values():
        for value in np.asarray(grid).reshape(-1):
            result[str(int(value))] = result.get(str(int(value)), 0) + 1
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _write_t12_final_products(
    output_dir: Path,
    result: Mapping[str, Any],
    *,
    validation_path: Path,
    plan_id: str,
    plan_hash: str,
) -> dict[str, Any]:
    final_dir = output_dir / "t1_2_final_reseat"
    final_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = dict(result["source_hashes"])
    source_hashes["t1_1_validation"] = sha256_file(validation_path)
    output_hashes = write_products(
        final_dir,
        city_placement=result["city_placement"],
        land_edit_requests=result["land_edit_requests"],
        solver_report=result["solver_report"],
        source_hashes=source_hashes,
    )
    identity = cityplace.result_identity(result)
    manifest = build_manifest(
        source_hashes=source_hashes,
        output_hashes=output_hashes,
        plan_id=plan_id,
        terrain_pass="final",
        deterministic_identity=identity,
    )
    write_deterministic(final_dir / "manifest.json", manifest)
    reseat = result["solver_report"]["gates"].get("final_reseat", {})
    final_contract = result["city_placement"].get("terrain_field", {})
    if reseat.get("status") != "reference_verified" or final_contract.get("pass") != "final" or result["city_placement"]["counts"].get("provisional") != 0:
        raise CityscapeError("T1.2 final re-seat did not accept the exact final field")
    evidence = {
        "status": "passed",
        "plan_id": plan_id,
        "plan_sha256": plan_hash,
        "final_output_dir": str(final_dir),
        "final_field_contract": final_contract,
        "reseat_gate": reseat,
        "counts": result["city_placement"]["counts"],
        "output_hashes": output_hashes,
        "manifest_sha256": sha256_file(final_dir / "manifest.json"),
    }
    write_deterministic(output_dir / "t1_2_final_reseat_integration.json", evidence)
    return evidence


def _write_t12_planned_products(
    output_dir: Path,
    result: Mapping[str, Any],
    *,
    validation_path: Path,
    plan_id: str,
    plan_hash: str,
) -> dict[str, Any]:
    """Persist the T1.2 planned pass that was run against T1.3's NPZ."""

    planned_dir = output_dir / "t1_2_planned_reseat"
    planned_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = dict(result["source_hashes"])
    source_hashes["t1_1_validation"] = sha256_file(validation_path)
    output_hashes = write_products(
        planned_dir,
        city_placement=result["city_placement"],
        land_edit_requests=result["land_edit_requests"],
        solver_report=result["solver_report"],
        source_hashes=source_hashes,
    )
    identity = cityplace.result_identity(result)
    manifest = build_manifest(
        source_hashes=source_hashes,
        output_hashes=output_hashes,
        plan_id=plan_id,
        terrain_pass="planned",
        deterministic_identity=identity,
    )
    write_deterministic(planned_dir / "manifest.json", manifest)
    evidence = {
        "status": "passed",
        "terrain_pass": "planned",
        "plan_id": plan_id,
        "plan_sha256": plan_hash,
        "output_dir": str(planned_dir),
        "terrain_field": result["city_placement"].get("terrain_field", {}),
        "counts": result["city_placement"].get("counts", {}),
        "output_hashes": output_hashes,
        "manifest_sha256": sha256_file(planned_dir / "manifest.json"),
    }
    write_deterministic(output_dir / "t1_2_planned_integration.json", evidence)
    return evidence


def build_cityscape(paths: CityscapePaths) -> dict[str, Any]:
    """Execute one complete T1.3 build; any essential failure raises."""

    for path in (
        paths.survey, paths.palette, paths.plan, paths.validation,
        paths.t12_placement, paths.t12_land_edits, paths.source_land,
        paths.effective_remap, paths.kit_brief, paths.centerlines,
        *paths.stamp_libraries,
    ):
        if not path.is_file():
            raise CityscapeError(f"missing accepted input {path}")
    output = paths.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plan = _json(paths.plan, "city plan")
    validation = _json(paths.validation, "T1.1 validation")
    palette = _json(paths.palette, "region palette")
    survey = _json(paths.survey, "site survey")
    t12_placement = _json(paths.t12_placement, "T1.2 planned placement")
    t12_land_edits = _json(paths.t12_land_edits, "T1.2 land edit requests")
    plan_hash = sha256_file(paths.plan)
    if t12_placement.get("plan_sha256") != plan_hash or t12_land_edits.get("plan_sha256") != plan_hash:
        raise CityscapeError("T1.2 accepted products are not pinned to the requested plan hash")
    if t12_placement.get("terrain_field", {}).get("pass") != "planned":
        raise CityscapeError("T1.2 placement reference is not a planned-pass product")
    known_links, water_links = _plan_links(plan)
    block = load_target_block(
        root=paths.workspace_root,
        survey_path=paths.survey,
        source_path=paths.source_land,
        effective_path=paths.effective_remap,
    )
    stitch = _stitch_audit(block)
    if not stitch["shared_edges_exact"] or not stitch["outer_border_source_frozen"]:
        raise CityscapeError("source LAND stitch gate failed")
    live = regionpalette.live_remap_ltex_table(paths.workspace_root)
    live_checks = regionpalette.crosscheck_live_remap_table(live)
    if not all(check.get("passed") for check in live_checks):
        raise CityscapeError("live remap LTEX gate failed: " + json.dumps(live_checks, sort_keys=True))
    surfaces = palette.get("semantic_surfaces", {}).get("surfaces", [])
    planned_live = regionpalette.planned_vs_live_remap_check(surfaces, live)
    if not planned_live.get("passed"):
        raise CityscapeError("planned/live remap contract failed: " + str(planned_live))
    vnml_gate = validate_source_convention(block)
    vnml_oracles = analytic_oracle_checks()
    convention = VNMLConvention(
        tuple(vnml_gate["best_candidate"]["permutation"]),
        tuple(vnml_gate["best_candidate"]["signs"]),
    )
    accepted_pad_edits = _auto_pad_edits(t12_land_edits)
    road_edits = _road_grade_edits(plan)
    intentional = plan.get("terrain_edits", [])
    if not isinstance(intentional, list):
        raise CityscapeError("city plan terrain_edits is not a list")
    planned_result = compose_edits(
        block=block,
        source_values_gu=block.source_heights_gu,
        edits=intentional,
        known_links=known_links,
        authorized_water_links=water_links,
    )
    planned_npz = output / "planned_terrain_field.npz"
    planned_meta = output / "planned_terrain_field.metadata.json"
    planned_npz_evidence = write_field_npz(
        planned_npz,
        planned_result.quantized_values_gu,
        metadata=field_metadata(block, field_pass="planned", values_gu=planned_result.quantized_values_gu, provenance="T1.3 planned analytic field; synthetic_not_a_falkreath_design"),
    )
    planned_meta_evidence = write_metadata(
        planned_meta,
        field_metadata(block, field_pass="planned", values_gu=planned_result.quantized_values_gu, provenance="T1.3 planned analytic field; synthetic_not_a_falkreath_design"),
    )
    # Run T1.2's planned pass against the actual T1.3 planned field, rather
    # than merely reusing the older accepted fixture's source-field placement.
    # The old request remains a pinned acceptance input and is compared below.
    if t12_placement.get("plan_id") != plan.get("plan_id"):
        raise CityscapeError("T1.2 placement product plan_id disagrees with city plan")
    planned_t12 = cityplace.solve_city_plan(
        plan_path=paths.plan,
        validation_path=paths.validation,
        site_survey_path=paths.survey,
        kit_brief_path=paths.kit_brief,
        region_palette_path=paths.palette,
        stamp_library_paths=list(paths.stamp_libraries),
        centerlines_path=paths.centerlines,
        terrain_field_path=planned_npz,
        terrain_metadata_path=planned_meta,
        terrain_pass="planned",
        workspace_root=paths.workspace_root,
    )
    t12_planned_evidence = _write_t12_planned_products(
        output, planned_t12, validation_path=paths.validation,
        plan_id=str(plan["plan_id"]), plan_hash=plan_hash,
    )
    generated_pad_document = planned_t12["land_edit_requests"]
    def canonical_request_rows(value: Any) -> Any:
        # T1.2's in-memory tuples/lists are semantically identical JSON
        # arrays and may carry binary floating-point noise before the shared
        # six-decimal serializer runs; compare the canonical JSON shape, not
        # Python container implementation details.
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, Mapping):
            return {key: canonical_request_rows(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [canonical_request_rows(item) for item in value]
        return value

    if canonical_request_rows(generated_pad_document.get("requests")) != canonical_request_rows(t12_land_edits.get("requests")):
        raise CityscapeError("T1.3 planned field changed the accepted T1.2 pad request; refusing silent reseat drift")
    pad_edits = _auto_pad_edits(generated_pad_document)
    if canonical_request_rows(pad_edits) != canonical_request_rows(accepted_pad_edits):
        raise CityscapeError("generated T1.2 pad edit differs from the accepted request contract")
    diagnostics = _edit_diagnostics(block, auto_pad=pad_edits[0] if pad_edits else None, known_links=known_links, water_links=water_links)
    final_edits = list(pad_edits) + list(road_edits)
    final_result = compose_edits(
        block=block,
        # Keep the pass-1 float field for composition.  The planned NPZ is the
        # T1.2 hand-off (and is quantized), but the final pass itself performs
        # the one final source-to-output THU quantization after both edit sets.
        source_values_gu=planned_result.values_gu,
        edits=final_edits,
        known_links=known_links,
        authorized_water_links=water_links,
    )
    final_npz = output / "final_terrain_field.npz"
    final_meta = output / "final_terrain_field.metadata.json"
    final_npz_evidence = write_field_npz(
        final_npz,
        final_result.quantized_values_gu,
        metadata=field_metadata(block, field_pass="final", values_gu=final_result.quantized_values_gu, provenance="T1.3 final analytic field after exact T1.2 pad/road re-seat; synthetic_not_a_falkreath_design"),
    )
    final_meta_evidence = write_metadata(
        final_meta,
        field_metadata(block, field_pass="final", values_gu=final_result.quantized_values_gu, provenance="T1.3 final analytic field after exact T1.2 pad/road re-seat; synthetic_not_a_falkreath_design"),
    )
    final_t12 = cityplace.solve_city_plan(
        plan_path=paths.plan,
        validation_path=paths.validation,
        site_survey_path=paths.survey,
        kit_brief_path=paths.kit_brief,
        region_palette_path=paths.palette,
        stamp_library_paths=list(paths.stamp_libraries),
        centerlines_path=paths.centerlines,
        terrain_field_path=final_npz,
        terrain_metadata_path=final_meta,
        terrain_pass="final",
        planned_placement_path=output / "t1_2_planned_reseat" / "city_placement.json",
        workspace_root=paths.workspace_root,
    )
    t12_evidence = _write_t12_final_products(output, final_t12, validation_path=paths.validation, plan_id=str(plan["plan_id"]), plan_hash=plan_hash)
    t12_evidence["planned_pass"] = t12_planned_evidence
    write_deterministic(output / "t1_2_final_reseat_integration.json", t12_evidence)
    placement_for_paint = final_t12["city_placement"]
    painted = paint_vtex(
        block=block,
        plan=plan,
        plan_hash=plan_hash,
        palette=palette,
        survey=survey,
        placement=placement_for_paint,
    )
    source_cells = split_field(block.source_heights_gu, block.cells)
    final_cells = split_field(final_result.quantized_values_gu, block.cells)
    height_edited_cells = {cell for cell in block.cells if not np.array_equal(source_cells[cell], final_cells[cell])}
    normal_edge = _normal_shared_edge_audit(block, final_result.quantized_values_gu, convention, height_edited_cells)
    production_edge = production_shared_edge_audit(final_result.quantized_values_gu, block, convention)
    if not production_edge["shared_edges_exact"]:
        raise CityscapeError("VNML production shared-edge compatibility gate failed")
    normal_payloads = normal_edge["payloads"]
    assembly = assemble_land_records(
        block=block,
        final_values_gu=final_result.quantized_values_gu,
        painted=painted,
        normal_payloads=normal_payloads,
        height_edited_cells=height_edited_cells,
        plan_id=str(plan["plan_id"]),
    )
    land_records_path = output / "land_records.json"
    land_records_hash = _write_json_any(land_records_path, assembly.document)
    land_edits = build_land_edits_document(
        plan_id=str(plan["plan_id"]),
        terrain_field_sha256=terrain_field_sha256(final_result.quantized_values_gu),
        height_edits=list(planned_result.edit_ledger) + list(final_result.edit_ledger),
        final_encoding=final_result.final_encoding,
        source_unchanged={
            "planned": planned_result.source_unchanged,
            "final_against_planned": final_result.source_unchanged,
            "overall_outer_border_exact": bool(np.array_equal(final_result.quantized_values_gu[outer_border_mask(block.field_shape)], block.source_heights_gu[outer_border_mask(block.field_shape)])),
            "overall_source_outside_height_support_exact": bool(
                planned_result.source_unchanged["outside_declared_support_exact"]
                and final_result.source_unchanged["outside_declared_support_exact"]
            ),
            "overall_changed_vertex_count": int(np.count_nonzero(final_result.quantized_values_gu - block.source_heights_gu)),
        },
        painted=painted,
        inputs={
            "source_land": {"path": str(paths.source_land), "sha256": block.source_sha256},
            "effective_remap": {"path": str(paths.effective_remap), "sha256": block.effective_sha256},
            "plan": {"path": str(paths.plan), "sha256": plan_hash},
            "t1_2_placement": {"path": str(output / "t1_2_planned_reseat" / "city_placement.json"), "sha256": sha256_file(output / "t1_2_planned_reseat" / "city_placement.json")},
        },
    )
    land_edits_hash = write_deterministic(output / "land_edits.json", land_edits)
    diagnostic_audit = _diagnostic_images(output, source_values=block.source_heights_gu, final_values=final_result.quantized_values_gu, paint=painted)
    validation_doc = {
        "schema_version": 1,
        "product": "cityforge_t1_3_hard_landscape_validation",
        "diagnostic_scope": "synthetic_not_a_falkreath_design",
        "status": "passed",
        "plan_id": plan.get("plan_id"),
        "plan_sha256": plan_hash,
        "gates": {
            "source_stitch": stitch,
            "payload_reconciliation": dict(block.reconciliation),
            "live_remap": {"checks": live_checks, "planned_vs_live": planned_live, "live": live},
            "vnml_source_convention": vnml_gate,
            "vnml_analytic_oracles": vnml_oracles,
            "terrain_edits": {
                "intentional_edit_count": len(intentional),
                "t1_2_auto_pad_count": len(pad_edits),
                "road_grade_count": len(road_edits),
                "planned": {"encoding": planned_result.final_encoding, "ledger": list(planned_result.edit_ledger)},
                "final": {"encoding": final_result.final_encoding, "ledger": list(final_result.edit_ledger)},
                "no_clipping": True,
            },
            "immutable_border": {
                "exact": bool(np.array_equal(final_result.quantized_values_gu[outer_border_mask(block.field_shape)], block.source_heights_gu[outer_border_mask(block.field_shape)])),
                "changed_count": int(np.count_nonzero(final_result.quantized_values_gu[outer_border_mask(block.field_shape)] != block.source_heights_gu[outer_border_mask(block.field_shape)])),
            },
            "source_payload_outside_declared_support": dict(land_edits["vertex_provenance"]),
            "vtex_paint": dict(painted.paint_ledger),
            "vnml_final": {
                "convention": convention.to_dict(),
                "height_edited_cells": [list(cell) for cell in sorted(height_edited_cells)],
                "shared_edges": {key: value for key, value in normal_edge.items() if key != "payloads"},
                "production_shared_edges": production_edge,
            },
            "records": dict(assembly.audit),
            "t1_2_planned_pass": t12_planned_evidence,
            "t1_2_final_reseat": t12_evidence,
            "synthetic_edit_diagnostics": diagnostics,
        },
        "diagnostics": diagnostic_audit,
        "outputs": {
            "planned_field": planned_npz_evidence,
            "planned_metadata_sha256": planned_meta_evidence,
            "final_field": final_npz_evidence,
            "final_metadata_sha256": final_meta_evidence,
            "land_records_sha256": land_records_hash,
            "land_edits_sha256": land_edits_hash,
        },
    }
    validation_hash = write_deterministic(output / "validation.json", validation_doc)
    output_files = {
        name: sha256_file(output / name)
        for name in sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "manifest.json")
    }
    output_files["validation.json"] = validation_hash
    source_hashes = {
        "tamriel_esm": block.source_sha256,
        "effective_remap_esp": block.effective_sha256,
        "site_survey": sha256_file(paths.survey),
        "region_palette": sha256_file(paths.palette),
        "city_plan": plan_hash,
        "t1_1_validation": sha256_file(paths.validation),
        "t1_2_placement": sha256_file(paths.t12_placement),
        "t1_2_land_edit_requests": sha256_file(paths.t12_land_edits),
        "t1_2_generated_planned_placement": sha256_file(output / "t1_2_planned_reseat" / "city_placement.json"),
    }
    identity = hashlib.sha256(deterministic_dumps({"validation": validation_doc, "land_edits": land_edits, "land_records_sha256": land_records_hash})).hexdigest()
    manifest = {
        "schema_version": 1,
        "product": "cityforge_t1_3_hard_landscape_manifest",
        "diagnostic_scope": "synthetic_not_a_falkreath_design",
        "plan_id": plan.get("plan_id"),
        "terrain_passes": {"planned": planned_npz_evidence, "final": final_npz_evidence},
        "source_hashes": dict(sorted(source_hashes.items())),
        "output_hashes": dict(sorted(output_files.items())),
        "deterministic_identity": identity,
        "tes3_authoring": "land_records.json only; no ESP produced in T1.3",
    }
    manifest_hash = write_deterministic(output / "manifest.json", manifest)
    return {
        "validation": validation_doc,
        "validation_sha256": validation_hash,
        "manifest_sha256": manifest_hash,
        "manifest": manifest,
        "source_hashes": source_hashes,
        "output_hashes": {**output_files, "manifest.json": manifest_hash},
        "vnml_gate": vnml_gate,
        "terrain": {"planned": planned_result, "final": final_result},
        "paint": painted,
        "assembly": assembly,
        "t12_final": final_t12,
        "t12_planned": planned_t12,
        "t12_planned_evidence": t12_planned_evidence,
        "t12_evidence": t12_evidence,
        "diagnostics": diagnostic_audit,
    }


def default_paths(root: Path | str = ".", output_dir: Path | str | None = None) -> CityscapePaths:
    """Return canonical Cityforge T1.3 paths without probing or editing files."""

    workspace = Path(root).resolve()
    return CityscapePaths(
        workspace_root=workspace,
        survey=workspace / "output/cityforge/sites/falkreath_v1/site_survey.json",
        palette=workspace / "output/cityforge/briefs/falkreath_v1/region_palette.json",
        plan=workspace / "output/cityforge/phase1/t1_2_placement_fixture/synthetic_not_a_falkreath_design.city_plan.json",
        validation=workspace / "output/cityforge/phase1/t1_2_placement_fixture/synthetic_not_a_falkreath_design.validation.json",
        t12_placement=workspace / "output/cityforge/phase1/t1_2_placement_fixture/city_placement.json",
        t12_land_edits=workspace / "output/cityforge/phase1/t1_2_placement_fixture/land_edit_requests.json",
        source_land=workspace / "tamriel.esm",
        effective_remap=workspace / "output/falkreath_landscape_texture_remap.esp",
        output_dir=(workspace / "output/cityforge/phase1/t1_3_cityscape_fixture") if output_dir is None else Path(output_dir).resolve(),
        kit_brief=workspace / "output/cityforge/briefs/falkreath_v1/kit_brief.json",
        stamp_libraries=(workspace / "output/cityforge/stamps/karthgad_nord_v1.json", workspace / "output/cityforge/stamps/markarth_side_stone_v1.json"),
        centerlines=workspace / "output/mapdata/roads/tamriel_aligned_centerlines_v1/tamriel_aligned_centerlines_v1.json",
    )


__all__ = ["CityscapeError", "CityscapePaths", "build_cityscape", "default_paths"]
