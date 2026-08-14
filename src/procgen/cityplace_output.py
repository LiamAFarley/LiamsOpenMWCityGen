"""Deterministic T1.2 output assembly and audit serialization.

Pipeline position
------------------
This module is the final host-side stage of Cityforge T1.2.  It receives the
orchestrator's measured placement records, rejected-lot records, pad requests,
source/oracle evidence, and input pins, and writes the four required JSON
products.  It never emits TES3 records or invokes tes3conv; that belongs to
T1.4 after the user-gated plan/landscape stages.

Outputs
-------
``city_placement.json`` contains only emitted house refs and their exact raw
TES3 transforms; ``land_edit_requests.json`` contains explicit provisional pad
requests; ``solver_report.json`` contains every measured check and structured
outcome; ``manifest.json`` records all source/output hashes.  JSON uses the
shared ``censusio`` serializer so reruns are byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .censusio import deterministic_dumps, sha256_file, write_deterministic


def build_city_placement(
    *,
    plan_id: str,
    plan_sha256: str,
    terrain_contract: Mapping[str, Any],
    frame: Mapping[str, Any],
    placements: list[Mapping[str, Any]],
    provisional: list[Mapping[str, Any]],
    rejected: list[Mapping[str, Any]],
    fine_collision_deferred: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the JSON-ready placement product with no hidden omissions."""

    return {
        "schema_version": 1,
        "product": "cityforge_t1_2_houses_only_placement",
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "terrain_field": dict(terrain_contract),
        "frame": dict(frame),
        "placements": sorted((dict(item) for item in placements), key=lambda item: str(item.get("lot_id"))),
        "provisional_pad_lots": sorted(
            (dict(item) for item in provisional), key=lambda item: str(item.get("lot_id"))
        ),
        "rejected_lots": sorted(
            (dict(item) for item in rejected), key=lambda item: str(item.get("lot_id"))
        ),
        "fine_collision_deferred": sorted(
            (dict(item) for item in fine_collision_deferred),
            key=lambda item: (str(item.get("lot_id")), str(item.get("stamp_id"))),
        ),
        "counts": {
            "accepted": len(placements),
            "provisional": len(provisional),
            "rejected": len(rejected),
            "fine_collision_deferred": len(fine_collision_deferred),
        },
    }


def build_land_edit_requests(
    *,
    plan_id: str,
    plan_sha256: str,
    terrain_contract: Mapping[str, Any],
    requests: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble exact footprint+margin pad requests, preserving provisional state."""

    return {
        "schema_version": 1,
        "product": "cityforge_t1_2_land_edit_requests",
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "terrain_field": dict(terrain_contract),
        "requests": sorted((dict(item) for item in requests), key=lambda item: str(item.get("lot_id"))),
        "final_reseat_required": bool(requests),
        "notes": (
            "T1.2 emits requests only; T1.3 must create a final field and rerun "
            "cityplace before any pad lot becomes accepted."
        ),
    }


def build_manifest(
    *,
    source_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
    plan_id: str,
    terrain_pass: str,
    deterministic_identity: str,
) -> dict[str, Any]:
    """Return the final hash/audit manifest."""

    return {
        "schema_version": 1,
        "product": "cityforge_t1_2_placement_manifest",
        "plan_id": plan_id,
        "terrain_pass": terrain_pass,
        "source_hashes": dict(sorted(source_hashes.items())),
        "output_hashes": dict(sorted(output_hashes.items())),
        "deterministic_identity": deterministic_identity,
        "tes3_authoring": "not produced in T1.2",
    }


def write_products(
    output_dir: Path | str,
    *,
    city_placement: Mapping[str, Any],
    land_edit_requests: Mapping[str, Any],
    solver_report: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, str]:
    """Write products in a fixed order and return their output hashes.

    The manifest is written last because it pins the already-written product
    bytes.  It is then re-read and its own hash is included in the returned
    audit map by the CLI's final manifest update pass.
    """

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payloads = {
        "city_placement.json": city_placement,
        "land_edit_requests.json": land_edit_requests,
        "solver_report.json": solver_report,
    }
    hashes: dict[str, str] = {}
    for name in sorted(payloads):
        hashes[name] = write_deterministic(target / name, payloads[name])
    return hashes


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Expose the exact bytes used for diagnostic identity tests."""

    return deterministic_dumps(payload)
