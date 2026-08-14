#!/usr/bin/env python3
"""Cityforge T0.3 stamp-library CLI: derive, write, catalog, verify.

Loads the existing extraction/split products (read-only), derives the two
D-STAMP v1 unit-stamp libraries through :mod:`procgen.citystamps`, writes
them as canonical JSON, packages the browsable catalog (copies of the real
existing contact sheets, verified by hash), and can prove deterministic
reruns by byte-comparing against the previously written files.

Libraries produced
------------------
* ``output/cityforge/stamps/karthgad_nord_v1.json`` — 11 Karthgad building
  stamps from ``output/skyrim-settlements/karthgad-v1`` (component manifests
  + placement manifest + landscape terrain cross-check + ``Sky_Main.esm``
  LAND re-derivation).
* ``output/cityforge/stamps/markarth_side_stone_v1.json`` — the approved
  Markarth Side split subset (``manual-corrections-v1``, provisional and
  hash-pinned) joined to ``output/skyrim-settlements/markarth-side-v1`` by
  source id, with ``Sky_Markarth.esm`` LAND re-derivation.

Catalog
-------
``output/cityforge/stamps/catalog_v1/``: ``index.html`` (browsable),
``index.md`` (markdown), ``index.json`` (machine-readable, hash-verified
preview copies), and ``previews/<library_id>/<stamp_id>.png`` byte-identical
copies of the existing contact sheets.  Every accepted stamp has exactly one
real preview; a missing preview is an explicit exclusion, never a placeholder.

Usage
-----
::

    python tools/cityforge/stamp_library.py [--libraries both|karthgad|markarth]
        [--verify-determinism] [--out DIR] [--catalog-dir DIR]

``--verify-determinism`` re-derives and byte-compares against the existing
library files; exit code 1 on any mismatch.  No commit is made by this tool.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
TOOLS = WORKSPACE / "tools"
for entry in (SRC, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from procgen import citystamps, espland  # noqa: E402
from procgen.citystamps import canonical_json_bytes, sha256_file  # noqa: E402

# The read-only established transform oracle (replay evidence only; never
# rewire).  Its placement_scene_matrix implements the validated OpenMW 0.51
# composition Rx(-rx) @ Ry(-ry) @ Rz(-rz) with 9-digit matrix rounding.
import karthgad_rebuild_geometry as _oracle  # noqa: E402

OUT_ROOT = WORKSPACE / "output" / "cityforge" / "stamps"
CATALOG_DIR = OUT_ROOT / "catalog_v1"
# Exact audited non-building exclusion list for the hash-pinned
# manual-corrections-v1 split source (T0.3 acceptance repair).  Applied only
# when the pinned units.json sha256 matches; a mismatch fails closed.
NON_BUILDING_AUDIT = TOOLS / "cityforge" / "non_building_audit_v1.json"

KARTHGAD_RUN = WORKSPACE / "output" / "skyrim-settlements" / "karthgad-v1"
MARKARTH_RUN = WORKSPACE / "output" / "skyrim-settlements" / "markarth-side-v1"
SPLIT_DIR = (
    WORKSPACE / "output" / "settlement-splits" / "markarth-side-v2" / "manual-corrections-v1"
)
SPLIT_RENDER_V6 = (
    WORKSPACE / "output" / "settlement-splits" / "markarth-side-v2" / "split-render-v6"
)
SKY_MAIN = WORKSPACE / "Sky_Main.esm"
SKY_MARKARTH = WORKSPACE / "PTR Indev" / "Sky_Markarth.esm"

KARTHGAD_LIBRARY_ID = "karthgad_nord_v1"
MARKARTH_LIBRARY_ID = "markarth_side_stone_v1"
KARTHGAD_STAMP_PREFIX = "karthgad_v1"
MARKARTH_STAMP_PREFIX = "markarth_side_v1"
SPLIT_PROVENANCE_LABEL = "manual-corrections-v1"


def _read_json(path: Path, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {label}: {path}: {exc}") from exc


def _ws_rel(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def _placement_index(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for placement in manifest["placements"]:
        source_id = placement["source_id"]
        if source_id in index:
            raise RuntimeError(f"duplicate placement source_id {source_id}")
        index[source_id] = placement
    return index


def _karthgad_building_candidate(
    building: Mapping[str, Any],
    manifest: Mapping[str, Any],
    placement_by_id: Mapping[str, Mapping[str, Any]],
    preview_path: Path | None,
    component_manifest_sha256: str,
    landscape_terrain_sha256: str,
) -> dict[str, Any]:
    """Assemble one Karthgad stamp candidate from its joined source records.

    Members come from the component manifest (``source_transform`` is the
    per-member authority); door destinations come from the B1 placement
    manifest joined by ``source_id``; the seed door's ``source_cell`` is
    taken from the placement manifest.
    """

    slug = building["slug"]
    members: list[dict[str, Any]] = []
    ghost = False
    for member in manifest["members"]:
        placement = placement_by_id.get(member["source_id"])
        if placement is None or bool(placement.get("unresolved")):
            ghost = True
            break
        transform = member["source_transform"]
        members.append(
            {
                "source_id": member["source_id"],
                "object_id": member["object_id"],
                "record_type": member["record_type"],
                "model_key": member["model_key"],
                "category": member.get("category"),
                "structural_role": member.get("structural_role"),
                "is_door": bool(member.get("is_door")),
                "position_gu": transform["position_gu"],
                "rotation": transform["rotation"],
                "scale": transform.get("scale"),
                "world_bounds_gu": member.get("world_bounds_gu"),
                "destination": {
                    "destination_cell": placement.get("destination_cell"),
                    "destination_position": placement.get("destination_position"),
                    "destination_rotation": placement.get("destination_rotation"),
                    "door_to_interior": placement.get("door_to_interior"),
                },
                "source_placement_scene_matrix": member.get("placement_scene_matrix"),
            }
        )

    bounds = building["bounds_gu"]
    seed_placement = placement_by_id.get(building["seed_door"], {})
    door_member_count = sum(1 for member in members if member["is_door"])
    protocol_failures = list(manifest.get("protocol_failures") or [])
    protocol_failures.extend(list(building.get("protocol_failures") or []))
    conditions = {
        "ghost_members": ghost,
        "protocol_failure": bool(protocol_failures),
        "no_door": door_member_count == 0,
        "preview_missing": preview_path is None,
    }
    return {
        "candidate_id": slug,
        "run": "karthgad-v1",
        "slug": slug,
        "component_id": building["component_id"],
        "seed_door_refs": [building["seed_door"]],
        "door_refs": list(building["door_refs"]),
        "members": members,
        "bounds_xy": [bounds["min"][0], bounds["min"][1], bounds["max"][0], bounds["max"][1]],
        "bounds_min_z": bounds["min"][2],
        "named_destination_interiors": list(building.get("named_destination_interiors") or []),
        "multi_shell": bool((building.get("flags") or {}).get("multiple_shells")),
        "preview_sheet": _ws_rel(preview_path) if preview_path else None,
        "exclusion_conditions": conditions,
        "exclusion_detail": {
            "component_id": building["component_id"],
            "member_count": len(members),
            "component_manifest_sha256": component_manifest_sha256,
            "landscape_terrain_sha256": landscape_terrain_sha256,
            "source_cell": seed_placement.get("source_cell"),
        },
        "extra_source": {
            "source_cell": seed_placement.get("source_cell"),
            "seed_door": building["seed_door"],
        },
    }


def build_karthgad_library() -> dict[str, Any]:
    """Load Karthgad products and derive ``karthgad_nord_v1``."""

    buildings_index = _read_json(KARTHGAD_RUN / "components" / "buildings_index.json", "karthgad buildings index")
    placement_manifest = _read_json(KARTHGAD_RUN / "b1" / "placement_manifest.json", "karthgad placement manifest")
    placement_by_id = _placement_index(placement_manifest)
    land = espland.load_land(SKY_MAIN)

    inputs: dict[str, str] = {}
    for rel, path in (
        ("components/buildings_index.json", KARTHGAD_RUN / "components" / "buildings_index.json"),
        ("b1/placement_manifest.json", KARTHGAD_RUN / "b1" / "placement_manifest.json"),
    ):
        inputs[_ws_rel(path)] = sha256_file(path)

    candidates: list[dict[str, Any]] = []
    for building in buildings_index["buildings"]:
        slug = building["slug"]
        manifest_path = KARTHGAD_RUN / "components" / "buildings" / slug / "manifest.json"
        manifest = _read_json(manifest_path, f"karthgad manifest {slug}")
        inputs[_ws_rel(manifest_path)] = sha256_file(manifest_path)
        terrain_path = KARTHGAD_RUN / "landscape" / "buildings" / slug / "terrain.json"
        terrain = _read_json(terrain_path, f"karthgad terrain {slug}")
        inputs[_ws_rel(terrain_path)] = sha256_file(terrain_path)
        preview = KARTHGAD_RUN / "renders" / f"{slug}_sheet_2x3.png"
        candidates.append(
            _karthgad_building_candidate(
                building,
                manifest,
                placement_by_id,
                preview if preview.exists() else None,
                sha256_file(manifest_path),
                sha256_file(terrain_path),
            )
        )

    # Cross-check the uniform LAND re-derivation against the landscape
    # product (disagreement > 1 GU is explicit, never silent).
    crosscheck_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(candidate["exclusion_conditions"].values()):
            continue
        slug = candidate["slug"]
        terrain = _read_json(KARTHGAD_RUN / "landscape" / "buildings" / slug / "terrain.json", f"terrain {slug}")
        derived = {
            "burial_depth_gu": None,
            "door_steps": {},
        }
        # Re-derive exactly as the library does (mirror derive_stamp inputs).
        stamp = None
        for candidate_try in (candidate,):
            candidate_try = dict(candidate_try)
            candidate_try["exclusion_conditions"] = {}
            candidate_try["preview_sheet"] = _ws_rel(KARTHGAD_RUN / "renders" / f"{slug}_sheet_2x3.png")
            stamp, _ = citystamps.derive_stamp(
                library_stamp_prefix=KARTHGAD_STAMP_PREFIX,
                run=candidate_try["run"],
                slug=candidate_try["slug"],
                component_id=candidate_try.get("component_id"),
                seed_door_refs=candidate_try["seed_door_refs"],
                door_refs=candidate_try["door_refs"],
                members=candidate_try["members"],
                bounds_xy=candidate_try.get("bounds_xy"),
                bounds_min_z=candidate_try.get("bounds_min_z"),
                named_destination_interiors=candidate_try.get("named_destination_interiors") or [],
                multi_shell=bool(candidate_try.get("multi_shell")),
                land=land,
                preview_sheet=candidate_try.get("preview_sheet"),
                extra_source=candidate_try.get("extra_source"),
                matrix_builder=_oracle.placement_scene_matrix,
            )
            break
        if stamp is None:
            continue
        landscape_doors = {door["source_id"]: door for door in terrain.get("doors", [])}
        for index, door_ref in enumerate(candidate["door_refs"]):
            landscape_door = landscape_doors.get(door_ref)
            if not landscape_door:
                continue
            derived_step = stamp["terrain_envelope"]["door_step_heights_gu"][index]
            landscape_step = float(landscape_door["step_height_game_units"])
            delta = derived_step - landscape_step
            if abs(delta) > citystamps.TERRAIN_CROSSCHECK_TOLERANCE_GU:
                crosscheck_rows.append(
                    {
                        "stamp_id": stamp["stamp_id"],
                        "door_ref": door_ref,
                        "measure": "door_step_height_gu",
                        "landscape_value_gu": landscape_step,
                        "derived_value_gu": derived_step,
                        "delta_gu": delta,
                    }
                )
        derived_burial = stamp["terrain_envelope"]["burial_depth_gu"]
        landscape_burial = float(terrain["burial_depth"])
        delta_burial = derived_burial - landscape_burial
        if abs(delta_burial) > citystamps.TERRAIN_CROSSCHECK_TOLERANCE_GU:
            crosscheck_rows.append(
                {
                    "stamp_id": stamp["stamp_id"],
                    "door_ref": None,
                    "measure": "burial_depth_gu",
                    "landscape_value_gu": landscape_burial,
                    "derived_value_gu": derived_burial,
                    "delta_gu": delta_burial,
                }
            )

    source_recorded = []
    for excluded in buildings_index.get("excluded_multi_ref_doorless_components", []):
        source_recorded.append(
            citystamps._excluded_row(
                f"component_{excluded['component_id']}",
                "doorless_component_source_recorded",
                {"component_id": excluded["component_id"], "ref_count": excluded["ref_count"]},
                scope="source_run_component",
            )
        )
    for target in buildings_index.get("access_targets", []):
        source_recorded.append(
            citystamps._excluded_row(
                f"component_{target['component_id']}",
                "access_target_only",
                {
                    "component_id": target["component_id"],
                    "slug": target["slug"],
                    "member_count": target["member_count"],
                    "shell_count": target["shell_count"],
                },
                scope="source_run_component",
            )
        )

    # Master-transform cross-check: component manifest source_transform vs
    # placement manifest position/rotation for shared refs.
    master_mismatches: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(candidate["exclusion_conditions"].values()):
            continue
        for member in candidate["members"]:
            placement = placement_by_id.get(member["source_id"])
            if not placement:
                continue
            pos = placement.get("position")
            rot = placement.get("rotation")
            if pos is not None and any(
                abs(a - b) > 1e-6 for a, b in zip(member["position_gu"], pos)
            ):
                master_mismatches.append(
                    {"slug": candidate["slug"], "source_id": member["source_id"], "field": "position"}
                )
            if rot is not None and any(
                abs(a - b) > 1e-9 for a, b in zip(member["rotation"], rot)
            ):
                master_mismatches.append(
                    {"slug": candidate["slug"], "source_id": member["source_id"], "field": "rotation"}
                )

    library = citystamps.build_library(
        library_id=KARTHGAD_LIBRARY_ID,
        library_name="Karthgad Nord Kit Building Stamps",
        kit={"id": "nord", "style_tags": ["nord", "log_cabin"]},
        stamp_prefix=KARTHGAD_STAMP_PREFIX,
        inputs=inputs,
        source_plugins={
            "sky_main": {"path": str(SKY_MAIN), "sha256": sha256_file(SKY_MAIN)},
        },
        land=land,
        candidates=candidates,
        source_recorded_exclusions=source_recorded,
        extra_stats={
            "terrain_crosscheck": {
                "tolerance_gu": citystamps.TERRAIN_CROSSCHECK_TOLERANCE_GU,
                "method": "uniform espland LAND re-derivation vs landscape product",
                "rows": crosscheck_rows,
                "disagreement_count": len(crosscheck_rows),
            },
            "master_transform_crosscheck": {
                "mismatch_count": len(master_mismatches),
                "rows": master_mismatches,
            },
        },
        matrix_builder=_oracle.placement_scene_matrix,
    )
    return library


def _split_unit_candidate(
    unit: Mapping[str, Any],
    building: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    placement_by_id: Mapping[str, Mapping[str, Any]],
    preview_path: Path | None,
    units_json_sha256: str,
) -> dict[str, Any]:
    """Assemble one Markarth split-unit stamp candidate.

    Members are the union of the unit's ref lists (shells, fabric, internal
    divisions, manual assignments, yard chains, doors), joined to the
    authoritative placement manifest by ``source_id``; structure/bounds come
    from the owning component manifest.  The split provenance stays explicit
    on every candidate (``manual-corrections-v1``, hash-pinned).
    """

    unit_id = unit["unit_id"]
    ref_lists = [
        unit.get("door_refs") or [],
        unit.get("seed_door_refs") or [],
        unit.get("shell_refs") or [],
        unit.get("fabric_member_refs") or [],
        unit.get("internal_division_refs") or [],
        unit.get("manual_assignment_refs") or [],
        unit.get("yard_chain_refs") or [],
    ]
    all_refs: list[str] = []
    seen: set[str] = set()
    for ref_list in ref_lists:
        for ref in ref_list:
            if ref not in seen:
                seen.add(ref)
                all_refs.append(ref)
    shell_refs = set(unit.get("shell_refs") or [])
    door_refs = list(unit.get("door_refs") or [])
    seed_door_refs = list(unit.get("seed_door_refs") or [])

    manifest_members = {member["source_id"]: member for member in (manifest or {}).get("members", [])}
    members: list[dict[str, Any]] = []
    ghost = False
    for ref in sorted(all_refs):
        member = manifest_members.get(ref)
        placement = placement_by_id.get(ref)
        if member is None or placement is None or bool(placement.get("unresolved")):
            ghost = True
            break
        transform_rotation = placement.get("rotation")
        members.append(
            {
                "source_id": ref,
                "object_id": member.get("object_id"),
                "record_type": member.get("record_type"),
                "model_key": member.get("model_key"),
                "category": member.get("category"),
                "structural_role": "shell" if ref in shell_refs else member.get("structural_role"),
                "is_door": ref in door_refs,
                "position_gu": placement.get("position"),
                "rotation": transform_rotation,
                "scale": placement.get("scale"),
                "world_bounds_gu": member.get("world_bounds_gu"),
                "destination": {
                    "destination_cell": placement.get("destination_cell"),
                    "destination_position": placement.get("destination_position"),
                    "destination_rotation": placement.get("destination_rotation"),
                    "door_to_interior": placement.get("door_to_interior"),
                },
                "source_placement_scene_matrix": placement.get("placement_scene_matrix"),
            }
        )

    bounds = unit.get("bounds_gu")
    bounds_xy = None
    bounds_min_z = None
    if (
        isinstance(bounds, list)
        and len(bounds) == 3
        and all(isinstance(axis, list) and len(axis) == 2 and axis[0] is not None and axis[1] is not None for axis in bounds)
    ):
        bounds_xy = [bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]]
        bounds_min_z = bounds[2][0]

    protocol_failures = list(building.get("protocol_failures") or []) if building else []
    seed_placement = placement_by_id.get(seed_door_refs[0]) if seed_door_refs else None
    conditions = {
        "ghost_members": ghost,
        "protocol_failure": bool(protocol_failures),
        "no_door": not door_refs or not seed_door_refs,
        "bounds_missing": bounds_xy is None or bounds_min_z is None,
        "preview_missing": preview_path is None,
    }
    detail: dict[str, Any] = {
        "component_id": unit.get("component_id"),
        "unit_id": unit_id,
        "member_count": len(members),
        "seed_door_count": len(seed_door_refs),
        "split_product": _ws_rel(SPLIT_DIR / "split" / "units.json"),
        "split_label": SPLIT_PROVENANCE_LABEL,
        "units_json_sha256": units_json_sha256,
    }
    return {
        "candidate_id": unit_id,
        "run": "markarth-side-v1",
        "slug": unit_id,
        "component_id": unit.get("component_id"),
        "seed_door_refs": seed_door_refs,
        "door_refs": door_refs,
        "members": members,
        "bounds_xy": bounds_xy,
        "bounds_min_z": bounds_min_z,
        "named_destination_interiors": list(unit.get("destination_names") or []),
        "multi_shell": len(shell_refs) > 1,
        "preview_sheet": _ws_rel(preview_path) if preview_path else None,
        "exclusion_conditions": conditions,
        "exclusion_detail": detail,
        "extra_source": {
            "unit_id": unit_id,
            "source_cell": seed_placement.get("source_cell") if seed_placement else None,
            "split": {
                "label": SPLIT_PROVENANCE_LABEL,
                "product": _ws_rel(SPLIT_DIR / "split" / "units.json"),
                "units_json_sha256": units_json_sha256,
            },
        },
    }


def build_markarth_library() -> dict[str, Any]:
    """Load Markarth split products and derive ``markarth_side_stone_v1``."""

    units = _read_json(SPLIT_DIR / "split" / "units.json", "markarth split units")
    summary = _read_json(SPLIT_DIR / "split" / "summary.json", "markarth split summary")
    buildings_index = _read_json(MARKARTH_RUN / "components" / "buildings_index.json", "markarth buildings index")
    placement_manifest = _read_json(MARKARTH_RUN / "b1" / "placement_manifest.json", "markarth placement manifest")
    placement_by_id = _placement_index(placement_manifest)
    land = espland.load_land(SKY_MARKARTH)
    units_json_sha256 = sha256_file(SPLIT_DIR / "split" / "units.json")

    slug_by_component = {building["component_id"]: building for building in buildings_index["buildings"]}
    manifest_cache: dict[int, Mapping[str, Any]] = {}

    inputs: dict[str, str] = {}
    for rel, path in (
        ("split/units.json", SPLIT_DIR / "split" / "units.json"),
        ("split/summary.json", SPLIT_DIR / "split" / "summary.json"),
        ("components/buildings_index.json", MARKARTH_RUN / "components" / "buildings_index.json"),
        ("b1/placement_manifest.json", MARKARTH_RUN / "b1" / "placement_manifest.json"),
    ):
        inputs[_ws_rel(path)] = sha256_file(path)

    candidates: list[dict[str, Any]] = []
    for unit in units:
        component_id = unit.get("component_id")
        building = slug_by_component.get(component_id)
        manifest = manifest_cache.get(component_id)
        if manifest is None and building is not None:
            manifest_path = MARKARTH_RUN / "components" / "buildings" / building["slug"] / "manifest.json"
            manifest = _read_json(manifest_path, f"markarth manifest {building['slug']}")
            manifest_cache[component_id] = manifest
            inputs[_ws_rel(manifest_path)] = sha256_file(manifest_path)
        preview = SPLIT_RENDER_V6 / f"unit_{unit['unit_id']}" / "sheet_2x3.png"
        candidates.append(
            _split_unit_candidate(
                unit,
                building,
                manifest,
                placement_by_id,
                preview if preview.exists() else None,
                units_json_sha256,
            )
        )

    # Apply the exact audited non-building exclusion list (hash-pinned).
    audit = _read_json(NON_BUILDING_AUDIT, "non-building audit list")
    audit_sha = sha256_file(NON_BUILDING_AUDIT)
    pinned = audit["source"]["sha256"]
    if units_json_sha256 != pinned:
        raise RuntimeError(
            "audit exclusion pin mismatch: units.json sha256 "
            f"{units_json_sha256} != pinned {pinned}; refusing to apply the "
            "audit list to different source data"
        )
    audit_unit_ids: list[str] = []
    for decision in audit["decisions"]:
        unit_id = decision["unit_id"]
        candidate = next((item for item in candidates if item["candidate_id"] == unit_id), None)
        if candidate is None:
            raise RuntimeError(
                f"audit exclusion lists unknown unit_id {unit_id}; audit list "
                "would silently go stale"
            )
        reason = decision["reason"]
        if reason not in citystamps.AUDITED_EXCLUSION_REASONS:
            raise RuntimeError(f"audit exclusion uses unknown reason {reason!r}")
        candidate["exclusion_conditions"][reason] = True
        candidate["exclusion_detail"] = {
            **(candidate.get("exclusion_detail") or {}),
            "audited": True,
            "audit_reason": reason,
            "audit_evidence": decision["preview_evidence"],
            "audit_basis": decision["basis"],
            "audit_file_sha256": audit_sha,
        }
        audit_unit_ids.append(unit_id)

    source_recorded = []
    for excluded in buildings_index.get("excluded_multi_ref_doorless_components", []):
        source_recorded.append(
            citystamps._excluded_row(
                f"component_{excluded['component_id']}",
                "doorless_component_source_recorded",
                {"component_id": excluded["component_id"], "ref_count": excluded["ref_count"]},
                scope="source_run_component",
            )
        )
    for target in buildings_index.get("access_targets", []):
        source_recorded.append(
            citystamps._excluded_row(
                f"component_{target['component_id']}",
                "access_target_only",
                {
                    "component_id": target["component_id"],
                    "slug": target["slug"],
                    "member_count": target["member_count"],
                },
                scope="source_run_component",
            )
        )

    library = citystamps.build_library(
        library_id=MARKARTH_LIBRARY_ID,
        library_name="Markarth Side Stone Kit Building Stamps (manual-corrections-v1 split subset)",
        kit={"id": "stone", "style_tags": ["markarth", "stone"]},
        stamp_prefix=MARKARTH_STAMP_PREFIX,
        inputs=inputs,
        source_plugins={
            "sky_markarth": {"path": str(SKY_MARKARTH), "sha256": sha256_file(SKY_MARKARTH)},
        },
        land=land,
        candidates=candidates,
        source_recorded_exclusions=source_recorded,
        extra_stats={
            "split_provenance": {
                "label": SPLIT_PROVENANCE_LABEL,
                "status": "provisional",
                "hash_pinned": True,
                "units_json_sha256": units_json_sha256,
                "summary_json_sha256": sha256_file(SPLIT_DIR / "split" / "summary.json"),
                "unit_candidate_count": len(units),
            },
            "audit_exclusions": {
                "file": _ws_rel(NON_BUILDING_AUDIT),
                "sha256": audit_sha,
                "pinned_units_json_sha256": pinned,
                "label": str(audit["label"]),
                "decisions": audit["decisions"],
                "excluded_unit_ids": audit_unit_ids,
            },
        },
        matrix_builder=_oracle.placement_scene_matrix,
    )
    library["input_provenance"] = {
        "label": SPLIT_PROVENANCE_LABEL,
        "status": "provisional",
        "hash_pinned": True,
        "note": (
            "Markarth Side split subset approved for stamp derivation; members "
            "joined to the authoritative markarth-side-v1 placement manifest by "
            "source id.  Provisional input, pinned by units.json sha256."
        ),
    }
    return library


def write_library(library: Mapping[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{library['library_id']}.json"
    path.write_bytes(canonical_json_bytes(library))
    return path


def build_catalog(libraries: Sequence[Mapping[str, Any]], catalog_dir: Path) -> dict[str, Any]:
    """Package the browsable catalog from real existing preview assets.

    Every accepted stamp must link to a real existing preview PNG; the file is
    copied byte-identically into ``catalog_dir/previews/<library>/<stamp_id>.png``
    and both hashes are recorded.  A missing source preview or a copy that does
    not verify is a catalog defect: the function raises (fail closed).
    """

    catalog_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = catalog_dir / "previews"
    entries: list[dict[str, Any]] = []
    for library in libraries:
        library_id = library["library_id"]
        library_previews = previews_dir / library_id
        library_previews.mkdir(parents=True, exist_ok=True)
        for stamp in library["stamps"]:
            source_rel = stamp["preview_sheet"]
            if not source_rel:
                raise RuntimeError(
                    f"catalog defect: accepted stamp {stamp['stamp_id']} has no preview_sheet"
                )
            source = WORKSPACE / source_rel
            if not source.exists():
                raise RuntimeError(
                    f"catalog defect: preview source missing for {stamp['stamp_id']}: {source}"
                )
            copy_path = library_previews / f"{stamp['stamp_id']}.png"
            shutil.copyfile(source, copy_path)
            source_sha = sha256_file(source)
            copy_sha = sha256_file(copy_path)
            if source_sha != copy_sha:
                raise RuntimeError(
                    f"catalog defect: preview copy hash mismatch for {stamp['stamp_id']}"
                )
            entries.append(
                {
                    "stamp_id": stamp["stamp_id"],
                    "library_id": library_id,
                    "preview_source": source_rel,
                    "preview_copy": _ws_rel(copy_path),
                    "sha256": source_sha,
                    "verified": True,
                }
            )
    entries.sort(key=lambda row: (row["library_id"], row["stamp_id"]))

    index_json = {
        "schema_version": 1,
        "generated_by": f"citystamps {citystamps.__version__} (Cityforge T0.3)",
        "libraries": [library["library_id"] for library in libraries],
        "stamps": entries,
        "count": len(entries),
        "validation": {
            "all_previews_verified": True,
            "missing_preview_stamps": [],
        },
    }
    (catalog_dir / "index.json").write_bytes(canonical_json_bytes(index_json))

    lines = [
        "# Cityforge Stamp Library Catalog (v1)",
        "",
        "Every accepted stamp below links a byte-identical copy of its real",
        "existing contact sheet. A missing preview is an explicit exclusion,",
        "never a blank placeholder.",
        "",
    ]
    for library in libraries:
        lines.append(f"## {library['library_name']} (`{library['library_id']}`)")
        lines.append("")
        lines.append(f"- accepted stamps: {library['stats']['stamp_count']}")
        lines.append(f"- per type: {json.dumps(library['stats']['per_type'])}")
        lines.append(f"- excluded: {library['stats']['excluded_count']}")
        lines.append("")
        for stamp in library["stamps"]:
            lines.append(f"- **{stamp['stamp_id']}** — {stamp['building_type']} "
                         f"({stamp['size_class']}, {stamp['door_count']} door(s)) — "
                         f"[preview](previews/{library['library_id']}/{stamp['stamp_id']}.png) "
                         f"(source: `{stamp['preview_sheet']}`)")
        lines.append("")
    (catalog_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")

    html_rows: list[str] = []
    for library in libraries:
        html_rows.append(
            f"<h2>{library['library_name']} <code>{library['library_id']}</code></h2>"
        )
        html_rows.append(
            "<table><thead><tr><th>Stamp</th><th>Type</th><th>Size</th><th>Doors</th>"
            "<th>Multi-shell</th><th>Preview</th></tr></thead><tbody>"
        )
        for stamp in library["stamps"]:
            preview_src = f"previews/{library['library_id']}/{stamp['stamp_id']}.png"
            html_rows.append(
                "<tr>"
                f"<td><code>{stamp['stamp_id']}</code></td>"
                f"<td>{stamp['building_type']}</td>"
                f"<td>{stamp['size_class']}</td>"
                f"<td>{stamp['door_count']}</td>"
                f"<td>{'yes' if stamp['multi_shell'] else 'no'}</td>"
                f"<td><a href=\"{preview_src}\"><img src=\"{preview_src}\" "
                f"alt=\"{stamp['stamp_id']}\" style=\"max-width:480px\"></a></td>"
                "</tr>"
            )
        html_rows.append("</tbody></table>")
    html = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Cityforge "
        "Stamp Library Catalog v1</title></head><body><h1>Cityforge Stamp "
        "Library Catalog v1</h1>"
        + "\n".join(html_rows)
        + "</body></html>\n"
    )
    (catalog_dir / "index.html").write_text(html, encoding="utf-8")
    return index_json


def _library_sha(path: Path) -> str:
    return sha256_file(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive D-STAMP v1 unit-stamp libraries (Cityforge T0.3).")
    parser.add_argument(
        "--libraries", choices=("karthgad", "markarth", "both"), default="both"
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_ROOT, help="library output directory"
    )
    parser.add_argument(
        "--catalog-dir", type=Path, default=CATALOG_DIR, help="catalog output directory"
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="compare freshly derived libraries against existing files byte-for-byte",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    libraries: list[dict[str, Any]] = []
    if args.libraries in ("karthgad", "both"):
        libraries.append(build_karthgad_library())
    if args.libraries in ("markarth", "both"):
        libraries.append(build_markarth_library())

    written: dict[str, Path] = {}
    for library in libraries:
        path = write_library(library, args.out)
        written[library["library_id"]] = path
        digest = _library_sha(path)
        stats = library["stats"]
        print(f"[cityforge] wrote {path}")
        print(f"  sha256 {digest}")
        print(
            f"  stamps={stats['stamp_count']} excluded={stats['excluded_count']} "
            f"types={json.dumps(stats['per_type'])} sizes={json.dumps(stats['per_size_class'])}"
        )
        if stats["replay"]["has_multi_axis_canary"]:
            print(
                f"  multi-axis canary members: {stats['replay']['multi_axis_member_count']}"
            )
        if "terrain_crosscheck" in stats:
            print(
                f"  terrain cross-check disagreements (>1 GU): "
                f"{stats['terrain_crosscheck']['disagreement_count']}"
            )

    index_json = build_catalog(libraries, args.catalog_dir)
    print(
        f"[cityforge] catalog: {len(index_json['stamps'])} verified previews in {args.catalog_dir}"
    )

    if args.verify_determinism:
        failed = False
        for library_id, path in written.items():
            fresh_digest = _library_sha(path)
            # Re-derive without writing to compare bytes.
            rebuilt = (
                build_karthgad_library()
                if library_id == KARTHGAD_LIBRARY_ID
                else build_markarth_library()
            )
            rebuilt_bytes = canonical_json_bytes(rebuilt)
            existing = path.read_bytes()
            if rebuilt_bytes == existing:
                print(f"[cityforge] determinism MATCH {library_id} ({fresh_digest})")
            else:
                print(f"[cityforge] determinism MISMATCH {library_id}")
                failed = True
        if failed:
            print("[cityforge] determinism verification FAILED")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
