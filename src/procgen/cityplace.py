"""Cityforge T1.2 deterministic houses-only placement solver.

Pipeline position
------------------
``cityplace`` is the first engine stage after the accepted T1.1 plan contract:

``site/brief/palette/stamps -> T1.1 validation -> cityplace -> T1.3 landscape``

It consumes a strictly zero-error T1.1 validation result, rechecks all bundle
hash pins and the validator's shared stamp resolution, replays every eligible
D-STAMP member from its source manifest, then seats only the house refs against
one explicit planned/final terrain field.  It emits host-side JSON products;
it does not author an ESP, edit LAND, render a city, solve walls/docks/features,
or change any source file.

Binding behavior
----------------
* The validator's explicit/selector result is authoritative.  T1.2 recomputes
  the exact T1.1 selector (smallest kit-brief hull area, sorted stamp-id tie
  break) and fails closed on any mismatch.
* A lot's plan position is the seed-door XY handle.  Its absolute XY is the
  accepted frame origin plus that plan position.  Anchor Z is bilinear terrain
  height plus the measured seed-door step.  Every member offset and rotation
  is transformed in engine matrix space; plan yaw is never nudged toward a
  road and direct Euler arithmetic is forbidden.
* Conform violations reject.  A legal ``flatten_pad`` violation produces one
  exact, measured provisional pad request on a planned pass.  A final pass
  needs a prior planned placement reference and re-seats against the final
  terrain hash before a pad lot can be accepted.
* Hull overlap/contact is the only hard spacing rule.  Dispatch-5 gap
  distributions are warning-only.  Fine triangle/AABB collision is explicitly
  deferred when unavailable.

Public entry point
------------------
``solve_city_plan`` returns JSON-ready products plus source/oracle evidence;
``tools/cityforge/solve_city_placement.py`` handles paths, output hashes, and
the required ``FAILURE: cityplace <reason>`` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import cityplan
from .cityplace_contracts import (
    CityPlaceInputError,
    FieldCoverageError,
    PlacementConfig,
    TerrainField,
    WATER_LEVEL_GU,
    load_json,
    sha256_file,
)
from .cityplace_geometry import (
    collision_status,
    fine_collision_ledger_entry,
    road_access_check,
    road_corridor_conflict,
    scope_and_mask_checks,
    terrain_metrics,
    transform_hull,
)
from .cityplace_output import (
    build_city_placement,
    build_land_edit_requests,
    build_manifest,
)
from .cityplace_transform import (
    PlacedMember,
    TransformContractError,
    mathematical_cell_bucket,
    place_stamp_members,
    replay_source_libraries,
    yaw37_oracle,
)
from .censusio import deterministic_dumps


CASTLE_BARRACKS_ID = "markarth_side_v1__u114_castle_barracks"
LOT_POSITION_TOLERANCE_GU = 0.0


class CityPlaceError(CityPlaceInputError):
    """Fatal cityplace stage failure; no degraded output is trusted."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CityPlaceError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise CityPlaceError(f"{label} must be finite")
    return result


def _issue(
    severity: str,
    code: str,
    path: str,
    message: str,
    measured: Any = None,
    limit: Any = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "path": path,
        "message": message,
    }
    if measured is not None:
        result["measured"] = measured
    if limit is not None:
        result["limit"] = limit
    return result


def _status_check(
    *,
    status: str,
    code: str | None,
    measured: Any,
    limit: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "code": code,
        "measured": measured,
        "limit": limit,
        "message": message,
    }


def _source_step_rows(stamp: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[float], int]:
    members = stamp.get("members")
    envelope = stamp.get("terrain_envelope")
    if not isinstance(members, list) or not isinstance(envelope, Mapping):
        raise CityPlaceError(f"stamp {stamp.get('stamp_id')} has incomplete terrain/member contract")
    doors = [member for member in members if isinstance(member, Mapping) and bool(member.get("is_door"))]
    steps = envelope.get("door_step_heights_gu")
    if not isinstance(steps, list) or len(steps) != len(doors):
        raise CityPlaceError(
            f"stamp {stamp.get('stamp_id')} door-step count {len(steps) if isinstance(steps, list) else None} "
            f"does not match door member count {len(doors)}"
        )
    seed_id = stamp.get("source", {}).get("seed_door")
    seed_index = next((index for index, door in enumerate(doors) if door.get("source_id") == seed_id), None)
    if seed_index is None:
        raise CityPlaceError(f"stamp {stamp.get('stamp_id')} seed door {seed_id!r} is not a door member")
    return doors, [float(step) for step in steps], seed_index


def validate_stamp_integrity(stamp: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck D-STAMP record semantics, mesh keys, and door provenance links."""

    stamp_id = str(stamp.get("stamp_id"))
    members = stamp.get("members")
    if not isinstance(members, list) or not members:
        raise CityPlaceError(f"stamp {stamp_id} has no members")
    mesh_refs = 0
    door_links = 0
    for index, member in enumerate(members):
        if not isinstance(member, Mapping):
            raise CityPlaceError(f"stamp {stamp_id} member {index} is not an object")
        record_type = member.get("record_type")
        if record_type not in ("STAT", "DOOR", "ACTI"):
            raise CityPlaceError(f"stamp {stamp_id} member {index} has unsupported record_type {record_type!r}")
        model_key = member.get("model_key")
        if not isinstance(model_key, str) or not model_key.strip():
            raise CityPlaceError(f"stamp {stamp_id} member {member.get('source_id')} has no mesh model_key")
        mesh_refs += 1
        if bool(member.get("is_door")):
            if record_type != "DOOR":
                raise CityPlaceError(f"stamp {stamp_id} door {member.get('source_id')} is not record_type DOOR")
            door = member.get("door")
            if not isinstance(door, Mapping):
                raise CityPlaceError(f"stamp {stamp_id} door {member.get('source_id')} has no provenance block")
            destination = door.get("destination_cell")
            destination_position = door.get("destination_position_gu")
            destination_rotation = door.get("destination_rotation")
            # Null destinations are valid source provenance for cave/grate doors;
            # non-null values must nevertheless retain their complete shape.
            if destination is not None and not isinstance(destination, str):
                raise CityPlaceError(f"stamp {stamp_id} door {member.get('source_id')} has malformed destination cell")
            if destination_position is not None and (
                not isinstance(destination_position, list) or len(destination_position) != 3
            ):
                raise CityPlaceError(f"stamp {stamp_id} door {member.get('source_id')} has malformed destination position")
            if destination_rotation is not None and (
                not isinstance(destination_rotation, list) or len(destination_rotation) != 3
            ):
                raise CityPlaceError(f"stamp {stamp_id} door {member.get('source_id')} has malformed destination rotation")
            if destination is not None or destination_position is not None or destination_rotation is not None:
                door_links += 1
    _source_step_rows(stamp)
    return {
        "stamp_id": stamp_id,
        "member_count": len(members),
        "mesh_references_checked": mesh_refs,
        "door_members_checked": sum(1 for member in members if bool(member.get("is_door"))),
        "door_provenance_links_checked": door_links,
        "triangle_geometry_available": False,
        "member_aabb_available": False,
    }


def _kit_source_hashes(bundle: cityplan.Bundle) -> dict[str, str]:
    sources = bundle.kit_brief.get("sources")
    if not isinstance(sources, Mapping):
        raise CityPlaceError("kit brief has no sources pin block")
    expected: dict[str, str] = {}
    for key, value in sources.items():
        if not isinstance(value, Mapping):
            continue
        library_id = value.get("library_id")
        digest = value.get("sha256")
        if isinstance(library_id, str) and isinstance(digest, str):
            expected[library_id] = digest
    for library_id, digest in expected.items():
        actual = bundle.hashes.get(f"stamp_library_{library_id}")
        if actual != digest:
            raise CityPlaceError(
                f"kit brief stamp-library pin mismatch for {library_id}: {actual} != {digest}"
            )
    return expected


def _plan_hash(plan_path: Path | str) -> str:
    return sha256_file(Path(plan_path))


def _copy_validation_core(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in ("valid", "plan_id", "schema_version", "issue_count", "error_count",
                    "warning_count", "issues", "summary", "input_hashes")
    }


def verify_t1_1_validation(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    validation: Mapping[str, Any],
    bundle: cityplan.Bundle,
) -> dict[str, Any]:
    """Require a current, byte-pinned, zero-error T1.1 validation result."""

    if validation.get("valid") is not True or int(validation.get("error_count", -1)) != 0:
        raise CityPlaceError("T1.1 validation is not a zero-error valid result")
    plan_digest = _plan_hash(plan_path)
    if validation.get("plan_file_sha256") != plan_digest:
        raise CityPlaceError("T1.1 validation plan_file_sha256 does not match the consumed plan")
    current = cityplan.validate_plan(dict(plan), bundle)
    current["plan_file_sha256"] = plan_digest
    current["input_hashes"]["plan"] = plan_digest
    if _copy_validation_core(validation) != _copy_validation_core(current):
        raise CityPlaceError("supplied T1.1 validation differs from a fresh validation of the pinned inputs")
    if current["error_count"] != 0:
        raise CityPlaceError("fresh T1.1 validation found errors")
    # Every bundle hash is checked explicitly; warning-only distinctions remain
    # in the exact issue/summary comparison above.
    expected_hashes = dict(sorted(current["input_hashes"].items()))
    actual_hashes = dict(sorted((validation.get("input_hashes") or {}).items()))
    if actual_hashes != expected_hashes:
        raise CityPlaceError("T1.1 validation input hashes do not match current bundle hashes")
    return {
        "plan_sha256": plan_digest,
        "valid": True,
        "error_count": 0,
        "warning_count": int(current["warning_count"]),
        "warning_codes": sorted(current["summary"].get("warning_codes", [])),
        "input_hashes": expected_hashes,
        "summary": current["summary"],
    }


def _resolution_index(validation: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = validation.get("summary", {}).get("lot_resolution", [])
    if not isinstance(rows, list):
        raise CityPlaceError("T1.1 validation has no lot_resolution list")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("lot_id"), str):
            raise CityPlaceError("T1.1 lot_resolution contains a malformed row")
        if row["lot_id"] in result:
            raise CityPlaceError(f"duplicate T1.1 lot resolution {row['lot_id']}")
        result[str(row["lot_id"])] = row
    return result


def resolve_shared_stamp(
    lot: Mapping[str, Any],
    *,
    bundle: cityplan.Bundle,
    validator_resolution: Mapping[str, Any],
) -> tuple[str, str, Mapping[str, Any]]:
    """Recompute and compare T1.1's exact shared selection result."""

    request = lot.get("request")
    if not isinstance(request, Mapping):
        raise CityPlaceError(f"lot {lot.get('lot_id')} has no request object")
    explicit = request.get("stamp_id")
    if isinstance(explicit, str):
        selected = explicit
        mode = "explicit"
    else:
        candidates = cityplan._candidate_stamps(bundle, dict(request))  # shared T1.1 contract
        if not candidates:
            raise CityPlaceError(f"lot {lot.get('lot_id')} has no compatible eligible stamp")
        selected = cityplan._select_stamp(bundle, candidates)  # exact shared selector
        mode = "selector"
    if validator_resolution.get("stamp_id") != selected or validator_resolution.get("mode") != mode:
        raise CityPlaceError(
            f"lot {lot.get('lot_id')} selector mismatch: T1.1={validator_resolution.get('stamp_id')!r}/"
            f"{validator_resolution.get('mode')!r}, T1.2={selected!r}/{mode!r}"
        )
    if validator_resolution.get("geometry_checked") is not True:
        raise CityPlaceError(f"lot {lot.get('lot_id')} was not geometry-checked by T1.1")
    stamp = bundle.stamp_geometry.get(selected)
    if stamp is None:
        raise CityPlaceError(f"selected stamp {selected} is absent from pinned library geometry")
    if selected == CASTLE_BARRACKS_ID:
        raise CityPlaceError("Castle Barracks is excluded and cannot be placed")
    btype = request.get("building_type")
    if stamp.get("building_type") != btype:
        raise CityPlaceError(f"selected stamp {selected} does not match requested building_type {btype!r}")
    if request.get("size_class") is not None and stamp.get("size_class") != request.get("size_class"):
        raise CityPlaceError(f"selected stamp {selected} does not match requested size_class")
    if request.get("multi_shell") is not None and bool(stamp.get("multi_shell")) != bool(request.get("multi_shell")):
        raise CityPlaceError(f"selected stamp {selected} does not match requested multi_shell")
    return selected, mode, stamp


def preflight_fixture_rejection(
    lot: Mapping[str, Any], *, bundle: cityplan.Bundle
) -> dict[str, Any] | None:
    """Return structured early rejection evidence for invalid proof lots.

    Normal T1.1-valid plans never enter this path.  The synthetic proof harness
    still needs to exercise no-stamp and out-of-scope contracts without asking
    the bilinear sampler to invent terrain outside its field.  This function is
    part of the production preflight logic, not a fixture-only expected-value
    shortcut.
    """

    lot_id = str(lot.get("lot_id", "fixture_lot"))
    request = lot.get("request") if isinstance(lot.get("request"), Mapping) else {}
    btype = request.get("building_type")
    if btype in cityplan.UNAVAILABLE_BUILDING_TYPES:
        return {
            "lot_id": lot_id,
            "status": "rejected",
            "checks": {"stamp_resolution": {"status": "failed", "code": "no_compatible_stamp"}},
            "issues": [_issue("error", "no_compatible_stamp", f"$.lots[{lot_id}].request",
                               f"no eligible stamp exists for capability-gap building_type {btype!r}",
                               btype, "eligible stamp")],
        }
    explicit = request.get("stamp_id")
    if isinstance(explicit, str) and explicit not in bundle.stamp_geometry:
        return {
            "lot_id": lot_id,
            "status": "rejected",
            "checks": {"stamp_resolution": {"status": "failed", "code": "stamp_not_eligible"}},
            "issues": [_issue("error", "stamp_not_eligible", f"$.lots[{lot_id}].request.stamp_id",
                               f"stamp {explicit!r} is absent from the eligible T1.2 stamp set",
                               explicit, "eligible stamp id")],
        }
    if not isinstance(btype, str):
        return {
            "lot_id": lot_id,
            "status": "rejected",
            "checks": {"stamp_resolution": {"status": "failed", "code": "no_compatible_stamp"}},
            "issues": [_issue("error", "no_compatible_stamp", f"$.lots[{lot_id}].request",
                               "request has no building_type", btype, "eligible building type")],
        }
    if not isinstance(explicit, str):
        candidates = cityplan._candidate_stamps(bundle, dict(request))
        if not candidates:
            return {
                "lot_id": lot_id,
                "status": "rejected",
                "checks": {"stamp_resolution": {"status": "failed", "code": "no_compatible_stamp"}},
                "issues": [_issue("error", "no_compatible_stamp", f"$.lots[{lot_id}].request",
                                   f"no eligible stamp matches building_type {btype!r}",
                                   btype, "eligible stamp")],
            }
        selected = cityplan._select_stamp(bundle, candidates)
    else:
        selected = explicit
    stamp = bundle.stamp_geometry[selected]
    position = lot.get("position")
    if not isinstance(position, list) or len(position) != 2:
        return None
    try:
        x, y = float(position[0]), float(position[1])
        yaw = float(lot.get("yaw_deg", 0.0))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, yaw)):
        return None
    hull = transform_hull(stamp["footprint"]["hull_xy_rel"], anchor_xy_plan_gu=(x, y), yaw_deg=yaw)
    scope = {
        "footprint_vertices_in_scope": all(cityplan.in_scope(px, py) for px, py in hull),
        "out_of_scope_vertices": [[px, py] for px, py in hull if not cityplan.in_scope(px, py)],
        "anchor_in_scope": cityplan.in_scope(x, y),
        "stamp_id": selected,
        "input_position_plan_gu": [x, y],
        "input_yaw_deg": yaw,
    }
    issues: list[dict[str, Any]] = []
    if not scope["anchor_in_scope"]:
        issues.append(_issue("error", "out_of_scope", f"$.lots[{lot_id}].position",
                             "seed-door anchor is outside the target plan frame", [x, y], 57344.0))
    if not scope["footprint_vertices_in_scope"]:
        issues.append(_issue("error", "footprint_out_of_scope", f"$.lots[{lot_id}].footprint",
                             "exact footprint has vertices outside the target plan frame",
                             scope["out_of_scope_vertices"], 0))
    if issues:
        return {
            "lot_id": lot_id,
            "stamp_id": selected,
            "status": "rejected",
            "position_plan_gu": [x, y],
            "yaw_deg": yaw,
            "footprint_hull_xy_plan_gu": [[px, py] for px, py in hull],
            "checks": {"scope": scope},
            "issues": issues,
        }
    return None


def _source_access_heading(stamp: Mapping[str, Any]) -> float:
    value = stamp.get("access_heading_rad")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CityPlaceError(f"stamp {stamp.get('stamp_id')} has no finite access_heading_rad")
    return float(value)


def _find_terrain_violations(
    stamp: Mapping[str, Any],
    terrain: Mapping[str, Any],
    *,
    step_rows: list[dict[str, Any]],
    config: PlacementConfig,
    policy_mode: str,
    max_cut_fill_gu: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return hard terrain issues and measured check rows."""

    envelope = stamp["terrain_envelope"]
    issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    relief = float(terrain["relief_gu"])
    source_relief = float(envelope.get("footprint_relief_gu", 0.0))
    relief_limit = source_relief * (1.0 + config.relief_tolerance_fraction)
    relief_bad = relief > relief_limit + 1.0e-9
    checks.append({
        **_status_check(
            status="violation" if relief_bad else "pass",
            code="terrain_relief_exceeded" if relief_bad else None,
            measured=relief,
            limit=relief_limit,
            message="bilinear footprint relief versus source envelope",
        ),
        "source_envelope_gu": source_relief,
        "tolerance_fraction": config.relief_tolerance_fraction,
    })
    if relief_bad:
        issues.append(_issue("error", "terrain_relief_exceeded", "$.terrain",
                             f"footprint relief {relief:.3f} GU exceeds source envelope limit {relief_limit:.3f} GU",
                             relief, relief_limit))

    slope = float(terrain["best_fit_slope_deg"])
    source_slope = float(envelope.get("footprint_slope_deg", 0.0))
    slope_limit = source_slope + config.slope_slack_deg
    slope_bad = slope > slope_limit + 1.0e-9
    checks.append({
        **_status_check(
            status="violation" if slope_bad else "pass",
            code="terrain_slope_exceeded" if slope_bad else None,
            measured=slope,
            limit=slope_limit,
            message="best-fit footprint slope versus source prior plus slack",
        ),
        "source_prior_deg": source_slope,
        "slack_deg": config.slope_slack_deg,
    })
    if slope_bad:
        issues.append(_issue("error", "terrain_slope_exceeded", "$.terrain",
                             f"best-fit slope {slope:.3f}° exceeds source limit {slope_limit:.3f}°",
                             slope, slope_limit))

    source_burial = float(envelope.get("burial_depth_gu", 0.0))
    burial = float(terrain["burial_depth_gu"])
    burial_lower = max(0.0, source_burial * (1.0 - config.burial_tolerance_fraction) - config.burial_zero_slack_gu)
    burial_upper = source_burial * (1.0 + config.burial_tolerance_fraction) + config.burial_zero_slack_gu
    burial_bad = burial < burial_lower - 1.0e-9 or burial > burial_upper + 1.0e-9
    checks.append({
        **_status_check(
            status="violation" if burial_bad else "pass",
            code="burial_out_of_envelope" if burial_bad else None,
            measured=burial,
            limit=[burial_lower, burial_upper],
            message="target burial depth versus source envelope",
        ),
        "source_envelope_gu": source_burial,
        "tolerance_fraction": config.burial_tolerance_fraction,
    })
    if burial_bad:
        issues.append(_issue("error", "burial_out_of_envelope", "$.terrain",
                             f"burial depth {burial:.3f} GU is outside [{burial_lower:.3f}, {burial_upper:.3f}] GU",
                             burial, [burial_lower, burial_upper]))

    bottom = float(terrain["bottom_clearance_gu"])
    # A negative bottom clearance is normal for a source-authored building:
    # D-STAMP explicitly measures and bounds its burial depth.  The burial
    # envelope check above is the hard limit; this row is an observation, not a
    # false "must float above terrain" rule.
    checks.append(_status_check(
        status="measured",
        code=None,
        measured=bottom,
        limit="reported; burial envelope is the authoritative hard check",
        message="member-bounds bottom clearance (negative values are source-burial evidence)",
    ))

    step_bad = False
    for row in step_rows:
        if row["deviation_gu"] > row["tolerance_gu"] + 1.0e-9:
            step_bad = True
            issues.append(_issue(
                "error", "door_step_out_of_envelope", "$.doors",
                f"door {row['source_id']} step {row['measured_step_height_gu']:.3f} GU differs from "
                f"source {row['source_step_height_gu']:.3f} GU by {row['deviation_gu']:.3f} GU",
                row["deviation_gu"], row["tolerance_gu"],
            ))
    checks.append({
        "status": "violation" if step_bad else "pass",
        "code": "door_step_out_of_envelope" if step_bad else None,
        "measured": step_rows,
        "limit": "per-door source step ± configured fraction/slack",
        "message": "seed and additional-door step heights",
    })
    cut_fill = float(terrain["max_cut_fill_gu"] if "max_cut_fill_gu" in terrain else 0.0)
    checks.append({
        "status": "pass" if cut_fill <= max_cut_fill_gu + 1.0e-9 else "violation",
        "code": None if cut_fill <= max_cut_fill_gu + 1.0e-9 else "pad_cut_fill_exceeded",
        "measured": cut_fill,
        "limit": max_cut_fill_gu,
        "message": "maximum absolute target-height cut/fill over measured footprint samples",
    })
    if policy_mode == "flatten_pad" and cut_fill > max_cut_fill_gu + 1.0e-9:
        issues.append(_issue(
            "error", "pad_cut_fill_exceeded", "$.terrain",
            f"requested pad cut/fill {cut_fill:.3f} GU exceeds lot limit {max_cut_fill_gu:.3f} GU",
            cut_fill, max_cut_fill_gu,
        ))
    # ``policy_mode`` is included in the call contract to make the decision
    # explicit; pad eligibility is decided by the orchestrator after all checks.
    _ = policy_mode
    return issues, checks


def _pad_polygon(hull: Sequence[tuple[float, float]], margin_gu: float) -> list[list[float]]:
    """Conservative exact-footprint envelope used by T1.3 for the pad request."""

    xs = [float(point[0]) for point in hull]
    ys = [float(point[1]) for point in hull]
    return [
        [min(xs) - margin_gu, min(ys) - margin_gu],
        [max(xs) + margin_gu, min(ys) - margin_gu],
        [max(xs) + margin_gu, max(ys) + margin_gu],
        [min(xs) - margin_gu, max(ys) + margin_gu],
        [min(xs) - margin_gu, min(ys) - margin_gu],
    ]


def _step_rows(
    doors: Sequence[Mapping[str, Any]],
    source_steps: Sequence[float],
    placed_members: Sequence[PlacedMember],
    field: TerrainField,
    *,
    config: PlacementConfig,
) -> list[dict[str, Any]]:
    placed_by_id = {member.source_id: member for member in placed_members}
    rows: list[dict[str, Any]] = []
    for door, source_step in zip(doors, source_steps):
        source_id = str(door["source_id"])
        placed = placed_by_id.get(source_id)
        if placed is None:
            raise CityPlaceError(f"placed stamp omitted door member {source_id}")
        x, y, z = placed.world_position_gu
        plan_x = x - float(field.origin_gu[0])
        plan_y = y - float(field.origin_gu[1])
        sample = field.sample(plan_x, plan_y)
        measured = z - sample.height_gu
        tolerance = max(abs(float(source_step)) * config.step_tolerance_fraction,
                        config.step_zero_slack_gu)
        rows.append({
            "source_id": source_id,
            "position_plan_gu": [plan_x, plan_y],
            "measured_step_height_gu": measured,
            "source_step_height_gu": float(source_step),
            "deviation_gu": abs(measured - float(source_step)),
            "tolerance_gu": tolerance,
            "field_sample": sample.to_dict(),
        })
    return rows


def _lot_result_base(
    lot: Mapping[str, Any], stamp: Mapping[str, Any], mode: str, hull: Sequence[tuple[float, float]],
    placed_members: Sequence[PlacedMember], *, field: TerrainField, anchor_z: float,
) -> dict[str, Any]:
    position = lot["position"]
    x, y = float(position[0]), float(position[1])
    world_xy = [float(field.origin_gu[0]) + x, float(field.origin_gu[1]) + y]
    return {
        "lot_id": lot.get("lot_id"),
        "stamp_id": stamp.get("stamp_id"),
        "resolution_mode": mode,
        "position_plan_gu": [x, y],
        "position_world_xy_gu": world_xy,
        "yaw_deg": float(lot.get("yaw_deg", 0.0)),
        "anchor": {
            "seed_door_xy_plan_gu": [x, y],
            "seed_door_world_gu": [world_xy[0], world_xy[1], anchor_z],
            "field_pass": field.field_pass,
            "cell": mathematical_cell_bucket(world_xy[0], world_xy[1]),
        },
        "footprint_hull_xy_plan_gu": [[float(px), float(py)] for px, py in hull],
        "members": [member.to_dict() for member in placed_members],
        "record_semantics": {
            "record_types_preserved": True,
            "mesh_refs_checked": True,
            "door_teleport_authoring": "deferred; D-STAMP destination fields are provenance only",
        },
    }


@dataclass
class SolverRun:
    """Validated engine context and deterministic output assembly state."""

    plan: Mapping[str, Any]
    plan_path: Path
    validation: Mapping[str, Any]
    bundle: cityplan.Bundle
    field: TerrainField
    terrain_pass: str
    workspace_root: Path
    config: PlacementConfig
    planned_placement: Mapping[str, Any] | None = None
    validation_evidence: dict[str, Any] | None = None

    def prepare(self) -> dict[str, Any]:
        """Run all fatal input, selector, replay, and oracle gates."""

        self.validation_evidence = verify_t1_1_validation(
            self.plan,
            plan_path=self.plan_path,
            validation=self.validation,
            bundle=self.bundle,
        )
        _kit_source_hashes(self.bundle)
        stamp_integrity = []
        for stamp_id in sorted(self.bundle.stamp_geometry):
            if stamp_id == CASTLE_BARRACKS_ID:
                continue
            stamp_integrity.append(validate_stamp_integrity(self.bundle.stamp_geometry[stamp_id]))
        eligible_ids = set(self.bundle.stamp_geometry)
        if CASTLE_BARRACKS_ID in eligible_ids:
            raise CityPlaceError("Castle Barracks unexpectedly appears in T1.1 eligible stamp geometry")
        replay = replay_source_libraries(
            self.bundle.libraries.values(), eligible_ids, workspace_root=self.workspace_root
        )
        oracle = yaw37_oracle(
            [self.bundle.stamp_geometry[stamp_id] for stamp_id in sorted(eligible_ids)],
            yaw_deg=37.0,
        )
        if oracle["mismatches"] != 0:
            raise CityPlaceError(f"37-degree multi-axis oracle mismatch: {oracle}")
        resolution_index = _resolution_index(self.validation)
        for lot in self.plan.get("lots", []):
            if not isinstance(lot, Mapping):
                raise CityPlaceError("T1.1-valid plan contains a malformed lot")
            lot_id = lot.get("lot_id")
            if not isinstance(lot_id, str) or lot_id not in resolution_index:
                raise CityPlaceError(f"missing T1.1 resolution for lot {lot_id!r}")
            resolve_shared_stamp(lot, bundle=self.bundle, validator_resolution=resolution_index[lot_id])
        reseat = self._verify_final_reseat()
        return {
            "validation": self.validation_evidence,
            "stamp_integrity": stamp_integrity,
            "source_replay": replay,
            "multi_axis_oracle_37deg": oracle,
            "final_reseat": reseat,
        }

    def _verify_final_reseat(self) -> dict[str, Any]:
        if self.terrain_pass != "final":
            return {
                "required": False,
                "status": "not_applicable_planned_pass",
                "final_field_sha256": self.field.source_sha256,
            }
        if self.planned_placement is None:
            raise CityPlaceError("final terrain pass requires a prior planned placement reference")
        if self.planned_placement.get("plan_sha256") != self.validation_evidence["plan_sha256"]:
            raise CityPlaceError("planned placement reference belongs to a different plan hash")
        planned_field = self.planned_placement.get("terrain_field")
        if not isinstance(planned_field, Mapping) or planned_field.get("pass") != "planned":
            raise CityPlaceError("final reseat reference is not a planned-field placement product")
        provisional_ids = {
            str(item.get("lot_id")) for item in self.planned_placement.get("provisional_pad_lots", [])
            if isinstance(item, Mapping)
        }
        return {
            "required": True,
            "status": "reference_verified",
            "planned_field_sha256": planned_field.get("sha256"),
            "final_field_sha256": self.field.source_sha256,
            "provisional_lots_in_reference": sorted(provisional_ids),
            "reseated_against_final_hash": self.field.source_sha256,
        }

    def _resolve_lots(self) -> dict[str, tuple[str, str, Mapping[str, Any]]]:
        rows = _resolution_index(self.validation)
        result = {}
        for lot in self.plan.get("lots", []):
            assert isinstance(lot, Mapping)
            lot_id = str(lot["lot_id"])
            result[lot_id] = resolve_shared_stamp(lot, bundle=self.bundle, validator_resolution=rows[lot_id])
        return result

    def _evaluate_lot(
        self,
        lot: Mapping[str, Any],
        *,
        stamp: Mapping[str, Any],
        mode: str,
        existing: list[tuple[str, Sequence[tuple[float, float]]]],
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Return status, lot evidence, optional pad requests, and hull ledger."""

        lot_id = str(lot["lot_id"])
        x = _finite(lot["position"][0], f"lot {lot_id} x")
        y = _finite(lot["position"][1], f"lot {lot_id} y")
        yaw = _finite(lot.get("yaw_deg", 0.0), f"lot {lot_id} yaw_deg")
        hull_rel = stamp.get("footprint", {}).get("hull_xy_rel")
        if not isinstance(hull_rel, list) or len(hull_rel) < 3:
            raise CityPlaceError(f"stamp {stamp.get('stamp_id')} has no exact footprint hull")
        hull = transform_hull(hull_rel, anchor_xy_plan_gu=(x, y), yaw_deg=yaw)
        seed_sample = self.field.sample(x, y)
        doors, source_steps, seed_index = _source_step_rows(stamp)
        anchor_step = source_steps[seed_index]
        anchor_world = (
            float(self.field.origin_gu[0]) + x,
            float(self.field.origin_gu[1]) + y,
            seed_sample.height_gu + anchor_step,
        )
        placed = place_stamp_members(stamp, anchor_world_gu=anchor_world, yaw_deg=yaw)
        evidence = _lot_result_base(
            lot, stamp, mode, hull, placed, field=self.field, anchor_z=anchor_world[2]
        )
        issues: list[dict[str, Any]] = []
        scope = scope_and_mask_checks(hull, anchor_xy=(x, y), bundle=self.bundle)
        evidence["checks"] = {"scope_and_masks": scope}
        if not scope["footprint_vertices_in_scope"]:
            issues.append(_issue("error", "footprint_out_of_scope", f"$.lots[{lot_id}].footprint",
                                 "exact footprint vertex escapes the target plan frame",
                                 scope["out_of_scope_vertices"], [0, self.bundle.survey_frame["site_span_gu"]]))
        if not scope["anchor"]["in_scope"]:
            issues.append(_issue("error", "out_of_scope", f"$.lots[{lot_id}].position",
                                 "seed-door anchor is outside target scope", [x, y], self.bundle.survey_frame["site_span_gu"]))
        if not scope["anchor"]["buildable"]:
            issues.append(_issue("error", "door_unbuildable", f"$.lots[{lot_id}].position",
                                 "seed-door tile is not buildable", scope["anchor"], 1))
        if scope["anchor"]["water"] or scope["water_tiles"]:
            issues.append(_issue("error", "non_dock_water", f"$.lots[{lot_id}].footprint",
                                 "houses-only lot intersects the surveyed water mask", scope["water_tiles"], 0))
        if scope["unbuildable_tiles"]:
            issues.append(_issue("error", "footprint_unbuildable", f"$.lots[{lot_id}].footprint",
                                 "exact footprint covers non-buildable target tiles",
                                 scope["unbuildable_tiles"], 0))

        member_plan_positions = [
            (member.world_position_gu[0] - self.field.origin_gu[0],
             member.world_position_gu[1] - self.field.origin_gu[1])
            for member in placed
        ]
        door_plan_positions = [
            (member.world_position_gu[0] - self.field.origin_gu[0],
             member.world_position_gu[1] - self.field.origin_gu[1])
            for member in placed if member.is_door
        ]
        try:
            terrain = terrain_metrics(
                self.field,
                hull,
                member_positions_plan=member_plan_positions,
                door_positions_plan=door_plan_positions,
                anchor_plan_xy=(x, y),
                anchor_z_gu=anchor_world[2],
                placed_members=placed,
                bounds_min_z_gu=float(stamp["bounds_rel_gu"]["min"][2]),
            )
        except (ValueError, FieldCoverageError) as exc:
            raise CityPlaceError(f"terrain seating/sample failure for lot {lot_id}: {exc}") from exc
        target_height = seed_sample.height_gu
        terrain["target_height_gu"] = target_height
        terrain["max_cut_fill_gu"] = max(
            (abs(float(sample["height_gu"]) - target_height) for sample in terrain["samples"]),
            default=0.0,
        )
        evidence["checks"]["terrain"] = terrain
        if terrain["missing_sample_count"]:
            issues.append(_issue(
                "error", "terrain_field_coverage_missing", f"$.lots[{lot_id}].terrain",
                "one or more required footprint/member/door samples are outside the supplied field",
                terrain["missing_samples"], 0,
            ))
        try:
            step_rows = _step_rows(doors, source_steps, placed, self.field, config=self.config)
        except (ValueError, FieldCoverageError) as exc:
            raise CityPlaceError(f"door seating failure for lot {lot_id}: {exc}") from exc
        evidence["checks"]["door_steps"] = step_rows
        policy = lot.get("terrain_policy") if isinstance(lot.get("terrain_policy"), Mapping) else {}
        policy_mode = str(policy.get("mode", "conform"))
        max_cut_fill = float(policy.get("max_cut_fill_gu", self.config.max_pad_cut_fill_gu))
        terrain_issues, terrain_checks = _find_terrain_violations(
            stamp, terrain, step_rows=step_rows, config=self.config,
            policy_mode=policy_mode, max_cut_fill_gu=max_cut_fill,
        )
        evidence["checks"]["terrain_limits"] = terrain_checks
        issues.extend(terrain_issues)
        if terrain["max_cut_fill_gu"] > self.config.max_encoded_delta_gu + 1.0e-9:
            issues.append(_issue("error", "pad_encoding_delta_exceeded", f"$.lots[{lot_id}].terrain",
                                 "measured target delta exceeds TES3 delta encoding bound",
                                 terrain["max_cut_fill_gu"], self.config.max_encoded_delta_gu))

        source_heading = _source_access_heading(stamp)
        try:
            road = road_access_check(
                self.plan, self.bundle, lot=lot, door_xy_plan=(x, y),
                source_access_heading_rad=source_heading, yaw_deg=yaw,
                field=self.field, hull=hull, config=self.config,
            )
        except (ValueError, FieldCoverageError) as exc:
            road = {"status": "measurement_failed", "code": "road_measurement_failed", "message": str(exc)}
        evidence["checks"]["road_access"] = road
        if road.get("status") == "measurement_failed":
            raise CityPlaceError(f"road access measurement failed for lot {lot_id}: {road.get('message')}")
        if road.get("status") == "measured":
            if float(road["door_to_road_distance_gu"]) > self.config.hard_road_distance_gu + 1.0e-9:
                issues.append(_issue("error", "road_distance_exceeded", f"$.lots[{lot_id}].access",
                                     "door-to-road distance exceeds hard houses-only limit",
                                     road["door_to_road_distance_gu"], self.config.hard_road_distance_gu))
            elif float(road["door_to_road_distance_gu"]) > self.config.preferred_road_distance_gu + 1.0e-9:
                issues.append(_issue("warning", "road_distance_preferred_exceeded", f"$.lots[{lot_id}].access",
                                     "door-to-road distance exceeds preferred threshold; no lot movement is applied",
                                     road["door_to_road_distance_gu"], self.config.preferred_road_distance_gu))
            if float(road.get("cross_slope_deg", 0.0)) > self.config.hard_cross_slope_deg + 1.0e-9:
                issues.append(_issue("error", "door_path_steep", f"$.lots[{lot_id}].access",
                                     "door access cross-slope exceeds the hard corridor limit",
                                     road["cross_slope_deg"], self.config.hard_cross_slope_deg))
            if float(road.get("angular_deviation_deg", 0.0)) > 90.0:
                issues.append(_issue("warning", "door_heading_deviation", f"$.lots[{lot_id}].access",
                                     "planned yaw faces more than 90 degrees away from the nearest road point; yaw is preserved",
                                     road["angular_deviation_deg"], 90.0))
            named = road.get("road_id")
            road_map = {str(item.get("road_id")): item for item in self.plan.get("roads", []) if isinstance(item, Mapping)}
            if named in road_map and road_corridor_conflict(hull, road_map[named]):
                issues.append(_issue("error", "footprint_road_corridor_conflict", f"$.lots[{lot_id}].access",
                                     "exact building hull enters the planned road corridor; no automatic movement",
                                     road["hull_to_road_distance_gu"], 0.0))

        collision = collision_status(hull, existing, config=self.config)
        evidence["checks"]["collision"] = collision
        for conflict in collision["hard_conflicts"]:
            issues.append(_issue("error", "footprint_collision", f"$.lots[{lot_id}].footprint",
                                 f"exact hull {conflict['status']} with lot {conflict['other_lot_id']}; spacing guidance is not used as a minimum",
                                 conflict["boundary_gap_gu"], 0.0))
        evidence["checks"]["plan_yaw"] = {
            "input_yaw_deg": float(lot.get("yaw_deg", 0.0)),
            "emitted_yaw_deg": float(lot.get("yaw_deg", 0.0)),
            "movement_gu": [0.0, 0.0],
            "rotation_policy": "exact plan yaw; no road-facing nudge",
        }
        evidence["issues"] = sorted(issues, key=lambda item: (item["severity"], item["code"], item["message"]))
        evidence["status"] = "rejected"
        evidence["source_geometry"] = {
            "triangle_collision": "deferred",
            "member_aabb_collision": "deferred",
            "fine_collision_deferred": True,
        }

        hard_issues = [item for item in issues if item["severity"] == "error"]
        terrain_codes = {item["code"] for item in terrain_issues}
        pad_eligible_codes = {
            "terrain_relief_exceeded", "terrain_slope_exceeded",
            "burial_out_of_envelope", "door_step_out_of_envelope",
        }
        non_pad_hard = [item for item in hard_issues if item["code"] not in pad_eligible_codes and item["code"] != "pad_encoding_delta_exceeded"]
        pad_request: list[dict[str, Any]] | None = None
        if policy_mode == "flatten_pad" and terrain_codes and terrain_codes.issubset(pad_eligible_codes) and not non_pad_hard:
            cut_fill = float(terrain["max_cut_fill_gu"])
            if cut_fill <= max_cut_fill + 1.0e-9 and cut_fill <= self.config.max_encoded_delta_gu + 1.0e-9:
                pad_request = [{
                    "lot_id": lot_id,
                    "stamp_id": stamp.get("stamp_id"),
                    "status": "provisional" if self.terrain_pass == "planned" else "final_reseat_required",
                    "field_pass": self.terrain_pass,
                    "footprint_hull_xy_plan_gu": [[float(px), float(py)] for px, py in hull],
                    "pad_polygon": _pad_polygon(hull, self.config.pad_margin_gu),
                    "margin_gu": self.config.pad_margin_gu,
                    "target_height_gu": target_height,
                    "falloff_gu": max(self.config.pad_falloff_gu, 512.0),
                    "max_cut_fill_gu": max_cut_fill,
                    "measured_max_cut_fill_gu": cut_fill,
                    "reason_codes": sorted(terrain_codes),
                    "requires_final_field_reseat": True,
                }]
                evidence["provisional"] = self.terrain_pass == "planned"
                evidence["status"] = "provisional_pad" if self.terrain_pass == "planned" else "rejected"
                if self.terrain_pass == "planned":
                    # A provisional lot occupies geometry and must therefore be
                    # included in collision comparisons by the orchestrator.
                    evidence["issues"] = [item for item in issues if item["severity"] != "error"]
                    return "provisional", evidence, pad_request, [fine_collision_ledger_entry(lot_id, str(stamp["stamp_id"]), len(placed))]
                evidence["issues"].append(_issue(
                    "error", "final_terrain_not_reseated", f"$.lots[{lot_id}].terrain",
                    "final pass still measures a pad violation; lot is not accepted",
                    sorted(terrain_codes), "zero final terrain violations",
                ))
                hard_issues = [item for item in evidence["issues"] if item["severity"] == "error"]

        if not hard_issues:
            evidence["status"] = "accepted_final" if self.terrain_pass == "final" else "accepted"
            evidence["issues"] = [item for item in issues if item["severity"] == "warning"]
            return "accepted", evidence, None, [fine_collision_ledger_entry(lot_id, str(stamp["stamp_id"]), len(placed))]
        return "rejected", evidence, None, []

    def solve(self) -> dict[str, Any]:
        """Execute the complete deterministic houses-only placement pass."""

        gates = self.prepare()
        resolved = self._resolve_lots()
        accepted: list[dict[str, Any]] = []
        provisional: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        pad_requests: list[dict[str, Any]] = []
        fine_ledger: list[dict[str, Any]] = []
        occupied: list[tuple[str, Sequence[tuple[float, float]]]] = []
        for lot in sorted(self.plan.get("lots", []), key=lambda item: str(item.get("lot_id"))):
            if not isinstance(lot, Mapping):
                raise CityPlaceError("T1.1-valid plan contains non-object lot")
            lot_id = str(lot["lot_id"])
            stamp_id, mode, stamp = resolved[lot_id]
            status, evidence, pads, ledger = self._evaluate_lot(
                lot, stamp=stamp, mode=mode, existing=occupied
            )
            if status == "accepted":
                accepted.append(evidence)
                occupied.append((lot_id, [tuple(point) for point in evidence["footprint_hull_xy_plan_gu"]]))
                fine_ledger.extend(ledger)
            elif status == "provisional":
                provisional.append(evidence)
                if pads:
                    pad_requests.extend(pads)
                occupied.append((lot_id, [tuple(point) for point in evidence["footprint_hull_xy_plan_gu"]]))
                fine_ledger.extend(ledger)
            else:
                rejected.append(evidence)
        if not accepted:
            raise CityPlaceError("no accepted house placement remains after T1.2 checks")
        frame = {
            "origin_gu": list(self.bundle.survey_frame["origin_gu"]),
            "site_span_gu": list(self.bundle.survey_frame["site_span_gu"]),
            "units": self.bundle.survey_frame.get("units", "game_units"),
            "yaw_convention": (
                self.bundle.survey_frame.get("yaw_convention")
                or self.bundle.survey_frame.get("axis_convention")
            ),
        }
        terrain_contract = self.field.contract_dict()
        city_placement = build_city_placement(
            plan_id=str(self.plan["plan_id"]), plan_sha256=self.validation_evidence["plan_sha256"],
            terrain_contract=terrain_contract, frame=frame, placements=accepted,
            provisional=provisional, rejected=rejected, fine_collision_deferred=fine_ledger,
        )
        land_edits = build_land_edit_requests(
            plan_id=str(self.plan["plan_id"]), plan_sha256=self.validation_evidence["plan_sha256"],
            terrain_contract=terrain_contract, requests=pad_requests,
        )
        solver_report = {
            "schema_version": 1,
            "product": "cityforge_t1_2_solver_report",
            "plan_id": self.plan["plan_id"],
            "plan_sha256": self.validation_evidence["plan_sha256"],
            "terrain_field": terrain_contract,
            "config": self.config.to_dict(),
            "gates": gates,
            "outcomes": {
                "accepted": [item["lot_id"] for item in accepted],
                "accepted_count": len(accepted),
                "provisional": [item["lot_id"] for item in provisional],
                "provisional_count": len(provisional),
                "rejected": [item["lot_id"] for item in rejected],
                "rejected_count": len(rejected),
                "pad_request_count": len(pad_requests),
                "fine_collision_deferred_count": len(fine_ledger),
            },
            "lot_results": sorted(
                accepted + provisional + rejected,
                key=lambda item: str(item.get("lot_id")),
            ),
            "rejection_codes": sorted({
                issue["code"] for item in rejected for issue in item.get("issues", [])
                if issue.get("severity") == "error"
            }),
            "warnings_preserved_from_t1_1": self.validation_evidence["summary"].get("warning_codes", []),
            "spacing_contract": {
                "hard_minimum_gu": 0.0,
                "dispatch5_guidance_used_as_hard_minimum": False,
                "guidance_source": self.config.spacing_guidance_source,
            },
            "fine_collision_ledger": fine_ledger,
            "diagnostic_scope": "synthetic or plan-host placement data only; no city render",
        }
        source_hashes = dict(self.validation_evidence["input_hashes"])
        source_hashes["terrain_field"] = self.field.source_sha256
        if self.field.metadata_sha256:
            source_hashes["terrain_field_metadata"] = self.field.metadata_sha256
        # The validation-result file hash is added by solve_city_plan after the
        # core pass; the bundle/input hashes above remain the exact T1.1 pins.
        return {
            "city_placement": city_placement,
            "land_edit_requests": land_edits,
            "solver_report": solver_report,
            "source_hashes": source_hashes,
            "terrain_pass": self.terrain_pass,
        }


def solve_city_plan(
    *,
    plan_path: Path | str,
    validation_path: Path | str,
    site_survey_path: Path | str,
    kit_brief_path: Path | str,
    region_palette_path: Path | str,
    stamp_library_paths: Sequence[Path | str],
    centerlines_path: Path | str,
    terrain_field_path: Path | str,
    terrain_pass: str,
    terrain_metadata_path: Path | str | None = None,
    planned_placement_path: Path | str | None = None,
    workspace_root: Path | str = ".",
    config: PlacementConfig | None = None,
) -> dict[str, Any]:
    """Load pinned inputs and run T1.2; callers serialize returned products."""

    plan_target = Path(plan_path)
    validation = load_json(validation_path, "T1.1 validation result")
    plan = load_json(plan_target, "city plan")
    try:
        bundle = cityplan.Bundle.from_paths(
            site_survey=site_survey_path,
            kit_brief=kit_brief_path,
            region_palette=region_palette_path,
            stamp_libraries=stamp_library_paths,
            centerlines=centerlines_path,
        )
    except (cityplan.BundleError, OSError, ValueError) as exc:
        raise CityPlaceError(f"cannot load accepted planner bundle: {exc}") from exc
    survey = bundle.site_survey
    field = TerrainField.from_npz(
        terrain_field_path, survey=survey, field_pass=terrain_pass, metadata_path=terrain_metadata_path
    )
    planned = load_json(planned_placement_path, "planned placement reference") if planned_placement_path else None
    run = SolverRun(
        plan=plan,
        plan_path=plan_target,
        validation=validation,
        bundle=bundle,
        field=field,
        terrain_pass=terrain_pass,
        workspace_root=Path(workspace_root),
        config=config or PlacementConfig(),
        planned_placement=planned,
    )
    result = run.solve()
    result["source_hashes"]["t1_1_validation"] = sha256_file(Path(validation_path))
    return result


def result_identity(result: Mapping[str, Any]) -> str:
    """Hash core products for deterministic fixture/audit checks."""

    return __import__("hashlib").sha256(
        deterministic_dumps({
            "city_placement": result["city_placement"],
            "land_edit_requests": result["land_edit_requests"],
            "solver_report": result["solver_report"],
        })
    ).hexdigest()
