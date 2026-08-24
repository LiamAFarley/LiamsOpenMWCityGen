"""Phase 5 compiled rule-kit and eligibility products.

Pipeline position: consumes the immutable Phase 1-4 JSON products and emits
the consumer-facing index used by later palette/composer phases.  This module
does not discover files, call Blender, rewrite source evidence, or author
D-STAMP/ESP data.  Every row is retained with checks and stable rejection
codes; only rows with no hard rejection are placed in selectable indexes.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    validate_connection_sample,
    validate_normalized_member,
)
from .normalize import canonicalize
from .palette import load_policy_document, resolve_policy, resolve_selection


def canonical_model_key(value: str) -> str:
    return str(value).replace("/", "\\").casefold()


def _profile_id(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_sequence(value: Any, length: int | None = None) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    if length is not None and len(value) != length:
        return False
    return all(_finite(item) for item in value)


def _finite_mapping_intervals(value: Any, names: Sequence[str]) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(_finite_sequence(value.get(name), 2) and value[name][0] <= value[name][1] for name in names)


def _check(name: str, passed: bool, code: str | None = None, detail: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "code": None if passed else code,
        "detail": detail,
    }


def _audit_row(
    kind: str,
    row_id: str,
    source: Mapping[str, Any],
    checks: list[dict[str, Any]],
    payload: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    reasons = sorted({str(check["code"]) for check in checks if check["status"] != "passed" and check.get("code")})
    result: dict[str, Any] = {
        "kind": kind,
        "profile_id": row_id,
        "selectable": not reasons,
        "rejection_codes": reasons,
        "warnings": sorted(set(str(value) for value in warnings)),
        "checks": checks,
        "source": dict(source),
    }
    if payload is not None:
        result["profile"] = copy.deepcopy(dict(payload))
    return canonicalize(result)


def _source_member_map(stamp: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(member.get("source_id")): member for member in stamp.get("members", []) if isinstance(member, Mapping)}


def _validate_terrain_prior(template: Mapping[str, Any]) -> list[dict[str, Any]]:
    envelope = template.get("terrain_envelope")
    checks = [_check("terrain_envelope_object", isinstance(envelope, Mapping), "terrain_prior_missing")]
    if not isinstance(envelope, Mapping):
        return checks
    steps = envelope.get("door_step_heights_gu")
    door_count = template.get("door_count")
    checks.append(_check("door_step_count", isinstance(steps, list) and int(door_count) == len(steps) if _finite(door_count) else False, "terrain_prior_door_steps"))
    checks.append(_check("door_steps_finite", isinstance(steps, list) and all(_finite(value) for value in steps), "terrain_prior_nonfinite"))
    checks.append(_check("footprint_samples_present", _finite(envelope.get("footprint_sample_count")) and envelope["footprint_sample_count"] > 0, "terrain_prior_missing_samples"))
    for field in ("burial_depth_gu", "footprint_relief_gu", "footprint_slope_deg"):
        checks.append(_check(f"{field}_finite", _finite(envelope.get(field)), "terrain_prior_nonfinite"))
    return checks


def _validate_template(template: Mapping[str, Any], inventory: Mapping[str, Mapping[str, Any]], tolerances: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    members = template.get("members")
    checks.append(_check("template_evidence", template.get("evidence_class") == "observed_exact", "template_not_observed_exact"))
    checks.append(_check("template_seed_door", isinstance(template.get("seed_door"), str) and bool(template.get("seed_door")), "template_no_seed_door"))
    checks.append(_check("template_members", isinstance(members, list) and bool(members), "template_no_members"))
    if isinstance(members, list):
        doors = [member for member in members if isinstance(member, Mapping) and bool(member.get("is_door"))]
        checks.append(_check("template_seed_is_door", template.get("seed_door") in {member.get("source_id") for member in doors}, "template_seed_not_door"))
        checks.append(_check("template_members_valid", _validate_members(members), "template_member_invalid"))
        checks.append(_check("template_models_indexed", all(canonical_model_key(str(member.get("model_key"))) in inventory for member in members if isinstance(member, Mapping)), "template_model_unindexed"))
    else:
        checks.extend([_check("template_seed_is_door", False, "template_seed_not_door"), _check("template_members_valid", False, "template_member_invalid")])
    roundtrip = template.get("roundtrip", {})
    checks.append(_check(
        "template_roundtrip",
        _finite(roundtrip.get("max_position_residual_gu"))
        and float(roundtrip["max_position_residual_gu"]) <= float(tolerances["template_position_gu"])
        and _finite(roundtrip.get("max_rotation_matrix_residual"))
        and float(roundtrip["max_rotation_matrix_residual"]) <= float(tolerances["template_rotation_matrix"]),
        "template_roundtrip_failed",
    ))
    checks.extend(_validate_terrain_prior(template))
    return checks


def _validate_members(members: Sequence[Any]) -> bool:
    try:
        for member in members:
            if not isinstance(member, Mapping):
                return False
            validate_normalized_member(member)
    except (TypeError, ValueError, KeyError):
        return False
    return True


def _validate_facades(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    facades = profile.get("facades")
    checks = [_check("facade_rows_present", isinstance(facades, list) and bool(facades), "facade_empty")]
    if not isinstance(facades, list):
        return checks
    valid = True
    for facade in facades:
        polygon = facade.get("polygon_uz") if isinstance(facade, Mapping) else None
        frame = facade.get("outward_frame") if isinstance(facade, Mapping) else None
        valid = valid and isinstance(facade, Mapping) and isinstance(facade.get("facade_id"), str)
        valid = valid and isinstance(polygon, list) and len(polygon) >= 3
        valid = valid and all(_finite_sequence(point, 2) for point in polygon) if isinstance(polygon, list) else False
        valid = valid and isinstance(frame, Mapping) and all(_finite_sequence(frame.get(axis), 3) for axis in ("n", "u", "v"))
        valid = valid and _finite(facade.get("area_gu2")) and float(facade["area_gu2"]) > 0.0
    checks.append(_check("facade_geometry", valid, "facade_geometry_invalid"))
    return checks


def _validate_native_profile(profile: Mapping[str, Any]) -> bool:
    bounds = profile.get("bounds_local_gu")
    if not isinstance(bounds, Mapping):
        return False
    if not _finite_sequence(bounds.get("min"), 3) or not _finite_sequence(bounds.get("max"), 3):
        return False
    if any(float(lo) > float(hi) for lo, hi in zip(bounds["min"], bounds["max"])):
        return False
    return _finite_sequence(profile.get("z_band_gu"), 2) and profile["z_band_gu"][0] <= profile["z_band_gu"][1]


def _validate_mount(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    frame = profile.get("mount_frame")
    checks = [
        _check("mount_frame_object", isinstance(frame, Mapping), "mount_frame_missing"),
        _check("mount_contact_present", profile.get("contact_geometry_kind") in {"polygon", "interval"}, "mount_contact_missing"),
        _check("mount_authored_scales", isinstance(profile.get("authored_scales"), list) and bool(profile.get("authored_scales")) and all(_finite(value) and float(value) > 0.0 for value in profile["authored_scales"]), "mount_scale_invalid"),
        _check("mount_orientation_evidence", isinstance(profile.get("orientation_evidence"), str) and bool(profile.get("orientation_evidence")), "mount_orientation_missing"),
    ]
    if isinstance(frame, Mapping):
        checks.extend([
            _check("mount_normal_axis", frame.get("normal_axis") in {"x", "y", "z"}, "mount_frame_invalid"),
            _check("mount_frame_axes", all(_finite_sequence(frame.get(axis), 3) for axis in ("n", "u_tangent", "v_up")), "mount_frame_invalid"),
        ])
    for envelope_name in ("occupied_envelope_gu", "clearance_envelope_gu"):
        checks.append(_check(envelope_name, _finite_mapping_intervals(profile.get(envelope_name), ("normal_gu", "tangent_gu", "up_gu")), "mount_envelope_invalid"))
    evidence = profile.get("front_back_evidence")
    checks.append(_check("front_back_evidence", isinstance(evidence, Mapping) and _finite(evidence.get("front_face_area_gu2")) and _finite(evidence.get("back_face_area_gu2")), "mount_front_back_missing"))
    return checks


def _validate_roof(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    patches = [patch for patch in profile.get("patches", []) if isinstance(patch, Mapping) and patch.get("status") == "eligible"]
    checks = [_check("roof_eligible_patch", bool(patches), "roof_no_eligible_patch")]
    valid = True
    for patch in patches:
        valid = valid and isinstance(patch.get("polygon_pieces_uv"), list) and bool(patch["polygon_pieces_uv"])
        valid = valid and all(_finite_sequence(patch.get(axis), 3) for axis in ("u", "v", "n"))
        valid = valid and _finite_sequence(patch.get("origin_gu"), 3)
        valid = valid and _finite(patch.get("area_gu2")) and float(patch["area_gu2"]) > 0.0
    checks.append(_check("roof_patch_geometry", valid, "roof_geometry_invalid"))
    return checks


def _validate_relation(
    relation: Mapping[str, Any],
    roof_by_model: Mapping[str, Mapping[str, Any]],
    selection_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    stamp_by_site_and_id: Mapping[tuple[str, str], Mapping[str, Any]],
    tolerances: Mapping[str, Any],
) -> list[dict[str, Any]]:
    key = (str(relation.get("source_stamp_id")), str(relation.get("dormer_member_source_id")))
    selected = selection_by_key.get(key)
    checks = [
        _check("relation_observed_exact", relation.get("evidence_class") == "observed_exact", "relation_not_observed_exact"),
        _check("relation_selection", selected is not None, "relation_not_selected"),
        _check("relation_source_stamp", (str(relation.get("site_id")), str(relation.get("source_stamp_id"))) in stamp_by_site_and_id, "relation_source_missing"),
    ]
    if selected is None:
        checks.append(_check("relation_host_profile", False, "relation_host_missing"))
    else:
        model_key = canonical_model_key(str(selected["shell_model_key"]))
        roof = roof_by_model.get(model_key)
        checks.append(_check("relation_host_profile", roof is not None, "relation_host_missing"))
        patch_ids = {str(patch.get("patch_id")) for patch in roof.get("patches", []) if patch.get("status") == "eligible"} if roof else set()
        checks.append(_check("relation_roof_patch", str(relation.get("roof_patch_id")) in patch_ids, "relation_roof_patch_missing"))
        checks.append(_check("relation_model_keys", canonical_model_key(str(relation.get("dormer_model_key"))) == canonical_model_key(str(selected["dormer_model_key"])), "relation_model_mismatch"))
    contact = relation.get("contact_evidence")
    clearance = relation.get("clearance_evidence")
    checks.append(_check("relation_contact_evidence", isinstance(contact, Mapping) and _finite(contact.get("nearest_plane_distance_gu")) and _finite(contact.get("projected_overlap_area_gu2")), "relation_contact_missing"))
    checks.append(_check("relation_clearance_evidence", isinstance(clearance, Mapping) and _finite(clearance.get("projected_min_distance_to_usable_region_gu")), "relation_clearance_missing"))
    roundtrip = relation.get("roundtrip")
    checks.append(_check("relation_roundtrip", isinstance(roundtrip, Mapping) and _finite(roundtrip.get("position_residual_gu")) and float(roundtrip["position_residual_gu"]) <= float(tolerances["roof_relation_position_gu"]) and _finite(roundtrip.get("rotation_matrix_residual")) and float(roundtrip["rotation_matrix_residual"]) <= float(tolerances["roof_relation_rotation_matrix"]), "relation_roundtrip_failed"))
    return checks


def _source_dstamp_preflight(site_id: str, stamp: Mapping[str, Any]) -> dict[str, Any]:
    """Audit source-stamp shape without pretending it is generated output."""

    members = stamp.get("members")
    checks = {
        "members_present": isinstance(members, list) and bool(members),
        "member_transforms_finite": bool(members) and all(
            isinstance(member, Mapping)
            and _finite_sequence(member.get("offset_gu"), 3)
            and _finite_sequence(member.get("rotation"), 3)
            and _finite(member.get("scale"))
            and float(member["scale"]) > 0.0
            for member in members
        ) if isinstance(members, list) else False,
        "door_count_matches": isinstance(members, list) and _finite(stamp.get("door_count")) and int(stamp["door_count"]) == sum(bool(member.get("is_door")) for member in members if isinstance(member, Mapping)),
        "terrain_prior_present": isinstance(stamp.get("terrain_envelope"), Mapping),
        "door_provenance_complete": isinstance(members, list) and all(isinstance(member.get("door"), Mapping) for member in members if isinstance(member, Mapping) and bool(member.get("is_door"))),
    }
    return {
        "site_id": site_id,
        "stamp_id": stamp.get("stamp_id"),
        "checks": checks,
        "source_shape_valid": all(checks[key] for key in ("members_present", "member_transforms_finite", "door_count_matches", "terrain_prior_present")),
        "generated_dstamp_ready": checks["door_provenance_complete"],
        "note": "source rows are not generated D-STAMP output; missing door provenance is retained for the composer to author",
    }


def _validate_connection(sample: Mapping[str, Any]) -> tuple[bool, str | None]:
    try:
        validate_connection_sample(sample)
    except (TypeError, ValueError, KeyError) as exc:
        return False, str(exc).split(":", 1)[0]
    return True, None


def _compile_connection_rows(
    site_id: str,
    document: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]],
    audit_rows: list[dict[str, Any]],
    compatibility_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    for rule in list(document.get("rules", [])) + list(document.get("attachment_contacts", [])):
        if not isinstance(rule, Mapping):
            audit_rows.append(_audit_row("connection", f"connection:{site_id}:invalid", {"site_id": site_id}, [_check("row_object", False, "connection_row_invalid")]))
            continue
        sample_rows = rule.get("samples") if isinstance(rule.get("samples"), list) else [rule]
        checks: list[dict[str, Any]] = []
        valid_samples = True
        for sample in sample_rows:
            valid, code = _validate_connection(sample) if isinstance(sample, Mapping) else (False, "relation_frame_invalid")
            valid_samples = valid_samples and valid
            if not valid and code:
                checks.append(_check("sample_contract", False, code))
            if isinstance(sample, Mapping):
                checks.append(_check("sample_evidence", sample.get("evidence_class") == "observed_exact", "connection_not_observed_exact"))
                checks.append(_check("sample_models_indexed", canonical_model_key(str(sample.get("model_a"))) in inventory and canonical_model_key(str(sample.get("model_b"))) in inventory, "connection_model_unindexed"))
        if "samples" in rule:
            checks.append(_check("aggregate_occurrence_count", rule.get("occurrence_count") == len(sample_rows), "connection_count_mismatch"))
            sample_sites = {str(sample.get("witness", {}).get("site_id")) for sample in sample_rows if isinstance(sample, Mapping)}
            sample_stamps = {str(sample.get("witness", {}).get("source_stamp_id")) for sample in sample_rows if isinstance(sample, Mapping)}
            checks.append(_check("aggregate_independent_site_count", rule.get("independent_site_count") == len(sample_sites), "connection_count_mismatch"))
            checks.append(_check("aggregate_independent_stamp_count", rule.get("independent_stamp_count") == len(sample_stamps), "connection_count_mismatch"))
        checks.append(_check("sample_contract", valid_samples, "relation_frame_invalid"))
        model_a = canonical_model_key(str(rule.get("model_a", sample_rows[0].get("model_a") if sample_rows and isinstance(sample_rows[0], Mapping) else "")))
        model_b = canonical_model_key(str(rule.get("model_b", sample_rows[0].get("model_b") if sample_rows and isinstance(sample_rows[0], Mapping) else "")))
        identity = f"{model_a}|{model_b}" if "samples" in rule else str(sample_rows[0].get("sample_id", f"{model_a}|{model_b}"))
        row_id = _profile_id("connection", f"{site_id}:{identity}")
        compiled_row = {"site_id": site_id, "connection_id": row_id, **dict(rule)}
        audit = _audit_row("connection", row_id, {"site_id": site_id, "source_sample_ids": [sample.get("sample_id") for sample in sample_rows if isinstance(sample, Mapping)]}, checks, compiled_row)
        audit_rows.append(audit)
        if audit["selectable"]:
            compiled.append(compiled_row)
            for sample in sample_rows:
                compatibility_edges.append({
                    "edge_id": f"{row_id}:{sample.get('sample_id')}",
                    "connection_id": row_id,
                    "model_a": sample.get("model_a"),
                    "model_b": sample.get("model_b"),
                    "authored_scale_a": sample.get("authored_scale_a"),
                    "authored_scale_b": sample.get("authored_scale_b"),
                    "evidence_class": sample.get("evidence_class"),
                    "witness": sample.get("witness"),
                })
    return compiled


def compile_rule_kit(config: Mapping[str, Any], documents: Mapping[str, Any], palette_document: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compile loaded JSON documents into kit, eligibility, and palette products."""

    inputs = config["inputs"]
    inventory_rows = {canonical_model_key(row["model_key"]): row for row in documents["phase02_inventory"]["models"]}
    native_profiles = {canonical_model_key(row["model_key"]): row for row in documents["phase02_profiles"]["meshes"]}
    facades = {canonical_model_key(row["model_key"]): row for row in documents["phase03b_facades"]["profiles"]}
    mounts = {canonical_model_key(row["model_key"]): row for row in documents["phase03b_mounts"]["profiles"]}
    roof_profiles = {canonical_model_key(row["model_key"]): row for row in documents["phase04_roofs"]["profiles"]}
    review = config["review_registry"]
    reviewed_shells = {canonical_model_key(value) for value in review["shell_models"]}
    reviewed_mounts = {canonical_model_key(value) for value in review["mount_models"]}
    reviewed_roofs = {canonical_model_key(value) for value in review["roof_models"]}
    audit_rows: list[dict[str, Any]] = []
    selectable: dict[str, list[str]] = {kind: [] for kind in ("shells", "mounts", "access", "roofs", "dormers", "templates", "connections")}
    shell_rows: list[dict[str, Any]] = []
    mount_rows: list[dict[str, Any]] = []
    access_rows: list[dict[str, Any]] = []
    roof_rows: list[dict[str, Any]] = []
    dormer_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    connection_rows: list[dict[str, Any]] = []
    compatibility_edges: list[dict[str, Any]] = []

    for model_key, profile in sorted(facades.items()):
        checks = _validate_facades(profile)
        checks.append(_check("inventory_cross_reference", model_key in inventory_rows, "profile_model_unindexed"))
        checks.append(_check("native_profile_cross_reference", model_key in native_profiles, "profile_native_missing"))
        checks.append(_check("native_profile_geometry", model_key in native_profiles and _validate_native_profile(native_profiles[model_key]), "native_profile_invalid"))
        checks.append(_check("review_registry", model_key in reviewed_shells, "profile_not_reviewed"))
        row_id = _profile_id("shell", model_key)
        audit = _audit_row("shell", row_id, {"model_key": model_key, "resolved_path": profile.get("resolved_path")}, checks, profile)
        audit_rows.append(audit)
        shell_rows.append(audit)
        if audit["selectable"]:
            selectable["shells"].append(row_id)

    for model_key, profile in sorted(mounts.items()):
        checks = _validate_mount(profile)
        checks.append(_check("inventory_cross_reference", model_key in inventory_rows, "profile_model_unindexed"))
        checks.append(_check("native_profile_cross_reference", model_key in native_profiles, "profile_native_missing"))
        checks.append(_check("native_profile_geometry", model_key in native_profiles and _validate_native_profile(native_profiles[model_key]), "native_profile_invalid"))
        checks.append(_check("review_registry", model_key in reviewed_mounts, "profile_not_reviewed"))
        row_id = _profile_id("mount", model_key)
        audit = _audit_row("mount", row_id, {"model_key": model_key, "resolved_path": profile.get("resolved_path")}, checks, profile)
        audit_rows.append(audit)
        mount_rows.append(audit)
        if audit["selectable"]:
            selectable["mounts"].append(row_id)
    mount_selectable = {row["profile_id"] for row in mount_rows if row["selectable"]}

    for bundle in documents["phase03b_access"].get("bundles", []):
        if not isinstance(bundle, Mapping):
            continue
        checks = [
            _check("access_evidence", bundle.get("evidence_class") == "observed_exact", "access_not_observed_exact"),
            _check("door_member", isinstance(bundle.get("door_member"), Mapping) and bundle["door_member"].get("record_type") == "DOOR", "access_door_missing"),
            _check("frame_member", isinstance(bundle.get("frame_member"), Mapping) and bundle["frame_member"].get("record_type") == "STAT", "access_frame_missing"),
            _check("heading_finite", _finite(bundle.get("outward_heading_in_slot_deg")), "access_heading_missing"),
            _check("grade_support", isinstance(bundle.get("grade_support"), Mapping) and isinstance(bundle["grade_support"].get("door_step_heights_gu"), list) and all(_finite(value) for value in bundle["grade_support"]["door_step_heights_gu"]), "access_grade_invalid"),
        ]
        frame_key = canonical_model_key(str(bundle.get("frame_member", {}).get("model_key", "")))
        checks.append(_check("frame_mount_profile", _profile_id("mount", frame_key) in mount_selectable, "access_frame_unselectable"))
        row_id = _profile_id("access", str(bundle.get("access_bundle_id", "")))
        audit = _audit_row("access", row_id, {"site_id": bundle.get("witness", {}).get("site_id"), "stamp_id": bundle.get("witness", {}).get("stamp_id")}, checks, bundle)
        audit_rows.append(audit)
        access_rows.append(audit)
        if audit["selectable"]:
            selectable["access"].append(row_id)

    for model_key, profile in sorted(roof_profiles.items()):
        checks = _validate_roof(profile)
        checks.append(_check("review_registry", model_key in reviewed_roofs, "profile_not_reviewed"))
        row_id = _profile_id("roof", model_key)
        audit = _audit_row("roof", row_id, {"model_key": model_key, "resolved_path": profile.get("resolved_path")}, checks, profile)
        audit_rows.append(audit)
        roof_rows.append(audit)
        if audit["selectable"]:
            selectable["roofs"].append(row_id)

    source_stamps: dict[tuple[str, str], Mapping[str, Any]] = {}
    dstamp_preflight: list[dict[str, Any]] = []
    for site in config["inputs"]["source_sites"]:
        for stamp in documents[f"stamps:{site['site_id']}"]["stamps"]:
            source_stamps[(site["site_id"], str(stamp.get("stamp_id")))] = stamp
            dstamp_preflight.append(_source_dstamp_preflight(site["site_id"], stamp))
    selection_by_key = {
        (str(row["stamp_id"]), str(row["dormer_member_source_id"])): row
        for row in documents["phase04_selection"].get("selected_relations", [])
    }
    roof_by_model = roof_profiles
    tolerances = config["tolerances"]
    for relation in documents["phase04_dormers"].get("relations", []):
        key = f"{relation.get('source_stamp_id')}:{relation.get('dormer_member_source_id')}"
        checks = _validate_relation(relation, roof_by_model, selection_by_key, source_stamps, tolerances)
        row_id = _profile_id("dormer", f"{relation.get('site_id')}:{key}")
        audit = _audit_row("dormer", row_id, {"site_id": relation.get("site_id"), "source_stamp_id": relation.get("source_stamp_id")}, checks, relation)
        audit_rows.append(audit)
        dormer_rows.append(audit)
        if audit["selectable"]:
            selectable["dormers"].append(row_id)

    for site in config["inputs"]["source_sites"]:
        site_id = site["site_id"]
        template_doc = documents[f"templates:{site_id}"]
        for template in template_doc.get("templates", []):
            row_id = _profile_id("template", f"{site_id}:{template.get('stamp_id')}")
            source_preflight = next((row for row in dstamp_preflight if row["site_id"] == site_id and row["stamp_id"] == template.get("stamp_id")), None)
            warnings = [] if source_preflight and source_preflight["generated_dstamp_ready"] else ["dstamp_provenance_missing"]
            audit = _audit_row("template", row_id, {"site_id": site_id, "stamp_id": template.get("stamp_id")}, _validate_template(template, inventory_rows, config["tolerances"]), template, warnings)
            audit_rows.append(audit)
            template_rows.append(audit)
            if audit["selectable"]:
                selectable["templates"].append(row_id)
        site_connection_rows = _compile_connection_rows(site_id, documents[f"connections:{site_id}"], inventory_rows, audit_rows, compatibility_edges)
        connection_rows.extend(site_connection_rows)
        selectable["connections"].extend(row["connection_id"] for row in site_connection_rows)

    # Cross-reference palette IDs only after all eligibility decisions exist.
    policy_document = load_policy_document(palette_document)
    for policy_name, policy in [("settlement_defaults", policy_document["settlement_defaults"])]:
        _validate_palette_references(policy_name, policy, selectable)
    for group in ("district_overrides", "parcel_overrides"):
        for key, override in policy_document[group].items():
            resolved = resolve_policy(policy_document, key.split(":", 1)[0], key.split(":", 1)[1] if ":" in key else None) if group == "parcel_overrides" else resolve_policy(policy_document, key)
            _validate_palette_references(f"{group}:{key}", resolved, selectable)

    resolution_rows = []
    for request in config.get("selection_requests", []):
        policy = resolve_policy(policy_document, request["district_id"], request["parcel_id"])
        _validate_palette_references(f"request:{request['request_id']}", policy, selectable)
        selection = resolve_selection(policy, request_id=request["request_id"], master_seed=request["master_seed"], requested_size=request["requested_size"])
        resolution_rows.append({
            "request": copy.deepcopy(request),
            "resolved_policy": policy,
            "selection": selection,
        })

    source_provenance = {
        "config_inputs": canonicalize(inputs),
        "review_evidence": list(config["review_registry"]["evidence"]),
        "documents": [
            {"key": key, "schema_version": value.get("schema_version") if isinstance(value, Mapping) else None}
            for key, value in sorted(documents.items()) if not key.startswith("stamps:") and not key.startswith("templates:") and not key.startswith("connections:")
        ],
    }
    compiled = canonicalize({
        "schema_version": 1,
        "rule_kit_id": config["rule_kit_id"],
        "kit_id": config["kit_id"],
        "source_provenance": source_provenance,
        "model_profiles": [
            {"profile_id": _profile_id("model", model_key), "model_key": model_key, "profile": profile}
            for model_key, profile in sorted(native_profiles.items())
        ],
        "selectable": {key: sorted(value) for key, value in selectable.items()},
        "shell_profiles": shell_rows,
        "mount_profiles": mount_rows,
        "access_bundles": access_rows,
        "roof_profiles": roof_rows,
        "dormer_relations": dormer_rows,
        "templates": template_rows,
        "connection_rules": connection_rows,
        "compatibility_edges": compatibility_edges,
        "terrain_priors": [
            {
                "template_id": row["profile_id"],
                "site_id": row["source"].get("site_id"),
                "stamp_id": row["source"].get("stamp_id"),
                "terrain_envelope": row["profile"].get("terrain_envelope", {}),
            }
            for row in template_rows
            if row["selectable"]
        ],
        "dstamp_preflight": dstamp_preflight,
        "counts": {"audit_rows": len(audit_rows), "selectable_rows": sum(1 for row in audit_rows if row["selectable"]), "rejected_rows": sum(1 for row in audit_rows if not row["selectable"])},
    })
    eligibility = canonicalize({
        "schema_version": 1,
        "rule_kit_id": config["rule_kit_id"],
        "rows": [{key: value for key, value in row.items() if key != "profile"} for row in audit_rows],
        "dstamp_preflight": dstamp_preflight,
        "counts": {
            "total": len(audit_rows),
            "selectable": sum(1 for row in audit_rows if row["selectable"]),
            "rejected": sum(1 for row in audit_rows if not row["selectable"]),
            "warnings": {
                warning: sum(warning in row["warnings"] for row in audit_rows)
                for warning in sorted({warning for row in audit_rows for warning in row["warnings"]})
            },
            "rejection_codes": {
                code: sum(code in row["rejection_codes"] for row in audit_rows)
                for code in sorted({code for row in audit_rows for code in row["rejection_codes"]})
            },
        },
    })
    resolution = canonicalize({"schema_version": 1, "rule_kit_id": config["rule_kit_id"], "requests": resolution_rows})
    return compiled, eligibility, resolution


def _validate_palette_references(name: str, palette: Mapping[str, Any], selectable: Mapping[str, Sequence[str]]) -> None:
    domains = {
        "shells.allowed_profile_ids": (palette["shells"]["allowed_profile_ids"], set(selectable["shells"])),
        "shells.weights": (list(palette["shells"]["weights"]), set(selectable["shells"])),
        "access.primary_bundle_weights": (list(palette["access"]["primary_bundle_weights"]), set(selectable["access"])),
        "attachments.allowed_profile_ids": (palette["attachments"]["allowed_profile_ids"], set(selectable["mounts"])),
        "attachments.window_rates": (list(palette["attachments"]["window_rates"]), set(selectable["mounts"])),
    }
    for field, (values, allowed) in domains.items():
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"palette_reference_invalid: {name}.{field} references ineligible or unknown ID(s): {', '.join(unknown)}")
    shell_allowed = set(palette["shells"]["allowed_profile_ids"])
    if set(palette["shells"]["weights"]) - shell_allowed:
        raise ValueError(f"palette_reference_invalid: {name}.shells.weights contains an ID outside allowed_profile_ids")
    mount_allowed = set(palette["attachments"]["allowed_profile_ids"])
    if set(palette["attachments"]["window_rates"]) - mount_allowed:
        raise ValueError(f"palette_reference_invalid: {name}.attachments.window_rates contains an ID outside allowed_profile_ids")


__all__ = ["canonical_model_key", "compile_rule_kit"]
