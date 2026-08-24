"""Strict Phase 0 contracts for the building-generation rule kit.

The validators are deliberately dependency-light and fail closed with stable
reason-code prefixes.  They validate JSON-ready mappings only; geometry and
relation semantics remain in the normalization/rebuild modules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


EVIDENCE_CLASSES = {"observed_exact", "measured_transfer", "reviewed_semantic", "ineligible"}
WINDOW_MODES = {"observed_slots", "measured_rhythm", "none"}
RECORD_TYPES = {"STAT", "DOOR", "ACTI"}


def _fail(code: str, detail: str) -> None:
    raise ValueError(f"{code}: {detail}")


def _mapping(doc: Any, code: str = "contract_invalid") -> Mapping[str, Any]:
    if not isinstance(doc, Mapping):
        _fail(code, "document must be an object")
    return doc


def _required(doc: Mapping[str, Any], names: Sequence[str], code: str) -> None:
    missing = [name for name in names if name not in doc]
    if missing:
        _fail(code, "missing required field(s): " + ", ".join(missing))


def _string(value: Any, field: str, code: str = "contract_invalid") -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code, f"{field} must be a non-empty string")
    return value


def _number(value: Any, field: str, code: str = "contract_invalid") -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        _fail(code, f"{field} must be finite")
    return float(value)


def _triplet(value: Any, field: str, code: str = "contract_invalid") -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        _fail(code, f"{field} must contain three numbers")
    return [_number(item, f"{field}[{index}]", code) for index, item in enumerate(value)]


def _matrix3(value: Any, field: str, code: str = "contract_invalid") -> list[list[float]]:
    if not isinstance(value, Sequence) or len(value) != 3:
        _fail(code, f"{field} must be a 3x3 matrix")
    rows = []
    for index, row in enumerate(value):
        if not isinstance(row, Sequence) or len(row) != 3:
            _fail(code, f"{field}[{index}] must contain three numbers")
        rows.append([_number(item, f"{field}[{index}][{j}]", code) for j, item in enumerate(row)])
    return rows


def _interval(value: Any, field: str, *, nullable: bool = False, code: str = "contract_invalid") -> list[float | None]:
    if not isinstance(value, Sequence) or len(value) != 2:
        _fail(code, f"{field} must contain two values")
    if nullable and all(item is None for item in value):
        return [None, None]
    if any(item is None for item in value):
        _fail(code, f"{field} must be fully numeric or [null, null]")
    result = [_number(item, f"{field}[{index}]", code) for index, item in enumerate(value)]
    if result[0] > result[1]:
        _fail(code, f"{field} lower bound exceeds upper bound")
    return result


def _optional_destination(door: Mapping[str, Any], code: str) -> None:
    for field in ("destination_cell", "destination_position_gu", "destination_rotation"):
        if field not in door:
            _fail(code, f"door.{field} is required")
    if door["destination_cell"] is not None:
        _string(door["destination_cell"], "door.destination_cell", code)
    if door["destination_position_gu"] is not None:
        _triplet(door["destination_position_gu"], "door.destination_position_gu", code)
    if door["destination_rotation"] is not None:
        _triplet(door["destination_rotation"], "door.destination_rotation", code)


def _validate_door_block(value: Any, code: str) -> None:
    door = _mapping(value, code)
    _optional_destination(door, code)


def validate_source_member(doc: Any) -> None:
    """Validate an immutable source-world xFa member row."""

    member = _mapping(doc, "source_member_invalid")
    _required(
        member,
        (
            "source_id", "object_id", "model_key", "record_type", "category",
            "is_door", "structural_role", "offset_gu", "rotation", "scale",
            "outward_heading_deg",
        ),
        "source_member_invalid",
    )
    for field in ("source_id", "object_id", "model_key", "category", "structural_role"):
        _string(member[field], f"member.{field}", "source_member_invalid")
    if member["record_type"] not in RECORD_TYPES:
        _fail("source_member_invalid", "member.record_type is unsupported")
    if not isinstance(member["is_door"], bool):
        _fail("source_member_invalid", "member.is_door must be boolean")
    _triplet(member["offset_gu"], "member.offset_gu", "source_member_invalid")
    rotation = _triplet(member["rotation"], "member.rotation", "source_member_invalid")
    # Source transforms are float32.  Values one ulp above +/-2*pi are valid
    # serialized radians; materially larger values remain a unit failure.
    if any(abs(value) > 2 * math.pi + 1.0e-5 for value in rotation):
        _fail("source_member_invalid", "member.rotation is not radians")
    scale = _number(member["scale"], "member.scale", "source_member_invalid")
    if scale <= 0:
        _fail("source_member_invalid", "member.scale must be positive")
    heading = member["outward_heading_deg"]
    if heading is not None:
        _number(heading, "member.outward_heading_deg", "source_member_invalid")
    if member["is_door"] and member["record_type"] != "DOOR":
        _fail("source_member_invalid", "door member must have record_type DOOR")
    if "door" in member:
        _validate_door_block(member["door"], "source_member_invalid")


def validate_normalized_member(doc: Any) -> None:
    """Validate a template-local member with full TES3 rotation preserved."""

    member = _mapping(doc, "normalized_member_invalid")
    _required(
        member,
        (
            "source_id", "object_id", "model_key", "record_type", "structural_role",
            "offset_local_gu", "rotation_local_rad", "scale",
        ),
        "normalized_member_invalid",
    )
    for field in ("source_id", "object_id", "model_key", "structural_role"):
        _string(member[field], f"member.{field}", "normalized_member_invalid")
    if member["record_type"] not in RECORD_TYPES:
        _fail("normalized_member_invalid", "member.record_type is unsupported")
    _triplet(member["offset_local_gu"], "member.offset_local_gu", "normalized_member_invalid")
    rotation = _triplet(member["rotation_local_rad"], "member.rotation_local_rad", "normalized_member_invalid")
    if any(abs(value) > 2 * math.pi for value in rotation):
        _fail("normalized_member_invalid", "member.rotation_local_rad exceeds 2*pi")
    if _number(member["scale"], "member.scale", "normalized_member_invalid") <= 0:
        _fail("normalized_member_invalid", "member.scale must be positive")
    if "is_door" in member and not isinstance(member["is_door"], bool):
        _fail("normalized_member_invalid", "member.is_door must be boolean")
    if bool(member.get("is_door")):
        if member["record_type"] != "DOOR":
            _fail("normalized_member_invalid", "door member must have record_type DOOR")
        if "door" in member:
            _validate_door_block(member["door"], "normalized_member_invalid")


def validate_connection_sample(doc: Any) -> None:
    """Validate one ordered A-local shell/piece relation sample."""

    sample = _mapping(doc, "relation_frame_invalid")
    _required(
        sample,
        (
            "sample_id", "model_a", "model_b", "authored_scale_a", "authored_scale_b",
            "offset_b_in_a_frame_gu", "relative_engine_matrix_3x3",
            "source_rotation_a_rad", "source_rotation_b_rad", "contact_distance_gu",
            "allowed_contact_interval_gu", "witness", "evidence_class",
        ),
        "relation_frame_invalid",
    )
    _string(sample["sample_id"], "sample.sample_id", "relation_frame_invalid")
    _string(sample["model_a"], "sample.model_a", "relation_frame_invalid")
    _string(sample["model_b"], "sample.model_b", "relation_frame_invalid")
    for field in ("authored_scale_a", "authored_scale_b"):
        if _number(sample[field], f"sample.{field}", "relation_frame_invalid") <= 0:
            _fail("relation_frame_invalid", f"sample.{field} must be positive")
    _triplet(sample["offset_b_in_a_frame_gu"], "sample.offset_b_in_a_frame_gu", "relation_frame_invalid")
    _matrix3(sample["relative_engine_matrix_3x3"], "sample.relative_engine_matrix_3x3", "relation_frame_invalid")
    for field in ("source_rotation_a_rad", "source_rotation_b_rad"):
        rotation = _triplet(sample[field], f"sample.{field}", "relation_frame_invalid")
        if any(abs(value) > 2 * math.pi for value in rotation):
            _fail("relation_frame_invalid", f"sample.{field} is not radians")
    distance = sample["contact_distance_gu"]
    if distance is not None and _number(distance, "sample.contact_distance_gu", "relation_frame_invalid") < 0:
        _fail("relation_frame_invalid", "sample.contact_distance_gu must be non-negative")
    _interval(sample["allowed_contact_interval_gu"], "sample.allowed_contact_interval_gu", nullable=True, code="relation_frame_invalid")
    witness = _mapping(sample["witness"], "relation_frame_invalid")
    _required(witness, ("site_id", "source_stamp_id", "ref_a", "ref_b", "direct_contact_id"), "relation_frame_invalid")
    for field in ("site_id", "source_stamp_id", "ref_a", "ref_b"):
        _string(witness[field], f"sample.witness.{field}", "relation_frame_invalid")
    if witness["direct_contact_id"] is not None:
        _string(witness["direct_contact_id"], "sample.witness.direct_contact_id", "relation_frame_invalid")
    if sample["evidence_class"] not in EVIDENCE_CLASSES:
        _fail("relation_frame_invalid", "sample.evidence_class is unsupported")


def validate_access_bundle(doc: Any) -> None:
    """Validate a complete door/frame/grade access bundle."""

    bundle = _mapping(doc, "access_bundle_invalid")
    _required(
        bundle,
        (
            "access_bundle_id", "slot_interface_id", "outward_heading_in_slot_deg",
            "door_member", "frame_member", "optional_grade_members",
            "members_in_door_frame", "grade_support", "door_record_provenance",
            "witness", "evidence_class",
        ),
        "access_bundle_invalid",
    )
    _string(bundle["access_bundle_id"], "bundle.access_bundle_id", "access_bundle_invalid")
    _string(bundle["slot_interface_id"], "bundle.slot_interface_id", "access_bundle_invalid")
    _number(bundle["outward_heading_in_slot_deg"], "bundle.outward_heading_in_slot_deg", "access_bundle_invalid")
    door = _mapping(bundle["door_member"], "access_bundle_invalid")
    _required(door, ("model_key", "record_type", "scale"), "access_bundle_invalid")
    _string(door["model_key"], "bundle.door_member.model_key", "access_bundle_invalid")
    if door["record_type"] != "DOOR":
        _fail("access_bundle_invalid", "door_member.record_type must be DOOR")
    if _number(door["scale"], "bundle.door_member.scale", "access_bundle_invalid") <= 0:
        _fail("access_bundle_invalid", "door_member.scale must be positive")
    frame = bundle["frame_member"]
    if not isinstance(frame, Mapping):
        _fail("door_frame_missing", "ordinary access bundle has no frame_member")
    _required(frame, ("model_key", "record_type", "scale"), "access_bundle_invalid")
    _string(frame["model_key"], "bundle.frame_member.model_key", "access_bundle_invalid")
    if frame["record_type"] != "STAT":
        _fail("access_bundle_invalid", "frame_member.record_type must be STAT")
    if _number(frame["scale"], "bundle.frame_member.scale", "access_bundle_invalid") <= 0:
        _fail("access_bundle_invalid", "frame_member.scale must be positive")
    if not isinstance(bundle["optional_grade_members"], list) or not isinstance(bundle["members_in_door_frame"], list):
        _fail("access_bundle_invalid", "grade/member lists must be arrays")
    support = _mapping(bundle["grade_support"], "access_bundle_invalid")
    _required(support, ("min_step_gu", "max_step_gu", "allowed_ground_penetration_gu"), "access_bundle_invalid")
    if _number(support["min_step_gu"], "grade_support.min_step_gu", "access_bundle_invalid") > _number(support["max_step_gu"], "grade_support.max_step_gu", "access_bundle_invalid"):
        _fail("access_bundle_invalid", "grade step interval is inverted")
    _interval(support["allowed_ground_penetration_gu"], "grade_support.allowed_ground_penetration_gu", code="access_bundle_invalid")
    if not isinstance(bundle["door_record_provenance"], Mapping) or not isinstance(bundle["witness"], Mapping):
        _fail("access_bundle_invalid", "provenance and witness must be objects")
    if bundle["evidence_class"] not in EVIDENCE_CLASSES:
        _fail("access_bundle_invalid", "bundle.evidence_class is unsupported")


def validate_model_profile(doc: Any) -> None:
    """Validate the minimal native model profile contract."""

    profile = _mapping(doc, "model_profile_invalid")
    _required(profile, ("model_key", "native_bounds", "ground_band_z_range", "bottom_penetration_range"), "model_profile_invalid")
    _string(profile["model_key"], "profile.model_key", "model_profile_invalid")
    bounds = _mapping(profile["native_bounds"], "model_profile_invalid")
    _required(bounds, ("min", "max"), "model_profile_invalid")
    _triplet(bounds["min"], "profile.native_bounds.min", "model_profile_invalid")
    _triplet(bounds["max"], "profile.native_bounds.max", "model_profile_invalid")
    _interval(profile["ground_band_z_range"], "profile.ground_band_z_range", code="model_profile_invalid")
    _interval(profile["bottom_penetration_range"], "profile.bottom_penetration_range", code="model_profile_invalid")


def _weights(value: Any, field: str, code: str) -> None:
    if not isinstance(value, Mapping):
        _fail(code, f"{field} must be an object")
    for key, weight in value.items():
        _string(key, f"{field} key", code)
        if _number(weight, f"{field}.{key}", code) < 0:
            _fail(code, f"{field}.{key} must be non-negative")


def _string_list(value: Any, field: str, code: str) -> None:
    if not isinstance(value, list):
        _fail(code, f"{field} must be an array")
    for index, item in enumerate(value):
        _string(item, f"{field}[{index}]", code)


def _rate(value: Any, field: str, code: str) -> None:
    rate = _number(value, field, code)
    if not 0.0 <= rate <= 1.0:
        _fail("palette_rate_invalid", f"{field} must be in [0, 1]")


def validate_palette(doc: Any) -> None:
    """Validate a strict district palette, including all three window modes."""

    palette = _mapping(doc, "palette_invalid")
    required = ("palette_id", "rule_kit_id", "shells", "access", "attachments", "extensions")
    _required(palette, required, "palette_invalid_missing_field")
    allowed = set(required)
    unknown = sorted(set(palette) - allowed)
    if unknown:
        _fail("palette_unknown_key", "unknown top-level key(s): " + ", ".join(unknown))
    for field in ("palette_id", "rule_kit_id"):
        _string(palette[field], f"palette.{field}", "palette_invalid")
    shells = _mapping(palette["shells"], "palette_invalid")
    _required(shells, ("allowed_profile_ids", "weights", "size_weights", "observed_template_rate", "multi_shell_rate", "max_shells"), "palette_invalid_missing_field")
    shell_allowed = {"allowed_profile_ids", "weights", "size_weights", "observed_template_rate", "multi_shell_rate", "max_shells"}
    unknown = sorted(set(shells) - shell_allowed)
    if unknown:
        _fail("palette_unknown_key", "unknown shells key(s): " + ", ".join(unknown))
    _rate(shells["observed_template_rate"], "shells.observed_template_rate", "palette_invalid")
    _rate(shells["multi_shell_rate"], "shells.multi_shell_rate", "palette_invalid")
    _weights(shells["weights"], "shells.weights", "palette_invalid")
    _weights(shells["size_weights"], "shells.size_weights", "palette_invalid")
    _string_list(shells["allowed_profile_ids"], "shells.allowed_profile_ids", "palette_invalid")
    max_shells = _number(shells["max_shells"], "shells.max_shells", "palette_invalid")
    if max_shells < 1 or max_shells != int(max_shells):
        _fail("palette_invalid", "shells.max_shells must be an integer >= 1")
    access = _mapping(palette["access"], "palette_invalid")
    _required(access, ("primary_bundle_weights", "secondary_door_rate", "max_secondary_doors", "require_named_access_surface"), "palette_invalid_missing_field")
    access_allowed = {"primary_bundle_weights", "secondary_door_rate", "max_secondary_doors", "require_named_access_surface"}
    unknown = sorted(set(access) - access_allowed)
    if unknown:
        _fail("palette_unknown_key", "unknown access key(s): " + ", ".join(unknown))
    _weights(access["primary_bundle_weights"], "access.primary_bundle_weights", "palette_invalid")
    _rate(access["secondary_door_rate"], "access.secondary_door_rate", "palette_invalid")
    max_secondary = _number(access["max_secondary_doors"], "access.max_secondary_doors", "palette_invalid")
    if max_secondary < 0 or max_secondary != int(max_secondary):
        _fail("palette_invalid", "access.max_secondary_doors must be an integer >= 0")
    if not isinstance(access["require_named_access_surface"], bool):
        _fail("palette_invalid", "access.require_named_access_surface must be boolean")
    attachments = _mapping(palette["attachments"], "palette_invalid")
    _required(attachments, ("window_mode", "window_rates", "porch_rate", "dormer_rate", "tent_rate", "allowed_profile_ids"), "palette_invalid_missing_field")
    attachment_allowed = {"window_mode", "window_rates", "porch_rate", "dormer_rate", "tent_rate", "allowed_profile_ids"}
    unknown = sorted(set(attachments) - attachment_allowed)
    if unknown:
        _fail("palette_unknown_key", "unknown attachments key(s): " + ", ".join(unknown))
    if attachments["window_mode"] not in WINDOW_MODES:
        _fail("palette_invalid_window_mode", "attachments.window_mode is unsupported")
    _weights(attachments["window_rates"], "attachments.window_rates", "palette_invalid")
    for field in ("porch_rate", "dormer_rate", "tent_rate"):
        _rate(attachments[field], f"attachments.{field}", "palette_invalid")
    _string_list(attachments["allowed_profile_ids"], "attachments.allowed_profile_ids", "palette_invalid")
    extensions = _mapping(palette["extensions"], "palette_invalid")
    _required(extensions, ("allowed_kinds", "max_revisions"), "palette_invalid_missing_field")
    extension_allowed = {"allowed_kinds", "max_revisions"}
    unknown = sorted(set(extensions) - extension_allowed)
    if unknown:
        _fail("palette_unknown_key", "unknown extensions key(s): " + ", ".join(unknown))
    _string_list(extensions["allowed_kinds"], "extensions.allowed_kinds", "palette_invalid")
    max_revisions = _number(extensions["max_revisions"], "extensions.max_revisions", "palette_invalid")
    if max_revisions < 0 or max_revisions != int(max_revisions):
        _fail("palette_invalid", "extensions.max_revisions must be an integer >= 0")


def _validate_request_common(doc: Any, code: str) -> Mapping[str, Any]:
    request = _mapping(doc, code)
    _required(
        request,
        (
            "settlement_id", "district_id", "parcel_id", "request_id", "master_seed",
            "revision", "palette_id", "requested_size", "use_tags", "lot_polygon",
            "setback_polygon", "frontage_segment", "primary_access_surface",
            "secondary_access_surfaces", "occupied_reserved_polygons", "terrain_context",
            "terrain_edit_policy", "runtime_caps",
        ),
        code,
    )
    for field in ("settlement_id", "district_id", "parcel_id", "request_id", "palette_id"):
        _string(request[field], f"request.{field}", code)
    _number(request["master_seed"], "request.master_seed", code)
    revision = _number(request["revision"], "request.revision", code)
    if revision < 0 or revision != int(revision):
        _fail(code, "request.revision must be an integer >= 0")
    _string(request["requested_size"], "request.requested_size", code)
    if not isinstance(request["use_tags"], list):
        _fail(code, "request.use_tags must be an array")
    for field in ("lot_polygon", "setback_polygon"):
        if not isinstance(request[field], list) or len(request[field]) < 3:
            _fail(code, f"request.{field} must contain at least three points")
        for point in request[field]:
            _triplet([point[0], point[1], 0.0], f"request.{field} point", code) if isinstance(point, Sequence) and len(point) == 2 else _triplet(point, f"request.{field} point", code)
    if not isinstance(request["frontage_segment"], Mapping) or not isinstance(request["primary_access_surface"], Mapping):
        _fail(code, "frontage and primary access must be objects")
    if not isinstance(request["secondary_access_surfaces"], list) or not isinstance(request["occupied_reserved_polygons"], list):
        _fail(code, "secondary access and occupied polygons must be arrays")
    if not isinstance(request["terrain_context"], Mapping) or not isinstance(request["terrain_edit_policy"], Mapping) or not isinstance(request["runtime_caps"], Mapping):
        _fail(code, "terrain and runtime contexts must be objects")
    return request


def validate_building_request(doc: Any) -> None:
    """Validate the complete base-building request contract."""

    _validate_request_common(doc, "building_request_invalid")


def validate_building_extension_request(doc: Any) -> None:
    """Validate a lot-directed extension request with complete prior output."""

    request = _mapping(doc, "building_extension_request_invalid")
    _required(
        request,
        (
            "settlement_id", "district_id", "parcel_id", "request_id", "master_seed",
            "revision", "palette_id", "previous_generated_building", "world_placement",
            "free_space_polygons", "new_access_surfaces", "occupied_reserved_polygons",
            "terrain_context", "terrain_edit_policy", "allowed_extension_kinds", "palette_caps",
        ),
        "building_extension_request_invalid",
    )
    for field in ("settlement_id", "district_id", "parcel_id", "request_id", "palette_id"):
        _string(request[field], f"extension.{field}", "building_extension_request_invalid")
    _number(request["master_seed"], "extension.master_seed", "building_extension_request_invalid")
    revision = _number(request["revision"], "extension.revision", "building_extension_request_invalid")
    if revision < 1 or revision != int(revision):
        _fail("building_extension_request_invalid", "extension.revision must be an integer >= 1")
    validate_generated_building(request["previous_generated_building"])
    for field in ("world_placement", "terrain_context", "terrain_edit_policy", "palette_caps"):
        if not isinstance(request[field], Mapping):
            _fail("building_extension_request_invalid", f"extension.{field} must be an object")
    for field in ("free_space_polygons", "new_access_surfaces", "occupied_reserved_polygons", "allowed_extension_kinds"):
        if not isinstance(request[field], list):
            _fail("building_extension_request_invalid", f"extension.{field} must be an array")


def validate_generated_building(doc: Any) -> None:
    """Validate and pass the complete output through the real D-STAMP consumer."""

    building = _mapping(doc, "generated_building_invalid")
    _required(
        building,
        (
            "stamp_id", "revision_id", "members", "source", "access_heading_rad",
            "footprint", "bounds_rel_gu", "terrain_envelope", "building_type",
            "size_class", "multi_shell", "door_count",
        ),
        "generated_building_invalid",
    )
    _string(building["stamp_id"], "building.stamp_id", "generated_building_invalid")
    _string(building["revision_id"], "building.revision_id", "generated_building_invalid")
    if not isinstance(building["members"], list) or not building["members"]:
        _fail("generated_building_invalid", "building.members must be non-empty")
    source = _mapping(building["source"], "generated_building_invalid")
    if not isinstance(source.get("seed_door"), str) or not source["seed_door"]:
        _fail("generated_building_invalid", "source.seed_door must name a door")
    _number(building["access_heading_rad"], "building.access_heading_rad", "generated_building_invalid")
    footprint = _mapping(building["footprint"], "generated_building_invalid")
    _required(footprint, ("hull_xy_rel", "aabb_rel", "components_xy_rel"), "generated_building_invalid")
    if not isinstance(footprint["hull_xy_rel"], list) or not isinstance(footprint["components_xy_rel"], list):
        _fail("generated_building_invalid", "footprint polygons must be arrays")
    for bounds_field in (footprint["aabb_rel"], building["bounds_rel_gu"]):
        bounds = bounds_field if isinstance(bounds_field, Mapping) else None
        if bounds is None:
            _fail("generated_building_invalid", "bounds must be objects")
        _required(bounds, ("min", "max"), "generated_building_invalid")
        _triplet(bounds["min"], "building bounds.min", "generated_building_invalid")
        _triplet(bounds["max"], "building bounds.max", "generated_building_invalid")
    envelope = _mapping(building["terrain_envelope"], "generated_building_invalid")
    _required(envelope, ("door_step_heights_gu",), "generated_building_invalid")
    if not isinstance(envelope["door_step_heights_gu"], list):
        _fail("generated_building_invalid", "terrain_envelope.door_step_heights_gu must be an array")
    if not isinstance(building["multi_shell"], bool):
        _fail("generated_building_invalid", "multi_shell must be boolean")
    door_count = _number(building["door_count"], "building.door_count", "generated_building_invalid")
    if door_count < 0 or door_count != int(door_count):
        _fail("generated_building_invalid", "door_count must be a non-negative integer")
    for member in building["members"]:
        if not isinstance(member, Mapping):
            _fail("generated_building_invalid", "every member must be an object")
        _required(member, ("source_id", "object_id", "record_type", "model_key", "offset_gu", "rotation", "scale", "structural_role"), "generated_building_invalid")
        _string(member["source_id"], "building member.source_id", "generated_building_invalid")
        _string(member["object_id"], "building member.object_id", "generated_building_invalid")
        _string(member["model_key"], "building member.model_key", "generated_building_invalid")
        if member["record_type"] not in RECORD_TYPES:
            _fail("generated_building_invalid", "member.record_type unsupported")
        _triplet(member["offset_gu"], "building member.offset_gu", "generated_building_invalid")
        rotation = _triplet(member["rotation"], "building member.rotation", "generated_building_invalid")
        if any(abs(value) > 2 * math.pi for value in rotation):
            _fail("generated_building_invalid", "member.rotation is not radians")
        if _number(member["scale"], "building member.scale", "generated_building_invalid") <= 0:
            _fail("generated_building_invalid", "member.scale must be positive")
        _string(member["structural_role"], "building member.structural_role", "generated_building_invalid")
        if bool(member.get("is_door")):
            if member["record_type"] != "DOOR":
                _fail("generated_building_invalid", "door member must have record_type DOOR")
            if "outward_heading_deg" not in member:
                _fail("generated_building_invalid", "door member has no outward heading")
            _number(member["outward_heading_deg"], "building member.outward_heading_deg", "generated_building_invalid")
            _validate_door_block(member.get("door"), "generated_building_invalid")
    doors = [member for member in building["members"] if bool(member.get("is_door"))]
    if int(door_count) != len(doors):
        _fail("generated_building_invalid", "door_count does not match emitted door members")
    if len(envelope["door_step_heights_gu"]) != len(doors):
        _fail("generated_building_invalid", "door step count does not match emitted doors")
    if source["seed_door"] not in {member["source_id"] for member in doors}:
        _fail("generated_building_invalid", "source.seed_door is not an emitted door")

    # This is intentionally the real current consumer, not a duplicate local
    # D-STAMP check.  CityPlaceError is allowed to propagate to callers.
    from ..cityplace import validate_stamp_integrity

    validate_stamp_integrity(building)
