"""Pure Phase 1 normalization and witnessed-relation rebuilding.

The module consumes parsed source-library/grammar documents and returns JSON-
ready derived products.  It makes no filesystem or Blender calls; the CLI owns
I/O and config selection.  Failed gates become explicit ineligible/rejection
rows rather than silently repaired evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from ..engine_transform import matrix_to_tes3_euler, tes3_euler_to_matrix
from .contracts import validate_connection_sample, validate_normalized_member, validate_source_member
from .normalize import canonicalize, matrix_max_error, rebase_connection, to_source_world, to_template_local, vector_max_error


STRUCTURAL_ROLES = {"shell", "wall", "piece", "wall_piece"}
ROLE_ORDER = {"shell": 0, "wall": 1, "piece": 2, "wall_piece": 2}


def _source_member(row: Mapping[str, Any]) -> dict[str, Any]:
    member = {
        "source_id": row["source_id"],
        "object_id": row["object_id"],
        "model_key": row["model_key"],
        "record_type": row["record_type"],
        "category": row["category"],
        "is_door": bool(row["is_door"]),
        "structural_role": row["structural_role"],
        "offset_gu": list(row["offset_gu"]),
        "rotation": list(row["rotation"]),
        "scale": float(row["scale"]),
        "outward_heading_deg": row.get("outward_heading_deg"),
    }
    if "door" in row:
        member["door"] = row["door"]
    validate_source_member(member)
    return member


def _edge_mentions(edge: Any, source_id: str) -> bool:
    if isinstance(edge, Mapping):
        return edge.get("ref_a") == source_id or edge.get("ref_b") == source_id
    return isinstance(edge, Sequence) and len(edge) == 2 and source_id in edge


def _witness_count(stamp: Mapping[str, Any], source_id: str) -> int:
    return sum(_edge_mentions(edge, source_id) for edge in stamp.get("touching_pairs", [])) + sum(
        _edge_mentions(edge, source_id) for edge in stamp.get("shell_attachment_edges", [])
    )


def _door_provenance(member: Mapping[str, Any]) -> dict[str, Any] | None:
    value = member.get("door")
    if isinstance(value, Mapping):
        return dict(value)
    return None


def choose_seed_door(stamp: Mapping[str, Any]) -> str | None:
    """Select the most witnessed door with a stable source-id tie break."""

    doors = [member for member in stamp.get("members", []) if bool(member.get("is_door"))]
    if not doors:
        return None
    return min(
        (str(member["source_id"]) for member in doors),
        key=lambda source_id: (-_witness_count(stamp, source_id), source_id),
    )


def normalize_stamp(
    stamp: Mapping[str, Any],
    *,
    site_id: str,
    position_tolerance_gu: float,
    rotation_tolerance: float,
) -> dict[str, Any]:
    """Normalize one complete observed source stamp around its seed door."""

    members = [_source_member(row) for row in stamp.get("members", [])]
    shell_count = sum(member["structural_role"] == "shell" for member in members)
    seed_door = choose_seed_door(stamp)
    base = {
        "stamp_id": stamp["stamp_id"],
        "seed_door": seed_door,
        "evidence_class": "observed_exact" if shell_count and seed_door else "ineligible",
        "members": [],
        "source_evidence": {
            "site_id": site_id,
            "source_stamp_id": stamp["stamp_id"],
            "frame": "source_world",
            "touching_pairs": stamp.get("touching_pairs", []),
            "shell_attachment_edges": stamp.get("shell_attachment_edges", []),
            "member_contact_edges": stamp.get("member_contact_edges", []),
        },
        "terrain_envelope": stamp.get("terrain_envelope", {}),
        "bounds_rel_gu": stamp.get("bounds_rel_gu", {}),
        "building_type": stamp.get("building_type"),
        "size_class": stamp.get("size_class"),
        "multi_shell": bool(stamp.get("multi_shell")),
        "door_count": int(sum(member["is_door"] for member in members)),
    }
    if not shell_count:
        base["rejection"] = "no_shell_members"
        return base
    if seed_door is None:
        base["rejection"] = "no_seed_door"
        return base
    by_id = {member["source_id"]: member for member in members}
    seed = by_id[seed_door]
    p0 = seed["offset_gu"]
    R0 = tes3_euler_to_matrix(seed["rotation"])
    normalized: list[dict[str, Any]] = []
    for member in members:
        local_offset, local_rotation = to_template_local(member["offset_gu"], member["rotation"], p0, R0)
        row = {
            "source_id": member["source_id"],
            "object_id": member["object_id"],
            "model_key": member["model_key"],
            "record_type": member["record_type"],
            "structural_role": member["structural_role"],
            "offset_local_gu": local_offset,
            "rotation_local_rad": local_rotation,
            "scale": member["scale"],
            "is_door": member["is_door"],
        }
        if member.get("outward_heading_deg") is not None:
            row["outward_heading_deg"] = member["outward_heading_deg"]
        provenance = _door_provenance(member)
        if provenance is not None:
            row["door"] = provenance
        validate_normalized_member(row)
        normalized.append(row)
    base["members"] = sorted(normalized, key=lambda row: str(row["source_id"]))
    base = canonicalize(base)
    # The seed assertion is against the emitted six-place values, not the
    # unpersisted intermediate arrays.
    emitted_seed = next(row for row in base["members"] if row["source_id"] == seed_door)
    seed_position, seed_rotation = to_source_world(
        emitted_seed["offset_local_gu"], emitted_seed["rotation_local_rad"], p0, R0
    )
    position_residual = vector_max_error(seed_position, p0)
    rotation_residual = matrix_max_error(tes3_euler_to_matrix(seed_rotation), R0)
    report = {
        "max_position_residual_gu": position_residual,
        "max_rotation_matrix_residual": rotation_residual,
        "source_member_count": len(members),
    }
    base["roundtrip"] = report
    failures = []
    for original in members:
        emitted = next(row for row in base["members"] if row["source_id"] == original["source_id"])
        position, rotation = to_source_world(emitted["offset_local_gu"], emitted["rotation_local_rad"], p0, R0)
        p_error = vector_max_error(position, original["offset_gu"])
        r_error = matrix_max_error(tes3_euler_to_matrix(rotation), tes3_euler_to_matrix(original["rotation"]))
        scale_equal = emitted["scale"] == round(float(original["scale"]), 6)
        if p_error > position_tolerance_gu or r_error > rotation_tolerance or not scale_equal:
            failures.append({"source_id": original["source_id"], "position_residual_gu": p_error, "rotation_matrix_residual": r_error, "scale_equal": scale_equal})
    if failures:
        base["evidence_class"] = "ineligible"
        base["rejection"] = "source_roundtrip_failed"
        base["roundtrip"]["failures"] = failures
    return base


def _role_key(member: Mapping[str, Any]) -> tuple[int, str]:
    role = str(member.get("structural_role", ""))
    return (ROLE_ORDER.get(role, 3), str(member["source_id"]))


def _witness_pairs(stamp: Mapping[str, Any]) -> list[tuple[str, str, float | None, str | None]]:
    rows: dict[tuple[str, str, str | None], tuple[str, str, float | None, str | None]] = {}
    for pair in stamp.get("touching_pairs", []):
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            continue
        left, right = str(pair[0]), str(pair[1])
        key = (min(left, right), max(left, right), None)
        rows.setdefault(key, (left, right, None, None))
    for edge in stamp.get("shell_attachment_edges", []):
        if not isinstance(edge, Mapping):
            continue
        if "ref_a" not in edge or "ref_b" not in edge:
            continue
        left, right = str(edge["ref_a"]), str(edge["ref_b"])
        distance = edge.get("minimum_distance_gu")
        distance_value = None if distance is None else float(distance)
        edge_id = f"{left}->{right}"
        key = (min(left, right), max(left, right), edge_id)
        rows.setdefault(key, (left, right, distance_value, edge_id))
    for edge in stamp.get("member_contact_edges", []):
        if not isinstance(edge, Mapping):
            continue
        if "ref_a" not in edge or "ref_b" not in edge:
            continue
        left, right = str(edge["ref_a"]), str(edge["ref_b"])
        distance = edge.get("minimum_distance_gu")
        distance_value = None if distance is None else float(distance)
        edge_id = f"{left}->{right}"
        key = (min(left, right), max(left, right), edge_id)
        rows.setdefault(key, (left, right, distance_value, edge_id))
    return sorted(rows.values(), key=lambda row: (min(row[0], row[1]), max(row[0], row[1]), row[3] or ""))


def rebuild_connections(
    stamp: Mapping[str, Any],
    *,
    site_id: str,
    contact_interval_half_width_gu: float = 0.0,
) -> dict[str, Any]:
    """Rebuild witnessed ordered shell/piece samples and attachment contacts."""

    members = {_member["source_id"]: _source_member(_member) for _member in stamp.get("members", [])}
    rules: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    attachments: list[dict[str, Any]] = []
    for ref_left, ref_right, distance, direct_id in _witness_pairs(stamp):
        if ref_left not in members or ref_right not in members:
            continue
        left = members[ref_left]
        right = members[ref_right]
        ordered = sorted((left, right), key=_role_key)
        a, b = ordered
        relation = rebase_connection(a["offset_gu"], a["rotation"], b["offset_gu"], b["rotation"])
        is_structural = a["structural_role"] in STRUCTURAL_ROLES and b["structural_role"] in STRUCTURAL_ROLES
        sample_id = f"{site_id}__{stamp['stamp_id']}__{a['source_id']}__{b['source_id']}"
        interval = [None, None] if distance is None else [distance - contact_interval_half_width_gu, distance + contact_interval_half_width_gu]
        sample = canonicalize({
            "sample_id": sample_id,
            "model_a": str(a["model_key"]).lower().replace("/", "\\"),
            "model_b": str(b["model_key"]).lower().replace("/", "\\"),
            "authored_scale_a": a["scale"],
            "authored_scale_b": b["scale"],
            **relation,
            "source_rotation_a_rad": a["rotation"],
            "source_rotation_b_rad": b["rotation"],
            "contact_distance_gu": distance,
            "allowed_contact_interval_gu": interval,
            "witness": {
                "site_id": site_id,
                "source_stamp_id": stamp["stamp_id"],
                "ref_a": a["source_id"],
                "ref_b": b["source_id"],
                "direct_contact_id": direct_id,
            },
            "evidence_class": "observed_exact",
        })
        validate_connection_sample(sample)
        if is_structural:
            rules[(sample["model_a"], sample["model_b"])].append(sample)
        else:
            attachments.append(sample)
    output_rules = []
    for (model_a, model_b), samples in sorted(rules.items()):
        samples = sorted(samples, key=lambda row: row["sample_id"])
        output_rules.append({
            "model_a": model_a,
            "model_b": model_b,
            "samples": samples,
            "occurrence_count": len(samples),
            "independent_stamp_count": len({row["witness"]["source_stamp_id"] for row in samples}),
            "independent_site_count": len({row["witness"]["site_id"] for row in samples}),
        })
    return canonicalize({
        "schema_version": 1,
        "site_id": site_id,
        "frame": "ordered_a_local",
        "rules": output_rules,
        "attachment_contacts": sorted(attachments, key=lambda row: row["sample_id"]),
    })


def build_connection_document(
    source_library: Mapping[str, Any],
    *,
    site_id: str,
    contact_interval_half_width_gu: float = 0.0,
) -> dict[str, Any]:
    """Aggregate ordered witnessed relations from every stamp in one site."""

    all_samples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    attachments: list[dict[str, Any]] = []
    for stamp in sorted(source_library.get("stamps", []), key=lambda row: str(row.get("stamp_id"))):
        document = rebuild_connections(
            stamp,
            site_id=site_id,
            contact_interval_half_width_gu=contact_interval_half_width_gu,
        )
        for rule in document["rules"]:
            all_samples[(rule["model_a"], rule["model_b"])].extend(rule["samples"])
        attachments.extend(document["attachment_contacts"])
    rules = []
    for (model_a, model_b), samples in sorted(all_samples.items()):
        ordered = sorted(samples, key=lambda row: row["sample_id"])
        rules.append({
            "model_a": model_a,
            "model_b": model_b,
            "samples": ordered,
            "occurrence_count": len(ordered),
            "independent_stamp_count": len({row["witness"]["source_stamp_id"] for row in ordered}),
            "independent_site_count": len({row["witness"]["site_id"] for row in ordered}),
        })
    return canonicalize({
        "schema_version": 1,
        "site_id": site_id,
        "frame": "ordered_a_local",
        "rules": rules,
        "attachment_contacts": sorted(attachments, key=lambda row: row["sample_id"]),
    })


def build_template_document(
    source_library: Mapping[str, Any],
    *,
    site_id: str,
    position_tolerance_gu: float,
    rotation_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize every stamp and return the document plus per-site statistics."""

    templates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for stamp in sorted(source_library.get("stamps", []), key=lambda row: str(row.get("stamp_id"))):
        template = normalize_stamp(
            stamp,
            site_id=site_id,
            position_tolerance_gu=position_tolerance_gu,
            rotation_tolerance=rotation_tolerance,
        )
        templates.append(template)
        if template.get("evidence_class") == "ineligible":
            rejected.append({"stamp_id": template["stamp_id"], "reason": template.get("rejection")})
    document = canonicalize({
        "schema_version": 1,
        "frame": "template_local",
        "site_id": site_id,
        "templates": templates,
        "rejections": rejected,
    })
    stats = {
        "templates_processed": len(templates),
        "templates_eligible": sum(row.get("evidence_class") == "observed_exact" for row in templates),
        "templates_rejected": len(rejected),
    }
    return document, stats


def roundtrip_connection_sample(
    sample: Mapping[str, Any],
    members: Mapping[str, Mapping[str, Any]],
    *,
    position_tolerance_gu: float = 0.25,
    rotation_tolerance: float = 1.0e-6,
) -> dict[str, float | bool]:
    """Reconstruct B from A and a stored sample for the hard relation gate."""

    witness = sample["witness"]
    a = members[witness["ref_a"]]
    b = members[witness["ref_b"]]
    relation = rebase_connection(a["offset_gu"], a["rotation"], b["offset_gu"], b["rotation"])
    R_a = tes3_euler_to_matrix(a["rotation"])
    reconstructed_position = np.asarray(a["offset_gu"], dtype=np.float64) + R_a @ np.asarray(sample["offset_b_in_a_frame_gu"], dtype=np.float64)
    reconstructed_rotation = R_a @ np.asarray(sample["relative_engine_matrix_3x3"], dtype=np.float64)
    position_error = vector_max_error(reconstructed_position.tolist(), b["offset_gu"])
    matrix_error = matrix_max_error(reconstructed_rotation, tes3_euler_to_matrix(b["rotation"]))
    stored_offset_error = vector_max_error(relation["offset_b_in_a_frame_gu"], sample["offset_b_in_a_frame_gu"])
    stored_matrix_error = matrix_max_error(relation["relative_engine_matrix_3x3"], sample["relative_engine_matrix_3x3"])
    return {
        "position_residual_gu": position_error,
        "rotation_matrix_residual": matrix_error,
        "stored_offset_residual_gu": stored_offset_error,
        "stored_matrix_residual": stored_matrix_error,
        "passed": position_error <= position_tolerance_gu and matrix_error <= rotation_tolerance,
    }
