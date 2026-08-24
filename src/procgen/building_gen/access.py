"""Door slots and access bundles from observed stamp members (Phase 3a).

Plan §7.6. Doors are first-class access contracts, not decoration. Every
bundle pairs a DOOR member with its doorframe from the same source stamp,
derives the outward heading from the frame's measured mount normal (never raw
door rotz), and records grade support from the stamp's terrain envelope.

Hard rule: ordinary eligible bundles always contain door plus frame. A door
without a frame is recorded as ``frameless_observation`` and is not eligible.
Ambiguous pairing (multiple frames at equal distance) is recorded ineligible.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(a, b)


def build_access_bundles(
    site_stamp_libraries: Mapping[str, Mapping[str, Any]],
    mount_profiles: Mapping[str, dict[str, Any]],
    engine_rotation,
    pair_distance_gu: float,
) -> dict[str, Any]:
    bundles: list[dict[str, Any]] = []
    for site_id in sorted(site_stamp_libraries):
        for stamp in site_stamp_libraries[site_id].get("stamps", []):
            members = stamp.get("members", [])
            doors = [m for m in members if m.get("is_door")]
            frames = [m for m in members if m.get("structural_role") == "doorframe"]
            if not doors:
                continue
            contact_pairs = set()
            for edge_key in ("shell_attachment_edges", "member_contact_edges", "internal_edges"):
                for edge in stamp.get(edge_key, []) or []:
                    if isinstance(edge, Mapping):
                        pair = (str(edge.get("ref_a")), str(edge.get("ref_b")))
                    else:
                        pair = (str(edge[0]), str(edge[1]))
                    contact_pairs.add(tuple(sorted(pair)))
            for door in doors:
                door_key = str(door["model_key"]).replace("/", "\\").casefold()
                candidates = []
                for frame in frames:
                    frame_key = str(frame["model_key"]).replace("/", "\\").casefold()
                    distance = _distance(door["offset_gu"], frame["offset_gu"])
                    direct = tuple(sorted((str(door["source_id"]), str(frame["source_id"])))) in contact_pairs
                    candidates.append({"frame": frame, "distance": distance, "direct": direct, "key": frame_key})
                candidates.sort(key=lambda c: (not c["direct"], c["distance"]))
                chosen = [c for c in candidates if c["direct"] or c["distance"] <= pair_distance_gu]
                base = {
                    "site_id": site_id,
                    "stamp_id": stamp["stamp_id"],
                    "door_ref": door["source_id"],
                }
                if not chosen:
                    bundles.append({
                        **base,
                        "evidence_class": "ineligible",
                        "rejection_reason": "frameless_observation",
                        "door_member": {
                            "model_key": door_key,
                            "record_type": "DOOR",
                            "scale": float(door["scale"]),
                        },
                    })
                    continue
                best = chosen[0]
                tied = [c for c in chosen if not c["direct"] and not best["direct"]
                        and abs(c["distance"] - best["distance"]) <= 1.0]
                if len(tied) > 1:
                    bundles.append({
                        **base,
                        "evidence_class": "ineligible",
                        "rejection_reason": "ambiguous_frame_pairing",
                        "candidate_frames": [c["frame"]["source_id"] for c in tied],
                    })
                    continue
                frame = best["frame"]
                frame_profile = mount_profiles.get(best["key"])
                if frame_profile is None:
                    heading = None
                    heading_note = "frame mount profile not measured in this run"
                else:
                    axis = "xyz".index(frame_profile["mount_frame"]["normal_axis"])
                    local_n = [0.0, 0.0, 0.0]
                    local_n[axis] = 1.0 if frame_profile["front_back_evidence"]["front_axis_sign"] == "+" else -1.0
                    rotated = engine_rotation(frame["rotation"]) @ np.asarray(local_n, dtype=np.float64)
                    heading = round(math.degrees(math.atan2(float(rotated[1]), float(rotated[0]))), 3)
                    heading_note = "frame mount normal rotated by frame TES3 rotation"
                bundles.append({
                    "access_bundle_id": f"{site_id}__{stamp['stamp_id']}__{door['source_id']}",
                    "slot_interface_id": f"facade_mount:{best['key']}",
                    "outward_heading_in_slot_deg": heading,
                    "heading_basis": heading_note,
                    "door_member": {
                        "model_key": door_key,
                        "record_type": "DOOR",
                        "scale": float(door["scale"]),
                    },
                    "frame_member": {
                        "model_key": best["key"],
                        "record_type": "STAT",
                        "scale": float(frame["scale"]),
                    },
                    "optional_grade_members": [],
                    "members_in_door_frame": [],
                    "grade_support": {
                        "door_step_heights_gu": stamp.get("terrain_envelope", {}).get("door_step_heights_gu", []),
                    },
                    "door_record_provenance": {
                        "object_id": door.get("object_id"),
                        "door_ref": door["source_id"],
                        "frame_ref": frame["source_id"],
                        "direct_contact": best["direct"],
                        "pair_distance_gu": round(best["distance"], 3),
                    },
                    "witness": {"site_id": site_id, "stamp_id": stamp["stamp_id"]},
                    "evidence_class": "observed_exact",
                })
    eligible = [b for b in bundles if b.get("evidence_class") == "observed_exact"]
    return {
        "schema_version": 1,
        "bundle_count": len(bundles),
        "eligible_count": len(eligible),
        "frameless_count": sum(1 for b in bundles if b.get("rejection_reason") == "frameless_observation"),
        "ambiguous_count": sum(1 for b in bundles if b.get("rejection_reason") == "ambiguous_frame_pairing"),
        "bundles": bundles,
    }
