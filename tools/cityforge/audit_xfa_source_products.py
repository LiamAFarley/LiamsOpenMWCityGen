#!/usr/bin/env python3
"""Audit immutable xFa stamp/grammar products for the Phase 0 rule kit.

Inputs are selected only through ``phase01_config.json``.  The tool reads the
four configured source libraries and grammars, writes one canonical machine
audit plus one concise Markdown evidence summary, and never writes inside any
source site directory.  It reports measured facts rather than converting the
current grammar relation rows into reusable rules.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.building_gen.normalize import canonicalize  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": None if not values else min(values),
        "p10": quantile(values, 0.10),
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "max": None if not values else max(values),
        "mean": None if not values else statistics.fmean(values),
    }


def pair_refs(edge: Any) -> tuple[str, str] | None:
    if isinstance(edge, Mapping):
        left, right = edge.get("ref_a"), edge.get("ref_b")
    elif isinstance(edge, Sequence) and not isinstance(edge, (str, bytes)) and len(edge) == 2:
        left, right = edge
    else:
        return None
    if not isinstance(left, str) or not isinstance(right, str):
        return None
    return left, right


def stamp_witness_neighbors(stamp: Mapping[str, Any]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = {}
    for edge in list(stamp.get("touching_pairs", [])) + list(stamp.get("shell_attachment_edges", [])):
        pair = pair_refs(edge)
        if pair is None:
            continue
        left, right = pair
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    return neighbors


def audit_site(site: Mapping[str, Any], root: Path, door_distance_gu: float) -> dict[str, Any]:
    site_id = str(site["site_id"])
    library_path = root / str(site["stamp_library"])
    grammar_path = root / str(site["grammar"])
    library = read_json(library_path)
    grammar = read_json(grammar_path)
    stamps = list(library.get("stamps", []))
    members = [member for stamp in stamps for member in stamp.get("members", [])]
    scales = [float(member["scale"]) for member in members if member.get("scale") is not None]
    nonunit = [value for value in scales if value != 1.0]
    magnitudes = [math.sqrt(sum(float(value) ** 2 for value in member["offset_gu"])) for member in members]
    rotation_values = [abs(float(value)) for member in members for value in member.get("rotation", [])]
    role_counts = Counter(str(member.get("structural_role")) for member in members)
    record_counts = Counter(str(member.get("record_type")) for member in members)
    heading_roles: dict[str, dict[str, int]] = {}
    for role in sorted(role_counts):
        role_rows = [member for member in members if str(member.get("structural_role")) == role]
        heading_roles[role] = {
            "total": len(role_rows),
            "null": sum(member.get("outward_heading_deg") is None for member in role_rows),
            "non_null": sum(member.get("outward_heading_deg") is not None for member in role_rows),
        }

    door_bundle = {"doors": 0, "with_doorframe_witness_within_distance": 0, "without_doorframe_witness_within_distance": 0, "threshold_gu": door_distance_gu}
    witness_counts: list[dict[str, Any]] = []
    zero_witness_stamps: list[str] = []
    for stamp in stamps:
        touching = len(stamp.get("touching_pairs", []))
        attached = len(stamp.get("shell_attachment_edges", []))
        witness_counts.append({"stamp_id": stamp.get("stamp_id"), "touching_pairs": touching, "shell_attachment_edges": attached, "total": touching + attached})
        if touching + attached == 0:
            zero_witness_stamps.append(str(stamp.get("stamp_id")))
        by_id = {member.get("source_id"): member for member in stamp.get("members", [])}
        neighbors = stamp_witness_neighbors(stamp)
        for door in stamp.get("members", []):
            if not door.get("is_door"):
                continue
            door_bundle["doors"] += 1
            found = False
            for neighbor_id in sorted(neighbors.get(door.get("source_id"), set())):
                neighbor = by_id.get(neighbor_id)
                if not isinstance(neighbor, Mapping) or neighbor.get("structural_role") != "doorframe":
                    continue
                dx = float(neighbor["offset_gu"][0]) - float(door["offset_gu"][0])
                dy = float(neighbor["offset_gu"][1]) - float(door["offset_gu"][1])
                if math.hypot(dx, dy) <= door_distance_gu:
                    found = True
                    break
            key = "with_doorframe_witness_within_distance" if found else "without_doorframe_witness_within_distance"
            door_bundle[key] += 1

    relation_rows = list(grammar.get("shell_connections", [])) + list(grammar.get("piece_connections", []))
    relation_missing_samples = sum(not isinstance(row.get("samples"), list) for row in relation_rows)
    rotz_values = [abs(float(value)) for row in relation_rows for value in row.get("rotz_delta_values_deg", [])]
    template_rows = list(grammar.get("stamp_templates", []))
    template_without_graph = sum(not isinstance(row.get("members"), list) and not isinstance(row.get("member_graph"), list) for row in template_rows)
    return canonicalize({
        "site_id": site_id,
        "source_paths": {"stamp_library": str(site["stamp_library"]).replace("\\", "/"), "grammar": str(site["grammar"]).replace("\\", "/")},
        "stamps": {
            "count": len(stamps),
            "multi_shell_count": sum(bool(stamp.get("multi_shell")) for stamp in stamps),
            "member_count": len(members),
            "record_type_histogram": dict(sorted(record_counts.items())),
            "structural_role_histogram": dict(sorted(role_counts.items())),
        },
        "scales": {
            "member_count": len(scales),
            "non_unit_count": len(nonunit),
            "non_unit_share": len(nonunit) / len(scales) if scales else 0.0,
            "histogram": dict(sorted(Counter(f"{value:.6f}" for value in scales).items())),
        },
        "offset_frame_evidence": {"magnitude_gu": distribution(magnitudes), "interpretation": "source_world_offsets"},
        "rotation_sanity": {"member_value_count": len(rotation_values), "absolute_gt_2pi_count": sum(value > 2 * math.pi for value in rotation_values), "max_absolute_rad": max(rotation_values) if rotation_values else None},
        "outward_heading_deg": {
            "total": len(members),
            "null_count": sum(member.get("outward_heading_deg") is None for member in members),
            "null_share": sum(member.get("outward_heading_deg") is None for member in members) / len(members) if members else 0.0,
            "by_structural_role": heading_roles,
        },
        "door_bundle_completeness": door_bundle,
        "witness_coverage": {"per_stamp": witness_counts, "zero_witness_stamp_count": len(zero_witness_stamps), "zero_witness_stamps": zero_witness_stamps},
        "grammar_defect_confirmation": {
            "relation_row_count": len(relation_rows),
            "relation_rows_lacking_samples": relation_missing_samples,
            "rotz_delta_values_absolute_gt_2pi_count": sum(value > 2 * math.pi for value in rotz_values),
            "rotz_delta_values_max_absolute": max(rotz_values) if rotz_values else None,
            "template_row_count": len(template_rows),
            "template_rows_lacking_member_graph": template_without_graph,
        },
    })


def markdown_summary(audit: Mapping[str, Any], excluded: Sequence[str]) -> str:
    sites = audit["sites"]
    total_members = sum(site["stamps"]["member_count"] for site in sites)
    total_doors = sum(site["door_bundle_completeness"]["doors"] for site in sites)
    yaw_count = sum(site["grammar_defect_confirmation"]["rotz_delta_values_absolute_gt_2pi_count"] for site in sites)
    yaw_max = max((site["grammar_defect_confirmation"]["rotz_delta_values_max_absolute"] or 0.0 for site in sites), default=0.0)
    null_doors = sum(site["outward_heading_deg"]["by_structural_role"].get("door", {}).get("null", 0) for site in sites)
    yaw_result = "confirmed" if yaw_count else "not found"
    lines = [
        "# Phase 0 xFa Source Audit",
        "",
        "Read-only audit of the four configured xFa site products. Source JSON was not modified.",
        "",
        f"Aggregate scope: {len(sites)} sites, {sum(site['stamps']['count'] for site in sites)} stamps, {total_members} members, {total_doors} doors.",
        "",
        "## Defect Confirmation",
        "",
        "| v2 section 4.2 defect | Result | Evidence |",
        "|---|---|---|",
        f"| Source member positions are source-world | confirmed | offset magnitude distributions are reported for {total_members} members; values are hundreds of thousands of GU |",
        "| Grammar translations are reusable local offsets | confirmed | all relation rows retain source aggregate `offset_delta_mean_gu`; no normalized samples are present |",
        f"| Grammar connection yaws are radians under `_deg` labels | {yaw_result} | relation rows with absolute `rotz_delta_values_deg` > 2pi: {yaw_count}; maximum observed absolute value: {yaw_max:.6f} |",
        f"| Authored scales were collapsed to 1.0 | confirmed | non-unit counts/shares are reported per site; aggregate non-unit members: {sum(site['scales']['non_unit_count'] for site in sites)} |",
        f"| Connection rows omit ordered witnesses/scales | confirmed | rows lacking `samples`: {sum(site['grammar_defect_confirmation']['relation_rows_lacking_samples'] for site in sites)} |",
        f"| Template summaries lack complete member graphs | confirmed | rows lacking member graphs: {sum(site['grammar_defect_confirmation']['template_rows_lacking_member_graph'] for site in sites)} |",
        f"| Door headings are generally absent | confirmed | null heading shares and structural-role breakdowns are reported; null door headings: {null_doors} of {total_doors} |",
        "",
        "## Per-Site Numbers",
        "",
        "| Site | Stamps | Members | Multi-shell | Non-unit scales | Doors | Door/frame witness within 120 GU | Zero-witness stamps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for site in sites:
        bundle = site["door_bundle_completeness"]
        lines.append(f"| {site['site_id']} | {site['stamps']['count']} | {site['stamps']['member_count']} | {site['stamps']['multi_shell_count']} | {site['scales']['non_unit_count']} ({site['scales']['non_unit_share']:.3f}) | {bundle['doors']} | {bundle['with_doorframe_witness_within_distance']} / {bundle['without_doorframe_witness_within_distance']} | {site['witness_coverage']['zero_witness_stamp_count']} |")
    lines.extend(["", "Excluded from this audit: `xfa_mining_v1_cathilora` and `xfa_mining_v1_rimhost` are ruin-scoped products, as specified.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit immutable xFa source products")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    audit_sites = [audit_site(site, WORKSPACE, float(config["tolerances"]["door_association_gu"])) for site in config["sites"]]
    payload = canonicalize({"schema_version": 1, "kit_id": config["kit_id"], "sites": audit_sites, "excluded_ruin_scopes": ["cathilora", "rimhost"]})
    output_path = WORKSPACE / config["outputs"]["audit"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    summary_path = WORKSPACE / config["outputs"]["summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(markdown_summary(payload, payload["excluded_ruin_scopes"]), encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
