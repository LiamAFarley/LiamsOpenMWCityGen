#!/usr/bin/env python3
"""Build Phase 3a or Phase 3b wall/mount profiles.

Pipeline position: Phase 3 driver (specs:
``.opencode/runs/2026-08-21-building-generation-rule-kit/2026-08-22_phase3a_implementation_spec.md``
and ``2026-08-22_phase3b_implementation_spec.md``). It exports evaluated
triangle evidence in one Blender pass, reconstructs facade profiles for shells,
mount profiles for attachments, witness occupancy, and observed access bundles.
The existing config selects the Phase 3a small set; the Phase 3b config selects
models by unambiguous inventory roles and writes separate products.

Usage::

    python tools/cityforge/build_wall_mount_profiles.py --config configs/kits/xfa_sky_nord_house/phase03_config.json
    python tools/cityforge/build_wall_mount_profiles.py --config configs/kits/xfa_sky_nord_house/phase03b_config.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen import engine_transform  # noqa: E402
from procgen.building_gen.access import build_access_bundles  # noqa: E402
from procgen.building_gen.facades import build_facade_profile, record_witness_occupancy  # noqa: E402
from procgen.building_gen.mounts import build_mount_profile, measure_sink_intervals  # noqa: E402
from procgen.building_gen.normalize import canonicalize  # noqa: E402

EVIDENCE_SCRIPT = WORKSPACE / "tools" / "cityforge" / "blender_wall_mount_evidence.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonicalize(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def normalize_model_key(value: str) -> str:
    return str(value).replace("/", "\\").casefold()


def resolve_profile_selection(config: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    """Resolve the configured small or inventory-role model partition."""
    selection = config.get("selection")
    if selection is None:
        small = config["small_set"]
        return {
            "mode": "small_set",
            "facade_models": list(small["shells"]),
            "mount_models": list(small["attachments"]),
            "facade_roles": [],
            "mount_roles": [],
            "skipped_models": [],
        }

    if selection.get("mode") != "inventory_roles":
        raise ValueError(f"unsupported Phase 3 selection mode: {selection.get('mode')!r}")
    facade_roles = {str(role).casefold() for role in selection["facade_roles"]}
    mount_roles = {str(role).casefold() for role in selection["mount_roles"]}
    overlap = sorted(facade_roles & mount_roles)
    if overlap:
        raise ValueError(f"facade/mount role sets overlap: {overlap}")

    facade_models: list[str] = []
    mount_models: list[str] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows = sorted(inventory["models"], key=lambda row: normalize_model_key(row["model_key"]))
    for row in rows:
        model_key = str(row["model_key"])
        normalized = normalize_model_key(model_key)
        if normalized in seen:
            skipped.append({"model_key": model_key, "reason": "duplicate_inventory_key"})
            continue
        seen.add(normalized)
        roles = [str(role).casefold() for role in row.get("observed_roles", [])]
        if not row.get("profile_eligible", False):
            reason = "profile_ineligible"
        elif len(roles) != 1:
            reason = "ambiguous_observed_roles"
        elif roles[0] in facade_roles:
            facade_models.append(model_key)
            continue
        elif roles[0] in mount_roles:
            mount_models.append(model_key)
            continue
        else:
            reason = "role_not_selected"
        skipped.append({
            "model_key": model_key,
            "observed_roles": roles,
            "reason": reason,
        })
    return {
        "mode": "inventory_roles",
        "facade_models": facade_models,
        "mount_models": mount_models,
        "facade_roles": sorted(facade_roles),
        "mount_roles": sorted(mount_roles),
        "skipped_models": skipped,
    }


def run_evidence(meshes: list[str], evidence_out: Path, blender: str) -> dict:
    blender_exe = shutil.which(blender)
    if blender_exe is None:
        print(f"FAILURE: blender not found on PATH ({blender!r})", file=sys.stderr)
        raise SystemExit(1)
    evidence_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wm_evidence_") as tmp:
        job_path = Path(tmp) / "job.json"
        job_path.write_text(json.dumps({"out": str(evidence_out), "meshes": meshes, "decimals": 4},
                                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [blender_exe, "-b", "--python", str(EVIDENCE_SCRIPT), "--", str(job_path)]
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=WORKSPACE, check=False)
        if not evidence_out.exists():
            print("FAILURE: Blender evidence export wrote no output", file=sys.stderr)
            raise SystemExit(1)
    return read_json(evidence_out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 3a or Phase 3b wall/mount profiles")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()

    config = read_json(args.config)
    inventory = read_json(WORKSPACE / config["phase02"]["inventory"])
    phase02_profiles = read_json(WORKSPACE / config["phase02"]["profiles"])
    bottom_by_key = {normalize_model_key(row["model_key"]): row["bottom_z_percentile_gu"]
                     for row in phase02_profiles["meshes"]}
    site_libraries = {str(s["site_id"]): read_json(WORKSPACE / s["stamp_library"]) for s in config["sites"]}
    measurement = config["measurement"]
    outputs = config["outputs"]
    axis_overrides = measurement.get("mount_normal_axis_overrides", {})
    if not isinstance(axis_overrides, dict):
        raise ValueError("measurement.mount_normal_axis_overrides must be an object")
    normalized_axis_overrides = {}
    for model_key, axis in axis_overrides.items():
        normalized_axis = str(axis).casefold()
        if normalized_axis not in {"x", "y"}:
            raise ValueError(
                f"mount normal axis override for {model_key!r} must be 'x' or 'y'"
            )
        normalized_axis_overrides[normalize_model_key(model_key)] = normalized_axis

    selection = resolve_profile_selection(config, inventory)
    shell_models = selection["facade_models"]
    attachment_models = selection["mount_models"]
    shell_keys = [normalize_model_key(k) for k in shell_models]
    attachment_keys = [normalize_model_key(k) for k in attachment_models]
    if len(set(shell_keys)) != len(shell_keys) or len(set(attachment_keys)) != len(attachment_keys):
        print("FAILURE: phase3 selection contains duplicate selected model keys", file=sys.stderr)
        return 1
    evidence_meshes = sorted({k.replace("\\", "/") for k in shell_models + attachment_models})
    scope = "phase3b" if selection["mode"] == "inventory_roles" else "phase3a"

    evidence_path = WORKSPACE / outputs.get("evidence", str(Path(outputs["root"]) / "wall_mount_evidence.json"))
    evidence = run_evidence(evidence_meshes, evidence_path, args.blender)
    if evidence["failures"]:
        print(f"FAILURE: evidence export failures: {evidence['failures']}", file=sys.stderr)
        return 1
    evidence_by_key = {normalize_model_key(row["model_key"]): row for row in evidence["models"]}
    missing = sorted(set(shell_keys + attachment_keys) - set(evidence_by_key))
    if missing:
        print(f"FAILURE: phase3 evidence missing selected models: {missing}", file=sys.stderr)
        return 1

    role_by_key = {}
    scales_by_key = {}
    for row in inventory["models"]:
        key = normalize_model_key(row["model_key"])
        if len(row["observed_roles"]) == 1:
            role_by_key[key] = row["observed_roles"][0]
        scales_by_key[key] = row["scales_observed"]

    facade_profiles: dict[str, dict] = {}
    for key in shell_keys:
        row = evidence_by_key[key]
        profile = build_facade_profile(
            row["model_key"],
            row["triangles"],
            {
                "max_wall_tilt_deg": measurement["max_wall_tilt_deg"],
                "wall_band_lo_hi": measurement["wall_band_fractions"]["default"],
                "azimuth_quantum_deg": measurement["azimuth_quantum_deg"],
                "plane_offset_tolerance_gu": measurement["plane_offset_tolerance_gu"],
                "vertex_weld_gu": measurement["vertex_weld_gu"],
                "merge_interval_iou": measurement["merge_interval_iou"],
                "merge_vertical_gap_gu": measurement["merge_vertical_gap_gu"],
                "merge_horizontal_gap_gu": measurement["merge_horizontal_gap_gu"],
                "facade_inset_gu": measurement["facade_inset_gu"],
                "min_facade_area_fraction": measurement["min_facade_area_fraction"],
                "max_wall_skin_gap_gu": measurement["max_wall_skin_gap_gu"],
                "max_facade_start_offset_gu": measurement.get("max_facade_start_offset_gu"),
            },
            row["bounds_local_gu"]["min"][2],
            row["bounds_local_gu"]["max"][2],
            band_base_z=bottom_by_key.get(key),
        )
        profile["resolved_path"] = row["resolved_path"]
        facade_profiles[key] = profile
        if profile["facade_count"] < 1:
            print(f"FAILURE: {scope} shell {key} produced zero facades", file=sys.stderr)
            return 1

    mount_profiles: dict[str, dict] = {}
    for key in attachment_keys:
        row = evidence_by_key[key]
        mount_profiles[key] = build_mount_profile(
            row["model_key"],
            role_by_key.get(key, "unresolved"),
            row["triangles"],
            row["bounds_local_gu"],
            scales_by_key.get(key, []),
            float(measurement["clearance_margin_gu"]),
            float(measurement["sink_tolerance_gu"]),
            normal_axis_override=normalized_axis_overrides.get(key),
        )
        mount_profiles[key]["resolved_path"] = row["resolved_path"]

    record_witness_occupancy(facade_profiles, site_libraries, inventory,
                             engine_transform.tes3_euler_to_matrix,
                             max_offset_gu=float(measurement["occupancy_max_offset_gu"]),
                             bounds_tolerance_gu=float(measurement["occupancy_bounds_tolerance_gu"]))
    measure_sink_intervals(mount_profiles, facade_profiles, site_libraries, inventory,
                           engine_transform.tes3_euler_to_matrix)
    access = build_access_bundles(site_libraries, mount_profiles,
                                  engine_transform.tes3_euler_to_matrix,
                                  float(measurement["door_frame_pair_distance_gu"]))

    selection_report = {
        "schema_version": 1,
        "scope": scope,
        "mode": selection["mode"],
        "facade_roles": selection["facade_roles"],
        "mount_roles": selection["mount_roles"],
        "selected_facade_models": sorted(shell_models, key=normalize_model_key),
        "selected_mount_models": sorted(attachment_models, key=normalize_model_key),
        "skipped_models": selection["skipped_models"],
        "selected_facade_count": len(shell_models),
        "selected_mount_count": len(attachment_models),
        "skipped_count": len(selection["skipped_models"]),
    }
    selection_report_path = outputs.get("selection_report")
    if selection_report_path:
        write_json(WORKSPACE / selection_report_path, selection_report)
    metadata = {
        "scope": scope,
        "selected_model_count": len(shell_models) + len(attachment_models),
        "selection_report": selection_report_path,
    }
    write_json(WORKSPACE / outputs["facades"], {
        "schema_version": 1,
        "origin": "evaluated_wall_band_triangles",
        "measurement": measurement,
        **metadata,
        "profiles": [facade_profiles[key] for key in shell_keys],
    })
    write_json(WORKSPACE / outputs["mounts"], {
        "schema_version": 1,
        "origin": "evaluated_attachment_geometry",
        "measurement": measurement,
        **metadata,
        "profiles": [mount_profiles[key] for key in attachment_keys],
    })
    access["scope"] = scope
    access["selected_model_count"] = len(shell_models) + len(attachment_models)
    access["selection_report"] = selection_report_path
    write_json(WORKSPACE / outputs["access"], access)

    summary = {
        "scope": scope,
        "selected_shell_count": len(shell_keys),
        "selected_mount_count": len(attachment_keys),
        "skipped_count": len(selection["skipped_models"]),
        "shells": {key: facade_profiles[key]["facade_count"] for key in shell_keys},
        "attachments": list(mount_profiles),
        "access_bundles": access["bundle_count"],
        "access_eligible": access["eligible_count"],
        "access_frameless": access["frameless_count"],
        "access_ambiguous": access["ambiguous_count"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
