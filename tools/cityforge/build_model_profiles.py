#!/usr/bin/env python3
"""Build the Phase 2 model inventory and native-scale profile products.

Pipeline position: host driver for Phase 2 of the xFa building rule kit
(spec: ``.opencode/runs/2026-08-21-building-generation-rule-kit/2026-08-22_phase2_implementation_spec.md``).
Builds the member/role inventory in pure Python
(``src/procgen/building_gen/inventory.py``), runs one Blender job
(``tools/cityforge/blender_model_profile.py``) to measure every observed model
at native scale, merges profiles into the inventory with explicit eligibility
decisions, and emits alias-equivalence evidence and the rejection list.

Usage::

    python tools/cityforge/build_model_profiles.py --config configs/kits/xfa_sky_nord_house/phase02_config.json
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

from procgen.building_gen.inventory import alias_families, build_inventory, canonical_model_key  # noqa: E402
from procgen.building_gen.normalize import canonicalize  # noqa: E402

BLENDER_SCRIPT = WORKSPACE / "tools" / "cityforge" / "blender_model_profile.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonicalize(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_blender_profile(meshes: list[str], measurement: dict, profiles_out: Path, blender: str) -> dict:
    blender_exe = shutil.which(blender)
    if blender_exe is None:
        print(f"FAILURE: blender not found on PATH ({blender!r})", file=sys.stderr)
        raise SystemExit(1)
    with tempfile.TemporaryDirectory(prefix="model_profile_") as tmp:
        job_path = Path(tmp) / "job.json"
        job = {
            "out": str(profiles_out),
            "meshes": meshes,
            "z_band_fractions": measurement["z_band_fractions"],
            "bottom_percentile": measurement["bottom_percentile"],
            "digest_decimals": measurement["digest_decimals"],
        }
        job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [blender_exe, "-b", "--python", str(BLENDER_SCRIPT), "--", str(job_path)]
        print("running:", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        if not profiles_out.exists():
            print("FAILURE: Blender profiling wrote no output", file=sys.stderr)
            raise SystemExit(1)
    return read_json(profiles_out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 2 model inventory and profiles")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()

    config = read_json(args.config)
    site_libraries = {}
    site_connections = {}
    for site in config["sites"]:
        site_id = str(site["site_id"])
        site_libraries[site_id] = read_json(WORKSPACE / site["stamp_library"])
        site_connections[site_id] = read_json(WORKSPACE / site["connections"])

    inventory = build_inventory(site_libraries, site_connections)
    models = inventory["models"]
    meshes = sorted({row["display_key"].replace("\\", "/") for row in models})
    print(f"[phase02] inventory: {len(models)} distinct models across {len(site_libraries)} sites", flush=True)

    outputs = config["outputs"]
    profiles_path = WORKSPACE / outputs["profiles"]
    profiles = run_blender_profile(meshes, config["measurement"], profiles_path, args.blender)
    profile_by_key = {canonical_model_key(row["model_key"]): row for row in profiles["meshes"]}
    failed_keys = {canonical_model_key(str(f).split(":", 1)[0]) for f in profiles["failures"]}

    rejections: list[dict[str, Any]] = []
    for row in models:
        key = row["model_key"]
        profile = profile_by_key.get(key)
        if profile is not None:
            row["resolved_path"] = profile.get("resolved_path")
            row["profile_eligible"] = True
            row["rejection_reason"] = None
        elif key in failed_keys:
            row["profile_eligible"] = False
            row["rejection_reason"] = "model_unresolved"
            rejections.append({"model_key": key, "rejection_reason": "model_unresolved"})
        else:
            row["profile_eligible"] = False
            row["rejection_reason"] = "profile_missing"
            rejections.append({"model_key": key, "rejection_reason": "profile_missing"})
        if row["classification_authority"] in ("unresolved", "source_role_mixed"):
            rejections.append({
                "model_key": key,
                "rejection_reason": f"semantics_{row['classification_authority']}",
                "observed_roles": row["observed_roles"],
            })
    inventory["profile_failures"] = profiles["failures"]

    digest_by_key = {key: row.get("geometry_digest") for key, row in profile_by_key.items()}
    families = alias_families(inventory)
    alias_rows = []
    for stem, members in sorted(families.items()):
        digests = {key: digest_by_key.get(key) for key in members}
        known = [d for d in digests.values() if d]
        equivalent = len(known) == len(members) and len(set(known)) == 1
        alias_rows.append({
            "family": stem,
            "members": members,
            "geometry_digests": digests,
            "equivalent": equivalent,
            "evidence": "identical evaluated-geometry digests" if equivalent else "geometry differs; not an alias",
        })

    write_json(WORKSPACE / outputs["inventory"], inventory)
    write_json(WORKSPACE / outputs["alias_evidence"], {
        "schema_version": 1,
        "origin": "evaluated_geometry_digest",
        "families": alias_rows,
    })
    write_json(WORKSPACE / outputs["rejection_list"], {
        "schema_version": 1,
        "rejections": sorted(rejections, key=lambda r: (str(r["rejection_reason"]), str(r["model_key"]))),
    })

    unresolved = sum(1 for r in rejections if r["rejection_reason"] in ("model_unresolved", "profile_missing"))
    gate_ok = unresolved == 0 and not profiles["failures"]
    print(json.dumps({
        "models": len(models),
        "profiled": len(profile_by_key),
        "unresolved": unresolved,
        "alias_families": len(alias_rows),
        "alias_equivalent": sum(1 for r in alias_rows if r["equivalent"]),
        "review_rows": len(rejections) - unresolved,
        "gate_ok": gate_ok,
    }, sort_keys=True))
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
