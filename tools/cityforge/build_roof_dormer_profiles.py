#!/usr/bin/env python3
"""Config-driven Phase 4 host driver: evidence -> pure roof/relation products."""
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
sys.path.insert(0, str(WORKSPACE / "src"))

from procgen.building_gen.normalize import canonicalize  # noqa: E402
from procgen.building_gen.roofs import build_dormer_relation, extract_roof_profile  # noqa: E402

EVIDENCE = WORKSPACE / "tools" / "cityforge" / "blender_roof_dormer_evidence.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(canonicalize(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def model_key(value: str) -> str:
    return str(value).replace("/", "\\").casefold()


def load_sources(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(site["site_id"]): read_json(WORKSPACE / str(site["stamp_library"]))
        for site in config["sites"]
    }


def validate_selection(config: dict[str, Any], phase02_inventory: dict[str, Any], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    inventory = {model_key(row["model_key"]): row for row in phase02_inventory["models"]}
    errors: list[str] = []
    for selected in config["selection"]["shells"]:
        row = inventory.get(model_key(selected["model_key"]))
        if row is None:
            errors.append(f"shell_not_in_inventory:{selected['model_key']}")
            continue
        if not row.get("profile_eligible"):
            errors.append(f"shell_not_profile_eligible:{selected['model_key']}")
        roles = [str(role).casefold() for role in row.get("observed_roles", [])]
        if roles != ["shell"]:
            errors.append(f"shell_role_not_exact:{selected['model_key']}:{roles}")
    for relation in config["selection"]["dormer_relations"]:
        site_id = str(relation["site_id"])
        source = sources.get(site_id)
        if source is None:
            errors.append(f"relation_site_missing:{site_id}")
            continue
        stamp = next((row for row in source.get("stamps", []) if row.get("stamp_id") == relation["stamp_id"]), None)
        if stamp is None:
            errors.append(f"relation_stamp_missing:{site_id}:{relation['stamp_id']}")
            continue
        member_ids = [str(member.get("source_id")) for member in stamp.get("members", [])]
        if len(member_ids) != len(set(member_ids)):
            errors.append(f"relation_duplicate_member_ids:{relation['stamp_id']}")
        by_id = {str(member.get("source_id")): member for member in stamp.get("members", [])}
        shell = by_id.get(str(relation["shell_member_source_id"]))
        dormer = by_id.get(str(relation["dormer_member_source_id"]))
        if shell is None:
            errors.append(f"relation_shell_member_missing:{relation['stamp_id']}:{relation['shell_member_source_id']}")
        elif str(shell.get("structural_role", "")).casefold() != "shell":
            errors.append(f"relation_shell_role_invalid:{relation['stamp_id']}:{relation['shell_member_source_id']}")
        if shell is not None and model_key(shell.get("model_key", "")) != model_key(relation.get("shell_model_key", "")):
            errors.append(f"relation_shell_model_mismatch:{relation['stamp_id']}:{shell.get('model_key')}")
        if dormer is None:
            errors.append(f"relation_dormer_member_missing:{relation['stamp_id']}:{relation['dormer_member_source_id']}")
        elif str(dormer.get("structural_role", "")).casefold() != "dormer":
            errors.append(f"relation_dormer_role_invalid:{relation['stamp_id']}:{relation['dormer_member_source_id']}")
        if dormer is not None and model_key(dormer.get("model_key", "")) != model_key(relation.get("dormer_model_key", "")):
            errors.append(f"relation_dormer_model_mismatch:{relation['stamp_id']}:{dormer.get('model_key')}")
        if shell is not None and model_key(shell["model_key"]) not in {model_key(x["model_key"]) for x in config["selection"]["shells"]}:
            errors.append(f"relation_shell_not_selected:{relation['stamp_id']}:{shell['model_key']}")
    if errors:
        raise RuntimeError("invalid Phase 4 selection: " + "; ".join(errors))
    return inventory


def run_evidence(meshes: list[str], output: Path, blender: str) -> dict[str, Any]:
    executable = shutil.which(blender)
    if executable is None:
        raise RuntimeError(f"blender not found on PATH ({blender!r})")
    with tempfile.TemporaryDirectory(prefix="roof_evidence_") as temp_name:
        job = Path(temp_name) / "job.json"
        job.write_text(json.dumps({"out": str(output), "meshes": meshes}, sort_keys=True) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [executable, "-b", "--python", str(EVIDENCE), "--", str(job)],
            cwd=WORKSPACE,
            check=False,
        )
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError("Blender evidence export failed; no fallback is permitted")
    evidence = read_json(output)
    if evidence.get("failures"):
        raise RuntimeError(f"Blender evidence failures: {evidence['failures']}")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 4 roof/dormer pilot products")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    config = read_json(args.config)
    outputs = {name: WORKSPACE / path for name, path in config["outputs"].items() if name not in {"root", "diagnostics"}}
    (WORKSPACE / config["outputs"]["root"]).mkdir(parents=True, exist_ok=True)
    sources = load_sources(config)
    inventory = read_json(WORKSPACE / config["phase02"]["inventory"])
    validate_selection(config, inventory, sources)

    shells = [dict(row) for row in config["selection"]["shells"]]
    shell_meshes = [str(row["model_key"]) for row in shells]
    relation_rows = list(config["selection"]["dormer_relations"])
    evidence_meshes = list(shell_meshes)
    relation_context = []
    for relation in relation_rows:
        source = sources[str(relation["site_id"])]
        stamp = next(row for row in source["stamps"] if row["stamp_id"] == relation["stamp_id"])
        dormer = next(member for member in stamp["members"] if member["source_id"] == relation["dormer_member_source_id"])
        evidence_meshes.append(str(dormer["model_key"]))
        relation_context.append((relation, source, stamp))
    evidence_meshes = sorted(set(evidence_meshes), key=model_key)
    evidence = run_evidence(evidence_meshes, outputs["evidence"], args.blender)
    write_json(outputs["evidence"], evidence)
    evidence_by_key = {model_key(row["model_key"]): row for row in evidence["models"]}
    missing_meshes = sorted({model_key(mesh) for mesh in evidence_meshes} - set(evidence_by_key), key=model_key)
    if missing_meshes:
        raise RuntimeError(f"Blender evidence omitted configured meshes: {missing_meshes}")

    measurement = dict(config["measurement"])
    profiles = []
    for selected in shells:
        row = evidence_by_key[model_key(selected["model_key"])]
        profile_measurement = dict(measurement)
        overrides = measurement.get("roof_floor_fraction_overrides", {})
        profile_measurement["roof_floor_fraction"] = next(
            (value for candidate, value in overrides.items() if model_key(candidate) == model_key(selected["model_key"])),
            measurement["roof_floor_fraction"],
        )
        profile = extract_roof_profile(
            row["model_key"], row["triangles"], row["bounds_local_gu"], profile_measurement
        )
        profile["case"] = selected["case"]
        profile["resolved_path"] = row["resolved_path"]
        profiles.append(profile)
    if any(profile["status"] != "eligible" for profile in profiles):
        raise RuntimeError("Phase 4 roof pilot contains a shell with no eligible roof patch")
    write_json(outputs["roofs"], {"schema_version": 1, "origin": "evaluated_roof_triangles", "measurement": measurement, "profiles": profiles})

    relations = []
    for relation, _source, stamp in relation_context:
        by_id = {str(member["source_id"]): member for member in stamp["members"]}
        shell = by_id[str(relation["shell_member_source_id"])]
        dormer = by_id[str(relation["dormer_member_source_id"])]
        profile = next(profile for profile in profiles if model_key(profile["model_key"]) == model_key(shell["model_key"]))
        dormer_evidence = evidence_by_key[model_key(dormer["model_key"])]
        relations.append(build_dormer_relation(str(relation["site_id"]), stamp, shell, dormer, profile, dormer_evidence["triangles"], measurement))
    write_json(outputs["dormers"], {"schema_version": 1, "relations": relations})
    write_json(outputs["selection_report"], {
        "schema_version": 1,
        "selected_shells": sorted(shell_meshes, key=model_key),
        "evidence_meshes": evidence_meshes,
        "selected_relations": relation_rows,
        "skipped": [],
    })
    print(json.dumps({"shells": len(profiles), "relations": len(relations), "evidence_meshes": len(evidence_meshes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
