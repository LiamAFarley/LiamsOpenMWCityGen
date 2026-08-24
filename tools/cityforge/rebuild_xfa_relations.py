#!/usr/bin/env python3
"""Build Phase 1 normalized templates and witnessed relation products.

The CLI is config-driven and reads the immutable xFa source libraries only.
It writes per-site canonical derived JSON, a hard source/relation round-trip
report, and exactly three Falkreath source-versus-reconstruction sheet pairs.
Rendering uses the existing ``render_generated_house.py`` bridge in a temporary
directory; only the final pair PNGs remain under the configured audit folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.building_gen.normalize import canonicalize, to_source_world  # noqa: E402
from procgen.building_gen.rebuild import (  # noqa: E402
    build_connection_document,
    build_template_document,
    choose_seed_door,
    roundtrip_connection_sample,
)
from procgen.engine_transform import matrix_to_tes3_euler, tes3_euler_to_matrix  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonicalize(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def source_world_stamp(stamp: Mapping[str, Any], template: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct a template as a source-world D-STAMP-like render input."""

    source_members = {member["source_id"]: member for member in stamp["members"]}
    seed_id = str(template["seed_door"])
    seed = source_members[seed_id]
    p0 = seed["offset_gu"]
    R0 = tes3_euler_to_matrix(seed["rotation"])
    members = []
    for local in template["members"]:
        position, rotation = to_source_world(local["offset_local_gu"], local["rotation_local_rad"], p0, R0)
        row = dict(source_members[local["source_id"]])
        row["offset_gu"] = position
        row["rotation"] = rotation
        row["scale"] = local["scale"]
        members.append(row)
    result = dict(stamp)
    result["members"] = members
    result["stamp_id"] = f"{stamp['stamp_id']}__reconstructed"
    result["source"] = {"kind": "phase01_reconstruction", "source_stamp_id": stamp["stamp_id"], "seed_door": seed_id}
    return canonicalize(result)


def render_pair(
    stamp: Mapping[str, Any],
    template: Mapping[str, Any],
    output_dir: Path,
    blender: str,
) -> dict[str, Any]:
    """Render two full sheets in scratch space and retain one side-by-side PNG."""

    render_script = WORKSPACE / "tools" / "cityforge" / "render_generated_house.py"
    with tempfile.TemporaryDirectory(prefix="building_rule_kit_render_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_path = temp_dir / "source.json"
        reconstructed_path = temp_dir / "reconstructed.json"
        source_png = temp_dir / "source.png"
        reconstructed_png = temp_dir / "reconstructed.png"
        source_path.write_text(json.dumps(canonicalize(stamp), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        reconstructed = source_world_stamp(stamp, template)
        reconstructed_path.write_text(json.dumps(reconstructed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for input_path, png_path in ((source_path, source_png), (reconstructed_path, reconstructed_png)):
            command = [sys.executable, str(render_script), "--stamp", str(input_path), "--out", str(png_path), "--blender", blender]
            completed = subprocess.run(command, cwd=WORKSPACE, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"render bridge failed for {input_path.name} with exit {completed.returncode}")
        with Image.open(source_png) as source_image, Image.open(reconstructed_png) as reconstructed_image:
            if source_image.size != reconstructed_image.size:
                raise RuntimeError("source and reconstruction render sizes differ")
            pair = Image.new("RGB", (source_image.width * 2, source_image.height), (24, 24, 24))
            pair.paste(source_image.convert("RGB"), (0, 0))
            pair.paste(reconstructed_image.convert("RGB"), (source_image.width, 0))
            # Eevee can round one edge pixel differently across identical CPU
            # runs. Canonicalize only the final 8-bit evidence sheet; geometry,
            # source renders, and reconstructed member data remain unchanged.
            pair = pair.point(lambda value: int(value) & ~1)
            output_path = output_dir / f"{stamp['stamp_id']}_source_vs_reconstructed.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            pair.save(output_path, format="PNG")
    return {"stamp_id": stamp["stamp_id"], "output": str(output_path.relative_to(WORKSPACE)).replace("\\", "/"), "notes": "source left, reconstructed right; visual review required"}


def template_roundtrip(stamp: Mapping[str, Any], template: Mapping[str, Any], position_tolerance: float, rotation_tolerance: float) -> dict[str, Any]:
    if template.get("evidence_class") != "observed_exact":
        return {"stamp_id": template["stamp_id"], "passed": False, "excluded": True, "reason": template.get("rejection")}
    source_members = {member["source_id"]: member for member in stamp["members"]}
    seed_id = str(template["seed_door"])
    seed = source_members[seed_id]
    p0 = seed["offset_gu"]
    R0 = tes3_euler_to_matrix(seed["rotation"])
    failures = []
    max_position = 0.0
    max_rotation = 0.0
    for member in template["members"]:
        original = source_members[member["source_id"]]
        position, rotation = to_source_world(member["offset_local_gu"], member["rotation_local_rad"], p0, R0)
        position_error = max(abs(float(left) - float(right)) for left, right in zip(position, original["offset_gu"]))
        rotation_error = float(abs(tes3_euler_to_matrix(rotation) - tes3_euler_to_matrix(original["rotation"])).max())
        max_position = max(max_position, position_error)
        max_rotation = max(max_rotation, rotation_error)
        scale_equal = member["scale"] == round(float(original["scale"]), 6)
        if position_error > position_tolerance or rotation_error > rotation_tolerance or not scale_equal:
            failures.append({"source_id": member["source_id"], "position_residual_gu": position_error, "rotation_matrix_residual": rotation_error, "scale_equal": scale_equal})
    seed_local = next(member for member in template["members"] if member["source_id"] == seed_id)
    seed_position, seed_rotation = to_source_world(seed_local["offset_local_gu"], seed_local["rotation_local_rad"], p0, R0)
    seed_rotation_error = float(abs(tes3_euler_to_matrix(seed_rotation) - R0).max())
    return {
        "stamp_id": template["stamp_id"],
        "passed": not failures and max(abs(value) for value in seed_local["offset_local_gu"]) <= position_tolerance and seed_rotation_error <= rotation_tolerance,
        "excluded": False,
        "failure_count": len(failures),
        "failures": failures,
        "max_position_residual_gu": max_position,
        "max_rotation_matrix_residual": max_rotation,
        "seed_position_residual_gu": max(abs(float(left) - float(right)) for left, right in zip(seed_position, p0)),
        "seed_rotation_matrix_residual": seed_rotation_error,
    }


def select_render_stamps(source_library: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    candidates = [stamp for stamp in source_library["stamps"] if any(member.get("structural_role") == "shell" for member in stamp.get("members", [])) and any(member.get("is_door") for member in stamp.get("members", []))]
    candidates = sorted(candidates, key=lambda stamp: str(stamp["stamp_id"]))
    single = next((stamp for stamp in candidates if not stamp.get("multi_shell")), None)
    multi = next((stamp for stamp in candidates if stamp.get("multi_shell")), None)
    nonunit = next((stamp for stamp in candidates if any(float(member.get("scale", 1.0)) != 1.0 for member in stamp.get("members", []))), None)
    used_ids = {stamp["stamp_id"] for stamp in (single, multi) if stamp is not None}
    nonunit = next(
        (
            stamp
            for stamp in candidates
            if stamp["stamp_id"] not in used_ids
            and any(float(member.get("scale", 1.0)) != 1.0 for member in stamp.get("members", []))
        ),
        None,
    )
    selected = [("single_shell", single), ("multi_shell", multi), ("non_unit_scale", nonunit)]
    if any(stamp is None for _, stamp in selected):
        raise RuntimeError("cannot select the required three real Falkreath render stamps")
    return [(kind, stamp) for kind, stamp in selected if stamp is not None]


def run_site(site: Mapping[str, Any], config: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    site_id = str(site["site_id"])
    library = read_json(root / str(site["stamp_library"]))
    templates, template_stats = build_template_document(
        library,
        site_id=site_id,
        position_tolerance_gu=float(config["tolerances"]["position_gu"]),
        rotation_tolerance=float(config["tolerances"]["rotation_matrix"]),
    )
    connections = build_connection_document(
        library,
        site_id=site_id,
        contact_interval_half_width_gu=float(config["tolerances"]["contact_interval_half_width_gu"]),
    )
    source_by_stamp = {stamp["stamp_id"]: stamp for stamp in library["stamps"]}
    template_results = [
        template_roundtrip(source_by_stamp[template["stamp_id"]], template, float(config["tolerances"]["position_gu"]), float(config["tolerances"]["rotation_matrix"]))
        for template in templates["templates"]
    ]
    sample_results = []
    for rule in connections["rules"]:
        for sample in rule["samples"]:
            source_stamp = source_by_stamp[sample["witness"]["source_stamp_id"]]
            source_members = {member["source_id"]: member for member in source_stamp["members"]}
            result = roundtrip_connection_sample(
                sample,
                source_members,
                position_tolerance_gu=float(config["tolerances"]["position_gu"]),
                rotation_tolerance=float(config["tolerances"]["rotation_matrix"]),
            )
            result["sample_id"] = sample["sample_id"]
            sample_results.append(result)
    for sample in connections["attachment_contacts"]:
        source_stamp = source_by_stamp[sample["witness"]["source_stamp_id"]]
        source_members = {member["source_id"]: member for member in source_stamp["members"]}
        result = roundtrip_connection_sample(
            sample,
            source_members,
            position_tolerance_gu=float(config["tolerances"]["position_gu"]),
            rotation_tolerance=float(config["tolerances"]["rotation_matrix"]),
        )
        result["sample_id"] = sample["sample_id"]
        sample_results.append(result)
    report = {
        "site_id": site_id,
        "templates_processed": len(template_results),
        "templates_eligible": sum(not row.get("excluded") for row in template_results),
        "templates_excluded": sum(bool(row.get("excluded")) for row in template_results),
        "templates_failed": sum(not row["passed"] and not row.get("excluded") for row in template_results),
        "connection_samples_processed": len(sample_results),
        "connection_samples_passed": sum(bool(row["passed"]) for row in sample_results),
        "connection_samples_failed": sum(not row["passed"] for row in sample_results),
        "max_position_residual_gu": max([row.get("max_position_residual_gu", row.get("position_residual_gu", 0.0)) for row in template_results + sample_results] or [0.0]),
        "max_rotation_matrix_residual": max([row.get("max_rotation_matrix_residual", row.get("rotation_matrix_residual", 0.0)) for row in template_results + sample_results] or [0.0]),
        "template_results": template_results,
        "connection_results": sample_results,
        "source_stamp_count": len(library["stamps"]),
        "template_stats": template_stats,
    }
    return templates, connections, canonicalize(report), library


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild xFa normalized templates and witnessed relations")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true", help="skip the required render gate for local debugging")
    args = parser.parse_args()
    config = read_json(args.config)
    root = WORKSPACE
    output_root = root / config["outputs"]["root"]
    all_reports = []
    render_reports = []
    falkreath_library = None
    falkreath_templates = None
    for site in config["sites"]:
        templates, connections, report, library = run_site(site, config, root)
        site_id = str(site["site_id"])
        site_dir = output_root / "derived" / site_id
        write_json(site_dir / "templates_v1.json", templates)
        write_json(site_dir / "connections_v1.json", connections)
        all_reports.append(report)
        if site_id == str(config["render"]["site_id"]):
            falkreath_library = library
            falkreath_templates = templates
    if not args.no_render:
        if falkreath_library is None or falkreath_templates is None:
            raise RuntimeError("configured render site was not processed")
        templates_by_id = {template["stamp_id"]: template for template in falkreath_templates["templates"]}
        blender = str(config.get("render", {}).get("blender", "blender"))
        render_dir = root / config["outputs"]["render_dir"]
        for kind, stamp in select_render_stamps(falkreath_library):
            template = templates_by_id.get(stamp["stamp_id"])
            if template is None or template.get("evidence_class") != "observed_exact":
                raise RuntimeError(f"selected render stamp {stamp['stamp_id']} has no eligible normalized template")
            result = render_pair(stamp, template, render_dir, blender)
            result["selection"] = kind
            render_reports.append(result)
    failed = sum(report["templates_failed"] + report["connection_samples_failed"] for report in all_reports)
    if failed:
        raise RuntimeError(f"Phase 1 round-trip hard gate failed with {failed} unexpected failures")
    roundtrip = canonicalize({"schema_version": 1, "position_tolerance_gu": config["tolerances"]["position_gu"], "rotation_matrix_tolerance": config["tolerances"]["rotation_matrix"], "sites": all_reports, "render_pairs": render_reports})
    write_json(root / config["outputs"]["roundtrip"], roundtrip)
    print(json.dumps({"sites": len(all_reports), "render_pairs": len(render_reports), "unexpected_failures": failed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
