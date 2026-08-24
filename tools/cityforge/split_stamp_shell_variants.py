#!/usr/bin/env python3
"""Split an observed multi-shell stamp into shell-specific render variants.

Purpose
-------
Reconstruct a source stamp from complementary diagnostic JSONs when a prior
review variant removed a different shell from each file, then emit one
standalone variant per shell. Every variant retains all non-shell members and
filters contact evidence to the retained member set.

Inputs
-------
``--config`` JSON with ``source_stamps`` (one or more stamp JSON paths) and
``output_dir``. The source files are merged by ``source_id``; duplicate
members must have identical canonical JSON. Optional ``extra_touching_pairs``
can restore explicitly witnessed shell pairs omitted by prior exclusions.

Outputs
-------
The reconstructed source stamp, one filtered stamp and one 2x3 Blender sheet
per shell, plus ``split_summary.json``. Source files are never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

WORKSPACE = Path(__file__).resolve().parents[2]
RENDERER = WORKSPACE / "tools" / "cityforge" / "render_generated_house.py"
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.building_gen.normalize import to_source_world, to_template_local  # noqa: E402
from procgen.engine_transform import tes3_euler_to_matrix  # noqa: E402


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _edge_key(edge: Mapping[str, Any]) -> tuple[str, str]:
    return tuple(sorted((str(edge.get("ref_a")), str(edge.get("ref_b")))))


def _merge_stamps(stamps: list[Mapping[str, Any]], extra_pairs: list[list[str]]) -> dict[str, Any]:
    if not stamps:
        raise ValueError("source_stamps must not be empty")
    result = copy.deepcopy(dict(stamps[0]))
    members: dict[str, dict[str, Any]] = {}
    for stamp in stamps:
        for member in stamp.get("members") or ():
            source_id = str(member.get("source_id"))
            previous = members.get(source_id)
            if previous is not None and previous != member:
                raise ValueError(f"conflicting member definitions for {source_id}")
            members[source_id] = copy.deepcopy(member)
    result["members"] = [members[key] for key in sorted(members)]
    for field in ("member_contact_edges", "shell_attachment_edges", "internal_edges"):
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for stamp in stamps:
            for edge in stamp.get(field) or ():
                merged.setdefault(_edge_key(edge), copy.deepcopy(edge))
        result[field] = [merged[key] for key in sorted(merged)]
    pairs = {
        tuple(sorted((str(pair[0]), str(pair[1]))))
        for stamp in stamps
        for pair in stamp.get("touching_pairs") or ()
    }
    pairs.update(tuple(sorted((str(pair[0]), str(pair[1])))) for pair in extra_pairs)
    result["touching_pairs"] = [list(pair) for pair in sorted(pairs)]
    result["stamp_id"] = f"{result.get('stamp_id')}__reconstructed_original"
    result["multi_shell"] = True
    result["source"] = {
        "kind": "complementary_variant_reconstruction",
        "source_stamp_ids": [stamp.get("stamp_id") for stamp in stamps],
    }
    return result


def _split_stamp(
    stamp: Mapping[str, Any], shell_id: str, attachment_host_shell_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    members_by_id = {
        str(member.get("source_id")): member for member in stamp.get("members") or ()
    }
    host = members_by_id.get(attachment_host_shell_id)
    if host is None or host.get("structural_role") != "shell":
        raise ValueError(f"attachment host shell not found: {attachment_host_shell_id}")
    target = members_by_id[shell_id]
    retained = {
        str(member.get("source_id"))
        for member in stamp.get("members") or ()
        if member.get("structural_role") != "shell" or str(member.get("source_id")) == shell_id
    }
    result = copy.deepcopy(dict(stamp))
    result["stamp_id"] = f"{stamp.get('stamp_id')}__shell_{shell_id.replace('-', '_')}"
    host_frame = tes3_euler_to_matrix(host["rotation"])
    target_frame = tes3_euler_to_matrix(target["rotation"])
    target_members = []
    for member in stamp.get("members") or ():
        source_id = str(member.get("source_id"))
        if source_id not in retained:
            continue
        copied = copy.deepcopy(member)
        if source_id != shell_id:
            local_position, local_rotation = to_template_local(
                member["offset_gu"], member["rotation"], host["offset_gu"], host_frame
            )
            copied["offset_gu"], copied["rotation"] = to_source_world(
                local_position, local_rotation, target["offset_gu"], target_frame
            )
        target_members.append(copied)
    result["members"] = target_members
    # The original contact rows are not authoritative after a frame copy.
    result["member_contact_edges"] = []
    result["shell_attachment_edges"] = []
    result["internal_edges"] = []
    result["touching_pairs"] = []
    result["multi_shell"] = False
    result["source"] = {
        "kind": "shell_specific_attachment_copy",
        "source_stamp_id": stamp.get("stamp_id"),
        "shell_source_id": shell_id,
        "attachment_host_shell_id": attachment_host_shell_id,
        "attachment_frame_copy": "host_shell_local_to_target_shell_frame",
        "copied_attachment_source_ids": sorted(retained - {shell_id}),
    }
    summary = {
        "shell_source_id": shell_id,
        "retained_member_count": len(result["members"]),
        "copied_attachment_count": len(retained) - 1,
        "retained_contact_edge_count": 0,
        "shell_attachment_edges": [],
    }
    return result, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Split a multi-shell stamp into shell-specific attachment copies")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    config = _read(_path(str(args.config)))
    source_stamps = [_read(_path(str(path))) for path in config.get("source_stamps") or ()]
    original = _merge_stamps(source_stamps, config.get("extra_touching_pairs") or [])
    output_dir = _path(str(config["output_dir"]))
    attachment_host_shell_id = str(config["attachment_host_shell_id"])
    original_path = output_dir / "c101_reconstructed_original.json"
    _write(original_path, original)
    shells = sorted(
        str(member.get("source_id"))
        for member in original.get("members") or ()
        if member.get("structural_role") == "shell"
    )
    summaries: list[dict[str, Any]] = []
    for shell_id in shells:
        variant, summary = _split_stamp(original, shell_id, attachment_host_shell_id)
        slug = shell_id.replace("-", "_")
        variant_path = output_dir / f"shell_{slug}.json"
        image_path = output_dir / f"shell_{slug}_sheet_2x3.png"
        _write(variant_path, variant)
        command = [sys.executable, str(RENDERER), "--stamp", str(variant_path), "--out", str(image_path), "--blender", str(args.blender)]
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"variant render failed: {shell_id} exit {completed.returncode}")
        summary.update({
            "stamp_json": str(variant_path.relative_to(WORKSPACE)).replace("\\", "/"),
            "render_png": str(image_path.relative_to(WORKSPACE)).replace("\\", "/"),
        })
        summaries.append(summary)
    _write(output_dir / "split_summary.json", {
        "schema_version": 1,
        "reconstructed_source": str(original_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "variants": summaries,
    })
    print(json.dumps({"shell_count": len(shells), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
