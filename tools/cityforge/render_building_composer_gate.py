#!/usr/bin/env python3
"""Render a config-selected Phase 6 building set for the Phase 7 visual gate.

Pipeline position
------------------
Consumes the real Phase 6 base/extension result JSON and sends only selected
accepted generated stamps through the existing ``tools/mesh_thumbs.py`` Blender
bridge. It writes flat, browsable 2x3 sheets plus canonical scene, stamp-copy,
per-sheet audit, and manifest evidence under a fresh output directory.

Inputs
-------
JSON config with Phase 6 result paths, accepted request IDs, render settings,
and a diagnostic ground-plane specification. Source NIFs are resolved by the
existing configured read-only data-root resolver inside Blender.

Outputs and invariants
----------------------
Every selected case must be accepted, every scene member must import, and the
bridge audit must report the same piece count with no excluded building piece.
The driver never edits the selected building members or substitutes meshes; a
case failure aborts the run rather than producing an acceptance manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
TOOLS = WORKSPACE / "tools"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.kit_house_grammar import stamp_to_sheet_scene  # noqa: E402

MESH_THUMBS = TOOLS / "mesh_thumbs.py"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _workspace_path(value: str, label: str) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else WORKSPACE / path
    if not resolved.exists():
        raise RuntimeError(f"{label} does not exist: {resolved}")
    return resolved


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _result_index(document: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    rows = document.get("results")
    if not isinstance(rows, list):
        raise RuntimeError(f"{label} has no results list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("request_id"), str):
            raise RuntimeError(f"{label} contains a malformed result row")
        request_id = str(row["request_id"])
        if request_id in indexed:
            raise RuntimeError(f"{label} contains duplicate request_id {request_id!r}")
        indexed[request_id] = row
    return indexed


def _fresh_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"phase07 output path is not a directory: {path}")
        if any(path.iterdir()):
            raise RuntimeError(f"phase07 output directory is not fresh: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def _relative(path: Path) -> str:
    try:
        return path.relative_to(WORKSPACE).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def _render_case(
    *,
    case: Mapping[str, Any],
    result: Mapping[str, Any],
    output_dir: Path,
    render_config_path: Path,
    ground_plane: Mapping[str, Any],
    blender: str,
) -> dict[str, Any]:
    request_id = str(case["request_id"])
    slug = str(case["output_slug"])
    if not slug or Path(slug).name != slug:
        raise RuntimeError(f"invalid output_slug for {request_id!r}")
    building = result.get("building")
    if not isinstance(building, Mapping):
        raise RuntimeError(f"accepted result has no building: {request_id}")
    stamp = copy.deepcopy(dict(building))
    stamp["ground_plane"] = copy.deepcopy(dict(ground_plane))

    stem = output_dir / f"{slug}_sheet_2x3"
    stamp_path = output_dir / f"{slug}_stamp.json"
    scene_path = stem.with_suffix(".scene.json")
    png_path = stem.with_suffix(".png")
    audit_path = output_dir / f"{slug}_sheet_2x3_audit.json"
    stamp_path.write_bytes(_canonical_bytes(stamp))

    scene = stamp_to_sheet_scene(stamp, scene_name=f"phase07_{slug}_sheet")
    scene["ground_plane"] = copy.deepcopy(dict(ground_plane))
    scene_path.write_bytes(_canonical_bytes(scene))

    command = [
        blender,
        "-b",
        "--factory-startup",
        "--python",
        str(MESH_THUMBS),
        "--",
        str(scene_path),
        str(png_path),
        "--config",
        str(render_config_path),
    ]
    print(f"[phase07] rendering {request_id}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=WORKSPACE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Blender failed for {request_id} with exit {completed.returncode}")
    if not png_path.is_file() or png_path.stat().st_size <= 0:
        raise RuntimeError(f"render did not produce a non-empty PNG for {request_id}")
    if not audit_path.is_file():
        raise RuntimeError(f"render did not produce the mesh audit for {request_id}")
    audit = _load_json(audit_path, f"mesh audit for {request_id}")
    meshes = scene.get("meshes")
    expected = len(meshes) if isinstance(meshes, list) else 0
    actual = audit.get("rendered_piece_count_actual")
    excluded = audit.get("excluded_pieces")
    if actual != expected:
        raise RuntimeError(f"piece-count mismatch for {request_id}: scene={expected}, audit={actual}")
    if not isinstance(excluded, list) or excluded:
        raise RuntimeError(f"building pieces were excluded for {request_id}: {excluded}")
    return {
        "request_id": request_id,
        "output_slug": slug,
        "status": str(result["status"]),
        "member_count": len(stamp["members"]),
        "door_count": int(stamp["door_count"]),
        "stamp": _relative(stamp_path),
        "scene": _relative(scene_path),
        "png": _relative(png_path),
        "audit": _relative(audit_path),
        "expected_piece_count": expected,
        "actual_piece_count": actual,
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path, "phase07 config")
    if int(config.get("phase", -1)) != 7:
        raise RuntimeError("phase07 config must have phase=7")
    inputs = config.get("input_documents")
    render = config.get("render")
    cases = config.get("cases")
    ground_plane = config.get("ground_plane")
    if not isinstance(inputs, Mapping) or not isinstance(render, Mapping) or not isinstance(cases, list):
        raise RuntimeError("phase07 config requires input_documents, render, and cases")
    if not isinstance(ground_plane, Mapping):
        raise RuntimeError("phase07 config requires a ground_plane object")
    output_value = Path(str(config["output_dir"]))
    output_dir = output_value if output_value.is_absolute() else WORKSPACE / output_value
    if not output_dir.parent.exists():
        raise RuntimeError(f"phase07 output parent does not exist: {output_dir.parent}")
    _fresh_output(output_dir)
    base_path = _workspace_path(str(inputs["base_results"]), "base_results")
    extension_path = _workspace_path(str(inputs["extension_results"]), "extension_results")
    indexes = {
        "base_results": _result_index(_load_json(base_path, "base_results"), "base_results"),
        "extension_results": _result_index(_load_json(extension_path, "extension_results"), "extension_results"),
    }
    if not cases:
        raise RuntimeError("phase07 config has no render cases")

    render_config = dict(render)
    render_config.pop("blender", None)
    render_config.pop("ground_plane", None)
    render_config_path = output_dir / "mesh_thumbs_config.json"
    render_config_path.write_bytes(_canonical_bytes(render_config))
    blender_value = str(render.get("blender", "blender"))
    blender = blender_value if Path(blender_value).is_file() else shutil.which(blender_value)
    if blender is None:
        raise RuntimeError(f"Blender not found: {blender_value}")

    rendered: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise RuntimeError("phase07 cases must be objects")
        source_document = str(case.get("source_document", ""))
        request_id = str(case.get("request_id", ""))
        if source_document not in indexes:
            raise RuntimeError(f"unknown source_document {source_document!r} for {request_id!r}")
        result = indexes[source_document].get(request_id)
        if result is None:
            raise RuntimeError(f"render case request_id is absent: {request_id}")
        if result.get("status") != "accepted":
            raise RuntimeError(f"render case is not accepted: {request_id}")
        rendered.append(
            _render_case(
                case=case,
                result=result,
                output_dir=output_dir,
                render_config_path=render_config_path,
                ground_plane=ground_plane,
                blender=blender,
            )
        )

    manifest = {
        "schema_version": 1,
        "phase": 7,
        "kit_id": str(config.get("kit_id", "")),
        "input_documents": {
            "base_results": _relative(base_path),
            "extension_results": _relative(extension_path),
        },
        "render": copy.deepcopy(dict(render)),
        "ground_plane": copy.deepcopy(dict(ground_plane)),
        "cases": rendered,
        "status": "complete",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(_canonical_bytes(manifest))
    print(f"[phase07] complete: {len(rendered)} sheets under {output_dir}", flush=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args.config.resolve())
    except Exception as exc:
        print(f"FAILURE: phase07_render {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
