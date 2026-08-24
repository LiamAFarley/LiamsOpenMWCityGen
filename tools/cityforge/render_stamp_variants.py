#!/usr/bin/env python3
"""Render source-stamp variants after explicit member exclusions.

Purpose
-------
Create auditable visual variants of an observed stamp without changing the
source library.  Each variant is defined by JSON as a list of source member
ids to exclude; members and contact edges involving those ids are removed
together before rendering.

Inputs
------
``--config`` JSON containing ``input_stamp``, ``output_dir``, and
``variants``.  Each variant requires ``variant_id`` and
``exclude_source_ids``.  Paths are workspace-relative unless absolute.

Outputs
-------
One filtered stamp JSON and one 2x3 PNG per variant, plus
``variant_summary.json`` containing retained members and contact counts.
The input stamp is never modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

WORKSPACE = Path(__file__).resolve().parents[2]
RENDERER = WORKSPACE / "tools" / "cityforge" / "render_generated_house.py"


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else WORKSPACE / candidate


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _edge_kept(edge: Mapping[str, Any], excluded: set[str]) -> bool:
    return str(edge.get("ref_a")) not in excluded and str(edge.get("ref_b")) not in excluded


def _variant(stamp: Mapping[str, Any], variant: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_id = str(variant.get("variant_id") or "").strip()
    if not variant_id:
        raise ValueError("variant requires non-empty variant_id")
    excluded = {str(value) for value in variant.get("exclude_source_ids") or ()}
    source_ids = {str(member.get("source_id")) for member in stamp.get("members") or []}
    unknown = sorted(excluded - source_ids)
    if unknown:
        raise ValueError(f"{variant_id}: excluded source ids are not in stamp: {unknown}")
    result = dict(stamp)
    result["stamp_id"] = f"{stamp.get('stamp_id')}__{variant_id}"
    result["members"] = [
        member for member in stamp.get("members") or []
        if str(member.get("source_id")) not in excluded
    ]
    for key in ("member_contact_edges", "shell_attachment_edges", "internal_edges"):
        if key in result:
            result[key] = [edge for edge in result.get(key) or () if _edge_kept(edge, excluded)]
    result["source"] = dict(result.get("source") or {})
    result["source"].update({
        "kind": "explicit_member_exclusion_variant",
        "variant_of": stamp.get("stamp_id"),
        "excluded_source_ids": sorted(excluded),
    })
    shells = [
        str(member.get("source_id"))
        for member in result["members"]
        if member.get("structural_role") == "shell"
    ]
    retained_edges = list(result.get("member_contact_edges") or ())
    summary = {
        "variant_id": variant_id,
        "stamp_id": result["stamp_id"],
        "excluded_source_ids": sorted(excluded),
        "retained_member_count": len(result["members"]),
        "retained_shell_refs": shells,
        "member_contact_edge_count": len(retained_edges),
        "shell_contact_edges": [
            edge for edge in retained_edges
            if str(edge.get("ref_a")) in shells and str(edge.get("ref_b")) in shells
        ],
    }
    return result, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Render explicit source-stamp member variants")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()

    config = _read(_path(str(args.config)))
    stamp_path = _path(str(config["input_stamp"]))
    output_dir = _path(str(config["output_dir"]))
    source_document = _read(stamp_path)
    stamp_id = config.get("input_stamp_id")
    if stamp_id is None:
        stamp = source_document
    else:
        stamp = next(
            (candidate for candidate in source_document.get("stamps") or ()
             if str(candidate.get("stamp_id")) == str(stamp_id)),
            None,
        )
        if stamp is None:
            raise ValueError(f"input stamp id not found: {stamp_id}")
    summaries: list[dict[str, Any]] = []
    for raw_variant in config.get("variants") or ():
        variant, summary = _variant(stamp, raw_variant)
        variant_id = summary["variant_id"]
        variant_path = output_dir / f"{variant_id}.json"
        image_path = output_dir / f"{variant_id}_sheet_2x3.png"
        _write(variant_path, variant)
        command = [
            sys.executable,
            str(RENDERER),
            "--stamp",
            str(variant_path),
            "--out",
            str(image_path),
            "--blender",
            str(args.blender),
        ]
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"variant render failed: {variant_id} exit {completed.returncode}")
        summary["stamp_json"] = str(variant_path.relative_to(WORKSPACE)).replace("\\", "/")
        summary["render_png"] = str(image_path.relative_to(WORKSPACE)).replace("\\", "/")
        summaries.append(summary)
    _write(output_dir / "variant_summary.json", {"schema_version": 1, "variants": summaries})
    print(json.dumps({"variant_count": len(summaries), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
