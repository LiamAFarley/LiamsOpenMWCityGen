#!/usr/bin/env python3
"""Render representative extracted Falkreath wall, gate, and fence pieces.

The structural JSON library stores source-positioned palettes separately from
house stamps.  This preview selects one resolved source member per role and
normalizes them into a small D-STAMP so the actual meshes can be inspected
before a wall-chain or gatehouse composer is written.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.kit_house_grammar import canonical_json_bytes  # noqa: E402


def build_preview(library_path: Path, roles: list[str]) -> dict:
    library = json.loads(library_path.read_text(encoding="utf-8"))
    components = library.get("components") or []
    selected = []
    for role in roles:
        candidates = [row for row in components if row.get("structural_role") == role]
        if not candidates:
            raise ValueError(f"structural library has no component role {role!r}")
        selected.append(max(candidates, key=lambda row: int(row.get("source_count") or 0)))

    members = []
    anchor = [0.0, 0.0, 0.0]
    for index, component in enumerate(selected):
        source = component["source_refs"][0]
        # This is a palette preview, not an assembly reconstruction.  Keep
        # each actual mesh at a readable fixed separation; source placement is
        # still preserved verbatim in structural_library.json.
        position = [float(index * 1800.0), 0.0, 0.0]
        rotation = list(source.get("rotation") or [0.0, 0.0, 0.0])
        members.append({
            "source_id": f"preview_{component['structural_role']}_{index:03d}",
            "object_id": component["object_ids"][0],
            "model_key": component["model_key"],
            "record_type": "STAT",
            "category": "exterior",
            "is_door": False,
            "offset_gu": position,
            "rotation": rotation,
            "scale": float(source.get("scale") or 1.0),
            "structural_role": component["structural_role"],
        })
    return {
        "stamp_id": "falkreath_structural_preview",
        "source": {"kind": "structural_library_preview", "roles": roles},
        "building_type": "structural_preview",
        "size_class": "small",
        "door_count": 0,
        "multi_shell": False,
        "anchor": {"kind": "source_component", "source_position_gu": anchor},
        "access_heading_rad": 0.0,
        "members": members,
        "footprint": {"aabb_rel": {"min": [0, 0, 0], "max": [0, 0, 0], "span": [0, 0, 0]}, "hull_xy_rel": []},
        "bounds_rel_gu": {"min": [0, 0, 0], "max": [0, 0, 0], "span": [0, 0, 0]},
        "terrain_envelope": {"door_step_heights_gu": [], "footprint_relief_gu": 0.0, "footprint_slope_deg": 0.0, "burial_depth_gu": 0.0},
    }


def build_wall_orientation_test(library_path: Path) -> dict:
    """Place the same extracted wall in four rotations for visual comparison."""
    library = json.loads(library_path.read_text(encoding="utf-8"))
    candidates = [row for row in library.get("components", []) if row.get("structural_role") == "wall"]
    component = max(candidates, key=lambda row: int(row.get("source_count") or 0), default=None)
    if component is None:
        raise ValueError("structural library has no wall component")
    source = component["source_refs"][0]
    raw = list(source.get("rotation") or [0.0, 0.0, 0.0])
    variants = [("raw", [0.0, 0.0, 0.0]), ("rx_pi", [math.pi, 0.0, 0.0]),
                ("ry_pi", [0.0, math.pi, 0.0]), ("rz_pi", [0.0, 0.0, math.pi])]
    members = []
    for index, (label, delta) in enumerate(variants):
        members.append({
            "source_id": f"wall_orientation_{label}",
            "object_id": component["object_ids"][0],
            "model_key": component["model_key"],
            "record_type": "STAT",
            "category": "exterior",
            "is_door": False,
            "offset_gu": [float((index % 2) * 1800.0), float((index // 2) * 1800.0), 0.0],
            "rotation": [raw[i] + delta[i] for i in range(3)],
            "scale": float(source.get("scale") or 1.0),
            "structural_role": "wall",
            "orientation_test_label": label,
        })
    return {
        "stamp_id": "falkreath_wall_orientation_test",
        "source": {"kind": "structural_library_orientation_test", "model": component["model_key"]},
        "building_type": "structural_preview", "size_class": "small", "door_count": 0,
        "multi_shell": False, "anchor": {"kind": "source_component", "source_position_gu": [0, 0, 0]},
        "access_heading_rad": 0.0, "members": members,
        "footprint": {"aabb_rel": {"min": [0, 0, 0], "max": [0, 0, 0], "span": [0, 0, 0]}, "hull_xy_rel": []},
        "bounds_rel_gu": {"min": [0, 0, 0], "max": [0, 0, 0], "span": [0, 0, 0]},
        "terrain_envelope": {"door_step_heights_gu": [], "footprint_relief_gu": 0.0, "footprint_slope_deg": 0.0, "burial_depth_gu": 0.0},
    }


def build_source_cell_preview(library_path: Path, grid: tuple[int, int]) -> dict:
    """Render structural members in one authoritative source cell in place."""
    library = json.loads(library_path.read_text(encoding="utf-8"))
    rows = []
    for component in library.get("components", []):
        if component.get("structural_role") not in {"wall", "gate", "fence"}:
            continue
        for source in component.get("source_refs", []):
            if tuple(source.get("cell", {}).get("grid") or ()) == grid:
                rows.append((component, source))
    if not rows:
        raise ValueError(f"no structural members in source cell {grid}")
    anchor = rows[0][1]["position"]
    members = []
    for index, (component, source) in enumerate(rows):
        position = source["position"]
        members.append({
            "source_id": source.get("source_id") or f"cell_{index:03d}",
            "object_id": component["object_ids"][0],
            "model_key": component["model_key"],
            "record_type": "STAT", "category": "exterior", "is_door": False,
            "offset_gu": [float(position[i]) - float(anchor[i]) for i in range(3)],
            "rotation": list(source.get("rotation") or [0.0, 0.0, 0.0]),
            "scale": float(source.get("scale") or 1.0),
            "structural_role": component["structural_role"],
        })
    return {
        "stamp_id": f"falkreath_structural_cell_{grid[0]}_{grid[1]}",
        "source": {"kind": "structural_library_source_cell", "grid": list(grid)},
        "building_type": "structural_preview", "size_class": "large", "door_count": 0,
        "multi_shell": False, "anchor": {"kind": "source_component", "source_position_gu": anchor},
        "access_heading_rad": 0.0, "members": members,
        "footprint": {"aabb_rel": {"min": [0, 0, 0], "max": [0, 0, 0], "span": [0, 0, 0]}, "hull_xy_rel": []},
        "bounds_rel_gu": {"min": [0, 0, 0], "max": [0, 0, 0], "span": [0, 0, 0]},
        "terrain_envelope": {"door_step_heights_gu": [], "footprint_relief_gu": 0.0, "footprint_slope_deg": 0.0, "burial_depth_gu": 0.0},
    }


def build_castle_system_preview(inventory_path: Path) -> dict:
    """Render the complete extracted exterior CS_RE system in source layout."""
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = [
        row for row in inventory.get("references", [])
        if "cs_re" in str(row.get("model") or "").casefold()
        and row.get("category") in {"exterior", "door"}
    ]
    if not rows:
        raise ValueError("inventory has no exterior CS_RE references")
    anchor = [min(float(row["position"][i]) for row in rows) for i in range(3)]
    members = []
    for index, row in enumerate(rows):
        text = (str(row.get("model") or "") + " " + str(row.get("object_id") or "")).casefold()
        role = "door" if row.get("category") == "door" else "castle_component"
        if "wl" in text or "wc" in text:
            role = "castle_wall"
        elif "gh" in text:
            role = "castle_gatehouse"
        elif "twr" in text or "tw_" in text:
            role = "castle_tower"
        members.append({
            "source_id": row.get("source_id") or f"castle_{index:04d}",
            "object_id": row.get("object_id") or "",
            "model_key": row["model"],
            "record_type": row.get("record_type") or "STAT",
            "category": row.get("category") or "exterior",
            "is_door": row.get("category") == "door",
            "offset_gu": [float(row["position"][i]) - anchor[i] for i in range(3)],
            "rotation": list(row.get("rotation") or [0.0, 0.0, 0.0]),
            "scale": float(row.get("scale") or 1.0),
            "structural_role": role,
        })
    return {
        "stamp_id": "falkreath_complete_castle_cs_re_system",
        "source": {"kind": "inventory_source_layout", "family": "CS_RE", "reference_count": len(rows)},
        "building_type": "structural_preview", "size_class": "large", "door_count": sum(m["is_door"] for m in members),
        "multi_shell": False, "anchor": {"kind": "source_component", "source_position_gu": anchor},
        "access_heading_rad": 0.0, "members": members,
        "footprint": {"aabb_rel": {"min": [0, 0, 0], "max": [0, 0, 0], "span": [0, 0, 0]}, "hull_xy_rel": []},
        "bounds_rel_gu": {"min": [0, 0, 0], "max": [0, 0, 0], "span": [0, 0, 0]},
        "terrain_envelope": {"door_step_heights_gu": [], "footprint_relief_gu": 0.0, "footprint_slope_deg": 0.0, "burial_depth_gu": 0.0},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--orientation-test", action="store_true")
    parser.add_argument("--cell", nargs=2, type=int, metavar=("X", "Y"))
    parser.add_argument("--castle-system", action="store_true")
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    if args.castle_system:
        if not args.inventory:
            parser.error("--castle-system requires --inventory")
        stamp = build_castle_system_preview(args.inventory)
    elif args.cell:
        stamp = build_source_cell_preview(args.library, tuple(args.cell))
    elif args.orientation_test:
        stamp = build_wall_orientation_test(args.library)
    else:
        stamp = build_preview(args.library, ["wall", "gate", "fence"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json_bytes(stamp))
    print(f"wrote {args.out} members={len(stamp['members'])}")
    if not args.render:
        return 0
    sheet = args.out.with_name(args.out.stem + "_sheet_2x3.png")
    command = [
        sys.executable,
        str(WORKSPACE / "tools" / "cityforge" / "render_generated_house.py"),
        "--stamp", str(args.out), "--out", str(sheet),
    ]
    return subprocess.run(command, cwd=WORKSPACE, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
