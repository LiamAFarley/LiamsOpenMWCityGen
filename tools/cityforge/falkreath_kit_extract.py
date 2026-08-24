#!/usr/bin/env python3
"""Extract authoritative Falkreath modular-kit evidence from the xFa ESP set.

The five Falkreath reference layers are scanned after loading Sky_Main and
Tamriel_Data definitions, following the dependency-aware Markarth workflow.
Only named Falkreath exterior-cell construction references are emitted; source
plugins and meshes are never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))
from procgen.espscan import CellReference, scan_file  # noqa: E402
from procgen.meshcheck import AssetResolver  # noqa: E402

DEFAULT_SOURCE_DIR = WORKSPACE / "Extra Reference Mods" / "Sky_xFa_02_clean_20260509"
DEFAULT_SKY = WORKSPACE / "Sky_Main.esm"
DEFAULT_TD = Path(r"C:\Modding\OpenMWOverhaul\one-day-morrowind-modernization\ModdingResources\TamrielData\00 Data Files\Tamriel_Data.esm")
DEFAULT_OUTPUT = WORKSPACE / "output" / "cityforge" / "falkreath_kit_extraction_v1"
LAYER_NAMES = ("Sky_xFa_01.esp", "Sky_xFa_02_clean_20260509.esp", "Sky_xFa_034.esp", "Sky_xFa_04.esp", "Sky_xFa_05.esp")
CONSTRUCTION_CATEGORIES = frozenset({"exterior", "door", "interior"})
FAMILY_RE = re.compile(r"sky[_\\]fk[_\\]([a-z0-9_]+)", re.IGNORECASE)
HOUSE_RE = re.compile(r"sky[_\\]x[_\\]sky_fk_house_[0-9]+_[a-z]", re.IGNORECASE)
STAMP_ATTACH_RADIUS_GU = 2200.0
OPENING_ATTACH_RADIUS_GU = 1450.0
ATTACH_Z_DELTA_GU = 800.0


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def _source_id(path: Path, grid: tuple[int, int] | None, ref: CellReference) -> str:
    if grid is None:
        raise ValueError(f"reference {ref.object_id!r} has no exterior grid")
    return f"{path.stem}:{grid[0]}_{grid[1]}_ref_{ref.refr_index:06d}"


def _row(path: Path, cell: Any, ref: CellReference) -> dict[str, Any]:
    return {
        "source_id": _source_id(path, cell.grid, ref),
        "source_layer": path.name,
        "cell": {"grid": list(cell.grid or ()), "name": cell.name or ""},
        "object_id": ref.object_id,
        "record_type": ref.record_type,
        "model": ref.model,
        "category": ref.category,
        "building": bool(ref.building),
        "door_to_interior": bool(ref.door_to_interior),
        "position": list(ref.position or ()),
        "rotation": list(ref.rotation or ()),
        "scale": ref.scale,
    }


def _component_role(row: dict[str, Any]) -> str | None:
    """Classify an extracted reference by construction role.

    Role detection intentionally uses the resolved TES3 category and source
    object id as well as the mesh path.  Falkreath houses reuse generic Nord
    and castle doors/windows, so a kit-prefix-only classifier silently loses
    real openings.  Walls, gates, and fences are emitted into the structural
    library rather than being treated as house attachments.
    """
    model = str(row.get("model") or "")
    object_id = str(row.get("object_id") or "")
    text = (model + " " + object_id).replace("/", "\\").casefold()
    if HOUSE_RE.search(model.replace("/", "\\").casefold()):
        return "shell"
    if "trapdoor" in text or "_td_" in text:
        return "trapdoor"
    if row.get("category") == "door":
        return "door"
    if "window" in text or "_ww_" in text or "_win_" in text:
        return "window"
    if "dframe" in text or "doorframe" in text or "_df_" in text:
        return "doorframe"
    if "gatehouse" in text or "_gh_" in text or "gate" in text:
        return "gate"
    if "_wl_" in text or "wall" in text or "stonewall" in text:
        return "wall"
    if "fence" in text:
        return "fence"
    if "chimney" in text:
        return "chimney"
    if "dormer" in text:
        return "dormer"
    if "porch" in text:
        return "porch"
    if "tent" in text:
        return "tent"
    if "_str_" in text or "stair" in text or "wdstr" in text:
        return "stair"
    # Preserve non-structural building props (market/fish/well/etc.) as
    # source decorations.  They are attached by the same distance/Z gate as
    # tents and porches; walls, gates, and fences were classified above and
    # remain structural-library inputs instead.
    if row.get("building"):
        return "decoration"
    return None


def _source_stamp_library(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn extracted source refs into localizable single-shell stamp inputs.

    The source ESP records are world placements.  Stamps intentionally retain
    those placements here; the existing kit-house grammar converts members to
    shell-local coordinates while mining door slots and accessory bundles.
    """
    typed = [(row, _component_role(row)) for row in rows]
    shells = [(row, role) for row, role in typed if role == "shell"]
    attachable = [(row, role) for row, role in typed
                  if role in {"door", "doorframe", "window", "chimney", "dormer",
                              "porch", "tent", "stair", "decoration"}]
    shells_by_grid: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for shell, _role in shells:
        grid = tuple(int(value) for value in shell["cell"].get("grid") or ())
        if len(grid) == 2:
            shells_by_grid[grid].append(shell)
    assigned: defaultdict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for row, role in attachable:
        grid = tuple(int(value) for value in row["cell"].get("grid") or ())
        candidates = shells_by_grid.get(grid, [])
        if not candidates:
            continue
        x, y = (float(row["position"][0]), float(row["position"][1]))
        shell = min(
            candidates,
            key=lambda candidate: math.hypot(
                x - float(candidate["position"][0]),
                y - float(candidate["position"][1]),
            ),
        )
        distance = math.hypot(
            x - float(shell["position"][0]),
            y - float(shell["position"][1]),
        )
        z_delta = abs(float(row["position"][2]) - float(shell["position"][2]))
        max_distance = OPENING_ATTACH_RADIUS_GU if role in {"door", "doorframe"} else STAMP_ATTACH_RADIUS_GU
        if distance <= max_distance and z_delta <= ATTACH_Z_DELTA_GU:
            assigned[str(shell["source_id"])].append((row, role))
    stamps: list[dict[str, Any]] = []
    for shell, _role in shells:
        sx, sy, sz = (float(value) for value in shell["position"])
        members = [{
            "source_id": shell["source_id"],
            "object_id": shell["object_id"],
            "model_key": shell["model"],
            "record_type": "STAT",
            "category": "exterior",
            "is_door": False,
            "offset_gu": [sx, sy, sz],
            "rotation": list(shell.get("rotation") or (0.0, 0.0, 0.0)),
            "scale": float(shell.get("scale") or 1.0),
            "structural_role": "shell",
        }]
        for row, role in assigned.get(str(shell["source_id"]), []):
            x, y, z = (float(value) for value in row["position"])
            if math.hypot(x - sx, y - sy) > STAMP_ATTACH_RADIUS_GU:
                continue
            members.append({
                "source_id": row["source_id"],
                "object_id": row["object_id"],
                "model_key": row["model"],
                "record_type": "DOOR" if role == "door" else "STAT",
                "category": "door" if role == "door" else "exterior",
                "is_door": role == "door",
                "offset_gu": [x, y, z],
                "rotation": list(row.get("rotation") or (0.0, 0.0, 0.0)),
                "scale": float(row.get("scale") or 1.0),
                "structural_role": role,
            })
        if len(members) > 1:
            stamps.append({
                "stamp_id": f"fk_source__{shell['source_id']}",
                "building_type": "house",
                "size_class": "unknown",
                "door_count": sum(1 for member in members if member["is_door"]),
                "multi_shell": False,
                "members": members,
            })
    return {
        "schema_version": 1,
        "library_id": "falkreath_source_v1",
        "kit_id": "falkreath",
        "stamps": stamps,
        "source": {
            "kind": "falkreath_xfa_exterior_refs",
            "association_rule": (
                f"same exterior cell; house openings XY radius <= {OPENING_ATTACH_RADIUS_GU:.1f} GU, "
                f"other attachments <= {STAMP_ATTACH_RADIUS_GU:.1f} GU, "
                f"all attachment Z delta <= {ATTACH_Z_DELTA_GU:.1f} GU"
            ),
            "stamp_count": len(stamps),
        },
    }


def _structural_library(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Emit source-positioned wall/gate/fence palettes for city generation.

    These records remain world-positioned and traceable to their source cell;
    consumers can choose a wall or gate family and transform the whole local
    component without confusing it with a house shell attachment.
    """
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    role_counts: Counter[str] = Counter()
    for row in rows:
        role = _component_role(row)
        if role not in {"wall", "gate", "fence"}:
            continue
        role_counts[role] += 1
        groups[(role, str(row.get("model") or "").casefold())].append(row)
    components: list[dict[str, Any]] = []
    for (role, model_key), members in sorted(groups.items()):
        first = members[0]
        component_key = model_key.replace("\\", "_").replace("/", "_")
        components.append({
            "component_id": f"fk_{role}__{component_key}",
            "structural_role": role,
            "model_key": first.get("model"),
            "object_ids": sorted({str(row.get("object_id") or "") for row in members}),
            "source_count": len(members),
            "source_refs": [
                {
                    "source_id": row["source_id"],
                    "cell": row["cell"],
                    "position": row["position"],
                    "rotation": row.get("rotation") or [0.0, 0.0, 0.0],
                    "scale": float(row.get("scale") or 1.0),
                }
                for row in members
            ],
        })
    return {
        "schema_version": 1,
        "library_id": "falkreath_structural_components_v1",
        "kit_id": "falkreath",
        "roles": sorted(role_counts),
        "role_counts": dict(sorted(role_counts.items())),
        "components": components,
        "source": {
            "kind": "falkreath_xfa_exterior_refs",
            "classification": "TES3 category plus object/model construction tokens",
        },
    }


def run(source_dir: Path, sky: Path, td: Path, output: Path) -> None:
    definitions: dict[str, Any] = {}
    object_types: dict[str, str] = {}
    for path, source_kit in ((sky, "sky"), (td, "tr")):
        if not path.is_file():
            raise FileNotFoundError(f"missing definition plugin: {path}")
        result = scan_file(path, source_kit=source_kit, collect_cells=False, max_seconds=180.0,
                           initial_object_models=definitions, initial_object_types=object_types)
        definitions.update(result.object_models)
        object_types.update(result.object_types)

    paths = [source_dir / name for name in LAYER_NAMES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Falkreath source layer(s): " + ", ".join(map(str, missing)))

    rows: list[dict[str, Any]] = []
    scan_stats: dict[str, Any] = {}
    scanned: list[tuple[Path, Any]] = []
    for path in paths:
        result = scan_file(path, source_kit="sky", collect_cells=True, max_seconds=180.0,
                           initial_object_models=definitions, initial_object_types=object_types)
        scanned.append((path, result))

    named_grids = {
        cell.grid
        for _path, result in scanned
        for cell in result.cells
        if cell.grid is not None and "falkreath" in (cell.name or "").casefold()
    }
    if not named_grids:
        raise ValueError("no named Falkreath exterior cells found in the source layers")

    for path, result in scanned:
        # Later xFa patches commonly retain the city CELL grid but omit its
        # name.  Grid membership keeps those authoritative overrides in scope.
        selected = [cell for cell in result.cells if cell.grid in named_grids]
        selected_refs = [(cell, ref) for cell in selected for ref in cell.references
                         if ref.model and ref.category in CONSTRUCTION_CATEGORIES and ref.building]
        rows.extend(_row(path, cell, ref) for cell, ref in selected_refs)
        scan_stats[path.name] = {
            "exterior_cells": result.exterior_cells,
            "reference_count": result.reference_count,
            "resolved_mesh_reference_count": result.resolved_mesh_reference_count,
            "unresolved_reference_count": result.unresolved_reference_count,
            "selected_cells": len(selected),
            "selected_construction_refs": len(selected_refs),
        }

    rows.sort(key=lambda row: str(row["source_id"]).casefold())
    model_counts = Counter(str(row["model"]).casefold() for row in rows)
    resolver = AssetResolver()
    model_rows: list[dict[str, Any]] = []
    family_rows: defaultdict[str, list[str]] = defaultdict(list)
    for model_key, count in sorted(model_counts.items()):
        model = next(str(row["model"]) for row in rows if str(row["model"]).casefold() == model_key)
        match = FAMILY_RE.search(model)
        family = match.group(1).casefold() if match else "other"
        family_rows[family].append(model)
        resolved = resolver.resolve(model, "mesh")
        model_rows.append({"model": model, "count": count, "family": family,
                           "asset_path": str(resolved) if resolved else None,
                           "asset_resolved": resolved is not None})

    summary = {
        "schema": 1,
        "source_layers": [str(path) for path in paths],
        "selected_cell_rule": "grids discovered from named Falkreath exterior CELLs, applied to every xFa layer",
        "selected_grids": [list(grid) for grid in sorted(named_grids)],
        "construction_rule": "resolved model and category in exterior/door/interior with building=true",
        "definition_plugins": [str(sky), str(td)],
        "scan_stats": scan_stats,
        "selected_reference_count": len(rows),
        "unique_model_count": len(model_rows),
        "asset_resolved_model_count": sum(1 for row in model_rows if row["asset_resolved"]),
        "asset_unresolved_models": [row["model"] for row in model_rows if not row["asset_resolved"]],
        "component_role_counts": dict(sorted(Counter(
            role for row in rows
            for role in [_component_role(row)]
            if role is not None
        ).items())),
    }
    _json(output / "inventory.json", {"schema": 1, "references": rows})
    _json(output / "kit.json", {"schema": 1, "models": model_rows,
                                 "families": {key: sorted(value, key=str.casefold) for key, value in sorted(family_rows.items())},
                                 "source_summary": summary})
    _json(output / "stamp_library.json", _source_stamp_library(rows))
    _json(output / "structural_library.json", _structural_library(rows))
    _json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--sky", type=Path, default=DEFAULT_SKY)
    parser.add_argument("--tamriel-data", type=Path, default=DEFAULT_TD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.source_dir, args.sky, args.tamriel_data, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
