"""town_author.py — assemble the masterless town ESP records (Stage 5).

Purpose
-------
Build the tes3conv JSON ``PluginDoc`` for the settlement ``<name>_town.esp``:
edited LAND + its local LTEX table (from the city generation's edited-land
JSON), the seated building statics/refs (from the realization's seated
objects), and the scatter statics/refs (from the v6 scatter document) merged
into the same cells.  One plugin carries buildings, terrain, and scatter so
seeds, bounds and exclusions cannot drift apart.

Coordinate frames
-----------------
The seated objects' ``world_position_gu`` / ``world_cell`` are in the city's
PLAN frame (origin at the settlement's frame-origin cell, e.g. (-95,-11));
the TES3 ESP needs global game units and global cells, so every ref is
converted ``world = plan + frame_origin_gu`` and each CELL record is keyed by
``floor(world / 8192)``. REFR POS1 translations remain global game units, as
verified against the working scatter and groundcover plugins.
Rotations are written verbatim from the seated objects' ``raw_tes3_rotation_rad``
(the final TES3 Euler triple under the engine composition).

Doors
-----
All seated objects (STAT and DOOR record types) are authored as STAT refs
pointing at their model.  Door travel data (DODT/DNAM interior links) is out of
scope (no interiors/quest content per the integration plan); a door renders as
a static until the interior authoring stage.

Pipeline position
-----------------
Approved plan ``.opencode/runs/cityforge-scatter-groundcover-integration/
plan.md`` Stage 5, step 4.  Consumed by ``tools/generate_settlement_wilderness.py``.
The plugin is authored masterless (``Header.masters: []``) per workspace rule
#24/#9; ``tamriel.esm`` is loaded separately by the user.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .tes3json import (
    JsonObject,
    PluginDoc,
    build_cell,
    build_reference,
    build_static,
    new_plugin,
)

BUILDING_STAT_PREFIX = "PTSB_"
WALL_STAT_PREFIX = "PTWL_"
CELL_SIZE_GU = 8192.0


def _frame_origin_gu(town_placements: Mapping[str, Any]) -> tuple[float, float]:
    """Plan-frame origin in world GU from the seated/realization document."""
    tf = town_placements.get("terrain_field")
    if isinstance(tf, Mapping):
        origin = tf.get("frame_origin_gu")
        if isinstance(origin, (list, tuple)) and len(origin) >= 2:
            return float(origin[0]), float(origin[1])
    origin = town_placements.get("frame_origin_gu")
    if isinstance(origin, (list, tuple)) and len(origin) >= 2:
        return float(origin[0]), float(origin[1])
    raise ValueError("town placements carry no frame_origin_gu (terrain_field.frame_origin_gu)")


def author_building_records(
    objects: Iterable[Mapping[str, Any]],
    *,
    frame_origin_gu: Sequence[float],
) -> tuple[list[JsonObject], dict[tuple[int, int], list[JsonObject]], dict[str, str]]:
    """Build building STAT records and per-global-cell refs from seated objects.

    Returns ``(stat_records, refs_by_grid, mesh_to_stat)``.  ``refs_by_grid``
    values are dicts ready for ``build_cell``; ``refr_index`` is renumbered by
    the caller after scatter refs are merged.
    """
    origin_x, origin_y = (float(frame_origin_gu[0]), float(frame_origin_gu[1]))
    rows = [row for row in objects if isinstance(row, Mapping)]
    mesh_keys = sorted({str(row["model_key"]) for row in rows if row.get("model_key")})
    mesh_to_stat = {
        key: f"{BUILDING_STAT_PREFIX}{index:04d}" for index, key in enumerate(mesh_keys)
    }
    stat_records = [build_static(mesh_to_stat[key], key) for key in mesh_keys]

    refs_by_grid: dict[tuple[int, int], list[JsonObject]] = {}
    for row in sorted(rows, key=lambda r: str(r.get("object_id") or r.get("reference_id") or "")):
        model_key = row.get("model_key")
        position = row.get("world_position_gu")
        rotation = row.get("raw_tes3_rotation_rad")
        if not model_key or not position or len(position) < 3 or not rotation:
            raise ValueError(
                f"seated object {row.get('object_id')!r} missing model_key/"
                "world_position_gu/raw_tes3_rotation_rad"
            )
        plan_x, plan_y, pos_z = (float(value) for value in position[:3])
        world_x = plan_x + origin_x
        world_y = plan_y + origin_y
        grid_x = math.floor(world_x / CELL_SIZE_GU)
        grid_y = math.floor(world_y / CELL_SIZE_GU)
        scale = float(row.get("scale") or 1.0)
        reference = build_reference(
            mesh_to_stat[str(model_key)],
            0,
            translation=(world_x, world_y, pos_z),
            rotation=[float(value) for value in rotation],
            mast_index=0,
            temporary=True,
            scale=scale,
        )
        refs_by_grid.setdefault((grid_x, grid_y), []).append(reference)
    return stat_records, refs_by_grid, mesh_to_stat


def merge_scatter_refs(
    refs_by_grid: dict[tuple[int, int], list[JsonObject]],
    scatter_document: Mapping[str, Any],
) -> list[JsonObject]:
    """Author scatter refs from the v6 scatter document with cell-local positions.

    Returns the scatter STAT records.  The v6 scatter document stores refs with
    GLOBAL TES3 game-unit positions (``cell * 8192 + local``); TES3 REFR
    translations must be cell-local, so every ref is converted before it is
    appended to ``refs_by_grid`` (cells are created if the town has no
    buildings there).  This deliberately does NOT reuse
    ``scatter_author.build_scatter_plugin`` verbatim, whose standalone path
    copies the global position as the translation.
    """
    from .scatter_author import allocate_stat_ids

    density = scatter_document.get("density")
    cells = density.get("cells") if isinstance(density, Mapping) else []
    if not isinstance(cells, list):
        raise ValueError("scatter document has no density.cells list")

    mesh_values = [
        str(ref["mesh"])
        for cell in cells
        if isinstance(cell, Mapping)
        for ref in (cell.get("refs") or [])
        if isinstance(ref, Mapping) and ref.get("mesh")
    ]
    mesh_to_stat = allocate_stat_ids(mesh_values)
    id_to_mesh: dict[str, str] = {}
    for mesh in mesh_values:
        id_to_mesh.setdefault(mesh_to_stat[mesh.casefold()], mesh)
    stat_records = [
        build_static(stat_id, id_to_mesh[stat_id]) for stat_id in sorted(id_to_mesh)
    ]

    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        grid_raw = cell.get("grid")
        if not isinstance(grid_raw, (list, tuple)) or len(grid_raw) != 2:
            continue
        grid_x, grid_y = int(grid_raw[0]), int(grid_raw[1])
        for ref in cell.get("refs") or []:
            if not isinstance(ref, Mapping):
                continue
            mesh = ref.get("mesh")
            position = ref.get("position_gu")
            rotation = ref.get("rotation_radians")
            if not mesh or not position or len(position) < 3:
                continue
            plan_x, plan_y, pos_z = (float(value) for value in position[:3])
            scale = float(ref.get("scale") or 1.0)
            reference = build_reference(
                mesh_to_stat[str(mesh).casefold()],
                0,
                translation=(plan_x, plan_y, pos_z),
                rotation=[float(value) for value in rotation] if rotation else None,
                mast_index=0,
                temporary=True,
                scale=scale,
            )
            refs_by_grid.setdefault((grid_x, grid_y), []).append(reference)
    return stat_records


def author_wall_records(
    wall_document: Mapping[str, Any],
    *,
    frame_origin_gu: Sequence[float],
) -> tuple[list[JsonObject], dict[tuple[int, int], list[JsonObject]]]:
    """Author composed wall members into the same world/cell contract."""

    members = wall_document.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("wall document has no members")
    origin = wall_document.get("origin_gu") or [0.0, 0.0]
    if not isinstance(origin, (list, tuple)) or len(origin) < 2:
        raise ValueError("wall document has no origin_gu")
    mesh_keys = sorted({str(row["model_key"]) for row in members if row.get("model_key")})
    mesh_to_stat = {
        key: f"{WALL_STAT_PREFIX}{index:04d}" for index, key in enumerate(mesh_keys)
    }
    stat_records = [build_static(mesh_to_stat[key], key) for key in mesh_keys]
    origin_x, origin_y = float(origin[0]), float(origin[1])
    frame_x, frame_y = float(frame_origin_gu[0]), float(frame_origin_gu[1])
    refs_by_grid: dict[tuple[int, int], list[JsonObject]] = {}
    for index, row in enumerate(
        sorted(members, key=lambda r: str(r.get("object_id") or r.get("source_id") or ""))
    ):
        model_key = row.get("model_key")
        offset = row.get("offset_gu")
        rotation = row.get("rotation")
        if not model_key or not isinstance(offset, list) or len(offset) < 3:
            raise ValueError(f"wall member {row.get('source_id')!r} lacks model/offset")
        if not isinstance(rotation, list) or len(rotation) < 3:
            raise ValueError(f"wall member {row.get('source_id')!r} lacks rotation")
        world_x = origin_x + float(offset[0]) + frame_x
        world_y = origin_y + float(offset[1]) + frame_y
        world_z = float(offset[2])
        grid_x = math.floor(world_x / CELL_SIZE_GU)
        grid_y = math.floor(world_y / CELL_SIZE_GU)
        reference = build_reference(
            mesh_to_stat[str(model_key)],
            0,
            translation=(world_x, world_y, world_z),
            rotation=[float(value) for value in rotation[:3]],
            mast_index=0,
            temporary=True,
            scale=float(row.get("scale") or 1.0),
        )
        refs_by_grid.setdefault((grid_x, grid_y), []).append(reference)
    return stat_records, refs_by_grid


def _renumber_cell_refs(cells: Mapping[tuple[int, int], list[JsonObject]]) -> list[JsonObject]:
    """Sort refs per cell and assign deterministic sequential ``refr_index``."""
    cell_records: list[JsonObject] = []
    for grid in sorted(cells):
        refs = sorted(
            cells[grid],
            key=lambda ref: (
                str(ref.get("id")),
                tuple(float(v) for v in (ref.get("translation") or ())),
            ),
        )
        for index, ref in enumerate(refs, start=1):
            ref["refr_index"] = index
        cell_records.append(build_cell("", list(grid), references=refs))
    return cell_records


def build_town_plugin(
    *,
    edited_land_doc: Sequence[Mapping[str, Any]],
    town_placements: Mapping[str, Any],
    scatter_document: Mapping[str, Any],
    wall_document: Mapping[str, Any] | None = None,
    description: str,
    author: str = "ProcGen",
) -> PluginDoc:
    """Assemble the masterless town plugin records (see module docstring)."""

    objects = town_placements.get("objects")
    if not isinstance(objects, list):
        raise ValueError("town placements has no 'objects' list")
    frame_origin = _frame_origin_gu(town_placements)

    records: PluginDoc = new_plugin(
        {
            "author": author,
            "description": description,
            "file_type": "Esp",
            "num_objects": 0,
            "masters": [],
        }
    )
    records.extend(
        dict(record)
        for record in edited_land_doc
        if isinstance(record, Mapping) and record.get("type") == "LandscapeTexture"
    )
    records.extend(
        dict(record)
        for record in edited_land_doc
        if isinstance(record, Mapping) and record.get("type") == "Landscape"
    )

    building_stats, refs_by_grid, _mesh_to_stat = author_building_records(
        objects, frame_origin_gu=frame_origin
    )
    records.extend(building_stats)
    if wall_document is not None:
        wall_stats, wall_refs = author_wall_records(
            wall_document, frame_origin_gu=frame_origin
        )
        records.extend(wall_stats)
        for grid, refs in wall_refs.items():
            refs_by_grid.setdefault(grid, []).extend(refs)
    scatter_stats = merge_scatter_refs(refs_by_grid, scatter_document)
    records.extend(scatter_stats)
    records.extend(_renumber_cell_refs(refs_by_grid))

    records[0]["num_objects"] = len(records) - 1
    if records[0]["num_objects"] != (
        sum(1 for record in records if record.get("type") in {"Static", "Cell"})
        + sum(1 for record in records if record.get("type") in {"Landscape", "LandscapeTexture"})
    ):
        raise ValueError("town plugin Header.num_objects does not match authored records")
    return records
