"""Cityforge D-BRIEF region-palette and LAND/VTEX census engine (T0.5).

Pipeline position
------------------
Consumes the read-only plugins (``tamriel.esm``, the masterless
``falkreath_landscape_texture_remap.esp``, and the SHOTN scratch
``Sky_Main.esm`` for the Karthgad core reproduction), the planning polygons
(``output/mapdata/regions.json``), the accepted site survey
(``output/cityforge/sites/falkreath_v1/site_survey.json``), the corrected
road centerline bundle, and the Vorndgad ecology proxies, and produces
``region_palette.json`` plus the land/water/road sections of ``census.json``.
The CLI ``tools/cityforge/build_city_brief.py`` orchestrates everything.

Invariants
----------
- Two ground scopes are never merged: the R072 PTR planning polygon (191
  cells from ``regions.json``, censused in ``tamriel.esm``) and the effective
  Falkreath target block (49 cells from ``site_survey.json``, censused in the
  remap ESP over the base ESM).  R072 membership is rasterized from the
  polygon at integer cell centers and must reproduce the declared
  ``cell_count`` exactly.
- VTEX/LTEX resolution is the workspace-validated **internal/toolchain
  convention** pinned by ``src/procgen/espland.py`` and the remap round-trip
  evidence (the remap ESP's LAND payloads resolve through its own 4-record
  LTEX table to the identities recorded in
  ``falkreath_landscape_texture_remap_report.json``): raw VTEX ``0`` is the
  base sentinel (never an LTEX record); raw ``N > 0`` resolves through the
  LAND-owning plugin's local LTEX index ``N - 1``.  **OpenMW 0.51 internal
  API confirmation is unavailable from the connected openmw-docs index (no
  LTEX/VTEX resolution coverage); verify against the engine source
  (``terrainstorage``/``ESMStore`` LandTexture handling) before Phase 1
  authoring.**  Identity labels are therefore load-order sensitive: R072
  ``base_textures`` labels describe base-ESM-only load order; with the remap
  ESP loaded, raw 33/78/92 render under the remap identities (see the
  ``load_order_caveats`` emitted in the palette).
- Road: raw 78 is the only protected source identity (``MA_sulphur_rock02``
  in the base ESM; ``T_Hr_TerrRoadOH_01`` in the remap output); raw 1 Sand is
  never road.  Direct LAND/VTEX raw-78 tiles are the in-game occupancy
  authority; the aligned consumer product
  (``tamriel_aligned_centerlines_v1.json``) is the only vector view planners
  consume, through ``src/procgen/aligned_roads.py``.  The source-space
  bundle and the XCF/BMP are provenance only and never planner inputs.
- Tile fractions sum exactly to each scope's tile total; unknown surface
  references fail closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import espland
from .censusio import PinnedFile, quantile_summary

TAMRIEL_ESM = "tamriel.esm"
REMAP_ESP = "output/falkreath_landscape_texture_remap.esp"
REMAP_REPORT = "output/falkreath_landscape_texture_remap_report.json"
SKY_MAIN_ESM = "Sky_Main.esm"
REGIONS_JSON = "output/mapdata/regions.json"
SITE_SURVEY = "output/cityforge/sites/falkreath_v1/site_survey.json"
TERRAIN_CELLS = "output/terrain_cells.json"
GROUND_RULES = "output/skyrim_ground_rules.json"
GROUNDCOVER_INI = "configs/groundcover_falkreath_v1_currenttextures.ini"
SCATTER_ANALYSIS = "output/vorndgad_scatter_analysis.json"
CLIFF_ANALYSIS = "output/vorndgad_cliff_analysis.json"
CENTERLINES_JSON = (
    "output/mapdata/roads/tamriel_aligned_centerlines_v1/"
    "tamriel_aligned_centerlines_v1.json"
)
CENTERLINES_SOURCE_JSON = (
    "output/mapdata/roads/tamriel_source_centerlines_v1/"
    "tamriel_road_centerlines_v1.json"
)
CENTERLINES_METADATA = (
    "output/mapdata/roads/tamriel_source_centerlines_v1/source_metadata.json"
)

SURVEY_REGION = (
    ".opencode/runs/karthgad-city-authoring/"
    "2026-08-04_region_palette_and_siting_survey.md"
)

REGION_ID = "R072"
REGION_NAME = "KREATHI DALE"

#: The remap ESP's expected local LTEX table (index -> (record id, texture
#: path)), validated against the remap report.  This constant is only the
#: EXPECTATION: the census cross-checks it against the live table read from
#: the pinned remap ESP via ``espland.load_ltex`` (``live_remap_ltex_table``)
#: and fails closed on any divergence (see ``crosscheck_live_remap_table``).
REMAP_LTEX_TABLE: dict[int, tuple[str, str]] = {
    0: ("Sand", "Tx_sand_01.tga"),
    32: ("T_Sky_TerrGrassRE_01", "Tx_Skyrim_grass_03.dds"),
    77: ("T_Hr_TerrRoadOH_01", r"hr\lnd\hr_oh_road_01.dds"),
    91: ("T_Sky_TerrPine_01", "Tx_Skyrim_pineneedles_01.dds"),
}


def live_remap_ltex_table(root: str | Path) -> dict[str, Any]:
    """Load the live remap ESP LTEX table (measured, hash-pinned).

    This is the authoritative cross-check source for the authoring
    assignment contract: the constant ``REMAP_LTEX_TABLE`` is only an
    expectation, and every gate below compares against the actual records
    read from the pinned ``falkreath_landscape_texture_remap.esp``.
    """
    path = Path(root) / REMAP_ESP
    records = espland.load_ltex(path)
    pin = PinnedFile(path)
    return {
        "esp_path": REMAP_ESP,
        "esp_sha256": pin.sha256(),
        "record_count": len(records),
        "records": [
            {
                "ltex_index": record.index,
                "ltex_id": record.record_id,
                "texture_path": record.file_name,
                "record_index": record.record_index,
                "record_offset": record.record_offset,
            }
            for record in records.values()
        ],
    }


def crosscheck_live_remap_table(live: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fail-closed checks of the live remap ESP LTEX table.

    The live table must contain exactly the expected indices {0, 32, 77,
    91}, each with the expected record id AND texture path; index 77 (road)
    is protected and checked separately.  Any divergence is a hard gate
    failure.
    """
    checks: list[dict[str, Any]] = []
    records = {record["ltex_index"]: record for record in live["records"]}
    live_indices = set(records)
    expected_indices = set(REMAP_LTEX_TABLE)
    checks.append({
        "id": "remap.live_table_coverage",
        "passed": live_indices == expected_indices,
        "detail": f"live indices {sorted(live_indices)} == expected "
                  f"{sorted(expected_indices)}",
    })
    for index in sorted(expected_indices):
        expected_id, expected_path = REMAP_LTEX_TABLE[index]
        record = records.get(index)
        present = record is not None
        id_ok = present and record["ltex_id"] == expected_id
        path_ok = present and record["texture_path"] == expected_path
        checks.append({
            "id": f"remap.live_identity.index{index}",
            "passed": present and id_ok and path_ok,
            "detail": (
                f"index {index}: live="
                f"{record['ltex_id'] if present else None}/"
                f"{record['texture_path'] if present else None} expected="
                f"{expected_id}/{expected_path}"
            ),
        })
    road = records.get(77)
    checks.append({
        "id": "remap.road77_live_protected",
        "passed": (road is not None
                   and road["ltex_id"] == "T_Hr_TerrRoadOH_01"
                   and road["texture_path"] == r"hr\lnd\hr_oh_road_01.dds"),
        "detail": (
            f"live index 77 = {road['ltex_id'] if road else None}/"
            f"{road['texture_path'] if road else None} (must be "
            f"T_Hr_TerrRoadOH_01 / hr\\lnd\\hr_oh_road_01.dds)"
        ),
    })
    return checks


def planned_vs_live_remap_check(
    surfaces: Iterable[Mapping[str, Any]],
    live: Mapping[str, Any],
) -> dict[str, Any]:
    """Planned assignments must agree with the live remap table on every
    shared index, and planned settlement indices must not collide with any
    live record (the settlement classes are Sky_Main-measured and have no
    remap identity)."""
    planned_by_index = {
        assignment["planned_ltex_index"]: assignment["planned_ltex_id"]
        for surface in surfaces
        if isinstance(assignment := surface.get("planned_assignment"), dict)
    }
    live_by_index = {
        record["ltex_index"]: record["ltex_id"] for record in live["records"]
    }
    shared = sorted(set(planned_by_index) & set(live_by_index))
    mismatches = [
        f"index {index}: planned {planned_by_index[index]} vs live "
        f"{live_by_index[index]}"
        for index in shared
        if planned_by_index[index] != live_by_index[index]
    ]
    settlement_indices = sorted(
        index for index in planned_by_index if index not in set(REMAP_LTEX_TABLE)
    )
    stray = sorted(index for index in settlement_indices if index in live_by_index)
    passed = not mismatches and not stray
    detail = (
        f"shared indices {shared} agree; settlement indices "
        f"{settlement_indices} have no live remap collision"
        if passed else
        "; ".join(mismatches + [f"settlement index {i} collides with live "
                                f"remap record" for i in stray])
    )
    return {
        "id": "authoring.planned_vs_live_remap",
        "passed": passed,
        "detail": detail,
    }

#: Closed semantic-surface vocabulary for Phase 1 plan/place/scape/mask
#: stages.  Each entry carries:
#: - ``measured_identity``: the measured source/remap raw + LTEX identity
#:   (raw 0 is never used; every surface raw is > 0),
#: - ``planned_assignment``: the explicit authoring contract for the future
#:   masterless city output plugin — ``planned_raw_vtex``,
#:   ``planned_ltex_index`` (= raw - 1), ``planned_ltex_id``, and the plugin
#:   scope/ownership.  ``planned_raw_vtex`` is NEVER inferred from any
#:   ordinal; the ordinal (``surface_ordinal``) is a pure enumeration label
#:   that must not be used for raw derivation.
#: The measured identities follow the review-corrected contract: remap
#: identities where present (Sand 1/0, base grass 33/32, protected road
#: 78/77) and the measured Sky_Main identities for settlement dirt/
#: grass-dirt/cobble (241/240, 142/141, 144/143; verified against
#: ``census.json#land.karthgad_core_reproduction``).
PHASE1_SURFACES: list[dict[str, Any]] = [
    {
        "surface": "base",
        "surface_ordinal": 0,
        "ordinal_note": "pure enumeration label; never derive raw_vtex from "
                        "this ordinal (raw is never ordinal + 1)",
        "measured_identity": {
            "raw_vtex": 33,
            "ltex_index": 32,
            "ltex_id": "AI_Grass",
            "scope": "tamriel.esm local LTEX (base-ESM-only load order)",
            "remap_identity": {
                "ltex_index": 32,
                "ltex_id": "T_Sky_TerrGrassRE_01",
                "scope": "falkreath_landscape_texture_remap.esp local LTEX "
                         "(remap load order)",
            },
        },
        "planned_assignment": {
            "planned_raw_vtex": 33,
            "planned_ltex_index": 32,
            "planned_ltex_id": "T_Sky_TerrGrassRE_01",
            "plugin_scope": "masterless city output plugin (masters: [])",
            "local_ltex_record_required": True,
        },
    },
    {
        "surface": "settlement_dirt",
        "surface_ordinal": 1,
        "ordinal_note": "pure enumeration label; never derive raw_vtex from "
                        "this ordinal",
        "measured_identity": {
            "raw_vtex": 241,
            "ltex_index": 240,
            "ltex_id": "T_Sky_TerrDirtRE_01",
            "scope": "Sky_Main.esm local LTEX (measured at Karthgad cell "
                     "(-102,11))",
            "provenance": "census.json#land.karthgad_core_reproduction",
        },
        "planned_assignment": {
            "planned_raw_vtex": 241,
            "planned_ltex_index": 240,
            "planned_ltex_id": "T_Sky_TerrDirtRE_01",
            "plugin_scope": "masterless city output plugin (masters: [])",
            "local_ltex_record_required": True,
        },
    },
    {
        "surface": "settlement_grass_dirt",
        "surface_ordinal": 2,
        "ordinal_note": "pure enumeration label; never derive raw_vtex from "
                        "this ordinal",
        "measured_identity": {
            "raw_vtex": 142,
            "ltex_index": 141,
            "ltex_id": "T_Sky_TerrGrassDirtRE_01",
            "scope": "Sky_Main.esm local LTEX (measured at Karthgad cell "
                     "(-102,11))",
            "provenance": "census.json#land.karthgad_core_reproduction",
        },
        "planned_assignment": {
            "planned_raw_vtex": 142,
            "planned_ltex_index": 141,
            "planned_ltex_id": "T_Sky_TerrGrassDirtRE_01",
            "plugin_scope": "masterless city output plugin (masters: [])",
            "local_ltex_record_required": True,
        },
    },
    {
        "surface": "settlement_cobble",
        "surface_ordinal": 3,
        "ordinal_note": "pure enumeration label; never derive raw_vtex from "
                        "this ordinal",
        "measured_identity": {
            "raw_vtex": 144,
            "ltex_index": 143,
            "ltex_id": "T_Nor_Set_TxCobbleStone_01",
            "scope": "Sky_Main.esm local LTEX (measured at Karthgad cell "
                     "(-102,11))",
            "provenance": "census.json#land.karthgad_core_reproduction",
        },
        "planned_assignment": {
            "planned_raw_vtex": 144,
            "planned_ltex_index": 143,
            "planned_ltex_id": "T_Nor_Set_TxCobbleStone_01",
            "plugin_scope": "masterless city output plugin (masters: [])",
            "local_ltex_record_required": True,
        },
    },
    {
        "surface": "road",
        "surface_ordinal": 4,
        "ordinal_note": "pure enumeration label; never derive raw_vtex from "
                        "this ordinal",
        "measured_identity": {
            "raw_vtex": 78,
            "ltex_index": 77,
            "ltex_id": "MA_sulphur_rock02",
            "scope": "tamriel.esm local LTEX (base-ESM-only load order)",
            "remap_identity": {
                "ltex_index": 77,
                "ltex_id": "T_Hr_TerrRoadOH_01",
                "scope": "falkreath_landscape_texture_remap.esp local LTEX "
                         "(remap load order)",
            },
        },
        "planned_assignment": {
            "planned_raw_vtex": 78,
            "planned_ltex_index": 77,
            "planned_ltex_id": "T_Hr_TerrRoadOH_01",
            "plugin_scope": "masterless city output plugin (masters: [])",
            "local_ltex_record_required": True,
        },
        "protected_source_identity": True,
    },
    {
        "surface": "water_edge_sand",
        "surface_ordinal": 5,
        "ordinal_note": "pure enumeration label; never derive raw_vtex from "
                        "this ordinal",
        "measured_identity": {
            "raw_vtex": 1,
            "ltex_index": 0,
            "ltex_id": "Sand",
            "scope": "tamriel.esm local LTEX (base-ESM-only load order)",
            "remap_identity": {
                "ltex_index": 0,
                "ltex_id": "Sand",
                "scope": "falkreath_landscape_texture_remap.esp local LTEX "
                         "(remap load order)",
            },
        },
        "planned_assignment": {
            "planned_raw_vtex": 1,
            "planned_ltex_index": 0,
            "planned_ltex_id": "Sand",
            "plugin_scope": "masterless city output plugin (masters: [])",
            "local_ltex_record_required": True,
        },
    },
]


def validate_authoring_assignments(
    surfaces: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Closed checks over the planned authoring assignment contract.

    Every check must pass for the vocabulary to be consumable by Phase 1:
    - each surface carries an explicit ``planned_assignment`` (never an
      abstract ordinal that invites ``raw = ordinal + 1`` inference),
    - ``planned_raw_vtex`` values are unique,
    - ``planned_ltex_index == planned_raw_vtex - 1`` for every raw > 0
      (espland convention), and the planned id matches the measured id,
    - road is exactly raw 78 / index 77 / ``T_Hr_TerrRoadOH_01`` (the
      protected identity is never reassigned),
    - no planned LTEX index collides with the remap ESP table unless the
      identity at that index is identical,
    - every emitted raw > 0 assignment requires a local LTEX record in the
      planned masterless output plugin, and the required table covers every
      emitted raw.
    Returns a list of {id, passed, detail} check records (all must pass).
    """
    records: list[dict[str, Any]] = []
    surface_list = list(surfaces)
    raws = []
    for surface in surface_list:
        surface_id = surface.get("surface")
        assignment = surface.get("planned_assignment")
        records.append({
            "id": f"authoring.{surface_id}.explicit_assignment",
            "passed": isinstance(assignment, dict),
            "detail": "planned_assignment present" if isinstance(assignment, dict)
                      else "planned_assignment missing (abstract ordinal "
                           "contract rejected)",
        })
        if not isinstance(assignment, dict):
            continue
        planned_raw = assignment.get("planned_raw_vtex")
        planned_index = assignment.get("planned_ltex_index")
        planned_id = assignment.get("planned_ltex_id")
        raw_ok = isinstance(planned_raw, int) and planned_raw > 0
        index_ok = isinstance(planned_index, int) and planned_raw == planned_index + 1
        records.append({
            "id": f"authoring.{surface_id}.raw_index_consistency",
            "passed": raw_ok and index_ok,
            "detail": f"planned_raw={planned_raw} planned_index={planned_index} "
                      f"(index must equal raw - 1)",
        })
        if raw_ok:
            raws.append((surface_id, planned_raw, planned_index, planned_id))

    unique = len({raw for _, raw, _, _ in raws}) == len(raws)
    records.append({
        "id": "authoring.raw_uniqueness",
        "passed": unique,
        "detail": f"planned raw values: {sorted(raw for _, raw, _, _ in raws)}",
    })
    road = next((s for s in surface_list if s.get("surface") == "road"), None)
    road_raw = road.get("planned_assignment", {}).get("planned_raw_vtex")
    road_index = road.get("planned_assignment", {}).get("planned_ltex_index")
    road_id = road.get("planned_assignment", {}).get("planned_ltex_id")
    records.append({
        "id": "authoring.road_protected",
        "passed": road_raw == 78 and road_index == 77
                  and road_id == "T_Hr_TerrRoadOH_01",
        "detail": f"road planned raw/index/id = {road_raw}/{road_index}/{road_id} "
                  f"(must be 78/77/T_Hr_TerrRoadOH_01)",
    })
    collisions = []
    for _, planned_index, planned_id in [(r[0], r[2], r[3]) for r in raws]:
        remap_entry = REMAP_LTEX_TABLE.get(planned_index)
        if remap_entry is not None and remap_entry[0] != planned_id:
            collisions.append(f"index {planned_index}: remap {remap_entry[0]} vs "
                              f"planned {planned_id}")
    records.append({
        "id": "authoring.no_remap_collision",
        "passed": not collisions,
        "detail": "no planned LTEX index collides with a different remap "
                  "identity" if not collisions else "; ".join(collisions),
    })
    required = sorted({raw for _, raw, _, _ in raws})
    # direct re-derivation: every planned index must appear in the emitted
    # required-local-LTEX inventory below
    emitted_indices = sorted(
        a["planned_ltex_index"]
        for s in surface_list
        if isinstance(a := s.get("planned_assignment"), dict)
    )
    records.append({
        "id": "authoring.local_ltex_coverage",
        "passed": emitted_indices == sorted(raw - 1 for raw in required)
                  and len(emitted_indices) == len(set(emitted_indices)),
        "detail": f"required local LTEX indices for masterless output "
                  f"plugin: {emitted_indices} (one record per emitted raw > 0)",
    })
    return records


class RegionPaletteError(RuntimeError):
    """Hard failure of a palette/census stage (abort, never degrade)."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegionPaletteError(message)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _point_in_polygon(
    p: tuple[float, float], polygon: Sequence[Sequence[float]]
) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > p[1]) != (yj > p[1])) and (
            p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def rasterize_region_cells(region: Mapping[str, Any]) -> list[tuple[int, int]]:
    """Enumerate the integer cells inside a planning polygon.

    Cell centers are placed with the documented map transform
    (``cell center px = ((x+251+0.5)*64, (59-y+0.5)*64)``) and tested with
    point-in-polygon ray casting over the polygon's bounding box.  The count
    must reproduce the polygon's declared ``cell_count`` exactly, otherwise
    the census aborts (a mismatch would mean the region definition changed).
    """
    bbox = region["bbox_cells"]
    min_x, min_y, max_x, max_y = (int(v) for v in bbox)
    polygon = region["polygon_map_px"]
    cells: list[tuple[int, int]] = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            center = ((x + 251 + 0.5) * 64.0, (59 - y + 0.5) * 64.0)
            if _point_in_polygon(center, polygon):
                cells.append((x, y))
    declared = int(region["cell_count"])
    _require(len(cells) == declared,
             f"rasterized {region['region_id']} gives {len(cells)} cells, "
             f"declared cell_count is {declared}")
    return cells


def load_region(root: str | Path, region_id: str = REGION_ID) -> dict[str, Any]:
    payload = _load_json(Path(root) / REGIONS_JSON)
    matches = [r for r in payload["regions"] if r["region_id"] == region_id]
    _require(len(matches) == 1,
             f"regions.json: expected exactly one {region_id}, got {len(matches)}")
    return matches[0]


def load_target_cells(root: str | Path) -> list[tuple[int, int]]:
    """The 49 effective Falkreath target cells from the accepted site survey."""
    survey = _load_json(Path(root) / SITE_SURVEY)
    cells = [tuple(int(v) for v in cell["grid"]) for cell in survey["cells"]]
    _require(len(cells) == 49,
             f"site_survey.json: expected 49 target cells, got {len(cells)}")
    _require(len(set(cells)) == 49, "site_survey.json: duplicate target cells")
    return cells


def census_land_records(
    records: Mapping[tuple[int, int], espland.LandRecord],
    ltex: Mapping[int, espland.LandscapeTexture],
    cells: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Census normalized LAND VTEX tiles for one cell set over already-loaded
    records and the LAND-owning plugin's LTEX table.

    Used by ``census_land_scope`` and by fixture-based tests; the plugin-local
    semantics live here: raw 0 is the base sentinel (never an LTEX record),
    raw ``N > 0`` resolves through the owning plugin's LTEX index ``N - 1``.
    Every tile of every cell must be counted; a missing LAND record or an
    unresolvable raw value is a hard failure.
    """
    raw_counts: dict[int, int] = {}
    base_sentinel_tiles = 0
    cell_totals: dict[str, int] = {}
    missing: list[list[int]] = []
    no_vtex_cells: list[list[int]] = []
    for cell in cells:
        key = tuple(int(v) for v in cell)
        record = records.get(key)
        if record is None:
            missing.append([key[0], key[1]])
            continue
        if not record.has_textures:
            no_vtex_cells.append([key[0], key[1]])
            base_sentinel_tiles += 256
            cell_totals[f"{key[0]},{key[1]}"] = 256
            continue
        counts: dict[int, int] = {}
        for raw in record.texture_indices:
            counts[raw] = counts.get(raw, 0) + 1
        total = 0
        for raw, count in counts.items():
            raw_counts[raw] = raw_counts.get(raw, 0) + count
            if raw == 0:
                base_sentinel_tiles += count
            total += count
        _require(total == 256, f"LAND {key}: tile total {total} != 256")
        cell_totals[f"{key[0]},{key[1]}"] = total
    _require(not missing,
             f"cells without LAND records: {missing[:10]}")
    _require(not no_vtex_cells,
             f"cells without VTEX payload: {no_vtex_cells[:10]}")

    resolved: dict[str, dict[str, Any]] = {}
    for raw, count in sorted(raw_counts.items()):
        index = espland.resolve_vtex_to_ltex_index(raw)
        if index is None:
            resolved["0"] = {
                "raw_vtex": 0,
                "class": "base_sentinel",
                "ltex_index": None,
                "ltex_id": None,
                "texture_path": None,
                "tile_count": count,
                "semantics": "raw 0 is the engine base texture sentinel, not "
                             "an LTEX record",
            }
            continue
        texture = ltex.get(index)
        _require(texture is not None,
                 f"raw {raw} -> LTEX index {index} not in plugin table "
                 f"({len(ltex)} records)")
        resolved[str(raw)] = {
            "raw_vtex": raw,
            "class": "ltex",
            "ltex_index": index,
            "ltex_id": texture.record_id,
            "texture_path": texture.file_name,
            "tile_count": count,
        }
    tile_total = sum(raw_counts.values())
    _require(tile_total == len(cells) * 256,
             f"scope tile total {tile_total} != {len(cells) * 256}")
    return {
        "cell_count": len(cells),
        "tile_total": tile_total,
        "base_sentinel_tiles": base_sentinel_tiles,
        "per_raw_vtex": resolved,
        "per_cell_tile_totals": cell_totals,
    }


def census_land_scope(
    plugin_path: str | Path,
    cells: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Census one plugin's LAND VTEX tiles for one cell set (real files)."""
    records = espland.load_land(plugin_path)
    ltex = espland.load_ltex(plugin_path)
    census = census_land_records(records, ltex, cells)
    return {"plugin": str(plugin_path), **census}


def require_surface(known: Iterable[str], name: str) -> None:
    """Fail closed on unknown semantic-surface references.

    Phase 1 plan/place/scape/mask stages may only reference surfaces present
    in the palette vocabulary; an unknown surface is a hard error rather than
    a silent pass-through.
    """
    _require(name in set(known),
             f"unknown semantic surface {name!r}; closed vocabulary is "
             f"{sorted(known)}")


def census_karthgad_core(root: str | Path) -> dict[str, Any]:
    """Reproduce the Karthgad core texture census from source LAND.

    Source: SHOTN ``Sky_Main.esm`` (read-only; hash-pinned and identical to
    the D-STAMP source pin ``fb57c808...``).  Cell ``(-102, 11)`` is the
    measured Karthgad city core.  The result is compared with the survey
    table (region survey §3.2) and marked ``recomputed``; any mismatch is
    reported, not hidden.
    """
    path = Path(root) / SKY_MAIN_ESM
    scope = census_land_scope(path, [(-102, 11)])
    per_raw = scope["per_raw_vtex"]
    rows = [
        {
            "raw_vtex": int(raw),
            "ltex_id": entry["ltex_id"],
            "texture_path": entry["texture_path"],
            "tile_count": entry["tile_count"],
        }
        for raw, entry in sorted(per_raw.items(), key=lambda kv: int(kv[0]))
    ]
    survey_table = [
        {"ltex_id": "T_Sky_TerrDirtRE_01", "tile_count": 159},
        {"ltex_id": "T_Sky_TerrGrassDirtRE_01", "tile_count": 42},
        {"ltex_id": "T_Nor_Set_TxCobbleStone_01", "tile_count": 35},
        {"ltex_id": "T_Sky_TerrRoadDirtRE_01", "tile_count": 16},
        {"ltex_id": "T_Sky_TerrGravel_01", "tile_count": 2},
        {"ltex_id": "T_Sky_TerrRockRE_01", "tile_count": 1},
    ]
    recomputed = {r["ltex_id"]: r["tile_count"] for r in rows}
    survey = {r["ltex_id"]: r["tile_count"] for r in survey_table}
    mismatches = {
        ltex_id: {"survey": survey[ltex_id], "recomputed": recomputed.get(ltex_id)}
        for ltex_id in survey
        if survey[ltex_id] != recomputed.get(ltex_id)
    }
    return {
        "status": "recomputed",
        "source_plugin": str(path),
        "cell": [-102, 11],
        "tile_total": scope["tile_total"],
        "rows": rows,
        "survey_comparison": {
            "survey_file": SURVEY_REGION,
            "survey_section": "3.2",
            "survey_lines": [226, 233],
            "matches_survey": not mismatches,
            "mismatches": mismatches,
            "note": "survey omitted the single T_Sky_TerrDirtRE_03 tile "
                    "present in the source LAND (recomputed count includes it)",
        },
    }


def survey_core_fractions() -> dict[str, Any]:
    """The survey-measured clearance fractions (consumed, with provenance)."""
    return {
        "status": "survey_measured",
        "survey_file": SURVEY_REGION,
        "survey_section": "1. Executive summary / 3.2",
        "survey_lines": [26, 34, 226, 233],
        "core_fractions": {
            "settlement_dirt": 0.62,
            "settlement_grass_dirt": 0.16,
            "settlement_cobble": 0.14,
            "road": 0.06,
        },
        "tile_basis": "256 VTEX tiles of cell (-102,11); fractions rounded "
                      "to 2 decimals in the survey (recomputed census gives "
                      "159/256, 42/256, 35/256, 16/256)",
    }


def water_edge_evidence(root: str | Path, cells: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Quantitative water-edge class evidence for the effective block.

    Evidence: raw-1 (Sand) tile counts per cell from the remap census; the
    site survey's water mask definition, water-cell list, and shore length;
    per-cell water fraction from ``terrain_cells.json``.
    """
    survey = _load_json(Path(root) / SITE_SURVEY)
    terrain = _load_json(Path(root) / TERRAIN_CELLS)
    terrain_by_cell = {tuple(row[0:2]): row for row in terrain["cells"]}
    water_cells = [tuple(c) for c in survey["water"]["water_cells_measured"]]
    cell_set = set(cells)
    return {
        "water_level_gu": survey["water"]["level_gu"],
        "water_mask_definition": survey["water"]["mask_definition"],
        "water_cell_count": len(water_cells),
        "water_cells": [list(c) for c in sorted(water_cells)],
        "shore_length_gu": survey["water"]["bodies"][0]["shore_length_gu"],
        "shore_length_definition": survey["water"]["bodies"][0][
            "shore_length_definition"
        ],
        # terrain_cells fields: x, y, e_min_gu, e_med_gu, e_max_gu,
        # slope_mean_deg, water_frac, wdist_gu, band, land  (row[6] = water)
        "per_cell_water_fraction": {
            f"{row[0]},{row[1]}": row[6]
            for row in terrain["cells"]
            if tuple(row[0:2]) in cell_set
        },
        "per_cell_band": {
            f"{row[0]},{row[1]}": row[8]
            for row in terrain["cells"]
            if tuple(row[0:2]) in cell_set
        },
        "notes": [
            "raw 1 (Sand) tiles are the measured water-edge/littoral class "
            "of the block (see land_census effective scope)",
            "terrain_cells water fraction uses z<=0 mask; site_survey water "
            "mask uses 5x5 128-GU LAND vertex check",
        ],
    }


def roads_evidence(root: str | Path) -> dict[str, Any]:
    """Pinned road identity + aligned consumer-product records."""
    centerlines = PinnedFile(Path(root) / CENTERLINES_JSON)
    source_centerlines = PinnedFile(Path(root) / CENTERLINES_SOURCE_JSON)
    metadata = PinnedFile(Path(root) / CENTERLINES_METADATA)
    report = _load_json(Path(root) / REMAP_REPORT)
    survey = _load_json(Path(root) / SITE_SURVEY)
    site_roads = survey["roads"]
    return {
        "source_road_identity": {
            "raw_vtex": 78,
            "base_esm_ltex_index": 77,
            "base_esm_ltex_id": "MA_sulphur_rock02",
            "base_esm_texture": "Tx_MA_sulphur_rock02.tga",
            "remap_output_ltex_index": 77,
            "remap_output_ltex_id": "T_Hr_TerrRoadOH_01",
            "remap_output_texture": "hr\\lnd\\hr_oh_road_01.dds",
            "note": "raw 78 is the only protected source road identity; raw 1 "
                    "Sand is never road",
        },
        "road_network_ref": {
            "bundle": "output/mapdata/roads/tamriel_aligned_centerlines_v1/",
            "consumer_geometry": "tamriel_aligned_centerlines_v1.json",
            "sha256": centerlines.sha256(),
            "alignment_manifest": {
                "file": "output/mapdata/roads/tamriel_aligned_centerlines_v1/"
                        "alignment_manifest.json",
            },
            "source_bundle": {
                "dir": "output/mapdata/roads/tamriel_source_centerlines_v1/",
                "canonical": "tamriel_road_centerlines_v1.json",
                "sha256": source_centerlines.sha256(),
            },
            "source_metadata": {
                "file": CENTERLINES_METADATA,
                "sha256": metadata.sha256(),
            },
            "geometry_authority": "direct tamriel.esm LAND/VTEX raw-78 tiles",
            "note": "consumers load the aligned product only through "
                    "src/procgen/aligned_roads.py; the source-space bundle "
                    "and the XCF/BMP are topology/provenance storage and are "
                    "never planner inputs; old roads_graph_clean.json and "
                    "raw-78-only land_roads.json geometry are not consumed",
        },
        "remap_report_provenance": {
            "file": REMAP_REPORT,
            "sha256": PinnedFile(Path(root) / REMAP_REPORT).sha256(),
        },
        "site_survey_road_tiles_78": site_roads["road_tiles"],
        "site_survey_raw_vtex": site_roads["raw_vtex"],
        "site_survey_rejected_vector_graph": site_roads["rejected_vector_graph"],
    }


def build_region_palette(
    *,
    root: str | Path,
    date: str,
    door_steps: Mapping[str, Any],
    ground_rules: Mapping[str, Any],
    r072_census: Mapping[str, Any],
    effective_census: Mapping[str, Any],
    effective_base_census: Mapping[str, Any],
    target_cells: Sequence[tuple[int, int]],
    r072_cells: Sequence[tuple[int, int]],
    water: Mapping[str, Any],
    roads: Mapping[str, Any],
    karthgad_core: Mapping[str, Any],
    live_remap: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble ``region_palette.json``."""
    region = load_region(root)
    survey = _load_json(Path(root) / SITE_SURVEY)
    r072_by_raw = r072_census["per_raw_vtex"]
    eff_by_raw = effective_census["per_raw_vtex"]

    def fraction_table(by_raw: Mapping[str, Any], total: int) -> list[dict[str, Any]]:
        rows = []
        for raw, entry in sorted(by_raw.items(), key=lambda kv: int(kv[0])):
            rows.append({
                "raw_vtex": int(raw),
                "ltex_id": entry["ltex_id"],
                "texture_path": entry["texture_path"],
                "class": entry["class"],
                "tile_count": entry["tile_count"],
                "tile_fraction": entry["tile_count"] / total,
            })
        return rows

    r072_total = r072_census["tile_total"]
    eff_total = effective_census["tile_total"]
    r072_rows = fraction_table(r072_by_raw, r072_total)
    eff_rows = fraction_table(eff_by_raw, eff_total)
    _require(abs(sum(r["tile_fraction"] for r in r072_rows) - 1.0) < 1e-9,
             "R072 fractions do not sum to 1")
    _require(abs(sum(r["tile_fraction"] for r in eff_rows) - 1.0) < 1e-9,
             "effective fractions do not sum to 1")

    target_set = set(target_cells)
    r072_set = set(r072_cells)
    overlap = sorted(target_set & r072_set)
    outside = sorted(target_set - r072_set)
    # base ESM composition of the effective block
    base_rows = fraction_table(effective_base_census["per_raw_vtex"],
                               effective_base_census["tile_total"])

    survey_pin = {
        "file": SURVEY_REGION,
        "sha256": PinnedFile(Path(root) / SURVEY_REGION).sha256(),
    }
    # The palette references the site-survey road-tile evidence instead of
    # duplicating it: the raw tile list lives in census.json (both files are
    # emitted by the same build, so the reference is self-contained).
    roads_view = dict(roads)
    raw_tiles = roads_view.get("site_survey_road_tiles_78")
    if isinstance(raw_tiles, list):
        roads_view["site_survey_road_tiles_78"] = {
            "tile_count": len(raw_tiles),
            "census_ref": "census.json#land.roads.site_survey_road_tiles_78",
            "note": "raw 1275-tile list preserved in census.json; not "
                    "duplicated here",
        }
    return {
        "schema_version": 1,
        "date": date,
        "region_id": region["region_id"],
        "region_name": region["name"],
        "cell_set_definition": "ptr_planning_polygon",
        "cell_set_note": "R072 is a PTR planning polygon, NOT a game RGNN; "
                         "no game RGNN covers the block",
        "cells": [list(c) for c in sorted(r072_cells)],
        "cells_count": len(r072_cells),
        "effective_target": {
            "definition": "site_survey.json target_cells (7x7 remap block)",
            "cells": [list(c) for c in sorted(target_cells)],
            "cell_count": len(target_cells),
            "membership": {
                "inside_r072_count": len(overlap),
                "outside_r072_cells": [list(c) for c in outside],
                "outside_r072_regions": _regions_containing(root, outside),
                "note": "R072 (191-cell polygon) and the 49-cell effective "
                        "block are distinct scopes and are never merged",
            },
        },
        "base_textures": {
            "scope": "R072 PTR polygon, 191 cells, tamriel.esm LAND",
            "tile_total": r072_total,
            "per_raw_vtex": r072_rows,
            "raw_0_semantics": "raw 0 is the engine base sentinel, not an "
                               "LTEX record; no raw-0 tiles occur in this "
                               "scope",
            "load_order_caveats": {
                "identity_labels": "raw -> LTEX identities above resolve "
                                   "through tamriel.esm's own local LTEX "
                                   "table (base-ESM-only load order)",
                "with_remap_esp_loaded": "when "
                                         "falkreath_landscape_texture_remap."
                                         "esp is loaded after the base ESM, "
                                         "raw 33/78/92 render under the "
                                         "remap identities "
                                         "(T_Sky_TerrGrassRE_01 / "
                                         "T_Hr_TerrRoadOH_01 / "
                                         "T_Sky_TerrPine_01); tile counts "
                                         "are unaffected",
                "verification_status": "OpenMW 0.51 internal API "
                                       "confirmation unavailable from the "
                                       "connected openmw-docs index; the "
                                       "plugin-local convention is a "
                                       "workspace-validated internal/"
                                       "toolchain contract (espland + remap "
                                       "round-trip evidence); verify against "
                                       "engine source before Phase 1 "
                                       "authoring",
            },
        },
        "effective_block_textures": {
            "scope": "49-cell remap block, falkreath_landscape_texture_remap.esp "
                     "LAND with its own LTEX table",
            "tile_total": eff_total,
            "per_raw_vtex": eff_rows,
            "base_esm_composition": {
                "note": "same LAND payloads in tamriel.esm; raw identities "
                        "below are the base-ESM local LTEX resolution",
                "tile_total": effective_base_census["tile_total"],
                "per_raw_vtex": base_rows,
            },
            "load_order_caveats": {
                "identity_labels": "per_raw_vtex identities above resolve "
                                   "through the remap ESP's own local LTEX "
                                   "table and match in-game appearance when "
                                   "the remap ESP is loaded",
                "verification_status": "OpenMW 0.51 internal API "
                                       "confirmation unavailable from the "
                                       "connected openmw-docs index; "
                                       "workspace-validated internal/"
                                       "toolchain contract (espland + remap "
                                       "round-trip evidence)",
            },
            "provenance": {
                "remap_esp": REMAP_ESP,
                "base_esm": TAMRIEL_ESM,
                "remap_report": REMAP_REPORT,
            },
        },
        "settlement_clearance": {
            "note": "Karthgad-measured clearance pattern (one culture sample; "
                    "cell (-102,11))",
            "core_fractions": survey_core_fractions()["core_fractions"],
            "recomputed_census": karthgad_core,
            "road": {
                "ltex_id": "T_Hr_TerrRoadOH_01",
                "raw_vtex": 78,
                "note": "raw-78 identity is a hard invariant; scatter and "
                        "groundcover road gates key on it",
            },
        },
        "water_edge": {
            "classes": [
                {
                    "class": "water_edge_sand",
                    "ltex_id": "Sand",
                    "raw_vtex": 1,
                    "tile_count": eff_by_raw.get("1", {}).get("tile_count", 0),
                    "tile_fraction": eff_by_raw.get("1", {}).get(
                        "tile_count", 0
                    ) / eff_total,
                    "evidence": water,
                    "note": "quantitative: raw-1 Sand tiles concentrate in "
                            "the 11 water cells; not a name guess",
                }
            ],
            "karthgad_measured_out_of_block": {
                "note": "T_Sky_TerrGravelRiver_01 + T_Sky_TerrRockRE_01 "
                        "riverside classes measured at Karthgad (survey 3.2) "
                        "are not present in the Falkreath block; not part of "
                        "this block's closed vocabulary",
                "provenance": {**survey_pin, "section": "3.2", "lines": [232]},
            },
        },
        "road": roads_view,
        "flora_rock": {
            "profiles": {
                "file": SCATTER_ANALYSIS,
                "sha256": PinnedFile(Path(root) / SCATTER_ANALYSIS).sha256(),
            },
            "cliffs": {
                "file": CLIFF_ANALYSIS,
                "sha256": PinnedFile(Path(root) / CLIFF_ANALYSIS).sha256(),
            },
            "proxy_region": {
                "status": "proxy",
                "source": "Vorndgad Forest (59-cell RGNN scope) measured "
                          "ecology",
                "note": "Vorndgad measured ecology is the Falkreath proxy "
                        "until a Kreathi Dale profile is measured",
            },
        },
        "door_step_prior_gu": {
            "stamp_aggregated": {
                "p10": door_steps["p10"],
                "p50": door_steps["p50"],
                "p90": door_steps["p90"],
                "sample_count": door_steps["sample_count"],
            },
            "ground_rules": {
                "p10": ground_rules["door_step_p10_game_units"],
                "p50": ground_rules["door_step_p50_game_units"],
                "p90": ground_rules["door_step_p90_game_units"],
                "sample_count": ground_rules["door_step_count"],
                "status": "survey_measured",
            },
        },
        "groundcover": {
            "ini_sections": GROUNDCOVER_INI,
            "ini_sha256": PinnedFile(Path(root) / GROUNDCOVER_INI).sha256(),
            "road_mask_regex": ".*hr_oh_road.*",
            "note": "pre-remap temporary palette keys on current-texture "
                    "names; road mask stays coupled to LTEX record ids",
        },
        "semantic_surfaces": {
            "phase": 1,
            "note": "closed vocabulary: plan/place/scape/mask stages may "
                    "only reference surfaces listed here; unknown surface "
                    "references fail closed; a Phase-1 planner validator "
                    "must reject any authored raw != planned_raw_vtex for a "
                    "listed surface",
            "authoring_contract_note": (
                "each surface carries an explicit planned_assignment "
                "(planned_raw_vtex / planned_ltex_index / planned_ltex_id) "
                "for the future masterless city output plugin; raw_vtex is "
                "NEVER derived from the surface ordinal (ordinal + 1 is not "
                "a raw value); the masterless plugin must define a local "
                "LTEX record for every emitted planned raw > 0"
            ),
            "surfaces": list(PHASE1_SURFACES),
        },
        "planned_output_plugin": {
            "plugin_scope": "masterless city output plugin (masters: [])",
            "note": "per ground rule 11, generated plugins declare no "
                    "masters; the city plugin owns its own local LTEX table",
            "required_local_ltex": [
                {
                    "ltex_index": assignment["planned_ltex_index"],
                    "ltex_id": assignment["planned_ltex_id"],
                    "for_surface": surface["surface"],
                    "planned_raw_vtex": assignment["planned_raw_vtex"],
                }
                for surface in PHASE1_SURFACES
                if isinstance(assignment := surface.get("planned_assignment"),
                              dict)
            ],
            "live_remap_evidence": {
                "esp_path": live_remap["esp_path"],
                "esp_sha256": live_remap["esp_sha256"],
                "record_count": live_remap["record_count"],
                "records": live_remap["records"],
                "note": "measured live remap ESP LTEX table read via "
                        "espland.load_ltex from the pinned remap ESP; "
                        "planned assignments at shared indices must match "
                        "these identities (validation gates fail closed on "
                        "divergence, especially index 77 road)",
            },
            "fail_closed_rule": "any additional raw value (>0) authored into "
                                "the city plugin's LAND must add its own "
                                "local LTEX record; an undefined local LTEX "
                                "index is an authoring error",
        },
    }


def _regions_containing(
    root: str | Path, cells: Sequence[tuple[int, int]]
) -> list[dict[str, Any]]:
    if not cells:
        return []
    payload = _load_json(Path(root) / REGIONS_JSON)
    result: list[dict[str, Any]] = []
    for cell in cells:
        center = ((cell[0] + 251 + 0.5) * 64.0, (59 - cell[1] + 0.5) * 64.0)
        for region in payload["regions"]:
            if _point_in_polygon(center, region["polygon_map_px"]):
                result.append({
                    "cell": [cell[0], cell[1]],
                    "region_id": region["region_id"],
                    "region_name": region["name"],
                })
                break
    return result


def build_census_land_section(
    *,
    r072_cells: Sequence[tuple[int, int]],
    target_cells: Sequence[tuple[int, int]],
    r072_census: Mapping[str, Any],
    effective_census: Mapping[str, Any],
    effective_base_census: Mapping[str, Any],
    karthgad_core: Mapping[str, Any],
    water: Mapping[str, Any],
    roads: Mapping[str, Any],
    site_survey_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Land/water/road raw vectors and provenance for ``census.json``."""
    return {
        "scopes": {
            "r072": {
                "definition": "R072 KREATHI DALE ptr_planning_polygon "
                              "(regions.json)",
                "cell_count": len(r072_cells),
                "cells": [list(c) for c in sorted(r072_cells)],
            },
            "effective_falkreath": {
                "definition": "site_survey.json 49-cell target block (remap "
                              "ESP over base ESM)",
                "cell_count": len(target_cells),
                "cells": [list(c) for c in sorted(target_cells)],
                "note": "separate from the 191-cell R072 scope; their texture "
                        "fractions are never merged",
            },
        },
        "r072_tamriel_esm": r072_census,
        "effective_remap_esp": effective_census,
        "effective_base_esm_composition": effective_base_census,
        "karthgad_core_reproduction": karthgad_core,
        "water_edge": water,
        "roads": roads,
        "site_survey_stats": site_survey_stats,
    }
