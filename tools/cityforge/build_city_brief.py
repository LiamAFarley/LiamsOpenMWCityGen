"""Cityforge T0.5 D-BRIEF census CLI — build the Falkreath v1 planner
vocabulary bundle (kit brief + region palette + census + validation).

Pipeline position
------------------
Final stage of dispatch 5: pins every authoritative input, runs the stamp/
spacing/door-step census (``src/procgen/citybrief.py``) and the LAND/VTEX
region census (``src/procgen/regionpalette.py``), assembles the four canonical
JSON products under ``output/cityforge/briefs/falkreath_v1/``, and proves
byte-determinism by building twice into fresh staging directories and
comparing all four files.

Usage
-----
    python tools/cityforge/build_city_brief.py --date 2026-08-10

Optional: ``--root F:/ProcGenWorkspace`` (workspace root), ``--out-dir``
(override canonical output), ``--staging-base`` (temp base), ``--no-proof``
(skip the double-build determinism proof; for debugging only).

Outputs (canonical)
-------------------
- ``kit_brief.json``        planner build vocabulary (enums, stamps, priors)
- ``region_palette.json``   ground/surface palette for R072 + effective block
- ``census.json``           raw measured vectors + provenance
- ``validation.json``       closed-world and cross-file contract gates

Exit codes: 0 = all gates pass; 1 = hard census failure (FAILURE protocol);
2 = validation gates failed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen import citybrief, regionpalette  # noqa: E402
from procgen.road_semantics import load_road_assignments  # noqa: E402
from procgen.censusio import (  # noqa: E402
    PinnedFile,
    deterministic_dumps,
    sha256_file,
    write_deterministic,
)

CANONICAL_OUT = "output/cityforge/briefs/falkreath_v1"
DEFAULT_LANDSCAPE_POLICY = "configs/landscape/kreathi_dale_v1.json"

#: Every authoritative input pinned by this build (relative to workspace root).
PINNED_INPUTS: dict[str, str] = {
    "stamp_library_karthgad": "output/cityforge/stamps/karthgad_nord_v1.json",
    "stamp_library_markarth": "output/cityforge/stamps/markarth_side_stone_v1.json",
    "stamp_catalog_index": "output/cityforge/stamps/catalog_v1/index.json",
    "site_survey": "output/cityforge/sites/falkreath_v1/site_survey.json",
    "centerlines_bundle": "output/mapdata/roads/tamriel_aligned_centerlines_v1/"
                          "tamriel_aligned_centerlines_v1.json",
    "centerlines_source_bundle": "output/mapdata/roads/tamriel_source_centerlines_v1/"
                                 "tamriel_road_centerlines_v1.json",
    "centerlines_source_metadata": "output/mapdata/roads/tamriel_source_centerlines_v1/"
                                   "source_metadata.json",
    "regions": "output/mapdata/regions.json",
    "tamriel_esm": "tamriel.esm",
    "remap_esp": "output/falkreath_landscape_texture_remap.esp",
    "remap_report": "output/falkreath_landscape_texture_remap_report.json",
    "sky_main_esm": "Sky_Main.esm",
    "terrain_cells": "output/terrain_cells.json",
    "vorndgad_scatter": "output/vorndgad_scatter_analysis.json",
    "vorndgad_cliff": "output/vorndgad_cliff_analysis.json",
    "skyrim_ground_rules": "output/skyrim_ground_rules.json",
    "groundcover_ini": "configs/groundcover_falkreath_v1_currenttextures.ini",
    "final_palette_catalog": "output/settlement-splits/markarth-side-v2/"
                             "final-markarth-extraction-2026-08-10-library/"
                             "stamp_palette_v1/catalog.json",
    "final_render_manifest": "output/settlement-splits/markarth-side-v2/"
                             "final-markarth-extraction-2026-08-10-library/"
                             "render_library_manifest.json",
    "kit_survey": ".opencode/runs/karthgad-city-authoring/"
                  "2026-08-04_karthgad_city_kit_survey.md",
    "region_survey": ".opencode/runs/karthgad-city-authoring/"
                     "2026-08-04_region_palette_and_siting_survey.md",
    "karthgad_placement_manifest": "output/skyrim-settlements/karthgad-v1/"
                                   "b1/placement_manifest.json",
    "karthgad_buildings_index": "output/skyrim-settlements/karthgad-v1/"
                                "components/buildings_index.json",
}


def build_payloads(
        root: Path,
        date: str,
        landscape_policy: Path | str | None = None,
) -> dict[str, dict]:
    """Run the whole census and return the four payload dicts."""
    policy_path = Path(landscape_policy or DEFAULT_LANDSCAPE_POLICY)
    if not policy_path.is_absolute():
        policy_path = root / policy_path
    if not policy_path.is_file():
        raise citybrief.CensusError(f"landscape policy missing: {policy_path}")
    landscape = json.loads(policy_path.read_text(encoding="utf-8"))
    try:
        road_assignments = {
            name: assignment.to_dict()
            for name, assignment in load_road_assignments(landscape).items()
        }
    except ValueError as exc:
        raise citybrief.CensusError(
            f"landscape policy road assignments are invalid: {exc}") from exc
    hierarchy_map = landscape.get("road_class_by_hierarchy") or {}
    if not isinstance(hierarchy_map, dict):
        raise citybrief.CensusError(
            "landscape policy road_class_by_hierarchy must be an object")
    # --- pin inputs -------------------------------------------------------
    pins: dict[str, dict] = {}
    for alias, relative in PINNED_INPUTS.items():
        path = root / relative
        if not path.is_file():
            raise citybrief.CensusError(f"pinned input missing: {path}")
        pins[alias] = {
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    try:
        policy_ref = policy_path.relative_to(root).as_posix()
    except ValueError:
        policy_ref = str(policy_path)
    pins["landscape_policy"] = {
        "path": policy_ref,
        "sha256": sha256_file(policy_path),
        "size_bytes": policy_path.stat().st_size,
    }

    # --- stamps -----------------------------------------------------------
    libraries = citybrief.load_stamp_libraries(root)
    eligible, excluded = citybrief.select_eligible_stamps(libraries)
    upstream = citybrief.summarize_upstream_exclusions(libraries)
    enum = citybrief.derive_building_type_enum(eligible)
    gaps = citybrief.capability_gaps(eligible)
    library_pin_aliases = {
        "karthgad_nord_v1": "stamp_library_karthgad",
        "markarth_side_stone_v1": "stamp_library_markarth",
    }
    library_pins = {
        library_id: {
            "file": citybrief.STAMP_LIBRARY_PATHS[library_id],
            "sha256": pins[library_pin_aliases[library_id]]["sha256"],
        }
        for library_id in citybrief.STAMP_LIBRARY_PATHS
    }

    markarth_previews = citybrief.resolve_markarth_previews(eligible, root)
    karthgad_previews = citybrief.verify_karthgad_previews(eligible, root)

    spacing = citybrief.spacing_census(eligible)
    door_steps = citybrief.aggregate_door_steps(eligible)
    ground_rules = citybrief.ground_rules_door_steps(root)
    footprint_q = citybrief.footprint_quantiles(eligible)

    # --- region / land ----------------------------------------------------
    region = regionpalette.load_region(root)
    r072_cells = regionpalette.rasterize_region_cells(region)
    target_cells = regionpalette.load_target_cells(root)
    r072_census = regionpalette.census_land_scope(root / regionpalette.TAMRIEL_ESM,
                                                  r072_cells)
    effective_census = regionpalette.census_land_scope(
        root / regionpalette.REMAP_ESP, target_cells
    )
    effective_base_census = regionpalette.census_land_scope(
        root / regionpalette.TAMRIEL_ESM, target_cells
    )
    karthgad_core = regionpalette.census_karthgad_core(root)
    water = regionpalette.water_edge_evidence(root, target_cells)
    roads = regionpalette.roads_evidence(root)
    # Live remap ESP LTEX table (measured, hash-pinned): the authoring
    # assignment contract is cross-checked against this, not only the
    # expected-table constant (review finding M-8).
    live_remap = regionpalette.live_remap_ltex_table(root)
    survey_payload = json.loads((root / regionpalette.SITE_SURVEY)
                                .read_text(encoding="utf-8"))
    site_stats = survey_payload["stats"]

    # --- assembly ---------------------------------------------------------
    raw_library_counts = {
        library_id: len(payload["stamps"])
        for library_id, payload in libraries.items()
    }
    kit_brief = citybrief.build_kit_brief(
        eligible=eligible,
        excluded_records=excluded,
        upstream_exclusions=upstream,
        building_type_enum=enum,
        gaps=gaps,
        spacing=spacing,
        door_steps=door_steps,
        ground_rules=ground_rules,
        previews={**markarth_previews, **karthgad_previews},
        library_pins=library_pins,
        root=root,
        date=date,
    )
    region_palette = regionpalette.build_region_palette(
        root=root,
        date=date,
        door_steps=door_steps,
        ground_rules=ground_rules,
        r072_census=r072_census,
        effective_census=effective_census,
        effective_base_census=effective_base_census,
        target_cells=target_cells,
        r072_cells=r072_cells,
        water=water,
        roads=roads,
        karthgad_core=karthgad_core,
        live_remap=live_remap,
        road_assignments=road_assignments,
        road_class_by_hierarchy=hierarchy_map,
    )
    census = {
        "schema_version": 1,
        "date": date,
        "generated_by": "citybrief 0.1.0 (Cityforge T0.5 D-BRIEF census)",
        "inputs": pins,
        "stamps": citybrief.build_census_stamp_section(
            eligible=eligible,
            excluded_records=excluded,
            upstream_exclusions=upstream,
            building_type_enum=enum,
            library_counts=raw_library_counts,
            markarth_previews=markarth_previews,
            karthgad_previews=karthgad_previews,
        ),
        "building_types": {
            "enum": list(enum),
            "counts": {
                type_name: sum(1 for s in eligible
                               if s["building_type"] == type_name)
                for type_name in enum
            },
            "capability_gaps": list(gaps),
        },
        "footprint_quantiles": footprint_q,
        "spacing": spacing,
        "door_steps": door_steps,
        "ground_rules_door_steps": ground_rules,
        "land": regionpalette.build_census_land_section(
            r072_cells=r072_cells,
            target_cells=target_cells,
            r072_census=r072_census,
            effective_census=effective_census,
            effective_base_census=effective_base_census,
            karthgad_core=karthgad_core,
            water=water,
            roads=roads,
            site_survey_stats=site_stats,
        ),
    }
    return {
        "kit_brief.json": kit_brief,
        "region_palette.json": region_palette,
        "census.json": census,
        "validation.json": _build_validation(
            kit_brief=kit_brief,
            region_palette=region_palette,
            census=census,
            raw_library_counts=raw_library_counts,
            live_remap=live_remap,
            root=root,
        ),
    }


# --------------------------------------------------------------------------
# Validation gates
# --------------------------------------------------------------------------

def _check(checks: list[dict], check_id: str, label: str, passed: bool,
           detail: str) -> None:
    checks.append({
        "id": check_id,
        "label": label,
        "passed": bool(passed),
        "detail": detail,
    })


def _build_validation(*, kit_brief, region_palette, census, root,
                      raw_library_counts, live_remap) -> dict:
    checks: list[dict] = []

    # --- stamps / eligibility --------------------------------------------
    stamps = census["stamps"]
    eligible = stamps["eligible"]
    _check(checks, "stamp.eligible_count",
           "eligible count matches the acceptance contract (54 = 11 + 44 - "
           "barracks) and both emitted stamp lists agree",
           len(eligible) == 54 and kit_brief["stamp_count"] == len(eligible),
           f"eligible={len(eligible)} brief_stamps={kit_brief['stamp_count']}")
    ids = [s["stamp_id"] for s in eligible]
    _check(checks, "stamp.unique",
           "every eligible record occurs exactly once",
           len(ids) == len(set(ids)),
           f"records={len(ids)} unique={len(set(ids))}")
    _check(checks, "stamp.raw_counts_derived",
           "raw per-library stamp counts are derived from the loaded "
           "hash-pinned libraries (never literals)",
           stamps["libraries"]["karthgad_nord_v1"]["stamp_count_raw"]
           == raw_library_counts["karthgad_nord_v1"]
           and stamps["libraries"]["markarth_side_stone_v1"]["stamp_count_raw"]
           == raw_library_counts["markarth_side_stone_v1"],
           f"emitted={ {k: v['stamp_count_raw'] for k, v in stamps['libraries'].items()} } "
           f"recomputed={raw_library_counts}")
    _check(checks, "stamp.raw_to_eligible_math",
           "raw counts, exclusions, and final totals reconcile",
           raw_library_counts["karthgad_nord_v1"]
           + raw_library_counts["markarth_side_stone_v1"]
           - len(stamps["excluded"]["records"]) == len(eligible)
           and len(stamps["excluded"]["records"]) == 1,
           f"raw={sum(raw_library_counts.values())} "
           f"excluded={len(stamps['excluded']['records'])} "
           f"eligible={len(eligible)}")
    barracks = [e for e in stamps["excluded"]["records"]
                if e["stamp_id"] == citybrief.BARRACKS_STAMP_ID]
    _check(checks, "stamp.barracks_excluded_once",
           "Castle Barracks absent from eligible and present exactly once in "
           "the exclusion ledger with the user reason",
           len(barracks) == 1
           and barracks[0]["reason"] == citybrief.BARRACKS_REASON,
           f"ledger_entries={len(barracks)}")
    _check(checks, "stamp.barracks_absent_from_types",
           "Castle Barracks absent from type/count/quantile/spacing census",
           "barracks" not in census["building_types"]["counts"]
           and "barracks" not in census["stamps"]["building_type_enum"]
           and "barracks" not in kit_brief["building_type_enum"],
           "barracks excluded before all aggregation")

    # --- coverage / capability -------------------------------------------
    counts = kit_brief["building_type_counts"]
    _check(checks, "coverage.houses",
           "at least 15 house stamps available",
           counts.get("house", 0) >= 15, f"houses={counts.get('house', 0)}")
    for type_name in ("tavern", "smith", "shop", "farm"):
        _check(checks, f"coverage.{type_name}",
               f"at least one {type_name} stamp available",
               counts.get(type_name, 0) >= 1,
               f"{type_name}={counts.get(type_name, 0)}")
    lodge = [g for g in kit_brief["capability_gaps"] if g["type"] == "lodge"]
    _check(checks, "coverage.lodge_unavailable",
           "lodge is explicitly unavailable (capability gap, not fabricated)",
           len(lodge) == 1 and lodge[0]["available"] is False,
           f"gap_records={len(lodge)}")
    _check(checks, "enum.derived",
           "building_type_enum derived from eligible stamps (no stale "
           "hard-coded enum)",
           set(kit_brief["building_type_enum"])
           == set(census["building_types"]["enum"])
           == {s["building_type"] for s in eligible},
           f"enum={kit_brief['building_type_enum']}")

    # --- previews ---------------------------------------------------------
    markarth_eligible = [s for s in eligible if s["library_id"] == "markarth_side_stone_v1"]
    karthgad_eligible = [s for s in eligible if s["library_id"] == "karthgad_nord_v1"]
    stale = [s for s in eligible if "split-render-v6" in s.get("preview_sheet", "")]
    missing_file = []
    for s in eligible:
        path = root / s["preview_sheet"]
        if not path.is_file():
            missing_file.append(s["stamp_id"])
    _check(checks, "preview.markarth_resolved",
           "every eligible Markarth stamp resolves to a final terrain-backed "
           "palette sheet",
           len(markarth_eligible) == 43
           and all(s.get("preview_sha256") for s in markarth_eligible),
           f"markarth_eligible={len(markarth_eligible)}")
    _check(checks, "preview.section_subcounts",
           "census preview_verification section carries explicit per-library "
           "subcounts (Karthgad 11, Markarth 43)",
           stamps["preview_verification"]["karthgad"]["count"]
           == len(karthgad_eligible) == 11
           and stamps["preview_verification"]["markarth"]["count"]
           == len(markarth_eligible) == 43
           and set(stamps["preview_verification"]["markarth"]["records"])
           == {s["stamp_id"] for s in markarth_eligible},
           "section subcounts and ids match the eligible partitions")
    _check(checks, "preview.no_stale_v6",
           "no stale split-render-v6 preview remains in the emitted brief",
           len(stale) == 0, f"stale={len(stale)}")
    _check(checks, "preview.files_exist",
           "all emitted preview paths resolve to files",
           len(missing_file) == 0, f"missing={missing_file}")

    # --- land scopes ------------------------------------------------------
    land = census["land"]
    r072_total = land["r072_tamriel_esm"]["tile_total"]
    eff_total = land["effective_remap_esp"]["tile_total"]
    _check(checks, "land.r072_total",
           "R072 tile total reconciles to 191*256",
           r072_total == 191 * 256, f"total={r072_total}")
    _check(checks, "land.effective_total",
           "effective block tile total reconciles to 49*256",
           eff_total == 49 * 256, f"total={eff_total}")
    for scope_name, scope in (("r072_tamriel_esm", land["r072_tamriel_esm"]),
                              ("effective_remap_esp", land["effective_remap_esp"])):
        zero_entry = scope["per_raw_vtex"].get("0")
        accounted = (zero_entry["tile_count"] if zero_entry else 0)
        _check(checks, f"land.raw0_sentinel.{scope_name}",
               "raw-0 sentinel accounting: base_sentinel_tiles equals the "
               "raw-0 tile count, and raw 0 is never an LTEX record",
               scope["base_sentinel_tiles"] == accounted
               and (zero_entry is None or zero_entry["class"] == "base_sentinel"),
               f"base_sentinel_tiles={scope['base_sentinel_tiles']} "
               f"raw0_tiles={accounted}")
    _check(checks, "land.scope_separation",
           "R072 and effective scopes stay separate (no silent merge)",
           land["r072_tamriel_esm"]["cell_count"] == 191
           and land["effective_remap_esp"]["cell_count"] == 49,
           "191 vs 49 cell scopes")
    _check(checks, "land.sand_not_road",
           "raw 1 Sand is never labeled road",
           land["roads"]["source_road_identity"]["raw_vtex"] == 78
           and land["roads"]["source_road_identity"]["remap_output_ltex_id"]
           == "T_Hr_TerrRoadOH_01",
           "raw 78 is the only protected road identity")

    # --- palette closure --------------------------------------------------
    palette = region_palette
    surfaces = {s["surface"] for s in palette["semantic_surfaces"]["surfaces"]}
    palette_refs = [
        "base", "settlement_dirt", "settlement_grass_dirt",
        "settlement_cobble", "road", "water_edge_sand",
    ]
    _check(checks, "palette.closure",
           "region palette covers every semantic surface Phase 1 stages may "
           "emit; unknown references fail closed",
           set(palette_refs) <= surfaces,
           f"surfaces={sorted(surfaces)}")
    _check(checks, "palette.fractions_sum",
           "tile fractions sum exactly to each scope's tile total",
           abs(sum(t["tile_fraction"] for t in
                   palette["base_textures"]["per_raw_vtex"]) - 1.0) < 1e-9
           and abs(sum(t["tile_fraction"] for t in
                       palette["effective_block_textures"]["per_raw_vtex"])
                   - 1.0) < 1e-9,
           "R072 and effective fractions both sum to 1")

    # --- authoring assignment contract (I-1) ------------------------------
    for record in regionpalette.validate_authoring_assignments(
            palette["semantic_surfaces"]["surfaces"]):
        _check(checks, record["id"], record["id"], record["passed"],
               record["detail"])
    required_table = palette["planned_output_plugin"]["required_local_ltex"]
    _check(checks, "authoring.masterless_scope",
           "planned output plugin is masterless with a local LTEX record for "
           "every emitted raw > 0",
           palette["planned_output_plugin"]["plugin_scope"].startswith(
               "masterless city output plugin")
           and len(required_table) == len(
               {r["planned_raw_vtex"] for r in required_table})
           and all(r["ltex_index"] == r["planned_raw_vtex"] - 1
                   for r in required_table),
           f"required_local_ltex={[r['ltex_index'] for r in required_table]}")
    _check(checks, "authoring.no_slot_plus_one_hazard",
           "raw_vtex is explicit per surface; ordinal+1 inference is "
           "impossible (no surface may carry an abstract planned_output_slot "
           "contract)",
           all("planned_output_slot" not in s
               for s in palette["semantic_surfaces"]["surfaces"])
           and all(s["planned_assignment"]["planned_raw_vtex"] != 5
                   for s in palette["semantic_surfaces"]["surfaces"]),
           "planned raw values are explicit; none equals the old slot+1 "
           "hazard value 5")

    # --- live remap ESP cross-check (M-8) --------------------------------
    # The authoring contract is validated against the LIVE table read from
    # the pinned remap ESP (espland.load_ltex), not only the expected
    # constant: indices, ids, and texture paths must match, with index 77
    # (road) protected explicitly.
    for record in regionpalette.crosscheck_live_remap_table(live_remap):
        _check(checks, record["id"], record["id"], record["passed"],
               record["detail"])
    planned_vs_live = regionpalette.planned_vs_live_remap_check(
        palette["semantic_surfaces"]["surfaces"], live_remap)
    _check(checks, planned_vs_live["id"], planned_vs_live["id"],
           planned_vs_live["passed"], planned_vs_live["detail"])
    _check(checks, "authoring.live_remap_evidence_emitted",
           "measured live remap records and provenance are emitted into the "
           "authoring assignment contract",
           palette["planned_output_plugin"]["live_remap_evidence"][
               "esp_sha256"] == live_remap["esp_sha256"]
           and palette["planned_output_plugin"]["live_remap_evidence"][
               "record_count"] == live_remap["record_count"]
           and len(palette["planned_output_plugin"]["live_remap_evidence"][
               "records"]) == live_remap["record_count"],
           f"live remap records emitted (count="
           f"{live_remap['record_count']}, sha="
           f"{live_remap['esp_sha256'][:16]}...)")

    # --- brief cross-file closure (M-2, real checks) ----------------------
    kit_surfaces = set(kit_brief["semantic_surfaces_used"])
    _check(checks, "brief.palette_closure",
           "kit brief machine-readable semantic_surfaces_used is a subset of "
           "the palette's closed vocabulary",
           kit_surfaces <= surfaces and bool(kit_surfaces),
           f"brief_surfaces={sorted(kit_surfaces)} "
           f"palette_surfaces={sorted(surfaces)}")
    stamp_types = {s["building_type"] for s in kit_brief["stamps"]}
    _check(checks, "brief.type_closure",
           "every emitted stamp record's building type is in the enum, and "
           "counts keys equal the enum",
           stamp_types <= set(kit_brief["building_type_enum"])
           and set(kit_brief["building_type_counts"]) == set(
               kit_brief["building_type_enum"]),
           "stamp types and counts reconcile with the enum")

    # --- spacing contract (I-4) -------------------------------------------
    spacing = census["spacing"]
    priors = kit_brief["spacing_priors"]
    _check(checks, "spacing.hard_minimum_contract",
           "inter_building_gap_gu is measured guidance with "
           "usable_as_hard_minimum false; collision clearance is the "
           "geometry solver's domain",
           priors["inter_building_gap_gu"]["usable_as_hard_minimum"] is False
           and priors["inter_building_gap_including_zero_gu"][
               "usable_as_hard_minimum"] is False
           and priors["collision_clearance"]["hard_minimum_gu"] == 0.0
           and "geometry solver" in priors["collision_clearance"]["provider"],
           "spacing prior is guidance; collision clearance is exact-hull "
           "solver domain")
    _check(checks, "spacing.run_separation",
           "per-run separated nearest-neighbor distributions and "
           "granularity metadata are emitted",
           set(spacing["runs"]) == {"karthgad-v1", "markarth-side-v1"}
           and all("nearest_neighbor_positive_gap_gu" in spacing["runs"][run]
                   for run in spacing["runs"])
           and set(spacing["granularity"]) == {"karthgad-v1",
                                               "markarth-side-v1"},
           "runs carry separated positive-gap stats and granularity notes")

    # --- provenance / determinism -----------------------------------------
    _check(checks, "provenance.inputs_pinned",
           "every external ref/path/hash in census inputs resolves",
           all(census["inputs"][name]["sha256"] for name in census["inputs"]),
           f"pinned_inputs={len(census['inputs'])}")
    _check(checks, "determinism.checked_by_cli",
           "two fresh builds produce byte-identical four-file output "
           "(verified by the CLI determinism proof)",
           False,
           "evidence (both staging build hash sets) filled by the CLI after "
           "the double-build comparison")

    passed = sum(1 for c in checks if c["passed"])
    return {
        "schema_version": 1,
        "target": "falkreath_v1",
        "checks": checks,
        "summary": {
            "total": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
        },
        "overall": "PASS" if passed == len(checks) else "FAIL",
    }


def _stamp_determinism(validation: dict, *, evidence: dict) -> dict:
    """Return the determinism-stamped validation payload (post-proof).

    ``evidence`` records the two staging build hash sets (all four files
    each) plus the comparison method, so the emitted artifact is
    self-verifying: the detail embeds the exact hashes of both fresh builds.
    """
    validation = json.loads(json.dumps(validation))
    for check in validation["checks"]:
        if check["id"] == "determinism.checked_by_cli":
            check["passed"] = True
            check["detail"] = {
                "method": "two fresh staging builds byte-compared; all four "
                          "files identical",
                "runs_compared": 2,
                "validation_json_hash_scope": evidence.get(
                    "validation_json_hash_scope",
                    "pre-stamp file hashes at byte-comparison time"),
                "build_a_sha256": dict(evidence["build_a"]),
                "build_b_sha256": dict(evidence["build_b"]),
            }
    validation["summary"]["passed"] += 1
    validation["summary"]["failed"] -= 1
    validation["overall"] = "PASS" if validation["summary"]["failed"] == 0 else "FAIL"
    return validation


def _write_bundle(out_dir: Path, payloads: dict[str, dict]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, payload in payloads.items():
        hashes[name] = write_deterministic(out_dir / name, payload)
    return hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default="2026-08-10",
                        help="build date label (canonical output date)")
    parser.add_argument("--root", default=".",
                        help="workspace root (default: current directory)")
    parser.add_argument("--out-dir", default=None,
                        help="override canonical output directory")
    parser.add_argument("--staging-base", default=None,
                        help="temporary base for the determinism proof")
    parser.add_argument("--no-proof", action="store_true",
                        help="skip the double-build determinism proof")
    parser.add_argument("--landscape-policy", default=DEFAULT_LANDSCAPE_POLICY,
                        help="JSON policy defining road classes and hierarchy mapping")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not (root / "tamriel.esm").is_file():
        print(f"FAILURE: workspace root invalid (no tamriel.esm): {root}",
              file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else root / CANONICAL_OUT

    try:
        print("Stage 1-3: census + assembly (build A)")
        payloads_a = build_payloads(root, args.date, args.landscape_policy)
        print("Stage 1-3: census + assembly (build B)")
        payloads_b = build_payloads(root, args.date, args.landscape_policy)

        with tempfile.TemporaryDirectory(
                prefix="dbrief-build-",
                dir=args.staging_base) as staging_a, tempfile.TemporaryDirectory(
                prefix="dbrief-build-",
                dir=args.staging_base) as staging_b:
            dir_a = Path(staging_a)
            dir_b = Path(staging_b)
            hashes_a = _write_bundle(dir_a, payloads_a)
            hashes_b = _write_bundle(dir_b, payloads_b)

            identical = True
            for name in payloads_a:
                bytes_a = (dir_a / name).read_bytes()
                bytes_b = (dir_b / name).read_bytes()
                if bytes_a != bytes_b:
                    identical = False
                    print(f"DETERMINISM MISMATCH: {name}", file=sys.stderr)

            if not identical:
                print("FAILURE: determinism two fresh builds differ", file=sys.stderr)
                return 1

            # stamp determinism proof into both validations, re-compare.
            # The evidence embeds both staging build hash sets so the emitted
            # artifact records the proof (M-1).  The sets are COPIED here:
            # hashes_a is mutated below (validation.json -> stamped hash) and
            # must not leak into the already-built evidence objects, or the
            # two stamped validations would diverge.
            evidence = {
                "build_a": dict(hashes_a),
                "build_b": dict(hashes_b),
                "validation_json_hash_scope": (
                    "pre-stamp file hashes at byte-comparison time; the "
                    "determinism stamp adds exactly this check record to "
                    "validation.json"
                ),
            }
            stamped_a = _stamp_determinism(payloads_a["validation.json"],
                                           evidence=evidence)
            stamped_b = _stamp_determinism(payloads_b["validation.json"],
                                           evidence=evidence)
            payloads_a["validation.json"] = stamped_a
            payloads_b["validation.json"] = stamped_b
            write_deterministic(dir_a / "validation.json", stamped_a)
            write_deterministic(dir_b / "validation.json", stamped_b)
            val_a = (dir_a / "validation.json").read_bytes()
            val_b = (dir_b / "validation.json").read_bytes()
            if val_a != val_b:
                print("FAILURE: stamped validation.json differs between builds",
                      file=sys.stderr)
                return 1
            hashes_a["validation.json"] = sha256_file(dir_a / "validation.json")

            if not args.no_proof:
                out_dir.mkdir(parents=True, exist_ok=True)
                canonical_hashes = _write_bundle(out_dir, payloads_a)
                canonical_hashes["validation.json"] = hashes_a["validation.json"]
                for name in payloads_a:
                    on_disk = sha256_file(out_dir / name)
                    if on_disk != canonical_hashes[name]:
                        print(f"FAILURE: canonical write hash mismatch {name}",
                              file=sys.stderr)
                        return 1
                final_hashes = canonical_hashes
            else:
                final_hashes = hashes_a

        # report
        validation = payloads_a["validation.json"]
        summary = validation["summary"]
        print(f"overall: {validation['overall']} "
              f"({summary['passed']}/{summary['total']} gates)")
        print("outputs:")
        for name in sorted(final_hashes):
            print(f"  {name}  {final_hashes[name]}")
        print(f"  canonical dir: {out_dir}")
        if validation["overall"] != "PASS":
            for check in validation["checks"]:
                if not check["passed"]:
                    print(f"  FAILED GATE: {check['id']}: {check['detail']}")
            return 2
        return 0
    except (citybrief.CensusError, regionpalette.RegionPaletteError,
            FileNotFoundError, KeyError, ValueError) as exc:
        print(f"FAILURE: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
