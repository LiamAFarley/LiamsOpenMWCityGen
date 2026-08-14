# Cityforge D-PLAN validator + overlay (T1.1)

**Family:** Cityforge plan contract
**Entry points:** `tools/cityforge/validate_city_plan.py`,
`tools/cityforge/render_plan.py`, `tools/cityforge/build_cityplan_fixture.py`
**Core logic:** `src/procgen/cityplan.py`
**Schema:** `src/procgen/schemas/city_plan_schema_v1.json` (draft 2020-12,
emitted by `cityplan.emit_json_schema()` — the file and the emitter cannot
drift; `tests/test_cityplan.py::SchemaEmitTests` proves they agree)
**Date:** 2026-08-11 (substage T1.1 of Cityforge dispatch 6)

---

## 1. Pipeline position

```
site survey (T0.2) -> stamp libraries (T0.3) -> kit brief / palette (T0.5)
   -> [T1.1 THIS FAMILY: strict plan gate + 2D overlay]
   -> placement solver (T1.2) -> landscape editor (T1.3) -> authoring (T1.4)
   -> review renderer (T1.5) -> first real Falkreath plan (T1.6, user gate)
```

T1.1 implements the **plan contract only**: it takes one declarative
`city_plan.json` plus the accepted planner-input bundle and returns a
deterministic structured issue list (`error`/`warning`, code, JSON path,
message, measured/limit) plus a summary and input hashes. It never authors
a plan, never runs placement, and never writes plugin data. The first real
Falkreath plan is a lead-driven T1.6 task; this family is proven on the
synthetic fixture only (see section 6).

## 2. What the validator enforces (in one paragraph)

Everything in the D-PLAN spec v1 (`2026-08-09_dplan_city_plan_schema_spec.md`)
as corrected by the dispatch-6 plan and request documents:

1. **Strict schema** — recursive unknown-key rejection at every level,
   types, enums, required fields, non-empty ids, and NaN/Infinity
   rejection (bool is never a number). `lots` is the only mandatory
   non-empty array.
2. **Frame pin** — the plan frame must match the accepted site survey
   exactly (origin `[-778240, -90112]`, units `game_units`, the survey's
   yaw-convention text, and the survey file's exact SHA-256), and the
   settlement block must match the survey's seed settlement.
3. **Vocabulary pins** — `building_type`/`size_class`/`stamp_id` come from
   the accepted kit brief (54 eligible stamps) and the hash-pinned D-STAMP
   libraries; surface names come only from the region palette's closed
   `semantic_surfaces` vocabulary; capability gaps fail closed (`lodge`,
   `stone_wall`, `fence` — the last two because no spacing rule is
   measured). Roads of class `street`/`approach` must keep the protected
   raw-78 `road` surface.
4. **Road network** — `connects` references resolve to plan roads, gates,
   real **aligned-centerline** edge/node ids
   (`tamriel_aligned_centerlines_v1.json` membership, loaded through
   `src/procgen/aligned_roads.py`), or **measured map-edge exits**
   (`exit_<side>_<edge_id>`, measured by clipping each aligned smooth
   polyline to the site rectangle in plan-frame GU).  The loader fails
   closed on the source-space bundle, hash drift, translation/topology
   drift, and coordinate invariants; the aligned product hash appears in
   `input_hashes` and `summary.external_references`.  Every road component
   must reach at least one external element (no orphan streets); polyline
   self-intersection and self-connection are errors.  Direct LAND/VTEX-78
   is the in-game occupancy authority; the source-space bundle and the
   XCF/BMP are provenance only; old `roads_graph_clean.json` and
   raw-78-only `land_roads.json` geometry are never consumed.
5. **Lot geometry** — the door anchor must be in scope and on a buildable
   tile (docks are the only water-position exception, and they are
   features, not lots). Explicit `stamp_id` requests get **exact
   transformed-footprint checks**: hull rotated by yaw about the anchor
   (standard CCW matrix, plan frame), in-scope, buildable/water tile
   coverage, and pairwise strict-overlap. Non-explicit requests are
   resolved by the documented **shared deterministic selector** (section 3)
   and then checked exactly; no candidate is a hard failure; unresolved
   geometry is never claimed checked (`geometry_checked` is reported per
   lot).
6. **Boundaries** — palisade rings must be explicitly closed simple rings;
   gates must lie on the ring (<=128 GU) and have a planned road within
   512 GU. `stone_wall` and `fence` fail closed.
7. **Terrain edits** — inside target cells, linked to existing plan ids
   (no orphan terraforming), target/falloff finite; a soft proxy warning
   fires when |target - surveyed cell median| exceeds the +/-1016 GU
   per-vertex delta encoding bound (exact deltas are solver-domain).
8. **Texture zones** — closed vocabulary, weights sum to ~1 (hard band
   0.5..1.5, warning outside 1.0 +/- 0.01), and zones must never paint
   the protected `road` surface.
9. **Soft diagnostics** (warnings, never fatal) — door-to-road distance
   > 1500 GU, door heading deviating > 90 deg from `face_road`, slope risk
   (cell mean slope above the conform limit without `flatten_pad`),
   measured spacing guidance below p10 (never a hard minimum; strict
   polygon overlap is the only hard spacing rule, 0.0 GU hard minimum with
   the 0.25 GU contact epsilon reported as `footprint_touch`),
   `max_cut_fill_gu` above the surveyed site constraint,
   dock far from water, external reference far from its road.

The issue list is deterministic (sorted by path/code/message) and complete
(no early exit). Invalid plans produce no trusted validated-plan artifact.

## 3. The shared deterministic selector

For a lot request without `stamp_id`, the candidate set is every eligible
kit-brief stamp matching `building_type`, plus `size_class` and
`multi_shell` when the request constrains them. The selector picks the
candidate with the **smallest `footprint_hull_area_gu2`** (best fit to a
compact lot), ties broken by sorted `stamp_id` — the same sorted-stamp-id
tie-break D-PLACE declares, without D-PLACE's per-lot seeded ranking (that
ranking is solver-stage T1.2). The choice is reported per lot as
`resolution: "selector"`; the planner remains free to make it explicit.
This is the documented shared selector T1.2 will default to.

## 4. The overlay renderer

`render_plan.py` renders only a **validated** plan (it runs the strict
validator internally and exits 1 on any error, writing nothing). It
composites on the accepted 4096x4096 `site_topdown.png` using the survey's
exact GU<->pixel mapping (`frame.render_mapping`), then draws, in order:

- faint corrected-centerline edges inside the site (context),
- translucent district polygons + labels (+ texture-zone line),
- roads with class colors and widths, direction arrows, ids, and
  external-connection markers (at the road endpoint nearest each external
  ref),
- exact yawed footprints for every resolved lot (explicit **and**
  selector), door anchors, access-heading arrows, id labels on white
  boxes, and orange warning markers for lots with warnings,
- boundary rings + gate diamonds, feature markers per kind (dock, well,
  statue, market, boat, signpost, keep-trees), terrain-edit polygons
  (translucent fill), wilderness hints (dotted rings + labels),
- a fixed legend band below the map and an optional banner band above it.

The canvas is 4096 wide and 4096 + banner(56 px) + legend(232 px) tall;
banner and legend live **outside** the map band, so no title or legend
pixel can cover planned geometry (`geometry_under_bands_px` is always 0 in
the audit, verified per run). The banner is opt-in via `--banner-text`;
the synthetic fixture run passes
`SYNTHETIC VALIDATION FIXTURE - NOT A FALKREATH DESIGN`.

**Determinism:** identical inputs produce a byte-identical PNG (Pillow's
PNG encoder is deterministic; fonts are Pillow's embedded font). The
render audit JSON records every input hash, the output PNG SHA-256,
drawn-element counts, and the band geometry.

**Implementation notes:** translucent fills are drawn through
`_composite_polygon` (temp layer + `alpha_composite`) because
`ImageDraw` on an RGBA image *replaces* pixels including alpha — direct
fills would erase everything underneath (this was found and fixed during
fixture inspection).

## 5. Commands

```text
# validate the synthetic fixture (canonical bundle defaults)
python tools/cityforge/validate_city_plan.py \
    --plan output/cityforge/phase1/t1_1_validation_fixture/synthetic_not_a_falkreath_design.city_plan.json

# render the overlay (validates first; refuses invalid plans)
python tools/cityforge/render_plan.py \
    --plan <city_plan.json> --out overlay.png \
    --banner-text "SYNTHETIC VALIDATION FIXTURE - NOT A FALKREATH DESIGN" \
    [--audit-out render_audit.json]

# (re)build the synthetic fixture and self-validate it
python tools/cityforge/build_cityplan_fixture.py

# (re)emit the machine-readable JSON schema
python tools/cityforge/validate_city_plan.py --emit-schema <path>
```

Exit codes: `0` = valid / rendered; `1` = plan invalid (nothing rendered,
no trusted artifact); `2` = configuration/bundle failure. All bundle paths
default to the canonical accepted files and can be overridden with
`--site-survey --kit-brief --region-palette --stamp-libraries --centerlines`.

## 6. The synthetic fixture (proof, not design)

`output/cityforge/phase1/t1_1_validation_fixture/` holds
`synthetic_not_a_falkreath_design.city_plan.json` (banner-labelled in the
overlay, manifest, and `design_notes`), built deterministically by
`build_cityplan_fixture.py` on a fixed 8192-GU lattice. It contains:

- 12 lots: 8 explicit stamp requests + 4 selector-resolved requests
  (house small/medium/large, tavern, smith, hall, shop, stable, manor,
  mill, farm, guild);
- 6 roads connected to real aligned-centerline ids: the junction node
  `road_node_fe5ab61f1218c960` inside the site, the aligned edge id
  `road_edge_31938970a750dc24` (after the +4096 GU registration correction
  that road no longer crosses the site border, so no west exit exists to
  name — see the road-authority investigation section 5.2), and the
  measured aligned exits
  `exit_south_road_edge_f200c85cfe673343`,
  `exit_east_road_edge_f36abb2dc60cb6fc`,
  `exit_south_road_edge_ed14e373290dcd8f`;
- a measured-capability palisade ring with 3 gates, each on the ring and
  within 512 GU of a planned road;
- a dock feature in the water mask (the only water-position exception),
  a well, a signpost, keep-trees;
- 2 terrain edits linked to plan elements; 3 closed-vocabulary texture
  zones; 6 districts; 2 wilderness hints.

It validates with **0 errors** and a known warning set (door distance /
heading deviation / slope risk) that exercises the soft-diagnostic path.
The fixture is regenerable and its manifest records the expected content
counts and input hashes.

## 7. Tests

| File | Covers |
|---|---|
| `tests/test_cityplan.py` (78) | strict schema (unknown keys at every level, types, enums, required, NaN/Inf, bool-as-number), frame pin, settlement pin, vocabulary pins, capability gaps, references, boundaries, zones, schema emitter agreement, determinism |
| `tests/test_cityplan_geometry.py` (50) | frame conversion, ring validity (bowtie/degenerate), overlap classification (disjoint/containment/rotated/touching), contact epsilon, footprint scope/water/buildable failures, pairwise overlap vs touch, selector (smallest-area, multi-shell, no-candidate, determinism), road graph (reachability, orphan, unknown ref, self-connect, loops, self-intersection, degenerate), soft diagnostics, dock exception, malformed data (NaN in every section must not crash) |
| `tests/test_cityplan_validate_cli.py` (11) | real-bundle CLI: fixture exit 0, summary counts, two-run byte determinism, invalid plan exit 1 + no trusted artifact, schema emit, render exit 0/determinism/fail-closed/banner pixels |
| `tests/test_cityplan_render.py` (5) | band layout, banner, byte determinism, overlay content, background-size mismatch fail |

Run: `python -m pytest tests/test_cityplan.py tests/test_cityplan_geometry.py
tests/test_cityplan_validate_cli.py tests/test_cityplan_render.py`

## 8. Known limitations (honest)

- The `edit_delta_proxy_exceeds_encoding_bound` check compares absolute
  targets against *surveyed cell medians* — an approximation; the exact
  per-vertex delta is computed at placement (T1.3).
- Road width/margins and building-to-road corridor separation are
  D-PLACE pass-4 concerns and are not checked here.
- The selector is validator-side; the planner may override any selector
  pick by making the request explicit.
- `road_external_ref_distance` (soft) uses the referenced edge's polyline
  start point as the anchor; for very long edges this is only a sanity
  bound, not a connection proof.
