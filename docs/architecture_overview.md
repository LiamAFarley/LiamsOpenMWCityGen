# Procedural Tamriel — Architecture Overview

This document explains how a city is generated and where every piece of data
comes from.  It is the first thing to read in this core checkout.  The goal of
the whole pipeline is to **deterministically generate** the unpopulated parts
of Tamriel for OpenMW 0.51 (TES3/Morrowind engine): region-correct terrain
decor, roads, settlements, and authorable plugins — without any hand-authored
city geometry.

Everything is deterministic: identical inputs produce byte-identical outputs.
Measured facts and decisions are kept separate, and every transformation can
be replayed and verified.

The reference city in current work is **Falkreath** in `tamriel.esm` region
R072, around cells x=-95..-89, y=-11..-5 (a 7x7 cell site).

---

## 1. The data acquisition layer: reading the real game world

The pipeline does **not** invent terrain or roads.  It reads them from the
read-only source plugin and the mod data roots.

### 1.1 The ESM and its land records

`tamriel.esm` is Habasi's "Tamriel Full Map" — a master plugin containing the
landscape for the whole continent (32,086 land cells, x in [-251, 57], y in
[-122, 59]), with no statics of its own.  Its landscape is a regular grid of
CELL records, each holding:

- **LAND** subrecords: 65x65 height points (signed 16-bit, 8 game units per
  unit) plus a 64x64 texture (VTEX) tile grid with a texture index per tile.
- **LTEX** records: the texture definitions referenced by VTEX indices
  (landscape texture "landmasses").
- Only 160 cells carry region definitions; the rest are unpopulated.

Core readers:
- `src/procgen/espscan.py` — record-level scanning (record tags, sizes,
  cell refs, header).
- `src/procgen/espland.py` — LAND/VTEX/LTEX parsing, height fields, VTEX tile
  layout, `transpose_vtex_serialized_to_openmw` for authoring.
- `src/procgen/worldcontext.py` — the coordinate frame between the exported
  heightmap, cells, and game units.
- `src/procgen/coords.py` — the fixed grid constants and cell/pixel mapping.

Key constants (all measured):
- 1 LAND height unit = 8 game units (GU).
- Cell size = 8192 GU = 64 LAND points.
- Point `(px, py)` of the full-map heightmap maps to cell
  `((px // 64) - 251, (py // 64) - 122)`.

### 1.2 The site survey (D-SITE)

`tools/cityforge/build_site_survey.py` + `src/procgen/citysite.py`,
`espland.py`, `landroads.py` build a deterministic field bundle for the site
(the target plus a one-cell perimeter):

- Normalized **elevation field** from LAND heights.
- **Water / buildability masks** derived from heights (z <= 0 water) and the
  measured terrain envelope rules.
- The **road mask** — the exact set of VTEX tiles classified as road — and
  the derived canonical `land_roads.json` polylines.
- Four 4096x4096 real-texture planner views (top-down, elevation, roads,
  oblique) with an exact camera mapping recorded in `site_survey.json`'s
  `frame.render_mapping`.

Outputs (in the workspace, under `output/cityforge/sites/<site>/`):
`site_survey.json` (frame, inputs, region, field metadata, render mapping),
`survey_fields.npz` (the actual elevation/water/buildability arrays),
`land_roads.json`, and the four PNG views.  The survey is the *authority* for
everything downstream — plans and validators are pinned to its SHA-256.

### 1.3 Road centerlines (reconstructing the actual road network)

Vanilla/mod road geometry is not vector data, so the network is reconstructed:

1. **Source mask** (`src/procgen/road_source.py`,
   `tools/cityforge/build_tamriel_road_centerlines.py`) decodes the XCF
   source's `road network` layer with `gimpformats`, yielding a raster of
   road pixels aligned to the LAND texture grid.
2. **Repair** (`src/procgen/road_repair.py`) fixes endpoint-endpoint and
   endpoint-to-corridor T-junction gaps deterministically.
3. **Graph** (`src/procgen/road_graph.py`) skeletonizes the repaired mask into
   a node/edge graph.
4. **Vectors** (`src/procgen/road_vectors.py`) fits smooth polylines to the
   graph, then **registers** them (+4096 GU in X, measured) so they coincide
   with the in-game LAND/VTEX road occupancy.
5. **Consumption** (`src/procgen/aligned_roads.py`,
   `tools/cityforge/build_aligned_road_centerlines.py`) loads the aligned
   network, fail-closed (source-space coordinates refused, hash/translation/
   topology gates), and exposes lookup, rect queries, local frames, corridor
   width, nearest-centerline, and corridor polygons.

Road centerlines are the reference geometry that buildings and lots are placed
against.

---

## 2. Building stamps (produced beforehand)

Stamps are **unit stamps**: one JSON entry per building, encoding the
building's members (walls, doors, roof) in a building-aligned local frame.
They are derived *before* the planning stage from accepted extraction/split
products and are consumed as data:

- `tools/cityforge/stamp_library.py` derives stamps (member offsets anchored
  at the seed door, verbatim source Euler rotations/scales, measured terrain
  envelope, footprint/hull, access heading).
- `tools/cityforge/normalize_stamp_orientation.py` +
  `stamp_local_bounds.py` re-express stamps in **building-aligned frames**
  (`F = Rz(+theta)`, `offset' = Rz(+theta).offset`, `rotz' = rotz - theta`)
  so member boxes are near axis-aligned at yaw 0.
- `tools/cityforge/build_stamp_volumes.py` adds per-member 3D bounding boxes
  for Z-aware overlap checks.
- `src/procgen/citystamps.py` is the deterministic consumer API.

Each stamp's door members carry `outward_heading_deg`: the **geometric**
approach direction (the door box's thin axis, sign away from the building
centroid), which is what plan arrows and door-reach checks use.  Raw door rotz
is never treated as a facing.

Two libraries are used for Falkreath: `karthgad_nord_v2.json` (wooden North
houses) and `markarth_side_stone_v2.json` (stone houses).  They are large but
plain JSON, and `docs/guides/stamp_library.md` documents the format.

---

## 3. The planning bundle

`tools/cityforge/build_planning_bundle.py` precompiles one self-contained
input set per visual design session:

- **Planning canvas PNG** with a labelled GU graticule
  (`src/procgen/planning_canvas.py`, the shared projection used by later
  renderers).
- **`stamps.json`** — the eligible stamps only, with a preview contact sheet
  (fail-closed eligibility gate, `src/procgen/visual_planner_eligibility.py`).
- **`site.json`** — the clipped source roads and site geometry.
- **`rules.md`** — the fixed design rules for the session.
- A tool-written manifest.

The bundle is the single input to design; nothing downstream may silently
invent roads, spaces, stamps, or targets.

---

## 4. Frontage fitting: from intent to resolved placement

### 4.1 The intent (sketch) contract

An intent JSON is the authored design: `site`, `roads` (street/alley polylines
with widths), `spaces` (plazas/courts), `lots` (each with a `marker`, a
`stamp`, a `role`, and one or more door `frontages` declaring which door
should reach which target with which intent — public/service), and optional
`districts`.  `src/procgen/frontage_fit.py` strictly normalizes/validates it
(unknown keys rejected, exactly one primary frontage per lot, etc.).

Composition is declared with optional `road.purpose`,
`road.max_unsupported_frontage_gu`, `lot.intentional_outlier`, and `lot_groups`
(`character`, `lot_ids`, optional `shared_target_id`, `along_order`,
`max_span_gu`, `max_consecutive_gap_gu`, `max_consecutive_same_side`,
`plaza_sectors`) — the vocabulary consumed by the composition evaluator
(Section 4.3).

### 4.2 The fitter

`tools/cityforge/fit_intent_sketch.py` + `src/procgen/frontage_fit.py`:

1. Build the **world-target map**: source-road edges + authored road/space
   targets, in absolute world GU.
2. For each lot, project its marker onto its primary frontage target and
   **generate candidates** across along-offset x door-gap x yaw-perturbation
   (a bounded beam), transforming the stamp hull to each candidate position.
3. **Unary gates** reject candidates that fail geometry, terrain
   buildability, or door reach/facing limits — recording a rejection
   histogram per lot.
4. **Complete search**: an MRV/forward-check bitset search over the
   compatibility matrix (exact 2D hull overlap/contact) with a finite node
   budget.  Budget exhaustion is reported (`search_budget_exhausted`) and is
   never claimed as unsatisfiability.
5. Emits `intent.copy.json`, `resolved.sketch.json`, and `fit_report.json`
   (candidates, rejection histograms, selected centroid/anchor/yaw/door
   facts, terminal code).  Exit is non-zero unless the report status is
   `solved`.

`src/procgen/frontage_targets.py` provides the projection helpers
(`nearest_polyline`, `corridor_rings`).

### 4.3 Composition evaluation

`src/procgen/composition_eval.py` is a **pure, observational** evaluator: it
measures a resolved assignment against the authored composition vocabulary and
returns canonical metrics and findings.  It never selects candidates or
assigns status.  Hard bounds (road support intervals, group span, consecutive
gaps, order, same-side runs, plaza-sector occupancy) are mandatory gates;
preference profiles (`urban_unsupported_profile_gu`,
`compact_span_profile_gu`, `compact_gap_profile_gu`,
`irregular_repeated_gap_pairs`) compare arrangements once the hard bounds pass.
Outlier exemption applies only to the non-outlier-distance metric, never to
road support/span/order.

---

## 5. Plan derivation, validation, and rendering

### 5.1 Sketch → visual plan

`tools/cityforge/plan_sketch.py` turns a resolved sketch into the full
format-v1 visual plan: lot position is the **footprint centroid** (anchor
derived internally), automatic connection targets (snap radius 1536 GU),
door intents/access links via the library door transform, and one composite
PNG on the byte-identical planning canvas (kit-colored hulls, intent-colored
door arrows along the **measured door facing**).  `checks.json` carries only
hard errors plus door/space facts.

### 5.2 Validation

`tools/cityforge/validate_city_plan.py` + `src/procgen/cityplan.py` is the
strict D-PLAN gate: recursive unknown-key rejection, site/brief/palette pins
by exact SHA-256, capability gaps failing closed, aligned-centerline road
connections (source-space refused), exact yawed-footprint geometry, palisade
gates, linked terrain edits, closed texture zones, and a deterministic Pillow
overlay.  `src/procgen/schemas/city_plan_schema_v1.json` is the emitted
machine-readable schema.

`tools/cityforge/render_plan.py` renders the 2D plan overlay
(see `src/procgen/cityplan.py` for the render helpers).

---

## 6. Placement: engine-transform matrix space

`tools/cityforge/solve_city_placement.py` +
`src/procgen/cityplace*.py`:

1. Recheck the shared selector and replay every eligible source member.
2. Run the independent 37-axis matrix oracle.
3. **Seat exact plan yaws in engine-transform matrix space** using
   `src/procgen/engine_transform.py` — the single NumPy implementation of
   the engine composition `Rx(-rx) @ Ry(-ry) @ Rz(-rz)` (column vectors).
   Never re-derive rotation math inline (see
   `docs/guides/rotation_conventions.md`).
4. Measure dense-field terrain, door reach, and road access; reject exact
   hull overlap/contact.
5. Emit provisional pad requests with margin/falloff, and raw TES3 transform
   evidence — without authoring an ESP yet.

`src/procgen/cityplace_contracts.py`, `cityplace_geometry.py`,
`cityplace_transform.py`, `cityplace_output.py` split the contracts,
geometry, transforms, and output handling.

---

## 7. Landscape editing

`tools/cityforge/build_city_landscape.py` + `src/procgen/cityscape*.py`
(`cityscape.py`, `cityscape_edits.py`, `cityscape_field.py`,
`cityscape_output.py`, `cityscape_vnml.py`, `cityscape_vtex.py`):

- Stitch the site LAND block into a dense float64-GU field with exact seams
  and an immutable border.
- Apply strict analytic terrain edits (pads, flattening) and re-seat placed
  buildings.
- Paint deterministic explicit raw VTEX classes (with road/water/support
  gates), close local LTEX records, and validate masterless tes3conv JSON
  LAND records.
- LTEX/VTEX semantics are **plugin-local**: an override plugin that remaps
  landscape textures must carry its own LTEX records for every VTEX index it
  references.

---

## 8. Rendering

- `tools/cityforge/render_city.py` + `src/procgen/cityrender.py`: builds a
  deterministic render-only scene from the accepted final placement and
  exact terrain, copies only to a scratch masterless plugin, imports real
  resolved NIF/LTEX assets in Blender, applies exact placement matrices
  without normalization, and renders base + door-height detail views with
  camera LOS/finite-edge selection.  Fails closed on missing textures,
  matrix drift, terrain mismatch, or readability defects.
- `tools/cityforge/render_site.py` (site survey views) and
  `tools/cityforge/render_plan.py` (2D plan overlay) complete the render set.
- `tools/cityforge/blender_render_city.py` is the Blender-side helper.

These are the only stages that require Blender (`bpy`).

---

## 9. ESP authoring

`src/procgen/tes3json.py` is the TES3 plugin JSON authoring layer: it emits
the masterless plugin structure consumed by `tes3conv` (ESP/ESM ↔ JSON).
`tools/cityforge/door_tes3conv_proof.py` (`docs/guides/door_tes3conv_proof.md`)
is the reproducible proof that real TES3 DOOR records and linked
exterior/interior door pairs survive the JSON ↔ ESP round-trip.

Authoring rules:
- Generated plugins are **masterless**: `Header.masters: []`.  `tamriel.esm`
  is never declared as a master; the user loads it separately before the
  generated plugin.
- At the ESP boundary, plan-map yaw must be written as engine rotations:
  member `rotz = rotz' - yaw_map` (and positions composed with `Rz(-yaw)`),
  because the engine composes `Rz(-rotz)`.  This sign flip lives **inside**
  the ESP writer; everything plan-side stays map-CCW.
- An override plugin that remaps landscape textures carries its own LTEX
  records for every VTEX index it references.

---

## 10. How the data flows (quick map)

```
tamriel.esm ──► espscan/espland/worldcontext ──► site survey (fields+roads)
     │                                              │
     │                                              ▼
     ├──► road_source/repair/graph/vectors ──► aligned centerlines
     │                                              │
     │                                              ▼
mod data ──► stamp libraries (precomputed) ──► planning bundle (canvas,
     │                                   │      stamps, site, rules)
     │                                   ▼
     │                        intent/sketch ──► frontage fit ──► composition eval
     │                                   │
     │                                   ▼
     │                        visual plan ──► validate ──► place (matrix space)
     │                                   │
     │                                   ▼
     │                        landscape edits ──► render ──► ESP authoring
     └───────────────────────────────────────────────────────► masterless .esp
```

---

## 11. Invariants and conventions (bind all stages)

- **Determinism**: same inputs → same outputs.  Input hashes are recorded in
  every product; byte-determinism proofs are run before canonical writes.
- **Measurement vs. decision**: censuses and rankings are evidence, not
  decisions; category rules interpret them.
- **Fail-closed**: a stage that cannot prove its gate fails loudly rather
  than degrading.  Plans pin to accepted surveys/briefs by SHA-256.
- **Never modify originals**: `tamriel.esm`, mod data roots, and source
  plugins are read-only; experiments happen on scratch copies.
- **Rotation**: the one measured composition rule
  (`Rx(-rx) @ Ry(-ry) @ Rz(-rz)`, yaw-only `Rz(-rotz)`) is implemented once
  in `src/procgen/engine_transform.py` and used by all new cityforge code.
  See `docs/guides/rotation_conventions.md`.
- **Doors face geometrically**: `outward_heading_deg` is the approach
  direction; plan arrows and reach use it, never raw door rotz.

## 12. Where to look next

- `docs/guides/cityforge_visual_planning.md` — the planning canvas, intent
  arrows, eligibility.
- `docs/guides/cityforge_cityplan_validator.md` — the D-PLAN gate.
- `docs/guides/cityforge_cityplace.md` — placement details.
- `docs/guides/cityforge_cityscape.md` — landscape editing engine.
- `docs/guides/cityforge_render_city.md` — the render host.
- `docs/guides/tamriel_road_centerlines.md` — road reconstruction.
- `docs/guides/rotation_conventions.md` — the binding geometry rules.
- `tools/cityforge/` dated `*.md` notes — per-tool operational detail.
- `examples/` — synthetic contract fixtures.
