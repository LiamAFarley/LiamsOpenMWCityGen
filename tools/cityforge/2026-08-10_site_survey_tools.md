# 2026-08-10 — Cityforge T0.2 site-survey tools (LAND-road correction)

## Pipeline position

These tools implement the visual survey checkpoint between source analysis and
future Cityforge planning/placement stages:

```text
read-only LAND + metadata
  -> build_site_survey.py / procgen.citysite.py + procgen.landroads.py
  -> land_roads.json + site_survey.json + survey_fields.npz
  -> render_site.py + blender_flat_render.py helpers
  -> four real-texture planner PNGs + render_audit.json
```

## `src/procgen/citysite.py`

The reusable host-side core streams selected LAND records, verifies the
remap-ESP heights against the selected `tamriel.esm` records, stitches a
449×449 float64 game-unit field at 128 GU spacing, calculates slope and
water-distance fields, and serializes 112×112 512-GU masks.  The road stage
now decodes the complete target-plus-one-cell perimeter directly from
`tamriel.esm` through the normalized `procgen.espland` VTEX view.  It emits
one row for every raw VTEX 78 tile, deterministic 8-neighbour components,
4-neighbour diagnostic statistics, confirmed boundary spans, and an explicit
ledger of unconfirmed edge tiles.  It does not open or use
`roads_graph_clean.json`.  The core also counts existing v6 scatter refs by
tile and applies the single anchored `(-92,-10)` grammar correction without
reformatting unrelated JSON bytes.

`procgen.landroads.py` is the focused source-analysis helper.  Its invariant
is that road topology is occupied LAND tiles only: no polylines, graph
junctions, headings, or synthetic centerlines are produced.  Perimeter
continuations use only the orthogonally adjacent outside tile with raw VTEX
78.  The canonical JSON embeds a 112×112 mask and source-count metadata; the
Cityforge core adds source paths/hashes, the remap-report count cross-check,
and the explicit `used_as_input=false` legacy-graph provenance row.

Inputs are read-only and no TES3 records are authored here.  The core returns
ordinary dictionaries/NumPy arrays to the driver and writes only the host-side
survey/evidence JSON and dense NPZ products under the requested output
directory.

## `tools/cityforge/build_site_survey.py`

Run from the workspace root with:

```text
python tools/cityforge/build_site_survey.py
```

It resolves the authoritative Falkreath source paths, invokes the core, writes
`output/cityforge/sites/falkreath_v1/land_roads.json`,
`site_survey.json`, and `survey_fields.npz`, and patches
`output/town_grammars.json` only when the stale `Farm`/`marker:0399` row is
present.  It fails closed if the raw-VTEX 78 count does not match the existing
remap report or if the render LAND VTEX differs from the authoritative
`tamriel.esm` target payload.

## `tools/cityforge/render_site.py`

Run:

```text
python tools/cityforge/render_site.py
```

The host command first resolves every used LTEX image under the configured
read-only roots.  Its Blender worker imports `blender_flat_render` terrain and
water helpers directly, so it bypasses only the shared non-empty-mesh input
gate; it does not edit that renderer.  Four 4096² Cycles CPU renders use the
real remap LAND textures and an exact z=0 plane.  The wrapper replaces the
shared helper's temporary rectangular water plane with terrain-triangle
clipping at `z<=0`, so perspective views cannot expose a rectangular water
skirt around elevated terrain.  A host-side annotation pass adds the
published affine grid/ruler mapping and elevation/hillshade tint.  The road
view dims the real terrain, maps each output pixel centre back to plan GU, and
uses a nearest lookup into the canonical 112×112 mask.  Its orange fill and
occupied-pixel-only outline cannot bridge an empty tile; cyan brackets/labels
identify only the perimeter-proven continuation spans.  `render_audit.json`
records camera data, scale, source terrain hash, texture-resolution evidence,
water z, before/after water extents and surface area, per-image SHA-256 hashes,
the measured GU→pixel→GU round-trip, and the road tile/pixel source agreement
audit.

The worker's temporary raw PNGs are deleted after annotation; the four final
PNGs are the required review artifacts.

## Validation

The focused tests are in `tests/test_citysite_survey.py`.  They do not launch
Blender; they validate the generated bundle, dense-field dtypes/shapes,
base64 masks, LAND road tile-to-plan coordinates, 8/4 component behavior,
perimeter-confirmed and rejected edge tiles, deterministic ordering, exact
render-mask mapping, source cross-check evidence, camera mapping, and
idempotent grammar patch behavior.  The full render command is the visual
checkpoint and must be followed by opening the complete-resolution
`site_roads.png`.
