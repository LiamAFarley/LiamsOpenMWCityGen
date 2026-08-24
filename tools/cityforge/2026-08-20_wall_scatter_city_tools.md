# 2026-08-20 Wall + Scatter City Render

`render_wall_scatter_city.py` appends a composed wall document and a regional
scatter document to an existing city flat-render scene. It preserves the
scene's LAND/VTEX road-retexture plugins and terrain anchor, converts wall
plan-frame offsets through the survey origin, filters scatter to loaded terrain
cells, rejects measured scatter footprints intersecting the wall meshes or
fitted inner-wall domain, and invokes `blender_flat_render.py` with all statics
enabled. When `--filtered-scatter` is supplied, the same filtered document is
also written for downstream scatter plugin authoring.

Example:

```text
python tools/cityforge/render_wall_scatter_city.py \
  --base-scene <city scene JSON> \
  --wall <composed wall JSON> \
  --scatter <regional scatter JSON> \
  --survey <site survey JSON> \
  --city-layout <wall-aware city layout JSON> \
  --output-scene <output scene JSON> \
  --output-png <output PNG>
```

Add `--close-up` to frame only the building/wall subject rather than the full
terrain extent.

`render_wall_mesh_layout.py` overlays the fitted wall's measured z-slice
footprints on an existing `city_layout_terrain.png`, using the same survey
mapping and wall-kit piece profiles.

The tool is render-only: it does not modify source scenes, LAND records,
scatter documents, or plugins. Wall offsets are expected to carry
`origin_gu`; scatter positions are expected to be global TES3 GU.
