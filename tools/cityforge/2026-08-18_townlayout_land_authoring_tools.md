# 2026-08-18 townlayout LAND authoring tools

`author_townlayout_land.py` consumes the circulation realization, seated
objects, an optional composed wall document, and a read-only source plugin,
then writes a masterless tes3conv JSON LAND product plus a sidecar authoring
manifest for affected cells. It preserves normals, vertex colors, world-map
data, flags, and untouched VTEX tiles.

When `--grading-policy` is supplied, all grading dimensions, grades, water-safe
floor, and cut/fill limits come from that JSON. Existing building lowering runs
first. Wall and slope footprints use the composer-published 5% bottom embedding
target. Broad roads receive a smoothed longitudinal profile with one height
across each section. Each gate uses the exact landing rectangle/elevation chosen
before gate placement, stays level across the complete entrance, and meets the
existing road through independently capped approaches. Shared LAND borders are
synchronized before serialization so grading cannot open terrain cracks.

Paint order is explicit: source raw-78 road tiles inside the city become raw
33 Sky grass first; authored broad roads/civic polygons and the wall-derived
gate approach rectangles then become raw 78 HR road. Narrow alley/apron geometry
remains deferred to terrain-following mesh authoring. The output is preparation
data for the scatter/render pipeline and is not yet an ESP.
