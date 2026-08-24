# 2026-08-23 Wall Walk-Anchor Tools

## `blender_mesh_walk_anchors.py`

Imports configured NIFs through the standard Blender mesh resolver and extracts
evaluated upward-facing surface clusters. A JSON job supplies every normal,
area, clustering, axis, mesh, and role value. It selects the largest horizontal
surface for a wall walkway and the lowest/highest tread clusters for a stair,
then writes native local-GU entry/exit and lateral bounds.

Input: `configs/kits/falkreath/wall_stair_anchor_measure.json` or another job of
the same schema. Output path is supplied inside the job JSON. This tool is the
measurement stage before semantic anchors are authored into a wall kit.

## `build_wall_stair_anchor_diagnostic.py`

Loads a validated wall kit and solves a minimal low-wall/stair/high-wall stamp
from its `walk_surface` and `stair_assembly` fields. It derives stair scale,
lateral lanes, deck overlap, pivot Z, and the successor wall bottom without
hardcoded kit dimensions. The output is consumed by
`build_wall_stair_transition_scene.py` and `blender_flat_render.py`.

The diagnostic deliberately contains no inverted wall or inferred supports.
Its purpose is to establish real tread-to-deck continuity before production
terrain integration.

