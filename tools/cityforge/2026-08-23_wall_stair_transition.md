# 2026-08-23 Wall Stair Transition Tool

`build_wall_stair_transition_scene.py` converts a diagnostic stamp into a
direct-view Blender scene using a JSON camera profile. It preserves the stamp's
member placements and ground plane and is intended for side, top-down, or other
controlled inspection views where a generic 2x3 sheet hides seams or alignment.

Inputs are a stamp JSON, a profile JSON containing named camera views, and an
output scene path. The output is consumed by `tools/blender_flat_render.py`.

`build_wall_slope_anchor_diagnostic.py` builds the current minimal
wall–slope–wall connection from the wall-kit JSON. It uses the measured
walk-surface anchors, scales the source ramp uniformly to the adjoining wall's
measured walkway width, and snaps the resulting endpoints with zero overlap.
It contains no Falkreath placement coordinates.

`focus_wall_scene.py` crops a terrain-backed production scene around a member
role and arc distance. Its default role is `slope`, allowing close visual
inspection of real authored-ramp transitions without regenerating geometry.

`blender_mesh_walk_anchors.py` extracts evaluated walk surfaces from NIFs. In
`sloped_walkway` mode it groups connected coplanar upward faces, chooses the
inclined walking plane, and extends its semantic endpoints across adjacent
horizontal landings. In `segmented_sloped_walkway` mode it uses the lowest and
highest horizontal terminal landings of an authored multi-facet slope. The
resulting anchors drive composer placement rather than full-mesh AABB ends.
