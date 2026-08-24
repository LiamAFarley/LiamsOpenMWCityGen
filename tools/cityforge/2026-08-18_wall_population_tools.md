# 2026-08-18 Wall-aware population tool

`build_town_wall_population.py` consumes the accepted `r2c_minor_roads`
checkpoint plus the Falkreath kit/town briefs. It preserves the road and wall
geometry, splits development blocks at the planning wall, assigns dense
Markarth frontage inside and sparse Karthgad frontage outside, and writes
`wall_front_rows.json` with a same-extent terrain diagnostic.

The tool is the current R5 entry point. It does not place rear rows, derive
final parcels, author meshes, or modify the accepted road network.

