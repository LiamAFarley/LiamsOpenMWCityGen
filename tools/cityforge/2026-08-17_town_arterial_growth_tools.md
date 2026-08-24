# 2026-08-17 · Town arterial growth tools (Stages A-W-C)

Stage A of the arterial growth plan rev2
(`.opencode/runs/townlayout-phase21-recovery/2026-08-17_arterial_growth_plan_rev2.md`,
including the §0 user amendments), followed by the accepted inner-wall ordering
amendment. Arterials, road blocks, the central wall, and minor roads are active.

- `tools/cityforge/build_town_arterials.py` — Stage A CLI. Verifies the frozen
  R1/R2 input hashes, builds the rooted arterial tree, writes
  `r2a_arterials/arterials.json` plus `arterials_topology.png` /
  `arterials_terrain.png` review renders, and proves determinism with one
  in-process rebuild. Output dir must be empty. Runtime gate 10 s.
- `src/procgen/townlayout/arterial_graph.py` — strict R2 input projection
  (gates/ring/junction only), centi-GU planar fine graph from exact patch
  polygons, gate-lead and center-lead (root attachment) terminal geometry.
- `src/procgen/townlayout/arterial_routes.py` — directed edge-state Dijkstra
  with chord/turn cost, rooted shared-tree construction with merge zone,
  fillet smoothing with corridor validation (dry + city-land covered),
  Stage A product assembly via `build_arterials(macro, ports)`.
- `src/procgen/townlayout/road_review.py` — same-extent topology/terrain
  review renderers; never draws source-road polylines. Draws the full cell
  fabric (every fine edge as a grey cell outline) plus an emphasized white
  city outer boundary (2026-08-18: replaces leaf-trimmed stub pruning, which
  hid gate-vertex context and left boundary cells with no outline).
- `tools/cityforge/build_town_road_blocks.py` — Stage B CLI. Reads only the
  accepted arterial checkpoint, subtracts the real corridor, and writes the
  deterministic block checkpoint plus topology/terrain review renders.
- `src/procgen/townlayout/road_blocks.py` — atomic-face accounting,
  arterial-safe adjacency, varied deterministic block growth, explicit
  isolation/verge records, and Stage-B geometry/distribution gates.
- `tools/cityforge/build_town_inner_wall.py` and `inner_walls.py` — freeze a
  contiguous central wall near two-thirds city area by eroding arterial-rooted
  outskirts; gates are exact major-arterial crossings only.
- `tools/cityforge/build_town_minor_roads.py` and `minor_roads.py` — grow
  wall-aware side streets after the wall is frozen, draw exact links to arterial
  centerlines, clip road texture at the wall, and retain waterfront landscape.
- `tests/test_townlayout_arterials.py` — synthetic mechanics regressions plus
  real-checkpoint mutation-invariance, determinism, and hard-gate tests.

User amendments in force: ports are city-boundary entrances; inner-wall gates
are separate later objects and only major arterials create them. The junction anchor contributes no road stub; gate
leads are exterior linkage (town corridor starts at the fine-graph attach
node, which carries the palisade gate marker and the 256-GU cap exception);
review renders draw the full cell fabric plus a white city outer boundary
(2026-08-18; replaces stub pruning); gate-lead attach
points are chosen by alpha+beta smoothness scoring over fine_shared
nodes/edge feet in [256, 4096] GU (not the first tangent-ray crossing), with
the 5-degree tangent residual reduced to a report-only metric; and used
arterial edges with < 2,048 GU roadside city-land depth promote adjacent
unselected dry fringe patches one cell deep (recorded as
`promoted_patch_ids`, max 2 routing passes).
