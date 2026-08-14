# Cityforge visual planning (2026-08-11)

The visual planner is a render-only design aid between the accepted D-SITE,
aligned-road consumer product, and a future T1.1/T1.2 plan handoff. It does not
author TES3 records, modify LAND/VTEX, run Blender, or turn interpolated pixels
into game data.

## Pipeline

```text
site_survey.json + survey_fields.npz
  + aligned centerline API + D-STAMP libraries
  -> visual_plan_extension_v1 structural gate
  -> Pillow terrain/layer renderer
  -> separated hard-error/advisory report
  -> human visual review
  -> future T1.1 city_plan.json (not performed here)
```

`src/procgen/visual_planner_terrain.py` retains exact dense height, water,
buildable, road, and raw-VTEX arrays. Material colours are bilinearly
interpolated only for the image. Water and shoreline are nearest-sampled from
the exact survey vertex mask. `src/procgen/visual_planner_symbols.py` draws
aligned source roads/corridors, authored roads, alleys, courts, plaza
polygons, districts, stamp hulls, every measured door, per-door intent arrows,
source terrain/burial notes, cell labels, legend, and optional context inset.
The map body uses a deterministic collision-aware label placer: short local
labels are rejected if they would cover a footprint, door/arrow shaft, or
another accepted label, and long leader lines are never emitted. A selected lot
is described in a separate right-hand detail panel with full stamp/source
identity, kit/category, all doors and targets, headings, source slope/relief,
burial range, and recorded stair/access members. Roads use subdued surfaces
with independent edge/centreline strokes; alleys remain narrower, while
courts and plazas stay bounded translucent surfaces. Access links are either
short dashed stubs capped at 560 GU or explicitly supplied routed polylines;
the renderer never invents a straight route to a distant target centroid.

Before advisory analysis or rendering, the CLI loads
`src/procgen/visual_planner_eligibility.py`. The gate hashes and checks the
accepted Markarth palette and both D-STAMP libraries, rejects non-eligible
palette entries and library exclusions, and fails closed when a placement is
not in the accepted inventory. In particular, both Castle Barracks palette
records cannot enter a normal or synthetic clean proof. The proof's selected
lot is an eligible two-door Karthgad stamp and receives a bright local outline
glow plus an adjacent `SELECTED` tag; the arrows are composited above that
highlight so doors remain visible.

## Format and commands

The sibling format is `cityforge_visual_plan_extension` version 1. Its JSON
schema/structural gate is in `src/procgen/visual_planner_format.py`. Coordinates
are `site_survey_plan_gu`; source roads carry aligned edge IDs only. A compact
generic planner prompt is in `tools/cityforge/visual_planner_prompt.md`, and
the town-specific supplement is in
`tools/cityforge/visual_planner_town_supplement_template.md`.

Render one document:

```powershell
python tools/cityforge/visual_planner.py --plan <visual-plan.json> --out <canvas.png>
```

Build the only canonical proof:

```powershell
python tools/cityforge/visual_planner.py --proof `
  --proof-dir output/cityforge/phase1/visual_planner_fixture
```

The proof builder writes four labelled images and JSON manifests. It uses one
declared Pillow iteration (`iteration_count=1`, maximum 3), six varied stamps
from both accepted kits, four multi-door placements, one aligned source road,
one authored street, one alley, one shared court, one irregular plaza, and an
explicit slope-capable marker. The clean vocabulary is arranged as compact
road/court/plaza clusters rather than a full-canvas relationship diagram. The
adversarial image is generated from a distinct tandem/repetition/orientation/
road-overlap fixture and is rendered only through the explicit proof path;
normal rendering remains fail-closed.

## Advisory semantics

`src/procgen/visual_planner_advisory.py` never moves geometry. Physical overlap,
water/out-of-scope placement, road-corridor collision, unresolved stamp/door,
and disconnected declared connections are hard errors. Access, repetition,
orientation, tandem, frontage, open-space, terrain-envelope, and generic slope
findings are advisory. A generic 15-degree slope warning never becomes a hard
placement rejection. Advisory override reasons are preserved in the report.

## Limits

This renderer is not a layout solver and does not certify urban quality. The
proof is synthetic and explicitly not a Falkreath design. Do not use it as an
authoring plugin or substitute source `land_roads.json` for the aligned road
consumer API.
