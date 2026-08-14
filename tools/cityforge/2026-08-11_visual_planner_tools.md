# 2026-08-11 visual planner tools

`visual_planner.py` is the documented entry point for the deterministic Pillow
visual-planning worker. It loads the accepted site/stamp/aligned-road products,
validates the versioned visual-plan extension, calls the advisory analyser, and
writes a PNG plus advisory/render manifests. `--proof` writes only the labelled
synthetic proof in `output/cityforge/phase1/visual_planner_fixture/`.

The reusable implementation is split into:

* `src/procgen/visual_planner_terrain.py` — exact survey loading, plan-frame
  rectangle mapping, smoothed visual material sampling, hillshade, water,
  contours, slope, and context inset;
* `src/procgen/visual_planner_symbols.py` — deterministic layer order, aligned
  road/corridor drawing, transformed D-STAMP hulls and all doors, collision-aware
  local labels, capped dashed access stubs or explicit routes, selected-lot
  highlight/evidence panel, legends, and advisory markers;
* `src/procgen/visual_planner_format.py` — strict extension v1 shape gate and
  deterministic JSON serialization;
* `src/procgen/visual_planner_advisory.py` — non-solving hard-error/advisory
  analysis with measured geometry and override reasons.
* `src/procgen/visual_planner_eligibility.py` — fail-closed accepted-palette /
  D-STAMP metadata gate; rejected/quarantined IDs cannot be rendered as normal
  planning examples.

No tool in this family reads XCF/source-v1 road coordinates, writes a plugin,
or invokes Blender. The proof worker writes a distinct adversarial fixture and
uses the explicit `adversarial_proof` path only for that labelled image; normal
plans refuse hard errors. Door callouts are short/local and access relationships
are never inferred as full straight lines. Run focused non-rendering tests
before one bounded proof pass and inspect all four final PNGs; hashes alone are
not visual review.
