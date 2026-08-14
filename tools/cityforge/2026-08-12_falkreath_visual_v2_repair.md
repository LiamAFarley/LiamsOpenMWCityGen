# 2026-08-12 — Falkreath visual v2 repair authoring tool

`author_falkreath_visual_v2_repair.py` is the bounded design-stage author for
the post-restart Falkreath visual-plan repair.  It consumes the accepted D-SITE
survey, both hash-pinned D-STAMP libraries, and the Markarth eligibility
palette.  It writes a visual-plan extension, a parallel T1.1 `city_plan.json`,
and machine-readable stamp/circulation/lot evidence into a caller-selected
repair directory.

The script contains fixed, inspected coordinates for a compact sequence of
aligned regional approaches → short streets → market/plaza → civic rear lane
→ small courts/alleys → sparse timber edge.  It does not use cell quotas,
ASCII layout, nearest-point bulk placement, rejected-v1 geometry, Blender,
T1.2 placement, terrain editing, or plugin authoring.  D-STAMP hulls and door
members are loaded rather than duplicated.  A seeded `random.Random(20260812)`
reproducibility check is used only for the deterministic authoring contract;
no geometry is randomly sampled.

Use from the workspace root:

```powershell
python tools/cityforge/author_falkreath_visual_v2_repair.py `
  --out-dir output/cityforge/plans/falkreath_visual_v2_repair
```

The visual document must still pass `visual_planner.py` structural,
eligibility, aligned-road, terrain, and advisory gates before rendering.  The
T1.1 export is validated separately with `validate_city_plan.py`; it is a
validation handoff, not a placement result.

## Render ledger

The author writes a truthful JSON ledger after render evidence exists. The interrupted canonical v2 directory is recorded as rejected intermediate evidence; only the inspected fresh repair set is counted as successful.
