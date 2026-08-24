# Building Rule Kit Phase 2 Tools — Model Inventory and Profiles

Date: 2026-08-22

## Purpose

Phase 2 of the xFa building-generation rule kit (spec:
`.opencode/runs/2026-08-21-building-generation-rule-kit/2026-08-22_phase2_implementation_spec.md`)
builds the measurement inventory and native-scale evaluated model profiles
that later phases (wall/facade, roof, mounts, compiler) consume. It measures
meshes only; it does not compose buildings or change TownLayout/CityPlace.

## Entry Points

- `tools/cityforge/build_model_profiles.py` — host driver: builds the
  inventory, runs the Blender profiler once, merges eligibility, writes all
  four outputs.
- `tools/cityforge/blender_model_profile.py` — Blender-side job-JSON profiler
  (never run directly; spawned by the driver).
- `src/procgen/building_gen/inventory.py` — pure-Python inventory builder.

Driven by `configs/kits/xfa_sky_nord_house/phase02_config.json` (sites,
outputs, z-band fractions, bottom percentile, digest decimals).

## Outputs

Under `output/cityforge/building_rule_kit_v1/phase02/`:

- `model_inventory.json` — one row per distinct model: observed roles, sites,
  scales, stamps, contact neighbors, relation-rule references, classification
  authority, eligibility.
- `model_profiles.json` — Blender-measured rows: evaluated bounds, vertex/face
  counts, z-band ground polygon, bottom penetration evidence, principal XY
  axis (measurement only), geometry digest, `band_fallback` flag.
- `alias_evidence.json` — per suffix family, member geometry digests and
  `equivalent: true/false`. Equivalence requires identical digests; nothing is
  presumed from names.
- `rejection_list.json` — unresolved/profile-missing models (hard rejections)
  plus `semantics_*` rows labeled for review.

## Command

```powershell
python tools/cityforge/build_model_profiles.py --config configs/kits/xfa_sky_nord_house/phase02_config.json
```

Exit code 0 only when the Phase 2 gate holds: every observed model resolves
and has a profile row.

## Invariants

- Roles come from source stamp members only; no filename-substring
  classification.
- Footprint evidence uses the z-slice method (AGENTS.md binding convention);
  `band_fallback: true` rows are explicitly flagged full-z measurements.
- Alias equivalence is digest equality over order-independent evaluated
  geometry, never a name pattern.
- Outputs are canonical JSON and byte-identical across repeated runs with
  identical inputs.
