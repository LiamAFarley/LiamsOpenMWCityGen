# 2026-08-25 Terrain v3 Erosion Tools

## Purpose

This family is the targeted geomorphic stage after the v3 relief and broad
harmonic seam bridge. It operates on real configured solve regions and keeps
owner terrain immutable.

## Entry Points

- `tools/terrain/relief_preview.py` builds the relief-scaled target and runs
  AMG synthesis only for incomplete cells in the configured review frame.
- `tools/terrain/solve_region_v3.py` builds the production seam union and runs
  the broad direct harmonic bridge.
- `tools/terrain/erode_region_v3.py` routes the solved field through MFD
  erosion, renders cycle snapshots, and applies the narrow final seam lock.

## Command

```text
python tools/terrain/relief_preview.py --config <config.json> --gains 3 --region <region>
python tools/terrain/solve_region_v3.py --config <config.json> --region <region>
python tools/terrain/erode_region_v3.py --config <config.json> --region <region>
```

All terrain constants are read from the JSON config. The erosion stage uses an
8-neighbor MFD router, deterministic priority-flood routing copies, normalized
stream-power parameters, and a configured owner halo. Owner halo, exact seam,
and active-ring vertices are restored after every cycle.

## Outputs

The solve output root contains relief metrics/field products, harmonic target /
after / overlay renders, and an `erosion/` directory containing cycle renders,
the final render, metrics JSON, and the final field NPZ. These are diagnostic
terrain products only; VHGT authoring and decoded-plugin verification remain a
separate stage.

## Current Review Limitation

The first real run is the TR 61-cluster visible union. It is a review artifact,
not a claim that all Skyrim/Cyr seam regions are accepted. The broad straight
mountain/TR traces remain visible in the full-frame render and require visual
review before tuning or batch expansion.
