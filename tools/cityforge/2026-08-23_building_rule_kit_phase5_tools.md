# Building Rule Kit Phase 5 Tools

Date: 2026-08-23

## Purpose

`tools/cityforge/build_building_rule_kit.py` is the host-side Phase 5 compiler
driver. It reads the JSON-selected Phase 1-4 xFa products, calls the pure
compiler/palette modules, and writes one deterministic compiled rule-kit index,
eligibility report, and resolved policy product. It does not call Blender,
modify source evidence, compose buildings, or author plugins.

## Core Modules

- `src/procgen/building_gen/compiler.py` audits native profiles, facades,
  mounts, access bundles, roof/dormer relations, templates, connection samples,
  source terrain priors, and source D-STAMP shape without inventing missing
  provenance. Every row is retained with checks and stable rejection codes.
- `src/procgen/building_gen/palette.py` loads the strict Phase 0 palette shape,
  resolves settlement defaults -> district -> parcel overlays, rejects hidden
  fields/empty domains, and performs stable weighted choices from request
  identity and seed-derived subseeds.
- `configs/kits/xfa_sky_nord_house/phase05_config.json` supplies all input,
  review, tolerance, policy, request, and output paths. Values are not embedded
  in the compiler.

## Command

```text
python tools/cityforge/build_building_rule_kit.py \
  --config configs/kits/xfa_sky_nord_house/phase05_config.json
```

For a repeatability check, give two empty temporary directories through
`--output-dir` and byte-compare their three JSON files. The canonical products
are under `output/cityforge/building_rule_kit_v1/phase05/`.

## Output Contract

- `compiled_rule_kit.json` contains model, shell, mount, access, roof, dormer,
  template, connection, compatibility, terrain-prior, and source D-STAMP
  preflight indexes. Rejected rows stay visible; `selectable` lists only exact
  rows with no hard rejection code.
- `eligibility_report.json` contains concise per-row checks and rejection
  counts. It deliberately does not duplicate the full measured profile payload.
- `palette_resolution.json` contains each configured request's resolved policy,
  selected IDs, rate decisions, and all deterministic selection subseeds.

Direct attachment contacts use their exact source `sample_id` in the compiled
connection identity, preventing distinct witnesses with the same model pair
from collapsing into one selectable key.

## Limitations

Source xFa stamps are evidence, not generated D-STAMP output. The preflight
records whether source shape and terrain rows are present and whether door
provenance already exists; source template rows carry
`dstamp_provenance_missing` when it does not. This is a warning, not a claim
that generated output exists, and the compiler does not add destination blocks.
Actual generated door provenance, composition, placement, and visual acceptance
remain later phases.
