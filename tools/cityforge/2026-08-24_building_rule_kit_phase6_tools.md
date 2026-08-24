# Building Rule Kit Phase 6 Tools

Date: 2026-08-24

## Purpose

`tools/cityforge/build_building_composer.py` runs the bounded Phase 6 pure
composer against the real Phase 5 compiled xFa kit. It writes complete
composition-local generated stamps for exact observed-template replay and
single-shell primary access, then exercises a named secondary-access revision
and a blocked revision. It does not import Blender, mutate source evidence,
change TownLayout, author TES3 records, or render.

## Core Module

`src/procgen/building_gen/composer.py` owns the deterministic composition
contract. It reads selectable shell, mount, access, template, and native model
profiles from the compiled kit. Native ground polygons and all native bound
corners are transformed through `engine_transform.py`. Facade selection uses
the explicit request heading, measured usable regions, and occupied-region
evidence. Secondary revisions preserve prior member IDs, reject measured
occupied regions and configured proximity to existing generated doors, and
return the prior stamp unchanged when supplied free space cannot support the
candidate.

## Command

```text
python tools/cityforge/build_building_composer.py \
  --config configs/kits/xfa_sky_nord_house/phase06_config.json
```

Use `--output-dir` twice with separate approved temporary directories for the
byte-determinism check. The canonical products are under
`output/cityforge/building_rule_kit_v1/phase06/`.

## Output Contract

- `base_results.json` contains the observed-template and one-shell successful
  generated stamps.
- `extension_results.json` contains the accepted two-door revision and the
  unchanged blocked revision with explicit rejection evidence.
- `summary.json` contains only deterministic case counts and statuses.

Every accepted stamp passes `validate_generated_building`, including the real
`cityplace.validate_stamp_integrity` consumer. Every emitted door carries the
complete nullable destination shape. The outputs are composition-local and are
not yet registered with TownLayout or authored into a plugin.

## Limitations

This phase deliberately excludes facade windows, roof attachments, dormer
transfer, porches, tents, and new multi-shell structural graph composition. It
uses an explicit plane terrain context for numeric proof. The first visual
composer gate for generated stamp assembly is Phase 7; authoritative terrain
seating is deferred to the later integration gate.
