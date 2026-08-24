# Building Generation Rule Kit Tools

Date: 2026-08-21

## Purpose

This Phase 0/1 tool family audits the immutable xFa house products, validates
the executable building-generation contracts, and rebuilds source-world member
graphs into template-local and ordered A-local relation products. It is
evidence preparation only. It does not compose buildings, profile meshes,
author plugins, or modify TownLayout/CityPlace.

## Entry Points

- `tools/cityforge/audit_xfa_source_products.py`
- `tools/cityforge/rebuild_xfa_relations.py`

Both tools are driven by `configs/kits/xfa_sky_nord_house/phase01_config.json`.
The config supplies site paths, output paths, tolerances, and render selection.
No site or model values are embedded in the reusable normalization/relation
modules.

## Inputs and Outputs

The four configured source pairs are read-only:

- Falkreath `stamp_library.json` and `grammar.json`
- Hal Norvold `stamp_library.json` and `grammar.json`
- Neugrad `stamp_library.json` and `grammar.json`
- Rimgrad `stamp_library.json` and `grammar.json`

The audit writes `output/cityforge/building_rule_kit_v1/audit/`/
`xfa_source_audit_v1.json` and the dated source-audit report. The rebuild writes
`derived/<site_id>/templates_v1.json`, `derived/<site_id>/connections_v1.json`,
`audit/roundtrip_report_v1.json`, and three Falkreath source-versus-rebuilt
side-by-side render pairs.

## Commands

```powershell
python tools/cityforge/audit_xfa_source_products.py --config configs/kits/xfa_sky_nord_house/phase01_config.json
python tools/cityforge/rebuild_xfa_relations.py --config configs/kits/xfa_sky_nord_house/phase01_config.json
python -m pytest tests/test_building_gen_contracts.py tests/test_building_gen_normalize.py -q
```

Use `--no-render` on the rebuild CLI only for local numeric debugging; the
normal command includes the required Blender render gate.

## Invariants

- Source-world offsets, template-local offsets, and A-local relation offsets
  are distinct fields and frames.
- All rotation conversion uses `src/procgen/engine_transform.py`.
- Authored scales are discrete source evidence and are not fit or recomputed.
- Bare touching witnesses retain null contact distance; no distance is invented.
- Current grammar aggregate rows are never copied into compiled relations.
- Round-trip failures are explicit rejection rows; no silent repair occurs.
- The final side-by-side PNG applies deterministic even-level 8-bit channel
  rounding only to suppress Eevee's one-level edge-rounding variance; it does
  not change scene geometry or source/reconstruction inputs.
- Canonical JSON uses sorted keys, two-space indentation, a final newline, and
  six-decimal float values with negative zero normalized.
