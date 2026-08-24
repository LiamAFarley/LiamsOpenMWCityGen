# 2026-08-18 Kit house grammar tools

Tools for mining exterior house grammars from D-STAMP libraries and generating
new stamp-compatible buildings with Blender sheet renders.

## mine_house_grammar.py

Derives `configs/kits/<kit_id>/house_grammar_v1.json` from any stamp library.

```powershell
python tools/cityforge/mine_house_grammar.py `
  --library output/cityforge/stamps/markarth_side_stone_v2.json `
  --kit-id stone `
  --out configs/kits/stone/house_grammar_v1.json `
  --date 2026-08-18
```

Output: shell catalogs, door/window/chimney slots (shell-local poses), attachment
catalog, joined-block patterns, hard constraints.

## Falkreath (no stamps)

`tools/cityforge/generate_fk_house.py` + `src/procgen/fk_house.py` assemble
`sky_FK_*` shells from `configs/kits/falkreath/kit_bounds.json`. See
`tools/cityforge/2026-08-18_fk_house_tools.md`.

## generate_house.py

Synthesizes one stamp-shaped building JSON from a grammar + source library (for
model_key lookup).

```powershell
python tools/cityforge/generate_house.py `
  --grammar configs/kits/stone/house_grammar_v1.json `
  --library output/cityforge/stamps/markarth_side_stone_v2.json `
  --shell sky_ex_mk_h_m_02 `
  --seed 1 `
  --out output/cityforge/stamps/generated/stone/house_seed0001.json
```

Options: `--door-slots door_0 door_1`, `--stamp-template template_1`,
`--no-windows`, `--no-chimney`, `--stairs on|off|auto`,
`--window-facades pos_x,neg_y`.

## render_generated_house.py

Converts a generated stamp to `sheet_scene.json` and invokes `tools/mesh_thumbs.py`
(Blender headless) for a 2304×1536 `sheet_2x3.png`.

```powershell
python tools/cityforge/render_generated_house.py `
  --stamp output/cityforge/stamps/generated/stone/house_seed0001.json `
  --out output/cityforge/stamps/generated/stone/house_seed0001_sheet_2x3.png
```

Use `--scene-only` to skip Blender when debugging scene JSON.

## generate_house_milestones.py

Runs the Phase-4 acceptance batch (8 houses + renders) into
`output/cityforge/stamps/generated/stone/milestones/`.

## Core module

`src/procgen/kit_house_grammar.py` — mining, generation, scene conversion.

## Tests

`tests/test_kit_house_grammar.py`
