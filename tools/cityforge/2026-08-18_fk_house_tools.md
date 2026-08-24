# 2026-08-18 Falkreath kit house tools

Assemble simple Falkreath houses from `sky_FK_*` meshes using native NIF
AABBs. Texture variants `_a` / `_b` / `_c` share one family spec. `--window`
mixes any measured window onto any shell.

Wall measurement (Blender, engine-local origin) is documented in
`Documentation/guides/fk_house_wall_measurement.md`.

## measure_fk_kit.py

Native AABB dump (`normalize_to_position: false`).

```powershell
python tools/cityforge/measure_fk_kit.py
```

Writes `configs/kits/falkreath/kit_bounds.json`. Variant shells reuse the
family `_a` AABB.

## blender_fk_wall_slice.py

First-floor vertex AABB vs roof AABB. Use `wall_min_xy` / `wall_max_xy` for
rectangular cottages.

```powershell
blender -b --python tools/cityforge/blender_fk_wall_slice.py -- configs/kits/falkreath/wall_overhang.json
```

## blender_fk_l_wings.py

X/Y wall-plane clusters for L / U / recessed shells. Inner courtyard faces
are interior clusters, not the AABB.

```powershell
blender -b --python tools/cityforge/blender_fk_l_wings.py -- configs/kits/falkreath/house_wings.json
```

## generate_fk_house.py

```powershell
python tools/cityforge/generate_fk_house.py --all-pilots --render

python tools/cityforge/generate_fk_house.py --shell sky_FK_house_04_a --render
```

`--doors` / `--windows` are facade ids (`neg_x`, `inner_neg_y`, …). `--window`
is a window mesh id. `--secondary-doors` adds family-valid side/rear doors while
retaining the primary door; overlapping window slots are suppressed by the
generator. House 12 intentionally has no secondary-door candidates. Core:
`src/procgen/fk_house.py`. Families 01–12 (`_a` through lettered variants). House 08 wall slice is
`(160, 320)` GU; use inner clusters for brick. L-wings skips empty slices.
`--porch` plus `--porch-facades` adds a measured porch at generated doors;
all porch variants include integrated steps. `--stair sky_ex_mk_str_02
--stair-facades ...` adds the measured Markarth exterior stair to a bare door;
it cannot share a facade with a porch. The stair is seated at the door sill.
Sheets reuse
`render_generated_house.py`. `blender_fk_access_profile.py` reports internal
deck/landing planes when an accessory's total AABB is not a usable placement
reference.

Output: `output/cityforge/stamps/generated/falkreath/`
