# 2026-08-12 stamp volumes tools

`build_stamp_volumes.py` derives, for EVERY member of EVERY stamp in the two
cityforge stamp libraries, a **stamp-local 3D bounding box** and a
**below-source-ground classification**, and emits one deterministic sidecar:

    python tools/cityforge/build_stamp_volumes.py \
        [--out PATH] [--karthgad-lib PATH] [--markarth-lib PATH]

`--out` defaults to `output/cityforge/stamps/stamp_volumes_v2.json`; the
library args default to the **v2 (building-aligned)** libraries.  Passing the
v1 library paths re-processes the legacy world-aligned libraries (the
pre-normalization behavior).  The output must not already exist (fresh-file
write; refuses to overwrite).

## Inputs (read-only)

* `output/cityforge/stamps/karthgad_nord_v2.json` (11 stamps) and
  `output/cityforge/stamps/markarth_side_stone_v2.json` (44 stamps) —
  building-aligned v2 libraries (member offsets/rotations already
  normalized; `normalization_theta_deg` present per stamp).
* Per-ref evaluated world bounds from the SAME extraction products the stamp
  library joined: the component manifests'
  `output/skyrim-settlements/{markarth-side-v1,karthgad-v1}/components/buildings/<slug>/manifest.json`
  `members[].world_bounds_gu`.  Markarth manifests are located via
  `components/buildings_index.json` `component_id -> slug` (each stamp's
  `source.component_id`); Karthgad uses `stamp.source.slug` directly.  Every
  manifest's SHA-256 is verified against the hash the owning library recorded
  in its `inputs` map (fail closed on mismatch).
* Per-model local bounds from each run's A2 evidence directory
  (`<run>/a2/`; `stamp_local_bounds.load_a2_local_bounds` — the OBB primary
  source, validated identical to the surface-geometry cache `local_bounds`).
* Source LAND terrain via `procgen.espland`: `Sky_Main.esm` (Karthgad) and
  `PTR Indev/Sky_Markarth.esm` (Markarth) — the same authority as the stamp
  library's re-measured terrain envelopes.

## Output

`stamp_volumes_v2.json`: `schema_version` (2), tool version, UTC timestamp,
input paths+sha256, then per library/stamp:

* `members[]` — `{source_id, model_key, structural_role, is_door,
  box_local: {min, max}, below_ground, measured: {top_z, terrain_min_z,
  terrain_max_z}, obb_source, obb_rotz_prime_deg, box_tight}` for v2
  libraries (see derivation rule 2);
* `above_ground_xy_boxes[]` — per-role merged XY AABB (stamp-local) over
  above-ground members only (role = `structural_role`, else `door` for door
  members, else `unassigned`);
* `sanity` — `union_vs_bounds_rel_max_dev_gu` (max per-axis deviation of the
  member-box union vs the library's `bounds_rel_gu`; measured ~1e-12 GU for
  both kits — both derive from the same member OBB corners).

Measured z values are source world GU; z comparisons are translation-
invariant so the classification is identical in stamp-local coordinates.

## Derivation rules

1. **v2 stamp space is building-aligned** (normalization T0.4b): the v2
   library's member rows already carry normalized offsets/rotations, so
   `box_local` is the AABB of each member's OBB in that frame.  The OBB is
   composed by `stamp_local_bounds.member_obb_corners_rel` (called with
   theta=0 — the rows are already normalized): model local XY bounds x
   scale, rotated by the member rotz, translated by the member offset.
2. **OBB primary source + fallback ladder**, recorded per member as
   `obb_source`: (1) `model_local_bounds` from the A2 evidence docs (matched
   by model_key + source SHA-256; 744/744 members on 2026-08-12, 0
   fallbacks); (2) `uninflated_world_aabb` (numeric solve of the 2×2
   rotation-AABB system from the member world-AABB dims + rotz;
   ill-conditioned when |cos 2φ| < 0.2); (3) `world_aabb` (conservative).
   `box_tight` = true when `obb_rotz_prime_deg` is within 0.6 deg of 0 mod
   90 (the AABB-of-OBB then equals the tight model box); non-cardinal
   members keep the AABB-of-OBB, documented per member.
3. **Below-ground rule (conservative)**: `below_ground = top_z <
   terrain_min_z` where `terrain_min_z` is the MINIMUM LAND height over the
   member's world XY footprint (65×65 field, 256 GU margin — the exact
   `citystamps._sample_field` parameters and rounding).  True only when the
   whole box sits strictly below every terrain seating at the source.  The
   measured top/min/max z are recorded either way.  Missing footprint LAND
   samples fail closed (never converted to sea level).
4. **Sanity (report-only)**: per-stamp union of member boxes vs
   `bounds_rel_gu` — measured ~1e-12 GU for all 55 stamps (both derive from
   the same member OBB corners).

## Invariants

Deterministic (canonical JSON, sorted keys; the UTC timestamp is the only
varying field); no random generators, no Blender, no TES3 authoring, no edits
to originals; every number traces to a measured cache value.  Coverage gate:
every member must yield a finite box (`FAILURE: stamp_volumes <stamp_id>
<reason>` listing missing `source_id`s; no fabricated boxes).  Legacy v1
libraries (no `normalization_theta_deg`) fall back to world-aligned stamp
space: `box_local = world_bounds_gu − anchor` (the pre-normalization
behavior).  Findings of the 2026-08-12 v2 run: zero below-ground members in
BOTH kits is genuine — the stamp library's `burial_depth_gu` is a whole-bbox
measure (terrain max vs building min z on a hillside) and does not imply any
member box top lies below the terrain min over its own footprint; the closest
call is `markarth_side_v1__u31_imperial_guilds` at a 0.43 GU margin (slab at
grade).
