# Rotation conventions (measured, binding)

Scope: all Procedural Tamriel geometry code that places, transforms, or
reasons about TES3 refs (stamps, placements, renders, ESP authoring).
Status: measured ground truth 2026-08-12; supersedes any older hypothesis.
Evidence run: `.opencode/runs/stamp-normalization-2026-08-12/2026-08-12_rotation_convention_fix.md`.

## The one rule

A TES3 ref with Euler rotation `(rx, ry, rz)` places its model as:

```
world = pos + Rx(-rx) @ Ry(-ry) @ Rz(-rz) . (scale * local)   (column vectors)
```

For yaw-only refs (the common case): `world = pos + Rz(-rz) . (scale * local)`.
Angles here are CCW-positive on the map (x east, y north). Note the MINUS:
the engine's stored rotation applies negated.

Proven 2026-08-12 against per-ref manifest `world_bounds_gu` (the evaluated
ground truth of the accepted extraction): for 140 pivot-offset members of
both stamp libraries, solving the true yaw from world-AABB centers vs
local-box centers gives residual cluster **0.972** (mean −1.7°, box-slack
noise) for `Rz(-rotz)` vs **0.099** (= random) for `Rz(+rotz)`.
Post-fix, all 744/744 stamp members recompose onto their manifest AABBs
with mean center deviation ≤ 8 GU and zero spill.

## Why numeric self-checks failed to catch the wrong sign

The AABB of a rotated box uses `|cos|, |sin|` — identical for `+φ` and `-φ`.
Any "union vs bounds deviation" or area check is **sign-blind**. Orientation
or mirroring claims require one of:

1. **Pivot-offset center test**: for refs whose model-local box center is
   off-origin (doors hinge at an edge, walls pivot at a corner), compare the
   *center* of the composed box against the ground-truth world AABB center.
   Wrong sign → center displaced by up to 2× the pivot offset.
2. **Visual verification** against an accepted render.

## Stamp library v2 frame (normalized)

Per stamp, θ = modal shell rotz (mod 180). The frame transform is
**F = Rz(+θ)** — the only choice consistent with both the measured
composition and "modal walls axis-aligned at yaw 0":

- `offset_gu' = Rz(+θ) . offset` (z unchanged)
- `rotz' = rotz − θ`
- member OBB corners: `offset' + Rz(−rotz') . (scale · local_corner)`
- `access_heading' = access_heading + θ`

Single shared implementation: `tools/cityforge/stamp_local_bounds.py`
(`member_obb_corners_rel`). Call it with the v1 member + θ (library build),
or with the v2 member + θ=0 (consumers of v2 libraries — the frame is
already applied). Do not re-derive this math in new tools.

## Door facing

Raw door rotz is NOT a reliable facing: mesh forward axes are model-specific
(e.g. `sky_ex_rm_door_01` faces local −Y), and a source door can even be
mounted facing inward. v2 library door members carry
**`outward_heading_deg`**: the geometric approach direction — the door box's
thin horizontal axis composed with `Rz(−rotz')`, sign chosen away from the
non-door body centroid; body-centroid radial fallback when the door box is
near-square (span ratio < 1.15). Each door records `heading_source` in
`normalization_facts.door_facts`. Plan arrows and door-reach use
`outward_heading_deg + yaw`. Whether the door *leaf* visually faces in or
out in-game is verified in-game (fly-around), not assumed.

## Plan/map space vs ESP space

Sketch/plan yaw is **map-CCW**: the plan renderer and checks apply
`Rz(+yaw)` to stamp-local geometry, and that depiction is self-consistent.
At the ESP authoring boundary the same yaw must be written as engine
rotations: member ref `rotz = rotz' − yaw_map` (and positions composed with
`Rz(−yaw)`), because the engine composes `Rz(−rotz)`. Keep this sign flip
contained in the ESP writer; everything plan-side stays map-CCW.

## History

- The 2026-08-09 extraction pipeline (Blender scene builds, split renders)
  uses the correct composition — user-confirmed visually correct.
- 2026-08-12 first v2 stamp normalization re-derived the math inline with
  two non-cancelling sign errors (`Rz(−θ)` offsets, `Rz(+rotz')` compose);
  invisible to sign-blind numeric checks; caught by user visual review of
  the 5-stamp geometry check; fixed and measured same day.
