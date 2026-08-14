# Cityforge D-PLACE houses-only placement (T1.2)

## Scope

T1.2 consumes a **current, zero-error** T1.1 validation result and produces
host-side placement evidence for house/building stamps.  It does not solve
walls, docks, features, landscape edits, interiors, NPCs, or TES3 authoring.
The final authoring stage must consume the placement product only after the
T1.3 terrain stage has returned a final field.

## Transform contract

Every lot position is the seed-door plan-frame GU offset.  The solver adds the
accepted survey frame origin to obtain absolute world XY and samples anchor Z
from the dense field plus the source seed-door step.  For every member:

```text
R_source = engine_transform.tes3_euler_to_matrix(source_raw_tes3_euler)
R_yaw    = engine_transform.tes3_euler_to_matrix((0, 0, -radians(plan_yaw_deg)))
R_world  = R_yaw @ R_source
offset_world = R_yaw @ source_offset_gu
```

The negative raw `rz` above is only the OpenMW/TES3 encoding needed to obtain a
positive-CCW world yaw.  No `rz` values are added directly.  The authoritative
authoring field is `raw_tes3_rotation_rad`; optional Blender XYZ values are
stored under a render-only label.  Exact plan yaw is retained in the lot
record, and the solver never rotates a lot toward the nearest road.

Exterior buckets use mathematical floor: `floor(world_x_gu / 8192)` and
`floor(world_y_gu / 8192)`, including negative coordinates.

## Terrain and geometry decisions

The solver samples hull vertices, edge midpoints, every covered 128-GU field
node, member XY positions, and door positions.  It measures bilinear height,
analytic normal, local slope, best-fit footprint slope, relief, source-bounds
burial, and door steps.  A negative member-bottom clearance is source-burial
evidence, not an automatic floating-building error; the measured source burial
envelope is the hard limit.

Exact hull overlap or contact is the only hard inter-building spacing rule.
Dispatch-5 nearest-neighbour gaps and survey spacing are retained as warnings
only.  Mesh triangle/member-AABB collision is explicitly marked deferred when
the accepted libraries do not provide those surfaces/bounds.

`conform` lots with violations are rejected.  A `flatten_pad` lot can emit one
provisional request containing its exact hull, a conservative 256-GU margin,
target height, 512-GU falloff, measured cut/fill, and reason codes.  A planned
pad lot is never included as accepted.  A final run must point to the planned
placement and records the final terrain-field SHA-256; only a final field with
zero remaining terrain violations can accept the pad lot.

## Source replay and selector gates

The kit brief remains the selector vocabulary.  For implicit requests T1.2
recomputes the T1.1 smallest-hull-area / sorted-stamp-id selector and fails
closed if the result differs.  All 54 eligible stamps and all 736 eligible
members are replayed from hash-pinned source manifests in the synthetic proof;
source positions, raw rotations, scales, and source placement matrices must
match.  An independent 37° multi-axis matrix oracle includes source canary
`-102_11_ref_095307`.

## Proof command

```powershell
$env:PYTHONPATH = 'src'
python tools/cityforge/build_cityplace_fixture.py --workspace-root .
python -m unittest tests.test_cityplace -v
```

The output directory is synthetic and explicitly labelled; it is not a real
Falkreath design and contains no city render or ESP.
