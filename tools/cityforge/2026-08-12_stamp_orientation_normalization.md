# 2026-08-12 — Stamp orientation normalization (Cityforge T0.4b)

`tools/cityforge/normalize_stamp_orientation.py` converts a v1 D-STAMP
library (world-rotated stamp space) into a **building-aligned v2 library**:
one theta per stamp (modal non-door rotz, mod 180, 0.5-deg buckets, majority
by member count, mean of the winning bucket rounded to 0.1 deg) defines the
building's natural axis, and every member is re-expressed in the frame rotated
by -theta about the seed-door anchor.  Shared OBB geometry lives in
`tools/cityforge/stamp_local_bounds.py` (also consumed by
`build_stamp_volumes.py`).

## Why this exists

v1 stamps preserve each building's SOURCE world rotation, so most sit at
non-cardinal angles (e.g. walls 40.1 deg); their stored `hull_xy_rel` /
`bounds_rel_gu` were axis-aligned boxes around the ROTATED building —
inflated and orientation-mismatched, producing diagonal-looking door arrows
and false collision margins.  After normalization yaw 0 IS the building's
axis-aligned pose: door arrows point cardinal at yaw 0 and hulls are tight.
Preview sheets still show the source orientation (they are renders of the
source scene; the rotation is recorded in `normalization_theta_deg`).

## CLI

```
python tools/cityforge/normalize_stamp_orientation.py \
  --in  output/cityforge/stamps/karthgad_nord_v1.json \
  --out output/cityforge/stamps/karthgad_nord_v2.json
```

`--out` must not exist (refuse-overwrite).  Kit (bounds authority) is
auto-detected from `library_id`.  Every manifest read is SHA-256-verified
against the v1 library's `inputs` map.  Exit 0 = v2 written (canonical JSON);
exit 1 = FAILURE, no output.

## Per-stamp transform

> **CORRECTED 2026-08-12 (lead, measured):** the original build used
> `Rz(-theta)` offsets and `Rz(+rotz')` composition — two non-cancelling
> sign errors that produced transposed/meaningless hulls for non-cardinal
> stamps (user-confirmed).  The measured engine convention is
> `world = pos + Rz(-rotz).(scale.local)` (140-member world-AABB-center
> measurement, cluster 0.972), which forces F = Rz(+theta).  Fixed behavior
> below; full evidence in `.opencode/runs/stamp-normalization-2026-08-12/
> 2026-08-12_rotation_convention_fix.md`.

- `offset_gu` xy → `rot2d_ccw(offset.xy, +theta)`; z unchanged.
- `rotation[2]` → `(rotz - theta_rad) mod 2pi`; rx/ry/scale verbatim.
- Door members rotate like all members.  The raw door rotz is NOT a reliable
  facing (mesh forward axis is model-specific), so each door member gains
  `outward_heading_deg`: the geometric wall normal (door box thin axis
  composed with Rz(-rotz'), sign away from the non-door body centroid;
  body-centroid radial fallback for near-square door boxes, span ratio <
  1.15; per-door `heading_source` recorded in `door_facts`).
  `door.destination_*` interior data is never touched.
- **Footprint (OBB composition):** per-member OBB =
  `offset' + Rz(-rotz').(scale * model local XY bounds)` —
  in the normalized frame (rotz' = rotz − theta).  `hull_xy_rel` =
  convex hull over all member OBB corners; `bounds_rel_gu`/`aabb_rel` = axis
  union — tight in the building frame since modal shell rotz' ≈ 0/90.
  `access_heading_rad` → `v1 + theta_rad`.
- Model local bounds: the run's A2 evidence documents
  (`<run>/a2/nif_<model>--*.json`), matched by (model_key, source SHA-256);
  validated byte-identical (≤3e-5 GU float noise) to the surface-geometry
  cache `local_bounds` on the 398-model Markarth overlap.  **Fallback ladder
  per member, each recorded in `normalization_facts.member_bounds_sources`:**
  (1) model local bounds; (2) numeric un-inflation of the member world AABB
  (solve `X = w|c|+d|s|, Y = w|s|+d|c|`; ill-conditioned when
  `|cos 2φ| < 0.2`); (3) the world AABB rotated by +θ (conservative).
  Coverage 2026-08-12: **744/744 members on model local bounds, 0 fallbacks.**
- Carry-over unchanged: `terrain_envelope`, `anchor`, `source`, `stamp_id`,
  `preview_sheet`, `style_tags`, `building_type`, `size_class`, `door_count`,
  `multi_shell`.  `normalization_theta_deg` + `normalization_facts` stored
  per stamp.

## Gates and facts (fail-closed; no output on any failure)

- **Replay**: per member, `anchor + Rz(-theta).offset'` must equal v1
  `anchor + offset` ≤ 1e-6 GU and `(theta + rotz'_deg) mod 360` must equal v1
  rotz ≤ 1e-9 deg.  Evidence embedded mirroring the v1 structure (per-stamp
  `replay` + `stats.replay`; v1 evidence preserved as `replay_v1`).
  2026-08-12: 744/744 pass; max position error 0.0 GU, max rotation error
  6.4e-14 deg.
- **Shell-modal frame assert** (trivially true by construction; kept): modal
  shell rotz' ≈ 0 mod 90 (±0.6 deg).
- **Door cardinality is REPORT-ONLY** (lead ruling 2026-08-12 — the original
  ±0.6 deg hard gate was a spec error): every door's residual vs the nearest
  cardinal direction is recorded in `normalization_facts.door_facts` and
  `stats.normalization.door_facts` with an `off_cardinal` flag (threshold
  0.6 deg).  Source-mounted skews (0.6–3.1 deg) and structural diagonal
  doors (round towers/windmills, up to ~35 deg) are expected source geometry.
  2026-08-12: 20 of 70 doors off-cardinal (5 karthgad, 15 markarth).
- Cross-check fact (informational): seed-door-implied wall
  `(door rotz - 90) mod 180` vs shell theta; deviation > 10 deg flags the
  stamp (shell modal still wins).  36 of 55 stamps flagged (mostly doors on
  perpendicular walls).

## Output structure (v2)

`schema_version: 2`; `library_id` and all `stamp_id`s UNCHANGED (geometry
frame changed, identity did not — consumers keyed by library/stamp id keep
working); `normalized_from` + `inputs` (v1 library + manifests, sha256);
`stats.normalization`: theta map, flagged stamps, member-bounds-source
counts, door facts, and `bounds_area_ratio_v2_vs_v1` (per stamp + min/max;
2026-08-12: min 0.614, max 1.764 — most rotated stamps shrank noticeably
(0.65–0.98); 9 of 55 stamps grew >1.1 because the v1 manifest world bounds
under-estimated those models, the A2-local bounds being the validated
authority).

## Pipeline position

T0.4b, immediately after the T0.3 stamp libraries and before
`build_stamp_volumes.py` (v2 volumes), `visual_planner.py` (CANONICAL
libraries), `visual_planner_advisory.py` (v2 volumes sidecar), and the
planning-bundle builder.  v1 files are never modified.  Propagation
completed 2026-08-12: volumes v2, bundle rebuild, fixture re-verification
(`out_norm_*`), benchmark re-render (`out_norm_benchmark`), 196-test suite —
all green; see the run report
`.opencode/runs/stamp-normalization-2026-08-12/2026-08-12_stamp_normalization_report.md`.
