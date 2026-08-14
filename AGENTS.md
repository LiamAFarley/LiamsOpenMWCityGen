# AGENTS.md — Procedural Tamriel Core

This is the *core implementation reference* checkout for the Procedural Tamriel
project.  It is an architecture/study copy, not a runnable pipeline.  Read
`docs/architecture_overview.md` first, then the guides in `docs/guides/`.

## Source of truth

The authoritative workspace is separate and contains the full pipeline,
settlement extraction tooling, `tamriel.esm`, mod data roots, and the large
test suite.  This checkout is refreshed by
`tools/repository/export_public_core.py` against
`tools/repository/public_core_manifest.json` in the workspace.  Never treat
this checkout as the live source of truth for changes.

## Layout

- `src/procgen/` — the engine package.  Pure Python (NumPy/Pillow/SciPy/
  scikit-image), deterministic, no Blender imports at package level.
- `tools/cityforge/` — driver scripts.  Some require Blender (`bpy`) or
  workspace data; see the README tier list.
- `docs/architecture_overview.md` — the stage-by-stage pipeline explanation.
- `docs/guides/` — cityforge tool and geometry-convention guides.
- `examples/` — synthetic contract examples (no real game assets).

## Geometry conventions (binding)

Read `docs/guides/rotation_conventions.md`.  The short version:

- Engine placement: `world = pos + Rx(-rx) @ Ry(-ry) @ Rz(-rz) . (scale*local)`,
  all column vectors.  For yaw-only: `Rz(-rotz)`.  Single implementation in
  `src/procgen/engine_transform.py` — never re-derive rotation math inline.
- Stamp v2 libraries are already in building-aligned frames (`F = Rz(+theta)`);
  compose member OBBs as `offset' + Rz(-rotz') . (scale*local)` via
  `tools/cityforge/stamp_local_bounds.py`.
- Door facing is the geometric `outward_heading_deg`, **not** raw door rotz.
- Plan yaw is map-CCW (`Rz(+yaw)` depiction); at ESP authoring write member
  `rotz = rotz' - yaw_map`.

## Rules

- Never modify original game/mod data.  All inputs are read-only.
- Keep the deterministic computation and the ESP authoring separate:
  measurement facts and decisions are never silently mixed.
- Generated plugins are **masterless**: `Header.masters: []`.  Never declare
  `tamriel.esm` as a master.
- When claiming something is proven, cite the invariant or test that proves
  it; a numeric self-check that is sign-blind (AABB area/bounds agreement)
  does not prove orientation.
