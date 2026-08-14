#!/usr/bin/env python3
"""Cityforge T0.4: per-member stamp-local bounding boxes + below-ground flags.

Purpose
-------
For EVERY member of EVERY stamp in the two cityforge stamp libraries
(``markarth_side_stone_v2.json``, ``karthgad_nord_v2.json`` by default; the
library paths are CLI-parameterized so v1 libraries can still be processed),
derive:

* a stamp-local 3D AABB (``box_local``) -- the member's ORIENTED bounding
  box (OBB) axis-aligned in the stamp frame.  The OBB is composed from the
  model's LOCAL (unrotated) bounds x scale, rotated by the member rotz and
  translated by the member offset -- in the v2 (normalized) frame the
  member rotz' is already building-relative, so for members near 0 mod 90
  the box is the tight model box; non-cardinal members keep the AABB of
  their OBB (documented per member via ``obb_rotz_prime_deg`` and
  ``box_tight``).  Model local bounds come from the run's A2 evidence
  documents via ``stamp_local_bounds`` (validated identical to the
  surface-geometry cache ``local_bounds``); the same fallback ladder
  (un-inflated world AABB, world AABB) applies, recorded per member.
  ``box_local = OBB_corners - anchor`` (the anchor is the v2 origin).
* a ``below_ground`` classification: the member's box top vs the MINIMUM
  source LAND terrain height sampled over the member's WORLD XY footprint
  (conservative rule: true ONLY when the whole box sits strictly below every
  terrain seating at the source position).

Emits one deterministic sidecar JSON (``stamp_volumes_v2.json`` by default)
consumed by the Z-aware building-overlap check (underground-only overlap
should not count as collision) and per-role footprint decomposition.

Inputs (all read-only)
---------------------
* ``output/cityforge/stamps/karthgad_nord_v2.json`` (11 stamps) and
  ``output/cityforge/stamps/markarth_side_stone_v2.json`` (44 stamps) --
  the building-aligned v2 libraries (``normalization_theta_deg`` per stamp
  is consumed only to document the frame; the composition itself reads the
  already-normalized member offsets/rotations).
* Per-ref evaluated world bounds from the SAME extraction products the stamp
  library joined (per-ref ``world_bounds_gu``; used for the terrain footprint
  and as the fallback authority -- see Invariants):
  - Markarth: ``output/skyrim-settlements/markarth-side-v1/components/
    buildings/<slug>/manifest.json`` ``members[].world_bounds_gu``, located via
    ``components/buildings_index.json`` ``component_id`` -> slug
    (``stamp.source.component_id``).
  - Karthgad: ``output/skyrim-settlements/karthgad-v1/components/buildings/
    <stamp.source.slug>/manifest.json`` ``members[].world_bounds_gu``.
  Every manifest read is SHA-256-verified against the hash the owning library
  recorded in its own ``inputs`` map (fail closed on mismatch).
* Per-model local bounds from each run's A2 evidence directory
  (``<run>/a2/``; see ``stamp_local_bounds.load_a2_local_bounds``).
* Source LAND terrain -- the same authority the stamp library's re-measured
  terrain envelopes come from (``procgen.espland``): ``Sky_Main.esm`` for
  Karthgad, ``PTR Indev/Sky_Markarth.esm`` for Markarth.

Outputs
-------
``output/cityforge/stamps/stamp_volumes_v2.json`` (fresh-file write; refuses
to overwrite an existing file).  Per stamp_id:

* ``members``: ``{source_id, model_key, structural_role, is_door,
  box_local: {min:[x,y,z], max:[x,y,z]}, below_ground,
  measured: {top_z, terrain_min_z, terrain_max_z}}`` -- measured z values are
  recorded in source world GU (translation-invariant: the same facts in
  stamp-local z differ only by the constant anchor z).  v2 additionally
  records ``obb_source`` (model_local_bounds / uninflated_world_aabb /
  world_aabb), ``obb_rotz_prime_deg`` and ``box_tight`` (rotz' within
  0.6 deg of 0 mod 90 -> the AABB-of-OBB equals the model box).
* ``above_ground_xy_boxes``: per-role merged XY AABB (stamp-local) over
  above-ground members only.  Role key = ``structural_role`` when assigned,
  else ``"door"`` for door members, else ``"unassigned"``.
* ``sanity``: ``union_vs_bounds_rel_max_dev_gu`` -- max per-axis deviation
  between the union of member boxes and the library's whole-stamp
  ``bounds_rel_gu`` (report-only; both are derived from the same member
  OBB corners so agreement within float tolerance is expected).

Invariants
----------
* v2 stamp space is building-aligned: the library's members already carry
  normalized offsets/rotations, so ``box_local`` is the AABB of the member
  OBB in that frame (tight for axis-aligned members).  Legacy v1 libraries
  (no ``normalization_theta_deg``) fall back to world-aligned stamp space:
  ``box_local = world_bounds_gu - anchor`` (the pre-normalization behavior).
* Preferred source for the OBB: per-model local bounds from the A2 evidence
  docs.  The fallback ladder (un-inflated world AABB, then the world AABB
  itself) is recorded per member and never silent; coverage is asserted at
  runtime and any member without cached world bounds fails the run loudly
  (``FAILURE: stamp_volumes <stamp_id> <reason>`` listing the missing
  ``source_id``s) -- no Blender launches, no fabricated boxes.
* Below-ground rule is conservative: ``below_ground = top_z < terrain_min_z``
  where ``terrain_min_z`` is the minimum LAND height over the member's world
  XY footprint (65x65 field, 256 GU margin -- the exact
  ``citystamps._sample_field`` sampling parameters and rounding).  Missing
  footprint samples fail closed (never converted to sea level).
* Determinism: canonical JSON (sorted keys, 2-space indent, no NaN) plus a
  trailing newline; the only non-constant field is the UTC timestamp.
* Coverage gate: EVERY member of EVERY stamp must yield a finite box; the run
  stops without writing on the first gap (all gaps are reported).

Pipeline position
------------------
Cityforge T0.4/T0.4b.  Consumes the T0.3/T0.4b stamp libraries and the T0.2
extraction products; feeds the later Z-aware building-overlap check and
per-role footprint decomposition.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
TOOLS = WORKSPACE / "tools"
for entry in (SRC, TOOLS, TOOLS / "cityforge"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from procgen import espland  # noqa: E402
from procgen.citystamps import (  # noqa: E402
    HEIGHT_FIELD_MARGIN_GU,
    HEIGHT_FIELD_SIDE,
    canonical_json_bytes,
    sha256_file,
)
from stamp_local_bounds import (  # noqa: E402
    load_a2_local_bounds,
    member_obb_corners_rel,
)

SCHEMA_VERSION = 2
__version__ = "0.2.0"

OUT_DEFAULT = WORKSPACE / "output" / "cityforge" / "stamps" / "stamp_volumes_v2.json"

MARKARTH_LIB = WORKSPACE / "output" / "cityforge" / "stamps" / "markarth_side_stone_v2.json"
KARTHGAD_LIB = WORKSPACE / "output" / "cityforge" / "stamps" / "karthgad_nord_v2.json"
MARKARTH_RUN = WORKSPACE / "output" / "skyrim-settlements" / "markarth-side-v1"
KARTHGAD_RUN = WORKSPACE / "output" / "skyrim-settlements" / "karthgad-v1"
SKY_MAIN = WORKSPACE / "Sky_Main.esm"
SKY_MARKARTH = WORKSPACE / "PTR Indev" / "Sky_Markarth.esm"

# A member's AABB-of-OBB is the tight model box when its normalized rotz is
# within this tolerance of a cardinal direction.
BOX_TIGHT_TOLERANCE_DEG = 0.6


class VolumeCoverageError(RuntimeError):
    """One or more stamp members have no recoverable cached world bounds."""


def _read_json(path: Path, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {label}: {path}: {exc}") from exc


def _ws_rel(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def _sub(left: Sequence[float], right: Sequence[float]) -> list[float]:
    """Exact float64 subtraction (the stamp library's own ``_vec_sub``)."""
    return [float(a) - float(b) for a, b in zip(left, right)]


def _finite_box(bounds: Mapping[str, Sequence[float]]) -> bool:
    """A world-bounds dict is usable iff both corners are finite triplets."""
    try:
        mn = [float(v) for v in bounds["min"]]
        mx = [float(v) for v in bounds["max"]]
    except (KeyError, TypeError, ValueError):
        return False
    if len(mn) != 3 or len(mx) != 3:
        return False
    return all(v == v and abs(v) != float("inf") for v in mn + mx)


def _manifest_authority(
    run_dir: Path,
    library: Mapping[str, Any],
    manifest_slug: str,
    stamp_id: str,
) -> tuple[dict[str, Mapping[str, Any]], list[tuple[str, str]]]:
    """Load one component manifest's members and verify its recorded hash.

    Returns ``(member_by_source_id, [(ws_rel_path, sha256)])``.  The manifest
    path must appear in the library's own ``inputs`` map with the same
    SHA-256 (the library recorded every manifest it consumed); a mismatch or
    a missing record fails closed -- the volumes must not drift from the
    bounds authority the stamp library itself used.
    """
    manifest_path = run_dir / "components" / "buildings" / manifest_slug / "manifest.json"
    manifest = _read_json(manifest_path, f"component manifest {manifest_slug}")
    rel = _ws_rel(manifest_path)
    digest = sha256_file(manifest_path)
    recorded = (library.get("inputs") or {}).get(rel)
    if recorded != digest:
        raise RuntimeError(
            f"FAILURE: stamp_volumes {stamp_id} manifest hash mismatch for {rel}: "
            f"recorded {recorded!r} != read {digest}"
        )
    members = {
        member["source_id"]: member
        for member in manifest.get("members", [])
        if member.get("source_id")
    }
    return members, [(rel, digest)]


def _karthgad_authority(library: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[tuple[str, str]]]:
    """Karthgad bounds authority: one manifest per stamp, keyed by slug."""
    authority: dict[str, Mapping[str, Any]] = {}
    inputs: list[tuple[str, str]] = []
    for stamp in library["stamps"]:
        members, manifest_inputs = _manifest_authority(
            KARTHGAD_RUN, library, stamp["source"]["slug"], stamp["stamp_id"]
        )
        authority.update(members)
        inputs.extend(manifest_inputs)
    return authority, inputs


def _markarth_authority(library: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[tuple[str, str]]]:
    """Markarth bounds authority: component_id -> building slug -> manifest.

    The stamp carries only ``source.component_id``; the buildings index maps
    it to the building slug whose manifest holds the joined member records
    (the same join the stamp library performed).
    """
    index = _read_json(
        MARKARTH_RUN / "components" / "buildings_index.json", "markarth buildings index"
    )
    slug_by_component = {building["component_id"]: building for building in index["buildings"]}
    authority: dict[str, Mapping[str, Any]] = {}
    inputs: list[tuple[str, str]] = []
    for stamp in library["stamps"]:
        building = slug_by_component.get(stamp["source"].get("component_id"))
        if building is None:
            raise RuntimeError(
                f"FAILURE: stamp_volumes {stamp['stamp_id']} no buildings-index "
                f"entry for component_id {stamp['source'].get('component_id')!r}"
            )
        members, manifest_inputs = _manifest_authority(
            MARKARTH_RUN, library, building["slug"], stamp["stamp_id"]
        )
        authority.update(members)
        inputs.extend(manifest_inputs)
    return authority, inputs


def _role_key(member: Mapping[str, Any]) -> str:
    """Grouping role for footprint decomposition.

    The kit-role registry assigns ``shell``/``access``; unassigned members
    fall back to ``door`` (is_door) or ``unassigned``.  Doors with an
    explicit registry role keep that role (the registry is authoritative).
    """
    role = member.get("structural_role")
    if role:
        return str(role)
    return "door" if member.get("is_door") else "unassigned"


def _footprint_terrain(land: Mapping[tuple[int, int], espland.LandRecord], min_xy: Sequence[float], max_xy: Sequence[float]) -> tuple[float, float, int]:
    """Min/max LAND height (world GU) over a member's world XY footprint.

    Mirrors ``citystamps._sample_field`` exactly (same 65x65 field, 256 GU
    margin, footprint-only predicate, 6-decimal rounding) so the heights here
    are the same authority as the stamp library's re-measured envelopes.
    Returns ``(terrain_min_z, terrain_max_z, missing_footprint_samples)``.
    """
    field = espland.sample_height_field(
        land,
        list(min_xy),
        list(max_xy),
        margin_game_units=HEIGHT_FIELD_MARGIN_GU,
        side=HEIGHT_FIELD_SIDE,
    )
    values_gu = field["values_game_units"]  # type: ignore[index]
    spacing = field["spacing_game_units"]  # type: ignore[index]
    field_min = field["field_bbox_xy_game_units"]["min"]  # type: ignore[index]
    heights: list[float] = []
    missing = 0
    for row in range(HEIGHT_FIELD_SIDE):
        game_y = float(field_min[1]) + row * float(spacing[1])
        for column in range(HEIGHT_FIELD_SIDE):
            game_x = float(field_min[0]) + column * float(spacing[0])
            value = values_gu[row][column]
            if value is None:
                if min_xy[0] <= game_x <= max_xy[0] and min_xy[1] <= game_y <= max_xy[1]:
                    missing += 1
                continue
            if min_xy[0] <= game_x <= max_xy[0] and min_xy[1] <= game_y <= max_xy[1]:
                heights.append(float(value))
    if missing or not heights:
        return float("nan"), float("nan"), missing
    return min(heights), max(heights), 0


def _process_stamp(
    stamp: Mapping[str, Any],
    land: Mapping[tuple[int, int], espland.LandRecord],
    authority: Mapping[str, Mapping[str, Any]],
    local_bounds: Mapping[tuple[str, str], Mapping[str, Sequence[float]]],
) -> dict[str, Any]:
    """Derive volumes + classification for one stamp; raise on any gap."""
    anchor = stamp["anchor"]["source_position_gu"]
    if len(anchor) != 3:
        raise RuntimeError(
            f"FAILURE: stamp_volumes {stamp['stamp_id']} malformed anchor {anchor!r}"
        )
    is_v2 = "normalization_theta_deg" in stamp

    member_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for member in stamp["members"]:
        source_id = member["source_id"]
        cached = authority.get(source_id)
        world_bounds = (cached or {}).get("world_bounds_gu")
        if not _finite_box(world_bounds):  # type: ignore[arg-type]
            missing.append(source_id)
            continue
        if is_v2:
            # Building-aligned frame: the member rows already carry the
            # normalized offset/rotation, so the OBB composition runs with
            # theta=0 (member_obb_corners_rel would otherwise re-rotate).
            source_sha = (cached or {}).get("source", {}).get("sha256")
            model_key = member.get("model_key")
            local = local_bounds.get((str(model_key), str(source_sha))) if (
                model_key and source_sha
            ) else None
            corners, obb_source, obb_note = member_obb_corners_rel(
                member, 0.0, local, world_bounds, anchor
            )
            box_local = {
                "min": [min(corner[axis] for corner in corners) for axis in range(3)],
                "max": [max(corner[axis] for corner in corners) for axis in range(3)],
            }
            rotz_prime = float(member["rotation"][2]) * 180.0 / math.pi
            rotz_prime_deg = rotz_prime % 360.0
            rem = rotz_prime_deg % 90.0
            residual = min(rem, 90.0 - rem)
            obb_row = {
                "obb_source": obb_source,
                "obb_rotz_prime_deg": round(rotz_prime_deg, 3),
                "box_tight": bool(residual <= BOX_TIGHT_TOLERANCE_DEG),
                **{f"obb_{key}": value for key, value in obb_note.items()},
            }
        else:
            # Legacy v1 libraries: world-aligned stamp space, exact anchor
            # subtraction (the pre-normalization behavior).
            wb_min = [float(v) for v in world_bounds["min"]]  # type: ignore[index]
            wb_max = [float(v) for v in world_bounds["max"]]  # type: ignore[index]
            box_local = {
                "min": _sub(wb_min, anchor),
                "max": _sub(wb_max, anchor),
            }
            obb_row = {}
        terrain_min_z, terrain_max_z, missing_samples = _footprint_terrain(
            land, world_bounds["min"][:2], world_bounds["max"][:2]  # type: ignore[index]
        )
        if missing_samples:
            raise RuntimeError(
                f"FAILURE: stamp_volumes {stamp['stamp_id']} member {source_id}: "
                f"{missing_samples} missing LAND samples under its footprint; "
                "refusing to classify without terrain"
            )
        top_z = float(world_bounds["max"][2])  # type: ignore[index]
        member_rows.append(
            {
                "source_id": source_id,
                "model_key": member.get("model_key"),
                "structural_role": member.get("structural_role"),
                "is_door": bool(member.get("is_door")),
                "box_local": box_local,
                "below_ground": bool(top_z < terrain_min_z),
                "measured": {
                    "top_z": top_z,
                    "terrain_min_z": terrain_min_z,
                    "terrain_max_z": terrain_max_z,
                },
                **obb_row,
            }
        )
    if missing:
        raise VolumeCoverageError(
            f"FAILURE: stamp_volumes {stamp['stamp_id']} no cached world bounds "
            f"for {len(missing)} member(s): {', '.join(sorted(missing))}"
        )

    # Per-role merged XY AABBs over ABOVE-GROUND members only (stamp-local).
    role_boxes: dict[str, list[list[float]]] = {}
    for row in member_rows:
        if row["below_ground"]:
            continue
        role = _role_key(
            {"structural_role": row["structural_role"], "is_door": row["is_door"]}
        )
        box = row["box_local"]
        # Accumulate flat [x, y] corners (both box corners per member) so the
        # merged AABB below is the axis-wise min/max over every member corner.
        role_boxes.setdefault(role, []).extend(
            [
                [box["min"][0], box["min"][1]],
                [box["max"][0], box["max"][1]],
            ]
        )
    above_ground_boxes: list[dict[str, Any]] = []
    for role in sorted(role_boxes):
        corners = role_boxes[role]
        above_ground_boxes.append(
            {
                "role": role,
                "member_count": len(corners) // 2,
                "min_xy": [min(p[0] for p in corners), min(p[1] for p in corners)],
                "max_xy": [max(p[0] for p in corners), max(p[1] for p in corners)],
            }
        )

    # Sanity (report-only): union of member boxes vs the library's
    # whole-stamp bounds_rel_gu; both derive from the same member world
    # bounds, so agreement within float tolerance is expected.
    union_min = [
        min(float(row["box_local"]["min"][axis]) for row in member_rows)
        for axis in range(3)
    ]
    union_max = [
        max(float(row["box_local"]["max"][axis]) for row in member_rows)
        for axis in range(3)
    ]
    rel = stamp["bounds_rel_gu"]
    max_dev = max(
        [abs(union_min[axis] - float(rel["min"][axis])) for axis in range(3)]
        + [abs(union_max[axis] - float(rel["max"][axis])) for axis in range(3)]
    )

    return {
        "stamp_id": stamp["stamp_id"],
        "members": member_rows,
        "above_ground_xy_boxes": above_ground_boxes,
        "sanity": {"union_vs_bounds_rel_max_dev_gu": max_dev},
    }


def _process_library(
    lib_path: Path,
    land: Mapping[tuple[int, int], espland.LandRecord],
    authority_builder: Any,
    local_bounds: Mapping[tuple[str, str], Mapping[str, Sequence[float]]],
) -> dict[str, Any]:
    """Process one library; returns the per-stamp result and run stats."""
    library = _read_json(lib_path, f"stamp library {lib_path.name}")
    authority, manifest_inputs = authority_builder(library)
    stamp_results: list[dict[str, Any]] = []
    member_total = 0
    below_ground_total = 0
    max_dev = 0.0
    for stamp in library["stamps"]:
        result = _process_stamp(stamp, land, authority, local_bounds)
        stamp_results.append(result)
        member_total += len(result["members"])
        below_ground_total += sum(1 for row in result["members"] if row["below_ground"])
        max_dev = max(max_dev, result["sanity"]["union_vs_bounds_rel_max_dev_gu"])
    inputs: dict[str, str] = {str(path): digest for path, digest in manifest_inputs}
    inputs[_ws_rel(lib_path)] = sha256_file(lib_path)
    return {
        "library_id": library["library_id"],
        "stamp_results": stamp_results,
        "inputs": dict(sorted(inputs.items())),
        "stats": {
            "stamp_count": len(stamp_results),
            "member_count": member_total,
            "below_ground_member_count": below_ground_total,
            "max_union_vs_bounds_rel_dev_gu": max_dev,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive per-member stamp-local bounding boxes and "
        "below-source-ground flags for the cityforge stamp libraries."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DEFAULT,
        help="output sidecar path (default: output/cityforge/stamps/stamp_volumes_v2.json)",
    )
    parser.add_argument(
        "--karthgad-lib",
        type=Path,
        default=KARTHGAD_LIB,
        help="Karthgad stamp library (default: karthgad_nord_v2.json; pass "
        "the v1 path to process the world-aligned legacy library)",
    )
    parser.add_argument(
        "--markarth-lib",
        type=Path,
        default=MARKARTH_LIB,
        help="Markarth stamp library (default: markarth_side_stone_v2.json; "
        "pass the v1 path to process the world-aligned legacy library)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = Path(args.out)
    if out_path.exists():
        print(f"FAILURE: stamp_volumes refuse to overwrite existing output {out_path}", file=sys.stderr)
        return 1

    try:
        karthgad_local = load_a2_local_bounds(KARTHGAD_RUN / "a2")
        markarth_local = load_a2_local_bounds(MARKARTH_RUN / "a2")
        karthgad = _process_library(
            Path(args.karthgad_lib),
            espland.load_land(SKY_MAIN),
            _karthgad_authority,
            karthgad_local,
        )
        markarth = _process_library(
            Path(args.markarth_lib),
            espland.load_land(SKY_MARKARTH),
            _markarth_authority,
            markarth_local,
        )
    except (VolumeCoverageError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    document = {
        "schema_version": SCHEMA_VERSION,
        "tool": f"build_stamp_volumes {__version__} (Cityforge T0.4/T0.4b)",
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "units": "game units (GU); box_local and above_ground_xy_boxes are "
        "stamp-local (seed-door anchor at origin, BUILDING-ALIGNED for v2 "
        "libraries: member boxes are the AABB of each member's OBB, tight "
        "for members near 0 mod 90 -- see per-member obb_* fields); "
        "measured z values are source world GU",
        "inputs": {
            karthgad["library_id"]: karthgad["inputs"],
            markarth["library_id"]: markarth["inputs"],
        },
        "libraries": {
            karthgad["library_id"]: {
                "stats": karthgad["stats"],
                "stamps": karthgad["stamp_results"],
            },
            markarth["library_id"]: {
                "stats": markarth["stats"],
                "stamps": markarth["stamp_results"],
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(canonical_json_bytes(document))

    print(f"[stamp_volumes] wrote {out_path} (sha256 {sha256_file(out_path)})")
    for kit in (karthgad, markarth):
        stats = kit["stats"]
        print(
            f"[stamp_volumes] {kit['library_id']}: stamps={stats['stamp_count']} "
            f"members={stats['member_count']} below_ground={stats['below_ground_member_count']} "
            f"max_union_vs_bounds_rel_dev_gu={stats['max_union_vs_bounds_rel_dev_gu']:.6g}"
        )
    if markarth["stats"]["below_ground_member_count"] == 0:
        print(
            "[stamp_volumes] WARNING: zero below-ground members in the Markarth "
            "library; re-examine terrain sampling before trusting the "
            "classification",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
