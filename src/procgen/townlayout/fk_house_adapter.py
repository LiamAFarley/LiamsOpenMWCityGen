"""Adapter for procedurally generated Falkreath houses in stage 07.

Purpose
-------
Provide a bounded shell table (source-observed Falkreath shell variants plus reusable access variants) and deterministic
station-candidate iteration for wall front rows when a brief sets
development_policy.inner.house_generator == "fk_house". The dense frontage
search uses the base shell rows; R11 can reuse the porch and stair variants
when their fitted hulls have room.

Inputs
------
- SHELL_SPECS via fk_house.generate_fk_house
- master_seed / candidate_id / block / road / side / station for seeding

Outputs
-------
- ShellRow table (source-observed base rows plus access variants) with hull/door/envelope/
  OBB metrics and width class
- station_seed (derive_seed)
- iter_station_candidates (fit filter + deterministic usage-balanced order, ≤8)

Invariants
----------
- No RNG objects; all determinism via procgen.seeds.derive_seed
- Base stamp IDs are `fkgen__<shell_id>__base`; access variants append
  `__porch` or `__stairs`.
- Deterministic base, porch, and stair variants; dormers remain disabled
  until their measured attachment profiles are accepted.

Pipeline position
-----------------
Stage 07 seam (wall_population inner zone); consumed only there and via
spatial_roles _door_rows merge of generated_stamps.
"""

from __future__ import annotations

from typing import NamedTuple

from procgen import fk_house
from procgen.frontage_fit import DoorGeometry, _stamp_doors, _stamp_hull
from procgen.seeds import derive_seed
from procgen.townlayout.stamp_index import _obb_width_depth

GENERATED_LIBRARY_ID = "generated_fk_house_v1"
SIZE_CYCLE = ("small", "medium", "small", "medium", "large")


class ShellRow(NamedTuple):
    shell_id: str
    accessory: str
    stamp: dict
    hull: list
    door: DoorGeometry
    envelope: dict
    obb_width_gu: float
    obb_depth_gu: float
    width_class: str


def build_shell_table() -> list[ShellRow]:
    rows: list[ShellRow] = []
    # Generate reusable access variants once. Dormers remain disabled until
    # their measured attachment profiles are accepted for production use.
    # Preserve the variants actually observed in the authoritative xFa cells;
    # reference_shell_ids falls back to one `_a` shell per family if the
    # extraction bundle is unavailable.
    shell_ids = fk_house.reference_shell_ids()
    for shell_id in sorted(shell_ids):
        primary = tuple(fk_house.SHELL_SPECS[shell_id]["default_door_facades"])
        variants = (
            ("base", {}),
            ("porch", {"porch_facades": primary,
                       "porch_model": "sky_FK_Porch_01a"}),
            ("stairs", {"stair_facades": primary,
                         "stair_model": "sky_ex_mk_str_02"}),
        )
        for accessory, options in variants:
            stamp_id = f"fkgen__{shell_id}__{accessory}"
            stamp = fk_house.generate_fk_house(
                shell_id, generated_id=stamp_id, **options)
            hull = _stamp_hull(stamp)
            door = _stamp_doors(stamp)[0]
            envelope = stamp.get("terrain_envelope") or {}
            w, d = _obb_width_depth(hull, door.heading_deg)
            rows.append(
                ShellRow(
                    shell_id=shell_id,
                    accessory=accessory,
                    stamp=stamp,
                    hull=hull,
                    door=door,
                    envelope=dict(envelope),
                    obb_width_gu=float(w),
                    obb_depth_gu=float(d),
                    width_class="",  # filled below
                )
            )
    # width_class by sorted-width thirds
    class_rows = [row for row in rows if row.accessory == "base"]
    n = len(class_rows)
    cut1 = max(1, n // 3)
    cut2 = max(cut1 + 1, 2 * n // 3)
    sorted_by_width = sorted(class_rows,
                             key=lambda r: (r.obb_width_gu, r.shell_id))
    # map shell_id -> class
    class_by_id: dict[str, str] = {}
    for idx, r in enumerate(sorted_by_width):
        if idx < cut1:
            cls = "small"
        elif idx < cut2:
            cls = "medium"
        else:
            cls = "large"
        class_by_id[r.shell_id] = cls
    # rebuild with class filled
    out: list[ShellRow] = []
    for r in rows:
        out.append(
            ShellRow(
                shell_id=r.shell_id,
                accessory=r.accessory,
                stamp=r.stamp,
                hull=r.hull,
                door=r.door,
                envelope=r.envelope,
                obb_width_gu=r.obb_width_gu,
                obb_depth_gu=r.obb_depth_gu,
                width_class=class_by_id[r.shell_id],
            )
        )
    return out


def station_seed(
    master_seed: int,
    candidate_id: str,
    block_id: str,
    road_id: str,
    side: str,
    station_gu: float,
) -> int:
    return derive_seed(
        master_seed,
        "fk_house",
        candidate_id,
        block_id,
        road_id,
        side,
        int(round(station_gu)),
    )


def iter_station_candidates(
    table,
    *,
    along_gu,
    depth_gu,
    wanted,
    usage,
    seed,
):
    """Up to 8 ShellRows: fit filter then deterministic usage-balanced order."""
    fitting = [
        r
        for r in table
        if r.accessory == "base"
        and r.obb_width_gu <= float(along_gu)
        and r.obb_depth_gu <= float(depth_gu)
    ]
    ordered = sorted(
        fitting,
        key=lambda r: (
            usage[r.shell_id],
            0 if r.width_class == wanted else 1,
            derive_seed(seed, "choice", r.shell_id),
            r.accessory,
            r.shell_id,
        ),
    )
    return ordered[:8]
