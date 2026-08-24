"""cliff_seating.py — observed-pose cliff seating runtime (pure core).

Pipeline position: consumed by ``scatter_generate`` for ``category == cliff``
when a seating config + profile sidecar are supplied.  This module owns the
profile preflight, per-mesh member pools, observed-pose selection, matrix
composition, sample transforms, and the feasible-Z sweep.  The relief and
road-footprint audits stay in ``scatter_generate`` and run AFTER seating so a
seating-rejected pose never pays for them.

Binding semantics (plan + user rulings 2026-08-24):

* Every rotation is one recorded terrain-relative observation composed with
  the candidate terrain frame: ``M_candidate = T_candidate @ R_relative``.
  No Euler-quantile sampling, yaw search, or SO(3) perturbation exists.
* Source material governs seating depth: the per-mode feasible embed band is
  the members' recorded source embed range (clamped by config); the solved Z
  clamps the member's own recorded z-offset arrangement into it.
* Upslope alignment of a lateral burial heading is RECORD-ONLY (it reduces to
  the mode's canonical downhill-axis component, so it is a mode property, and
  source mouths are hidden by burial depth, not direction).  The lateral
  terrain-cover check and the visible-front check remain hard gates.
* A candidate whose terrain frame cannot be constructed (no real lower
  neighbor) rejects the attempt — patterns that cannot exist on the target
  terrain are absent from generations by design, never pipeline failures.
* At most one observed pose and ``maximum_total_samples_per_attempt`` LAND
  samples are evaluated per candidate; no mesh I/O and no retry loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .engine_transform import matrix_to_tes3_euler, tes3_euler_to_matrix
from .espland import THU_TO_GU, height_at_game_position
from .scatter_analysis import normalize_mesh_key, transformed_bbox
from .terrain_frame import (
    TerrainFrameError,
    build_terrain_frame,
    euler_round_trip,
    slope_bin_index,
    transfer_pose_matrix,
)


class CliffSeatingError(ValueError):
    """Preflight or configuration failure (essential-stage failure)."""


_REJECTION_PREFIX = "cliff_seating_"


@dataclass(frozen=True)
class _Member:
    """One recorded observation available to the runtime selector."""

    ref_id: str
    mode_id: str
    matrix: np.ndarray
    scale: float
    z_offset_gu: float
    slope_bin: int
    support_samples: tuple[tuple[float, float, float], ...]
    cover_samples: tuple[tuple[float, float, float], ...]
    front_samples: tuple[tuple[float, float, float], ...]
    burial_local: tuple[float, float, float] | None
    classification: str
    embed_band_gu: tuple[float, float]
    visible_height_gu: float | None


@dataclass
class _MeshState:
    mesh: str
    members: tuple[_Member, ...]
    members_by_bin: dict[int, tuple[_Member, ...]] = field(default_factory=dict)

    def pool_for(self, slope_bin: int) -> tuple[_Member, ...]:
        return self.members_by_bin.get(slope_bin) or self.members


class CliffSeatingRuntime:
    """Validated sidecar state plus the per-candidate seating evaluator."""

    def __init__(
        self,
        config: Mapping[str, Any],
        profiles: Mapping[str, Any],
        cliff_analysis: Mapping[str, Any],
        *,
        quarantined_keys: frozenset[str] | set[str],
    ) -> None:
        if str(config.get("schema_version")) != "1":
            raise CliffSeatingError("cliff seating config schema_version must be 1")
        profile_id = str(config.get("profile_id", ""))
        if not profile_id:
            raise CliffSeatingError("cliff seating config has no profile_id")
        if str(profiles.get("schema_version")) != "1":
            raise CliffSeatingError("cliff seating profiles schema_version must be 1")
        if str(profiles.get("profile_id", "")) != profile_id:
            raise CliffSeatingError(
                "cliff seating profile_id does not match the config "
                f"({profiles.get('profile_id')!r} != {profile_id!r})"
            )
        runtime_cfg = config.get("runtime")
        frame_cfg = config.get("terrain_frame")
        modes_cfg = config.get("pose_modes")
        for label, block in (("runtime", runtime_cfg), ("terrain_frame", frame_cfg), ("pose_modes", modes_cfg)):
            if not isinstance(block, Mapping):
                raise CliffSeatingError(f"cliff seating config has no {label} block")

        # Preflight 2: recorded analysis path AND per-mesh ref sets must match
        # the supplied analysis.  No path-only acceptance, no hashes.
        recorded_path = str((profiles.get("inputs") or {}).get("cliff_analysis", ""))
        if not recorded_path:
            raise CliffSeatingError("profile sidecar records no cliff_analysis input")
        giants = cliff_analysis.get("giants")
        if not isinstance(giants, list) or not giants:
            raise CliffSeatingError("cliff analysis contains no giants rows")
        refs_by_mesh: dict[str, set[str]] = {}
        for row in giants:
            if not isinstance(row, Mapping) or not row.get("mesh") or not row.get("ref_id"):
                raise CliffSeatingError("cliff analysis giants row lacks mesh/ref_id")
            refs_by_mesh.setdefault(normalize_mesh_key(str(row["mesh"])), set()).add(
                str(row["ref_id"])
            )
        profile_meshes = profiles.get("meshes")
        if not isinstance(profile_meshes, Mapping):
            raise CliffSeatingError("profile sidecar has no meshes mapping")
        for key, section in profile_meshes.items():
            normalized = normalize_mesh_key(str(key))
            recorded_refs = set(section.get("source_ref_ids") or []) if isinstance(section, Mapping) else set()
            if normalized in refs_by_mesh and recorded_refs != refs_by_mesh[normalized]:
                raise CliffSeatingError(
                    f"profile source ref set does not match the supplied cliff analysis: {key}"
                )

        # Preflight 3: every non-quarantined analysis mesh needs a profile row.
        missing_rows = sorted(
            key for key in refs_by_mesh
            if key not in quarantined_keys and normalize_mesh_key(key) not in {
                normalize_mesh_key(str(k)) for k in profile_meshes
            }
        )
        if missing_rows:
            raise CliffSeatingError(
                f"profile sidecar is missing rows for analysis meshes: {missing_rows}"
            )

        self.profile_id = profile_id
        self.config = config
        self.runtime_cfg = dict(runtime_cfg)
        self.frame_cfg = dict(frame_cfg)
        self.bin_edges = [float(v) for v in modes_cfg["slope_bin_edges_deg"]]
        self.maximum_samples = int(self.runtime_cfg["maximum_total_samples_per_attempt"])
        self.support_min_pass = int(self.runtime_cfg["support_min_pass_count"])
        self.cover_min_gu = float(self.runtime_cfg["lateral_cover_min_gu"])
        self.cover_min_pass = int(self.runtime_cfg["lateral_cover_min_pass_count"])
        self.alignment_min_dot = float(self.runtime_cfg["lateral_upslope_min_alignment_dot"])
        self.alignment_gate = str(self.runtime_cfg.get("lateral_alignment_gate", "record_only"))
        self.front_min_height = float(self.runtime_cfg["visible_front_min_height_gu"])
        self.front_min_pass = int(self.runtime_cfg["visible_front_min_pass_count"])
        self.front_visible_fraction = float(
            self.runtime_cfg.get("visible_front_min_fraction", 0.5)
        )
        if not 0.0 < self.front_visible_fraction <= 2.0:
            raise CliffSeatingError("visible_front_min_fraction must be in (0, 2]")
        self.embed_clamp_lo = float(self.runtime_cfg["support_embed_min_gu"])
        self.embed_clamp_hi = float(self.runtime_cfg["support_embed_max_gu"])
        self.embed_tolerance = float(
            self.runtime_cfg.get("support_embed_tolerance_gu", 0.0)
        )
        if self.embed_tolerance < 0.0:
            raise CliffSeatingError("support_embed_tolerance_gu must be non-negative")
        self.matrix_tolerance = float(frame_cfg["matrix_residual_tolerance"])
        self.frame_spacing = float(frame_cfg["sample_spacing_gu"])

        # Preflight 4 bookkeeping: quarantined + zero-eligible meshes removed
        # from quota allocation, with their measured frequency.
        frequency_by_mesh: dict[str, int] = {}
        for row in giants:
            key = normalize_mesh_key(str(row["mesh"]))
            frequency_by_mesh[key] = frequency_by_mesh.get(key, 0) + 1
        self.quarantined_meshes: dict[str, int] = {}
        self.excluded_meshes: dict[str, int] = {}
        self.mesh_states: dict[str, _MeshState] = {}
        for key in sorted(refs_by_mesh):
            if key in quarantined_keys:
                self.quarantined_meshes[key] = frequency_by_mesh.get(key, 0)
                continue
            section = next(
                (value for name, value in profile_meshes.items()
                 if normalize_mesh_key(str(name)) == key),
                None,
            )
            if not isinstance(section, Mapping):
                raise CliffSeatingError(f"profile row vanished during preflight: {key}")
            members = self._build_members(key, section)
            if not members:
                self.excluded_meshes[key] = frequency_by_mesh.get(key, 0)
                continue
            by_bin: dict[int, list[_Member]] = {}
            for member in members:
                by_bin.setdefault(member.slope_bin, []).append(member)
            self.mesh_states[key] = _MeshState(
                mesh=str(section.get("mesh", key)),
                members=tuple(members),
                members_by_bin={
                    bin_index: tuple(rows) for bin_index, rows in sorted(by_bin.items())
                },
            )
        # Preflight 5: at least one eligible profile must remain.
        if not self.mesh_states:
            raise CliffSeatingError("no eligible cliff seating profiles remain")

    # ------------------------------------------------------------------
    # Sidecar parsing
    # ------------------------------------------------------------------

    def _build_members(self, key: str, section: Mapping[str, Any]) -> list[_Member]:
        members: list[_Member] = []
        for mode in section.get("modes") or []:
            if not isinstance(mode, Mapping) or not mode.get("eligible"):
                continue
            classification = str(mode.get("classification", ""))
            if classification not in {"bottom_support_only", "lateral_burial_opening"}:
                continue
            burial = mode.get("burial_heading_local")
            for member in mode.get("members") or []:
                matrix = np.asarray(member["R_relative"], dtype=np.float64)
                if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                    raise CliffSeatingError(
                        f"{key}:{mode.get('mode_id')}: member matrix is invalid"
                    )
                # Source material governs seating depth: the feasible band is a
                # tolerance around THIS member's recorded source embed, so a
                # singleton observation still yields a workable interval while
                # the arrangement stays at its observed depth.
                source_embed = float(member["source_embed_depth_gu"])
                band = (
                    max(self.embed_clamp_lo, source_embed - self.embed_tolerance),
                    min(self.embed_clamp_hi, source_embed + self.embed_tolerance),
                )
                if band[0] > band[1]:
                    band = (band[1], band[0])
                members.append(_Member(
                    ref_id=str(member["ref_id"]),
                    mode_id=str(mode["mode_id"]),
                    matrix=matrix,
                    scale=float(member["scale"]),
                    z_offset_gu=float(member["z_offset_gu"]),
                    slope_bin=int(member["slope_bin"]),
                    support_samples=tuple(
                        (float(v[0]), float(v[1]), float(v[2]))
                        for v in mode.get("support_samples_local") or []
                    ),
                    cover_samples=tuple(
                        (float(v[0]), float(v[1]), float(v[2]))
                        for v in mode.get("lateral_cover_samples_local") or []
                    ),
                    front_samples=tuple(
                        (float(v[0]), float(v[1]), float(v[2]))
                        for v in mode.get("visible_front_samples_local") or []
                    ),
                    burial_local=(
                        (float(burial[0]), float(burial[1]), float(burial[2]))
                        if burial else None
                    ),
                    classification=classification,
                    embed_band_gu=band,
                    visible_height_gu=(
                        float(member["source_visible_height_gu"])
                        if member.get("source_visible_height_gu") is not None else None
                    ),
                ))
        return members

    # ------------------------------------------------------------------
    # Per-candidate evaluation
    # ------------------------------------------------------------------

    def has_profile(self, profile_key: str) -> bool:
        return normalize_mesh_key(profile_key) in self.mesh_states

    def select_member(self, profile_key: str, slope_deg: float, rng) -> _Member:
        state = self.mesh_states[normalize_mesh_key(profile_key)]
        pool = state.pool_for(slope_bin_index(slope_deg, self.bin_edges))
        return pool[rng.randrange(len(pool))]

    def evaluate_attempt(
        self,
        *,
        profile_key: str,
        member: _Member,
        candidate_x: float,
        candidate_y: float,
        candidate_terrain_z_gu: float,
        candidate_slope_deg: float,
        candidate_downhill_xy: Sequence[float],
        land_records: Mapping[Any, Any],
        bbox: Mapping[str, Any],
        clearing_index: Any | None,
    ) -> tuple[str, dict[str, Any]]:
        """Run one bounded seating attempt.

        Returns ``("reject", {"reason": ..., ...})`` or
        ``("accept", solution)`` where the solution carries the final euler
        triple, solved Z, transformed world AABB, and compact gate evidence.
        """

        sample_budget = self.maximum_samples
        needed = len(member.support_samples) + (
            len(member.cover_samples) + len(member.front_samples)
            if member.classification == "lateral_burial_opening" else 0
        )
        if needed > sample_budget:
            return "reject", {
                "reason": "sample_budget_exceeded",
                "detail": {"needed": needed, "budget": sample_budget},
            }

        # 1. Candidate terrain frame (target terrain).
        try:
            frame = build_terrain_frame(
                land_records,
                (candidate_x, candidate_y),
                sample_spacing_gu=self.frame_spacing,
                matrix_residual_tolerance=self.matrix_tolerance,
            )
        except TerrainFrameError as exc:
            return "reject", {
                "reason": "no_terrain_frame",
                "detail": {"detail": str(exc)},
            }

        # 2. Observed-pose composition and euler round trip.
        candidate_matrix = transfer_pose_matrix(frame.matrix, member.matrix)
        euler, residual = euler_round_trip(candidate_matrix)
        if residual > self.matrix_tolerance:
            return "reject", {
                "reason": "rotation_roundtrip",
                "detail": {"residual": residual},
            }
        local_up_z = math.cos(euler[0]) * math.cos(euler[1])
        if local_up_z <= 0.0:
            return "reject", {
                "reason": "flipped_orientation",
                "detail": {"local_up_world_z": local_up_z},
            }

        scale = member.scale
        preferred_z = candidate_terrain_z_gu + member.z_offset_gu

        # 3. Provisional-Z world AABB for the XY clearing/city-domain gate.
        provisional_bbox = transformed_bbox(
            bbox, [candidate_x, candidate_y, preferred_z], euler, scale
        )
        if clearing_index is not None:
            world_min = provisional_bbox["min"]
            world_max = provisional_bbox["max"]
            if clearing_index.blocks_aabb(
                float(world_min[0]), float(world_min[1]),
                float(world_max[0]), float(world_max[1]),
            ):
                return "reject", {
                    "reason": "rock_clearing_blocked",
                    "detail": {},
                }

        # 4. Transform the fixed sample set and query direct LAND once each.
        def world_offset(sample: tuple[float, float, float]) -> tuple[float, float, float]:
            rotated = candidate_matrix @ np.array(
                [sample[0] * scale, sample[1] * scale, sample[2] * scale],
                dtype=np.float64,
            )
            return (float(rotated[0]), float(rotated[1]), float(rotated[2]))

        def sample_heights(samples):
            offsets = []
            heights = []
            for sample in samples:
                off = world_offset(sample)
                height_thu = height_at_game_position(
                    land_records, (candidate_x + off[0], candidate_y + off[1])
                )
                if height_thu is None:
                    return None, None
                offsets.append(off)
                heights.append(float(height_thu) * THU_TO_GU)
            return offsets, heights

        support_offsets, support_heights = sample_heights(member.support_samples)
        if support_offsets is None:
            return "reject", {"reason": "missing_land", "detail": {"group": "support"}}

        # 5. Feasible-Z sweep over the support intervals.
        e_lo, e_hi = member.embed_band_gu
        intervals = [
            (h - off[2] - e_hi, h - off[2] - e_lo)
            for off, h in zip(support_offsets, support_heights)
        ]
        span = _covered_span_closest_to(intervals, self.support_min_pass, preferred_z)
        if span is None:
            return "reject", {
                "reason": "no_support_z",
                "detail": {"required_pass": self.support_min_pass,
                           "total": len(intervals)},
            }
        solved_z = min(max(preferred_z, span[0]), span[1])

        support_pass = sum(
            1 for lo, hi in intervals if solved_z >= lo - 1e-9 and solved_z <= hi + 1e-9
        )
        embed_values = [
            h - (solved_z + off[2])
            for off, h in zip(support_offsets, support_heights)
        ]
        support_slack = min(
            (min(solved_z - lo, hi - solved_z) for lo, hi in intervals
             if solved_z >= lo - 1e-9 and solved_z <= hi + 1e-9),
            default=0.0,
        )

        # 6. Final world AABB at the solved Z.
        world_bbox = transformed_bbox(
            bbox, [candidate_x, candidate_y, solved_z], euler, scale
        )

        alignment_dot = None
        cover_pass = None
        cover_total = None
        cover_margin = None
        front_pass = None
        front_total = None
        front_margin = None

        if member.classification == "lateral_burial_opening":
            burial_world = candidate_matrix @ np.array(
                [member.burial_local[0], member.burial_local[1], member.burial_local[2]],
                dtype=np.float64,
            )
            length = math.hypot(float(burial_world[0]), float(burial_world[1]))
            if length > 1e-9:
                downhill = candidate_downhill_xy
                downhill_len = math.hypot(downhill[0], downhill[1])
                if downhill_len > 1e-9:
                    alignment_dot = -(
                        (float(burial_world[0]) * downhill[0]
                         + float(burial_world[1]) * downhill[1])
                        / length / downhill_len
                    )
            cover_offsets, cover_heights = sample_heights(member.cover_samples)
            if cover_offsets is None:
                return "reject", {"reason": "missing_land", "detail": {"group": "cover"}}
            cover_margins = [
                h - (solved_z + off[2]) - self.cover_min_gu
                for off, h in zip(cover_offsets, cover_heights)
            ]
            cover_total = len(cover_margins)
            cover_pass = sum(1 for value in cover_margins if value >= 0.0)
            cover_margin = min(cover_margins)
            if cover_pass < self.cover_min_pass:
                return "reject", {
                    "reason": "lateral_uncovered",
                    "detail": {
                        "passed": cover_pass,
                        "required": self.cover_min_pass,
                        "worst_margin_gu": cover_margin,
                    },
                }

        front_pass = None
        front_total = None
        front_margin = None

        if member.front_samples:
            front_offsets, front_heights = sample_heights(member.front_samples)
            if front_offsets is None:
                return "reject", {"reason": "missing_land", "detail": {"group": "front"}}
            front_margins = [
                (solved_z + off[2]) - h - self.front_min_height
                for off, h in zip(front_offsets, front_heights)
            ]
            front_total = len(front_margins)
            front_pass = sum(1 for value in front_margins if value >= 0.0)
            front_margin = min(front_margins)

        # Visible-body gate, source-faithful: the arrangement must emerge from
        # the candidate anchor terrain at least a configured fraction as far as
        # the recorded source arrangement did (with the plan's absolute floor).
        # Height-quantile front samples cannot express this under deep source
        # seating (their lower quantiles are legitimately underground), so they
        # are recorded as evidence while the gate uses the achieved emergence.
        member_visible = member.visible_height_gu
        if member_visible is not None:
            world_max_z = float(world_bbox["max"][2])
            achieved_visible = world_max_z - candidate_terrain_z_gu
            required_visible = max(
                self.front_min_height,
                self.front_visible_fraction * member_visible,
            )
            front_total = 1
            front_pass = 1 if achieved_visible >= required_visible else 0
            front_margin = achieved_visible - required_visible
            if front_pass < 1:
                return "reject", {
                    "reason": "visible_front",
                    "detail": {
                        "achieved_visible_gu": achieved_visible,
                        "required_visible_gu": required_visible,
                        "source_visible_gu": member_visible,
                    },
                }

        margins = [support_slack]
        if cover_margin is not None:
            margins.append(cover_margin)
        if front_margin is not None:
            margins.append(front_margin)
        stability_margin = min(margins)

        solution = {
            "profile_id": self.profile_id,
            "mode_id": member.mode_id,
            "member_ref_id": member.ref_id,
            "source_slope_bin": member.slope_bin,
            "classification": member.classification,
            "recorded_scale": scale,
            "recorded_z_offset_gu": member.z_offset_gu,
            "candidate_frame": {
                "downhill_xy": [round(frame.downhill_xy[0], 6), round(frame.downhill_xy[1], 6)],
                "downhill_angle_deg": round(math.degrees(frame.downhill_angle_rad), 6),
            },
            "rotation_radians": [round(float(v), 6) for v in euler],
            "rotation_roundtrip_residual": residual,
            "embed_band_gu": [e_lo, e_hi],
            "solved_z_gu": solved_z,
            "preferred_z_gu": preferred_z,
            "z_adjustment_gu": solved_z - preferred_z,
            "solved_interval_gu": [span[0], span[1]],
            "support_passed": support_pass,
            "support_total": len(intervals),
            "support_embed_min_gu": min(embed_values),
            "support_embed_max_gu": max(embed_values),
            "lateral_alignment_dot": None if alignment_dot is None else round(alignment_dot, 6),
            "lateral_alignment_gate": self.alignment_gate,
            "lateral_cover_passed": cover_pass,
            "lateral_cover_total": cover_total,
            "lateral_cover_margin_gu": None if cover_margin is None else round(cover_margin, 3),
            "visible_front_passed": front_pass,
            "visible_front_total": front_total,
            "visible_front_margin_gu": (
                None if front_margin is None else round(front_margin, 3)
            ),
            "visible_front_rule": "achieved_emergence_vs_source_fraction",
            "source_visible_height_gu": member.visible_height_gu,
            "stability_margin_gu": round(stability_margin, 3),
            "world_bbox": world_bbox,
            "passed": True,
        }
        return "accept", solution


def _covered_span_closest_to(intervals, minimum_pass: int, preferred_z: float):
    """Qualifying span closest to ``preferred_z`` (ties: earliest start).

    A qualifying span is covered by >= ``minimum_pass`` intervals.  Returns
    ``None`` when no span qualifies; callers clamp the preferred Z into the
    returned span.
    """

    events: list[tuple[float, int]] = []
    for lo, hi in intervals:
        if hi < lo:
            lo, hi = hi, lo
        events.append((lo, 0))
        events.append((hi, 1))
    events.sort()
    coverage = 0
    span_start = None
    best: tuple[float, float] | None = None
    best_distance = math.inf
    for value, kind in events:
        if kind == 0:
            coverage += 1
            if coverage >= minimum_pass and span_start is None:
                span_start = value
        else:
            if coverage >= minimum_pass and span_start is not None:
                if value < span_start:
                    span_start, value = value, span_start
                distance = (
                    0.0
                    if preferred_z <= value and preferred_z >= span_start
                    else min(abs(preferred_z - span_start), abs(value - preferred_z))
                )
                if distance < best_distance - 1e-12:
                    best_distance = distance
                    best = (span_start, value)
                span_start = None
            coverage -= 1
    return best


__all__ = [
    "CliffSeatingError",
    "CliffSeatingRuntime",
]
