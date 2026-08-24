"""Stage W2: deterministic city-wall composer (wall kit system).

Pipeline position
-----------------
Consumes the stage W1 wall-kit JSON (``src/procgen/wall_kit.py`` schema), a
closed wall-path polyline (plan GU), gate crossings, and a terrain height
callable; emits a stamp-shaped member doc that existing render/ESP pipelines
consume unchanged. Kit-generic: composing a different wall kit is pure JSON
authoring.

Algorithm (deterministic; no RNG anywhere)
------------------------------------------
1. Path/anchor pass: normalize the closed ring to the kit's declared winding,
   place a tangent-matched gatehouse at every road crossing, extend each side
   with native straight neck modules to a real round-tower junction, and retain
   major-turn towers only where ``rules.corner_angle_threshold_deg`` is reached.
2. Segment fill: between consecutive anchors the remaining polyline arc is
   filled by an exact-length bounded coin-change over straight-piece lengths
   (quantized to ``rules.fill_quantum_gu``, maximizing mined usage weight,
   ties broken by fewer pieces then piece id). If no exact combination
   exists, anchor insets are grown symmetrically (then one-sided) in quantum
   steps up to ``rules.anchor_inset_tolerance_gu``; failing that,
   :class:`WallComposeError` ``wall_fill_infeasible`` aborts — no silent gaps.
   Scaled fillers run only for pieces flagged ``allow_scaled_fill``.
3. Terrain following: solve one cyclic discrete deck-height state around the
   complete ring. A straight consumes one module without changing height; a
   directional authored slope consumes its measured span and changes height by
   its exact measured rise. The closed solution cannot float, rejects immediate
   opposing slopes, and has net-zero height change. Towers and gates partition
   geometry but never conceal a deck-height change.
4. Gate assembly: keep the source gate assembly at its road-relative elevation,
   derive the gatehouse mesh-bottom plane from that placement, and publish the
   complete measured footprint at that bottom elevation for downstream LAND
   authoring. The road corridor remains centered on the measured arch rather
   than expanding to gatehouse width.
5. Output: members relative to the path origin (xy) with absolute z, real
   computed bounds/hull, and per-member role/model/scale/rotation.

Rotation convention: engine placement ``world = pos + Rz(-rotz)·(scale·local)``
via ``procgen.engine_transform``; a piece's long axis (local x or y) is aligned
to the desired world heading by solving that relation analytically and
verifying numerically against the shared matrix helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from procgen import engine_transform
from procgen.wall_kit import fill_candidates, piece_by_id


class WallComposeError(ValueError):
    """Raised for infeasible wall composition (never silently degraded)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class _Anchor:
    arc: float
    kind: str  # "gate" | "corner" | "tower"
    piece_id: str
    heading_rad: float
    inset_before: float  # arc consumed on the incoming side
    inset_after: float  # arc consumed on the outgoing side
    gate: dict | None = None
    position_xy: tuple[float, float] | None = None
    protected: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class _Placed:
    model_key: str
    structural_role: str
    piece_id: str
    position: tuple[float, float, float]
    rotz_rad: float
    scale: float
    rotx_rad: float = 0.0
    is_door: bool = False
    arc: float | None = None  # arc position along the path (ring pieces only)
    meta: dict = field(default_factory=dict)


def _wrap(pi: float) -> float:
    return (pi + math.pi) % (2.0 * math.pi) - math.pi


def _heading_of(direction: np.ndarray) -> float:
    return math.atan2(float(direction[1]), float(direction[0]))


def _slice_outline(piece: dict) -> list[list[float]]:
    footprint = piece.get("footprint_slice") or {}
    outline = footprint.get("slice_outline_xy")
    if outline and len(outline) >= 3:
        return [[float(point[0]), float(point[1])] for point in outline]
    lo = footprint.get("slice_min_xy")
    hi = footprint.get("slice_max_xy")
    if not lo or not hi:
        half_l = 0.5 * float(piece["length_gu"])
        half_t = 0.5 * float(piece["thickness_gu"])
        lo, hi = [-half_l, -half_t], [half_l, half_t]
    return [
        [float(lo[0]), float(lo[1])],
        [float(hi[0]), float(lo[1])],
        [float(hi[0]), float(hi[1])],
        [float(lo[0]), float(hi[1])],
    ]


def _rotz_for_heading(heading_rad: float, long_axis: str) -> float:
    """Engine rotz aligning the piece long axis to a world heading.

    Engine maps local +X to ``(cos rotz, -sin rotz)`` and local +Y to
    ``(sin rotz, cos rotz)`` (world = Rz(-rotz)·local). Verified numerically
    against ``engine_transform.tes3_euler_to_matrix`` in the self-check.
    """
    deg = math.degrees(heading_rad)
    return math.radians(-deg if long_axis == "x" else 90.0 - deg)


def _self_check_rotation() -> None:
    for heading in (0.0, 0.4, 1.2, 2.9, -1.7):
        for axis in ("x", "y"):
            rotz = _rotz_for_heading(heading, axis)
            matrix = engine_transform.tes3_euler_to_matrix([0.0, 0.0, rotz])
            col = matrix[:2, 0] if axis == "x" else matrix[:2, 1]
            want = np.array([math.cos(heading), math.sin(heading)])
            if float(np.linalg.norm(col - want)) > 1e-6:
                raise WallComposeError(
                    "rotation_self_check",
                    f"long axis {axis} heading {heading}: {col} != {want}",
                )


class _Path:
    """Closed polyline with arc-length parameterization."""

    def __init__(self, points: list[tuple[float, float]]) -> None:
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) < 3:
            raise WallComposeError("path_too_short", "closed wall path needs >= 3 points")
        if pts[0] == pts[-1]:
            pts = pts[:-1]
        self.points = pts
        self.seg_dirs: list[np.ndarray] = []
        self.seg_lens: list[float] = []
        self.seg_starts: list[float] = [0.0]
        for i in range(len(pts)):
            a = np.array(pts[i])
            b = np.array(pts[(i + 1) % len(pts)])
            delta = b - a
            length = float(np.linalg.norm(delta))
            if length <= 0.0:
                raise WallComposeError("path_degenerate", f"zero-length segment at index {i}")
            self.seg_dirs.append(delta / length)
            self.seg_lens.append(length)
            self.seg_starts.append(self.seg_starts[-1] + length)
        self.total_length = self.seg_starts[-1]

    def locate(self, arc: float) -> tuple[np.ndarray, float]:
        """Point and segment heading at an arc position (wraps around the loop)."""
        arc %= self.total_length
        i = len(self.seg_lens) - 1
        for j, start in enumerate(self.seg_starts[:-1]):
            if arc < self.seg_starts[j + 1]:
                i = j
                break
        t = arc - self.seg_starts[i]
        point = np.array(self.points[i]) + self.seg_dirs[i] * t
        return point, _heading_of(self.seg_dirs[i])

    def project(self, xy: tuple[float, float]) -> float:
        """Arc position of the nearest point on the path."""
        p = np.array(xy)
        best_arc, best_dist = 0.0, float("inf")
        for i in range(len(self.seg_lens)):
            a = np.array(self.points[i])
            t = float(np.clip((p - a) @ self.seg_dirs[i], 0.0, self.seg_lens[i]))
            candidate = self.seg_starts[i] + t
            dist = float(np.linalg.norm(p - (a + self.seg_dirs[i] * t)))
            if dist < best_dist:
                best_arc, best_dist = candidate, dist
        return best_arc

    def turn_at_vertex(self, i: int) -> tuple[float, float, float]:
        """Exterior turn angle (rad) at vertex i plus in/out headings."""
        h_in = _heading_of(self.seg_dirs[(i - 1) % len(self.seg_dirs)])
        h_out = _heading_of(self.seg_dirs[i])
        return abs(_wrap(h_out - h_in)), h_in, h_out


def _solve_fill(target_q: int, candidates: list[dict], max_pieces: int) -> list[dict] | None:
    """Exact coin-change maximizing weight; ties: fewer pieces, then piece ids.

    Returns the chosen candidate list (with repetition) or None.
    """
    lens_q = [max(1, round(c["fill_length_gu"] / c["quantum"])) for c in candidates]
    neg_inf = float("-inf")
    best_weight = [neg_inf] * (target_q + 1)
    best_count = [math.inf] * (target_q + 1)
    best_choice: list[tuple[int, int] | None] = [None] * (target_q + 1)
    best_weight[0] = 0.0
    best_count[0] = 0
    for s in range(1, target_q + 1):
        for idx, lq in enumerate(lens_q):
            if lq > s or best_weight[s - lq] == neg_inf:
                continue
            prev_count = best_count[s - lq]
            if prev_count + 1 > max_pieces:
                continue
            weight = best_weight[s - lq] + candidates[idx]["weight"]
            count = prev_count + 1
            if (weight, -count) > (best_weight[s], -best_count[s]):
                best_weight[s] = weight
                best_count[s] = count
                best_choice[s] = (idx, s - lq)
    if best_weight[target_q] == neg_inf:
        return None
    chosen: list[dict] = []
    s = target_q
    while s > 0:
        idx, prev = best_choice[s]  # type: ignore[misc]
        chosen.append(candidates[idx])
        s = prev
    return chosen


def compose_city_wall(
    path_points: list[tuple[float, float]],
    gates: list[dict],
    terrain,
    kit: dict,
    stamp_id: str = "composed_city_wall",
) -> dict:
    """Compose a city wall; returns a stamp-shaped member doc.

    ``terrain`` is any callable ``(x, y) -> z`` in the same GU frame as the
    path. Gates: ``[{"position_xy": [x, y], "heading_deg": float|omitted}]``;
    omitted headings use the path tangent at the projected crossing.
    """
    _self_check_rotation()
    rules = kit["rules"]
    quantum = float(rules["fill_quantum_gu"])
    tower_piece = _tower_piece(kit)

    requested_winding = str(rules.get("path_winding", "preserve")).lower()
    if requested_winding not in {"preserve", "clockwise", "counterclockwise"}:
        raise WallComposeError(
            "wall_orientation_infeasible",
            f"unknown path_winding {requested_winding!r}",
        )
    signed_area = 0.5 * sum(
        float(path_points[i][0]) * float(path_points[(i + 1) % len(path_points)][1])
        - float(path_points[(i + 1) % len(path_points)][0]) * float(path_points[i][1])
        for i in range(len(path_points))
    )
    if (
        requested_winding == "clockwise" and signed_area > 0.0
    ) or (
        requested_winding == "counterclockwise" and signed_area < 0.0
    ):
        path_points = list(reversed(path_points))

    path = _Path(path_points)

    gate_config = kit["gate"]
    junction_neck_piece = piece_by_id(
        kit, str(gate_config.get("junction_neck_piece_id", ""))
    )
    junction_neck_count = int(gate_config.get("junction_neck_piece_count", 0))
    junction_neck_span = (
        float(junction_neck_piece["length_gu"]) * junction_neck_count
        if junction_neck_piece is not None and junction_neck_count > 0
        else 0.0
    )

    # --- anchor pass -----------------------------------------------------
    anchors: list[_Anchor] = []
    for gate in gates:
        arc = path.project(tuple(gate["position_xy"]))
        _, tangent = path.locate(arc)
        if gate.get("heading_deg") is None:
            heading = tangent
        else:
            road_heading = math.radians(float(gate["heading_deg"]))
            # The gatehouse spans the wall and its passage runs along the
            # arterial.  Choose the road normal that agrees with the local
            # wall direction; the two normals have identical footprints but
            # opposite local door/frame offsets.
            normals = (road_heading + math.pi / 2.0, road_heading - math.pi / 2.0)
            heading = min(normals, key=lambda value: abs(_wrap(value - tangent)))
        gh_piece = piece_by_id(kit, kit["gate"]["gatehouse_piece"])
        if gh_piece is None:
            raise WallComposeError("gate_missing", "kit has no gatehouse piece")
        crossing_xy = np.asarray(gate["position_xy"], dtype=float)
        gate_yaw = _rotz_for_heading(heading, str(gh_piece["long_axis"]))
        gate_rot = engine_transform.tes3_euler_to_matrix(
            [0.0, 0.0, gate_yaw]
        )[:2, :2]
        door_offset = kit["gate"].get("door_offset_local")
        if not isinstance(door_offset, list) or len(door_offset) != 3:
            raise WallComposeError(
                "gate_missing", "gate requires a measured door_offset_local"
            )
        passage_center_y = float(
            kit["gate"].get("passage_center_local_y_gu", 0.0)
        )
        gate_pivot_xy = crossing_xy - gate_rot @ np.asarray(
            [float(door_offset[0]), passage_center_y], dtype=float
        )
        half = 0.5 * float(gh_piece["length_gu"]) + junction_neck_span
        anchors.append(
            _Anchor(
                arc=arc,
                kind="gate",
                piece_id=str(gh_piece["piece_id"]),
                heading_rad=heading,
                inset_before=half,
                inset_after=half,
                gate=gate,
                position_xy=(float(gate_pivot_xy[0]), float(gate_pivot_xy[1])),
                protected=True,
                meta={
                    "gate_id": gate.get("gate_id"),
                    "gate_crossing_xy": [float(crossing_xy[0]), float(crossing_xy[1])],
                },
            )
        )
    for vi in range(len(path.points)):
        turn, h_in, h_out = path.turn_at_vertex(vi)
        if math.degrees(turn) < float(rules["corner_angle_threshold_deg"]):
            continue
        arc = path.seg_starts[vi]
        # User design (2026-08-20): angle changes happen AT TOWERS — round
        # drums absorb any turn, matching the source castle. Corner blocks
        # stay in the kit for authored/small-wall use only.
        piece = tower_piece
        if piece is None:
            raise WallComposeError("anchor_missing", "no tower piece in kit")
        bisector = path.seg_dirs[(vi - 1) % len(path.seg_dirs)] + path.seg_dirs[vi]
        norm = float(np.linalg.norm(bisector))
        if norm < 1e-6:  # 180-degree reversal: orient across the vertex
            bisector = path.seg_dirs[vi] * 0.0 + np.array([-path.seg_dirs[vi][1], path.seg_dirs[vi][0]])
        else:
            bisector = bisector / norm
        heading = _heading_of(bisector)
        # The vertex is the wall endpoint.  Put the tower on that exact
        # vertex and let its footprint hide the two wall ends; consuming a
        # tower half-length here pulled the walls back and exposed seams.
        proj = 0.0
        anchors.append(
            _Anchor(
                arc=arc,
                kind="tower",
                piece_id=str(piece["piece_id"]),
                heading_rad=heading,
                inset_before=proj,
                inset_after=proj,
                meta={"corner_turn_deg": round(math.degrees(turn), 3)},
            )
        )

    # Periodic towers on long anchor-free stretches (user decision 2026-08-20;
    # postern gates are rejected — gates exist only at road crossings).
    spacing = float(rules.get("tower_spacing_gu") or 0.0)
    if spacing > 0.0 and tower_piece is not None:
        anchors.sort(key=lambda a: a.arc)
        periodic: list[_Anchor] = []
        n_anchors = len(anchors)
        for ai in range(n_anchors):
            a = anchors[ai]
            b = anchors[(ai + 1) % n_anchors]
            gap = (b.arc - a.arc) % path.total_length
            if gap <= spacing:
                continue
            count = int(gap // spacing)
            for k in range(count):
                arc = (a.arc + gap * (k + 1) / (count + 1)) % path.total_length
                _point, tangent = path.locate(arc)
                periodic.append(
                    _Anchor(
                        arc=arc,
                        kind="tower",
                        piece_id=str(tower_piece["piece_id"]),
                        heading_rad=tangent,
                        inset_before=0.0,
                        inset_after=0.0,
                    )
                )
        anchors.extend(periodic)

    # The purpose-built junction towers already absorb the wall turn at a
    # gate. Reserve a clear arc beyond them so an ordinary tower cannot form
    # the clipped, over-dense clusters that the junction is meant to avoid.
    gate_clearance = float(rules.get("gate_tower_clearance_gu") or 0.0)
    if gate_clearance > 0.0:
        gate_arcs = [anchor.arc for anchor in anchors if anchor.kind == "gate"]
        anchors = [
            anchor
            for anchor in anchors
            if anchor.kind != "tower"
            or all(
                min(
                    abs(anchor.arc - gate_arc),
                    path.total_length - abs(anchor.arc - gate_arc),
                )
                >= gate_clearance
                for gate_arc in gate_arcs
            )
        ]
    anchors.sort(key=lambda a: a.arc)

    # Merge anchors that land too close to leave room for even one fill
    # piece (dense vertex clusters on curvy rings): drop the non-gate member
    # of each too-close pair until every gap fits at least one piece.
    while len(anchors) > 3:
        merged = False
        for k in range(len(anchors)):
            a = anchors[k]
            b = anchors[(k + 1) % len(anchors)]
            gap = (b.arc - a.arc) % path.total_length
            min_tower_gap = float(rules.get("min_tower_separation_gu") or 0.0)
            if (min_tower_gap > 0.0 and a.kind == "tower" and b.kind == "tower"
                    and gap < min_tower_gap):
                if a.protected and b.protected:
                    continue
                if a.protected:
                    anchors.pop((k + 1) % len(anchors))
                else:
                    anchors.pop(k)
                merged = True
                break
            if gap >= a.inset_after + b.inset_before + quantum:
                continue
            if a.protected and b.protected:
                continue
            if a.protected:
                anchors.pop((k + 1) % len(anchors))
            elif b.protected:
                anchors.pop(k)
            elif a.kind != "gate":
                anchors.pop(k)
            else:
                anchors.pop((k + 1) % len(anchors))
            merged = True
            break
        if not merged:
            break
    anchors.sort(key=lambda a: a.arc)

    def gate_outer_connection(anchor: _Anchor, *, after: bool) -> np.ndarray:
        """Center of the real junction tower beyond a gate's straight neck."""

        if anchor.kind != "gate" or anchor.position_xy is None:
            raise WallComposeError("gate_missing", "gate junction requires a gate anchor")
        gate_piece = piece_by_id(kit, anchor.piece_id)
        assert gate_piece is not None
        gate_yaw = _rotz_for_heading(anchor.heading_rad, str(gate_piece["long_axis"]))
        gate_rot = engine_transform.tes3_euler_to_matrix(
            [0.0, 0.0, gate_yaw]
        )[:2, :2]
        end_a = np.asarray(gate_piece["end_a_local"], dtype=float)
        end_b = np.asarray(gate_piece["end_b_local"], dtype=float)
        axis = end_b - end_a
        axis /= float(np.linalg.norm(axis))
        endpoint = end_b if after else end_a
        local = endpoint + axis * (junction_neck_span if after else -junction_neck_span)
        return np.asarray(anchor.position_xy, dtype=float) + gate_rot @ local

    # --- segment fill ----------------------------------------------------
    placed: list[_Placed] = []
    n = len(anchors)
    fill_candidates_all = fill_candidates(kit)
    if not fill_candidates_all:
        raise WallComposeError("wall_fill_infeasible", "kit has no allowed straight fill pieces")
    shortest_fill = min(float(row["fill_length_gu"]) for row in fill_candidates_all)

    def cover_half(anchor: _Anchor) -> float:
        piece = piece_by_id(kit, anchor.piece_id)
        assert piece is not None
        if anchor.kind == "gate":
            return 0.5 * float(piece["length_gu"])
        return 0.5 * float(piece["thickness_gu"])

    for ai in range(n):
        a = anchors[ai]
        b = anchors[(ai + 1) % n]
        gap_arc = (b.arc - a.arc) % path.total_length
        start_arc = a.arc + a.inset_after
        end_arc = b.arc - b.inset_before
        start_point = path.locate(start_arc)[0]
        if a.kind == "gate" and a.position_xy is not None:
            start_point = gate_outer_connection(a, after=True)
        end_point = path.locate(end_arc)[0]
        if b.kind == "gate" and b.position_xy is not None:
            end_point = gate_outer_connection(b, after=False)
        chord = np.array(end_point) - np.array(start_point)
        chord_len = float(np.linalg.norm(chord))
        mesh_overlap = float(rules.get("anchor_mesh_overlap_gu", 0.0))
        # Gate-edge towers are deliberately positioned from the gatehouse axis
        # and can sit off the fitted centerline. Fit against the actual anchor
        # centers, not their unrelated path-arc separation.
        fill_len = chord_len + 2.0 * mesh_overlap
        if fill_len <= quantum / 2.0:
            continue
        # A short residual between anchor footprints is intentionally omitted:
        # the tower/gatehouse bodies already overlap across it.  This is the
        # kit-constrained collapse rule, not permission to use a smaller wall.
        if fill_len < shortest_fill and chord_len <= cover_half(a) + cover_half(b):
            continue
        candidates = []
        for cand in fill_candidates_all:
            entry = dict(cand)
            entry["quantum"] = quantum
            candidates.append(entry)
        tol_steps = int(float(rules["anchor_inset_tolerance_gu"]) // quantum)
        solution = None
        used_shift = 0.0
        for step in range(tol_steps + 1):
            shift = step * quantum
            for sign in ((1.0, -1.0) if step else (1.0,)):
                target = fill_len - sign * shift
                if target < quantum / 2.0:
                    continue
                solution = _solve_fill(
                    int(round(target / quantum)), candidates, int(rules["max_fill_pieces"])
                )
                if solution is not None:
                    used_shift = sign * shift
                    break
            if solution is not None:
                break
        if solution is None:
            raise WallComposeError(
                "wall_fill_infeasible",
                f"gap {fill_len:.1f} GU after arc {start_arc:.1f}"
                f" ({a.kind}@{a.arc:.1f} -> {b.kind}@{b.arc:.1f}) has no exact"
                f" combination within tolerance {rules['anchor_inset_tolerance_gu']} GU",
            )
        # The solver's target controls the selected chain length. Placement is
        # independent of the arc residue: center the actual chain on the
        # straight chord so both anchor footprints receive the same overlap.
        # The old cursor-based residue mixed arc and chord coordinates and
        # could move an entire run outside its two towers.
        a.inset_after += used_shift / 2.0
        b.inset_before += used_shift / 2.0
        start_arc = a.arc + a.inset_after
        # Chord layout (user design 2026-08-20): every piece of one run shares
        # a single heading — the straight line between the two anchor faces —
        # so runs read as straight lines between towers with no sub-threshold
        # kinks. Residual sagitta vs the polyline is bounded by the turn
        # threshold and dwarfed by the wall thickness.
        chain_length = sum(float(cand["fill_length_gu"]) for cand in solution)
        # Recompute the chord after any tolerance shift changed arc-based
        # anchors. Explicit gate-edge positions remain authoritative.
        start_point = path.locate(start_arc)[0]
        if a.kind == "gate" and a.position_xy is not None:
            start_point = gate_outer_connection(a, after=True)
        end_arc = b.arc - b.inset_before
        end_point = path.locate(end_arc)[0]
        if b.kind == "gate" and b.position_xy is not None:
            end_point = gate_outer_connection(b, after=False)
        chord = np.array(end_point) - np.array(start_point)
        chord_len = float(np.linalg.norm(chord))
        chord_dir = chord / chord_len if chord_len > 1e-9 else path.seg_dirs[0]
        heading = math.atan2(float(chord_dir[1]), float(chord_dir[0]))
        chain_offset = (chord_len - chain_length) / 2.0
        chain_cursor = chain_offset
        for cand in solution:
            piece = cand["piece"]
            scale = float(cand["scale"])
            length = float(cand["fill_length_gu"])
            offset = chain_cursor + length / 2.0
            point = np.array(start_point) + chord_dir * offset
            mid_arc = start_arc + offset
            rotz = _rotz_for_heading(heading, str(piece["long_axis"]))
            placed.append(
                _Placed(
                    model_key=str(piece["model_key"]),
                    structural_role="straight",
                    piece_id=str(piece["piece_id"]),
                    position=(float(point[0]), float(point[1]), 0.0),
                    rotz_rad=rotz,
                    scale=scale,
                    arc=mid_arc,
                    meta={"fill": True, "segment": ai},
                )
            )
            chain_cursor += length

    # Gate junctions are structural, not decorative overlaps. Native straight
    # modules run from each measured gatehouse end to a separated round tower;
    # the main ring also terminates at that tower center, so both wall chains
    # meet beneath the tower footprint without the tower clipping the gatehouse.
    if junction_neck_piece is not None and junction_neck_count > 0:
        neck_length = float(junction_neck_piece["length_gu"])
        neck_center_local = 0.5 * (
            np.asarray(junction_neck_piece["end_a_local"], dtype=float)
            + np.asarray(junction_neck_piece["end_b_local"], dtype=float)
        )
        for gate_index, anchor in enumerate(anchors):
            if anchor.kind != "gate" or anchor.position_xy is None:
                continue
            gate_piece = piece_by_id(kit, anchor.piece_id)
            assert gate_piece is not None
            yaw = _rotz_for_heading(anchor.heading_rad, str(gate_piece["long_axis"]))
            gate_rot = engine_transform.tes3_euler_to_matrix(
                [0.0, 0.0, yaw]
            )[:2, :2]
            end_a = np.asarray(gate_piece["end_a_local"], dtype=float)
            end_b = np.asarray(gate_piece["end_b_local"], dtype=float)
            axis = end_b - end_a
            axis /= float(np.linalg.norm(axis))
            for side, endpoint, sign, segment in (
                ("before", end_a, -1.0, (gate_index - 1) % n),
                ("after", end_b, 1.0, gate_index),
            ):
                for neck_index in range(junction_neck_count):
                    center_local = endpoint + axis * sign * (
                        (neck_index + 0.5) * neck_length
                    )
                    center_world = (
                        np.asarray(anchor.position_xy, dtype=float)
                        + gate_rot @ center_local
                    )
                    origin = center_world - gate_rot @ neck_center_local
                    arc = anchor.arc + sign * (
                        0.5 * float(gate_piece["length_gu"])
                        + (neck_index + 0.5) * neck_length
                    )
                    placed.append(
                        _Placed(
                            model_key=str(junction_neck_piece["model_key"]),
                            structural_role="gate_neck",
                            piece_id=str(junction_neck_piece["piece_id"]),
                            position=(float(origin[0]), float(origin[1]), 0.0),
                            rotz_rad=yaw,
                            scale=1.0,
                            arc=arc,
                            meta={
                                "fill": True,
                                "gate_neck": True,
                                "gate_side": side,
                                "gate_id": anchor.gate.get("gate_id") if anchor.gate else None,
                                "neck_index": neck_index,
                                "segment": segment,
                            },
                        )
                    )

    # --- anchor piece placement -----------------------------------------
    for a in anchors:
        if a.kind == "gate":
            continue  # gate assembly below places all gatehouse tiers
        piece = piece_by_id(kit, a.piece_id)
        assert piece is not None
        point = np.array(a.position_xy) if a.position_xy is not None else path.locate(a.arc)[0]
        rotz = _rotz_for_heading(a.heading_rad, str(piece["long_axis"]))
        # Center the piece's geometric long-axis midpoint on the anchor arc;
        # several kit meshes (e.g. wc_01) have pivots far off center.
        anchor_scale = (
            float(rules.get("tower_scale", 1.0)) if a.kind == "tower" else 1.0
        )
        center_local = np.array(
            [
                (float(piece["end_a_local"][0]) + float(piece["end_b_local"][0])) / 2.0,
                (float(piece["end_a_local"][1]) + float(piece["end_b_local"][1])) / 2.0,
            ]
        ) * anchor_scale
        rot2 = engine_transform.tes3_euler_to_matrix([0.0, 0.0, rotz])[:2, :2]
        pos_xy = np.array([float(point[0]), float(point[1])]) - rot2 @ center_local
        placed.append(
            _Placed(
                model_key=str(piece["model_key"]),
                structural_role=a.kind,
                piece_id=a.piece_id,
                position=(float(pos_xy[0]), float(pos_xy[1]), 0.0),
                rotz_rad=rotz,
                scale=anchor_scale,
                arc=a.arc,
                meta={"anchor_kind": a.kind, "anchor_arc": round(a.arc, 6), **a.meta},
            )
        )

    # --- terrain following -----------------------------------------------
    burial = float(rules["burial_depth_gu"])
    minimum_coverage = float(
        rules.get(
            "minimum_wall_bottom_coverage_fraction",
            rules.get("regular_wall_bottom_coverage_fraction", 0.005),
        )
    )
    maximum_coverage = float(
        rules.get("maximum_wall_bottom_coverage_fraction", 0.15)
    )
    minimum_foundation_ground = float(
        rules.get("minimum_foundation_ground_z_gu", 0.0)
    )
    regular_coverage = minimum_coverage
    doubled_coverage = float(rules.get("doubled_wall_bottom_coverage_fraction", 0.15))
    wall_tier_count = max(1, int(rules.get("wall_tier_count", 1)))
    if not 0.0 <= minimum_coverage <= maximum_coverage <= 1.0:
        raise WallComposeError(
            "height_transition_infeasible",
            "wall bottom coverage must satisfy 0 <= minimum <= maximum <= 1",
        )

    def footprint_ground(
        piece: dict, xy: tuple[float, float], rotz: float, scale: float
    ) -> float:
        return footprint_ground_range(piece, xy, rotz, scale)[0]

    def footprint_ground_range(
        piece: dict, xy: tuple[float, float], rotz: float, scale: float
    ) -> tuple[float, float]:
        outline = piece["footprint_slice"]["slice_outline_xy"]
        rot = engine_transform.tes3_euler_to_matrix([0.0, 0.0, rotz])[:2, :2]
        center = np.array(xy, dtype=float)
        samples = [
            float(
                terrain(
                    *(center + rot @ (np.asarray(local_xy, dtype=float) * scale))
                )
            )
            for local_xy in outline
        ]
        samples.append(float(terrain(float(center[0]), float(center[1]))))
        return min(samples), max(samples)

    def tier_count_for(pl: _Placed, piece: dict) -> int:
        if pl.structural_role != "straight" or not piece.get("stackable"):
            return 1
        if float(piece.get("tier_height_gu") or 0.0) <= 0.0:
            return 1
        return wall_tier_count

    def effective_height(piece: dict, scale: float, tiers: int) -> float:
        return float(scale) * (
            float(piece["height_gu"])
            + max(0, tiers - 1) * float(piece.get("tier_height_gu") or 0.0)
        )

    def required_coverage(piece: dict, scale: float, tiers: int) -> float:
        fraction = doubled_coverage if tiers > 1 else regular_coverage
        return max(burial, fraction * effective_height(piece, scale, tiers))

    def target_level(pl: _Placed, piece: dict) -> float:
        return footprint_ground(
            piece,
            (pl.position[0], pl.position[1]),
            pl.rotz_rad,
            pl.scale,
        ) - required_coverage(piece, pl.scale, tier_count_for(pl, piece))

    def sit(pl: _Placed) -> None:
        piece = piece_by_id(kit, pl.piece_id)
        assert piece is not None
        ground = footprint_ground(
            piece,
            (pl.position[0], pl.position[1]),
            pl.rotz_rad,
            pl.scale,
        )
        # Mesh hangs full_min (= -base_offset) below its pivot. The configured
        # coverage target moves the measured bottom below the lowest evaluated
        # terrain sample instead of relying on a fixed burial depth.
        base = float(piece["base_offset_gu"]) * pl.scale
        pl.position = (
            pl.position[0],
            pl.position[1],
            ground - required_coverage(piece, pl.scale, tier_count_for(pl, piece)) + base,
        )

    for pl in placed:
        sit(pl)

    # The deck height is one continuous state around the entire closed ring.
    # Anchors split XY runs, but neither towers nor gates may hide a height
    # discontinuity. A straight module preserves the state and an authored
    # slope changes it by one measured rise.
    slope_assembly = kit.get("slope_assembly", {})
    transition_piece = piece_by_id(kit, str(slope_assembly.get("slope_piece_id", "")))
    reverse_transition_piece = piece_by_id(
        kit, str(slope_assembly.get("reverse_slope_piece_id", ""))
    )
    transition_pieces = [
        piece for piece in (transition_piece, reverse_transition_piece) if piece is not None
    ]
    if not transition_pieces:
        raise WallComposeError("height_transition_infeasible", "wall kit has no slope piece")
    transition_wall = piece_by_id(kit, str(slope_assembly.get("wall_piece_id", "")))
    if transition_wall is None or not isinstance(transition_wall.get("walk_surface"), dict):
        raise WallComposeError(
            "height_transition_infeasible", "slope assembly wall has no walk anchors"
        )
    wall_lateral = transition_wall["walk_surface"].get("lateral_bounds_gu")
    if not isinstance(wall_lateral, list) or len(wall_lateral) != 2:
        raise WallComposeError(
            "height_transition_infeasible", "slope assembly lacks measured walkway widths"
        )
    wall_walk_width = float(wall_lateral[1]) - float(wall_lateral[0])

    def transition_parameters(piece: dict) -> tuple[float, float, float]:
        surface = piece.get("walk_surface")
        if not isinstance(surface, dict):
            raise WallComposeError("height_transition_infeasible", "slope has no walk anchors")
        lateral = surface.get("lateral_bounds_gu")
        if not isinstance(lateral, list) or len(lateral) != 2:
            raise WallComposeError(
                "height_transition_infeasible", "slope lacks measured walkway width"
            )
        slope_width = float(lateral[1]) - float(lateral[0])
        scale = wall_walk_width / slope_width
        scale_min, scale_max = (float(value) for value in piece["scale_range"])
        if not (scale_min - 1e-9 <= scale <= scale_max + 1e-9):
            raise WallComposeError(
                "height_transition_infeasible",
                f"walkway width match requires {piece['piece_id']} scale {scale:.9f}",
            )
        axis = 0 if piece["long_axis"] == "x" else 1
        entry = [float(value) for value in surface["entry_local_gu"]]
        exit_anchor = [float(value) for value in surface["exit_local_gu"]]
        start, end = (entry, exit_anchor) if entry[axis] < exit_anchor[axis] else (exit_anchor, entry)
        delta = (end[2] - start[2]) * scale
        if abs(delta) <= 1e-9:
            raise WallComposeError("height_transition_infeasible", "slope rise must be nonzero")
        return scale, delta, start[2]

    for piece in transition_pieces:
        transition_parameters(piece)
    transition_overlap = float(slope_assembly.get("minimum_deck_overlap_gu", 0.0))
    if abs(transition_overlap) > 1e-9:
        raise WallComposeError(
            "height_transition_infeasible",
            "authored slope endpoints require exact 0 GU overlap",
        )

    ring = sorted(
        (pl for pl in placed if pl.arc is not None),
        key=lambda pl: pl.arc,  # type: ignore[arg-type]
    )
    run_members: dict[int, list[_Placed]] = {}
    run_order: list[int] = []
    for pl in ring:
        if pl.structural_role != "straight":
            continue
        segment = int(pl.meta["segment"])
        run_members.setdefault(segment, []).append(pl)
        if segment not in run_order:
            run_order.append(segment)
    if not run_order:
        raise WallComposeError("height_transition_infeasible", "wall has no straight runs")

    tower_height_changes_enabled = bool(rules.get("tower_height_changes_enabled", False))
    if tower_height_changes_enabled:
        raise WallComposeError(
            "height_transition_infeasible",
            "tower_height_changes_enabled would permit hidden deck discontinuities",
        )

    def slice_area(piece: dict) -> float:
        outline = [np.asarray(point, dtype=float) for point in _slice_outline(piece)]
        return 0.5 * abs(
            sum(
                float(outline[i][0] * outline[(i + 1) % len(outline)][1]
                      - outline[(i + 1) % len(outline)][0] * outline[i][1])
                for i in range(len(outline))
            )
        )

    def geometric_center(pl: _Placed, piece: dict) -> np.ndarray:
        local = np.array(
            [
                (float(piece["end_a_local"][0]) + float(piece["end_b_local"][0])) / 2.0,
                (float(piece["end_a_local"][1]) + float(piece["end_b_local"][1])) / 2.0,
            ],
            dtype=float,
        ) * pl.scale
        rot = engine_transform.tes3_euler_to_matrix(
            [0.0, 0.0, pl.rotz_rad]
        )[:2, :2]
        return np.asarray(pl.position[:2], dtype=float) + rot @ local

    def option_target(option: dict) -> float:
        piece = option["piece"]
        scale = float(option["scale"])
        origin = option["origin"]
        terrain_floor = footprint_ground(
            piece,
            (float(origin[0]), float(origin[1])),
            float(option["rotz"]),
            scale,
        ) - required_coverage(piece, scale, 1)
        if option["kind"] == "straight":
            return terrain_floor
        return (
            terrain_floor
            - float(option["slot_deck_offset"])
            + float(option["start_anchor_z"]) * scale
            + float(piece["base_offset_gu"]) * scale
        )

    wall_length = float(transition_wall["length_gu"])
    slope_rise = abs(transition_parameters(transition_pieces[0])[1])
    if any(abs(abs(transition_parameters(piece)[1]) - slope_rise) > 1e-6 for piece in transition_pieces):
        raise WallComposeError("height_transition_infeasible", "directional slopes have different rises")
    if any(
        abs(float(piece["length_gu"]) * transition_parameters(piece)[0] - 2.0 * wall_length) > 1e-6
        for piece in transition_pieces
    ):
        raise WallComposeError(
            "height_transition_infeasible",
            "authored slope must replace exactly two straight wall modules",
        )

    slots: list[dict] = []
    run_bounds: dict[int, tuple[int, int]] = {}
    for segment in sorted(run_members):
        members = sorted(run_members[segment], key=lambda member: float(member.arc))
        run_members[segment] = members
        start_index = len(slots)
        centers = [
            geometric_center(member, piece_by_id(kit, member.piece_id))  # type: ignore[arg-type]
            for member in members
        ]
        if len(centers) > 1:
            chain_dir = centers[-1] - centers[0]
            chain_dir /= float(np.linalg.norm(chain_dir))
        else:
            chain_dir = np.asarray(path.locate(float(members[0].arc))[1], dtype=float)
        first_piece = piece_by_id(kit, members[0].piece_id)
        assert first_piece is not None
        run_start = centers[0] - chain_dir * float(first_piece["length_gu"]) * members[0].scale / 2.0
        for local_index, member in enumerate(members):
            straight = piece_by_id(kit, member.piece_id)
            assert straight is not None
            if abs(float(straight["length_gu"]) * member.scale - wall_length) > 1e-6:
                raise WallComposeError(
                    "height_transition_infeasible",
                    "continuous slope solver requires native wall-module spans",
                )
            straight_option = {
                "kind": "straight",
                "consume": 1,
                "delta": 0.0,
                "delta_steps": 0,
                "piece": straight,
                "scale": member.scale,
                "rotz": member.rotz_rad,
                "origin": np.asarray(member.position[:2], dtype=float),
                "area": slice_area(straight) * member.scale * member.scale,
                "member": member,
                "segment": segment,
                "local_index": local_index,
            }
            terrain_min, terrain_max = footprint_ground_range(
                straight,
                (float(member.position[0]), float(member.position[1])),
                member.rotz_rad,
                member.scale,
            )
            straight_option["terrain_ranges"] = [
                (terrain_min, terrain_max, 0.0)
            ]
            straight_option["wall_body_height_gu"] = float(straight["height_gu"]) * member.scale
            straight_option["minimum_level"] = (
                minimum_foundation_ground
                - maximum_coverage * float(straight["height_gu"]) * member.scale
            )
            straight_option["target"] = (
                terrain_max
                - required_coverage(straight, member.scale, 1)
            )
            options = [straight_option]
            if (
                local_index + 1 < len(members)
                and not member.meta.get("gate_neck")
                and not members[local_index + 1].meta.get("gate_neck")
            ):
                for directional_piece in transition_pieces:
                    scale, delta, start_anchor_z = transition_parameters(directional_piece)
                    desired_center = run_start + chain_dir * (
                        local_index * wall_length + float(directional_piece["length_gu"]) * scale / 2.0
                    )
                    center_local = np.array(
                        [
                            (float(directional_piece["end_a_local"][0]) + float(directional_piece["end_b_local"][0])) / 2.0,
                            (float(directional_piece["end_a_local"][1]) + float(directional_piece["end_b_local"][1])) / 2.0,
                        ],
                        dtype=float,
                    ) * scale
                    rot = engine_transform.tes3_euler_to_matrix(
                        [0.0, 0.0, member.rotz_rad]
                    )[:2, :2]
                    slot_surface = straight.get("walk_surface")
                    if not isinstance(slot_surface, dict):
                        raise WallComposeError(
                            "height_transition_infeasible", "candidate wall has no walk anchors"
                        )
                    option = {
                        "kind": "slope",
                        "consume": 2,
                        "delta": delta,
                        "delta_steps": int(round(delta / slope_rise)),
                        "piece": directional_piece,
                        "scale": scale,
                        "rotz": member.rotz_rad,
                        "origin": desired_center - rot @ center_local,
                        "area": slice_area(directional_piece) * scale * scale,
                        "member": member,
                        "segment": segment,
                        "local_index": local_index,
                        "slot_deck_offset": (
                            float(straight["base_offset_gu"])
                            + float(slot_surface["surface_z_gu"])
                        ) * member.scale,
                        "start_anchor_z": start_anchor_z,
                    }
                    next_member = members[local_index + 1]
                    next_piece = piece_by_id(kit, next_member.piece_id)
                    assert next_piece is not None
                    next_min, next_max = footprint_ground_range(
                        next_piece,
                        (float(next_member.position[0]), float(next_member.position[1])),
                        next_member.rotz_rad,
                        next_member.scale,
                    )
                    option["terrain_ranges"] = [
                        (terrain_min, terrain_max, float(delta) * 0.25),
                        (next_min, next_max, float(delta) * 0.75),
                    ]
                    option["wall_body_height_gu"] = float(transition_wall["height_gu"])
                    option["minimum_level"] = max(
                        minimum_foundation_ground
                        - maximum_coverage
                        * float(transition_wall["height_gu"])
                        * float(option["scale"])
                        - float(level_offset)
                        for _terrain_min, _terrain_max, level_offset
                        in option["terrain_ranges"]
                    )
                    option["target"] = sum(
                        float(high)
                        - required_coverage(transition_wall, float(option["scale"]), 1)
                        - float(level_offset)
                        for _low, high, level_offset in option["terrain_ranges"]
                    ) / len(option["terrain_ranges"])
                    options.append(option)
            slots.append({"segment": segment, "options": options})
        run_bounds[segment] = (start_index, len(slots))

    slope_candidate_count = sum(
        option["kind"] == "slope"
        for slot in slots
        for option in slot["options"]
    )
    phase_candidates = sorted(
        {
            round(float(option["target"]) % slope_rise, 7)
            for slot in slots
            for option in slot["options"]
        }
    )
    all_targets = [
        float(option["target"])
        for slot in slots
        for option in slot["options"]
    ]
    best: tuple[float, int, float, list[tuple[dict, int]]] | None = None
    slope_penalty = float(rules.get("slope_complexity_penalty_gu", 0.0))
    max_foundation_gap = float(rules.get("max_wall_foundation_gap_gu", 0.0))
    if slope_penalty < 0.0:
        raise WallComposeError(
            "height_transition_infeasible", "slope_complexity_penalty_gu must be nonnegative"
        )
    for phase in phase_candidates:
        q_min = math.floor((min(all_targets) - phase) / slope_rise) - 2
        q_max = math.floor((max(all_targets) - phase) / slope_rise)
        for start_q in range(q_min, q_max + 1):
            states: list[dict[int, tuple[float, int, list[tuple[dict, int]]]]] = [
                {} for _ in range(len(slots) + 1)
            ]
            states[0][start_q] = (0.0, 0, [])
            for slot_index in range(len(slots)):
                for q, (cost, slope_count, chosen) in list(states[slot_index].items()):
                    for option in slots[slot_index]["options"]:
                        if (
                            chosen
                            and chosen[-1][0]["kind"] == "slope"
                            and option["kind"] == "slope"
                            and int(chosen[-1][0]["delta_steps"])
                            * int(option["delta_steps"]) < 0
                        ):
                            continue
                        consume = int(option["consume"])
                        end_index = slot_index + consume
                        if end_index > len(slots):
                            continue
                        if any(
                            slots[index]["segment"] != slots[slot_index]["segment"]
                            for index in range(slot_index, end_index)
                        ):
                            continue
                        level = phase + q * slope_rise
                        target = float(option["target"])
                        if level > target + 1e-6:
                            continue
                        if level < float(option["minimum_level"]) - 1e-6:
                            continue
                        if any(
                            level + float(level_offset)
                            > float(terrain_min) + max_foundation_gap + 1e-6
                            for terrain_min, _terrain_max, level_offset
                            in option["terrain_ranges"]
                        ):
                            continue
                        next_q = q + int(option["delta_steps"])
                        if next_q < q_min or next_q > q_max:
                            continue
                        buried = abs(target - level) * float(option["area"])
                        if option["kind"] == "slope":
                            buried += slope_penalty * float(option["area"])
                        next_value = (
                            cost + buried,
                            slope_count + (1 if option["kind"] == "slope" else 0),
                            chosen + [(option, q)],
                        )
                        previous = states[end_index].get(next_q)
                        if previous is None or (next_value[0], next_value[1]) < (
                            previous[0], previous[1]
                        ):
                            states[end_index][next_q] = next_value
            closed = states[len(slots)].get(start_q)
            if closed is None:
                continue
            candidate = (closed[0], closed[1], phase, closed[2])
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    if best is None:
        raise WallComposeError(
            "height_transition_infeasible",
            "no closed continuous slope/straight terrain solution",
        )

    old_straights = {
        id(member)
        for members in run_members.values()
        for member in members
    }
    placed[:] = [member for member in placed if id(member) not in old_straights]
    solved_runs: dict[int, list[_Placed]] = {segment: [] for segment in run_members}
    slope_selected_count = 0
    solved_levels: list[float] = []
    for option, q in best[3]:
        level = float(best[2]) + q * slope_rise
        solved_levels.append(level)
        piece = option["piece"]
        member = option["member"]
        origin = option["origin"]
        member.model_key = str(piece["model_key"])
        member.piece_id = str(piece["piece_id"])
        member.structural_role = str(option["kind"])
        member.position = (
            float(origin[0]),
            float(origin[1]),
            level + float(piece["base_offset_gu"]) * float(option["scale"]),
        )
        if option["kind"] == "slope":
            slope_selected_count += 1
            member.position = (
                float(origin[0]),
                float(origin[1]),
                level + float(option["slot_deck_offset"])
                - float(option["start_anchor_z"]) * float(option["scale"]),
            )
            member.arc = sum(
                float(run_members[int(option["segment"])][int(option["local_index"]) + offset].arc)
                for offset in range(int(option["consume"]))
            ) / int(option["consume"])
            member.meta.update(
                {
                    "slope_rise_gu": round(float(option["delta"]), 3),
                    "slope_rise_axis": str(piece["rise_axis"]),
                    "height_level_from_gu": round(level, 3),
                    "height_level_to_gu": round(level + float(option["delta"]), 3),
                }
            )
        member.rotz_rad = float(option["rotz"])
        member.scale = float(option["scale"])
        member.meta.update(
            {
                "terrain_optimized": True,
                "continuous_ring_level_gu": round(level, 3),
            }
        )
        placed.append(member)
        solved_runs[int(option["segment"])].append(member)
    run_members = solved_runs

    # Keep each straight gate neck at the adjacent ring level. The junction
    # tower covers the shared XY endpoint, but it must not conceal an unrelated
    # height step between the main wall and the neck.
    neck_groups: dict[tuple[str, str, int], list[_Placed]] = {}
    for member in placed:
        if member.structural_role != "gate_neck":
            continue
        key = (
            str(member.meta.get("gate_id")),
            str(member.meta.get("gate_side")),
            int(member.meta["segment"]),
        )
        neck_groups.setdefault(key, []).append(member)
    for (_gate_id, side, segment), necks in neck_groups.items():
        adjacent = sorted(run_members.get(segment, []), key=lambda row: float(row.arc))
        if not adjacent:
            raise WallComposeError(
                "gate_junction_infeasible",
                f"gate neck segment {segment} has no adjacent wall run",
            )
        reference = adjacent[-1] if side == "before" else adjacent[0]
        reference_piece = piece_by_id(kit, reference.piece_id)
        assert reference_piece is not None
        if reference.structural_role == "slope":
            level = float(
                reference.meta[
                    "height_level_to_gu" if side == "before" else "height_level_from_gu"
                ]
            )
        else:
            level = (
                float(reference.position[2])
                - float(reference_piece["base_offset_gu"]) * reference.scale
            )
        for neck in necks:
            neck_piece = piece_by_id(kit, neck.piece_id)
            assert neck_piece is not None
            neck.position = (
                neck.position[0],
                neck.position[1],
                level + float(neck_piece["base_offset_gu"]) * neck.scale,
            )
            neck.meta.update(
                {
                    "terrain_optimized": True,
                    "continuous_ring_level_gu": round(level, 3),
                }
            )

    # Materialize additional measured wall tiers only after the base run has
    # been optimized. Tier copies therefore cannot corrupt run sequencing or
    # slope-substitution decisions.
    tiered: list[_Placed] = []
    for pl in placed:
        piece = piece_by_id(kit, pl.piece_id)
        assert piece is not None
        tiers = tier_count_for(pl, piece)
        if tiers <= 1:
            continue
        tier_height = float(piece["tier_height_gu"]) * pl.scale
        for tier in range(1, tiers):
            tiered.append(
                replace(
                    pl,
                    position=(
                        pl.position[0],
                        pl.position[1],
                        pl.position[2] + tier * tier_height,
                    ),
                    meta={**pl.meta, "tier": tier},
                )
            )
    placed.extend(tiered)

    # A footprint-bottom audit does not prove that a replacement slope reaches
    # its neighboring wall. Check the measured end connection points in world
    # XY and fail closed on a real longitudinal seam.
    seam_tolerance = float(rules.get("piece_seam_tolerance_gu", 16.0))

    def connection_points(pl: _Placed, piece: dict) -> list[np.ndarray]:
        rot = engine_transform.tes3_euler_to_matrix(
            [0.0, 0.0, pl.rotz_rad]
        )[:2, :2]
        origin = np.asarray(pl.position[:2], dtype=float)
        return [
            origin + rot @ (np.asarray(endpoint, dtype=float) * pl.scale)
            for endpoint in (piece["end_a_local"], piece["end_b_local"])
        ]

    for segment, members in run_members.items():
        ordered = sorted(members, key=lambda member: float(member.arc))
        for left, right in zip(ordered, ordered[1:]):
            left_piece = piece_by_id(kit, left.piece_id)
            right_piece = piece_by_id(kit, right.piece_id)
            assert left_piece is not None and right_piece is not None
            seam = min(
                float(np.linalg.norm(a - b))
                for a in connection_points(left, left_piece)
                for b in connection_points(right, right_piece)
            )
            allowed_seam = seam_tolerance
            if seam > allowed_seam + 1e-6:
                raise WallComposeError(
                    "wall_seam_infeasible",
                    f"segment {segment} has a {seam:.3f} GU slope/wall end seam",
                )
            def level_in(member: _Placed) -> float:
                if member.structural_role == "slope":
                    return float(member.meta["height_level_from_gu"])
                member_piece = piece_by_id(kit, member.piece_id)
                assert member_piece is not None
                return float(member.position[2]) - float(member_piece["base_offset_gu"]) * member.scale

            def level_out(member: _Placed) -> float:
                if member.structural_role == "slope":
                    return float(member.meta["height_level_to_gu"])
                member_piece = piece_by_id(kit, member.piece_id)
                assert member_piece is not None
                return float(member.position[2]) - float(member_piece["base_offset_gu"]) * member.scale

            level_seam = abs(level_out(left) - level_in(right))
            if level_seam > seam_tolerance:
                raise WallComposeError(
                    "wall_height_seam_infeasible",
                    f"segment {segment} has a {level_seam:.3f} GU slope/wall "
                    "height seam",
                )

    # --- gate assembly ----------------------------------------------------
    gate = kit["gate"]
    gh_piece = piece_by_id(kit, gate["gatehouse_piece"])
    assert gh_piece is not None
    tier_height = float(gh_piece["tier_height_gu"]) or float(gh_piece["height_gu"])
    tier_delta = math.radians(float(gate.get("tier_rotz_delta_deg", 0.0)))
    for a in anchors:
        if a.kind != "gate":
            continue
        point = np.array(a.position_xy) if a.position_xy is not None else path.locate(a.arc)[0]
        gx, gy = float(point[0]), float(point[1])
        yaw = _rotz_for_heading(a.heading_rad, str(gh_piece["long_axis"]))
        crossing_xy = a.meta.get("gate_crossing_xy")
        if not isinstance(crossing_xy, list) or len(crossing_xy) != 2:
            raise WallComposeError("gate_missing", "gate anchor lost its crossing position")
        door_piece = piece_by_id(kit, str(gate.get("door_model", "")))
        door_offset = gate.get("door_offset_local")
        if door_piece is None or not isinstance(door_offset, list) or len(door_offset) != 3:
            raise WallComposeError("gate_missing", "gate passage has no measured door anchor")
        gate_rot = engine_transform.tes3_euler_to_matrix(
            [0.0, 0.0, yaw]
        )[:2, :2]
        passage_center_local = np.asarray(
            [
                float(door_offset[0]),
                float(gate.get("passage_center_local_y_gu", door_offset[1])),
            ],
            dtype=float,
        )
        passage_center = np.asarray([gx, gy], dtype=float) + gate_rot @ passage_center_local
        crossing = np.asarray(crossing_xy, dtype=float)
        alignment_error = float(np.linalg.norm(passage_center - crossing))
        alignment_tolerance = float(gate.get("passage_alignment_tolerance_gu", 1.0))
        if alignment_error > alignment_tolerance:
            raise WallComposeError(
                "gate_passage_alignment_infeasible",
                f"gate {a.meta.get('gate_id')} passage is {alignment_error:.3f} GU "
                "from its road crossing",
            )
        half_width = float(gate.get("landing_half_width_across_road_gu", 512.0))
        half_length = float(gate.get("landing_half_length_along_road_gu", 1024.0))
        landing_spacing = float(gate.get("landing_sample_spacing_gu", 128.0))
        wall_axis = np.asarray([math.cos(yaw), -math.sin(yaw)], dtype=float)
        road_axis = np.asarray([math.sin(yaw), math.cos(yaw)], dtype=float)
        gate_slice = gh_piece["footprint_slice"]
        slice_min = np.asarray(gate_slice["slice_min_xy"], dtype=float)
        slice_max = np.asarray(gate_slice["slice_max_xy"], dtype=float)
        platform_local_corners = [
            np.asarray([x, y], dtype=float)
            for x, y in (
                (slice_min[0], slice_min[1]),
                (slice_max[0], slice_min[1]),
                (slice_max[0], slice_max[1]),
                (slice_min[0], slice_max[1]),
            )
        ]
        platform_world_corners = [
            np.asarray([gx, gy], dtype=float) + gate_rot @ point
            for point in platform_local_corners
        ]
        landing_samples = [
            float(terrain(*(passage_center + wall_axis * across + road_axis * along)))
            for across in np.arange(-half_width, half_width + landing_spacing * 0.5, landing_spacing)
            for along in np.arange(-half_length, half_length + landing_spacing * 0.5, landing_spacing)
        ]
        road_reference_z = max(
            max(landing_samples),
            float(gate.get("minimum_passage_threshold_z_gu", 0.0)),
        )
        gatehouse_bottom_local_z = -float(gh_piece["base_offset_gu"])
        unquantized_bottom_z = (
            road_reference_z
            - float(door_offset[2])
            + gatehouse_bottom_local_z
        )
        gatehouse_bottom_z = 8.0 * math.ceil(
            max(
                unquantized_bottom_z,
                float(gate.get("minimum_passage_threshold_z_gu", 0.0)),
            ) / 8.0
        )
        base_z = gatehouse_bottom_z - gatehouse_bottom_local_z
        for tier in range(int(gate["tier_count"])):
            placed.append(
                _Placed(
                    model_key=str(gh_piece["model_key"]),
                    structural_role="gatehouse",
                    piece_id=str(gh_piece["piece_id"]),
                    position=(gx, gy, base_z + tier * tier_height),
                    rotz_rad=yaw + tier * tier_delta,
                    scale=1.0,
                    arc=a.arc,
                    meta={
                        "tier": tier,
                        "arc": round(float(a.arc), 2),
                        "landing_center_xy_gu": [
                            round(float(passage_center[0]), 3),
                            round(float(passage_center[1]), 3),
                        ],
                        "gatehouse_platform_polygon_xy_gu": [
                            [round(float(point[0]), 3), round(float(point[1]), 3)]
                            for point in platform_world_corners
                        ],
                        "gatehouse_bottom_local_z_gu": round(gatehouse_bottom_local_z, 3),
                        "gatehouse_bottom_z_gu": round(gatehouse_bottom_z, 3),
                        "passage_floor_z_gu": round(gatehouse_bottom_z, 3),
                        "passage_threshold_z_gu": round(gatehouse_bottom_z, 3),
                        "landing_half_length_along_road_gu": half_length,
                        "landing_half_width_across_road_gu": half_width,
                        **a.meta,
                    },
                )
            )
        junction_tower = piece_by_id(
            kit, str(gate.get("junction_tower_piece_id", ""))
        )
        if junction_tower is not None and junction_neck_span > 0.0:
            junction_scale = float(gate.get("junction_tower_scale", 1.0))
            gate_top = base_z + float(gh_piece["height_gu"]) - float(gh_piece["base_offset_gu"])
            tower_top_local = junction_scale * (
                float(junction_tower["height_gu"]) - float(junction_tower["base_offset_gu"])
            )
            tower_center_local = np.asarray(
                [
                    (float(junction_tower["end_a_local"][0]) + float(junction_tower["end_b_local"][0])) / 2.0,
                    (float(junction_tower["end_a_local"][1]) + float(junction_tower["end_b_local"][1])) / 2.0,
                ],
                dtype=float,
            ) * junction_scale
            for side, after in (
                ("before", False),
                ("after", True),
            ):
                center_xy = gate_outer_connection(a, after=after)
                tower_origin = center_xy - gate_rot @ tower_center_local
                placed.append(
                    _Placed(
                        model_key=str(junction_tower["model_key"]),
                        structural_role="tower",
                        piece_id=str(junction_tower["piece_id"]),
                        position=(
                            float(tower_origin[0]),
                            float(tower_origin[1]),
                            gate_top - tower_top_local,
                        ),
                        rotz_rad=yaw,
                        scale=junction_scale,
                        arc=a.arc,
                        meta={
                            "gate_side": side,
                            "gate_junction": True,
                            "gate_id": a.gate.get("gate_id") if a.gate else None,
                            "arc": round(float(a.arc), 2),
                        },
                    )
                )
        for role_key, offset_key in (("door_model", "door_offset_local"), ("frame_model", "frame_offset_local")):
            pid = gate.get(role_key)
            off = gate.get(offset_key)
            if not pid or not off:
                continue
            piece = piece_by_id(kit, pid)
            if piece is None or len(off) != 3:
                continue
            lx, ly, lz = (float(v) for v in off)
            wx, wy = np.asarray([gx, gy], dtype=float) + gate_rot @ np.asarray(
                [lx, ly], dtype=float
            )
            placed.append(
                _Placed(
                    model_key=str(piece["model_key"]),
                    structural_role="door",
                    piece_id=str(piece["piece_id"]),
                    position=(wx, wy, base_z + lz),
                    rotz_rad=yaw,
                    scale=1.0,
                    arc=a.arc,
                    is_door=True,
                    meta={
                        "gate_member": role_key,
                        "gate_id": a.gate.get("gate_id") if a.gate else None,
                        "arc": round(float(a.arc), 2),
                    },
                )
            )

    # Keep elevated wall/tower tops aligned without leaving an exposed lower
    # edge. The source castle uses the same piece flipped beneath the primary
    # mesh. Its pivot is displaced by twice the measured base offset, so the
    # flipped mesh meets the primary mesh exactly at the primary bottom.
    underlays: list[_Placed] = []
    for primary in placed:
        if primary.structural_role not in {"straight", "gate_neck", "tower"}:
            continue
        piece = piece_by_id(kit, primary.piece_id)
        assert piece is not None
        terrain_min, _terrain_max = footprint_ground_range(
            piece,
            (primary.position[0], primary.position[1]),
            primary.rotz_rad,
            primary.scale,
        )
        primary_bottom = (
            float(primary.position[2])
            - float(piece["base_offset_gu"]) * primary.scale
        )
        minimum_burial_gu = (
            minimum_coverage * float(piece["height_gu"]) * primary.scale
        )
        foundation_gap = primary_bottom + minimum_burial_gu - terrain_min
        # Downstream building/road grading can lower terrain after composition.
        # Materialize the measured underlay for every ordinary wall and tower;
        # when it is not needed it remains completely underground, while any
        # later lowering reveals continuous masonry instead of an open seam.
        cross_center = 0.5 * (
            float(piece["end_a_local"][1]) + float(piece["end_b_local"][1])
        ) if str(piece["long_axis"]) == "x" else 0.5 * (
            float(piece["end_a_local"][0]) + float(piece["end_b_local"][0])
        )
        compensation_local = (
            np.asarray([0.0, 2.0 * cross_center], dtype=float)
            if str(piece["long_axis"]) == "x"
            else np.asarray([2.0 * cross_center, 0.0], dtype=float)
        )
        yaw_rot = engine_transform.tes3_euler_to_matrix(
            [0.0, 0.0, primary.rotz_rad]
        )[:2, :2]
        compensation_world = yaw_rot @ (compensation_local * primary.scale)
        primary.meta["foundation_underlay"] = True
        primary.meta["source_foundation_gap_gu"] = round(foundation_gap, 3)
        underlays.append(
            replace(
                primary,
                structural_role="foundation_underlay",
                position=(
                    primary.position[0] + float(compensation_world[0]),
                    primary.position[1] + float(compensation_world[1]),
                    primary.position[2]
                    - 2.0 * float(piece["base_offset_gu"]) * primary.scale,
                ),
                rotx_rad=math.pi,
                arc=None,
                meta={
                    "foundation_underlay_for_role": primary.structural_role,
                    "foundation_underlay_for_piece": primary.piece_id,
                    "source_foundation_gap_gu": round(foundation_gap, 3),
                },
            )
        )
    placed.extend(underlays)

    # --- output doc --------------------------------------------------------
    origin = (float(path.points[0][0]), float(path.points[0][1]))
    members = []
    boxes_xy: list[list[float]] = []
    z_min, z_max = math.inf, -math.inf
    for index, pl in enumerate(placed):
        piece = piece_by_id(kit, pl.piece_id)
        assert piece is not None
        # GU-space placement: world = pos + Rz(-rotz)·(scale·local)
        rot = engine_transform.tes3_euler_to_matrix(
            [pl.rotx_rad, 0.0, pl.rotz_rad]
        )
        pos = np.array(pl.position, dtype=float)
        # Local body box: long-axis extent from the slice end points, cross
        # extent centered on the end points' cross coordinate.
        ax, ay = float(piece["end_a_local"][0]), float(piece["end_a_local"][1])
        bx, by = float(piece["end_b_local"][0]), float(piece["end_b_local"][1])
        half_t = 0.5 * float(piece["thickness_gu"]) * pl.scale
        if str(piece["long_axis"]) == "x":
            lo = np.array([min(ax, bx), ay - half_t, -float(piece["base_offset_gu"]) * pl.scale])
            hi = np.array([max(ax, bx), ay + half_t, float(piece["height_gu"]) - float(piece["base_offset_gu"])])
        else:
            lo = np.array([ax - half_t, min(ay, by), -float(piece["base_offset_gu"]) * pl.scale])
            hi = np.array([ax + half_t, max(ay, by), float(piece["height_gu"]) - float(piece["base_offset_gu"])])
        for cx in (lo[0], hi[0]):
            for cy in (lo[1], hi[1]):
                for cz in (lo[2], hi[2]):
                    world = pos + rot @ (np.array([cx, cy, cz]) * pl.scale)
                    z_min = min(z_min, float(world[2]))
                    z_max = max(z_max, float(world[2]))
        footprint_xy_rel = []
        for lx, ly in _slice_outline(piece):
            world = pos + rot @ (np.array([lx, ly, 0.0]) * pl.scale)
            footprint_xy_rel.append([
                round(float(world[0] - origin[0]), 3),
                round(float(world[1] - origin[1]), 3),
            ])
        boxes_xy.extend(footprint_xy_rel)
        if pl.structural_role in {"straight", "slope", "gate_neck"}:
            end_world = [
                pos + rot @ (
                    np.array([float(endpoint[0]), float(endpoint[1]), 0.0])
                    * pl.scale
                )
                for endpoint in (piece["end_a_local"], piece["end_b_local"])
            ]
            if pl.structural_role == "slope":
                level_a = float(pl.meta["height_level_from_gu"])
                level_b = float(pl.meta["height_level_to_gu"])
                minimum_gu = (
                    minimum_coverage
                    * float(transition_wall["height_gu"])
                    * pl.scale
                )
                maximum_gu = (
                    maximum_coverage
                    * float(transition_wall["height_gu"])
                    * pl.scale
                )
                minimum_ground_a = level_a + minimum_gu
                minimum_ground_b = level_b + minimum_gu
                maximum_ground_a = level_a + maximum_gu
                maximum_ground_b = level_b + maximum_gu
                ground_a = minimum_ground_a
                ground_b = minimum_ground_b
            else:
                base_level = (
                    float(pl.position[2])
                    - float(piece["base_offset_gu"]) * pl.scale
                )
                minimum_gu = (
                    minimum_coverage * float(piece["height_gu"]) * pl.scale
                )
                maximum_gu = (
                    maximum_coverage * float(piece["height_gu"]) * pl.scale
                )
                if pl.meta.get("foundation_underlay"):
                    underlay_bottom = (
                        base_level - float(piece["height_gu"]) * pl.scale
                    )
                    minimum_ground_a = minimum_ground_b = underlay_bottom + minimum_gu
                    maximum_ground_a = maximum_ground_b = base_level + maximum_gu
                    source_a = float(
                        terrain(float(end_world[0][0]), float(end_world[0][1]))
                    )
                    source_b = float(
                        terrain(float(end_world[1][0]), float(end_world[1][1]))
                    )
                    ground_a = min(max(source_a, minimum_ground_a), maximum_ground_a)
                    ground_b = min(max(source_b, minimum_ground_b), maximum_ground_b)
                else:
                    minimum_ground_a = minimum_ground_b = base_level + minimum_gu
                    maximum_ground_a = maximum_ground_b = base_level + maximum_gu
                    ground_a = ground_b = minimum_ground_a
            pl.meta["terrain_grade_profile"] = {
                "end_a_xy": [round(float(end_world[0][0]), 3), round(float(end_world[0][1]), 3)],
                "end_b_xy": [round(float(end_world[1][0]), 3), round(float(end_world[1][1]), 3)],
                "ground_z_end_a_gu": round(ground_a, 3),
                "ground_z_end_b_gu": round(ground_b, 3),
                "minimum_ground_z_end_a_gu": round(minimum_ground_a, 3),
                "minimum_ground_z_end_b_gu": round(minimum_ground_b, 3),
                "maximum_ground_z_end_a_gu": round(maximum_ground_a, 3),
                "maximum_ground_z_end_b_gu": round(maximum_ground_b, 3),
                "burial_fraction": (
                    doubled_coverage
                    if tier_count_for(pl, piece) > 1
                    else regular_coverage
                ),
            }
        if pl.structural_role == "straight" and int(pl.meta.get("tier", 0)) == 0:
            ground = footprint_ground(
                piece,
                (pl.position[0], pl.position[1]),
                pl.rotz_rad,
                pl.scale,
            )
            bottom = float(pl.position[2]) - float(piece["base_offset_gu"]) * pl.scale
            coverage_gu = ground - bottom
            tiers = tier_count_for(pl, piece)
            required_gu = required_coverage(piece, pl.scale, tiers)
            coverage_fraction = required_gu / max(
                effective_height(piece, pl.scale, tiers), 1e-9
            )
            pl.meta.update(
                {
                    "source_bottom_coverage_gu": round(coverage_gu, 3),
                    "bottom_coverage_gu": round(required_gu, 3),
                    "bottom_coverage_fraction": round(coverage_fraction, 6),
                    "required_bottom_coverage_gu": round(required_gu, 3),
                    "wall_tier_count": tiers,
                }
            )
        members.append(
            {
                "source_id": f"{stamp_id}_m{index:04d}",
                "model_key": pl.model_key,
                "structural_role": pl.structural_role,
                "is_door": pl.is_door,
                "category": "exterior",
                "record_type": "STAT",
                "object_id": None,
                "offset_gu": [
                    round(pl.position[0] - origin[0], 3),
                    round(pl.position[1] - origin[1], 3),
                    round(pl.position[2], 3),
                ],
                "rotation": [
                    round(float(pl.rotx_rad), 9),
                    0.0,
                    round(float(pl.rotz_rad), 9),
                ],
                "scale": pl.scale,
                "piece_id": pl.piece_id,
                "footprint_xy_rel": footprint_xy_rel,
                "meta": {**pl.meta, "arc": None if pl.arc is None else round(float(pl.arc), 2)},
            }
        )
    arr = np.array(boxes_xy)
    aabb_min = arr.min(axis=0)
    aabb_max = arr.max(axis=0)
    hull = _convex_hull(arr)
    doc = {
        "stamp_id": stamp_id,
        "building_type": "city_wall",
        "kit_id": kit["kit_id"],
        "coordinates": "input_path_frame_gu_xy_rel_to_origin_gu_z_absolute",
        "origin_gu": [round(origin[0], 3), round(origin[1], 3)],
        "members": members,
        "bounds_rel_gu": {
            "min": [round(float(aabb_min[0]), 3), round(float(aabb_min[1]), 3), round(z_min, 3)],
            "max": [round(float(aabb_max[0]), 3), round(float(aabb_max[1]), 3), round(z_max, 3)],
            "span": [
                round(float(aabb_max[0] - aabb_min[0]), 3),
                round(float(aabb_max[1] - aabb_min[1]), 3),
                round(z_max - z_min, 3),
            ],
        },
        "footprint": {
            "aabb_rel": {
                "min": [round(float(aabb_min[0]), 3), round(float(aabb_min[1]), 3)],
                "max": [round(float(aabb_max[0]), 3), round(float(aabb_max[1]), 3)],
            },
            "hull_xy_rel": [[round(float(x), 3), round(float(y), 3)] for x, y in hull],
        },
        "provenance": {
            "gate_count": sum(1 for a in anchors if a.kind == "gate"),
            "corner_count": sum(1 for a in anchors if a.kind == "corner"),
            "tower_anchor_count": sum(
                member["structural_role"] == "tower" for member in members
            ),
            "step_insert_count": 0,
            "slope_insert_count": sum(
                member["structural_role"] == "slope" for member in members
            ),
            "slope_candidate_count": slope_candidate_count,
            "slope_selected_count": slope_selected_count,
            "member_count": len(members),
            "path_length_gu": round(path.total_length, 3),
            "regular_wall_bottom_coverage_fraction": regular_coverage,
            "doubled_wall_bottom_coverage_fraction": doubled_coverage,
            "minimum_wall_bottom_coverage_fraction": minimum_coverage,
            "maximum_wall_bottom_coverage_fraction": maximum_coverage,
            "foundation_underlay_count": len(underlays),
            "wall_tier_count": wall_tier_count,
            "tower_piece_id": str(tower_piece["piece_id"]),
            "tower_height_changes_enabled": tower_height_changes_enabled,
            "height_transition_priority": "authored_slopes_first",
        },
    }
    return doc


def _modal_piece_id(kit: dict, role: str) -> str | None:
    pieces = [p for p in kit["pieces"] if p["role"] == role]
    if not pieces:
        return None
    return str(max(pieces, key=lambda p: (int(p.get("weight", 1)), str(p["piece_id"])))["piece_id"])


def _modal_piece(kit: dict, role: str):
    pid = _modal_piece_id(kit, role)
    return piece_by_id(kit, pid) if pid else None


def _tower_piece(kit: dict):
    preferred = kit.get("rules", {}).get("tower_piece_id")
    if preferred:
        piece = piece_by_id(kit, str(preferred))
        if piece is None or piece.get("role") != "tower":
            raise WallComposeError(
                "tower_piece_infeasible",
                f"configured tower_piece_id {preferred!r} is not a tower in the kit",
            )
        return piece
    return _modal_piece(kit, "tower")


def _convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    """Andrew monotone chain; returns CCW hull without repeating the start."""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
