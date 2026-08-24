"""Wall kit JSON schema and IO for the city-wall composer (stages W1/W2).

Pipeline position
-----------------
Stage W1 (``tools/wall_kit_extract.py``) authors a wall-kit JSON from source
placements; stage W2 (``src/procgen/wall_compose.py`` /
``tools/compose_city_wall.py``) consumes it. This module is the single owner
of that file's schema: loading is fail-closed (missing/invalid required keys
raise ``WallKitError``), so a malformed kit can never silently reach
composition.

Kit document shape (all tunables live in the JSON, none in code)::

    {
      "kit_id": str,
      "pieces": [{
         "piece_id": str,            # unique within kit
         "model_key": str,           "role": straight|corner|tower|gatehouse|
                                           stair|slope|end|trim|door|doorframe,
         "length_gu": float,         # slice-measured long-axis extent at scale 1
         "thickness_gu": float,      "height_gu": float,
         "base_offset_gu": float,    # pivot height above the piece's lowest point
         "tier_height_gu": float,    # mined vertical stacking step (0 = never stacks)
         "footprint_slice": {...},   # slice min/max/percentiles (provenance)
         "end_a_local": [x, y],      "end_b_local": [x, y],   # pivot-relative 2D
         "long_axis": "x"|"y",       "square": bool,
         "stackable": bool,          "allow_scaled_fill": bool,
         "scale_range": [lo, hi],    "weight": int          # usage count
      }],
      "gate": {"gatehouse_piece": str, "tier_count": int,
               "tier_rotz_delta_deg": float,          # authored per-extra-tier yaw step
               "door_model": str|null, "frame_model": str|null,
               "door_offset_local": [x, y, z]|null,   # vs base-tier pivot, gh local frame
               "frame_offset_local": [x, y, z]|null},
      "rules": {"corner_angle_threshold_deg", "max_step_down_gu",
                "tower_spacing_gu", "stair_at_step_gu",
                "anchor_inset_tolerance_gu", "fill_quantum_gu",
                "max_fill_pieces", "burial_depth_gu",
                 "min_tower_separation_gu",
                 "anchor_mesh_overlap_gu",
                 "allowed_fill_piece_ids" (optional)},
      "stair_assembly": {                       # optional measured deck transition
        "wall_piece_id": str,
        "stair_piece_id": str,
        "stair_count": int,
        "target_lateral_bounds_gu": [float, float],
        "transition_span_gu": float,
        "minimum_deck_overlap_gu": float
      },
      "slope_assembly": {                       # optional authored terrain ramp
        "wall_piece_id": str,
        "slope_piece_id": str,
        "minimum_deck_overlap_gu": float
      },
      "provenance": {...}
    }

Units are TES3 game units (GU) throughout; angles in degrees in the JSON and
radians inside the composer.
"""

from __future__ import annotations

import json
from pathlib import Path

ROLES = (
    "straight",
    "corner",
    "tower",
    "gatehouse",
    "stair",
    "slope",
    "end",
    "trim",
    "door",
    "doorframe",
)

_PIECE_REQUIRED = (
    "piece_id",
    "model_key",
    "role",
    "length_gu",
    "thickness_gu",
    "height_gu",
    "end_a_local",
    "end_b_local",
    "long_axis",
)
_RULES_REQUIRED = (
    "corner_angle_threshold_deg",
    "corner_max_turn_deg",
    "max_step_down_gu",
    "anchor_inset_tolerance_gu",
    "fill_quantum_gu",
    "max_fill_pieces",
    "anchor_mesh_overlap_gu",
)


class WallKitError(ValueError):
    """Raised when a wall-kit JSON is missing or has invalid required data."""


def validate_kit(kit: dict) -> None:
    """Fail-closed validation of a loaded kit document."""
    if not isinstance(kit, dict):
        raise WallKitError("kit document must be an object")
    for key in ("kit_id", "pieces", "gate", "rules"):
        if key not in kit:
            raise WallKitError(f"kit missing required key {key!r}")
    pieces = kit["pieces"]
    if not isinstance(pieces, list) or not pieces:
        raise WallKitError("kit 'pieces' must be a non-empty list")
    seen: set[str] = set()
    for piece in pieces:
        if not isinstance(piece, dict):
            raise WallKitError("each piece must be an object")
        for key in _PIECE_REQUIRED:
            if key not in piece:
                raise WallKitError(f"piece missing required key {key!r}: {piece}")
        if piece["role"] not in ROLES:
            raise WallKitError(f"piece {piece['piece_id']!r} unknown role {piece['role']!r}")
        if piece["piece_id"] in seen:
            raise WallKitError(f"duplicate piece_id {piece['piece_id']!r}")
        seen.add(str(piece["piece_id"]))
        if float(piece["length_gu"]) <= 0.0:
            raise WallKitError(f"piece {piece['piece_id']!r} length_gu must be > 0")
        if piece["long_axis"] not in ("x", "y"):
            raise WallKitError(f"piece {piece['piece_id']!r} long_axis must be 'x' or 'y'")
        for end in ("end_a_local", "end_b_local"):
            if len(piece[end]) != 2:
                raise WallKitError(f"piece {piece['piece_id']!r} {end} must be [x, y]")
    gate = kit["gate"]
    if not isinstance(gate, dict) or "gatehouse_piece" not in gate or "tier_count" not in gate:
        raise WallKitError("kit 'gate' must be an object with gatehouse_piece and tier_count")
    if gate["gatehouse_piece"] is not None and gate["gatehouse_piece"] not in seen:
        raise WallKitError(f"gate.gatehouse_piece {gate['gatehouse_piece']!r} not in pieces")
    if float(gate.get("passage_burial_gu", 0.0)) < 0.0:
        raise WallKitError("gate.passage_burial_gu must be nonnegative")
    if not isinstance(gate.get("passage_center_local_y_gu", 0.0), (int, float)):
        raise WallKitError("gate.passage_center_local_y_gu must be numeric")
    if float(gate.get("passage_alignment_tolerance_gu", 0.0)) < 0.0:
        raise WallKitError("gate.passage_alignment_tolerance_gu must be nonnegative")
    for key in (
        "landing_half_length_along_road_gu",
        "landing_half_width_across_road_gu",
        "landing_sample_spacing_gu",
    ):
        if float(gate.get(key, 0.0)) <= 0.0:
            raise WallKitError(f"gate.{key} must be positive")
    junction_tower_id = gate.get("junction_tower_piece_id")
    junction_neck_id = gate.get("junction_neck_piece_id")
    junction_count = int(gate.get("junction_neck_piece_count", 0))
    if junction_tower_id is not None or junction_neck_id is not None or junction_count:
        junction_tower = piece_by_id(kit, str(junction_tower_id))
        junction_neck = piece_by_id(kit, str(junction_neck_id))
        if junction_tower is None or junction_tower.get("role") != "tower":
            raise WallKitError("gate.junction_tower_piece_id must name a tower piece")
        if junction_neck is None or junction_neck.get("role") != "straight":
            raise WallKitError("gate.junction_neck_piece_id must name a straight piece")
        if junction_count < 1:
            raise WallKitError("gate.junction_neck_piece_count must be positive")
        junction_scale = float(gate.get("junction_tower_scale", 1.0))
        scale_min, scale_max = (float(value) for value in junction_tower["scale_range"])
        if not (scale_min <= junction_scale <= scale_max):
            raise WallKitError("gate.junction_tower_scale is outside the tower scale_range")
    rules = kit["rules"]
    if not isinstance(rules, dict):
        raise WallKitError("kit 'rules' must be an object")
    for key in _RULES_REQUIRED:
        if key not in rules:
            raise WallKitError(f"kit 'rules' missing {key!r}")
    allowed = rules.get("allowed_fill_piece_ids")
    if allowed is not None:
        if not isinstance(allowed, list) or not allowed:
            raise WallKitError("kit rules allowed_fill_piece_ids must be a non-empty list")
        role_by_id = {str(piece["piece_id"]): piece["role"] for piece in pieces}
        missing = [str(pid) for pid in allowed if str(pid) not in role_by_id]
        if missing:
            raise WallKitError(f"kit rules allowed_fill_piece_ids missing pieces: {missing}")
        non_straight = [str(pid) for pid in allowed if role_by_id[str(pid)] != "straight"]
        if non_straight:
            raise WallKitError(
                f"kit rules allowed_fill_piece_ids must name straight pieces: {non_straight}"
            )
    if str(rules.get("path_winding", "preserve")).lower() not in {
        "preserve", "clockwise", "counterclockwise"
    }:
        raise WallKitError("kit rules path_winding must be preserve/clockwise/counterclockwise")
    tower_scale = float(rules.get("tower_scale", 1.0))
    if tower_scale <= 0.0:
        raise WallKitError("kit rules tower_scale must be positive")
    tower_piece_id = rules.get("tower_piece_id")
    tower_piece = piece_by_id(kit, str(tower_piece_id)) if tower_piece_id else None
    if tower_piece is not None:
        scale_min, scale_max = (float(value) for value in tower_piece["scale_range"])
        if not (scale_min <= tower_scale <= scale_max):
            raise WallKitError(
                f"kit rules tower_scale {tower_scale} is outside {tower_piece_id} scale_range"
            )
    if float(rules.get("slope_complexity_penalty_gu", 0.0)) < 0.0:
        raise WallKitError("kit rules slope_complexity_penalty_gu must be nonnegative")
    if float(rules.get("max_wall_foundation_gap_gu", 0.0)) < 0.0:
        raise WallKitError("kit rules max_wall_foundation_gap_gu must be nonnegative")
    if not isinstance(
        rules.get("minimum_foundation_ground_z_gu", 0.0), (int, float)
    ):
        raise WallKitError("kit rules minimum_foundation_ground_z_gu must be numeric")
    if float(rules.get("gate_tower_clearance_gu", 0.0)) < 0.0:
        raise WallKitError("kit rules gate_tower_clearance_gu must be nonnegative")
    minimum_coverage = float(
        rules.get("minimum_wall_bottom_coverage_fraction", 0.0)
    )
    maximum_coverage = float(
        rules.get("maximum_wall_bottom_coverage_fraction", 1.0)
    )
    if not 0.0 <= minimum_coverage <= maximum_coverage <= 1.0:
        raise WallKitError(
            "wall bottom coverage fractions must satisfy 0 <= minimum <= maximum <= 1"
        )
    assembly = kit.get("stair_assembly")
    if assembly is not None:
        if not isinstance(assembly, dict):
            raise WallKitError("kit stair_assembly must be an object")
        required = (
            "wall_piece_id",
            "stair_piece_id",
            "stair_count",
            "target_lateral_bounds_gu",
            "transition_span_gu",
            "minimum_deck_overlap_gu",
        )
        missing = [key for key in required if key not in assembly]
        if missing:
            raise WallKitError(f"kit stair_assembly missing keys: {missing}")
        stair = piece_by_id(kit, str(assembly["stair_piece_id"]))
        wall = piece_by_id(kit, str(assembly["wall_piece_id"]))
        if stair is None or stair.get("role") != "stair":
            raise WallKitError("stair_assembly.stair_piece_id must name a stair piece")
        if wall is None or wall.get("role") != "straight":
            raise WallKitError("stair_assembly.wall_piece_id must name a straight piece")
        if not isinstance(stair.get("walk_surface"), dict):
            raise WallKitError("stair assembly piece is missing walk_surface anchors")
        if not isinstance(wall.get("walk_surface"), dict):
            raise WallKitError("stair assembly wall is missing walk_surface anchors")
        for key in ("entry_local_gu", "exit_local_gu"):
            if len(stair["walk_surface"].get(key, [])) != 3:
                raise WallKitError(f"stair walk_surface.{key} must contain three values")
        for key in ("end_a_local_gu", "end_b_local_gu"):
            if len(wall["walk_surface"].get(key, [])) != 3:
                raise WallKitError(f"wall walk_surface.{key} must contain three values")
        for owner, surface in (("stair", stair["walk_surface"]), ("wall", wall["walk_surface"])):
            bounds = surface.get("lateral_bounds_gu")
            if not isinstance(bounds, list) or len(bounds) != 2 or float(bounds[1]) <= float(bounds[0]):
                raise WallKitError(f"{owner} walk_surface.lateral_bounds_gu must be increasing")
        if not isinstance(assembly["stair_count"], int) or assembly["stair_count"] < 1:
            raise WallKitError("stair_assembly.stair_count must be a positive integer")
        target_bounds = assembly["target_lateral_bounds_gu"]
        if (not isinstance(target_bounds, list) or len(target_bounds) != 2
                or float(target_bounds[1]) <= float(target_bounds[0])):
            raise WallKitError("stair_assembly.target_lateral_bounds_gu must be increasing")
        if float(assembly["transition_span_gu"]) <= 0.0:
            raise WallKitError("stair_assembly.transition_span_gu must be positive")
        if float(assembly["minimum_deck_overlap_gu"]) < 0.0:
            raise WallKitError("stair_assembly.minimum_deck_overlap_gu must be nonnegative")
    slope_assembly = kit.get("slope_assembly")
    if slope_assembly is not None:
        if not isinstance(slope_assembly, dict):
            raise WallKitError("kit slope_assembly must be an object")
        required = ("wall_piece_id", "slope_piece_id", "minimum_deck_overlap_gu")
        missing = [key for key in required if key not in slope_assembly]
        if missing:
            raise WallKitError(f"kit slope_assembly missing keys: {missing}")
        slope = piece_by_id(kit, str(slope_assembly["slope_piece_id"]))
        reverse_slope_id = slope_assembly.get("reverse_slope_piece_id")
        reverse_slope = (
            piece_by_id(kit, str(reverse_slope_id)) if reverse_slope_id is not None else None
        )
        wall = piece_by_id(kit, str(slope_assembly["wall_piece_id"]))
        if slope is None or slope.get("role") != "slope":
            raise WallKitError("slope_assembly.slope_piece_id must name a slope piece")
        if wall is None or wall.get("role") != "straight":
            raise WallKitError("slope_assembly.wall_piece_id must name a straight piece")
        if reverse_slope_id is not None and (
            reverse_slope is None or reverse_slope.get("role") != "slope"
        ):
            raise WallKitError(
                "slope_assembly.reverse_slope_piece_id must name a slope piece"
            )
        surface = slope.get("walk_surface")
        if not isinstance(surface, dict):
            raise WallKitError("slope assembly piece is missing walk_surface anchors")
        for key in ("entry_local_gu", "exit_local_gu"):
            if len(surface.get(key, [])) != 3:
                raise WallKitError(f"slope walk_surface.{key} must contain three values")
        if float(surface.get("rise_gu", 0.0)) <= 0.0:
            raise WallKitError("slope walk_surface.rise_gu must be positive")
        if float(slope_assembly["minimum_deck_overlap_gu"]) < 0.0:
            raise WallKitError("slope_assembly.minimum_deck_overlap_gu must be nonnegative")


def load_kit(path: str | Path) -> dict:
    """Load and validate a wall-kit JSON. Raises WallKitError on any defect."""
    path = Path(path)
    try:
        kit = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WallKitError(f"cannot read kit {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise WallKitError(f"kit {path} is not valid JSON: {exc}") from exc
    validate_kit(kit)
    return kit


def save_kit(kit: dict, path: str | Path) -> None:
    """Validate then write a wall-kit JSON with stable formatting."""
    validate_kit(kit)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pieces_with_role(kit: dict, role: str) -> list[dict]:
    return [p for p in kit["pieces"] if p["role"] == role]


def piece_by_id(kit: dict, piece_id: str) -> dict | None:
    for piece in kit["pieces"]:
        if piece["piece_id"] == piece_id:
            return piece
    return None


def fill_candidates(kit: dict) -> list[dict]:
    """Straight pieces usable as segment fill, longest first.

    Includes scaled variants of pieces flagged ``allow_scaled_fill``; each
    candidate carries ``scale`` (1.0 unless a scaled filler) and its effective
    ``fill_length_gu``.
    """
    candidates: list[dict] = []
    allowed_ids = kit.get("rules", {}).get("allowed_fill_piece_ids")
    allowed = {str(pid) for pid in allowed_ids} if allowed_ids is not None else None
    for piece in pieces_with_role(kit, "straight"):
        if allowed is not None and str(piece["piece_id"]) not in allowed:
            continue
        weight = int(piece.get("weight", 1))
        lo, hi = piece.get("scale_range", [1.0, 1.0])
        base = float(piece["length_gu"])
        if piece.get("allow_scaled_fill") and lo < hi:
            # Scaled filler: use the smallest observed scale as the filler
            # variant (source pattern, e.g. wl_01 @ 0.75).
            scale = round(float(lo), 4)
            candidates.append(
                {
                    "piece": piece,
                    "scale": scale,
                    "fill_length_gu": base * scale,
                    "weight": weight,
                }
            )
        candidates.append(
            {"piece": piece, "scale": 1.0, "fill_length_gu": base, "weight": weight}
        )
    candidates.sort(key=lambda c: (-c["weight"], -c["fill_length_gu"], str(c["piece"]["piece_id"])))
    return candidates
