"""Kit-general exterior house grammar: mine, validate, and generate stamp-shaped buildings."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import engine_transform

SCHEMA_VERSION = 1
__version__ = "0.3.0"

POSITION_CLUSTER_GU = 48.0
DOOR_ASSOC_GU = 120.0
STAIR_ASSOC_GU = 600.0
APPROACH_BLOCK_ASSOC_GU = 900.0
DOOR_CLEARANCE_XY_GU = 160.0
ELEVATION_STAIR_THRESHOLD_GU = 40.0
DEFAULT_STAIR_APPROACH_GU = 180.0
DEFAULT_DOUBLE_STAIR_LATERAL_GU = 118.0
DOOR_OUTWARD_NUDGE_GU = 24.0

# Compatibility default for grammar documents authored before shell prefixes
# became explicit JSON provenance.
DEFAULT_HOUSE_SHELL_PREFIXES = (
    "sky_ex_mk_h_",
    "sky_ex_farm_h",
    "sky_ex_rm_h",
    "sky_ex_mk_tv",
    "sky_ex_mk_h_gl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def classify_attachment(member: Mapping[str, Any]) -> str:
    if member.get("is_door") or member.get("record_type") == "DOOR":
        return "door"
    oid = (member.get("object_id") or "").lower()
    mk = (member.get("model_key") or "").lower()
    if member.get("structural_role") == "shell":
        return "shell"
    if "doorframe" in oid or "doorf" in oid or "_df_" in mk:
        return "doorframe"
    if "stair" in oid or "stair" in mk or "_str_" in mk:
        return "stair"
    if "window" in oid or "_wg_" in mk or "_ww_" in mk:
        return "window"
    if "chimney" in oid or "_cm_" in mk:
        return "chimney"
    if "dormer" in oid or "dormer" in mk:
        return "dormer"
    if "porch" in oid or "porch" in mk:
        return "porch"
    if "tent" in oid or "tent" in mk:
        return "tent"
    if "wall" in oid or "_wl_" in mk or "stnwl" in mk or "stonewall" in oid:
        return "wall"
    if "fence" in oid or "ffence" in mk:
        return "fence"
    return "decoration"


def is_house_shell_id(
    shell_id: str, prefixes: Sequence[str] | None = None
) -> bool:
    wanted = prefixes if prefixes is not None else DEFAULT_HOUSE_SHELL_PREFIXES
    normalized = str(shell_id).replace("/", "\\").casefold()
    return any(
        normalized.startswith(str(prefix).replace("/", "\\").casefold())
        for prefix in wanted
    )


def shell_key(member: Mapping[str, Any]) -> str:
    mk = (member.get("model_key") or "").replace("\\", "/")
    base = mk.rsplit("/", 1)[-1]
    return re.sub(r"\.nif$", "", base, flags=re.IGNORECASE)


def rotz_of(member: Mapping[str, Any]) -> float:
    rotation = member.get("rotation") or [0.0, 0.0, 0.0]
    return float(rotation[2])


def rotate_xy(dx: float, dy: float, angle_rad: float) -> tuple[float, float]:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return c * dx - s * dy, s * dx + c * dy


def to_shell_local(
    offset_gu: Sequence[float],
    rotation: Sequence[float],
    shell_offset: Sequence[float],
    shell_rotz: float,
) -> tuple[list[float], list[float]]:
    dx = float(offset_gu[0]) - float(shell_offset[0])
    dy = float(offset_gu[1]) - float(shell_offset[1])
    dz = float(offset_gu[2]) - float(shell_offset[2])
    # Engine placement uses world = shell_pos + Rz(-shell_rotz) * local.
    # Recovering local coordinates therefore uses the inverse Rz(+shell_rotz).
    lx, ly = rotate_xy(dx, dy, shell_rotz)
    local_rotz = float(rotation[2]) - shell_rotz
    return [lx, ly, dz], [float(rotation[0]), float(rotation[1]), local_rotz]


def offset_rel_door(member: Mapping[str, Any], door: Mapping[str, Any]) -> list[float]:
    return [
        float(member["offset_gu"][0]) - float(door["offset_gu"][0]),
        float(member["offset_gu"][1]) - float(door["offset_gu"][1]),
        float(member["offset_gu"][2]) - float(door["offset_gu"][2]),
    ]


def approach_unit(outward_heading_deg: float) -> tuple[float, float]:
    rad = math.radians(float(outward_heading_deg))
    return math.sin(rad), math.cos(rad)


def lateral_unit(outward_heading_deg: float) -> tuple[float, float]:
    ax, ay = approach_unit(outward_heading_deg)
    return -ay, ax


def access_rotz_from_outward(outward_heading_deg: float) -> float:
    """Map plan outward heading (deg) to Markarth access-mesh yaw (rad)."""
    return math.radians((270.0 - float(outward_heading_deg)) % 360.0)


def door_local_to_stamp(local_x: float, local_y: float, door_rotz: float) -> tuple[float, float]:
    """Door-local XY to stamp-relative XY. Local -Y is outward, local +X is lateral right."""
    return rotate_xy(local_x, local_y, door_rotz)


def stamp_to_door_local(stamp_x: float, stamp_y: float, door_rotz: float) -> tuple[float, float]:
    return rotate_xy(stamp_x, stamp_y, -door_rotz)


def _normalize_rotz_delta(delta: float) -> float:
    return (float(delta) + math.pi) % (2.0 * math.pi) - math.pi


def _stair_rotz_for_door(stair_rotz_ref: float, ref_door_rotz: float, door_rotz: float) -> float:
    """Align stair yaw with door unless the bundle uses a deliberate perpendicular offset."""
    delta = _normalize_rotz_delta(stair_rotz_ref - ref_door_rotz)
    if abs(delta) < 0.05:
        return float(door_rotz)
    return float(door_rotz) + delta


_A2_MODEL_BOUNDS_BY_KEY: dict[str, dict[str, list[float]]] | None = None


def _lookup_model_local_bounds(model_key: str | None) -> dict[str, list[float]] | None:
    global _A2_MODEL_BOUNDS_BY_KEY
    if not model_key:
        return None
    if _A2_MODEL_BOUNDS_BY_KEY is None:
        index: dict[str, dict[str, list[float]]] = {}
        a2_dir = Path(__file__).resolve().parents[2] / "output" / "skyrim-settlements" / "markarth-side-v1" / "a2"
        if a2_dir.is_dir():
            for doc_path in sorted(a2_dir.glob("nif_*.json")):
                try:
                    doc = json.loads(doc_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                mk = str(doc.get("model_key") or "").replace("/", "\\").lower()
                if not mk or mk in index:
                    continue
                mn = [float("inf")] * 3
                mx = [float("-inf")] * 3
                for shape in doc.get("source_shapes") or []:
                    if not isinstance(shape, Mapping):
                        continue
                    for obj in shape.get("blender_objects") or []:
                        wb = obj.get("evaluated_world_bounds_game_units")
                        if not isinstance(wb, Mapping):
                            continue
                        wb_min = wb.get("min")
                        wb_max = wb.get("max")
                        if not (
                            isinstance(wb_min, Sequence)
                            and isinstance(wb_max, Sequence)
                            and len(wb_min) >= 3
                            and len(wb_max) >= 3
                        ):
                            continue
                        for axis in range(3):
                            mn[axis] = min(mn[axis], float(wb_min[axis]))
                            mx[axis] = max(mx[axis], float(wb_max[axis]))
                if mn[0] != float("inf"):
                    index[mk] = {
                        "min": [float(v) for v in mn],
                        "max": [float(v) for v in mx],
                    }
        _A2_MODEL_BOUNDS_BY_KEY = index
    return _A2_MODEL_BOUNDS_BY_KEY.get(str(model_key).replace("/", "\\").lower())


def _door_local_xy_extents(
    local_bounds: Mapping[str, Sequence[float]],
    member_rotz: float,
    door_rotz: float,
    offset: Sequence[float],
    scale: float,
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    ox, oy = float(offset[0]), float(offset[1])
    for lx in (local_bounds["min"][0], local_bounds["max"][0]):
        for ly in (local_bounds["min"][1], local_bounds["max"][1]):
            wx, wy = rotate_xy(scale * float(lx), scale * float(ly), member_rotz)
            wx += ox
            wy += oy
            dx, dy = stamp_to_door_local(wx, wy, door_rotz)
            xs.append(dx)
            ys.append(dy)
    return min(xs), max(xs), min(ys), max(ys)


def _contact_stair_approach_gu(
    *,
    door_rotz: float,
    stair_rotz: float,
    door_bounds: Mapping[str, Sequence[float]],
    stair_bounds: Mapping[str, Sequence[float]],
    door_scale: float = 1.0,
    stair_scale: float = 1.0,
    outer_members: Sequence[tuple[Mapping[str, Sequence[float]], Sequence[float], float, float]] | None = None,
) -> float | None:
    """Outward pivot distance so the stair's door-facing edge meets the door's outer edge."""
    outer_ys: list[float] = []
    _dmin_lx, _dmax_lx, door_min_ly, _door_max_ly = _door_local_xy_extents(
        door_bounds, door_rotz, door_rotz, [0.0, 0.0, 0.0], door_scale
    )
    outer_ys.append(door_min_ly)
    for bounds, offset, member_rotz, scale in outer_members or ():
        _min_lx, _max_lx, member_min_ly, _member_max_ly = _door_local_xy_extents(
            bounds, member_rotz, door_rotz, offset, scale
        )
        outer_ys.append(member_min_ly)
    door_outer_ly = min(outer_ys)
    _smin_lx, _smax_lx, _stair_min_ly, stair_max_ly = _door_local_xy_extents(
        stair_bounds, stair_rotz, door_rotz, [0.0, 0.0, 0.0], stair_scale
    )
    pivot_ly = door_outer_ly - stair_max_ly
    if pivot_ly > -8.0:
        return None
    return -pivot_ly


def _bundle_door_rotz(bundle: Mapping[str, Any]) -> float:
    for row in bundle.get("members") or []:
        if row.get("role") == "door":
            return float((row.get("rotation") or [0.0, 0.0, 0.0])[2])
    return 0.0


def _stair_layout_from_bundle(
    bundle: Mapping[str, Any],
    door_rotz: float,
    *,
    access_defaults: Mapping[str, Any],
) -> dict[str, Any] | None:
    stairs = _bundle_stair_rows(bundle)
    if not stairs:
        return None
    ref_rotz = _bundle_door_rotz(bundle)
    approaches: list[float] = []
    laterals: list[float] = []
    zs: list[float] = []
    for row in stairs:
        rel = row["offset_rel_door_gu"]
        lx, ly = stamp_to_door_local(float(rel[0]), float(rel[1]), ref_rotz)
        if -ly > 32.0:
            approaches.append(-ly)
        if abs(lx) > 32.0:
            laterals.append(abs(lx))
        zs.append(float(rel[2]))
    if not approaches:
        return None
    if len(stairs) >= 2 and laterals:
        lateral = min(sorted(laterals)[len(laterals) // 2], DEFAULT_DOUBLE_STAIR_LATERAL_GU)
        scale = float(access_defaults.get("double_stair_scale", 1.0))
    else:
        lateral = DEFAULT_DOUBLE_STAIR_LATERAL_GU
        scale = float(access_defaults.get("stair_scale", 0.88))
    return {
        "approach_gu": sorted(approaches)[len(approaches) // 2],
        "lateral_gu": lateral,
        "stair_z_gu": sorted(zs)[len(zs) // 2]
        if zs
        else access_defaults.get("stair_z_rel_gu"),
        "stair_rotz": door_rotz,
        "object_id": stairs[0].get("object_id") or access_defaults.get("stair_object_id"),
        "model_key": stairs[0].get("model_key") or access_defaults.get("stair_model_key"),
        "scale": scale,
    }


def _append_stairs_door_local(
    rows: list[dict[str, Any]],
    *,
    door_rotz: float,
    is_double: bool,
    layout: Mapping[str, Any],
    access_defaults: Mapping[str, Any],
    door_model_key: str | None = None,
    doorframe_model_key: str | None = None,
    doorframe_offset_rel_door_gu: Sequence[float] | None = None,
    doorframe_rotz: float | None = None,
) -> None:
    """Place stairs in door-local XY then map to stamp space.

    Door-local: +X = lateral right, -Y = outward approach. This matches mined
    bundle offsets and is independent of geographic ``outward_heading_deg``.
    """
    if is_double:
        lateral = DEFAULT_DOUBLE_STAIR_LATERAL_GU
        stair_scale = float(access_defaults.get("double_stair_scale", 1.0))
    else:
        lateral = max(64.0, float(layout["lateral_gu"]))
        stair_scale = float(layout.get("scale") or access_defaults.get("stair_scale", 0.88))
    stair_z = float(layout["stair_z_gu"])
    stair_rotz = float(layout.get("stair_rotz", door_rotz))
    stair_rotation = [0.0, 0.0, stair_rotz]
    stair_object_value = layout.get("object_id") or access_defaults.get("stair_object_id")
    stair_model_value = layout.get("model_key") or access_defaults.get("stair_model_key")
    if not stair_object_value or not stair_model_value:
        raise ValueError("mined grammar has no stair object/model evidence")
    stair_object = str(stair_object_value)
    stair_model_key = str(stair_model_value)
    door_bounds = _lookup_model_local_bounds(door_model_key)
    stair_bounds = _lookup_model_local_bounds(stair_model_key)
    approach = float(layout["approach_gu"])
    if door_bounds and stair_bounds:
        outer_members: list[tuple[Mapping[str, Sequence[float]], Sequence[float], float, float]] = []
        frame_bounds = _lookup_model_local_bounds(doorframe_model_key)
        if frame_bounds and doorframe_offset_rel_door_gu is not None:
            outer_members.append(
                (
                    frame_bounds,
                    doorframe_offset_rel_door_gu,
                    float(doorframe_rotz if doorframe_rotz is not None else door_rotz),
                    1.0,
                )
            )
        contact = _contact_stair_approach_gu(
            door_rotz=door_rotz,
            stair_rotz=stair_rotz,
            door_bounds=door_bounds,
            stair_bounds=stair_bounds,
            door_scale=1.0,
            stair_scale=stair_scale,
            outer_members=outer_members,
        )
        if contact is not None:
            approach = contact
    approach = max(48.0, min(approach, 280.0))
    if is_double:
        for sign in (-1.0, 1.0):
            wx, wy = door_local_to_stamp(lateral * sign, -approach, door_rotz)
            rows.append(
                {
                    "role": "stair",
                    "object_id": stair_object,
                    "model_key": None,
                    "record_type": "STAT",
                    "category": "exterior",
                    "scale": stair_scale,
                    "offset_rel_door_gu": [round(wx, 3), round(wy, 3), round(stair_z, 3)],
                    "rotation": stair_rotation,
                }
            )
    else:
        wx, wy = door_local_to_stamp(0.0, -approach, door_rotz)
        rows.append(
            {
                "role": "stair",
                "object_id": stair_object,
                "model_key": None,
                "record_type": "STAT",
                "category": "exterior",
                "scale": stair_scale,
                "offset_rel_door_gu": [round(wx, 3), round(wy, 3), round(stair_z, 3)],
                "rotation": stair_rotation,
            }
        )


def _transform_rel_door_gu(
    rel: Sequence[float],
    ref_rotz: float,
    door_rotz: float,
) -> list[float]:
    lx, ly = stamp_to_door_local(float(rel[0]), float(rel[1]), ref_rotz)
    wx, wy = door_local_to_stamp(lx, ly, door_rotz)
    return [round(wx, 3), round(wy, 3), round(float(rel[2]), 3)]


def _window_facade_id(local_x: float, local_y: float) -> str:
    if abs(local_x) >= abs(local_y):
        return "pos_x" if local_x >= 0.0 else "neg_x"
    return "pos_y" if local_y >= 0.0 else "neg_y"


def _normalize_windows_by_facade(windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Snap window Z to one height per shell face so facades read as intentional bands."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in windows:
        offset = row.get("offset_gu") or [0.0, 0.0, 0.0]
        groups[_window_facade_id(float(offset[0]), float(offset[1]))].append(row)
    normalized: list[dict[str, Any]] = []
    for rows in groups.values():
        zs = sorted(float(row["offset_gu"][2]) for row in rows)
        band_z = zs[len(zs) // 2]
        for row in rows:
            offset = row["offset_gu"]
            copied = dict(row)
            copied["offset_gu"] = [float(offset[0]), float(offset[1]), round(band_z, 3)]
            normalized.append(copied)
    return normalized


def _filter_windows_to_facades(
    windows: Sequence[Mapping[str, Any]],
    facade_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    rows = _normalize_windows_by_facade(windows)
    if not facade_ids:
        return rows
    allowed = set(facade_ids)
    return [
        row
        for row in rows
        if _window_facade_id(float(row["offset_gu"][0]), float(row["offset_gu"][1])) in allowed
    ]


def _direct_shell_contacts(
    stamp: Mapping[str, Any], source_id: str,
) -> list[Mapping[str, Any]]:
    """Return direct shell-contact witnesses for one attached member."""

    contacts: list[Mapping[str, Any]] = []
    for edge in stamp.get("shell_attachment_edges") or []:
        ref_a, ref_b = str(edge.get("ref_a")), str(edge.get("ref_b"))
        if source_id not in {ref_a, ref_b}:
            continue
        contacts.append(edge)
    return contacts


def _direct_window_contacts(
    stamp: Mapping[str, Any], source_id: str,
) -> list[Mapping[str, Any]]:
    """Return evaluated direct contacts from one member to window members."""

    members = {
        str(member.get("source_id")): member
        for member in stamp.get("members") or []
    }
    contacts: list[Mapping[str, Any]] = []
    edges = list(stamp.get("member_contact_edges") or [])
    # Older source products only carried shell edges; retain compatibility for
    # those documents without inventing a dormer/window relation.
    if not edges:
        edges = list(stamp.get("shell_attachment_edges") or [])
    for edge in edges:
        ref_a, ref_b = str(edge.get("ref_a")), str(edge.get("ref_b"))
        if source_id not in {ref_a, ref_b}:
            continue
        other = ref_b if ref_a == source_id else ref_a
        if classify_attachment(members.get(other, {})) == "window":
            contacts.append(edge)
    return contacts


def _template_member_kept(member: Mapping[str, Any]) -> bool:
    """Keep modular kit composition; drop terrain/debris attachments only."""
    cls = classify_attachment(member)
    mk = (member.get("model_key") or "").lower()
    if cls == "fence" or "fence" in mk or "ffence" in mk:
        return False
    if cls == "wall" and "blck" in mk:
        return False
    if cls == "shell":
        return is_house_shell_id(shell_key(member))
    return cls in {"door", "doorframe", "stair", "window", "chimney", "dormer", "porch", "tent", "decoration"}


def is_double_door_object(object_id: str) -> bool:
    text = object_id.lower()
    return "double" in text or "ddr" in text


def normalize_access_policy(policy: Mapping[str, Any] | None) -> dict[str, str]:
    defaults = {
        "stairs": "auto",
        "terrain_at_door": "auto",
        "access_placement": "synthetic",
        "decorations": "source",
    }
    if not policy:
        return defaults
    merged = dict(defaults)
    for key in defaults:
        if key in policy:
            merged[key] = str(policy[key])
    return merged


def quantize(value: float, grid: float = POSITION_CLUSTER_GU) -> float:
    return round(value / grid) * grid


def quantize_vec3(values: Sequence[float], grid: float = POSITION_CLUSTER_GU) -> tuple[float, float, float]:
    return quantize(values[0], grid), quantize(values[1], grid), quantize(values[2], grid)


def shells_in_stamp(stamp: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [member for member in stamp["members"] if classify_attachment(member) == "shell"]


def cluster_key(values: Sequence[float]) -> str:
    qx, qy, qz = quantize_vec3(values)
    return f"{qx:.0f},{qy:.0f},{qz:.0f}"


def _bundle_member_row(member: Mapping[str, Any], door: Mapping[str, Any], role: str) -> dict[str, Any]:
    rotation = member.get("rotation") or [0.0, 0.0, 0.0]
    return {
        "role": role,
        "object_id": member.get("object_id"),
        "model_key": member.get("model_key"),
        "record_type": member.get("record_type", "STAT"),
        "category": member.get("category", "exterior"),
        "scale": float(member.get("scale") or 1.0),
        "offset_rel_door_gu": [round(v, 3) for v in offset_rel_door(member, door)],
        "rotation": [float(v) for v in rotation],
        "outward_heading_deg": member.get("outward_heading_deg"),
    }


def _extract_access_bundle(stamp: Mapping[str, Any], door: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    door_pos = door["offset_gu"]
    for member in stamp["members"]:
        cls = classify_attachment(member)
        rel = offset_rel_door(member, door)
        dist_xy = math.hypot(rel[0], rel[1])
        mk = (member.get("model_key") or "").lower()
        if member.get("source_id") == door.get("source_id") and cls == "door":
            rows.append(_bundle_member_row(member, door, "door"))
            continue
        if cls == "doorframe" and dist_xy <= DOOR_ASSOC_GU:
            rows.append(_bundle_member_row(member, door, "doorframe"))
        elif cls == "stair" and dist_xy <= STAIR_ASSOC_GU:
            rows.append(_bundle_member_row(member, door, "stair"))
        elif cls == "wall" and "blck" in mk and dist_xy <= APPROACH_BLOCK_ASSOC_GU:
            rows.append(_bundle_member_row(member, door, "approach_block"))
    return rows


def _bundle_score(bundle: Mapping[str, Any]) -> float:
    roles = {row["role"] for row in bundle["members"]}
    score = 0.0
    if "doorframe" in roles:
        score += 4
    if "stair" in roles:
        score += 4
    score += sum(1 for row in bundle["members"] if row["role"] == "approach_block")
    door = next((row for row in bundle["members"] if row["role"] == "door"), None)
    stair = next((row for row in bundle["members"] if row["role"] == "stair"), None)
    if door is not None and stair is not None:
        rel = stair["offset_rel_door_gu"]
        dist_xy = math.hypot(float(rel[0]), float(rel[1]))
        score -= dist_xy / 64.0
        if dist_xy > 320.0:
            score -= 8.0
    return score


def _windows_for_stamp(
    stamp: Mapping[str, Any],
    shell: Mapping[str, Any],
    doors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    shell_offset = shell["offset_gu"]
    shell_rotz = rotz_of(shell)
    door_locals = [
        to_shell_local(door["offset_gu"], door["rotation"], shell_offset, shell_rotz)[0]
        for door in doors
    ]
    rows: list[dict[str, Any]] = []
    for member in stamp["members"]:
        if classify_attachment(member) != "window":
            continue
        local_offset, local_rotation = to_shell_local(
            member["offset_gu"], member["rotation"], shell_offset, shell_rotz
        )
        if any(
            math.hypot(local_offset[0] - door_local[0], local_offset[1] - door_local[1])
            < DOOR_CLEARANCE_XY_GU
            for door_local in door_locals
        ):
            continue
        rows.append(
            {
                "object_id": member.get("object_id"),
                "model_key": member.get("model_key"),
                "offset_gu": [round(v, 3) for v in local_offset],
                "rotation": [round(v, 6) for v in local_rotation],
                "scale": float(member.get("scale") or 1.0),
            }
        )
    return rows


@dataclass
class SlotAccumulator:
    slot_id: str
    offset_gu: list[float]
    rotation: list[float]
    outward_heading_deg: float | None = None
    door_models: Counter = field(default_factory=Counter)
    occurrence_count: int = 0
    source_stamps: set[str] = field(default_factory=set)
    access_bundles: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ShellAccumulator:
    shell_id: str
    object_id: str
    model_key: str
    size_classes: Counter = field(default_factory=Counter)
    door_slots: dict[str, SlotAccumulator] = field(default_factory=dict)
    chimney_slots: dict[str, dict[str, Any]] = field(default_factory=dict)
    window_facades: dict[str, dict[str, Any]] = field(default_factory=dict)
    decorations: dict[str, dict[str, Any]] = field(default_factory=dict)
    stamp_count: int = 0


def _slot_from_door(
    shell_acc: ShellAccumulator,
    door: Mapping[str, Any],
    local_offset: Sequence[float],
    local_rotation: Sequence[float],
    stamp_id: str,
) -> SlotAccumulator:
    key = cluster_key(local_offset)
    if key in shell_acc.door_slots:
        slot = shell_acc.door_slots[key]
    else:
        slot = SlotAccumulator(
            slot_id=f"door_{len(shell_acc.door_slots)}",
            offset_gu=[float(v) for v in local_offset],
            rotation=[float(v) for v in local_rotation],
            outward_heading_deg=door.get("outward_heading_deg"),
        )
        shell_acc.door_slots[key] = slot
    slot.door_models[door.get("object_id") or ""] += 1
    slot.occurrence_count += 1
    slot.source_stamps.add(stamp_id)
    if door.get("outward_heading_deg") is not None:
        slot.outward_heading_deg = float(door["outward_heading_deg"])
    return slot


def _finalize_slot(slot: SlotAccumulator) -> dict[str, Any]:
    bundles = sorted(slot.access_bundles, key=lambda bundle: (-_bundle_score(bundle), bundle["source_stamp_id"]))
    payload: dict[str, Any] = {
        "slot_id": slot.slot_id,
        "offset_gu": [round(v, 3) for v in slot.offset_gu],
        "rotation": [round(v, 6) for v in slot.rotation],
        "occurrence_count": slot.occurrence_count,
        "source_stamp_count": len(slot.source_stamps),
        "door_models": [{"object_id": oid, "count": c} for oid, c in slot.door_models.most_common()],
        "access_bundles": bundles,
    }
    if slot.outward_heading_deg is not None:
        payload["outward_heading_deg"] = round(float(slot.outward_heading_deg), 2)
    if bundles:
        payload["canonical_bundle_source"] = bundles[0]["source_stamp_id"]
        payload["canonical_bundle_score"] = _bundle_score(bundles[0])
    return payload


def _finalize_shell(shell_acc: ShellAccumulator) -> dict[str, Any]:
    size_class = shell_acc.size_classes.most_common(1)[0][0] if shell_acc.size_classes else "unknown"
    door_slots = []
    for index, slot in enumerate(sorted(shell_acc.door_slots.values(), key=lambda s: s.slot_id)):
        finalized = _finalize_slot(slot)
        finalized["slot_id"] = f"door_{index}"
        door_slots.append(finalized)
    chimneys = sorted(shell_acc.chimney_slots.values(), key=lambda row: row["slot_id"])
    window_facades = []
    for facade_id in sorted(shell_acc.window_facades):
        row = shell_acc.window_facades[facade_id]
        zs = sorted(row["z_values"])
        slots = []
        for slot_index, (_key, slot) in enumerate(
            sorted(row.get("slots", {}).items())
        ):
            slots.append(
                {
                    "slot_id": f"{facade_id}_win_{slot_index}",
                    "offset_gu": slot["offset_gu"],
                    "occurrence_count": int(slot["occurrence_count"]),
                    "window_models": [
                        {"object_id": oid, "count": count}
                        for oid, count in slot["window_models"].most_common()
                    ],
                    "direct_shell_contact_count": int(slot["direct_shell_contact_count"]),
                    "contact_distances_gu": sorted(round(float(v), 3) for v in slot["contact_distances_gu"]),
                }
            )
        window_facades.append(
            {
                "facade_id": facade_id,
                "z_gu": round(float(zs[len(zs) // 2]), 3),
                "occurrence_count": int(row["occurrence_count"]),
                "window_models": [
                    {"object_id": oid, "count": count}
                    for oid, count in row["window_models"].most_common()
                ],
                "direct_shell_contact_count": int(row["direct_shell_contact_count"]),
                "window_slots": slots,
            }
        )
    decorations = []
    for _key, row in sorted(shell_acc.decorations.items()):
        finalized = dict(row)
        finalized["models"] = [
            {"object_id": oid, "count": count}
            for oid, count in row["models"].most_common()
        ]
        finalized["source_refs"] = sorted(row["source_refs"])
        finalized["direct_shell_contact_count"] = int(row["direct_shell_contact_count"])
        finalized["contact_distances_gu"] = sorted(
            round(float(value), 3) for value in row["contact_distances_gu"]
        )
        finalized["direct_window_contact_count"] = int(
            row.get("direct_window_contact_count", 0)
        )
        finalized["window_contact_source_refs"] = sorted(
            row.get("window_contact_source_refs", set())
        )
        finalized["window_attachment_evidence"] = (
            "attached_mesh"
            if finalized["direct_window_contact_count"]
            else "none_observed"
        )
        decorations.append(finalized)
    return {
        "shell_id": shell_acc.shell_id,
        "object_id": shell_acc.object_id,
        "model_key": shell_acc.model_key,
        "size_class": size_class,
        "stamp_count": shell_acc.stamp_count,
        "door_slots": door_slots,
        "chimney_slots": chimneys,
        "window_facades": window_facades,
        "decorations": decorations,
    }


def mine_stamp_for_shells(
    stamp: Mapping[str, Any],
    shell_registry: dict[str, ShellAccumulator],
    attachment_catalog: dict[str, Counter],
) -> None:
    stamp_id = str(stamp["stamp_id"])
    shells = shells_in_stamp(stamp)
    if not shells:
        return
    if len(shells) == 1:
        shell = shells[0]
        sid = shell_key(shell)
        if sid not in shell_registry:
            shell_registry[sid] = ShellAccumulator(
                shell_id=sid,
                object_id=str(shell.get("object_id") or ""),
                model_key=str(shell.get("model_key") or ""),
            )
        shell_acc = shell_registry[sid]
        shell_acc.stamp_count += 1
        shell_acc.size_classes[str(stamp.get("size_class") or "unknown")] += 1
        shell_rotz = rotz_of(shell)
        shell_offset = shell["offset_gu"]
        doors: list[Mapping[str, Any]] = []
        for member in stamp["members"]:
            cls = classify_attachment(member)
            if cls == "shell":
                continue
            attachment_catalog[cls][member.get("object_id") or ""] += 1
            if cls == "door":
                doors.append(member)
            elif cls == "chimney":
                local_offset, local_rotation = to_shell_local(
                    member["offset_gu"], member["rotation"], shell_offset, shell_rotz
                )
                key = cluster_key(local_offset)
                if key not in shell_acc.chimney_slots:
                    shell_acc.chimney_slots[key] = {
                        "slot_id": f"chimney_{len(shell_acc.chimney_slots)}",
                        "offset_gu": [round(v, 3) for v in local_offset],
                        "rotation": [round(v, 6) for v in local_rotation],
                        "chimney_models": Counter(),
                    }
                shell_acc.chimney_slots[key]["chimney_models"][member.get("object_id") or ""] += 1
            elif cls in {"dormer", "porch", "tent", "decoration"}:
                local_offset, local_rotation = to_shell_local(
                    member["offset_gu"], member["rotation"], shell_offset, shell_rotz
                )
                key = f"{cls}:{cluster_key(local_offset)}"
                decoration = shell_acc.decorations.get(key)
                if decoration is None:
                    decoration = {
                        "structural_role": cls,
                        "object_id": member.get("object_id"),
                        "model_key": member.get("model_key"),
                        "offset_gu": [round(v, 3) for v in local_offset],
                        "rotation": [round(v, 6) for v in local_rotation],
                        "scale": float(member.get("scale") or 1.0),
                        "occurrence_count": 0,
                        "models": Counter(),
                        "source_refs": set(),
                        "direct_shell_contact_count": 0,
                        "contact_distances_gu": [],
                        "direct_window_contact_count": 0,
                        "window_contact_source_refs": set(),
                    }
                    shell_acc.decorations[key] = decoration
                decoration["occurrence_count"] += 1
                decoration["models"][member.get("object_id") or ""] += 1
                decoration["source_refs"].add(str(member["source_id"]))
                contacts = _direct_shell_contacts(stamp, str(member["source_id"]))
                decoration["direct_shell_contact_count"] += len(contacts)
                decoration["contact_distances_gu"].extend(
                    float(edge["minimum_distance_gu"])
                    for edge in contacts
                    if edge.get("minimum_distance_gu") is not None
                )
                window_contacts = _direct_window_contacts(
                    stamp, str(member["source_id"])
                )
                decoration["direct_window_contact_count"] += len(window_contacts)
                decoration["window_contact_source_refs"].update(
                    str(
                        edge["ref_b"]
                        if str(edge.get("ref_a")) == str(member["source_id"])
                        else edge["ref_a"]
                    )
                    for edge in window_contacts
                )
        door_locals = [
            to_shell_local(door["offset_gu"], door["rotation"], shell_offset, shell_rotz)[0]
            for door in doors
        ]
        for member in stamp["members"]:
            if classify_attachment(member) != "window":
                continue
            local_offset, _local_rotation = to_shell_local(
                member["offset_gu"], member["rotation"], shell_offset, shell_rotz
            )
            if any(
                math.hypot(local_offset[0] - door_local[0], local_offset[1] - door_local[1])
                < DOOR_CLEARANCE_XY_GU
                for door_local in door_locals
            ):
                continue
            facade_id = _window_facade_id(local_offset[0], local_offset[1])
            if facade_id not in shell_acc.window_facades:
                shell_acc.window_facades[facade_id] = {
                    "z_values": [],
                    "occurrence_count": 0,
                    "window_models": Counter(),
                    "slots": {},
                    "direct_shell_contact_count": 0,
                }
            facade = shell_acc.window_facades[facade_id]
            facade["z_values"].append(float(local_offset[2]))
            facade["occurrence_count"] += 1
            facade["window_models"][member.get("object_id") or ""] += 1
            contacts = _direct_shell_contacts(stamp, str(member["source_id"]))
            facade["direct_shell_contact_count"] += len(contacts)
            slot_key = cluster_key(local_offset)
            slots = facade["slots"]
            if slot_key not in slots:
                slots[slot_key] = {
                    "offset_gu": [round(float(v), 3) for v in local_offset],
                    "occurrence_count": 0,
                    "window_models": Counter(),
                    "direct_shell_contact_count": 0,
                    "contact_distances_gu": [],
                }
            slots[slot_key]["occurrence_count"] += 1
            slots[slot_key]["window_models"][member.get("object_id") or ""] += 1
            slots[slot_key]["direct_shell_contact_count"] += len(contacts)
            slots[slot_key]["contact_distances_gu"].extend(
                float(edge["minimum_distance_gu"])
                for edge in contacts
                if edge.get("minimum_distance_gu") is not None
            )
        for door in doors:
            local_offset, local_rotation = to_shell_local(
                door["offset_gu"], door["rotation"], shell_offset, shell_rotz
            )
            slot = _slot_from_door(shell_acc, door, local_offset, local_rotation, stamp_id)
            bundle_members = _extract_access_bundle(stamp, door)
            if not any(row["role"] == "door" for row in bundle_members):
                bundle_members.insert(0, _bundle_member_row(door, door, "door"))
            slot.access_bundles.append(
                {
                    "source_stamp_id": stamp_id,
                    "members": bundle_members,
                    "windows": _windows_for_stamp(stamp, shell, doors),
                }
            )
        return
    # Multi-shell stamps are physical buildings, but their attachment evidence
    # still belongs to the individual shell surfaces.  Mine a shell-local view
    # rather than dropping all windows/dormers/porches from the aggregate.
    member_by_id = {
        str(member.get("source_id")): member
        for member in stamp.get("members") or []
    }
    all_edges = list(stamp.get("member_contact_edges") or [])
    if not all_edges:
        all_edges = list(stamp.get("shell_attachment_edges") or [])
    for shell in shells:
        shell_ref = str(shell.get("source_id"))
        included = {shell_ref}
        for edge in stamp.get("shell_attachment_edges") or []:
            left, right = str(edge.get("ref_a")), str(edge.get("ref_b"))
            if shell_ref not in {left, right}:
                continue
            included.add(right if left == shell_ref else left)
        # Doors can be separated from their shell by a frame or stair.  Keep
        # only this short access chain; arbitrary boundary chains remain out.
        changed = True
        while changed:
            changed = False
            for edge in all_edges:
                left, right = str(edge.get("ref_a")), str(edge.get("ref_b"))
                if left not in included and right not in included:
                    continue
                other = right if left in included else left
                if other in included or other not in member_by_id:
                    continue
                if classify_attachment(member_by_id[other]) in {"door", "doorframe", "stair"}:
                    included.add(other)
                    changed = True
        view = dict(stamp)
        view["stamp_id"] = f"{stamp['stamp_id']}__{shell_ref}"
        view["multi_shell"] = False
        view["members"] = [member_by_id[ref] for ref in sorted(included)]
        view["shell_attachment_edges"] = [
            edge
            for edge in (stamp.get("shell_attachment_edges") or [])
            if str(edge.get("ref_a")) in included and str(edge.get("ref_b")) in included
        ]
        view["member_contact_edges"] = [
            edge
            for edge in all_edges
            if str(edge.get("ref_a")) in included and str(edge.get("ref_b")) in included
        ]
        mine_stamp_for_shells(view, shell_registry, attachment_catalog)


def mine_stamp_templates(
    stamps: Sequence[Mapping[str, Any]],
    *,
    house_shell_prefixes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for stamp in stamps:
        shells = shells_in_stamp(stamp)
        if len(shells) < 2:
            continue
        shell_ids = [shell_key(shell) for shell in shells]
        house_shell_count = sum(
            1 for sid in shell_ids if is_house_shell_id(sid, house_shell_prefixes)
        )
        ineligible_reason: str | None = None
        eligible = True
        if house_shell_count < 2:
            eligible = False
            ineligible_reason = "fewer_than_two_house_shells"
        if any("barrow" in sid for sid in shell_ids):
            eligible = False
            ineligible_reason = "contains_barrow_shell"
        if any(sid.startswith("sky_ex_alt_") for sid in shell_ids):
            eligible = False
            ineligible_reason = "contains_direnni_or_keep_shell"
        templates.append(
            {
                "template_id": f"template_{len(templates)}",
                "source_stamp_id": stamp["stamp_id"],
                "shell_count": len(shells),
                "house_shell_count": house_shell_count,
                "shell_ids": shell_ids,
                "door_count": int(stamp.get("door_count") or 0),
                "eligible": eligible,
                "ineligible_reason": ineligible_reason,
            }
        )
    return templates


def _mine_attachment_defaults(stamps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame_offsets: list[list[float]] = []
    frame_objects: Counter[str] = Counter()
    frame_model_keys: dict[str, str] = {}
    stair_approaches: list[float] = []
    stair_z_values: list[float] = []
    stair_objects: Counter[str] = Counter()
    stair_model_keys: dict[str, str] = {}
    double_frame_objects: Counter[str] = Counter()
    double_stair_objects: Counter[str] = Counter()
    double_stair_scales: list[float] = []
    double_stair_lateral: list[float] = []
    approach_block_z_values: list[float] = []
    for stamp in stamps:
        doors = [member for member in stamp["members"] if classify_attachment(member) == "door"]
        for door in doors:
            heading = float(door.get("outward_heading_deg") or 0.0)
            ax, ay = approach_unit(heading)
            is_double = is_double_door_object(str(door.get("object_id") or ""))
            stair_rows: list[Mapping[str, Any]] = []
            for member in stamp["members"]:
                cls = classify_attachment(member)
                if cls == "doorframe":
                    rel = offset_rel_door(member, door)
                    if math.hypot(rel[0], rel[1]) > DOOR_ASSOC_GU:
                        continue
                    frame_offsets.append(rel)
                    oid = str(member.get("object_id") or "")
                    frame_objects[oid] += 1
                    if is_double_door_object(oid):
                        double_frame_objects[oid] += 1
                    if oid and member.get("model_key"):
                        frame_model_keys[oid] = str(member["model_key"])
                elif cls == "stair":
                    rel = offset_rel_door(member, door)
                    dist_xy = math.hypot(rel[0], rel[1])
                    if dist_xy > STAIR_ASSOC_GU:
                        continue
                    door_rotz = rotz_of(door)
                    lx, ly = stamp_to_door_local(rel[0], rel[1], door_rotz)
                    if -ly < 32.0 or dist_xy < 48.0:
                        continue
                    stair_approaches.append(-ly)
                    stair_z_values.append(rel[2])
                    oid = str(member.get("object_id") or "")
                    stair_objects[oid] += 1
                    if oid and member.get("model_key"):
                        stair_model_keys[oid] = str(member["model_key"])
                    stair_rows.append(member)
                elif cls == "wall":
                    mk = (member.get("model_key") or "").lower()
                    if "blck" not in mk:
                        continue
                    rel = offset_rel_door(member, door)
                    if math.hypot(rel[0], rel[1]) > APPROACH_BLOCK_ASSOC_GU:
                        continue
                    approach_block_z_values.append(rel[2])
            if is_double and len(stair_rows) >= 2:
                for row in stair_rows:
                    oid = str(row.get("object_id") or "")
                    double_stair_objects[oid] += 1
                    double_stair_scales.append(float(row.get("scale") or 1.0))
                    rel = offset_rel_door(row, door)
                    lx, ly = stamp_to_door_local(rel[0], rel[1], rotz_of(door))
                    double_stair_lateral.append(abs(lx))
    warnings: list[str] = []
    if not frame_offsets:
        warnings.append("no doorframe evidence in mined stamps")
    median_frame = (
        [sorted(value)[len(value) // 2] for value in zip(*frame_offsets)]
        if frame_offsets
        else None
    )
    top_frame = frame_objects.most_common(1)[0][0] if frame_objects else None
    top_stair = stair_objects.most_common(1)[0][0] if stair_objects else None
    top_double_frame = double_frame_objects.most_common(1)[0][0] if double_frame_objects else None
    if not stair_objects:
        warnings.append("no stair evidence in mined stamps")
    if not double_frame_objects:
        warnings.append("no double-doorframe evidence in mined stamps")
    stair_approach = (
        sorted(stair_approaches)[len(stair_approaches) // 2]
        if stair_approaches
        else None
    )
    if stair_approach is not None:
        stair_approach = max(120.0, min(float(stair_approach), 280.0))
    stair_z = sorted(stair_z_values)[len(stair_z_values) // 2] if stair_z_values else None
    block_z = (
        sorted(approach_block_z_values)[len(approach_block_z_values) // 2]
        if approach_block_z_values
        else None
    )
    top_double_stair = double_stair_objects.most_common(1)[0][0] if double_stair_objects else None
    double_lateral = (
        sorted(double_stair_lateral)[len(double_stair_lateral) // 2]
        if double_stair_lateral
        else None
    )
    double_scale = (
        sorted(double_stair_scales)[len(double_stair_scales) // 2] if double_stair_scales else None
    )
    if not approach_block_z_values:
        warnings.append("no approach-block evidence in mined stamps")
    return {
        "doorframe_object_id": top_frame,
        "doorframe_model_key": frame_model_keys.get(top_frame) if top_frame else None,
        "doorframe_offset_rel_door_gu": (
            [round(v, 3) for v in median_frame] if median_frame is not None else None
        ),
        "double_doorframe_object_id": top_double_frame,
        "stair_object_id": top_stair,
        "stair_model_key": stair_model_keys.get(top_stair) if top_stair else None,
        "double_stair_object_id": top_double_stair,
        "double_stair_scale": round(float(double_scale), 3) if double_scale is not None else None,
        "stair_approach_gu": round(float(stair_approach), 3) if stair_approach is not None else None,
        "stair_z_rel_gu": round(float(stair_z), 3) if stair_z is not None else None,
        "double_stair_lateral_gu": round(float(double_lateral), 3) if double_lateral is not None else None,
        "double_door_object_id": None,
        "approach_block_object_id": None,
        "approach_block_approach_fraction": 0.55,
        "approach_block_z_rel_gu": round(float(block_z), 3) if block_z is not None else None,
        "elevation_stair_threshold_gu": ELEVATION_STAIR_THRESHOLD_GU,
        "mining_warnings": warnings,
    }


def _wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _finalize_connection_rules(
    rules: dict[tuple[str, str], dict[str, Any]],
    *,
    id_field_a: str,
    id_field_b: str,
) -> list[dict[str, Any]]:
    """Deterministic mean/delta summary for accumulated connection pairs."""

    out: list[dict[str, Any]] = []
    for (lo, hi), rec in sorted(rules.items()):
        offsets = rec["offset_deltas_gu"]
        rots = rec["rotz_deltas_deg"]
        count = len(offsets)
        mean = [round(sum(d[i] for d in offsets) / count, 3) for i in range(3)]
        spread = [
            round(max(d[i] for d in offsets) - min(d[i] for d in offsets), 3)
            for i in range(3)
        ]
        rot_counter = Counter(rots)
        best = max(rot_counter.values())
        rot_mode = min(rot for rot, n in rot_counter.items() if n == best)
        out.append(
            {
                id_field_a: lo,
                id_field_b: hi,
                "sample_count": rec["sample_count"],
                "offset_delta_mean_gu": mean,
                "offset_delta_spread_gu": spread,
                "rotz_delta_mode_deg": round(rot_mode, 3),
                "rotz_delta_values_deg": sorted(round(v, 3) for v in set(rots)),
                "source_stamps": sorted(rec["source_stamps"]),
            }
        )
    return out


def mine_shell_connections(stamps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mine how house shells CONNECT from multi-shell stamps.

    Each multi-shell stamp carries ``touching_pairs`` (ref ids with direct
    contact edges, recorded by unit derivation).  For every touching pair
    this records the world offset delta and rotz delta between the two shell
    anchors — the rule a generator applies to attach one shell to another.
    """

    rules: dict[tuple[str, str], dict[str, Any]] = {}
    for stamp in stamps:
        pairs = stamp.get("touching_pairs") or []
        if not pairs:
            continue
        by_id = {str(m["source_id"]): m for m in stamp.get("members") or []}
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            member_a = by_id.get(str(pair[0]))
            member_b = by_id.get(str(pair[1]))
            if member_a is None or member_b is None:
                continue
            key_a, key_b = shell_key(member_a), shell_key(member_b)
            if key_a == key_b:
                continue
            lo, hi = sorted((key_a, key_b))
            first, second = (member_a, member_b) if key_a == lo else (member_b, member_a)
            first_offset, second_offset = first["offset_gu"], second["offset_gu"]
            delta = [
                round(float(second_offset[i]) - float(first_offset[i]), 3)
                for i in range(3)
            ]
            rotz_delta = _wrap180(rotz_of(second) - rotz_of(first))
            rec = rules.setdefault(
                (lo, hi),
                {
                    "sample_count": 0,
                    "offset_deltas_gu": [],
                    "rotz_deltas_deg": [],
                    "source_stamps": set(),
                },
            )
            rec["sample_count"] += 1
            rec["offset_deltas_gu"].append(delta)
            rec["rotz_deltas_deg"].append(rotz_delta)
            rec["source_stamps"].add(str(stamp["stamp_id"]))
    return _finalize_connection_rules(rules, id_field_a="shell_a", id_field_b="shell_b")


def mine_piece_connections(stamps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mine piece-to-piece attachment rules from piecewise stamps.

    Piecewise stamps carry ``internal_edges`` (contact edges among their own
    members).  Rules are keyed by model basename pair with the relative
    anchor transform, so a generator can assemble new multi-piece
    structures from the same connection vocabulary.
    """

    def basename(model_key: object) -> str:
        return str(model_key or "").replace("/", "\\").casefold().rsplit("\\", 1)[-1]

    rules: dict[tuple[str, str], dict[str, Any]] = {}
    for stamp in stamps:
        edges = stamp.get("internal_edges") or []
        if not edges:
            continue
        by_id = {str(m["source_id"]): m for m in stamp.get("members") or []}
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            member_a = by_id.get(str(edge.get("ref_a")))
            member_b = by_id.get(str(edge.get("ref_b")))
            if member_a is None or member_b is None:
                continue
            base_a, base_b = basename(member_a.get("model_key")), basename(member_b.get("model_key"))
            if base_a == base_b:
                continue
            lo, hi = sorted((base_a, base_b))
            first, second = (member_a, member_b) if base_a == lo else (member_b, member_a)
            first_offset, second_offset = first["offset_gu"], second["offset_gu"]
            delta = [
                round(float(second_offset[i]) - float(first_offset[i]), 3)
                for i in range(3)
            ]
            rotz_delta = _wrap180(rotz_of(second) - rotz_of(first))
            rec = rules.setdefault(
                (lo, hi),
                {
                    "sample_count": 0,
                    "offset_deltas_gu": [],
                    "rotz_deltas_deg": [],
                    "source_stamps": set(),
                },
            )
            rec["sample_count"] += 1
            rec["offset_deltas_gu"].append(delta)
            rec["rotz_deltas_deg"].append(rotz_delta)
            rec["source_stamps"].add(str(stamp["stamp_id"]))
    return _finalize_connection_rules(rules, id_field_a="piece_a", id_field_b="piece_b")


def mine_grammar_from_library(
    library_path: Path,
    *,
    kit_id: str,
    grammar_id: str | None = None,
    mined_at: str = "2026-08-18",
    house_shell_prefixes: Sequence[str] | None = None,
    size_class_thresholds_gu: Sequence[float] | None = None,
) -> dict[str, Any]:
    library_path = library_path.resolve()
    payload = json.loads(library_path.read_text(encoding="utf-8"))
    prefixes = tuple(
        str(value)
        for value in (
            house_shell_prefixes
            or payload.get("house_shell_prefixes")
            or DEFAULT_HOUSE_SHELL_PREFIXES
        )
    )
    raw_thresholds = (
        size_class_thresholds_gu
        or (payload.get("mining_settings") or {}).get("size_class_thresholds_gu")
        or (900.0, 1800.0)
    )
    thresholds = tuple(sorted(float(value) for value in raw_thresholds))
    if len(thresholds) != 2 or thresholds[0] <= 0 or thresholds[1] <= thresholds[0]:
        raise ValueError("size_class_thresholds_gu must contain two increasing positive values")

    def measured_size_class(stamp: Mapping[str, Any]) -> str:
        bounds = stamp.get("bounds_rel_gu") or (stamp.get("footprint") or {}).get("aabb_rel")
        spans = bounds.get("span") if isinstance(bounds, Mapping) else None
        if isinstance(spans, Sequence) and len(spans) >= 2:
            width = max(float(spans[0]), float(spans[1]))
        else:
            members = stamp.get("members") or []
            xs = [float(row["offset_gu"][0]) for row in members if row.get("offset_gu")]
            ys = [float(row["offset_gu"][1]) for row in members if row.get("offset_gu")]
            width = max(max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else 0.0
        return "small" if width < thresholds[0] else "medium" if width < thresholds[1] else "large"

    # Keep only stamps with at least one configured house shell.  Other shells
    # (forts, camps, mines) remain site-level provenance but never inflate the
    # house grammar's shell/template counts.
    stamps: list[dict[str, Any]] = []
    for original in payload["stamps"]:
        shells = [
            member
            for member in shells_in_stamp(original)
            if is_house_shell_id(shell_key(member), prefixes)
        ]
        if not shells:
            continue
        stamp = dict(original)
        stamp["size_class"] = measured_size_class(original)
        stamp["members"] = [
            member
            for member in original.get("members") or []
            if classify_attachment(member) != "shell"
            or is_house_shell_id(shell_key(member), prefixes)
        ]
        stamps.append(stamp)
    shell_registry: dict[str, ShellAccumulator] = {}
    attachment_catalog: dict[str, Counter] = defaultdict(Counter)
    for stamp in stamps:
        mine_stamp_for_shells(stamp, shell_registry, attachment_catalog)
    shells = [_finalize_shell(shell_acc) for shell_acc in sorted(shell_registry.values(), key=lambda s: s.shell_id)]
    for shell in shells:
        for row in shell["chimney_slots"]:
            row["chimney_models"] = [
                {"object_id": oid, "count": c} for oid, c in row["chimney_models"].most_common()
            ]
    attachments = {
        cls: [{"object_id": oid, "count": count} for oid, count in counter.most_common()]
        for cls, counter in sorted(attachment_catalog.items())
    }
    stamp_templates = mine_stamp_templates(stamps, house_shell_prefixes=prefixes)
    attachment_defaults = _mine_attachment_defaults(stamps)
    # Connection rules: shell-to-shell from multi-shell stamps, and
    # piece-to-piece from piecewise stamps.  Piecewise stamps carry no
    # configured house shell, so they are mined from the RAW library (the
    # filtered `stamps` list would have dropped them).
    shell_connections = mine_shell_connections(stamps)
    piece_connections = mine_piece_connections(payload.get("stamps") or [])
    grammar_id = grammar_id or f"{payload.get('library_id', kit_id)}_grammar_v1"
    return {
        "schema_version": SCHEMA_VERSION,
        "grammar_id": grammar_id,
        "kit_id": kit_id,
        "assembly_mode": "monolithic_shell",
        "source": {
            "stamp_library": library_path.as_posix(),
            "stamp_library_sha256": sha256_file(library_path),
            "stamp_count": len(stamps),
            "mined_at": mined_at,
            "mined_by": f"kit_house_grammar {__version__}",
            "house_shell_prefixes": list(prefixes),
            "size_class_thresholds_gu": list(thresholds),
        },
        "constraints": {
            "door_requires_doorframe": True,
            "door_requires_stair": True,
            "door_requires_approach_block_or_contact": True,
            "window_door_clearance_xy_gu": DOOR_CLEARANCE_XY_GU,
            "exterior_only": True,
            "house_shell_prefixes": list(prefixes),
        },
        "attachment_defaults": attachment_defaults,
        "shells": shells,
        "attachments": attachments,
        "stamp_templates": stamp_templates,
        "shell_connections": shell_connections,
        "piece_connections": piece_connections,
        "stats": {
            "shell_count": len(shells),
            "stamp_template_count": len(stamp_templates),
            "eligible_template_count": sum(1 for row in stamp_templates if row["eligible"]),
            "single_shell_stamps": sum(1 for stamp in stamps if len(shells_in_stamp(stamp)) == 1),
            "multi_shell_stamps": sum(1 for stamp in stamps if len(shells_in_stamp(stamp)) > 1),
        },
    }


def _shell_by_id(grammar: Mapping[str, Any], shell_id: str) -> Mapping[str, Any]:
    for shell in grammar["shells"]:
        if shell["shell_id"] == shell_id:
            return shell
    raise KeyError(f"unknown shell_id {shell_id!r}")


def _top_model(models: Sequence[Mapping[str, Any]]) -> str:
    if not models:
        return ""
    return str(models[0]["object_id"])


def _build_model_index(stamp_library: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for stamp in stamp_library["stamps"]:
        for member in stamp["members"]:
            oid = member.get("object_id")
            if not oid or oid in index:
                continue
            index[str(oid)] = {
                "model_key": member.get("model_key"),
                "record_type": member.get("record_type", "STAT"),
                "category": member.get("category", "exterior"),
                "scale": member.get("scale", 1.0),
            }
    return index


def _compose_member(
    *,
    source_id: str,
    object_id: str,
    model_index: Mapping[str, Mapping[str, Any]],
    offset_gu: Sequence[float],
    rotation: Sequence[float],
    is_door: bool = False,
    structural_role: str | None = None,
    outward_heading_deg: float | None = None,
    scale: float | None = None,
) -> dict[str, Any]:
    meta = model_index.get(object_id, {})
    member: dict[str, Any] = {
        "source_id": source_id,
        "object_id": object_id,
        "model_key": meta.get("model_key") or "",
        "record_type": "DOOR" if is_door else str(meta.get("record_type") or "STAT"),
        "category": "door" if is_door else str(meta.get("category") or "exterior"),
        "is_door": is_door,
        "offset_gu": [float(v) for v in offset_gu],
        "rotation": [float(v) for v in rotation],
        "scale": float(scale if scale is not None else meta.get("scale") or 1.0),
        "structural_role": structural_role,
    }
    if outward_heading_deg is not None:
        member["outward_heading_deg"] = outward_heading_deg
    return member


def _world_from_shell_local(
    local_offset: Sequence[float],
    local_rotation: Sequence[float],
    shell_offset: Sequence[float],
    shell_rotz: float,
) -> tuple[list[float], list[float]]:
    # Match the engine's Rz(-shell_rotz) placement convention.
    wx, wy = rotate_xy(local_offset[0], local_offset[1], -shell_rotz)
    world_offset = [wx + shell_offset[0], wy + shell_offset[1], float(local_offset[2]) + float(shell_offset[2])]
    world_rotation = [float(local_rotation[0]), float(local_rotation[1]), float(local_rotation[2]) + shell_rotz]
    return world_offset, world_rotation


def _compose_access_members(
    slot: Mapping[str, Any],
    seed: int,
    *,
    attachment_defaults: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bundles = list(slot.get("access_bundles") or [])
    if not bundles:
        raise ValueError(f"door slot {slot.get('slot_id')} has no access bundles")
    ranked = sorted(bundles, key=lambda bundle: (-_bundle_score(bundle), bundle["source_stamp_id"]))
    chosen = ranked[seed % len(ranked)]
    by_role: dict[str, dict[str, Any]] = {}
    for bundle in ranked:
        for row in bundle["members"]:
            if row["role"] not in by_role:
                by_role[row["role"]] = row
    for row in chosen["members"]:
        if row["role"] == "door":
            by_role["door"] = row
    if "doorframe" not in by_role and attachment_defaults:
        door_row = by_role.get("door")
        if door_row is not None:
            by_role["doorframe"] = {
                "role": "doorframe",
                "object_id": attachment_defaults.get("doorframe_object_id"),
                "model_key": attachment_defaults.get("doorframe_model_key"),
                "record_type": "STAT",
                "category": "exterior",
                "scale": 1.0,
                "offset_rel_door_gu": list(attachment_defaults.get("doorframe_offset_rel_door_gu") or [0, 0, 0]),
                "rotation": list(door_row["rotation"]),
            }
    return list(by_role.values()), list(chosen.get("windows") or [])


def _pick_bundle(slot: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    bundles = list(slot.get("access_bundles") or [])
    if not bundles:
        raise ValueError(f"door slot {slot.get('slot_id')} has no access bundles")
    ranked = sorted(bundles, key=lambda bundle: (-_bundle_score(bundle), bundle["source_stamp_id"]))
    return ranked[seed % len(ranked)]


def _pick_access_bundle(
    slot: Mapping[str, Any],
    seed: int,
    *,
    outward_heading: float | None = None,
) -> Mapping[str, Any]:
    bundles = list(slot.get("access_bundles") or [])
    if not bundles:
        raise ValueError(f"door slot {slot.get('slot_id')} has no access bundles")
    ranked = sorted(bundles, key=lambda bundle: (-_bundle_score(bundle), bundle["source_stamp_id"]))
    if outward_heading is not None:
        door_rotz = _bundle_door_rotz(ranked[0])
        aligned = [
            bundle
            for bundle in ranked
            if _bundle_stairs_usable(_bundle_stair_rows(bundle), door_rotz)
        ]
        if aligned:
            return aligned[seed % len(aligned)]
    return ranked[seed % len(ranked)]


def _door_object_for_slot(slot: Mapping[str, Any], bundle: Mapping[str, Any]) -> str:
    if slot.get("door_models"):
        return _top_model(slot["door_models"])
    for row in bundle["members"]:
        if row["role"] == "door":
            return str(row["object_id"])
    return ""


def _should_use_stairs(
    slot: Mapping[str, Any],
    policy: Mapping[str, str],
    access_defaults: Mapping[str, Any],
    *,
    size_class: str,
) -> bool:
    mode = policy["stairs"]
    if mode == "on":
        return True
    if mode == "off":
        return False
    threshold = float(access_defaults.get("elevation_stair_threshold_gu", ELEVATION_STAIR_THRESHOLD_GU))
    slot_z = float(slot["offset_gu"][2])
    if slot_z >= threshold:
        return True
    if size_class in {"large", "medium"} and slot_z > 0.0:
        return True
    return False


def _terrain_step_height(
    *,
    use_stairs: bool,
    policy: Mapping[str, str],
    slot: Mapping[str, Any],
) -> float:
    if use_stairs:
        return 96.0
    terrain_mode = policy["terrain_at_door"]
    if terrain_mode == "on":
        return 0.0
    if terrain_mode == "off":
        return max(0.0, float(slot["offset_gu"][2]))
    return 0.0 if float(slot["offset_gu"][2]) <= ELEVATION_STAIR_THRESHOLD_GU else float(slot["offset_gu"][2])


def _bundle_stair_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in bundle.get("members") or [] if row.get("role") == "stair"]


def _bundle_block_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in bundle.get("members") or [] if row.get("role") == "approach_block"]
    return [row for row in rows if float(row["offset_rel_door_gu"][2]) <= -200.0]


def _bundle_stairs_usable(stair_rows: Sequence[Mapping[str, Any]], door_rotz: float) -> bool:
    if not stair_rows:
        return False
    for row in stair_rows:
        rel = row["offset_rel_door_gu"]
        dist = math.hypot(float(rel[0]), float(rel[1]))
        if dist > 360.0:
            return False
        _lx, ly = stamp_to_door_local(float(rel[0]), float(rel[1]), door_rotz)
        if -ly < 48.0:
            return False
    return True


def _default_bundle_for_door(door: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_stamp_id": None,
        "members": [
            {
                "role": "door",
                "object_id": door.get("object_id"),
                "model_key": door.get("model_key"),
                "record_type": "DOOR",
                "category": "door",
                "scale": float(door.get("scale") or 1.0),
                "offset_rel_door_gu": [0.0, 0.0, 0.0],
                "rotation": [float(v) for v in (door.get("rotation") or [0.0, 0.0, 0.0])],
            }
        ],
        "windows": [],
    }


def _nearest_grammar_slot(
    grammar: Mapping[str, Any],
    door: Mapping[str, Any],
    shells: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    house_shells = [row for row in shells if row.get("structural_role") == "shell"]
    if not house_shells:
        return None
    dx = float(door["offset_gu"][0])
    dy = float(door["offset_gu"][1])
    nearest_shell = min(
        house_shells,
        key=lambda shell: math.hypot(
            float(shell["offset_gu"][0]) - dx,
            float(shell["offset_gu"][1]) - dy,
        ),
    )
    try:
        spec = _shell_by_id(grammar, shell_key(nearest_shell))
    except KeyError:
        return None
    slots = list(spec.get("door_slots") or [])
    if not slots:
        return None
    local, _rotation = to_shell_local(
        door["offset_gu"],
        door.get("rotation") or [0.0, 0.0, 0.0],
        nearest_shell["offset_gu"],
        rotz_of(nearest_shell),
    )
    return min(
        slots,
        key=lambda slot: math.hypot(
            float(slot["offset_gu"][0]) - local[0],
            float(slot["offset_gu"][1]) - local[1],
        ),
    )


def _slot_from_existing_door(door: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "slot_id": "template_door",
        "offset_gu": [float(v) for v in door["offset_gu"]],
        "rotation": [float(v) for v in (door.get("rotation") or [0.0, 0.0, 0.0])],
        "outward_heading_deg": door.get("outward_heading_deg"),
        "door_models": [{"object_id": door.get("object_id"), "count": 1}],
        "access_bundles": [_default_bundle_for_door(door)],
    }


def _synthesize_door_access_rows(
    slot: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    access_defaults: Mapping[str, Any],
    policy: Mapping[str, str],
    size_class: str,
    door_override: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if door_override is not None:
        door_object = str(door_override.get("object_id") or _door_object_for_slot(slot, bundle))
        door_rotation = [float(v) for v in (door_override.get("rotation") or slot["rotation"])]
        outward_heading = float(
            door_override.get("outward_heading_deg") or slot.get("outward_heading_deg") or 0.0
        )
        door_model_key = door_override.get("model_key")
    else:
        door_object = _door_object_for_slot(slot, bundle)
        door_rotation = [float(v) for v in slot["rotation"]]
        outward_heading = float(slot.get("outward_heading_deg") or 0.0)
        door_model_key = None
    rows: list[dict[str, Any]] = [
        {
            "role": "door",
            "object_id": door_object,
            "model_key": None,
            "record_type": "DOOR",
            "category": "door",
            "scale": 1.0,
            "offset_rel_door_gu": [0.0, 0.0, 0.0],
            "rotation": door_rotation,
            "outward_heading_deg": outward_heading,
        }
    ]
    is_double = is_double_door_object(door_object)
    ref_rotz = _bundle_door_rotz(bundle)
    door_rotz = float(door_rotation[2])
    frame_row = next((row for row in bundle.get("members") or [] if row.get("role") == "doorframe"), None)
    if is_double:
        default_frame = access_defaults.get("double_doorframe_object_id") or access_defaults.get("doorframe_object_id")
    else:
        default_frame = access_defaults.get("doorframe_object_id")
    if frame_row is not None:
        frame_object = str(frame_row.get("object_id") or default_frame)
        if is_double and not is_double_door_object(frame_object):
            frame_object = str(default_frame or frame_object)
        if not is_double and is_double_door_object(str(frame_object or "")):
            frame_object = str(access_defaults.get("doorframe_object_id") or frame_object)
        frame_offset = _transform_rel_door_gu(frame_row["offset_rel_door_gu"], ref_rotz, door_rotz)
        frame_rotz = _stair_rotz_for_door(float((frame_row.get("rotation") or door_rotation)[2]), ref_rotz, door_rotz)
    else:
        frame_object = default_frame
        frame_offset = list(access_defaults.get("doorframe_offset_rel_door_gu") or [0, 0, 0])
        frame_rotz = door_rotz
    if not frame_object:
        raise ValueError("mined grammar has no doorframe object evidence")
    rows.append(
        {
            "role": "doorframe",
            "object_id": frame_object,
            "model_key": None,
            "record_type": "STAT",
            "category": "exterior",
            "scale": 1.0,
            "offset_rel_door_gu": frame_offset,
            "rotation": [0.0, 0.0, frame_rotz],
        }
    )
    if not _should_use_stairs(slot, policy, access_defaults, size_class=size_class):
        return rows
    layout = _stair_layout_from_bundle(bundle, door_rotz, access_defaults=access_defaults)
    if layout is None:
        stair_object = access_defaults.get(
            "double_stair_object_id" if is_double else "stair_object_id"
        )
        stair_model = access_defaults.get("stair_model_key")
        if not stair_object or not stair_model:
            raise ValueError("mined grammar has no stair evidence for a stair-required door")
        layout = {
            "approach_gu": float(access_defaults.get("stair_approach_gu") or DEFAULT_STAIR_APPROACH_GU),
            "lateral_gu": float(access_defaults.get("double_stair_lateral_gu") or DEFAULT_DOUBLE_STAIR_LATERAL_GU),
            "stair_z_gu": float(access_defaults.get("stair_z_rel_gu") or -118.0),
            "stair_rotz": door_rotz,
            "object_id": stair_object,
            "model_key": stair_model,
            "scale": float(access_defaults.get("double_stair_scale" if is_double else "stair_scale") or 0.88),
        }
    if not door_model_key:
        for row in bundle.get("members") or []:
            if row.get("role") == "door" and row.get("model_key"):
                door_model_key = str(row["model_key"])
                break
    if not door_model_key:
        door_model_key = access_defaults.get("double_door_model_key") or access_defaults.get("door_model_key")
    frame_model_key = access_defaults.get("doorframe_model_key")
    for row in bundle.get("members") or []:
        if row.get("role") == "doorframe" and row.get("model_key"):
            frame_model_key = str(row["model_key"])
            break
    _append_stairs_door_local(
        rows,
        door_rotz=door_rotz,
        is_double=is_double,
        layout=layout,
        access_defaults=access_defaults,
        door_model_key=str(door_model_key) if door_model_key else None,
        doorframe_model_key=str(frame_model_key) if frame_model_key else None,
        doorframe_offset_rel_door_gu=[float(v) for v in frame_offset],
        doorframe_rotz=frame_rotz,
    )
    return rows


def _build_door_access_rows(
    slot: Mapping[str, Any],
    seed: int,
    *,
    access_defaults: Mapping[str, Any],
    policy: Mapping[str, str],
    size_class: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bundle = _pick_bundle(slot, seed)
    if policy["access_placement"] == "synthetic":
        rows = _synthesize_door_access_rows(
            slot,
            bundle,
            access_defaults=access_defaults,
            policy=policy,
            size_class=size_class,
        )
        return rows, _filter_windows_to_facades(list(bundle.get("windows") or []), None)
    rows, windows = _compose_access_members(slot, seed, attachment_defaults=access_defaults)
    return rows, windows


def _validate_access_bundle(bundle: Mapping[str, Any], *, require_stair: bool = True) -> None:
    roles = {row["role"] for row in bundle["members"]}
    if "door" not in roles:
        raise ValueError("access bundle missing door")
    if "doorframe" not in roles:
        raise ValueError("access bundle missing doorframe")
    if require_stair and "stair" not in roles:
        raise ValueError("access bundle missing stair")


def _validate_access_rows(rows: Sequence[Mapping[str, Any]], *, require_stair: bool) -> None:
    _validate_access_bundle({"members": list(rows)}, require_stair=require_stair)


def _place_composed_access(
    *,
    access_rows: Sequence[Mapping[str, Any]],
    door_world: Sequence[float],
    next_id,
    model_index: Mapping[str, Mapping[str, Any]],
    members: list[dict[str, Any]],
) -> dict[str, Any] | None:
    anchor_door = None
    for row in access_rows:
        world_offset = [
            float(door_world[0]) + float(row["offset_rel_door_gu"][0]),
            float(door_world[1]) + float(row["offset_rel_door_gu"][1]),
            float(door_world[2]) + float(row["offset_rel_door_gu"][2]),
        ]
        is_door = row["role"] == "door"
        member = _compose_member(
            source_id=next_id("door" if is_door else row["role"]),
            object_id=str(row["object_id"]),
            model_index=model_index,
            offset_gu=world_offset,
            rotation=row["rotation"],
            is_door=is_door,
            structural_role=None,
            outward_heading_deg=row.get("outward_heading_deg"),
            scale=float(row.get("scale") or 1.0),
        )
        members.append(member)
        if is_door and anchor_door is None:
            anchor_door = member
    return anchor_door


def generate_from_stamp_template(
    stamp_library: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    generated_id: str | None = None,
    seed: int = 0,
    grammar: Mapping[str, Any] | None = None,
    access_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not template.get("eligible"):
        raise ValueError(
            f"stamp template {template['template_id']!r} is ineligible: {template.get('ineligible_reason')}"
        )
    source = next(
        stamp for stamp in stamp_library["stamps"] if stamp["stamp_id"] == template["source_stamp_id"]
    )
    doors = [member for member in source["members"] if member.get("is_door")]
    if not doors:
        raise ValueError(f"template source stamp {template['source_stamp_id']!r} has no door")
    anchor = doors[seed % len(doors)]["offset_gu"]
    replace_access = grammar is not None
    members: list[dict[str, Any]] = []
    for index, member in enumerate(source["members"]):
        if not _template_member_kept(member):
            continue
        cls = classify_attachment(member)
        if replace_access and cls in {"stair", "doorframe"}:
            continue
        copied = dict(member)
        copied["source_id"] = f"gen_tpl_{index:04d}"
        copied["offset_gu"] = [
            float(member["offset_gu"][0]) - float(anchor[0]),
            float(member["offset_gu"][1]) - float(anchor[1]),
            float(member["offset_gu"][2]) - float(anchor[2]),
        ]
        members.append(copied)
    if replace_access:
        model_index = _build_model_index(stamp_library)
        policy = normalize_access_policy(access_policy)
        access_defaults = grammar.get("attachment_defaults") or {}
        access_index = 0

        def next_access_id(prefix: str) -> str:
            nonlocal access_index
            access_index += 1
            return f"gen_tpl_{prefix}_{access_index:04d}"

        if policy["stairs"] == "auto":
            policy = dict(policy)
            policy["stairs"] = "on"
        for door in [row for row in members if row.get("is_door")]:
            slot = _nearest_grammar_slot(grammar, door, members) or _slot_from_existing_door(door)
            if not slot.get("access_bundles"):
                slot = _slot_from_existing_door(door)
            bundle = _pick_bundle(slot, seed)
            access_rows = _synthesize_door_access_rows(
                slot,
                bundle,
                access_defaults=access_defaults,
                policy=policy,
                size_class="large",
                door_override=door,
            )
            _place_composed_access(
                access_rows=[row for row in access_rows if row.get("role") != "door"],
                door_world=door["offset_gu"],
                next_id=next_access_id,
                model_index=model_index,
                members=members,
            )
    door_count = sum(1 for member in members if member.get("is_door"))
    xs = [member["offset_gu"][0] for member in members]
    ys = [member["offset_gu"][1] for member in members]
    zs = [member["offset_gu"][2] for member in members]
    bounds = {
        "min": [min(xs), min(ys), min(zs)],
        "max": [max(xs), max(ys), max(zs)],
        "span": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
    }
    return {
        "stamp_id": generated_id or f"generated__{template['template_id']}__{seed:04d}",
        "source": {
            "kind": "generated_from_stamp_template",
            "grammar_id": None,
            "template_id": template["template_id"],
            "source_stamp_id": template["source_stamp_id"],
            "seed": seed,
        },
        "building_type": "house",
        "size_class": "large",
        "door_count": door_count,
        "multi_shell": True,
        "anchor": {"kind": "seed_door", "source_position_gu": [0.0, 0.0, 0.0]},
        "access_heading_rad": 0.0,
        "members": sorted(members, key=lambda member: (0 if member.get("is_door") else 1, member["source_id"])),
        "footprint": {"aabb_rel": bounds, "hull_xy_rel": []},
        "bounds_rel_gu": bounds,
        "terrain_envelope": {
            "door_step_heights_gu": [96.0],
            "footprint_relief_gu": 0.0,
            "footprint_slope_deg": 0.0,
            "burial_depth_gu": 0.0,
        },
    }


def generate_house(
    grammar: Mapping[str, Any],
    stamp_library: Mapping[str, Any],
    *,
    shell_id: str,
    door_slot_ids: Sequence[str] | None = None,
    include_windows: bool = True,
    include_chimney: bool = True,
    stamp_template_id: str | None = None,
    block_pattern_id: str | None = None,
    generated_id: str | None = None,
    seed: int = 0,
    access_policy: Mapping[str, Any] | None = None,
    window_facade_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if stamp_template_id is not None or block_pattern_id is not None:
        template_id = stamp_template_id or block_pattern_id.replace("block_", "template_")
        template = next(row for row in grammar["stamp_templates"] if row["template_id"] == template_id)
        stamp = generate_from_stamp_template(
            stamp_library,
            template,
            generated_id=generated_id,
            seed=seed,
            grammar=grammar,
            access_policy=access_policy,
        )
        stamp["source"]["grammar_id"] = grammar["grammar_id"]
        stamp["source"]["access_policy"] = normalize_access_policy(access_policy)
        return stamp

    model_index = _build_model_index(stamp_library)
    members: list[dict[str, Any]] = []
    member_index = 0

    def next_id(prefix: str) -> str:
        nonlocal member_index
        member_index += 1
        return f"gen_{prefix}_{member_index:04d}"

    shell_spec = _shell_by_id(grammar, shell_id)
    shell_offset = [0.0, 0.0, 0.0]
    shell_rotz = 0.0
    members.append(
        _compose_member(
            source_id=next_id("shell"),
            object_id=str(shell_spec["object_id"]),
            model_index=model_index,
            offset_gu=shell_offset,
            rotation=[0.0, 0.0, shell_rotz],
            structural_role="shell",
        )
    )
    slots = shell_spec.get("door_slots") or []
    if not slots:
        raise ValueError(f"shell {shell_id!r} has no mined door slots")
    if door_slot_ids:
        lookup = {slot["slot_id"]: slot for slot in slots}
        selected_slots = [lookup[slot_id] for slot_id in door_slot_ids]
    else:
        selected_slots = [max(slots, key=lambda slot: max((_bundle_score(b) for b in slot.get("access_bundles") or []), default=-999.0))]

    attachment_defaults = grammar.get("attachment_defaults") or {}
    policy = normalize_access_policy(access_policy)
    anchor_door = None
    windows_for_decor: list[dict[str, Any]] = []
    bundle_for_decor_source: str | None = None
    door_step_heights: list[float] = []
    for slot_index, slot in enumerate(selected_slots):
        bundle = _pick_bundle(slot, seed + slot_index)
        access_rows, window_rows = _build_door_access_rows(
            slot,
            seed + slot_index,
            access_defaults=attachment_defaults,
            policy=policy,
            size_class=str(shell_spec.get("size_class") or "unknown"),
        )
        use_stairs = _should_use_stairs(
            slot,
            policy,
            attachment_defaults,
            size_class=str(shell_spec.get("size_class") or "unknown"),
        )
        require_stair = slot_index == 0 and use_stairs
        _validate_access_rows(access_rows, require_stair=require_stair)
        door_step_heights.append(
            _terrain_step_height(use_stairs=use_stairs, policy=policy, slot=slot)
        )
        door_world, _door_rot = _world_from_shell_local(
            slot["offset_gu"], slot["rotation"], shell_offset, shell_rotz
        )
        door_row = next((row for row in access_rows if row.get("role") == "door"), None)
        if door_row and is_double_door_object(str(door_row.get("object_id") or "")):
            nudge_x, nudge_y = door_local_to_stamp(0.0, -DOOR_OUTWARD_NUDGE_GU, float(slot["rotation"][2]))
            door_world = [door_world[0] + nudge_x, door_world[1] + nudge_y, door_world[2]]
        placed = _place_composed_access(
            access_rows=access_rows,
            door_world=door_world,
            next_id=next_id,
            model_index=model_index,
            members=members,
        )
        if anchor_door is None:
            anchor_door = placed
            windows_for_decor = _filter_windows_to_facades(window_rows, window_facade_ids)
            if shell_spec.get("window_facades"):
                bands = {
                    str(row["facade_id"]): float(row["z_gu"])
                    for row in shell_spec["window_facades"]
                }
                snapped: list[dict[str, Any]] = []
                for window in windows_for_decor:
                    copied = dict(window)
                    offset = copied["offset_gu"]
                    facade_id = _window_facade_id(float(offset[0]), float(offset[1]))
                    if facade_id in bands:
                        copied["offset_gu"] = [float(offset[0]), float(offset[1]), round(bands[facade_id], 3)]
                    snapped.append(copied)
                windows_for_decor = snapped
            bundle_for_decor_source = bundle.get("source_stamp_id")

    if include_windows and windows_for_decor:
        for window in windows_for_decor:
            world_offset, world_rotation = _world_from_shell_local(
                window["offset_gu"], window["rotation"], shell_offset, shell_rotz
            )
            members.append(
                _compose_member(
                    source_id=next_id("window"),
                    object_id=str(window["object_id"]),
                    model_index=model_index,
                    offset_gu=world_offset,
                    rotation=world_rotation,
                    scale=float(window.get("scale") or 1.0),
                )
            )
    if include_chimney and shell_spec.get("chimney_slots"):
        slot = shell_spec["chimney_slots"][0]
        chimney_object = _top_model(slot.get("chimney_models") or [])
        if chimney_object:
            world_offset, world_rotation = _world_from_shell_local(
                slot["offset_gu"], slot["rotation"], shell_offset, shell_rotz
            )
            members.append(
                _compose_member(
                    source_id=next_id("chimney"),
                    object_id=chimney_object,
                    model_index=model_index,
                    offset_gu=world_offset,
                    rotation=world_rotation,
                )
            )
    if policy.get("decorations", "source") != "none":
        for decoration in shell_spec.get("decorations") or []:
            world_offset, world_rotation = _world_from_shell_local(
                decoration["offset_gu"], decoration["rotation"], shell_offset, shell_rotz
            )
            members.append(
                _compose_member(
                    source_id=next_id(str(decoration.get("structural_role") or "decoration")),
                    object_id=str(decoration.get("object_id") or ""),
                    model_index=model_index,
                    offset_gu=world_offset,
                    rotation=world_rotation,
                    scale=float(decoration.get("scale") or 1.0),
                    structural_role=str(decoration.get("structural_role") or "decoration"),
                )
            )
    if anchor_door is None:
        raise ValueError("generated house has no door anchor")

    anchor = anchor_door["offset_gu"]
    anchored_members = []
    for member in members:
        anchored = dict(member)
        anchored["offset_gu"] = [
            float(member["offset_gu"][0]) - anchor[0],
            float(member["offset_gu"][1]) - anchor[1],
            float(member["offset_gu"][2]) - anchor[2],
        ]
        anchored_members.append(anchored)
    door_count = sum(1 for member in anchored_members if member.get("is_door"))
    xs = [member["offset_gu"][0] for member in anchored_members]
    ys = [member["offset_gu"][1] for member in anchored_members]
    zs = [member["offset_gu"][2] for member in anchored_members]
    bounds = {
        "min": [min(xs), min(ys), min(zs)],
        "max": [max(xs), max(ys), max(zs)],
        "span": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
    }
    stamp_id = generated_id or f"generated__{shell_id}__{seed:04d}"
    return {
        "stamp_id": stamp_id,
        "source": {
            "kind": "generated_from_grammar",
            "grammar_id": grammar["grammar_id"],
            "shell_id": shell_id,
            "access_bundle_source": bundle_for_decor_source if anchor_door else None,
            "access_policy": policy,
            "seed": seed,
        },
        "building_type": "house",
        "size_class": shell_spec.get("size_class", "unknown"),
        "door_count": door_count,
        "multi_shell": False,
        "anchor": {"kind": "seed_door", "source_position_gu": [0.0, 0.0, 0.0]},
        "access_heading_rad": 0.0,
        "members": sorted(
            anchored_members,
            key=lambda member: (0 if member.get("is_door") else 1, member["source_id"]),
        ),
        "footprint": {"aabb_rel": bounds, "hull_xy_rel": []},
        "bounds_rel_gu": bounds,
        "terrain_envelope": {
            "door_step_heights_gu": door_step_heights or [96.0],
            "footprint_relief_gu": 0.0,
            "footprint_slope_deg": 0.0,
            "burial_depth_gu": 0.0,
        },
    }


SERIALIZED_EULER_DIGITS = 9
IMPORT_SPEC = {
    "ignore_animations": True,
    "ignore_collision_nodes": True,
    "normalize_to_position": False,
    "reuse_meshes": True,
    "scale_correction": 0.01,
    "use_existing_materials": True,
    "vertex_precision": 0.001,
}


def stamp_to_sheet_scene(stamp: Mapping[str, Any], *, scene_name: str | None = None) -> dict[str, Any]:
    meshes: list[dict[str, Any]] = []
    for member in stamp["members"]:
        position_gu = member["offset_gu"]
        rotation = member.get("rotation") or [0.0, 0.0, 0.0]
        scale = 1.0 if member.get("scale") is None else float(member["scale"])
        serialized_rotation = [
            round(value, SERIALIZED_EULER_DIGITS)
            for value in engine_transform.blender_xyz_euler_for_tes3_rotation(rotation)
        ]
        meshes.append(
            {
                "id": member["source_id"],
                "mesh": member["model_key"],
                "model_key": member["model_key"],
                "position": [round(0.01 * float(value), 6) for value in position_gu],
                "rotation": serialized_rotation,
                "scale": round(scale, 6),
                "source_id": member["source_id"],
                "source_object_id": member.get("object_id"),
                "source_record_type": member.get("record_type"),
                "source_category": member.get("category"),
            }
        )
    return {
        "schema_version": 1,
        "scene_name": scene_name or f"generated_{stamp['stamp_id']}_sheet",
        "mesh_path_semantics": (
            "TES3 relative MODL path; resolved case-insensitively by "
            "procgen.meshcheck against configured data roots (read-only)"
        ),
        "source": {
            "kind": "generated_stamp",
            "stamp_id": stamp["stamp_id"],
            "member_count": len(meshes),
        },
        # Generated-house sheets need the maximum approved framing margin;
        # chimneys and profile-driven offsets otherwise sit against the tile
        # edge even when the mesh bounds are technically included.
        "import": {**IMPORT_SPEC, "margin": 1.60},
        "terrain": {"enabled": False},
        "meshes": meshes,
    }
