"""Deterministic procedural groundcover generation for TES3 exterior cells.

Pipeline position
------------------
This is the generation-core stage of the procedural groundcover pipeline.  It
consumes an FGM-style palette (parsed by ``procgen.groundcover_ini``), the
read-only scratch ``tamriel.esm`` LAND/VTEX data (via ``procgen.espland``), an
accepted scatter document (used as static-exclusion input), and emits one
standalone ref document JSON that mirrors the scatter JSON ref schema.  A
companion author step converts that document into a TES3 plugin (ESP) through
``procgen.tes3json`` + tes3conv.  Like the scatter generator, this module
deliberately does not author CELL/FRMR records itself.

Placement model (mirrors the FGM/MWGroundcoverGenerator approach)
-----------------------------------------------------------------
For every cell in the target bounds a lattice of candidate points is built at
the section's ``iGap`` spacing (FGM uses 100-105 game units).  Each candidate
is jittered, then gated in order:

1. terrain height available (``procgen.espland.height_at_game_position``),
2. water: terrain below the 0-THU water plane is rejected,
3. ``fMinHeight`` / ``fMaxHeight`` elevation window,
4. ``fMaximumAngle`` slope cap (max of 8 neighbour gradients at 128 GU),
5. water is rejected unless the matched texture is explicitly listed in the
   run's ``water_flora_texture_ids``; raw VTEX 0 (base/unpainted tiles) is
   rejected,
6. the tile's LTEX record id must match a palette section for the run region,
7. texture bans: any ban substring in the LTEX record id or texture path, and
   any configured road-path regex against the texture path; bans with a
   nonzero offset additionally probe neighbour tiles at that distance,
8. static avoidance: the candidate must clear the exclusion index built from
   the accepted scatter refs (token-radius table, bbox-aware for rocks) and
   any settlement circles.

Accepted candidates receive: a terrain-normal tilt solved for OpenMW's ref
rotation composition ``Rx(-rx) @ Ry(-ry) @ Rz(-rz)`` (negated axes, derived
from ``components/misc/convert.hpp::makeOsgQuat`` lines 50-54 and
OpenSceneGraph's reversed ``osg::Quat::operator*`` operand semantics; the
rightmost Z rotation applies first).  The workspace ``_rotate_xyz`` helper
is a separate bbox utility and is not used by this normal solve.  A uniform
random yaw, a uniform scale in
``fSclMin``..``fSclMax``, and a weighted-random mesh from the section table
(``sChance`` weights).  The tilt is exact for any yaw, so a random spin never
lifts the quad off the terrain.  A prior FGM alignment statistic (up · normal
p50 = 0.983) was measured under an earlier convention and is not asserted as
a measurement for this implementation.
The position z is terrain + ``iZPositionModifier``.

Determinism
-----------
Every RNG is seeded with ``procgen.seeds.derive_seed(master_seed,
"groundcover", area, version, "cell", gx, gy)``; cell iteration order and
ref order are fixed, and the output document is written with sorted keys.

Output document schema (schema_version 1)
-----------------------------------------
Top-level keys: ``schema_version, tool, tool_version, seed, determinism,
scope, units, terrain, palette, masks, exclusion, density, placement_stats,
generation_failures``.  ``density.cells[]`` holds per-cell ``{grid, refs,
stats}``; each ref carries ``ref_id, cell, mesh, category="groundcover",
pass="C", section_texture, section_region, position_gu, rotation_radians,
rotation_mode, scale, terrain, lattice``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

from procgen.espland import (
    THU_TO_GU,
    CELL_SIZE_GAME_UNITS,
    LandRecord,
    LandscapeTexture,
    height_at_game_position,
    load_land,
    load_ltex,
)
from procgen.groundcover_ini import (
    GroundcoverIni,
    MeshOption,
    TextureBan,
    TextureSection,
)
from procgen.scatter_analysis import terrain_slope_deg, texture_at_position
from procgen.seeds import derive_seed
from .clearing_index import ClearingIndex, MultiClearingIndex, build_clearing_index
from .region_scope import region_cells

TOOL_NAME = "procgen.groundcover_generate"
TOOL_VERSION = "1.0.0"
SCHEMA_VERSION = 1
ROAD_REJECT = "road_texture"
DEFAULT_EXCLUSION_MARGIN_GU = 32.0

# ---------------------------------------------------------------------------
# Palette specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionSpec:
    """One active palette entry: placement behaviour for one ground texture."""

    texture_id: str
    region: str
    gap_gu: int
    scale_min: float
    scale_max: float
    jitter_min_gu: float
    jitter_max_gu: float
    max_angle_deg: float
    min_height_gu: float
    max_height_gu: float | None
    align_to_normal: bool
    bans: tuple[TextureBan, ...]
    options: tuple[MeshOption, ...]

    def describe(self) -> dict[str, Any]:
        return {
            "texture_id": self.texture_id,
            "region": self.region,
            "gap_gu": self.gap_gu,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "jitter_min_gu": self.jitter_min_gu,
            "jitter_max_gu": self.jitter_max_gu,
            "max_angle_deg": self.max_angle_deg,
            "min_height_gu": self.min_height_gu,
            "max_height_gu": self.max_height_gu,
            "align_to_normal": self.align_to_normal,
            "bans": [(ban.texture, ban.offset_gu) for ban in self.bans],
            "mesh_table": [[option.mesh, option.chance] for option in self.options],
        }


def build_palette(
    ini: GroundcoverIni,
    region: str,
    *,
    extra_banned_textures: Sequence[str] = (),
    jitter_gu: float | None = None,
) -> dict[str, SectionSpec]:
    """Assemble the active palette for one region from a parsed FGM INI.

    Only sections with ``bPlaceGrass=1`` whose region qualifier equals
    ``region`` participate.  ``extra_banned_textures`` are appended to every
    section's ban list (used for texture exclusions outside the FGM lists).
    ``jitter_gu`` overrides the per-section ``fPosMin``/``fPosMax`` jitter
    symmetrically.
    """

    palette: dict[str, SectionSpec] = {}
    extra_bans = tuple(TextureBan(name) for name in extra_banned_textures)
    for section in ini.sections_for_region(region):
        if not section.place_grass:
            continue
        if section.texture_id in palette:
            raise ValueError(
                f"duplicate palette section for texture {section.texture_id!r} in region {region!r}"
            )
        if jitter_gu is not None:
            jitter_min = -abs(float(jitter_gu))
            jitter_max = abs(float(jitter_gu))
        else:
            jitter_min = section.pos_min_gu
            jitter_max = section.pos_max_gu
        palette[section.texture_id] = SectionSpec(
            texture_id=section.texture_id,
            region=section.region,
            gap_gu=section.gap_gu,
            scale_min=section.scale_min,
            scale_max=section.scale_max,
            jitter_min_gu=jitter_min,
            jitter_max_gu=jitter_max,
            max_angle_deg=section.max_angle_deg,
            min_height_gu=section.min_height_gu,
            max_height_gu=section.max_height_gu,
            align_to_normal=section.align_to_normal,
            bans=section.bans + extra_bans,
            options=section.options,
        )
    return palette


# ---------------------------------------------------------------------------
# Exclusion index (static avoidance)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticExclusionRule:
    """One token-radius rule for classifying scatter refs as exclusions."""

    tokens: tuple[str, ...]
    radius_gu: float
    use_bbox: bool = False
    skip: bool = False


DEFAULT_EXCLUSION_RULES: tuple[StaticExclusionRule, ...] = (
    # Never block: markers, lights, triggers, decorative planks, invisible aids.
    StaticExclusionRule(
        ("marker", "light", "fx_", "trigger", "teleport", "invis", "collis",
         "bridge", "plank", "log", "sound", "door", "thiefdoor", "ward"),
        0.0,
        skip=True,
    ),
    # Low shrubs and undergrowth: small clearance, grass may grow around.
    StaticExclusionRule(("shrub", "bush", "fern", "undergrowth", "_bs_", "_lg_"), 80.0),
    # Trees: trunk-scale clearance (LawnMower uses 120 for tree/flora).
    StaticExclusionRule(("tree", "flora_tr", "tr_"), 120.0),
    # Rocks and cliffs: use the ref's own world bbox when available.
    StaticExclusionRule(("rock", "cliff", "boulder"), 400.0, use_bbox=True),
    # Large structures: fort-scale clearance.
    StaticExclusionRule(
        ("strongh", "pylon", "portal", "entrance", "necrom", "menhir",
         "pillar", "lighthouse", "statue", "tower", "temple", "palace",
         "canton", "striderport"),
        600.0,
    ),
    # Buildings and walls.
    StaticExclusionRule(
        ("house", "building", "ex_", "shack", "ruin", "keep", "fort", "docks",
         "gate", "tomb", "well", "stairs", "steps", "bazaar", "shrine"),
        400.0,
    ),
)


@dataclass(frozen=True)
class ExclusionCircle:
    """A fixed-radius circular exclusion zone in absolute game units."""

    x_gu: float
    y_gu: float
    radius_gu: float
    label: str = ""


class ExclusionIndex:
    """Cell-keyed 2-D exclusion test over scatter refs and circles.

    Scatter refs are indexed by their own cell; queries check the candidate's
    cell plus its eight neighbours (a static near a cell border can block
    groundcover in the adjacent cell).  Settlement circles are tested
    directly against every candidate.
    """

    def __init__(
        self,
        scatter_refs: Sequence[Mapping[str, Any]] = (),
        rules: Sequence[StaticExclusionRule] = DEFAULT_EXCLUSION_RULES,
        *,
        default_radius_gu: float = 150.0,
        margin_gu: float = DEFAULT_EXCLUSION_MARGIN_GU,
    ) -> None:
        self._rules = list(rules)
        self._default_radius_gu = float(default_radius_gu)
        self._margin_gu = float(margin_gu)
        self._by_cell: dict[tuple[int, int], list[tuple[float, float, float, str]]] = {}
        self._circles: list[ExclusionCircle] = []
        self._ref_count = 0
        for ref in scatter_refs:
            self.add_ref(ref)

    def add_ref(self, ref: Mapping[str, Any]) -> None:
        cell = ref.get("cell")
        position = ref.get("position_gu")
        if not isinstance(cell, Sequence) or len(cell) != 2 or not isinstance(position, Sequence) or len(position) < 2:
            return
        radius = self._radius_for(str(ref.get("mesh", "")), ref.get("bbox"))
        if radius <= 0.0:
            return
        grid = (int(cell[0]), int(cell[1]))
        self._by_cell.setdefault(grid, []).append(
            (float(position[0]), float(position[1]), radius, str(ref.get("ref_id", "?")))
        )
        self._ref_count += 1

    def add_circle(self, circle: ExclusionCircle) -> None:
        self._circles.append(circle)

    def circle_list(self) -> list[ExclusionCircle]:
        return list(self._circles)

    def _radius_for(self, mesh: str, bbox: Mapping[str, Any] | None) -> float:
        lowered = str(mesh).lower()
        for rule in self._rules:
            if any(token in lowered for token in rule.tokens):
                if rule.skip:
                    return 0.0
                if rule.use_bbox and isinstance(bbox, Mapping):
                    world = bbox.get("world_aabb_gu")
                    if isinstance(world, Mapping):
                        minimum = world.get("min")
                        maximum = world.get("max")
                        if (
                            isinstance(minimum, Sequence)
                            and isinstance(maximum, Sequence)
                            and len(minimum) >= 2
                            and len(maximum) >= 2
                        ):
                            half = 0.5 * max(
                                float(maximum[0]) - float(minimum[0]),
                                float(maximum[1]) - float(minimum[1]),
                            )
                            return half + self._margin_gu
                return rule.radius_gu + self._margin_gu
        return self._default_radius_gu + self._margin_gu

    def _add_ref(self, ref: Mapping[str, Any]) -> None:
        self.add_ref(ref)

    @property
    def ref_count(self) -> int:
        return self._ref_count

    @property
    def circle_count(self) -> int:
        return len(self._circles)

    def circle_list(self) -> list[ExclusionCircle]:
        return list(self._circles)

    def is_blocked(
        self, x_gu: float, y_gu: float, cell: Sequence[int] | None = None
    ) -> tuple[bool, str]:
        if cell is None:
            cell = (math.floor(x_gu / CELL_SIZE_GAME_UNITS), math.floor(y_gu / CELL_SIZE_GAME_UNITS))
        gx, gy = int(cell[0]), int(cell[1])
        for neighbor_x in (gx - 1, gx, gx + 1):
            for neighbor_y in (gy - 1, gy, gy + 1):
                for ex, ey, radius, label in self._by_cell.get((neighbor_x, neighbor_y), ()):
                    if radius <= 0.0:
                        continue
                    if (x_gu - ex) ** 2 + (y_gu - ey) ** 2 <= radius * radius:
                        return True, f"static:{label}"
        for circle in self._circles:
            if (x_gu - circle.x_gu) ** 2 + (y_gu - circle.y_gu) ** 2 <= circle.radius_gu ** 2:
                return True, f"settlement:{circle.label}"
        return False, ""


def load_settlement_circles(
    settlements_json: str | Path,
    bounds: Sequence[Sequence[int]],
    radius_gu: float,
) -> list[ExclusionCircle]:
    """Read settlement markers near the target bounds from ``settlements.json``.

    Every marker whose ``game_cell`` lies within the bounds expanded by one
    cell (plus the radius span) becomes a circle centred on its cell.  The
    marker file is the mapdata product produced by ``tools/map_decompose.py``
    + ``tools/settlement_names.py``; a marker here means a settlement or
    named place exists on the map, and groundcover should not be placed on it.
    """

    import json

    path = Path(settlements_json)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read settlements JSON {path}: {exc}") from exc
    entries = document.get("settlements") if isinstance(document, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError(f"settlements JSON {path} has no 'settlements' list")

    (min_x, min_y), (max_x, max_y) = bounds
    radius_cells = max(1, int(math.ceil(radius_gu / CELL_SIZE_GAME_UNITS)) + 1)
    circles: list[ExclusionCircle] = []
    for entry in entries:
        cell = entry.get("game_cell")
        if not isinstance(cell, Sequence) or len(cell) != 2:
            continue
        gx, gy = int(cell[0]), int(cell[1])
        if not (min_x - radius_cells <= gx <= max_x + radius_cells):
            continue
        if not (min_y - radius_cells <= gy <= max_y + radius_cells):
            continue
        name = str(entry.get("provisional_name") or entry.get("marker_id") or "settlement")
        circles.append(
            ExclusionCircle(
                gx * CELL_SIZE_GAME_UNITS + CELL_SIZE_GAME_UNITS / 2.0,
                gy * CELL_SIZE_GAME_UNITS + CELL_SIZE_GAME_UNITS / 2.0,
                radius_gu,
                f"{name}@{gx},{gy}",
            )
        )
    return circles


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass
class GroundcoverRunConfig:
    """Everything the generator needs for one deterministic run."""

    area: str
    version: str
    master_seed: int
    bounds: tuple[tuple[int, int], tuple[int, int]]
    region: str
    ini_path: str | Path
    land_plugin: str | Path
    scope_region_map: str | Path | None = None
    scope_region_id: str | None = None
    scope_pixels_per_cell: float = 64.0
    master_name: str = ""
    mesh_roots: tuple[str | Path, ...] = ()
    scatter_exclusions: str | Path | None = None
    settlements_json: str | Path | None = None
    settlement_radius_gu: float = 1400.0
    clearing_json: str | Path | tuple[Path, ...] | None = None
    edited_land_json: str | Path | None = None
    exclusion_rules: tuple[StaticExclusionRule, ...] = DEFAULT_EXCLUSION_RULES
    default_exclusion_radius_gu: float = 150.0
    exclusion_margin_gu: float = DEFAULT_EXCLUSION_MARGIN_GU
    road_texture_regexes: tuple[str, ...] = (
        ".*road.*",
        ".*mainroad.*",
        ".*dirtroad.*",
        ".*gravel.*",
        ".*beatenpath.*",
    )
    road_raw_vtex_values: tuple[int, ...] = ()
    water_flora_texture_ids: tuple[str, ...] = ()
    extra_banned_textures: tuple[str, ...] = ()
    jitter_gu: float | None = None
    z_modifier_gu: float | None = None
    object_prefix: str = "PTGC_"
    slope_spacing_gu: float = 128.0
    normal_spacing_gu: float = 128.0
    water_threshold_thu: int = 0
    limit_cells: int | None = None
    max_seconds: float | None = None


def config_from_mapping(values: Mapping[str, Any]) -> GroundcoverRunConfig:
    """Build a run config from a JSON mapping (validated with clear errors)."""

    def expect(key: str, kind: type) -> Any:
        if key not in values:
            raise ValueError(f"groundcover config: missing required key {key!r}")
        value = values[key]
        if not isinstance(value, kind):
            raise ValueError(f"groundcover config: {key!r} must be {kind.__name__}")
        return value

    area = str(expect("area", str))
    version = str(expect("version", str))
    bounds_raw = expect("bounds", list)
    if len(bounds_raw) != 2 or any(len(pair) != 2 for pair in bounds_raw):
        raise ValueError("groundcover config: 'bounds' must be [[min_x, min_y], [max_x, max_y]]")
    bounds = ((int(bounds_raw[0][0]), int(bounds_raw[0][1])), (int(bounds_raw[1][0]), int(bounds_raw[1][1])))
    if bounds[0][0] > bounds[1][0] or bounds[0][1] > bounds[1][1]:
        raise ValueError("groundcover config: bounds min must not exceed max")

    def path_list(key: str) -> tuple[str, ...]:
        raw = values.get(key, [])
        if not isinstance(raw, list):
            raise ValueError(f"groundcover config: {key!r} must be a list")
        return tuple(str(item) for item in raw)

    def rule_from_mapping(rule: Mapping[str, Any]) -> StaticExclusionRule:
        tokens = rule.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
            raise ValueError("groundcover config: exclusion rule needs a non-empty 'tokens' list")
        return StaticExclusionRule(
            tokens=tuple(tokens),
            radius_gu=float(rule.get("radius_gu", 0.0)),
            use_bbox=bool(rule.get("use_bbox", False)),
            skip=bool(rule.get("skip", False)),
        )

    rules_raw = values.get("exclusion_rules")
    rules: tuple[StaticExclusionRule, ...] = DEFAULT_EXCLUSION_RULES
    if rules_raw is not None:
        if not isinstance(rules_raw, list):
            raise ValueError("groundcover config: 'exclusion_rules' must be a list")
        rules = tuple(rule_from_mapping(rule) for rule in rules_raw)

    scope = values.get("scope")
    scope_region_map = None
    scope_region_id = None
    scope_pixels_per_cell = 64.0
    if scope is not None:
        if not isinstance(scope, Mapping):
            raise ValueError("groundcover config: 'scope' must be an object")
        if not isinstance(scope.get("region_map"), str) or not isinstance(scope.get("region_id"), str):
            raise ValueError("groundcover config: scope requires region_map and region_id")
        scope_region_map = str(scope["region_map"])
        scope_region_id = str(scope["region_id"])
        scope_pixels_per_cell = float(scope.get("map_pixels_per_cell", 64.0))
    seed = expect("master_seed", int)
    return GroundcoverRunConfig(
        area=area,
        version=version,
        master_seed=int(seed),
        bounds=bounds,
        scope_region_map=scope_region_map,
        scope_region_id=scope_region_id,
        scope_pixels_per_cell=scope_pixels_per_cell,
        region=str(expect("region", str)),
        ini_path=Path(str(expect("ini_path", str))),
        land_plugin=Path(str(expect("land_plugin", str))),
        master_name=str(values.get("master_name", "tamriel.esm")),
        mesh_roots=path_list("mesh_roots"),
        scatter_exclusions=(
            Path(str(values["scatter_exclusions"])) if values.get("scatter_exclusions") else None
        ),
        settlements_json=(
            Path(str(values["settlements_json"])) if values.get("settlements_json") else None
        ),
        settlement_radius_gu=float(values.get("settlement_radius_gu", 1400.0)),
        clearing_json=(
            tuple(Path(str(item)) for item in values["clearing_json"])
            if isinstance(values.get("clearing_json"), list)
            else (Path(str(values["clearing_json"])) if values.get("clearing_json") else None)
        ),
        edited_land_json=(
            Path(str(values["edited_land_json"])) if values.get("edited_land_json") else None
        ),
        exclusion_rules=rules,
        default_exclusion_radius_gu=float(values.get("default_exclusion_radius_gu", 150.0)),
        exclusion_margin_gu=float(values.get("exclusion_margin_gu", DEFAULT_EXCLUSION_MARGIN_GU)),
        road_texture_regexes=tuple(str(item) for item in values.get("road_texture_regexes", ())),
        road_raw_vtex_values=tuple(int(item) for item in values.get("road_raw_vtex_values", ())),
        water_flora_texture_ids=tuple(str(item) for item in values.get("water_flora_texture_ids", ())),
        extra_banned_textures=tuple(str(item) for item in values.get("extra_banned_textures", ())),
        jitter_gu=(
            float(values["jitter_gu"]) if values.get("jitter_gu") is not None else None
        ),
        z_modifier_gu=(
            float(values["z_modifier_gu"]) if values.get("z_modifier_gu") is not None else None
        ),
        object_prefix=str(values.get("object_prefix", "PTGC_")),
        slope_spacing_gu=float(values.get("slope_spacing_gu", 128.0)),
        normal_spacing_gu=float(values.get("normal_spacing_gu", 128.0)),
        water_threshold_thu=int(values.get("water_threshold_thu", 0)),
        limit_cells=(int(values["limit_cells"]) if values.get("limit_cells") is not None else None),
        max_seconds=(
            float(values["max_seconds"]) if values.get("max_seconds") is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Generation core
# ---------------------------------------------------------------------------


def _terrain_normal(
    land_records: Mapping[tuple[int, int], LandRecord],
    x_gu: float,
    y_gu: float,
    spacing_gu: float,
) -> tuple[float, float, float] | None:
    """Sample the terrain normal (unit vector, up is +Z) at a position.

    Heights are sampled at +/- ``spacing_gu`` along X and Y and converted to
    game units; the gradient then defines the normal ``(-gx, -gy, 1)``
    normalized.  ``None`` when any sample is outside the LAND set.
    """

    center = height_at_game_position(land_records, (x_gu, y_gu))
    if center is None:
        return None
    plus_x = height_at_game_position(land_records, (x_gu + spacing_gu, y_gu))
    minus_x = height_at_game_position(land_records, (x_gu - spacing_gu, y_gu))
    plus_y = height_at_game_position(land_records, (x_gu, y_gu + spacing_gu))
    minus_y = height_at_game_position(land_records, (x_gu, y_gu - spacing_gu))
    if any(value is None for value in (plus_x, minus_x, plus_y, minus_y)):
        return None
    gx = (float(plus_x) - float(minus_x)) * THU_TO_GU / (2.0 * spacing_gu)
    gy = (float(plus_y) - float(minus_y)) * THU_TO_GU / (2.0 * spacing_gu)
    length = math.sqrt(gx * gx + gy * gy + 1.0)
    return (-gx / length, -gy / length, 1.0 / length)


def _engine_tilt(normal: Sequence[float], yaw: float) -> tuple[float, float]:
    """Return (rx, ry) that makes the quad lie exactly on the terrain.

    OpenMW constructs ``Qz(-rz) * Qy(-ry) * Qx(-rx)`` in
    ``components/misc/convert.hpp::makeOsgQuat`` lines 50-54.  OpenSceneGraph
    reverses the Hamilton operands in ``osg::Quat::operator*``, giving the
    active column-vector matrix ``Rx(-rx) @ Ry(-ry) @ Rz(-rz)``.  For a flat
    quad, the engine's up vector is therefore
    ``(-sin ry, sin rx cos ry, cos rx cos ry)``: the rightmost yaw acts on the
    local quad before the two tilt axes and does not change its up vector.
    Solving ``up = normal`` gives ``ry = asin(-nx)`` and
    ``rx = asin(ny / cos ry)``.  This is exact for any yaw, so a random spin
    never lifts the quad off the terrain.  ``yaw`` remains an argument because
    it is part of the authored rotation triple, even though it is absent from
    this normal solve under the authoritative composition.
    """

    nx, ny, _nz = float(normal[0]), float(normal[1]), float(normal[2])
    _ = float(yaw)  # retained in the API; see the normal-composition note above
    ry = math.asin(max(-1.0, min(1.0, -nx)))
    cos_ry = math.sqrt(max(0.0, 1.0 - math.sin(ry) ** 2))
    if cos_ry < 1e-9:
        return 0.0, ry
    rx = math.asin(max(-1.0, min(1.0, ny / cos_ry)))
    return rx, ry


def _engine_up(rotation: Sequence[float]) -> tuple[float, float, float]:
    """Map (0,0,1) through OpenMW's ref rotation composition.

    Matches ``Misc::Convert::makeOsgQuat`` (convert.hpp lines 50-54) plus
    OpenSceneGraph's reversed Hamilton operand semantics: the active matrix is
    ``Rx(-rx) @ Ry(-ry) @ Rz(-rz)`` for column vectors.  The rightmost Z
    rotation is applied first and leaves the local up vector unchanged.
    """

    rx, ry, rz = (float(v) for v in rotation)
    # Apply Rz(-rz) first.  Starting from local +Z means this first step is
    # intentionally a no-op; keeping it explicit documents the matrix order.
    x, y, z = 0.0, 0.0, 1.0
    c, s = math.cos(-rz), math.sin(-rz)
    # The Z step belongs on the right of the authoritative matrix and thus
    # precedes Ry/Rx.  It is a no-op for this vector, but retain the operation
    # in the helper so the basis-vector derivation is visible.
    x, y = x * c - y * s, x * s + y * c
    c, s = math.cos(-ry), math.sin(-ry)
    x, y, z = x * c + z * s, y, -x * s + z * c
    c, s = math.cos(-rx), math.sin(-rx)
    x, y, z = x, y * c - z * s, y * s + z * c
    return x, y, z


def _fnv1a32(text: str) -> int:
    hash_value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return hash_value


class _Audit:
    """Counters for rejection reasons plus per-texture placement tallies."""

    def __init__(self) -> None:
        self.rejected: dict[str, int] = {}
        self.by_texture: dict[str, int] = {}
        self.unmatched_textures: dict[str, int] = {}
        self.road_raw_vtex_placement_count = 0
        self.placed = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def note_texture(self, name: str) -> None:
        self.by_texture[name] = self.by_texture.get(name, 0) + 1

    def note_unmatched(self, name: str) -> None:
        self.unmatched_textures[name] = self.unmatched_textures.get(name, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {
            "placed": self.placed,
            "rejected": dict(sorted(self.rejected.items())),
            "by_texture": dict(sorted(self.by_texture.items())),
            "unmatched_textures": dict(sorted(self.unmatched_textures.items())),
            "road_raw_vtex_placement_count": self.road_raw_vtex_placement_count,
        }


def _candidate_lattice(
    gx: int,
    gy: int,
    gap_gu: int,
    rng: random.Random,
    section: SectionSpec,
) -> Iterable[tuple[float, float]]:
    """Yield jittered lattice positions (absolute game units) for one cell."""

    origin_x = gx * CELL_SIZE_GAME_UNITS
    origin_y = gy * CELL_SIZE_GAME_UNITS
    jitter_min = section.jitter_min_gu
    jitter_span = section.jitter_max_gu - section.jitter_min_gu
    step = max(1, int(gap_gu))
    side = int(CELL_SIZE_GAME_UNITS)
    for ix in range(step // 2, side, step):
        for iy in range(step // 2, side, step):
            jx = rng.uniform(jitter_min, jitter_min + jitter_span) if jitter_span > 0 else 0.0
            jy = rng.uniform(jitter_min, jitter_min + jitter_span) if jitter_span > 0 else 0.0
            yield origin_x + ix + jx, origin_y + iy + jy


def generate_groundcover_document(
    config: GroundcoverRunConfig,
    ini: GroundcoverIni,
) -> dict[str, Any]:
    """Load the terrain plugin and run one deterministic generation pass.

    When ``config.edited_land_json`` is set, the base ``config.land_plugin``
    LAND is loaded first and the affected cells' records are replaced by the
    city generation's edited-LAND JSON (a tes3conv document) directly — no
    ESP conversion.  Outside the affected cells the base LAND is used as-is.
    """

    land_records = load_land(config.land_plugin)
    if config.edited_land_json is not None:
        from procgen.tes3json import land_records_from_json  # noqa: E402

        edited_doc = _read_json_document(config.edited_land_json)
        edited_records = land_records_from_json(edited_doc)
        merged = dict(land_records)
        merged.update(edited_records)
        land_records = merged
    ltex = load_ltex(config.land_plugin)
    return generate_groundcover_document_with_land(config, ini, land_records, ltex)


def generate_groundcover_document_with_land(
    config: GroundcoverRunConfig,
    ini: GroundcoverIni,
    land_records: Mapping[tuple[int, int], LandRecord],
    ltex: Mapping[int, LandscapeTexture] | None = None,
) -> dict[str, Any]:
    """Run one deterministic groundcover generation pass (see module docstring).

    ``land_records`` and ``ltex`` are caller-supplied (normally produced by
    ``generate_groundcover_document`` via ``procgen.espland``); tests inject
    synthetic records through this entry point.
    """

    if ltex is None:
        ltex = {}
    started = time.perf_counter()

    def check_budget() -> None:
        if config.max_seconds is not None and time.perf_counter() - started > config.max_seconds:
            raise TimeoutError(
                f"groundcover generation exceeded {config.max_seconds:.1f}s budget"
            )

    palette = build_palette(
        ini,
        config.region,
        extra_banned_textures=config.extra_banned_textures,
        jitter_gu=config.jitter_gu,
    )
    if not palette:
        raise ValueError(
            f"palette for region {config.region!r} is empty "
            f"(no bPlaceGrass=1 sections in {config.ini_path})"
        )

    used_meshes: set[str] = set()
    for section in palette.values():
        for option in section.options:
            if option.mesh:
                used_meshes.add(option.mesh)
    _validate_meshes(used_meshes, config.mesh_roots)

    exclusion_index = ExclusionIndex(
        rules=config.exclusion_rules,
        default_radius_gu=config.default_exclusion_radius_gu,
        margin_gu=config.exclusion_margin_gu,
    )
    if config.scatter_exclusions is not None:
        scatter = _read_json_mapping(config.scatter_exclusions)
        for cell_entry in scatter.get("density", {}).get("cells", []):
            if not isinstance(cell_entry, Mapping):
                continue
            for ref in cell_entry.get("refs", []):
                if isinstance(ref, Mapping):
                    exclusion_index.add_ref(ref)
    if config.settlements_json is not None:
        circles = load_settlement_circles(
            config.settlements_json, config.bounds, config.settlement_radius_gu
        )
        for circle in circles:
            exclusion_index.add_circle(circle)

    clearing_index: ClearingIndex | MultiClearingIndex | None = None
    if config.clearing_json is not None:
        clearing_paths = (
            [Path(p) for p in config.clearing_json]
            if isinstance(config.clearing_json, (list, tuple))
            else [Path(config.clearing_json)]
        )
        clearing_docs = [_read_json_mapping(path) for path in clearing_paths]
        clearing_index = build_clearing_index(clearing_docs)

    road_regexes = tuple(re.compile(pattern, re.IGNORECASE) for pattern in config.road_texture_regexes)
    z_modifier = (
        config.z_modifier_gu
        if config.z_modifier_gu is not None
        else float(ini.z_position_modifier_gu)
    )

    (min_x, min_y), (max_x, max_y) = config.bounds
    scope_cells: set[tuple[int, int]] | None = None
    scope_metadata: dict[str, Any] | None = None
    if config.scope_region_map is not None and config.scope_region_id is not None:
        scope_cells, scope_metadata = region_cells(
            config.scope_region_map,
            config.scope_region_id,
            pixels_per_cell=config.scope_pixels_per_cell,
        )
        outside = [cell for cell in scope_cells if not (min_x <= cell[0] <= max_x and min_y <= cell[1] <= max_y)]
        if outside:
            raise ValueError(f"groundcover scope extends outside configured bounds: {outside[:8]}")
    cells: list[dict[str, Any]] = []
    global_audit = _Audit()
    generated_cells = 0

    for gx in range(min_x, max_x + 1):
        if config.limit_cells is not None and generated_cells >= config.limit_cells:
            break
        for gy in range(min_y, max_y + 1):
            if scope_cells is not None and (gx, gy) not in scope_cells:
                continue
            if config.limit_cells is not None and generated_cells >= config.limit_cells:
                break
            generated_cells += 1
            check_budget()
            cell_audit = _Audit()
            cell_rng = random.Random(
                derive_seed(config.master_seed, "groundcover", config.area, config.version, "cell", gx, gy)
            )
            record = land_records.get((gx, gy))
            refs: list[dict[str, Any]] = []
            if record is None or not record.has_textures:
                cell_audit.reject("no_land")
                cells.append(
                    {"grid": [gx, gy], "refs": [], "stats": _cell_stats(cell_audit)}
                )
                _merge_audits(global_audit, cell_audit)
                continue

            sections_by_gap: dict[int, SectionSpec] = {}
            for spec in palette.values():
                sections_by_gap.setdefault(spec.gap_gu, spec)
            for gap, spec in sections_by_gap.items():
                for lattice_x, lattice_y in _candidate_lattice(gx, gy, gap, cell_rng, spec):
                    ref = _evaluate_candidate(
                        config=config,
                        land_records=land_records,
                        ltex=ltex,
                        palette=palette,
                        road_regexes=road_regexes,
                        z_modifier=z_modifier,
                        exclusion_index=exclusion_index,
                        clearing_index=clearing_index,
                        rng=cell_rng,
                        gx=gx,
                        gy=gy,
                        lattice_xy=(lattice_x, lattice_y),
                        audit=cell_audit,
                    )
                    if ref is not None:
                        refs.append(ref)
            for index, ref in enumerate(refs):
                ref["ref_id"] = f"gc_{gx}_{gy}_{index:06d}"
            _merge_audits(global_audit, cell_audit)
            cells.append({"grid": [gx, gy], "refs": refs, "stats": _cell_stats(cell_audit)})

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "seed": config.master_seed,
        "determinism": {
            "seeded_by": "procgen.seeds.derive_seed",
            "scope": ["groundcover", config.area, config.version, "cell", "gx", "gy"],
        },
            "scope": {
            "area": config.area,
            "version": config.version,
            "bounds": [list(bounds) for bounds in config.bounds],
            "region": config.region,
            "cell_count": len(cells),
            "cells": [[int(row["grid"][0]), int(row["grid"][1])] for row in cells],
            "region_scope": scope_metadata,
        },
        "units": {"cell_size_gu": CELL_SIZE_GAME_UNITS, "thu_to_gu": THU_TO_GU},
        "terrain": {
            "land_plugin": str(config.land_plugin),
            "edited_land_json": str(config.edited_land_json) if config.edited_land_json else None,
            "water_threshold_thu": config.water_threshold_thu,
        },
        "palette": {texture: spec.describe() for texture, spec in sorted(palette.items())},
        "masks": {
            "road_texture_regexes": list(config.road_texture_regexes),
            "road_raw_vtex_values": list(config.road_raw_vtex_values),
            "water_flora_texture_ids": list(config.water_flora_texture_ids),
            "extra_banned_textures": list(config.extra_banned_textures),
            "water": {"threshold_thu": config.water_threshold_thu},
        },
        "exclusion": {
            "scatter_json": str(config.scatter_exclusions) if config.scatter_exclusions else None,
            "static_ref_count": exclusion_index.ref_count,
            "settlement_circles": [
                [circle.x_gu, circle.y_gu, circle.radius_gu, circle.label]
                for circle in exclusion_index.circle_list()
            ],
            "rules": [
                [rule.tokens, rule.radius_gu, rule.use_bbox, rule.skip]
                for rule in config.exclusion_rules
            ],
            "margin_gu": config.exclusion_margin_gu,
        },
        "city_clearing": {
            "enabled": clearing_index is not None,
            "clearing_json": str(config.clearing_json) if config.clearing_json else None,
            "frame_origin_gu": (
                [list(o) for o in clearing_index.frame_origins_gu] if hasattr(clearing_index, "frame_origins_gu")
                else list(clearing_index.frame_origin_gu)
            ) if clearing_index is not None else None,
            "blocks_point_rejections": int(global_audit.rejected.get("clearing_blocked", 0)),
            "rule": (
                "groundcover candidates are rejected inside building footprints, "
                "circulation surfaces, and road corridors via "
                "ClearingIndex.blocks_point; city_domain does NOT block "
                "groundcover (grassy in-town tiles keep grass); scatter statics "
                "are excluded through the existing ExclusionIndex"
            ),
        },
        "density": {"cells": cells, "placement_stats": _placement_stats(cells, global_audit)},
        "generation_failures": [],
    }


def _merge_audits(target: _Audit, source: _Audit) -> None:
    """Fold one cell's counters into the document-wide audit."""

    for reason, count in source.rejected.items():
        target.rejected[reason] = target.rejected.get(reason, 0) + count
    for name, count in source.by_texture.items():
        target.by_texture[name] = target.by_texture.get(name, 0) + count
    for name, count in source.unmatched_textures.items():
        target.unmatched_textures[name] = target.unmatched_textures.get(name, 0) + count
    target.placed += source.placed
    target.road_raw_vtex_placement_count += source.road_raw_vtex_placement_count


def _cell_stats(audit: _Audit) -> dict[str, Any]:
    return {
        "ref_count": audit.placed,
        "rejected": dict(sorted(audit.rejected.items())),
    }


def _placement_stats(cells: Sequence[Mapping[str, Any]], audit: _Audit) -> dict[str, Any]:
    counts = [entry["stats"]["ref_count"] for entry in cells]
    return {
        "total": audit.placed,
        "per_cell": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(sum(counts) / len(counts), 3) if counts else 0.0,
        },
        "by_texture": dict(sorted(audit.by_texture.items())),
        "rejected": dict(sorted(audit.rejected.items())),
        "unmatched_textures": dict(sorted(audit.unmatched_textures.items())),
        "road_raw_vtex_placement_count": audit.road_raw_vtex_placement_count,
    }


def _evaluate_candidate(
    *,
    config: GroundcoverRunConfig,
    land_records: Mapping[tuple[int, int], LandRecord],
    ltex: Mapping[int, LandscapeTexture],
    palette: Mapping[str, SectionSpec],
    road_regexes: Sequence[re.Pattern[str]],
    z_modifier: float,
    exclusion_index: ExclusionIndex,
    clearing_index: ClearingIndex | None,
    rng: random.Random,
    gx: int,
    gy: int,
    lattice_xy: tuple[float, float],
    audit: _Audit,
) -> dict[str, Any] | None:
    x_gu, y_gu = lattice_xy
    terrain_thu = height_at_game_position(land_records, (x_gu, y_gu))
    if terrain_thu is None:
        audit.reject("no_terrain")
        return None
    terrain_gu = float(terrain_thu) * THU_TO_GU
    texture = _texture_at(land_records, ltex, (x_gu, y_gu))
    if texture is None:
        audit.reject("unmatched_texture")
        return None
    raw_vtex = int(texture["raw_vtex"])
    texture_name = str(texture["name"])
    texture_path = str(texture["path"])
    if terrain_gu <= config.water_threshold_thu * THU_TO_GU and texture_name not in config.water_flora_texture_ids:
        audit.reject("water")
        return None
    if raw_vtex == 0:
        audit.reject("base_texture")
        return None
    if raw_vtex in config.road_raw_vtex_values:
        audit.reject(ROAD_REJECT)
        return None
    # Road mask first: a road tile is rejected even when its texture happens
    # to be a palette key, and the audit counts the mask separately.
    for regex in road_regexes:
        if regex.search(texture_path):
            audit.reject(ROAD_REJECT)
            return None
    active = palette.get(texture_name)
    if active is None:
        audit.note_unmatched(texture_name)
        audit.reject("unmatched_texture")
        return None

    # All gates below use the texture-matched section, not the lattice
    # section, so a mixed-texture cell obeys each texture's own limits.
    if terrain_gu < active.min_height_gu:
        audit.reject("below_min_height")
        return None
    if active.max_height_gu is not None and terrain_gu > active.max_height_gu:
        audit.reject("above_max_height")
        return None

    slope = terrain_slope_deg(land_records, (x_gu, y_gu), spacing_game_units=config.slope_spacing_gu)
    if slope is None:
        audit.reject("no_slope")
        return None
    if slope > active.max_angle_deg:
        audit.reject("slope")
        return None

    ban_reason = _check_bans(
        active.bans, texture_name, texture_path, road_regexes, ltex, land_records, x_gu, y_gu
    )
    if ban_reason is not None:
        audit.reject(ban_reason)
        return None
    blocked, blocker = exclusion_index.is_blocked(x_gu, y_gu, (gx, gy))
    if blocked:
        audit.reject("static_exclusion" if blocker.startswith("static") else "settlement_exclusion")
        return None
    if clearing_index is not None and clearing_index.blocks_point(x_gu, y_gu):
        audit.reject("clearing_blocked")
        return None

    mesh = _choose_mesh(active.options, rng)
    if mesh is None:
        audit.reject("no_mesh")
        return None

    scale = rng.uniform(active.scale_min, active.scale_max)
    yaw = rng.uniform(0.0, 2.0 * math.pi)
    rx = ry = 0.0
    if active.align_to_normal:
        normal = _terrain_normal(land_records, x_gu, y_gu, config.normal_spacing_gu)
        if normal is not None:
            # OpenMW's Qz*Qy*Qx source expression (convert.hpp lines 50-54)
            # becomes Rx(-rx) @ Ry(-ry) @ Rz(-rz) under OSG's reversed
            # Hamilton operand semantics.  The tilt angles are solved for that
            # composition and the quad lies exactly on the terrain for any
            # yaw.
            rx, ry = _engine_tilt(normal, yaw)

    audit.placed += 1
    audit.note_texture(active.texture_id)
    return {
        "ref_id": "",
        "cell": [gx, gy],
        "mesh": mesh,
        "category": "groundcover",
        "pass": "C",
        "section_texture": active.texture_id,
        "section_region": active.region,
        "position_gu": [round(x_gu, 3), round(y_gu, 3), round(terrain_gu + z_modifier, 3)],
        "rotation_radians": [round(rx, 6), round(ry, 6), round(yaw, 6)],
        "rotation_mode": "normal_tilt_yaw" if active.align_to_normal else "yaw_only",
        "scale": round(scale, 6),
        "terrain": {
            "terrain_source": "tamriel.esm LAND via procgen.espland",
            "terrain_z_thu": round(float(terrain_thu), 3),
            "terrain_z_gu": round(terrain_gu, 3),
            "raw_vtex": raw_vtex,
            "ltex_name": texture_name,
            "ltex_path": texture_path,
            "slope_deg": round(float(slope), 3),
            "water_state": "above_water",
            "road_raw_vtex": bool(re.search(r"road|mainroad|dirtroad|gravel|beatenpath", texture_path, re.IGNORECASE)),
        },
        "lattice": {
            "gap_gu": active.gap_gu,
            "jitter_min_gu": active.jitter_min_gu,
            "jitter_max_gu": active.jitter_max_gu,
            "lattice_xy": [round(x_gu, 3), round(y_gu, 3)],
        },
    }


def _texture_at(
    land_records: Mapping[tuple[int, int], LandRecord],
    ltex: Mapping[int, LandscapeTexture],
    position: Sequence[float],
) -> dict[str, Any] | None:
    """Resolve the tile texture, treating missing LAND/LTEX as unmatched."""

    try:
        return texture_at_position(land_records, ltex, position)
    except ValueError:
        return None


def _check_bans(
    bans: Sequence[TextureBan],
    texture_name: str,
    texture_path: str,
    road_regexes: Sequence[re.Pattern[str]],
    ltex: Mapping[int, LandscapeTexture],
    land_records: Mapping[tuple[int, int], LandRecord],
    x_gu: float,
    y_gu: float,
) -> str | None:
    """Reject when the tile texture matches a ban or a road regex.

    Bans with a nonzero offset probe eight neighbour positions at that
    distance; a match on any probed tile also rejects.
    """

    for regex in road_regexes:
        if regex.search(texture_path):
            return ROAD_REJECT
    for ban in bans:
        if ban.texture and ban.texture.lower() in texture_name.lower():
            return "texture_ban"
        if ban.texture and ban.texture.lower() in texture_path.lower():
            return "texture_ban"
    for ban in bans:
        offset = ban.offset_gu
        if offset <= 0.0 or not ban.texture:
            continue
        for dx, dy in (
            (offset, 0.0),
            (-offset, 0.0),
            (0.0, offset),
            (0.0, -offset),
            (offset * 0.7071, offset * 0.7071),
            (-offset * 0.7071, offset * 0.7071),
            (offset * 0.7071, -offset * 0.7071),
            (-offset * 0.7071, -offset * 0.7071),
        ):
            probe_x, probe_y = x_gu + dx, y_gu + dy
            try:
                probe = texture_at_position(land_records, ltex, (probe_x, probe_y))
            except ValueError:
                continue
            probe_name = str(probe.get("name", ""))
            probe_path = str(probe.get("path", ""))
            if ban.texture.lower() in probe_name.lower() or ban.texture.lower() in probe_path.lower():
                return "texture_ban_offset"
    return None


def _choose_mesh(options: Sequence[MeshOption], rng: random.Random) -> str | None:
    meshes = [option.mesh for option in options if option.mesh]
    weights = [option.chance for option in options if option.mesh]
    if not meshes:
        return None
    return rng.choices(meshes, weights=weights, k=1)[0]


def _validate_meshes(meshes: set[str], mesh_roots: Sequence[str | Path]) -> None:
    """Fatal when a configured mesh root is missing a palette mesh.

    Mirrors greenmote's missing-mesh-is-fatal policy: generation must not
    emit references to meshes that do not exist in the installed data.
    """

    if not mesh_roots:
        return
    missing: list[str] = []
    for mesh in sorted(meshes):
        candidate = Path(mesh.replace("\\", "/"))
        found = False
        for root in mesh_roots:
            if (Path(root) / candidate).is_file():
                found = True
                break
        if not found:
            missing.append(mesh)
    if missing:
        raise FileNotFoundError(
            "palette meshes missing from all configured mesh roots: "
            + ", ".join(missing)
        )


def _read_json_mapping(path: str | Path) -> Mapping[str, Any]:
    import json

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _read_json_document(path: str | Path) -> Any:
    """Read any JSON document (object or list, e.g. a tes3conv record list)."""

    import json

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Plugin authoring (ref document -> tes3conv JSON -> .esp)
# ---------------------------------------------------------------------------


def _unique_meshes(document: Mapping[str, Any]) -> list[str]:
    meshes: set[str] = set()
    for cell_entry in document.get("density", {}).get("cells", []):
        if not isinstance(cell_entry, Mapping):
            continue
        for ref in cell_entry.get("refs", []):
            if isinstance(ref, Mapping) and ref.get("mesh"):
                meshes.add(str(ref["mesh"]))
    return sorted(meshes, key=lambda mesh: mesh.lower())


def _static_id_for(mesh: str, prefix: str, used: set[str]) -> str:
    candidate = f"{prefix}{_fnv1a32(mesh.lower()):08x}"
    counter = 0
    while candidate in used:
        counter += 1
        candidate = f"{prefix}{_fnv1a32(mesh.lower()):08x}_{counter}"
    used.add(candidate)
    return candidate


def build_plugin_document(
    document: Mapping[str, Any],
    *,
    master_name: str,
    master_size: int,
    object_prefix: str,
    author: str = "ProcGen Groundcover",
) -> list[dict[str, Any]]:
    """Convert a ref document into a tes3conv JSON plugin document.

    The output plugin contains: one Header, one ``Static`` definition per
    unique mesh with a deterministic ``<prefix><fnv1a32>`` id, and one
    ``Cell`` per generated cell holding its references (``temporary=True``
    so groundcover never bloats save games).  The plugin is deliberately
    **masterless**: every reference points at its own STAT and every cell is
    a new exterior cell, so no master is required (an empty ``masters`` list
    is written when ``master_name`` is empty).  Groundcover semantics are
    entirely registration-side (``groundcover=<file>.esp`` in ``openmw.cfg``),
    so no special records or flags are needed, matching FGM/OpenMW 0.51
    behaviour.
    """

    from procgen.tes3json import build_cell, build_reference, build_static, new_plugin

    meshes = _unique_meshes(document)
    used_ids: set[str] = set()
    static_by_mesh: dict[str, str] = {}
    for mesh in meshes:
        static_by_mesh[mesh] = _static_id_for(mesh, object_prefix, used_ids)

    masters: list[list[Any]] = []
    if master_name:
        masters = [[str(master_name), int(master_size)]]
    plugin = new_plugin(
        {
            "author": author,
            "description": f"Procedural groundcover {document.get('scope', {}).get('area', '')} "
            f"{document.get('scope', {}).get('version', '')}",
            "masters": masters,
        }
    )
    for mesh in meshes:
        plugin.append(build_static(static_by_mesh[mesh], mesh))

    cells = document.get("density", {}).get("cells", [])
    for cell_entry in sorted(cells, key=lambda item: (item["grid"][0], item["grid"][1])):
        if not isinstance(cell_entry, Mapping):
            continue
        grid = cell_entry.get("grid")
        refs = cell_entry.get("refs", [])
        if not isinstance(grid, Sequence) or len(grid) != 2 or not refs:
            continue
        references = [
            build_reference(
                static_by_mesh[str(ref["mesh"])],
                index + 1,
                translation=ref["position_gu"],
                rotation=ref["rotation_radians"],
                scale=float(ref["scale"]),
                temporary=True,
            )
            for index, ref in enumerate(refs)
            if isinstance(ref, Mapping) and str(ref["mesh"]) in static_by_mesh
        ]
        plugin.append(build_cell("", [int(grid[0]), int(grid[1])], references=references))
    return plugin


def author_plugin(
    document: Mapping[str, Any],
    *,
    master_plugin: str | Path,
    master_name: str,
    object_prefix: str,
    output_path: str | Path,
    scratch_dir: str | Path,
    tes3conv_path: str | Path,
) -> Path:
    """Write a ref document as an .esp via tes3conv (validating first).

    The tes3conv JSON is written into ``scratch_dir``, validated with
    ``procgen.tes3json.validate`` (which must return zero issues), converted
    with ``tes3conv -o -c``, and the resulting esp is copied to
    ``output_path``.  Raises on any validation error or converter failure.
    The plugin is authored masterless (``master_name`` empty) by default;
    when a master is declared its on-disk size is recorded.
    """

    from procgen.tes3json import validate, write_json

    master_size = 0
    if master_name:
        master_size = os.path.getsize(str(master_plugin))
    plugin = build_plugin_document(
        document,
        master_name=master_name,
        master_size=master_size,
        object_prefix=object_prefix,
    )
    issues = validate(plugin)
    if issues:
        raise ValueError(
            "groundcover plugin validation failed:\n" + "\n".join(str(issue) for issue in issues)
        )

    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    json_path = scratch / f"groundcover_{document.get('scope', {}).get('area', 'x')}_plugin.json"
    write_json(plugin, json_path)

    result = subprocess.run(
        [
            str(tes3conv_path),
            "-o",
            "-c",
            str(json_path),
            str(scratch / "groundcover_tmp.esp"),
        ],
        cwd=str(scratch),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"tes3conv failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    final = Path(output_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(scratch / "groundcover_tmp.esp"), str(final))
    return final
