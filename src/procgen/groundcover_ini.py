"""Parser for MWGroundcoverGenerator-style FGM groundcover INI files.

Pipeline position
------------------
This is the first stage of the procedural groundcover pipeline: it turns a
modmaker/mesh-generator INI (the format used to build the FGM_*.esp groundcover
family, e.g. ``FGM_SHOTN.ini``) into structured Python objects.  The next
stage is ``procgen.groundcover_generate`` which consumes these objects as its
palette.  No plugin, JSON, or mesh data is read or written here.

Input
-----
One INI file with:

* an optional ``[global]`` section holding ``iZPositionModifier`` and
  ``sObjectPrefix``, and
* one ``[TEXTURE_ID:REGION_NAME]`` (or bare ``[TEXTURE_ID]``) section per
  ground texture; each section carries placement behaviour (``iGap`` spacing,
  ``fSclMin``/``fSclMax`` scale range, ``fPosMin``/``fPosMax`` jitter,
  ``fMaximumAngle`` slope cap, ``fMinHeight``/``fMaxHeight`` elevation
  window), texture bans (``sBanN`` + ``iBanOffN``), and a weighted mesh table
  (``sMeshN``/``sChanceN`` or ``sIDN``).

Output
------
A frozen ``GroundcoverIni`` value object.  ``parse_ini`` raises
``GroundcoverIniError`` for structural problems (missing section key, missing
``iGap``, unknown boolean) and never silently drops a section's behaviour.

Invariants
----------
* All numeric fields are finite floats/ints; booleans are strict (0/1 or
  true/false, case-insensitive).
* Section headers are matched as ``[TEXTURE:REGION]``; a header may carry
  stray trailing ``]`` characters (seen in real FGM files) which are stripped.
* ``sIDN`` and ``sMeshN`` are mutually exclusive per index.
* Case is preserved for texture ids, region names, mesh paths, and bans
  (the original tool treats these as case-sensitive).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

_SECTION_RE = re.compile(r"^\[(.+)\]$")
_INT_KEYS = {"iZPositionModifier", "iGap", "iBanOff0"}
_BOOL_KEYS = {"bPlaceGrass", "bRandClump", "bAlignObjectNormalToGround"}
_FLOAT_KEYS = {
    "fMaximumAngle",
    "fMinHeight",
    "fMaxHeight",
    "fPosMin",
    "fPosMax",
    "fSclMin",
    "fSclMax",
}


class GroundcoverIniError(ValueError):
    """Raised when an INI file cannot be interpreted as a groundcover config."""


@dataclass(frozen=True)
class TextureBan:
    """One banned ground texture; placement is rejected on its tiles."""

    texture: str
    offset_gu: float = 0.0


@dataclass(frozen=True)
class MeshOption:
    """One weighted entry of a section's placement table."""

    index: int
    mesh: str = ""
    record_id: str = ""
    chance: int = 1


@dataclass(frozen=True)
class TextureSection:
    """Placement behaviour for one ground texture (optionally per region)."""

    texture_id: str
    region: str
    place_grass: bool = True
    random_clump: bool = False
    max_angle_deg: float = 180.0
    min_height_gu: float = 0.0
    max_height_gu: float | None = None
    pos_min_gu: float = 0.0
    pos_max_gu: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    gap_gu: int = 0
    align_to_normal: bool = True
    bans: tuple[TextureBan, ...] = ()
    options: tuple[MeshOption, ...] = ()


@dataclass(frozen=True)
class GroundcoverIni:
    """A parsed FGM INI: global settings plus per-texture sections."""

    z_position_modifier_gu: int = 10
    object_prefix: str = "GRS_"
    sections: tuple[TextureSection, ...] = ()
    warnings: tuple[str, ...] = ()

    def sections_for_region(self, region: str) -> tuple[TextureSection, ...]:
        """Return sections whose region qualifier equals ``region`` (exact)."""

        return tuple(section for section in self.sections if section.region == region)

    def sections_for_texture(self, texture_id: str) -> tuple[TextureSection, ...]:
        """Return sections keyed on ``texture_id`` (exact, case-sensitive)."""

        return tuple(section for section in self.sections if section.texture_id == texture_id)

    def select(
        self, texture_id: str, region: str | None
    ) -> TextureSection | None:
        """Pick one section: exact ``texture:region`` first, then texture-only.

        Mirror of the original tool's selector precedence (exact qualifier
        wins over unqualified texture sections).
        """

        if region:
            for section in self.sections:
                if section.texture_id == texture_id and section.region == region:
                    return section
        for section in self.sections:
            if section.texture_id == texture_id and not section.region:
                return section
        return None


def _parse_bool(key: str, raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise GroundcoverIniError(f"key {key!r}: expected a boolean, got {raw!r}")


def _parse_float(key: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise GroundcoverIniError(f"key {key!r}: expected a number, got {raw!r}") from exc
    if not _isfinite(value):
        raise GroundcoverIniError(f"key {key!r}: value is not finite: {raw!r}")
    return value


def _parse_int(key: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise GroundcoverIniError(f"key {key!r}: expected an integer, got {raw!r}") from exc
    return value


def _isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _split_section_header(line: str) -> tuple[str, str]:
    """Parse ``[TEXTURE:REGION]`` (or ``[TEXTURE]``), tolerating stray brackets.

    A real FGM file contains headers like ``[T_Sky_TerrGrassDirtHF_01:Falkheim
    Region]]``; the extra closing bracket is stripped.  The header text is
    split on the *first* colon because texture ids never contain one.
    """

    match = _SECTION_RE.match(line.strip())
    if not match:
        raise GroundcoverIniError(f"malformed section header: {line.strip()!r}")
    header = match.group(1).rstrip("]").strip()
    if not header:
        raise GroundcoverIniError("empty section header")
    if ":" in header:
        texture, region = header.split(":", 1)
        texture = texture.strip()
        region = region.strip()
    else:
        texture, region = header, ""
    if not texture:
        raise GroundcoverIniError(f"section header has no texture id: {line.strip()!r}")
    return texture, region


def parse_ini(path: str | Path) -> GroundcoverIni:
    """Parse one FGM-style groundcover INI file (see module docstring)."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise GroundcoverIniError(f"cannot read INI {source}: {exc}") from exc

    global_values: dict[str, object] = {}
    sections: list[dict[str, object]] = []
    warnings: list[str] = []
    current: dict[str, object] | None = None
    line_number = 0

    def note(message: str) -> None:
        warnings.append(f"{source.name}:{line_number}: {message}")

    for raw_line in lines:
        line_number += 1
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("["):
            texture, region = _split_section_header(line)
            if texture == "global":
                current = global_values
            else:
                current = {"_texture_id": texture, "_region": region}
                sections.append(current)
                current_texture = texture
                current_region = region
            continue
        if "=" not in line:
            note(f"ignoring non-key line {raw_line!r}")
            continue
        key, _, raw_value = line.partition("=")
        key = key.strip()
        value = raw_value.strip()
        if current is None:
            note(f"key {key!r} appears before any section; ignored")
            continue

        base = re.sub(r"\d+$", "", key)
        if base in _BOOL_KEYS:
            current[key] = _parse_bool(key, value)
        elif base in _FLOAT_KEYS:
            current[key] = _parse_float(key, value)
        elif base in {"iGap", "iZPositionModifier"}:
            current[key] = _parse_int(key, value)
        elif base == "iBanOff":
            current[key] = _parse_float(key, value)
        elif base in {"sBan", "sMesh", "sID", "sChance"} or base == "sObjectPrefix":
            current[key] = value
        else:
            note(f"unknown key {key!r} ignored")

    parsed_sections: list[TextureSection] = []
    for index, section in enumerate(sections):
        parsed_sections.append(_build_section(section, index))

    if global_values:
        try:
            z_modifier = int(global_values.get("iZPositionModifier", 10))
        except (TypeError, ValueError) as exc:
            raise GroundcoverIniError(f"global iZPositionModifier: {exc}") from exc
        object_prefix = str(global_values.get("sObjectPrefix", "GRS_"))
    else:
        z_modifier = 10
        object_prefix = "GRS_"
    return GroundcoverIni(
        z_position_modifier_gu=z_modifier,
        object_prefix=object_prefix,
        sections=tuple(parsed_sections),
        warnings=tuple(warnings),
    )


def _section_float(section: dict[str, object], key: str, default: float) -> float:
    value = section.get(key, default)
    if not isinstance(value, (int, float)):
        raise GroundcoverIniError(f"{key!r}: expected a number, got {value!r}")
    return float(value)


def _build_section(raw: dict[str, object], index: int) -> TextureSection:
    texture = str(raw.get("_texture_id", ""))
    region = str(raw.get("_region", ""))
    place_grass = bool(raw.get("bPlaceGrass", True))
    gap = raw.get("iGap")
    if gap is None:
        raise GroundcoverIniError(f"section {index} ({texture or '<global>'}): missing required key iGap")
    gap_value = int(gap)
    if gap_value <= 0:
        raise GroundcoverIniError(f"section {index}: iGap must be positive, got {gap_value}")

    bans: list[TextureBan] = []
    ban_names: dict[int, str] = {}
    for key, value in raw.items():
        match = re.fullmatch(r"sBan(\d+)", key)
        if match:
            ban_names[int(match.group(1))] = str(value)
    for key, value in raw.items():
        match = re.fullmatch(r"iBanOff(\d+)", key)
        if match:
            number = int(match.group(1))
            if number in ban_names:
                bans.append(TextureBan(ban_names[number], _section_float(raw, key, 0.0)))
    for number, name in sorted(ban_names.items()):
        if all(ban.texture != name for ban in bans):
            bans.append(TextureBan(name, 0.0))

    options: list[MeshOption] = []
    meshes: dict[int, str] = {}
    record_ids: dict[int, str] = {}
    chances: dict[int, int] = {}
    for key, value in raw.items():
        match = re.fullmatch(r"sMesh(\d+)", key)
        if match:
            meshes[int(match.group(1))] = str(value)
    for key, value in raw.items():
        match = re.fullmatch(r"sID(\d+)", key)
        if match:
            record_ids[int(match.group(1))] = str(value)
    for key, value in raw.items():
        match = re.fullmatch(r"sChance(\d+)", key)
        if match:
            chances[int(match.group(1))] = int(value)
    for number in sorted(set(meshes) | set(record_ids)):
        mesh = meshes.get(number, "")
        record_id = record_ids.get(number, "")
        if mesh and record_id:
            raise GroundcoverIniError(
                f"section {index} ({texture}): sMesh{number} and sID{number} are mutually exclusive"
            )
        if not mesh and not record_id:
            continue
        chance = chances.get(number, 1)
        if chance < 0:
            raise GroundcoverIniError(f"section {index}: sChance{number} must be non-negative")
        options.append(MeshOption(number, mesh=mesh, record_id=record_id, chance=chance))

    if place_grass and not options:
        raise GroundcoverIniError(
            f"section {index} ({texture}): bPlaceGrass=1 but no sMesh/sID entries"
        )

    min_height = _section_float(raw, "fMinHeight", 0.0)
    max_height = raw.get("fMaxHeight")
    return TextureSection(
        texture_id=texture,
        region=region,
        place_grass=place_grass,
        random_clump=bool(raw.get("bRandClump", False)),
        max_angle_deg=_section_float(raw, "fMaximumAngle", 180.0),
        min_height_gu=min_height,
        max_height_gu=None if max_height is None else float(max_height),
        pos_min_gu=_section_float(raw, "fPosMin", 0.0),
        pos_max_gu=_section_float(raw, "fPosMax", 0.0),
        scale_min=_section_float(raw, "fSclMin", 1.0),
        scale_max=_section_float(raw, "fSclMax", 1.0),
        gap_gu=gap_value,
        align_to_normal=bool(raw.get("bAlignObjectNormalToGround", True)),
        bans=tuple(bans),
        options=tuple(options),
    )
