"""Small, strict authoring helpers for the tes3conv JSON format.

The tes3 crate deliberately does not provide a permissive JSON schema: fields
which are not ``Option`` in the Rust structs must be present.  This module
keeps that contract close to the generator and reports structural errors
before tes3conv is invoked.  It is intentionally independent of OpenMW APIs.

The public document type is a plain list of dictionaries, matching tes3conv's
top-level JSON representation.  Builders return ordinary dictionaries so
callers can still add fields supported by a newer tes3conv build.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import copy
import io
import json
import math
import struct
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeAlias

import zstandard


JsonObject: TypeAlias = dict[str, Any]
PluginDoc: TypeAlias = list[JsonObject]

_LAND_HEIGHT_COUNT = 65 * 65
_LAND_NORMAL_COUNT = _LAND_HEIGHT_COUNT * 3
_LAND_COLOR_COUNT = _LAND_HEIGHT_COUNT * 3
_LAND_WORLD_MAP_COUNT = 9 * 9
_LAND_TEXTURE_COUNT = 256

_OBJECT_FLAGS = {
    "MODIFIED",
    "DELETED",
    "PERSISTENT",
    "IGNORED",
    "BLOCKED",
}
_CELL_FLAGS = {
    "IS_INTERIOR",
    "HAS_WATER",
    "RESTING_IS_ILLEGAL",
    "BEHAVES_LIKE_EXTERIOR",
}
_LANDSCAPE_FLAGS = {
    "USES_VERTEX_HEIGHTS_AND_NORMALS",
    "USES_VERTEX_COLORS",
    "USES_TEXTURES",
}
_CONTAINER_FLAGS = {"ORGANIC", "RESPAWNS", "IS_BASE"}
_NPC_FLAGS = {"FEMALE", "ESSENTIAL", "RESPAWN", "IS_BASE", "AUTO_CALCULATE"}
_CREATURE_FLAGS = {
    "BIPED",
    "RESPAWN",
    "WEAPON_AND_SHIELD",
    "IS_BASE",
    "SWIMS",
    "FLIES",
    "WALKS",
    "ESSENTIAL",
}
_SERVICE_FLAGS = {
    "BARTERS_WEAPONS",
    "BARTERS_ARMOR",
    "BARTERS_CLOTHING",
    "BARTERS_BOOKS",
    "BARTERS_INGREDIENTS",
    "BARTERS_LOCKPICKS",
    "BARTERS_PROBES",
    "BARTERS_LIGHTS",
    "BARTERS_APPARATUS",
    "BARTERS_REPAIR_ITEMS",
    "BARTERS_MISC_ITEMS",
    "OFFERS_SPELLS",
    "BARTERS_ENCHANTED_ITEMS",
    "BARTERS_ALCHEMY",
    "OFFERS_TRAINING",
    "OFFERS_SPELLMAKING",
    "OFFERS_ENCHANTING",
    "OFFERS_REPAIRS",
}
_DIALOGUE_TYPES = {"Topic", "Voice", "Greeting", "Persuasion", "Journal"}
_SEXES = {"Any", "Male", "Female"}
_FILTER_TYPES = {
    "None",
    "Function",
    "Global",
    "Local",
    "Journal",
    "Item",
    "Dead",
    "NotId",
    "NotFaction",
    "NotClass",
    "NotRace",
    "NotCell",
    "NotLocal",
}
_COMPARISONS = {"Equal", "NotEqual", "Greater", "GreaterEqual", "Less", "LessEqual"}


@dataclass(frozen=True)
class Issue:
    """A validation or mesh-resolution issue.

    ``path`` uses a compact JSONPath-like notation (for example,
    ``records[2].references[0].id``) and is suitable for a concise CLI report.
    """

    path: str
    message: str
    code: str = "invalid"

    def __str__(self) -> str:
        return f"{self.path}: {self.message} ({self.code})"


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _vec3(value: Sequence[Real] | None = None) -> list[float]:
    if value is None:
        return [0.0, 0.0, 0.0]
    if len(value) != 3:
        raise ValueError("a vector must contain exactly three values")
    return [float(item) for item in value]


def _color(value: Sequence[Integral] | None = None) -> list[int]:
    if value is None:
        return [0, 0, 0, 0]
    if len(value) != 4:
        raise ValueError("a map color must contain four values")
    result = [int(item) for item in value]
    if any(not 0 <= item <= 255 for item in result):
        raise ValueError("map colors must be bytes")
    return result


def _blob(raw: bytes | bytearray | memoryview) -> str:
    """Encode raw bytes as tes3conv's zstd level-0/base64 blob."""

    compressed = zstandard.ZstdCompressor(level=0).compress(bytes(raw))
    return base64.b64encode(compressed).decode("ascii")


def decode_blob(value: str | Mapping[str, Any]) -> bytes:
    """Decode a tes3conv blob field (either its string or object form)."""

    encoded: Any = value.get("data") if isinstance(value, Mapping) else value
    if not isinstance(encoded, str):
        raise ValueError("blob must be a base64 string or an object with data")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        try:
            return zstandard.ZstdDecompressor().decompress(compressed)
        except zstandard.ZstdError:
            # tes3conv may emit a valid streaming Zstandard frame without a
            # content-size field.  The one-shot decoder quite correctly
            # refuses to guess that size; stream_reader preserves strict frame
            # validation while supporting the converter's canonical output.
            with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
                return reader.read()
    except Exception as exc:  # zstd raises a package-specific exception
        raise ValueError(f"invalid zstd/base64 blob: {exc}") from exc


def new_plugin(header: Mapping[str, Any] | None = None) -> PluginDoc:
    """Return a complete plugin document with a tes3conv-compatible header.

    ``header`` is a convenient partial override.  In particular, callers may
    pass ``{"masters": [["Tamriel_Data.esm", size]]}``; all other required
    header fields receive deterministic defaults.
    """

    supplied = dict(header or {})
    supplied.pop("type", None)
    result: JsonObject = {
        "type": "Header",
        "flags": "",
        "version": 1.3,
        "file_type": "Esp",
        "author": "ProcGen",
        "description": "Procedural Tamriel plugin",
        "num_objects": 0,
        "masters": [],
    }
    result.update(_copy(supplied))
    result["type"] = "Header"
    return [result]


def build_static(
    record_id: str,
    mesh: str = "",
    *,
    flags: str = "",
) -> JsonObject:
    return {"type": "Static", "flags": flags, "id": record_id, "mesh": mesh}


def build_reference(
    record_id: str,
    refr_index: int,
    *,
    translation: Sequence[Real] | None = None,
    rotation: Sequence[Real] | None = None,
    mast_index: int = 0,
    temporary: bool = False,
    scale: float | None = None,
    destination: Mapping[str, Any] | None = None,
    **optional: Any,
) -> JsonObject:
    """Build a CELL reference while keeping optional TES3 fields optional."""

    result: JsonObject = {
        "mast_index": int(mast_index),
        "refr_index": int(refr_index),
        "id": record_id,
        "temporary": bool(temporary),
        "translation": _vec3(translation),
        "rotation": _vec3(rotation),
    }
    if scale is not None:
        result["scale"] = float(scale)
    if destination is not None:
        result["destination"] = _copy(destination)
    for key, value in optional.items():
        if value is not None:
            result[key] = _copy(value)
    return result


def build_cell(
    name: str,
    grid: Sequence[Integral],
    *,
    interior: bool = False,
    data_flags: str | None = None,
    region: str | None = None,
    map_color: Sequence[Integral] | None = None,
    water_height: float | None = None,
    atmosphere_data: Mapping[str, Any] | None = None,
    references: Iterable[Mapping[str, Any]] = (),
    flags: str = "",
) -> JsonObject:
    """Build an exterior or interior CELL with an always-present ref array."""

    if len(grid) != 2:
        raise ValueError("cell grid must contain x and y")
    cell_flags = data_flags if data_flags is not None else ("IS_INTERIOR" if interior else "")
    result: JsonObject = {
        "type": "Cell",
        "flags": flags,
        "name": name,
        "data": {"flags": cell_flags, "grid": [int(grid[0]), int(grid[1])]},
        "references": [_copy(reference) for reference in references],
    }
    if region is not None:
        result["region"] = region
    if map_color is not None:
        result["map_color"] = _color(map_color)
    if water_height is not None:
        result["water_height"] = float(water_height)
    if atmosphere_data is not None:
        result["atmosphere_data"] = _copy(atmosphere_data)
    return result


def build_region(
    record_id: str,
    name: str,
    *,
    weather_chances: Mapping[str, Integral] | None = None,
    sleep_creature: str = "",
    map_color: Sequence[Integral] | None = None,
    sounds: Iterable[Sequence[Any]] = (),
    flags: str = "",
) -> JsonObject:
    weather_defaults = {
        "clear": 100,
        "cloudy": 0,
        "foggy": 0,
        "overcast": 0,
        "rain": 0,
        "thunder": 0,
        "ash": 0,
        "blight": 0,
        "snow": 0,
        "blizzard": 0,
    }
    if weather_chances is not None:
        weather_defaults.update({key: int(value) for key, value in weather_chances.items()})
    return {
        "type": "Region",
        "flags": flags,
        "id": record_id,
        "name": name,
        "weather_chances": weather_defaults,
        "sleep_creature": sleep_creature,
        "map_color": _color(map_color),
        "sounds": [_copy(list(sound)) for sound in sounds],
    }


def build_ltex(
    record_id: str,
    index: int,
    file_name: str,
    *,
    flags: str = "",
) -> JsonObject:
    return {
        "type": "LandscapeTexture",
        "flags": flags,
        "id": record_id,
        "index": int(index),
        "file_name": file_name,
    }


def _height_grid(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    rows = list(value)
    if len(rows) != 65 or any(len(list(row)) != 65 for row in rows):
        raise ValueError("LAND heights must be a 65 by 65 grid")
    return [[float(item) for item in row] for row in rows]


def _height_deltas(height_thu: list[list[int]]) -> tuple[float, bytes]:
    offset = float(height_thu[0][0])
    deltas: list[int] = []
    previous_row_start = int(offset)
    for y, row in enumerate(height_thu):
        for x, current in enumerate(row):
            previous = int(offset) if x == 0 and y == 0 else (
                previous_row_start if x == 0 else row[x - 1]
            )
            delta = current - previous
            if not -128 <= delta <= 127:
                raise ValueError(
                    f"LAND height delta at ({x},{y})={delta} is outside signed i8 range"
                )
            deltas.append(delta)
        previous_row_start = row[0]
    return offset, bytes((delta & 0xFF) for delta in deltas)


def _raw_bytes(value: bytes | bytearray | memoryview | Sequence[int], expected: int, label: str) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        try:
            raw = bytes(int(item) & 0xFF for item in value)
        except Exception as exc:
            raise ValueError(f"{label} must be byte-like") from exc
    if len(raw) != expected:
        raise ValueError(f"{label} must contain {expected} bytes, got {len(raw)}")
    return raw


def build_land(
    grid: Sequence[Integral],
    heights: Any | None = None,
    *,
    heights_in_thu: bool = False,
    landscape_flags: str = (
        "USES_VERTEX_HEIGHTS_AND_NORMALS | USES_VERTEX_COLORS | USES_TEXTURES"
    ),
    vertex_normals: bytes | Sequence[int] | None = None,
    world_map_data: bytes | Sequence[int] | None = None,
    vertex_colors: bytes | Sequence[int] | None = None,
    texture_indices: bytes | Sequence[int] | None = None,
    flags: str = "",
) -> JsonObject:
    """Build a complete LAND record.

    ``heights`` is a 65x65 grid in game units by default.  Set
    ``heights_in_thu=True`` to pass the stored 1/8-game-unit values directly.
    All five blob fields are emitted even when the corresponding landscape
    flag is disabled, as required by the non-Option Rust fields.  The
    ``texture_indices`` argument is serialized LAND VTEX order; callers that
    start with OpenMW's normalized row-major tile grid must apply
    ``procgen.espland.transpose_vtex_openmw_to_serialized`` before passing the
    values here.
    """

    if len(grid) != 2:
        raise ValueError("LAND grid must contain x and y")
    if heights is None:
        height_thu = [[0 for _ in range(65)] for _ in range(65)]
    else:
        grid_values = _height_grid(heights)
        if heights_in_thu:
            height_thu = [[int(value) for value in row] for row in grid_values]
        else:
            height_thu = []
            for row in grid_values:
                converted: list[int] = []
                for value in row:
                    if not math.isfinite(value) or value % 8 != 0:
                        raise ValueError(
                            "LAND game-unit heights must be finite multiples of 8"
                        )
                    converted.append(int(value / 8))
                height_thu.append(converted)
    offset, height_deltas = _height_deltas(height_thu)
    normals = _raw_bytes(
        vertex_normals if vertex_normals is not None else bytes([0, 0, 127]) * _LAND_HEIGHT_COUNT,
        _LAND_NORMAL_COUNT,
        "vertex_normals",
    )
    colors = _raw_bytes(
        vertex_colors if vertex_colors is not None else bytes(_LAND_COLOR_COUNT),
        _LAND_COLOR_COUNT,
        "vertex_colors",
    )
    world = _raw_bytes(
        world_map_data if world_map_data is not None else bytes(_LAND_WORLD_MAP_COUNT),
        _LAND_WORLD_MAP_COUNT,
        "world_map_data",
    )
    if texture_indices is None:
        texture_raw = struct.pack("<256H", *([0] * _LAND_TEXTURE_COUNT))
    elif isinstance(texture_indices, (bytes, bytearray, memoryview)):
        texture_raw = bytes(texture_indices)
        if len(texture_raw) != _LAND_TEXTURE_COUNT * 2:
            raise ValueError("texture_indices must contain 512 bytes")
    else:
        indices = [int(item) for item in texture_indices]
        if len(indices) != _LAND_TEXTURE_COUNT or any(not 0 <= item <= 65535 for item in indices):
            raise ValueError("texture_indices must contain 256 u16 values")
        texture_raw = struct.pack("<256H", *indices)
    return {
        "type": "Landscape",
        "flags": flags,
        "grid": [int(grid[0]), int(grid[1])],
        "landscape_flags": landscape_flags,
        "vertex_normals": {"data": _blob(normals)},
        "vertex_heights": {"offset": offset, "data": _blob(height_deltas)},
        "world_map_data": {"data": _blob(world)},
        "vertex_colors": {"data": _blob(colors)},
        "texture_indices": {"data": _blob(texture_raw)},
    }


def decode_land_heights(record: Mapping[str, Any], *, game_units: bool = True) -> list[list[int]]:
    """Decode LAND VHGT deltas, returning game units by default."""

    field = record.get("vertex_heights")
    if not isinstance(field, Mapping) or "offset" not in field or "data" not in field:
        raise ValueError("LAND record has no complete vertex_heights field")
    offset = float(field["offset"])
    if not offset.is_integer():
        raise ValueError("LAND height offset must be an integral THU for decoding")
    deltas = decode_blob(field["data"])
    if len(deltas) != _LAND_HEIGHT_COUNT:
        raise ValueError(f"LAND height data must contain {_LAND_HEIGHT_COUNT} bytes")
    result: list[list[int]] = []
    cursor = 0
    row_start = int(offset)
    for y in range(65):
        row: list[int] = []
        for x in range(65):
            previous = int(offset) if x == 0 and y == 0 else (
                row_start if x == 0 else row[x - 1]
            )
            row.append(previous + struct.unpack("b", deltas[cursor : cursor + 1])[0])
            cursor += 1
        result.append(row)
        row_start = row[0]
    if game_units:
        return [[value * 8 for value in row] for row in result]
    return result


def _landscape_flags_value(value: object) -> int:
    """Parse a tes3conv ``landscape_flags`` string into the DATA u32 value."""
    result = 0
    for token in str(value).split("|"):
        token = token.strip()
        if not token:
            continue
        if token.startswith("0x"):
            result |= int(token, 16)
        elif token == "USES_VERTEX_HEIGHTS_AND_NORMALS":
            result |= 0x1
        elif token == "USES_VERTEX_COLORS":
            result |= 0x2
        elif token == "USES_TEXTURES":
            result |= 0x4
        else:
            raise ValueError(f"unknown landscape flag {token!r}")
    return result


def land_records_from_json(doc: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int], Any]:
    """Load the edited-LAND ``Landscape`` records from a tes3conv JSON document.

    The city generation emits its terrain edits as a masterless tes3conv JSON
    document (``Header`` + ``Landscape`` + ``LandscapeTexture`` records, with
    zstd/base64 blob subrecords) rather than an ESP.  This parses every
    ``Landscape`` record back into the same ``procgen.espland.LandRecord``
    objects that ``espland.load_land`` produces from a plugin, so scatter and
    groundcover generators can consume the city's edited landscape directly
    without a tes3conv/ESP conversion.

    ``vertex_heights`` is stored as ``{"offset": ..., "data": <zstd/base64 of
    the 4225 signed i8 deltas>}``; ``decode_land_heights`` is reused for the
    row-major THU grid.  The VHGT trailing padding bytes are not preserved by
    the JSON format and are emitted as empty (these records are read-only for
    placement; they are not re-authored).
    """

    from .espland import LandRecord, transpose_vtex_serialized_to_openmw

    result: dict[tuple[int, int], LandRecord] = {}
    for record in doc:
        if not isinstance(record, Mapping) or record.get("type") != "Landscape":
            continue
        grid_raw = record.get("grid")
        if not isinstance(grid_raw, (list, tuple)) or len(grid_raw) != 2:
            raise ValueError("Landscape record grid must be [x, y]")
        grid = (int(grid_raw[0]), int(grid_raw[1]))
        heights = decode_land_heights(record, game_units=False)
        height_offset = float(record.get("vertex_heights", {}).get("offset", 0.0))
        texture_raw = decode_blob(record.get("texture_indices"))
        if len(texture_raw) != _LAND_TEXTURE_COUNT * 2:
            raise ValueError(
                f"Landscape {grid} texture data must contain {_LAND_TEXTURE_COUNT * 2} bytes"
            )
        serialized_values = struct.unpack(f"<{_LAND_TEXTURE_COUNT}H", texture_raw)
        texture_indices = transpose_vtex_serialized_to_openmw(serialized_values)
        result[grid] = LandRecord(
            grid=grid,
            flags=_landscape_flags_value(record.get("landscape_flags", "")),
            heights_thu=tuple(tuple(row) for row in heights),
            offset_thu=height_offset,
            texture_indices=texture_indices,
            vhgt_tail=b"",
            vertex_normals=decode_blob(record.get("vertex_normals")),
            vertex_colors=decode_blob(record.get("vertex_colors")),
            world_map_data=decode_blob(record.get("world_map_data")),
        )
    return result


def build_container(
    record_id: str,
    *,
    name: str = "",
    mesh: str = "",
    script: str = "",
    encumbrance: float = 0.0,
    inventory: Iterable[Sequence[Any]] = (),
    flags: str = "PERSISTENT",
    container_flags: str = "IS_BASE",
) -> JsonObject:
    return {
        "type": "Container",
        "flags": flags,
        "id": record_id,
        "name": name,
        "script": script,
        "mesh": mesh,
        "encumbrance": float(encumbrance),
        "container_flags": container_flags,
        "inventory": [[int(count), str(item_id)] for count, item_id in inventory],
    }


def build_door(
    record_id: str,
    *,
    name: str = "Door",
    mesh: str = "",
    script: str = "",
    open_sound: str = "",
    close_sound: str = "",
    flags: str = "PERSISTENT",
) -> JsonObject:
    return {
        "type": "Door",
        "flags": flags,
        "id": record_id,
        "name": name,
        "script": script,
        "mesh": mesh,
        "open_sound": open_sound,
        "close_sound": close_sound,
    }


def _default_ai_package() -> JsonObject:
    return {
        "type": "Wander",
        "distance": 0,
        "duration": 5,
        "game_hour": 0,
        "idle2": 0,
        "idle3": 0,
        "idle4": 0,
        "idle5": 0,
        "idle6": 0,
        "idle7": 0,
        "idle8": 0,
        "idle9": 0,
        "reset": 1,
    }


def build_npc(
    record_id: str,
    *,
    name: str = "",
    mesh: str = "",
    script: str = "",
    inventory: Iterable[Sequence[Any]] = (),
    spells: Iterable[str] = (),
    ai_data: Mapping[str, Any] | None = None,
    ai_packages: Iterable[Mapping[str, Any]] | None = None,
    travel_destinations: Iterable[Mapping[str, Any]] = (),
    race: str = "Imperial",
    class_id: str = "Commoner",
    faction: str = "",
    head: str = "",
    hair: str = "",
    npc_flags: str = "IS_BASE | AUTO_CALCULATE",
    blood_type: int = 0,
    data: Mapping[str, Any] | None = None,
    stats: Mapping[str, Any] | None = None,
    flags: str = "PERSISTENT",
) -> JsonObject:
    npc_data = {
        "level": 1,
        "disposition": 50,
        "reputation": 0,
        "rank": 0,
        "gold": 0,
    }
    if data is not None:
        npc_data.update(_copy(dict(data)))
    if stats is not None:
        npc_data["stats"] = _copy(dict(stats))
    ai = {"hello": 30, "fight": 30, "flee": 30, "alarm": 0, "services": ""}
    if ai_data is not None:
        ai.update(_copy(dict(ai_data)))
    return {
        "type": "Npc",
        "flags": flags,
        "id": record_id,
        "name": name,
        "script": script,
        "mesh": mesh,
        "inventory": [[int(count), str(item_id)] for count, item_id in inventory],
        "spells": [str(item) for item in spells],
        "ai_data": ai,
        "ai_packages": [_copy(item) for item in (ai_packages if ai_packages is not None else [_default_ai_package()])],
        "travel_destinations": [_copy(item) for item in travel_destinations],
        "race": race,
        "class": class_id,
        "faction": faction,
        "head": head,
        "hair": hair,
        "npc_flags": npc_flags,
        "blood_type": int(blood_type),
        "data": npc_data,
    }


def build_dialogue(record_id: str, dialogue_type: str = "Greeting", *, flags: str = "") -> JsonObject:
    return {
        "type": "Dialogue",
        "flags": flags,
        "id": record_id,
        "dialogue_type": dialogue_type,
    }


def build_dialogue_info(
    record_id: str,
    text: str,
    *,
    dialogue_type: str = "Greeting",
    prev_id: str = "",
    next_id: str = "",
    disposition: int = 0,
    speaker_rank: int = -1,
    speaker_sex: str = "Any",
    player_rank: int = -1,
    speaker_id: str = "",
    speaker_race: str = "",
    speaker_class: str = "",
    speaker_faction: str = "",
    speaker_cell: str = "",
    player_faction: str = "",
    sound_path: str = "",
    filters: Iterable[Mapping[str, Any]] = (),
    script_text: str = "",
    quest_state: str | None = None,
    flags: str = "",
) -> JsonObject:
    result: JsonObject = {
        "type": "DialogueInfo",
        "flags": flags,
        "id": record_id,
        "prev_id": prev_id,
        "next_id": next_id,
        "data": {
            "dialogue_type": dialogue_type,
            "disposition": int(disposition),
            "speaker_rank": int(speaker_rank),
            "speaker_sex": speaker_sex,
            "player_rank": int(player_rank),
        },
        "speaker_id": speaker_id,
        "speaker_race": speaker_race,
        "speaker_class": speaker_class,
        "speaker_faction": speaker_faction,
        "speaker_cell": speaker_cell,
        "player_faction": player_faction,
        "sound_path": sound_path,
        "text": text,
        "filters": [_copy(item) for item in filters],
        "script_text": script_text,
    }
    if quest_state is not None:
        result["quest_state"] = quest_state
    return result


def build_pathgrid(
    cell: str,
    grid: Sequence[Integral],
    points: Iterable[Mapping[str, Any]] = (),
    connections: Iterable[Iterable[Integral]] | None = None,
    *,
    granularity: int = 256,
    flags: str = "",
) -> JsonObject:
    point_list = [_copy(point) for point in points]
    adjacency = [list(map(int, values)) for values in (connections or [[] for _ in point_list])]
    if len(adjacency) != len(point_list):
        raise ValueError("PathGrid connections must contain one list per point")
    normalized_points: list[JsonObject] = []
    for point, linked in zip(point_list, adjacency):
        item = {
            "location": [int(value) for value in point.get("location", [0, 0, 0])],
            "auto_generated": int(point.get("auto_generated", 0)),
            "connection_count": len(linked),
        }
        normalized_points.append(item)
    flat = [sum(len(values) for values in adjacency)]
    flat.extend(value for values in adjacency for value in values)
    packed = struct.pack(f"<{len(flat)}I", *flat)
    return {
        "type": "PathGrid",
        "flags": flags,
        "cell": cell,
        "data": {
            "grid": [int(grid[0]), int(grid[1])],
            "granularity": int(granularity),
            "point_count": len(normalized_points),
        },
        "points": normalized_points,
        "connections": _blob(packed),
    }


_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "Header": ("type", "flags", "version", "file_type", "author", "description", "num_objects", "masters"),
    "Static": ("type", "flags", "id", "mesh"),
    "Cell": ("type", "flags", "name", "data", "references"),
    "Region": ("type", "flags", "id", "name", "weather_chances", "sleep_creature", "map_color", "sounds"),
    "LandscapeTexture": ("type", "flags", "id", "index", "file_name"),
    "Landscape": ("type", "flags", "grid", "landscape_flags", "vertex_normals", "vertex_heights", "world_map_data", "vertex_colors", "texture_indices"),
    "Door": ("type", "flags", "id", "name", "script", "mesh", "open_sound", "close_sound"),
    "Container": ("type", "flags", "id", "name", "script", "mesh", "encumbrance", "container_flags", "inventory"),
    "Npc": ("type", "flags", "id", "name", "script", "mesh", "inventory", "spells", "ai_data", "ai_packages", "travel_destinations", "race", "class", "faction", "head", "hair", "npc_flags", "blood_type", "data"),
    "Dialogue": ("type", "flags", "id", "dialogue_type"),
    "DialogueInfo": ("type", "flags", "id", "prev_id", "next_id", "data", "speaker_id", "speaker_race", "speaker_class", "speaker_faction", "speaker_cell", "player_faction", "sound_path", "text", "filters", "script_text"),
    "PathGrid": ("type", "flags", "cell", "data", "points", "connections"),
}

_SUPPORTED_TYPES = frozenset(_REQUIRED_FIELDS)


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _add(issues: list[Issue], path: str, message: str, code: str = "invalid") -> None:
    issues.append(Issue(path, message, code))


def _required(record: Mapping[str, Any], record_path: str, issues: list[Issue]) -> None:
    record_type = record.get("type")
    for field in _REQUIRED_FIELDS.get(str(record_type), ()):
        if field not in record:
            _add(issues, f"{record_path}.{field}", "required non-Option field is missing", "missing-field")


def _string(record: Mapping[str, Any], field: str, path: str, issues: list[Issue]) -> None:
    if field in record and not isinstance(record[field], str):
        _add(issues, f"{path}.{field}", "must be a string", "type")


def _flag(value: Any, allowed: set[str], path: str, issues: list[Issue]) -> None:
    if not isinstance(value, str):
        _add(issues, path, "must be a bitflag string", "type")
        return
    if not value:
        return
    for part in value.split("|"):
        name = part.strip()
        if not name:
            _add(issues, path, "contains an empty flag name", "enum")
        elif name.startswith("0x"):
            try:
                int(name[2:], 16)
            except ValueError:
                _add(issues, path, f"invalid hexadecimal flag {name!r}", "enum")
        elif name not in allowed:
            _add(issues, path, f"unknown flag {name!r}", "enum")


def _array(value: Any, length: int, path: str, issues: list[Issue], kind: str = "number") -> None:
    if not isinstance(value, list) or len(value) != length:
        _add(issues, path, f"must be a {length}-item array", "type")
        return
    for index, item in enumerate(value):
        good = _is_number(item) if kind == "number" else _is_int(item)
        if not good:
            _add(issues, f"{path}[{index}]", f"must be a {kind}", "type")


def _byte_array(value: Any, path: str, issues: list[Issue]) -> None:
    if not isinstance(value, list) or len(value) != 4:
        _add(issues, path, "must be a four-byte array", "type")
        return
    for index, item in enumerate(value):
        if not _is_int(item) or not 0 <= int(item) <= 255:
            _add(issues, f"{path}[{index}]", "must be an integer in 0..255", "range")


def _blob_length(value: Any, expected: int, path: str, issues: list[Issue]) -> bytes | None:
    try:
        raw = decode_blob(value)
    except ValueError as exc:
        _add(issues, path, str(exc), "blob")
        return None
    if len(raw) != expected:
        _add(issues, path, f"decoded blob must contain {expected} bytes, got {len(raw)}", "length")
    return raw


def _validate_reference(reference: Any, path: str, issues: list[Issue]) -> None:
    required = ("mast_index", "refr_index", "id", "temporary", "translation", "rotation")
    if not isinstance(reference, Mapping):
        _add(issues, path, "must be an object", "type")
        return
    for field in required:
        if field not in reference:
            _add(issues, f"{path}.{field}", "required reference field is missing", "missing-field")
    if "mast_index" in reference and (not _is_int(reference["mast_index"]) or not 0 <= int(reference["mast_index"]) <= 255):
        _add(issues, f"{path}.mast_index", "must be a u8-compatible integer", "range")
    if "refr_index" in reference and (not _is_int(reference["refr_index"]) or not 0 <= int(reference["refr_index"]) <= 0xFFFFFF):
        _add(issues, f"{path}.refr_index", "must be a u32 in the low 24 bits", "range")
    if "id" in reference and not isinstance(reference["id"], str):
        _add(issues, f"{path}.id", "must be a string", "type")
    if "temporary" in reference and not isinstance(reference["temporary"], bool):
        _add(issues, f"{path}.temporary", "must be a boolean", "type")
    if "translation" in reference:
        _array(reference["translation"], 3, f"{path}.translation", issues)
    if "rotation" in reference:
        _array(reference["rotation"], 3, f"{path}.rotation", issues)
    if "scale" in reference and (not _is_number(reference["scale"]) or not 0.5 <= float(reference["scale"]) <= 2.0):
        _add(issues, f"{path}.scale", "must be a number in the TES3 0.5..2.0 range", "range")
    if "destination" in reference:
        destination = reference["destination"]
        if not isinstance(destination, Mapping):
            _add(issues, f"{path}.destination", "must be an object", "type")
        else:
            for field in ("translation", "rotation", "cell"):
                if field not in destination:
                    _add(issues, f"{path}.destination.{field}", "required destination field is missing", "missing-field")
            if "translation" in destination:
                _array(destination["translation"], 3, f"{path}.destination.translation", issues)
            if "rotation" in destination:
                _array(destination["rotation"], 3, f"{path}.destination.rotation", issues)
            if "cell" in destination and not isinstance(destination["cell"], str):
                _add(issues, f"{path}.destination.cell", "must be a string", "type")


def _validate_ai_package(package: Any, path: str, issues: list[Issue]) -> None:
    if not isinstance(package, Mapping) or not isinstance(package.get("type"), str):
        _add(issues, path, "must be an internally tagged AI package", "type")
        return
    package_type = package["type"]
    required: dict[str, tuple[str, ...]] = {
        "Wander": ("distance", "duration", "game_hour", "idle2", "idle3", "idle4", "idle5", "idle6", "idle7", "idle8", "idle9", "reset"),
        "Travel": ("location", "reset"),
        # CNDT/cell is an Option for Escort and Follow in the tes3 crate.
        "Escort": ("location", "duration", "target", "reset"),
        "Follow": ("location", "duration", "target", "reset"),
        "Activate": ("target", "reset"),
    }
    if package_type not in required:
        _add(issues, f"{path}.type", f"unknown AI package type {package_type!r}", "enum")
        return
    for field in required[package_type]:
        if field not in package:
            _add(issues, f"{path}.{field}", "required AI package field is missing", "missing-field")
    if package_type == "Travel" or package_type in {"Escort", "Follow"}:
        if "location" in package:
            _array(package["location"], 3, f"{path}.location", issues)
    if "target" in package and not isinstance(package["target"], str):
        _add(issues, f"{path}.target", "must be a string", "type")
    if "cell" in package and not isinstance(package["cell"], str):
        _add(issues, f"{path}.cell", "must be a string", "type")


def validate(doc: PluginDoc | Mapping[str, Any]) -> list[Issue]:
    """Validate supported records and cross-record TES3 linkage rules.

    This is a hard-gate validator: unknown record types, omitted non-Option
    fields, invalid enum strings, malformed blobs, broken DIAL/INFO adjacency,
    bad PathGrid counts, unknown cell regions, and invalid door destinations are
    all returned as issues.  It does not inspect the filesystem; use
    :func:`procgen.meshcheck.check_refs` for that separate read-only gate.
    """

    records: Any = [doc] if isinstance(doc, Mapping) else doc
    issues: list[Issue] = []
    if not isinstance(records, list):
        return [Issue("document", "must be a top-level JSON array", "type")]
    if not records:
        return [Issue("document", "must contain a Header record", "missing-header")]
    ids_by_type: dict[str, set[str]] = {}
    region_ids: set[str] = set()
    cells: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        path = f"records[{index}]"
        if not isinstance(record, Mapping):
            _add(issues, path, "record must be an object", "type")
            continue
        record_type = record.get("type")
        if record_type not in _SUPPORTED_TYPES:
            _add(issues, f"{path}.type", f"unsupported record type {record_type!r}", "unsupported")
            continue
        _required(record, path, issues)
        _flag(record.get("flags"), _OBJECT_FLAGS, f"{path}.flags", issues)
        record_id = record.get("id")
        if isinstance(record_id, str):
            ids_by_type.setdefault(str(record_type), set())
            if record_id in ids_by_type[str(record_type)]:
                _add(issues, f"{path}.id", f"duplicate {record_type} id {record_id!r}", "duplicate-id")
            ids_by_type[str(record_type)].add(record_id)
        if record_type == "Header":
            _validate_header(record, path, issues)
        elif record_type == "Static":
            _validate_object(record, path, issues, extra_flags=None)
        elif record_type == "Door":
            _validate_object(record, path, issues, extra_flags=None)
            for field in ("name", "script", "open_sound", "close_sound", "mesh"):
                _string(record, field, path, issues)
        elif record_type == "Container":
            _validate_container(record, path, issues)
        elif record_type == "Npc":
            _validate_npc(record, path, issues)
        elif record_type == "Cell":
            _validate_cell(record, path, issues)
            if isinstance(record, Mapping):
                cells.append(record)
        elif record_type == "Region":
            _validate_region(record, path, issues)
            if isinstance(record_id, str):
                region_ids.add(record_id)
        elif record_type == "LandscapeTexture":
            _validate_ltex(record, path, issues)
        elif record_type == "Landscape":
            _validate_land(record, path, issues)
        elif record_type == "Dialogue":
            _validate_dialogue(record, path, issues)
        elif record_type == "DialogueInfo":
            _validate_dialogue_info(record, path, issues)
        elif record_type == "PathGrid":
            _validate_pathgrid(record, path, issues)
    if records[0].get("type") != "Header" if isinstance(records[0], Mapping) else True:
        _add(issues, "records[0]", "Header must be the first record", "header-order")
    for index, cell in enumerate(cells):
        path = f"records[{records.index(cell)}]"
        region = cell.get("region")
        if isinstance(region, str) and region and region not in region_ids:
            _add(issues, f"{path}.region", f"unknown REGN id {region!r}", "unknown-region")
        refs = cell.get("references")
        if isinstance(refs, list):
            seen_refs: set[tuple[int, int]] = set()
            for ref_index, reference in enumerate(refs):
                if not isinstance(reference, Mapping):
                    continue
                if _is_int(reference.get("mast_index")) and _is_int(reference.get("refr_index")):
                    key = (int(reference["mast_index"]), int(reference["refr_index"]))
                    if key in seen_refs:
                        _add(issues, f"{path}.references[{ref_index}]", "duplicate reference index in cell", "duplicate-ref")
                    seen_refs.add(key)
    _validate_dialogue_linkage(records, issues)
    _validate_door_destinations(records, issues)
    return issues


def _validate_header(record: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    if "version" in record and not _is_number(record["version"]):
        _add(issues, f"{path}.version", "must be a number", "type")
    if record.get("file_type") not in {"Esp", "Esm", "Ess"}:
        _add(issues, f"{path}.file_type", "must be Esp, Esm, or Ess", "enum")
    for field in ("author", "description"):
        _string(record, field, path, issues)
    if "num_objects" in record and (not _is_int(record["num_objects"]) or int(record["num_objects"]) < 0):
        _add(issues, f"{path}.num_objects", "must be a non-negative u32", "range")
    masters = record.get("masters")
    if not isinstance(masters, list):
        _add(issues, f"{path}.masters", "must be an array", "type")
    else:
        for index, master in enumerate(masters):
            if not isinstance(master, list) or len(master) != 2 or not isinstance(master[0], str) or not _is_int(master[1]) or int(master[1]) < 0:
                _add(issues, f"{path}.masters[{index}]", "must be [string, non-negative integer]", "type")


def _validate_object(record: Mapping[str, Any], path: str, issues: list[Issue], extra_flags: set[str] | None) -> None:
    for field in ("id", "mesh"):
        _string(record, field, path, issues)


def _validate_container(record: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    for field in ("id", "name", "script", "mesh"):
        _string(record, field, path, issues)
    _flag(record.get("container_flags"), _CONTAINER_FLAGS, f"{path}.container_flags", issues)
    if "encumbrance" in record and not _is_number(record["encumbrance"]):
        _add(issues, f"{path}.encumbrance", "must be a number", "type")
    inventory = record.get("inventory")
    if not isinstance(inventory, list):
        _add(issues, f"{path}.inventory", "must be an array", "type")
    else:
        for index, item in enumerate(inventory):
            if not isinstance(item, list) or len(item) != 2 or not _is_int(item[0]) or not isinstance(item[1], str):
                _add(issues, f"{path}.inventory[{index}]", "must be [integer count, string id]", "type")


def _validate_npc(record: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    for field in ("id", "name", "script", "mesh", "race", "class", "faction", "head", "hair"):
        _string(record, field, path, issues)
    _flag(record.get("npc_flags"), _NPC_FLAGS, f"{path}.npc_flags", issues)
    if not _is_int(record.get("blood_type")) or not 0 <= int(record.get("blood_type", -1)) <= 7:
        _add(issues, f"{path}.blood_type", "must be a u8 blood type in 0..7", "range")
    inventory = record.get("inventory")
    if not isinstance(inventory, list):
        _add(issues, f"{path}.inventory", "must be an array", "type")
    else:
        for index, item in enumerate(inventory):
            if not isinstance(item, list) or len(item) != 2 or not _is_int(item[0]) or not isinstance(item[1], str):
                _add(issues, f"{path}.inventory[{index}]", "must be [integer count, string id]", "type")
    if not isinstance(record.get("spells"), list) or any(not isinstance(item, str) for item in record["spells"]):
        _add(issues, f"{path}.spells", "must be an array of strings", "type")
    ai = record.get("ai_data")
    if not isinstance(ai, Mapping):
        _add(issues, f"{path}.ai_data", "must be an object", "type")
    else:
        for field in ("hello", "fight", "flee", "alarm"):
            if field not in ai or not _is_int(ai[field]):
                _add(issues, f"{path}.ai_data.{field}", "must be an integer", "type")
        _flag(ai.get("services"), _SERVICE_FLAGS, f"{path}.ai_data.services", issues)
    packages = record.get("ai_packages")
    if not isinstance(packages, list):
        _add(issues, f"{path}.ai_packages", "must be an array", "type")
    else:
        for index, package in enumerate(packages):
            _validate_ai_package(package, f"{path}.ai_packages[{index}]", issues)
    destinations = record.get("travel_destinations")
    if not isinstance(destinations, list):
        _add(issues, f"{path}.travel_destinations", "must be an array", "type")
    else:
        for index, destination in enumerate(destinations):
            if not isinstance(destination, Mapping):
                _add(issues, f"{path}.travel_destinations[{index}]", "must be an object", "type")
                continue
            for field in ("translation", "rotation", "cell"):
                if field not in destination:
                    _add(issues, f"{path}.travel_destinations[{index}].{field}", "required field is missing", "missing-field")
            if "translation" in destination:
                _array(destination["translation"], 3, f"{path}.travel_destinations[{index}].translation", issues)
            if "rotation" in destination:
                _array(destination["rotation"], 3, f"{path}.travel_destinations[{index}].rotation", issues)
            if "cell" in destination and not isinstance(destination["cell"], str):
                _add(issues, f"{path}.travel_destinations[{index}].cell", "must be a string", "type")
    data = record.get("data")
    if not isinstance(data, Mapping):
        _add(issues, f"{path}.data", "must be an object", "type")
    else:
        for field in ("level", "disposition", "reputation", "rank", "gold"):
            if field not in data or not _is_int(data[field]):
                _add(issues, f"{path}.data.{field}", "must be an integer", "type")
        if "stats" in data:
            stats = data["stats"]
            if not isinstance(stats, Mapping):
                _add(issues, f"{path}.data.stats", "must be an object", "type")
            else:
                if not isinstance(stats.get("attributes"), list) or len(stats.get("attributes", [])) != 8:
                    _add(issues, f"{path}.data.stats.attributes", "must contain 8 values", "length")
                if not isinstance(stats.get("skills"), list) or len(stats.get("skills", [])) != 27:
                    _add(issues, f"{path}.data.stats.skills", "must contain 27 values", "length")
                for field in ("health", "magicka", "fatigue"):
                    if field not in stats or not _is_int(stats[field]):
                        _add(issues, f"{path}.data.stats.{field}", "must be an integer", "type")


def _validate_cell(record: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    _string(record, "flags", path, issues)
    _string(record, "name", path, issues)
    data = record.get("data")
    if not isinstance(data, Mapping):
        _add(issues, f"{path}.data", "must be an object", "type")
    else:
        _flag(data.get("flags"), _CELL_FLAGS, f"{path}.data.flags", issues)
        _array(data.get("grid"), 2, f"{path}.data.grid", issues, "integer")
    if "region" in record:
        _string(record, "region", path, issues)
    if "map_color" in record:
        _byte_array(record["map_color"], f"{path}.map_color", issues)
    if "water_height" in record and not _is_number(record["water_height"]):
        _add(issues, f"{path}.water_height", "must be a number", "type")
    if "atmosphere_data" in record:
        atmosphere = record["atmosphere_data"]
        if not isinstance(atmosphere, Mapping):
            _add(issues, f"{path}.atmosphere_data", "must be an object", "type")
        else:
            for field in ("ambient_color", "sunlight_color", "fog_color"):
                if field not in atmosphere:
                    _add(issues, f"{path}.atmosphere_data.{field}", "required field is missing", "missing-field")
                else:
                    _byte_array(atmosphere[field], f"{path}.atmosphere_data.{field}", issues)
            if "fog_density" not in atmosphere or not _is_number(atmosphere.get("fog_density")):
                _add(issues, f"{path}.atmosphere_data.fog_density", "must be a number", "type")
    refs = record.get("references")
    if not isinstance(refs, list):
        _add(issues, f"{path}.references", "must be an array of references", "type")
    else:
        for index, reference in enumerate(refs):
            _validate_reference(reference, f"{path}.references[{index}]", issues)


def _validate_region(record: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    for field in ("id", "name", "sleep_creature"):
        _string(record, field, path, issues)
    _byte_array(record.get("map_color"), f"{path}.map_color", issues)
    weather = record.get("weather_chances")
    weather_fields = ("clear", "cloudy", "foggy", "overcast", "rain", "thunder", "ash", "blight", "snow", "blizzard")
    if not isinstance(weather, Mapping):
        _add(issues, f"{path}.weather_chances", "must be an object", "type")
    else:
        for field in weather_fields:
            if field not in weather:
                _add(issues, f"{path}.weather_chances.{field}", "required weather field is missing", "missing-field")
            elif not _is_int(weather[field]) or not 0 <= int(weather[field]) <= 255:
                _add(issues, f"{path}.weather_chances.{field}", "must be a byte", "range")
    sounds = record.get("sounds")
    if not isinstance(sounds, list):
        _add(issues, f"{path}.sounds", "must be an array", "type")
    else:
        for index, sound in enumerate(sounds):
            if not isinstance(sound, list) or len(sound) != 2 or not isinstance(sound[0], str) or not _is_int(sound[1]) or not 0 <= int(sound[1]) <= 255:
                _add(issues, f"{path}.sounds[{index}]", "must be [string id, byte chance]", "type")


def _validate_ltex(record: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    for field in ("id", "file_name"):
        _string(record, field, path, issues)
    if "index" in record and (not _is_int(record["index"]) or int(record["index"]) < 0):
        _add(issues, f"{path}.index", "must be a non-negative u32", "range")


def _validate_land(record: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    _array(record.get("grid"), 2, f"{path}.grid", issues, "integer")
    _flag(record.get("landscape_flags"), _LANDSCAPE_FLAGS, f"{path}.landscape_flags", issues)
    normals = record.get("vertex_normals")
    if isinstance(normals, Mapping):
        _blob_length(normals.get("data"), _LAND_NORMAL_COUNT, f"{path}.vertex_normals.data", issues)
    else:
        _add(issues, f"{path}.vertex_normals", "must be an object", "type")
    heights = record.get("vertex_heights")
    if isinstance(heights, Mapping):
        if "offset" not in heights or not _is_number(heights.get("offset")):
            _add(issues, f"{path}.vertex_heights.offset", "must be a number", "type")
        _blob_length(heights.get("data"), _LAND_HEIGHT_COUNT, f"{path}.vertex_heights.data", issues)
    else:
        _add(issues, f"{path}.vertex_heights", "must be an object", "type")
    for field, expected in (("world_map_data", _LAND_WORLD_MAP_COUNT), ("vertex_colors", _LAND_COLOR_COUNT), ("texture_indices", _LAND_TEXTURE_COUNT * 2)):
        value = record.get(field)
        if isinstance(value, Mapping):
            _blob_length(value.get("data"), expected, f"{path}.{field}.data", issues)
        else:
            _add(issues, f"{path}.{field}", "must be an object", "type")


def _validate_dialogue(record: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    _string(record, "id", path, issues)
    if record.get("dialogue_type") not in _DIALOGUE_TYPES:
        _add(issues, f"{path}.dialogue_type", "unknown dialogue type", "enum")


def _validate_dialogue_info(record: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    for field in ("id", "prev_id", "next_id", "speaker_id", "speaker_race", "speaker_class", "speaker_faction", "speaker_cell", "player_faction", "sound_path", "text", "script_text"):
        _string(record, field, path, issues)
    data = record.get("data")
    if not isinstance(data, Mapping):
        _add(issues, f"{path}.data", "must be an object", "type")
    else:
        if data.get("dialogue_type") not in _DIALOGUE_TYPES:
            _add(issues, f"{path}.data.dialogue_type", "unknown dialogue type", "enum")
        if not _is_int(data.get("disposition")) or not -128 <= int(data.get("disposition", 0)) <= 127:
            _add(issues, f"{path}.data.disposition", "must be an i8", "range")
        if not _is_int(data.get("speaker_rank")) or not -128 <= int(data.get("speaker_rank", 0)) <= 127:
            _add(issues, f"{path}.data.speaker_rank", "must be an i8", "range")
        if data.get("speaker_sex") not in _SEXES:
            _add(issues, f"{path}.data.speaker_sex", "unknown sex enum", "enum")
        if not _is_int(data.get("player_rank")) or not -128 <= int(data.get("player_rank", 0)) <= 127:
            _add(issues, f"{path}.data.player_rank", "must be an i8", "range")
    if "quest_state" in record and record["quest_state"] not in {"Name", "Finished", "Restart"}:
        _add(issues, f"{path}.quest_state", "unknown quest state", "enum")
    filters = record.get("filters")
    if not isinstance(filters, list):
        _add(issues, f"{path}.filters", "must be an array", "type")
    else:
        for index, item in enumerate(filters):
            filter_path = f"{path}.filters[{index}]"
            if not isinstance(item, Mapping):
                _add(issues, filter_path, "must be an object", "type")
                continue
            for field in ("index", "filter_type", "function", "comparison", "id", "value"):
                if field not in item:
                    _add(issues, f"{filter_path}.{field}", "required filter field is missing", "missing-field")
            if "index" in item and not _is_int(item["index"]):
                _add(issues, f"{filter_path}.index", "must be an integer", "type")
            if "filter_type" in item and item["filter_type"] not in _FILTER_TYPES:
                _add(issues, f"{filter_path}.filter_type", "unknown filter type", "enum")
            if "function" in item and not isinstance(item["function"], str):
                _add(issues, f"{filter_path}.function", "must be a string", "type")
            if "comparison" in item and item["comparison"] not in _COMPARISONS:
                _add(issues, f"{filter_path}.comparison", "unknown comparison", "enum")
            if "id" in item and not isinstance(item["id"], str):
                _add(issues, f"{filter_path}.id", "must be a string", "type")
            value = item.get("value")
            if not isinstance(value, Mapping) or value.get("type") not in {"Float", "Integer"} or not _is_number(value.get("data")):
                _add(issues, f"{filter_path}.value", "must be {type: Float|Integer, data: number}", "type")


def _validate_pathgrid(record: Mapping[str, Any], path: str, issues: list[Issue]) -> None:
    _string(record, "cell", path, issues)
    data = record.get("data")
    if not isinstance(data, Mapping):
        _add(issues, f"{path}.data", "must be an object", "type")
        return
    _array(data.get("grid"), 2, f"{path}.data.grid", issues, "integer")
    for field in ("granularity", "point_count"):
        if field not in data or not _is_int(data[field]) or not 0 <= int(data[field]) <= 65535:
            _add(issues, f"{path}.data.{field}", "must be a u16", "range")
    points = record.get("points")
    if not isinstance(points, list):
        _add(issues, f"{path}.points", "must be an array", "type")
        return
    if _is_int(data.get("point_count")) and int(data["point_count"]) != len(points):
        _add(issues, f"{path}.data.point_count", "does not equal points length", "count-mismatch")
    counts: list[int] = []
    for index, point in enumerate(points):
        point_path = f"{path}.points[{index}]"
        if not isinstance(point, Mapping):
            _add(issues, point_path, "must be an object", "type")
            continue
        _array(point.get("location"), 3, f"{point_path}.location", issues, "integer")
        if point.get("auto_generated") not in {0, 1}:
            _add(issues, f"{point_path}.auto_generated", "must be 0 or 1", "enum")
        if not _is_int(point.get("connection_count")) or int(point.get("connection_count", -1)) < 0:
            _add(issues, f"{point_path}.connection_count", "must be a non-negative integer", "range")
        counts.append(int(point.get("connection_count", 0)))
    raw = _blob_length(record.get("connections"), 4 * (1 + sum(counts)), f"{path}.connections", issues)
    if raw is not None and len(raw) >= 4:
        values = list(struct.unpack(f"<{len(raw) // 4}I", raw))
        if values[0] != sum(counts):
            _add(issues, f"{path}.connections", "leading count does not equal point connection counts", "count-mismatch")
        if values[0] != len(values) - 1:
            _add(issues, f"{path}.connections", "leading count does not equal payload length minus one", "count-mismatch")
        if any(value >= len(points) for value in values[1:]):
            _add(issues, f"{path}.connections", "connection points must index the point array", "range")


def _validate_dialogue_linkage(records: list[Any], issues: list[Issue]) -> None:
    info_ids: set[str] = set()
    for index, record in enumerate(records):
        if isinstance(record, Mapping) and record.get("type") == "DialogueInfo" and isinstance(record.get("id"), str):
            if record["id"] in info_ids:
                _add(issues, f"records[{index}].id", "DialogueInfo ids must be unique", "duplicate-id")
            info_ids.add(record["id"])
    current_dialogue: Mapping[str, Any] | None = None
    saw_info = False
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            current_dialogue = None
            saw_info = False
            continue
        kind = record.get("type")
        if kind == "Dialogue":
            if current_dialogue is not None and not saw_info:
                _add(issues, f"records[{index - 1}]", "Dialogue must be followed by at least one DialogueInfo", "dialogue-link")
            current_dialogue = record
            saw_info = False
        elif kind == "DialogueInfo":
            if current_dialogue is None:
                _add(issues, f"records[{index}]", "DialogueInfo has no preceding Dialogue", "dialogue-link")
            else:
                saw_info = True
                data = record.get("data")
                if isinstance(data, Mapping) and data.get("dialogue_type") != current_dialogue.get("dialogue_type"):
                    _add(issues, f"records[{index}].data.dialogue_type", "does not match preceding Dialogue", "dialogue-link")
        else:
            if current_dialogue is not None and not saw_info:
                _add(issues, f"records[{index}]", "DialogueInfo must be adjacent to its Dialogue", "dialogue-link")
            current_dialogue = None
            saw_info = False
    if current_dialogue is not None and not saw_info:
        _add(issues, f"records[{len(records) - 1}]", "Dialogue must be followed by at least one DialogueInfo", "dialogue-link")
    for index, record in enumerate(records):
        if isinstance(record, Mapping) and record.get("type") == "DialogueInfo":
            for field in ("prev_id", "next_id"):
                value = record.get(field)
                if isinstance(value, str) and value and value not in info_ids:
                    _add(issues, f"records[{index}].{field}", f"unknown DialogueInfo id {value!r}", "dialogue-link")


def _validate_door_destinations(records: list[Any], issues: list[Issue]) -> None:
    cell_names = {record.get("name") for record in records if isinstance(record, Mapping) and record.get("type") == "Cell" and isinstance(record.get("name"), str)}
    regions = {record.get("region") for record in records if isinstance(record, Mapping) and record.get("type") == "Cell" and isinstance(record.get("region"), str) and record.get("region")}
    object_types = {record.get("id"): record.get("type") for record in records if isinstance(record, Mapping) and isinstance(record.get("id"), str)}
    for cell_index, cell in enumerate(records):
        if not isinstance(cell, Mapping) or cell.get("type") != "Cell" or not isinstance(cell.get("references"), list):
            continue
        for ref_index, reference in enumerate(cell["references"]):
            if not isinstance(reference, Mapping) or "destination" not in reference:
                continue
            destination = reference["destination"]
            if isinstance(destination, Mapping) and isinstance(destination.get("cell"), str):
                # An empty destination cell name is the TES3 return-door
                # grammar (interior -> exterior): the door carries DODT and
                # the engine resolves the exit position geometrically, so no
                # DNAM is serialized.  The tes3 crate (rev 51fae82) models
                # this as TravelDestination with a non-Option ``cell: String``
                # whose Save writes DNAM only when the string is non-empty;
                # tes3conv round-trips ``"cell": ""`` as DODT-without-DNAM
                # (proven 2026-08-10, cityforge T0.4 proof).  Only non-empty
                # names must name a cell or exterior region in this document.
                if destination["cell"] and destination["cell"] not in cell_names and destination["cell"] not in regions:
                    _add(issues, f"records[{cell_index}].references[{ref_index}].destination.cell", "door destination does not name a cell or exterior region in this document", "door-link")
            if reference.get("temporary") is True and object_types.get(reference.get("id")) == "Door":
                _add(issues, f"records[{cell_index}].references[{ref_index}].temporary", "door references with destinations must be persistent", "door-link")


def write_json(doc: PluginDoc, path: str | Path, *, indent: int | None = 2) -> None:
    """Write a plugin document without invoking tes3conv."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(doc, handle, indent=indent, ensure_ascii=False, separators=None if indent else (",", ":"))
        handle.write("\n")


def read_json(path: str | Path) -> PluginDoc:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("tes3conv JSON must be a top-level array")
    return value


# Short aliases are useful at call sites that already use TES3 record names.
build_landscape = build_land
build_landscape_texture = build_ltex
build_npc_ = build_npc
build_path_grid = build_pathgrid


__all__ = [
    "Issue",
    "JsonObject",
    "PluginDoc",
    "build_cell",
    "build_container",
    "build_dialogue",
    "build_dialogue_info",
    "build_door",
    "build_land",
    "build_landscape",
    "build_landscape_texture",
    "build_ltex",
    "build_npc",
    "build_pathgrid",
    "build_path_grid",
    "build_reference",
    "build_region",
    "build_static",
    "decode_blob",
    "decode_land_heights",
    "new_plugin",
    "read_json",
    "validate",
    "write_json",
]
