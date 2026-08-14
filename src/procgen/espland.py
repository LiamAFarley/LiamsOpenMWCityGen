"""Streaming TES3 ``LAND``/``VHGT``/``VNML``/``WNAM``/``VCLR``/``VTEX`` and
``LTEX`` reader.

The ESP scanner intentionally does not retain landscape payloads.  Rendering
needs the same payload, however, so this module provides a small, bounded
reader for exterior terrain records.  A TES3 ``VHGT`` is one float32 offset in
THU followed by 65*65 signed, row-major deltas and three trailing padding
bytes.  Each row starts from the first vertex of the previous row; this is the
same gradient convention used by :mod:`procgen.tes3json`.

The reader is read-only and streaming: non-LAND records are discarded as they
are encountered and a caller can request a hard time limit.  Empty LAND
records are retained as records with ``heights_thu=None`` because a few real
plugins use them as placeholders.
``LAND`` also carries the complete normal (``VNML``), world-map (``WNAM``),
and vertex-colour (``VCLR``) payloads.  Those byte fields are retained when a
caller is preparing a LAND override; the reader does not synthesize defaults
for records that actually contain them.  ``LAND/VTEX`` contains 256
little-endian ``u16`` values in serialized
macro-block order.  OpenMW transposes those values while loading the LAND
record: ``LandRecord.texture_indices`` is therefore the engine's row-major
16 by 16 tile order, not the byte order on disk.  A nonzero in-memory VTEX
value is one greater than the owning plugin's LTEX ``INTV`` index; zero is the
base ``_land_default.dds`` texture and does not perform an LTEX lookup.
``LTEX`` records are parsed from the plugin that owns the LAND record.  This
module deliberately does not invent a global load-order LTEX table.  The
``load_ltex_with_masters`` function remains only as an explicitly non-engine
overlay convenience for legacy visualization callers.  If an LTEX record has
no ``INTV`` field, its zero-based LTEX record ordinal is used as a deterministic
fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
import time
from typing import BinaryIO, Iterable, Iterator, Mapping, Sequence, TYPE_CHECKING

from .espscan import RECORD_HEADER_SIZE, SUBRECORD_HEADER_SIZE, TES3ParseError

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from .worldcontext import WorldContext


LAND_SIDE = 65
LAND_HEIGHT_COUNT = LAND_SIDE * LAND_SIDE
LAND_TEXTURE_SIDE = 16
LAND_TEXTURE_COUNT = LAND_TEXTURE_SIDE * LAND_TEXTURE_SIDE
VTEX_STANDARD_SIZE = LAND_TEXTURE_COUNT * 2
LAND_NORMAL_BYTES = LAND_HEIGHT_COUNT * 3
LAND_COLOR_BYTES = LAND_HEIGHT_COUNT * 3
LAND_WORLD_MAP_BYTES = 9 * 9
VHGT_MIN_SIZE = 4 + LAND_HEIGHT_COUNT
VHGT_STANDARD_SIZE = 4232
THU_TO_GU = 8
CELL_SIZE_GAME_UNITS = 8192.0
LAND_VERTEX_SPACING_GAME_UNITS = CELL_SIZE_GAME_UNITS / (LAND_SIDE - 1)
BASE_LAND_TEXTURE_NAME = "_land_default"
BASE_LAND_TEXTURE_PATH = "_land_default.dds"


def _validated_vtex_values(values: Sequence[int]) -> tuple[int, ...]:
    """Return one checked 256-value VTEX sequence for the permutations."""

    if len(values) != LAND_TEXTURE_COUNT:
        raise ValueError(
            f"VTEX values must contain {LAND_TEXTURE_COUNT} entries, got {len(values)}"
        )
    result = tuple(int(value) for value in values)
    if any(value < 0 or value > 65535 for value in result):
        raise ValueError("VTEX values must be unsigned 16-bit integers")
    return result


def resolve_vtex_to_ltex_index(raw_vtex: int) -> int | None:
    """Convert one raw VTEX value to an owning-plugin LTEX INTV index.

    This is the single value-semantic boundary shared by LAND consumers and
    diagnostic tools: raw zero is the engine's base texture sentinel, while a
    positive raw value is one greater than the plugin-local LTEX index.
    """

    value = int(raw_vtex)
    if value < 0 or value > 65535:
        raise ValueError("raw VTEX value must be an unsigned 16-bit integer")
    return None if value == 0 else value - 1


def transpose_vtex_serialized_to_openmw(values: Sequence[int]) -> tuple[int, ...]:
    """Transpose serialized LAND VTEX order into OpenMW's row-major order.

    OpenMW's ``transposeTextureData`` walks four 4x4 macro-block rows and
    columns, then walks each block in row-major order.  The resulting output
    is indexed as ``tile_y * 16 + tile_x`` by the terrain sampler.  Keeping
    this exact loop (rather than a visually similar reshape) makes the parser
    faithful to the engine's on-load permutation.
    """

    source = _validated_vtex_values(values)
    output = [0] * LAND_TEXTURE_COUNT
    read_position = 0
    for macro_y in range(4):
        for macro_x in range(4):
            for block_y in range(4):
                for block_x in range(4):
                    output[
                        (macro_y * 4 + block_y) * LAND_TEXTURE_SIDE
                        + (macro_x * 4 + block_x)
                    ] = source[read_position]
                    read_position += 1
    return tuple(output)


def transpose_vtex_openmw_to_serialized(values: Sequence[int]) -> tuple[int, ...]:
    """Inverse the OpenMW VTEX permutation for future LAND authoring.

    ``tes3json.build_land`` currently accepts already-serialized bytes and is
    intentionally unchanged.  New authoring code that starts with an
    OpenMW-order tile grid can use this helper before packing the 256 u16
    values into the LAND ``VTEX`` subrecord.
    """

    source = _validated_vtex_values(values)
    output = [0] * LAND_TEXTURE_COUNT
    write_position = 0
    for macro_y in range(4):
        for macro_x in range(4):
            for block_y in range(4):
                for block_x in range(4):
                    output[write_position] = source[
                        (macro_y * 4 + block_y) * LAND_TEXTURE_SIDE
                        + (macro_x * 4 + block_x)
                    ]
                    write_position += 1
    return tuple(output)


@dataclass(frozen=True)
class LandRecord:
    """One TES3 LAND record with its source payload fields retained.

    ``texture_indices`` is the OpenMW-normalized row-major view.  The three
    byte fields are optional for compatibility with deliberately minimal
    synthetic fixtures, but real full LAND records expose all three.
    """

    grid: tuple[int, int]
    flags: int
    heights_thu: tuple[tuple[int, ...], ...] | None
    offset_thu: float | None
    texture_indices: tuple[int, ...] | None = None
    vhgt_tail: bytes = b""
    vertex_normals: bytes | None = None
    world_map_data: bytes | None = None
    vertex_colors: bytes | None = None
    record_offset: int = 0

    @property
    def has_heights(self) -> bool:
        return self.heights_thu is not None

    @property
    def has_textures(self) -> bool:
        """Whether this LAND has its standard 16x16 VTEX payload."""

        return self.texture_indices is not None

    @property
    def dominant_texture_index(self) -> int | None:
        """Return the lowest raw-VTEX winner among the 256 normalized tiles."""

        if self.texture_indices is None:
            return None
        counts: dict[int, int] = {}
        for value in self.texture_indices:
            counts[value] = counts.get(value, 0) + 1
        return min(counts, key=lambda value: (-counts[value], value))

    def texture_index(self, tile_x: int, tile_y: int) -> int:
        """Return one raw VTEX value in OpenMW-normalized row-major order."""

        if self.texture_indices is None:
            raise ValueError(f"LAND {self.grid} has no VTEX payload")
        if not (0 <= tile_x < LAND_TEXTURE_SIDE and 0 <= tile_y < LAND_TEXTURE_SIDE):
            raise ValueError("LAND texture coordinates must be in [0, 16)")
        return self.texture_indices[tile_y * LAND_TEXTURE_SIDE + tile_x]

    def texture_ltex_index(self, tile_x: int, tile_y: int) -> int | None:
        """Resolve one raw VTEX value to an LTEX index or the base sentinel.

        OpenMW treats raw VTEX ``0`` as ``_land_default.dds`` and treats raw
        value ``N > 0`` as LTEX ``INTV`` index ``N - 1``.  Keeping this
        conversion beside the normalized LAND accessor prevents analysis,
        rendering, and future authoring consumers from duplicating the offset.
        The caller supplies the LTEX table for the LAND-owning plugin.
        """

        return resolve_vtex_to_ltex_index(self.texture_index(tile_x, tile_y))

    @property
    def heights_gu(self) -> tuple[tuple[int, ...], ...] | None:
        """Return a game-unit view without changing the stored THU values."""

        if self.heights_thu is None:
            return None
        return tuple(
            tuple(value * THU_TO_GU for value in row) for row in self.heights_thu
        )

    def height_thu(self, x: int, y: int) -> int:
        if self.heights_thu is None:
            raise ValueError(f"LAND {self.grid} has no VHGT payload")
        if not (0 <= x < LAND_SIDE and 0 <= y < LAND_SIDE):
            raise ValueError("LAND vertex coordinates must be in [0, 65)")
        return self.heights_thu[y][x]


@dataclass(frozen=True)
class LandValidation:
    """Comparison summary for selected LAND vertices."""

    grid: tuple[int, int]
    sample_count: int
    max_abs_delta_thu: int
    max_abs_delta_gu: int
    mismatches: int


@dataclass(frozen=True)
class LandscapeTexture:
    """One LTEX lookup entry from a single source plugin."""

    index: int
    record_id: str
    file_name: str
    record_index: int
    record_offset: int = 0

    @property
    def texture_path(self) -> str:
        """Alias used by renderers when resolving the data-root asset."""

        return self.file_name


def _subrecords(body: bytes) -> Iterator[tuple[bytes, bytes]]:
    pos = 0
    while pos + SUBRECORD_HEADER_SIZE <= len(body):
        tag = body[pos : pos + 4]
        size = struct.unpack_from("<I", body, pos + 4)[0]
        start = pos + SUBRECORD_HEADER_SIZE
        end = start + size
        if end > len(body):
            raise TES3ParseError(f"LAND subrecord {tag!r} overruns record body")
        yield tag, body[start:end]
        pos = end
    if pos != len(body):
        raise TES3ParseError("LAND has trailing bytes after subrecords")


def _decode_vhgt(payload: bytes) -> tuple[float, tuple[tuple[int, ...], ...], bytes]:
    if len(payload) < VHGT_MIN_SIZE:
        raise TES3ParseError(
            f"VHGT payload is {len(payload)} bytes; expected at least {VHGT_MIN_SIZE}"
        )
    offset = struct.unpack_from("<f", payload, 0)[0]
    if not math.isfinite(offset):
        raise TES3ParseError("VHGT offset is not finite")
    deltas = struct.unpack(f"<{LAND_HEIGHT_COUNT}b", payload[4 : 4 + LAND_HEIGHT_COUNT])
    heights: list[tuple[int, ...]] = []
    cursor = 0
    previous_row_start = int(round(offset))
    for y in range(LAND_SIDE):
        row: list[int] = []
        for x in range(LAND_SIDE):
            previous = (
                int(round(offset))
                if x == 0 and y == 0
                else previous_row_start
                if x == 0
                else row[x - 1]
            )
            row.append(previous + deltas[cursor])
            cursor += 1
        heights.append(tuple(row))
        previous_row_start = row[0]
    return offset, tuple(heights), payload[4 + LAND_HEIGHT_COUNT :]


def parse_land(body: bytes, *, record_offset: int = 0) -> LandRecord:
    """Parse one LAND record body, retaining every standard payload field."""

    grid: tuple[int, int] | None = None
    flags = 0
    heights: tuple[tuple[int, ...], ...] | None = None
    offset_thu: float | None = None
    texture_indices: tuple[int, ...] | None = None
    tail = b""
    vertex_normals: bytes | None = None
    world_map_data: bytes | None = None
    vertex_colors: bytes | None = None
    for tag, payload in _subrecords(body):
        if tag == b"INTV":
            if len(payload) < 8:
                raise TES3ParseError("LAND INTV must contain two signed integers")
            grid = tuple(struct.unpack_from("<ii", payload))
        elif tag == b"DATA":
            if len(payload) < 4:
                raise TES3ParseError("LAND DATA must contain a u32 flags value")
            (flags,) = struct.unpack_from("<I", payload)
        elif tag == b"VHGT":
            offset_thu, heights, tail = _decode_vhgt(payload)
        elif tag == b"VNML":
            if len(payload) != LAND_NORMAL_BYTES:
                raise TES3ParseError(
                    f"VNML payload is {len(payload)} bytes; expected {LAND_NORMAL_BYTES}"
                )
            vertex_normals = bytes(payload)
        elif tag == b"WNAM":
            if len(payload) != LAND_WORLD_MAP_BYTES:
                raise TES3ParseError(
                    f"WNAM payload is {len(payload)} bytes; expected {LAND_WORLD_MAP_BYTES}"
                )
            world_map_data = bytes(payload)
        elif tag == b"VCLR":
            if len(payload) != LAND_COLOR_BYTES:
                raise TES3ParseError(
                    f"VCLR payload is {len(payload)} bytes; expected {LAND_COLOR_BYTES}"
                )
            vertex_colors = bytes(payload)
        elif tag == b"VTEX":
            if len(payload) != VTEX_STANDARD_SIZE:
                raise TES3ParseError(
                    f"VTEX payload is {len(payload)} bytes; expected {VTEX_STANDARD_SIZE}"
                )
            serialized_values = struct.unpack(f"<{LAND_TEXTURE_COUNT}H", payload)
            texture_indices = transpose_vtex_serialized_to_openmw(serialized_values)
    if grid is None:
        raise TES3ParseError("LAND record has no INTV grid")
    return LandRecord(
        grid=grid,
        flags=flags,
        heights_thu=heights,
        offset_thu=offset_thu,
        texture_indices=texture_indices,
        vhgt_tail=tail,
        vertex_normals=vertex_normals,
        world_map_data=world_map_data,
        vertex_colors=vertex_colors,
        record_offset=record_offset,
    )


def _decode_text(data: bytes) -> str:
    """Decode a TES3 string field without depending on the host locale."""

    return data.split(b"\0", 1)[0].decode("cp1252", errors="replace").strip()


def _printable_text(data: bytes) -> str | None:
    """Return a path-like DATA string, rejecting binary u32 DATA fields."""

    text = _decode_text(data)
    if not text or any(ord(char) < 32 for char in text):
        return None
    if not any(char.isalnum() for char in text):
        return None
    return text


def parse_ltex(
    body: bytes,
    *,
    record_index: int = 0,
    record_offset: int = 0,
) -> LandscapeTexture:
    """Parse one LTEX record.

    Modern TES3 plugins expose the landscape texture index through ``INTV``
    and the asset path through string ``DATA``.  A few old master files use a
    short display name in ``NAME`` or binary DATA; those cases retain the
    display name as the best available path and remain visible to the missing
    texture fallback/reporting layer.
    """

    record_id = ""
    explicit_index: int | None = None
    file_name: str | None = None
    for tag, payload in _subrecords(body):
        if tag == b"NAME" and not record_id:
            record_id = _decode_text(payload)
        elif tag == b"INTV":
            if len(payload) < 4:
                raise TES3ParseError("LTEX INTV must contain a u32 index")
            (explicit_index,) = struct.unpack_from("<I", payload)
        elif tag == b"DATA":
            file_name = _printable_text(payload)
    index = record_index if explicit_index is None else int(explicit_index)
    if index < 0 or index > 65535:
        raise TES3ParseError(f"LTEX index out of u16 range: {index}")
    if not record_id:
        raise TES3ParseError("LTEX record has no NAME")
    return LandscapeTexture(
        index=index,
        record_id=record_id,
        file_name=file_name or record_id,
        record_index=record_index,
        record_offset=record_offset,
    )


def _read_record(handle: BinaryIO, offset: int) -> tuple[bytes, bytes] | None:
    header = handle.read(RECORD_HEADER_SIZE)
    if not header:
        return None
    if len(header) != RECORD_HEADER_SIZE:
        raise TES3ParseError(f"truncated record header at {offset}")
    tag = header[:4]
    size = struct.unpack_from("<I", header, 4)[0]
    body = handle.read(size)
    if len(body) != size:
        raise TES3ParseError(f"record {tag!r} at {offset} is truncated")
    return tag, body


def iter_land(path: str | Path, *, max_seconds: float | None = None) -> Iterator[LandRecord]:
    """Yield LAND records from a TES3 plugin without loading the file."""

    plugin = Path(path)
    started = time.perf_counter()
    with plugin.open("rb") as handle:
        offset = 0
        while True:
            if max_seconds is not None and time.perf_counter() - started > max_seconds:
                raise TimeoutError(f"LAND scan exceeded {max_seconds:.1f}s: {plugin}")
            record = _read_record(handle, offset)
            if record is None:
                return
            tag, body = record
            if tag == b"LAND":
                yield parse_land(body, record_offset=offset + RECORD_HEADER_SIZE)
            offset += RECORD_HEADER_SIZE + len(body)


def load_land(
    path: str | Path, *, max_seconds: float | None = None
) -> dict[tuple[int, int], LandRecord]:
    """Return a deterministic grid-indexed LAND map."""

    records: dict[tuple[int, int], LandRecord] = {}
    for record in iter_land(path, max_seconds=max_seconds):
        if record.grid in records:
            raise TES3ParseError(f"duplicate LAND grid {record.grid} in {path}")
        records[record.grid] = record
    return dict(sorted(records.items()))


def height_at_game_position(
    records: Mapping[tuple[int, int], LandRecord],
    position: Sequence[float],
) -> float | None:
    """Bilinearly sample one LAND height at an absolute TES3 position.

    ``VHGT`` stores 65 by 65 vertices over an 8192-game-unit cell.  The
    footprint analysis uses this helper rather than the rendered terrain mesh,
    so its accessibility numbers come directly from the read-only source
    plugin.  The return value is in THU and may be fractional because the
    footprint corner is not generally aligned to a LAND vertex.  ``None`` is
    returned for an absent/placeholder LAND record.
    """

    if len(position) != 2:
        raise ValueError("terrain position must contain game-unit x and y")
    game_x, game_y = (float(position[0]), float(position[1]))
    if not math.isfinite(game_x) or not math.isfinite(game_y):
        raise ValueError("terrain position must be finite")
    cell_x = math.floor(game_x / CELL_SIZE_GAME_UNITS)
    cell_y = math.floor(game_y / CELL_SIZE_GAME_UNITS)
    record = records.get((cell_x, cell_y))
    if record is None or record.heights_thu is None:
        return None
    local_x = (game_x - cell_x * CELL_SIZE_GAME_UNITS) / LAND_VERTEX_SPACING_GAME_UNITS
    local_y = (game_y - cell_y * CELL_SIZE_GAME_UNITS) / LAND_VERTEX_SPACING_GAME_UNITS
    # A footprint can lie exactly on a cell edge.  The floor-based cell choice
    # keeps that sample in the following cell; clamping handles floating-point
    # roundoff at the outer 64-vertex boundary.
    local_x = min(64.0, max(0.0, local_x))
    local_y = min(64.0, max(0.0, local_y))
    x0 = min(64, math.floor(local_x))
    y0 = min(64, math.floor(local_y))
    x1 = min(64, x0 + 1)
    y1 = min(64, y0 + 1)
    fraction_x = local_x - x0
    fraction_y = local_y - y0
    lower = (
        record.height_thu(x0, y0) * (1.0 - fraction_x)
        + record.height_thu(x1, y0) * fraction_x
    )
    upper = (
        record.height_thu(x0, y1) * (1.0 - fraction_x)
        + record.height_thu(x1, y1) * fraction_x
    )
    return lower * (1.0 - fraction_y) + upper * fraction_y


def footprint_corner_heights(
    records: Mapping[tuple[int, int], LandRecord],
    minimum_xy: Sequence[float],
    maximum_xy: Sequence[float],
) -> dict[str, dict[str, float | None]]:
    """Sample the four named corners of an x/y footprint bounding box.

    The result deliberately keeps ``None`` values visible for a missing LAND
    payload; callers must not silently turn absent terrain into sea level.
    Heights are reported in THU, game units, and renderer scene units.
    """

    if len(minimum_xy) != 2 or len(maximum_xy) != 2:
        raise ValueError("footprint corners require two-value minimum and maximum")
    minimum = (float(minimum_xy[0]), float(minimum_xy[1]))
    maximum = (float(maximum_xy[0]), float(maximum_xy[1]))
    points = {
        "southwest": (minimum[0], minimum[1]),
        "southeast": (maximum[0], minimum[1]),
        "northeast": (maximum[0], maximum[1]),
        "northwest": (minimum[0], maximum[1]),
    }
    result: dict[str, dict[str, float | None]] = {}
    for name, point in points.items():
        thu = height_at_game_position(records, point)
        result[name] = {
            "position_game_units": [point[0], point[1]],
            "height_thu": thu,
            "height_game_units": thu * THU_TO_GU if thu is not None else None,
            "height_scene_units": thu * THU_TO_GU * 0.01 if thu is not None else None,
        }
    return result


def sample_height_field(
    records: Mapping[tuple[int, int], LandRecord],
    minimum_xy: Sequence[float],
    maximum_xy: Sequence[float],
    *,
    margin_game_units: float = 256.0,
    side: int = LAND_SIDE,
) -> dict[str, object]:
    """Sample a deterministic terrain grid under and around a footprint.

    The natural LAND resolution is 65 by 65 vertices, so the default uses a
    65x65 regular grid over the footprint bbox expanded by ``margin_game_units``.
    Sampling is bilinear through :func:`height_at_game_position`, allowing a
    footprint to cross cell boundaries.  Missing/placeholder LAND remains
    ``None`` rather than being silently converted to sea level.

    Values are retained in both THU and game units because the chunker uses GU
    for burial and door-step diagnostics while source validation is most useful
    in THU.  ``stats.footprint`` covers only samples whose coordinates lie
    inside the unexpanded footprint; ``stats.field`` covers the complete field.
    """

    if len(minimum_xy) != 2 or len(maximum_xy) != 2:
        raise ValueError("height field bounds require two-value minimum and maximum")
    if side < 2:
        raise ValueError("height field side must be at least two")
    margin = float(margin_game_units)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("height field margin must be finite and non-negative")
    minimum = (float(minimum_xy[0]), float(minimum_xy[1]))
    maximum = (float(maximum_xy[0]), float(maximum_xy[1]))
    if minimum[0] > maximum[0] or minimum[1] > maximum[1]:
        raise ValueError("height field minimum must not exceed maximum")
    field_minimum = (minimum[0] - margin, minimum[1] - margin)
    field_maximum = (maximum[0] + margin, maximum[1] + margin)
    spacing = (
        (field_maximum[0] - field_minimum[0]) / float(side - 1),
        (field_maximum[1] - field_minimum[1]) / float(side - 1),
    )
    values_thu: list[list[float | None]] = []
    values_gu: list[list[float | None]] = []
    all_values: list[float] = []
    footprint_values: list[float] = []
    for row in range(side):
        row_thu: list[float | None] = []
        row_gu: list[float | None] = []
        game_y = field_minimum[1] + row * spacing[1]
        for column in range(side):
            game_x = field_minimum[0] + column * spacing[0]
            value = height_at_game_position(records, (game_x, game_y))
            rounded = round(float(value), 6) if value is not None else None
            game_units = round(float(value) * THU_TO_GU, 6) if value is not None else None
            row_thu.append(rounded)
            row_gu.append(game_units)
            if value is not None:
                all_values.append(float(value))
                if minimum[0] <= game_x <= maximum[0] and minimum[1] <= game_y <= maximum[1]:
                    footprint_values.append(float(value))
        values_thu.append(row_thu)
        values_gu.append(row_gu)

    def stats(values: Sequence[float]) -> dict[str, object]:
        if not values:
            return {"valid_count": 0, "min_thu": None, "max_thu": None, "mean_thu": None,
                    "min_game_units": None, "max_game_units": None, "mean_game_units": None}
        return {
            "valid_count": len(values),
            "min_thu": round(min(values), 6),
            "max_thu": round(max(values), 6),
            "mean_thu": round(sum(values) / len(values), 6),
            "min_game_units": round(min(values) * THU_TO_GU, 6),
            "max_game_units": round(max(values) * THU_TO_GU, 6),
            "mean_game_units": round(sum(values) / len(values) * THU_TO_GU, 6),
        }

    return {
        "resolution": [side, side],
        "sample_units": "THU and game units",
        "margin_game_units": margin,
        "footprint_bbox_xy_game_units": {"min": [minimum[0], minimum[1]], "max": [maximum[0], maximum[1]]},
        "field_bbox_xy_game_units": {"min": [field_minimum[0], field_minimum[1]], "max": [field_maximum[0], field_maximum[1]]},
        "spacing_game_units": [spacing[0], spacing[1]],
        "values_thu": values_thu,
        "values_game_units": values_gu,
        "stats": {"field": stats(all_values), "footprint": stats(footprint_values)},
    }


def iter_ltex(path: str | Path, *, max_seconds: float | None = None) -> Iterator[LandscapeTexture]:
    """Yield LTEX records from a TES3 plugin in source-file order."""

    plugin = Path(path)
    started = time.perf_counter()
    record_index = 0
    with plugin.open("rb") as handle:
        offset = 0
        while True:
            if max_seconds is not None and time.perf_counter() - started > max_seconds:
                raise TimeoutError(f"LTEX scan exceeded {max_seconds:.1f}s: {plugin}")
            record = _read_record(handle, offset)
            if record is None:
                return
            tag, body = record
            if tag == b"LTEX":
                yield parse_ltex(
                    body,
                    record_index=record_index,
                    record_offset=offset + RECORD_HEADER_SIZE,
                )
                record_index += 1
            offset += RECORD_HEADER_SIZE + len(body)


def load_ltex(
    path: str | Path, *, max_seconds: float | None = None
) -> dict[int, LandscapeTexture]:
    """Load one owning plugin's LTEX table keyed by its ``INTV`` index."""

    table: dict[int, LandscapeTexture] = {}
    for texture in iter_ltex(path, max_seconds=max_seconds):
        if texture.index in table:
            raise TES3ParseError(f"duplicate LTEX index {texture.index} in {path}")
        table[texture.index] = texture
    return dict(sorted(table.items()))


def load_ltex_with_masters(
    source: str | Path,
    masters: Sequence[str | Path] = (),
    *,
    max_seconds: float | None = None,
) -> dict[int, LandscapeTexture]:
    """Overlay LTEX tables for a hypothetical visualization only.

    OpenMW does not merge LTEX indices across plugins: a LAND's nonzero VTEX
    value is converted with ``value - 1`` and looked up in the table belonging
    to the plugin that owns that LAND.  This compatibility helper is retained
    for callers that intentionally need a synthetic master overlay, but it is
    not an engine-faithful resolution path and must not be used by production
    analysis or rendering.
    """

    merged: dict[int, LandscapeTexture] = {}
    paths = [Path(item) for item in masters] + [Path(source)]
    for path in paths:
        merged.update(load_ltex(path, max_seconds=max_seconds))
    return dict(sorted(merged.items()))


def validate_land_samples(
    record: LandRecord,
    context: "WorldContext",
    offsets: Sequence[tuple[int, int]],
) -> LandValidation:
    """Compare selected LAND vertices to the read-only composite heightmap.

    The composite is a cell raster assembled from several plugins, so callers
    deliberately supply sample vertices/cells rather than treating it as a
    byte-for-byte replacement for every source LAND record.  Values are
    compared in THU and also reported in GU (one THU is eight GU).
    """

    if record.heights_thu is None:
        raise ValueError(f"LAND {record.grid} has no VHGT payload")
    if not offsets:
        raise ValueError("at least one LAND sample offset is required")
    max_delta = 0
    mismatches = 0
    for x, y in offsets:
        if not (0 <= x < LAND_SIDE and 0 <= y < LAND_SIDE):
            raise ValueError("LAND sample offsets must be in [0, 65)")
        expected = record.height_thu(x, y)
        actual = int(context.height_at((record.grid[0], record.grid[1], x, y)))
        delta = expected - actual
        max_delta = max(max_delta, abs(delta))
        if delta:
            mismatches += 1
    return LandValidation(
        grid=record.grid,
        sample_count=len(offsets),
        max_abs_delta_thu=max_delta,
        max_abs_delta_gu=max_delta * THU_TO_GU,
        mismatches=mismatches,
    )


__all__ = [
    "BASE_LAND_TEXTURE_NAME",
    "BASE_LAND_TEXTURE_PATH",
    "LAND_HEIGHT_COUNT",
    "LAND_SIDE",
    "LAND_TEXTURE_COUNT",
    "LAND_TEXTURE_SIDE",
    "LAND_NORMAL_BYTES",
    "LAND_COLOR_BYTES",
    "LAND_WORLD_MAP_BYTES",
    "CELL_SIZE_GAME_UNITS",
    "LAND_VERTEX_SPACING_GAME_UNITS",
    "LandRecord",
    "LandValidation",
    "LandscapeTexture",
    "THU_TO_GU",
    "VHGT_MIN_SIZE",
    "VHGT_STANDARD_SIZE",
    "VTEX_STANDARD_SIZE",
    "iter_ltex",
    "iter_land",
    "load_ltex",
    "load_ltex_with_masters",
    "load_land",
    "height_at_game_position",
    "footprint_corner_heights",
    "sample_height_field",
    "parse_ltex",
    "parse_land",
    "resolve_vtex_to_ltex_index",
    "transpose_vtex_openmw_to_serialized",
    "transpose_vtex_serialized_to_openmw",
    "validate_land_samples",
]
