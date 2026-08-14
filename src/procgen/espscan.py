"""Streaming TES3 ESM/ESP town-reference scanner.

The scanner intentionally stays below tes3conv's JSON layer.  It reads one
TES3 record body at a time, collects object definitions, and yields CELL
summaries with their inline FRMR references.  A complete plugin is never
loaded into memory.  The binary layout follows the validated findings:

``tag(4) + size(4) + padding(4) + flags(4) + subrecords(size)``

CELL references are inline FRMR groups and CELL DATA is 12 bytes.  The object
map is populated as records are encountered; normal TES3 files place object
definitions before CELL records.  A reference seen before its definition is
reported as unresolved rather than retaining the potentially enormous CELL
payload for a second pass.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import struct
import time
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence

from .assets import MeshClassification, classify_mesh_path


RECORD_HEADER_SIZE = 16
SUBRECORD_HEADER_SIZE = 8
CELL_DATA_SIZE = 12
FRMR_PAYLOAD_SIZE = 4

# These record types expose a MODL and are useful for static/reference
# archaeology.  NPC/CREA are deliberately excluded: their MODL is a body or
# creature mesh, not a town tileset reference.
MESH_RECORD_TYPES = frozenset({"STAT", "DOOR", "CONT", "LIGH", "ACTI", "FURN", "MISC"})
REFERENCE_ONLY_RECORD_TYPES = frozenset({"NPC_", "CREA"})
BUILDING_CATEGORIES = frozenset({"exterior", "interior", "door"})


class TES3ParseError(ValueError):
    """Raised when a TES3 record or subrecord is truncated or malformed."""


@dataclass(frozen=True)
class ObjectDefinition:
    object_id: str
    record_type: str
    model: str
    classification: MeshClassification


@dataclass(frozen=True)
class CellReference:
    object_id: str | None
    record_type: str | None
    model: str | None
    mast_index: int
    refr_index: int
    frmr_raw: int
    temporary: bool
    position: tuple[float, float, float] | None
    rotation: tuple[float, float, float] | None
    scale: float | None
    owner: str | None
    destination_position: tuple[float, float, float] | None
    destination_rotation: tuple[float, float, float] | None
    destination_cell: str | None
    has_dodt: bool
    kit: str
    category: str
    building: bool
    unresolved: bool

    @property
    def door_to_interior(self) -> bool:
        """Whether this reference has the exterior-door link grammar.

        TES3 does not put a separate "exterior door" flag on a reference.  The
        anatomy report established the useful streaming predicate: a DOOR ref
        with a DODT payload and a non-empty DNAM cell name leads to an interior.
        Keeping the predicate here prevents callers from mistaking a door mesh
        or an empty return-door DNAM for a dwelling link.
        """

        return (
            self.record_type == "DOOR"
            and self.has_dodt
            and bool(self.destination_cell)
        )


@dataclass(frozen=True)
class CellSummary:
    name: str | None
    is_interior: bool
    grid: tuple[int, int] | None
    region: str | None
    flags: int
    references: tuple[CellReference, ...]
    offset: int = 0

    @property
    def ref_count(self) -> int:
        return len(self.references)


@dataclass
class ScanResult:
    path: str
    size_bytes: int
    elapsed_seconds: float
    sha256: str
    record_counts: Counter[str] = field(default_factory=Counter)
    record_bytes: Counter[str] = field(default_factory=Counter)
    stat_models: dict[str, str] = field(default_factory=dict)
    object_models: dict[str, ObjectDefinition] = field(default_factory=dict)
    object_types: dict[str, str] = field(default_factory=dict)
    cells: list[CellSummary] = field(default_factory=list)
    pathgrid_names: set[str] = field(default_factory=set)
    pathgrid_record_count: int = 0
    exterior_cells: int = 0
    interior_cells: int = 0
    reference_count: int = 0
    mesh_reference_count: int = 0
    resolved_mesh_reference_count: int = 0
    unresolved_reference_count: int = 0
    unresolved_object_ids: Counter[str] = field(default_factory=Counter)
    malformed_records: int = 0

    @property
    def cell_count(self) -> int:
        return self.exterior_cells + self.interior_cells

    @property
    def ref_count(self) -> int:
        return self.reference_count

    @property
    def static_count(self) -> int:
        return self.record_counts.get("STAT", 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "sha256": self.sha256,
            "record_counts": dict(sorted(self.record_counts.items())),
            "record_bytes": dict(sorted(self.record_bytes.items())),
            "stat_model_count": len(self.stat_models),
            "object_model_count": len(self.object_models),
            "object_type_count": len(self.object_types),
            "cell_count": self.cell_count,
            "exterior_cells": self.exterior_cells,
            "interior_cells": self.interior_cells,
            "reference_count": self.reference_count,
            "mesh_reference_count": self.mesh_reference_count,
            "resolved_mesh_reference_count": self.resolved_mesh_reference_count,
            "unresolved_reference_count": self.unresolved_reference_count,
            "unresolved_object_ids": dict(self.unresolved_object_ids.most_common(20)),
            "pathgrid_record_count": self.pathgrid_record_count,
            "pathgrid_cell_name_count": len(self.pathgrid_names),
            "malformed_records": self.malformed_records,
        }


def _decode_text(data: bytes) -> str:
    """Decode TES3's null-terminated byte strings without locale dependence."""

    return data.split(b"\0", 1)[0].decode("cp1252", errors="replace")


def _subrecords(body: bytes, *, base_offset: int = 0) -> Iterator[tuple[bytes, bytes, int]]:
    """Yield ``(tag, payload, absolute_payload_offset)`` from one record body."""

    pos = 0
    limit = len(body)
    while pos + SUBRECORD_HEADER_SIZE <= limit:
        tag = body[pos : pos + 4]
        (size,) = struct.unpack_from("<I", body, pos + 4)
        payload_start = pos + SUBRECORD_HEADER_SIZE
        payload_end = payload_start + size
        if payload_end > limit:
            raise TES3ParseError(
                f"subrecord {tag!r} at {base_offset + pos} overruns record body"
            )
        yield tag, body[payload_start:payload_end], base_offset + payload_start
        pos = payload_end
    if pos != limit:
        raise TES3ParseError(f"{limit - pos} trailing bytes after subrecords at {base_offset + pos}")


def _parse_object(body: bytes, record_type: str, source_kit: str) -> ObjectDefinition | None:
    object_id: str | None = None
    model = ""
    for tag, payload, _offset in _subrecords(body):
        if tag == b"NAME" and object_id is None:
            object_id = _decode_text(payload)
        elif tag == b"MODL":
            model = _decode_text(payload)
    if not object_id:
        return None
    classification = classify_mesh_path(model, source_kit=source_kit, record_type=record_type)
    return ObjectDefinition(object_id, record_type, model, classification)


def _parse_record_object_id(body: bytes) -> str | None:
    """Read only NAME from a non-mesh definition (NPC_/CREA).

    NPC records are deliberately not parsed deeply.  Their NAME is sufficient
    to resolve cheap reference counts while retaining the one-record-at-a-time
    parser architecture.
    """

    for tag, payload, _offset in _subrecords(body):
        if tag == b"NAME":
            value = _decode_text(payload)
            return value or None
    return None


def _parse_pathgrid_name(body: bytes) -> str | None:
    """Read the cell NAME from a PGRD record without parsing grid points."""

    return _parse_record_object_id(body)


def _parse_ref(
    body: bytes,
    frmr_payload_start: int,
    *,
    temporary: bool,
    object_models: Mapping[str, ObjectDefinition],
    object_types: Mapping[str, str],
    source_kit: str,
) -> tuple[CellReference, int]:
    if frmr_payload_start + FRMR_PAYLOAD_SIZE > len(body):
        raise TES3ParseError("FRMR has no 4-byte reference number")
    (raw,) = struct.unpack_from("<I", body, frmr_payload_start)
    mast_index = (raw >> 24) & 0xFF
    refr_index = raw & 0xFFFFFF
    object_id: str | None = None
    position: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None
    scale: float | None = None
    owner: str | None = None
    destination_position: tuple[float, float, float] | None = None
    destination_rotation: tuple[float, float, float] | None = None
    destination_cell: str | None = None
    has_dodt = False
    pos = frmr_payload_start + FRMR_PAYLOAD_SIZE
    limit = len(body)
    while pos + SUBRECORD_HEADER_SIZE <= limit:
        tag = body[pos : pos + 4]
        (size,) = struct.unpack_from("<I", body, pos + 4)
        payload_start = pos + SUBRECORD_HEADER_SIZE
        payload_end = payload_start + size
        if payload_end > limit:
            raise TES3ParseError(f"reference subrecord {tag!r} overruns CELL")
        payload = body[payload_start:payload_end]
        if tag == b"FRMR":
            # The next reference begins here.  Do not consume its header.
            break
        if tag == b"NAME" and object_id is None:
            object_id = _decode_text(payload)
        elif tag == b"ANAM":
            owner = _decode_text(payload) or None
        elif tag == b"DODT":
            has_dodt = size >= 24
            if has_dodt:
                values = struct.unpack_from("<6f", payload)
                destination_position = tuple(values[:3])
                destination_rotation = tuple(values[3:])
        elif tag == b"DNAM":
            destination_cell = _decode_text(payload) or None
        elif tag == b"XSCL" and size >= 4:
            (scale,) = struct.unpack_from("<f", payload)
        elif tag == b"DATA":
            if size >= 24:
                values = struct.unpack_from("<6f", payload)
                position = tuple(values[:3])
                rotation = tuple(values[3:])
            # DATA is the required transform payload, but XSCL may occur
            # after it in real plugins.  Continue until the next FRMR so the
            # complete reference transform is retained.
            pos = payload_end
            continue
        elif tag == b"DELE":
            pos = payload_end
            break
        pos = payload_end

    definition = object_models.get((object_id or "").casefold())
    if definition is None:
        model = None
        record_type = object_types.get((object_id or "").casefold())
        classification = classify_mesh_path(
            "", source_kit=source_kit, record_type=None
        )
        unresolved = bool(object_id)
    else:
        model = definition.model or None
        record_type = definition.record_type
        # A dependency ESM can store a legacy ``x\\``/``f\\`` path without a
        # province prefix.  Attribute that path to the referencing town file,
        # while explicit ``tr\\``/``sky\\``/``pc\\`` namespaces remain stable.
        classification = classify_mesh_path(
            definition.model,
            source_kit=source_kit,
            record_type=definition.record_type,
        )
        unresolved = not bool(model)
    return (
        CellReference(
            object_id=object_id,
            record_type=record_type,
            model=model,
            mast_index=mast_index,
            refr_index=refr_index,
            frmr_raw=raw,
            temporary=temporary,
            position=position,
            rotation=rotation,
            scale=scale,
            owner=owner,
            destination_position=destination_position,
            destination_rotation=destination_rotation,
            destination_cell=destination_cell,
            has_dodt=has_dodt,
            kit=classification.kit if model else "unknown",
            category=classification.category if model else "unknown",
            building=bool(model and classification.category in BUILDING_CATEGORIES),
            unresolved=unresolved,
        ),
        pos,
    )


def parse_cell(
    body: bytes,
    *,
    object_models: Mapping[str, ObjectDefinition],
    object_types: Mapping[str, str] | None = None,
    source_kit: str = "vanilla",
    offset: int = 0,
) -> CellSummary:
    """Parse one CELL body, including all inline FRMR groups."""

    name: str | None = None
    region: str | None = None
    grid: tuple[int, int] | None = None
    flags = 0
    is_interior = False
    refs: list[CellReference] = []
    temp_remaining = 0
    pos = 0
    limit = len(body)
    while pos + SUBRECORD_HEADER_SIZE <= limit:
        tag = body[pos : pos + 4]
        (size,) = struct.unpack_from("<I", body, pos + 4)
        payload_start = pos + SUBRECORD_HEADER_SIZE
        payload_end = payload_start + size
        if payload_end > limit:
            raise TES3ParseError(f"CELL subrecord {tag!r} overruns record body")
        payload = body[payload_start:payload_end]
        if tag == b"FRMR":
            ref, next_pos = _parse_ref(
                body,
                payload_start,
                temporary=temp_remaining > 0,
                object_models=object_models,
                object_types=object_types or {},
                source_kit=source_kit,
            )
            refs.append(ref)
            if temp_remaining:
                temp_remaining -= 1
            pos = next_pos
            continue
        if tag == b"NAME" and name is None:
            name = _decode_text(payload)
        elif tag == b"DATA" and size >= 4:
            (flags,) = struct.unpack_from("<I", payload)
            is_interior = bool(flags & 0x1)
            if size >= CELL_DATA_SIZE:
                parsed_grid = tuple(struct.unpack_from("<ii", payload, 4))
                # Interior DATA carries two junk/placeholder integers.  Keep
                # the identity in NAME + interior flag and expose grid only
                # for exterior cells where it is meaningful.
                grid = None if is_interior else parsed_grid
        elif tag == b"RGNN":
            region = _decode_text(payload) or None
        elif tag == b"NAM0" and size >= 4:
            (temp_remaining,) = struct.unpack_from("<i", payload)
            temp_remaining = max(0, temp_remaining)
        pos = payload_end
    if pos != limit:
        raise TES3ParseError(f"CELL has trailing bytes at {offset + pos}")
    return CellSummary(
        name=name,
        is_interior=is_interior,
        grid=grid,
        region=region,
        flags=flags,
        references=tuple(refs),
        offset=offset,
    )


def _read_record(handle: BinaryIO, offset: int) -> tuple[bytes, int, bytes] | None:
    header = handle.read(RECORD_HEADER_SIZE)
    if not header:
        return None
    if len(header) != RECORD_HEADER_SIZE:
        raise TES3ParseError(f"truncated record header at {offset}")
    tag = header[:4]
    (size,) = struct.unpack_from("<I", header, 4)
    body = handle.read(size)
    if len(body) != size:
        raise TES3ParseError(f"record {tag!r} at {offset} is truncated")
    (flags,) = struct.unpack_from("<I", header, 12)
    return tag, flags, body


def _scan(
    path: Path,
    *,
    source_kit: str,
    on_cell: Callable[[CellSummary], None] | None = None,
    collect_cells: bool,
    max_seconds: float | None = None,
    initial_object_models: Mapping[str, ObjectDefinition] | None = None,
    initial_object_types: Mapping[str, str] | None = None,
) -> ScanResult:
    started = time.perf_counter()
    digest = hashlib.sha256()
    result = ScanResult(path=str(path), size_bytes=path.stat().st_size, elapsed_seconds=0.0, sha256="")
    if initial_object_models:
        result.object_models.update(initial_object_models)
        for definition in initial_object_models.values():
            if definition.record_type == "STAT":
                result.stat_models.setdefault(definition.object_id, definition.model)
    if initial_object_types:
        result.object_types.update(initial_object_types)
    with path.open("rb") as handle:
        offset = 0
        while True:
            if max_seconds is not None and time.perf_counter() - started > max_seconds:
                raise TimeoutError(f"TES3 scan exceeded {max_seconds:.1f}s: {path}")
            header = handle.read(RECORD_HEADER_SIZE)
            if not header:
                break
            if len(header) != RECORD_HEADER_SIZE:
                raise TES3ParseError(f"truncated record header at {offset}")
            digest.update(header)
            tag = header[:4]
            (size,) = struct.unpack_from("<I", header, 4)
            body = handle.read(size)
            if len(body) != size:
                raise TES3ParseError(f"record {tag!r} at {offset} is truncated")
            digest.update(body)
            tag_text = tag.decode("latin1")
            result.record_counts[tag_text] += 1
            result.record_bytes[tag_text] += RECORD_HEADER_SIZE + size
            if tag_text in MESH_RECORD_TYPES:
                definition = _parse_object(body, tag_text, source_kit)
                if definition is not None:
                    result.object_models[definition.object_id.casefold()] = definition
                    if tag_text == "STAT":
                        result.stat_models[definition.object_id] = definition.model
            elif tag_text in REFERENCE_ONLY_RECORD_TYPES:
                object_id = _parse_record_object_id(body)
                if object_id:
                    result.object_types[object_id.casefold()] = tag_text
            elif tag == b"PGRD":
                result.pathgrid_record_count += 1
                pathgrid_name = _parse_pathgrid_name(body)
                if pathgrid_name:
                    result.pathgrid_names.add(pathgrid_name.casefold().strip())
            if tag == b"CELL":
                try:
                    cell = parse_cell(
                        body,
                        object_models=result.object_models,
                        object_types=result.object_types,
                        source_kit=source_kit,
                        offset=offset + RECORD_HEADER_SIZE,
                    )
                except TES3ParseError:
                    result.malformed_records += 1
                    raise
                if cell.is_interior:
                    result.interior_cells += 1
                else:
                    result.exterior_cells += 1
                result.reference_count += cell.ref_count
                result.mesh_reference_count += sum(1 for ref in cell.references if ref.model)
                result.resolved_mesh_reference_count += sum(
                    1 for ref in cell.references if ref.model and not ref.unresolved
                )
                for ref in cell.references:
                    if ref.unresolved:
                        result.unresolved_reference_count += 1
                        if ref.object_id:
                            result.unresolved_object_ids[ref.object_id] += 1
                if collect_cells:
                    result.cells.append(cell)
                if on_cell is not None:
                    on_cell(cell)
            offset += RECORD_HEADER_SIZE + size
    result.elapsed_seconds = time.perf_counter() - started
    result.sha256 = digest.hexdigest()
    return result


def scan_file(
    path: str | Path,
    *,
    source_kit: str = "vanilla",
    on_cell: Callable[[CellSummary], None] | None = None,
    collect_cells: bool = False,
    max_seconds: float | None = None,
    initial_object_models: Mapping[str, ObjectDefinition] | None = None,
    initial_object_types: Mapping[str, str] | None = None,
) -> ScanResult:
    """Scan a plugin once, invoking ``on_cell`` as CELLs are parsed.

    ``collect_cells=False`` is the production mode: the caller receives each
    cell through the callback and only compact counters are retained.
    ``collect_cells=True`` is convenient for small fixtures and interactive
    inspection, not for the large ESMs.
    """

    return _scan(
        Path(path),
        source_kit=source_kit,
        on_cell=on_cell,
        collect_cells=collect_cells,
        max_seconds=max_seconds,
        initial_object_models=initial_object_models,
        initial_object_types=initial_object_types,
    )


def iter_cells(
    path: str | Path,
    *,
    source_kit: str = "vanilla",
    max_seconds: float | None = None,
) -> Iterator[CellSummary]:
    """Yield CELL summaries without retaining a whole plugin in memory."""

    plugin = Path(path)
    started = time.perf_counter()
    object_models: dict[str, ObjectDefinition] = {}
    object_types: dict[str, str] = {}
    with plugin.open("rb") as handle:
        offset = 0
        while True:
            if max_seconds is not None and time.perf_counter() - started > max_seconds:
                raise TimeoutError(f"TES3 scan exceeded {max_seconds:.1f}s: {plugin}")
            record = _read_record(handle, offset)
            if record is None:
                return
            tag, _flags, body = record
            tag_text = tag.decode("latin1")
            if tag_text in MESH_RECORD_TYPES:
                definition = _parse_object(body, tag_text, source_kit)
                if definition is not None:
                    object_models[definition.object_id.casefold()] = definition
            elif tag_text in REFERENCE_ONLY_RECORD_TYPES:
                object_id = _parse_record_object_id(body)
                if object_id:
                    object_types[object_id.casefold()] = tag_text
            if tag == b"CELL":
                yield parse_cell(
                    body,
                    object_models=object_models,
                    object_types=object_types,
                    source_kit=source_kit,
                    offset=offset + RECORD_HEADER_SIZE,
                )
            offset += RECORD_HEADER_SIZE + len(body)


__all__ = [
    "BUILDING_CATEGORIES",
    "CELL_DATA_SIZE",
    "CellReference",
    "CellSummary",
    "MESH_RECORD_TYPES",
    "REFERENCE_ONLY_RECORD_TYPES",
    "ObjectDefinition",
    "ScanResult",
    "TES3ParseError",
    "iter_cells",
    "parse_cell",
    "scan_file",
]
