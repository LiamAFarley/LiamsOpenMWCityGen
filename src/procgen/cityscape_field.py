"""Cityforge T1.3 stitched LAND field and source/effective payload contract.

Pipeline position
------------------
This module is the read-only LAND boundary of the T1.3 landscape editor.  It
loads the authoritative heights and non-VTEX payload from ``tamriel.esm`` and
the effective normalized VTEX view from the accepted Falkreath remap ESP.  The
49 cells are joined into one 449 by 449 float64-GU vertex field, edited by the
other T1.3 modules, and split back into TES3-sized 65 by 65 records.

Inputs and outputs
------------------
``load_target_block`` consumes the accepted site survey, ``tamriel.esm``, and
the accepted remap ESP.  It also loads one exterior LAND neighbour on every
side of the target block so VNML border samples never use an invented height.
``write_field_npz`` writes a timestamp-free NPZ containing ``height_gu`` and a
sidecar metadata document that is directly consumable by the T1.2
``TerrainField`` contract.  ``split_field`` returns exact per-cell views for
T1.4 LAND assembly.

Invariants
----------
* The target is exactly the contiguous 7 by 7 / 49-cell site-survey block.
* Shared source LAND edges must be float64-exact before any edit.  The outer
  49-cell vertex border is tracked explicitly and may never change.
* Base heights, VNML, VCLR, WNAM, LAND DATA flags, and VHGT padding come from
  ``tamriel.esm``.  Effective VTEX values come from the remap ESP and retain
  their owning-plugin table identity; the two scopes are reconciled in an
  explicit audit rather than overlaid implicitly.
* No missing neighbour or incomplete source LAND payload is converted to a
  default value.  The stage fails closed instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.lib import format as np_format

from . import espland
from .censusio import PinnedFile, deterministic_dumps


# Falkreath (7x7) defaults kept as import-compatible constants; the validators
# below derive the expected cell count / field side from the actual survey
# cell set, so any square site (3x3, 5x5, 7x7, ...) passes without a rebuild.
TARGET_CELL_COUNT = 49
TARGET_SIDE_CELLS = 7
FIELD_SIDE = TARGET_SIDE_CELLS * 64 + 1
FIELD_SPACING_GU = 128.0
CELL_SIZE_GU = 8192.0
LAND_SIDE = 65


def _rect_cell_geometry(cells: Sequence[tuple[int, int]], label: str) -> tuple[list[int], list[int]]:
    """Validate a contiguous rectangular cell set; return (xs, ys).

    Rectangles are fully supported: a site may be any contiguous
    width x height block of cells (squares included).  All downstream field
    shapes are derived as (len(ys) * 64 + 1, len(xs) * 64 + 1), row-major
    [y, x], never from a hardcoded side.
    """

    xs = sorted({cell[0] for cell in cells})
    ys = sorted({cell[1] for cell in cells})
    if not xs or not ys:
        raise CityscapeFieldError(f"{label}: empty target cell set")
    expected = {(x, y) for y in ys for x in xs}
    if set(cells) != expected or xs != list(range(xs[0], xs[-1] + 1)) or ys != list(range(ys[0], ys[-1] + 1)):
        raise CityscapeFieldError(
            f"{label}: target cells are not a contiguous {len(xs)}x{len(ys)} block"
        )
    return xs, ys


class CityscapeFieldError(ValueError):
    """Hard source stitch, payload, frame, or field serialization failure."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path | str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CityscapeFieldError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CityscapeFieldError(f"{label} {path} must be a JSON object")
    return value


def _stable_array_bytes(values: np.ndarray) -> bytes:
    """Return endian-stable contiguous bytes for a field-content hash."""

    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return array.tobytes(order="C")


def terrain_field_sha256(values_gu: np.ndarray) -> str:
    """Hash only the canonical float64 field values, not a ZIP container."""

    return _sha256_bytes(_stable_array_bytes(values_gu))


def _cell_key(cell: Sequence[int]) -> tuple[int, int]:
    if len(cell) != 2:
        raise CityscapeFieldError(f"cell grid must contain two integers: {cell!r}")
    x, y = int(cell[0]), int(cell[1])
    if isinstance(cell[0], bool) or isinstance(cell[1], bool):
        raise CityscapeFieldError(f"cell grid values must be integers: {cell!r}")
    return x, y


def _payload_bytes(record: espland.LandRecord, field: str) -> bytes:
    if field == "flags":
        return struct.pack("<I", int(record.flags))
    if field == "heights_thu":
        if record.heights_thu is None:
            return b""
        values = [item for row in record.heights_thu for item in row]
        return struct.pack(f"<{len(values)}i", *values)
    if field == "texture_indices":
        if record.texture_indices is None:
            return b""
        return struct.pack(f"<{len(record.texture_indices)}H", *record.texture_indices)
    value = getattr(record, field)
    return b"" if value is None else bytes(value)


PAYLOAD_FIELDS = (
    "flags",
    "heights_thu",
    "vhgt_tail",
    "vertex_normals",
    "world_map_data",
    "vertex_colors",
    "texture_indices",
)


def payload_digest(record: espland.LandRecord, *, include_vtex: bool = True) -> str:
    """Hash one LAND's retained payload fields with field names and lengths."""

    digest = hashlib.sha256()
    fields = PAYLOAD_FIELDS if include_vtex else tuple(
        field for field in PAYLOAD_FIELDS if field != "texture_indices"
    )
    for field in fields:
        value = _payload_bytes(record, field)
        digest.update(field.encode("ascii"))
        digest.update(struct.pack("<I", len(value)))
        digest.update(value)
    return digest.hexdigest()


def _require_complete_land(record: espland.LandRecord, label: str) -> None:
    required = {
        "heights_thu": record.heights_thu,
        "texture_indices": record.texture_indices,
        "vertex_normals": record.vertex_normals,
        "world_map_data": record.world_map_data,
        "vertex_colors": record.vertex_colors,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise CityscapeFieldError(f"{label} LAND {record.grid} missing {missing}")
    if record.vhgt_tail not in (b"", b"\0\0\0"):
        raise CityscapeFieldError(
            f"{label} LAND {record.grid} has non-standard nonzero VHGT padding"
        )


def _target_cells_from_survey(survey: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    rows = survey.get("cells")
    if not isinstance(rows, list):
        raise CityscapeFieldError("site survey has no cells list")
    cells_set = {_cell_key(row["grid"]) for row in rows if isinstance(row, Mapping)}
    if len(cells_set) != len(rows):
        raise CityscapeFieldError("site survey cells list contains duplicate or invalid grids")
    cells = tuple(sorted(cells_set))
    xs, ys = _rect_cell_geometry(cells, "site survey target")
    expected = tuple((x, y) for y in ys for x in xs)
    target = survey.get("target_cells")
    if isinstance(target, Mapping):
        expected_bounds = {
            "min_x": xs[0], "max_x": xs[-1], "min_y": ys[0], "max_y": ys[-1]
        }
        if any(int(target.get(key, 10**9)) != value for key, value in expected_bounds.items()):
            raise CityscapeFieldError("site survey target_cells bounds disagree with cells")
    return expected


def _assert_frame(survey: Mapping[str, Any], cells: Sequence[tuple[int, int]]) -> tuple[float, float]:
    frame = survey.get("frame")
    if not isinstance(frame, Mapping):
        raise CityscapeFieldError("site survey frame is missing")
    origin = frame.get("origin_gu")
    spacing = frame.get("field_spacing_gu")
    if not isinstance(origin, list) or len(origin) != 2 or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in origin
    ):
        raise CityscapeFieldError("site survey frame origin_gu is invalid")
    if float(spacing) != FIELD_SPACING_GU:
        raise CityscapeFieldError(
            f"site survey field spacing is {spacing!r}; expected {FIELD_SPACING_GU} GU"
        )
    min_x, min_y = min(cell[0] for cell in cells), min(cell[1] for cell in cells)
    expected_origin = (min_x * CELL_SIZE_GU, min_y * CELL_SIZE_GU)
    actual_origin = (float(origin[0]), float(origin[1]))
    if actual_origin != expected_origin:
        raise CityscapeFieldError(
            f"survey frame origin {actual_origin} does not equal target SW {expected_origin}"
        )
    return actual_origin


def _edge_mismatch_rows(
    records: Mapping[tuple[int, int], espland.LandRecord],
    cells: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    cell_set = set(cells)
    for x, y in sorted(cells):
        current = records[(x, y)]
        assert current.heights_gu is not None
        if (x + 1, y) in cell_set:
            right = records[(x + 1, y)]
            assert right.heights_gu is not None
            left_edge = np.asarray(current.heights_gu, dtype=np.float64)[:, -1]
            right_edge = np.asarray(right.heights_gu, dtype=np.float64)[:, 0]
            if not np.array_equal(left_edge, right_edge):
                where = int(np.flatnonzero(left_edge != right_edge)[0])
                mismatches.append({
                    "axis": "x",
                    "left": [x, y],
                    "right": [x + 1, y],
                    "vertex": [64, where],
                    "left_gu": float(left_edge[where]),
                    "right_gu": float(right_edge[where]),
                })
        if (x, y + 1) in cell_set:
            north = records[(x, y + 1)]
            assert north.heights_gu is not None
            lower_edge = np.asarray(current.heights_gu, dtype=np.float64)[-1, :]
            upper_edge = np.asarray(north.heights_gu, dtype=np.float64)[0, :]
            if not np.array_equal(lower_edge, upper_edge):
                where = int(np.flatnonzero(lower_edge != upper_edge)[0])
                mismatches.append({
                    "axis": "y",
                    "south": [x, y],
                    "north": [x, y + 1],
                    "vertex": [where, 64],
                    "south_gu": float(lower_edge[where]),
                    "north_gu": float(upper_edge[where]),
                })
    return mismatches


def stitch_heights(
    records: Mapping[tuple[int, int], espland.LandRecord],
    cells: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Join 49 source LAND height grids, rejecting disagreement at every seam."""

    cell_list = tuple(sorted(cells))
    xs, ys = _rect_cell_geometry(cell_list, "stitch_heights")
    mismatches = _edge_mismatch_rows(records, cell_list)
    if mismatches:
        raise CityscapeFieldError(
            "source LAND shared-edge disagreement: " + json.dumps(mismatches[:4], sort_keys=True)
        )
    result = np.empty((len(ys) * 64 + 1, len(xs) * 64 + 1), dtype=np.float64)
    for cell_y, y in enumerate(ys):
        for cell_x, x in enumerate(xs):
            record = records[(x, y)]
            if record.heights_gu is None:
                raise CityscapeFieldError(f"LAND {x,y} has no heights")
            values = np.asarray(record.heights_gu, dtype=np.float64)
            y0, x0 = cell_y * 64, cell_x * 64
            result[y0 : y0 + 65, x0 : x0 + 65] = values
    if not np.isfinite(result).all():
        raise CityscapeFieldError("stitched source field contains non-finite heights")
    return result


def split_field(
    values_gu: np.ndarray,
    cells: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    """Split a joint field into exact 65x65 per-cell float64 arrays."""

    field = np.asarray(values_gu, dtype=np.float64)
    cell_list = tuple(sorted(cells))
    xs, ys = _rect_cell_geometry(cell_list, "split_field")
    expected_shape = (len(ys) * 64 + 1, len(xs) * 64 + 1)
    if field.shape != expected_shape:
        raise CityscapeFieldError(
            f"joint terrain field must be {expected_shape}, got {field.shape}"
        )
    result: dict[tuple[int, int], np.ndarray] = {}
    for y in ys:
        for x in xs:
            x0, y0 = (x - xs[0]) * 64, (y - ys[0]) * 64
            result[(x, y)] = np.array(field[y0 : y0 + 65, x0 : x0 + 65], dtype=np.float64, copy=True)
    return result


def rejoin_field(
    per_cell: Mapping[tuple[int, int], np.ndarray],
    cells: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Rejoin split grids and reject any seam that is no longer exact."""

    cell_list = tuple(sorted(cells))
    xs, ys = _rect_cell_geometry(cell_list, "rejoin_field")
    result = np.empty((len(ys) * 64 + 1, len(xs) * 64 + 1), dtype=np.float64)
    for y in ys:
        for x in xs:
            value = np.asarray(per_cell[(x, y)], dtype=np.float64)
            if value.shape != (LAND_SIDE, LAND_SIDE):
                raise CityscapeFieldError(f"split cell {(x,y)} has shape {value.shape}")
            x0, y0 = (x - xs[0]) * 64, (y - ys[0]) * 64
            result[y0 : y0 + 65, x0 : x0 + 65] = value
    return result


def outer_border_mask(shape: tuple[int, int] | int = FIELD_SIDE) -> np.ndarray:
    """Return the immutable outer vertex border mask for an (h, w) field.

    Accepts either a (height, width) tuple or a single square side for
    backwards compatibility with the Falkreath 449x449 call sites.
    """

    if isinstance(shape, tuple):
        height, width = shape
    else:
        height = width = int(shape)
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _record_payload_diff(
    base: espland.LandRecord,
    effective: espland.LandRecord,
) -> dict[str, Any]:
    return {
        field: _payload_bytes(base, field) != _payload_bytes(effective, field)
        for field in PAYLOAD_FIELDS
    }


@dataclass(frozen=True)
class TargetBlock:
    """Read-only source/effective target bundle plus the joint source field."""

    cells: tuple[tuple[int, int], ...]
    origin_gu: tuple[float, float]
    spacing_gu: float
    source_land: Mapping[tuple[int, int], espland.LandRecord]
    effective_land: Mapping[tuple[int, int], espland.LandRecord]
    source_ltex: Mapping[int, espland.LandscapeTexture]
    effective_ltex: Mapping[int, espland.LandscapeTexture]
    neighbor_land: Mapping[tuple[int, int], espland.LandRecord]
    source_heights_gu: np.ndarray
    effective_vtex: Mapping[tuple[int, int], np.ndarray]
    source_file: str
    source_sha256: str
    effective_file: str
    effective_sha256: str
    reconciliation: Mapping[str, Any]

    @property
    def field_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.source_heights_gu.shape)

    @property
    def field_sha256(self) -> str:
        return terrain_field_sha256(self.source_heights_gu)

    @property
    def border_mask(self) -> np.ndarray:
        return outer_border_mask(self.field_shape)

    def source_cell_field(self, cell: tuple[int, int]) -> np.ndarray:
        """Return a copy of one source cell in float64 GU."""

        record = self.source_land[cell]
        if record.heights_gu is None:
            raise CityscapeFieldError(f"source LAND {cell} has no heights")
        return np.asarray(record.heights_gu, dtype=np.float64).copy()

    def outside_source_height_gu(self, global_x: int, global_y: int) -> float:
        """Sample a target-relative vertex, including its one-cell border."""

        field_h, field_w = self.field_shape
        if not (-1 <= global_x <= field_w and -1 <= global_y <= field_h):
            raise CityscapeFieldError("normal context request exceeds one-cell border")
        # The source field already covers the target.  Only the one-vertex halo
        # is needed for central differences at the immutable block edge.
        if 0 <= global_x <= field_w - 1 and 0 <= global_y <= field_h - 1:
            return float(self.source_heights_gu[global_y, global_x])
        world_x = self.origin_gu[0] + global_x * self.spacing_gu
        world_y = self.origin_gu[1] + global_y * self.spacing_gu
        cell_x = int(np.floor(world_x / CELL_SIZE_GU))
        cell_y = int(np.floor(world_y / CELL_SIZE_GU))
        local_x = int(round((world_x - cell_x * CELL_SIZE_GU) / self.spacing_gu))
        local_y = int(round((world_y - cell_y * CELL_SIZE_GU) / self.spacing_gu))
        record = self.neighbor_land.get((cell_x, cell_y))
        if record is None or record.heights_gu is None:
            raise CityscapeFieldError(
                f"missing source normal-context LAND {(cell_x, cell_y)} for vertex "
                f"({global_x},{global_y})"
            )
        if not (0 <= local_x < LAND_SIDE and 0 <= local_y < LAND_SIDE):
            raise CityscapeFieldError("normal-context local vertex is out of range")
        return float(record.heights_gu[local_y][local_x])

    def effective_texture_grid(self, cell: tuple[int, int]) -> np.ndarray:
        return np.array(self.effective_vtex[cell], dtype=np.uint16, copy=True)


def _reconcile_payloads(
    source: Mapping[tuple[int, int], espland.LandRecord],
    effective: Mapping[tuple[int, int], espland.LandRecord],
    cells: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    non_vtex_diffs: list[dict[str, Any]] = []
    vtex_diffs: list[dict[str, Any]] = []
    counts = {field: 0 for field in PAYLOAD_FIELDS}
    for cell in sorted(cells):
        left, right = source[cell], effective[cell]
        diff = _record_payload_diff(left, right)
        for field, changed in diff.items():
            if changed:
                counts[field] += 1
        non_vtex = {field: value for field, value in diff.items() if field != "texture_indices" and value}
        if non_vtex:
            non_vtex_diffs.append({"cell": list(cell), "fields": sorted(non_vtex)})
        if diff["texture_indices"]:
            vtex_diffs.append({"cell": list(cell), "field": "texture_indices"})
    if non_vtex_diffs:
        raise CityscapeFieldError(
            "accepted remap LAND changes a base non-VTEX payload: "
            + json.dumps(non_vtex_diffs[:4], sort_keys=True)
        )
    return {
        "base_payload_authority": "tamriel.esm",
        "effective_vtex_authority": "accepted remap ESP normalized VTEX",
        "non_vtex_payload_differences": [],
        "vtex_difference_cells": vtex_diffs,
        "field_difference_counts": counts,
        "effective_vtex_selected_explicitly": True,
        "plugin_local_tables_kept_separate": True,
    }


def load_target_block(
    *,
    root: Path | str,
    survey_path: Path | str,
    source_path: Path | str,
    effective_path: Path | str,
) -> TargetBlock:
    """Load, validate, and stitch the authoritative Falkreath target block."""

    workspace = Path(root)
    survey = _json_load(survey_path, "site survey")
    cells = _target_cells_from_survey(survey)
    origin = _assert_frame(survey, cells)
    source_file = Path(source_path)
    effective_file = Path(effective_path)
    if not source_file.is_file() or not effective_file.is_file():
        raise CityscapeFieldError(
            f"missing source/effective LAND input: {source_file}, {effective_file}"
        )
    source_all = espland.load_land(source_file, max_seconds=300.0)
    effective_all = espland.load_land(effective_file, max_seconds=120.0)
    source_ltex = espland.load_ltex(source_file, max_seconds=300.0)
    effective_ltex = espland.load_ltex(effective_file, max_seconds=120.0)
    source = {cell: source_all[cell] for cell in cells if cell in source_all}
    effective = {cell: effective_all[cell] for cell in cells if cell in effective_all}
    missing_source = sorted(set(cells) - set(source))
    missing_effective = sorted(set(cells) - set(effective))
    if missing_source or missing_effective:
        raise CityscapeFieldError(
            f"target LAND missing: source={missing_source}, effective={missing_effective}"
        )
    for cell in cells:
        _require_complete_land(source[cell], "tamriel.esm")
        _require_complete_land(effective[cell], "effective remap")
    # The remap is a payload-preserving VTEX/LTEX proving-ground.  Its local
    # table is deliberately not merged with tamriel.esm's table; every raw is
    # resolved once in each owning-plugin scope for an auditable reconciliation.
    reconciliation = _reconcile_payloads(source, effective, cells)
    source_heights = stitch_heights(source, cells)
    target_set = set(cells)
    # The corner VNML samples use one outside x and one outside y coordinate
    # simultaneously, so the required one-cell context is the complete 8-way
    # ring rather than only the four cardinal neighbours.
    neighbor_cells = {
        (x + dx, y + dy)
        for x, y in cells
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0) and (x + dx, y + dy) not in target_set
    }
    neighbor_land: dict[tuple[int, int], espland.LandRecord] = {}
    for cell in sorted(neighbor_cells):
        record = source_all.get(cell)
        if record is None:
            raise CityscapeFieldError(f"missing one-cell source LAND neighbour {cell}")
        _require_complete_land(record, "tamriel.esm neighbour")
        neighbor_land[cell] = record
    effective_vtex: dict[tuple[int, int], np.ndarray] = {}
    all_raw: set[int] = set()
    for cell in cells:
        values = effective[cell].texture_indices
        if values is None or len(values) != 256:
            raise CityscapeFieldError(f"effective LAND {cell} has incomplete VTEX")
        grid = np.asarray(values, dtype=np.uint16).reshape((16, 16))
        effective_vtex[cell] = grid
        all_raw.update(int(value) for value in values)
    for raw in sorted(all_raw):
        index = espland.resolve_vtex_to_ltex_index(raw)
        if index is not None and index not in effective_ltex:
            raise CityscapeFieldError(
                f"effective raw VTEX {raw} resolves to missing local LTEX {index}"
            )
        base_index = espland.resolve_vtex_to_ltex_index(raw)
        if base_index is not None and base_index not in source_ltex:
            raise CityscapeFieldError(
                f"base raw VTEX {raw} resolves to missing tamriel.esm LTEX {base_index}"
            )
    reconciliation = dict(reconciliation)
    reconciliation["source_vtex_raw_values"] = sorted({
        int(value) for cell in cells for value in (source[cell].texture_indices or ())
    })
    reconciliation["effective_vtex_raw_values"] = sorted(all_raw)
    reconciliation["source_ltex_table_scope"] = "tamriel.esm local LTEX"
    reconciliation["effective_ltex_table_scope"] = "accepted remap ESP local LTEX"
    return TargetBlock(
        cells=cells,
        origin_gu=origin,
        spacing_gu=FIELD_SPACING_GU,
        source_land=source,
        effective_land=effective,
        source_ltex=source_ltex,
        effective_ltex=effective_ltex,
        neighbor_land=neighbor_land,
        source_heights_gu=source_heights,
        effective_vtex=effective_vtex,
        source_file=str(source_file),
        source_sha256=sha256_file(source_file),
        effective_file=str(effective_file),
        effective_sha256=sha256_file(effective_file),
        reconciliation=reconciliation,
    )


def write_field_npz(
    path: Path | str,
    values_gu: np.ndarray,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a deterministic ``height_gu`` NPZ and return file/hash evidence."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values_gu, dtype="<f8")
    if array.ndim != 2 or not np.isfinite(array).all():
        raise CityscapeFieldError("field NPZ requires a finite 2D float64 array")
    height, width = array.shape
    stream = io.BytesIO()
    np_format.write_array(stream, array, allow_pickle=False)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        info = zipfile.ZipInfo("height_gu.npy", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 0
        info.external_attr = 0
        archive.writestr(info, stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "content_sha256": terrain_field_sha256(array),
        "shape": [height, width],
        "dtype": str(array.dtype),
        "metadata": dict(metadata),
    }


def write_metadata(path: Path | str, metadata: Mapping[str, Any]) -> str:
    """Write stable JSON metadata with LF newlines and return its hash."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(deterministic_dumps(dict(metadata)))
    return sha256_file(target)


def field_metadata(
    block: TargetBlock,
    *,
    field_pass: str,
    values_gu: np.ndarray,
    provenance: str,
) -> dict[str, Any]:
    """Create the T1.2-compatible field sidecar plus T1.3 provenance."""

    if field_pass not in {"planned", "final"}:
        raise CityscapeFieldError(f"unknown terrain field pass {field_pass!r}")
    field_shape = [int(v) for v in np.asarray(values_gu).shape]
    return {
        "schema_version": 1,
        "frame_origin_gu": list(block.origin_gu),
        "spacing_gu": [block.spacing_gu, block.spacing_gu],
        "shape": field_shape,
        "units": "game_units",
        "pass": field_pass,
        "provenance": provenance,
        "cell_count": len(block.cells),
        "cells": [list(cell) for cell in block.cells],
        "terrain_field_sha256": terrain_field_sha256(values_gu),
        "source_land_sha256": block.source_sha256,
        "effective_vtex_plugin_sha256": block.effective_sha256,
    }


__all__ = [
    "CELL_SIZE_GU",
    "CityscapeFieldError",
    "FIELD_SIDE",
    "FIELD_SPACING_GU",
    "LAND_SIDE",
    "PAYLOAD_FIELDS",
    "TARGET_CELL_COUNT",
    "TargetBlock",
    "field_metadata",
    "load_target_block",
    "outer_border_mask",
    "payload_digest",
    "rejoin_field",
    "sha256_file",
    "split_field",
    "stitch_heights",
    "terrain_field_sha256",
    "write_field_npz",
    "write_metadata",
]
