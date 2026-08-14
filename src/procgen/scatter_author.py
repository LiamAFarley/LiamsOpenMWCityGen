"""Author and verify the accepted Falkreath v6 scatter document.

Pipeline position: this module is the deterministic authoring core between the
schema-6 scatter document produced by :mod:`procgen.scatter_generate` and the
tes3conv JSON/ESP boundary.  It does not read or write a source ESM, invoke a
converter, inspect assets, or implement OpenMW registration.  The command-line
driver owns those side effects; this module owns the input contract, local
STAT/CELL construction, and structural comparisons used after conversion.

Inputs are the v6 document's ``density.cells`` placement rows.  Outputs are a
flat tes3conv plugin document containing one Header, one local Static record
per case-insensitive mesh, and one exterior Cell per expected grid.  The
authoring invariants are deliberately stronger than the general TES3 JSON
validator: the Falkreath window is exactly 49 cells, the accepted document has
the declared reference count equals the actual placement rows, source
reference IDs are unique, and every authored CELL reference is temporary with
a zero master index and a one-based per-cell reference index.  Reference count
is deliberately not a historical constant: schema v6 accepts the validated
count emitted by the generator.

The verifier functions are converter-independent.  They compare records by
stable TES3 identity rather than relying on converter ordering and admit only
the representation changes empirically expected from tes3conv: float32
rounding and omission of a new-reference scale of exactly ``1.0``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from numbers import Integral, Real
from typing import Any, Callable, Iterable, Mapping, Sequence

from .tes3json import (
    JsonObject,
    PluginDoc,
    build_cell,
    build_reference,
    build_static,
    new_plugin,
)


SCHEMA_VERSION = 6
FALKREATH_BOUNDS = (-95, -89, -11, -5)
EXPECTED_CELL_COUNT = 49
STAT_ID_PREFIX = "PTSC_"

_CATEGORIES = frozenset({"flora", "rocks", "cliff"})


class ScatterAuthorError(ValueError):
    """Raised when a source scatter document cannot satisfy the v6 contract."""


@dataclass(frozen=True)
class SourceReference:
    """The source fields retained for one accepted placement."""

    source_id: str
    mesh: str
    mesh_key: str
    category: str
    grid: tuple[int, int]
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float]
    scale: float


@dataclass(frozen=True)
class SourceCell:
    """A validated source cell and its source-ID-sorted references."""

    grid: tuple[int, int]
    references: tuple[SourceReference, ...]


@dataclass(frozen=True)
class ValidatedScatter:
    """Validated v6 input facts used by the authoring builder."""

    cells: tuple[SourceCell, ...]
    references: tuple[SourceReference, ...]
    canonical_meshes: tuple[tuple[str, str], ...]
    declared_ref_count: int
    declared_category_counts: Mapping[str, int]


@dataclass(frozen=True)
class AuthoringResult:
    """Plugin document plus the deterministic manifest needed by verification."""

    plugin: PluginDoc
    mesh_to_stat: Mapping[str, str]
    stat_id_to_mesh: Mapping[str, str]
    stat_manifest: tuple[Mapping[str, str], ...]
    cell_grids: tuple[tuple[int, int], ...]
    reference_count: int

    def manifest(self) -> dict[str, Any]:
        """Return JSON-ready authoring facts without adding metadata to TES3 JSON."""

        return {
            "mesh_to_stat": dict(sorted(self.mesh_to_stat.items())),
            "stat_id_to_mesh": dict(sorted(self.stat_id_to_mesh.items())),
            "stat_manifest": [dict(row) for row in self.stat_manifest],
            "cell_grids": [list(grid) for grid in self.cell_grids],
            "reference_count": self.reference_count,
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScatterAuthorError(f"{path}: must be an object")
    return value


def _require_int(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if not _is_int(value):
        raise ScatterAuthorError(f"{path}: must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ScatterAuthorError(f"{path}: must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ScatterAuthorError(f"{path}: must be at most {maximum}")
    return result


def _require_grid(value: Any, path: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ScatterAuthorError(f"{path}: must be a two-item integer array")
    return (
        _require_int(value[0], f"{path}[0]"),
        _require_int(value[1], f"{path}[1]"),
    )


def _require_vector(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ScatterAuthorError(f"{path}: must be a three-item numeric array")
    result: list[float] = []
    for index, item in enumerate(value):
        if not _is_finite_number(item):
            raise ScatterAuthorError(f"{path}[{index}]: must be finite")
        result.append(float(item))
    return (result[0], result[1], result[2])


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or not value.strip() or "\x00" in value:
        raise ScatterAuthorError(f"{path}: must be a non-empty string without NUL")
    return value


def _expected_grids() -> set[tuple[int, int]]:
    min_x, max_x, min_y, max_y = FALKREATH_BOUNDS
    return {(x, y) for y in range(min_y, max_y + 1) for x in range(min_x, max_x + 1)}


def _canonical_meshes(meshes: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Return ``(casefolded key, canonical spelling)`` in stable key order."""

    spellings: dict[str, set[str]] = {}
    for mesh in meshes:
        spellings.setdefault(mesh.casefold(), set()).add(mesh)
    return tuple(
        (key, min(values, key=lambda value: (value.casefold(), value)))
        for key, values in sorted(spellings.items())
    )


def _stable_mesh_digest(mesh_key: str) -> str:
    """Return a process-independent digest for a case-folded mesh path."""

    return hashlib.sha256(mesh_key.encode("utf-8")).hexdigest()


def allocate_stat_ids(
    meshes: Iterable[str],
    *,
    digest_function: Callable[[str], str] | None = None,
) -> dict[str, str]:
    """Assign deterministic ``PTSC_`` IDs to case-insensitive mesh keys.

    Meshes are first grouped by ``casefold`` and processed in case-folded
    lexical order.  The normal ID uses the first 12 hexadecimal characters of
    SHA-256.  If a shortened digest collides, the already-stable order assigns
    ``_01``, ``_02`` and so on until the case-insensitive ID is free.  The
    injectable digest function is intentionally small and exists so a focused
    unit test can exercise the collision branch without relying on a
    cryptographic collision.
    """

    canonical = _canonical_meshes(meshes)
    digest = digest_function or _stable_mesh_digest
    result: dict[str, str] = {}
    used: dict[str, str] = {}
    for mesh_key, _mesh in canonical:
        raw_digest = digest(mesh_key)
        if not isinstance(raw_digest, str) or len(raw_digest) < 12:
            raise ScatterAuthorError("stat-id digest function returned fewer than 12 characters")
        if any(character not in "0123456789abcdefABCDEF" for character in raw_digest):
            raise ScatterAuthorError("stat-id digest function returned non-hexadecimal characters")
        base = f"{STAT_ID_PREFIX}{raw_digest[:12].lower()}"
        candidate = base
        suffix = 0
        while candidate.casefold() in used and used[candidate.casefold()] != mesh_key:
            suffix += 1
            candidate = f"{base}_{suffix:02d}"
        result[mesh_key] = candidate
        used[candidate.casefold()] = mesh_key
    if len({value.casefold() for value in result.values()}) != len(result):
        raise ScatterAuthorError("stat-id allocation produced a case-insensitive collision")
    return result


def validate_scatter_document(document: Mapping[str, Any]) -> ValidatedScatter:
    """Validate the exact v6 Falkreath input contract before authoring.

    This gate deliberately rejects incomplete documents instead of silently
    treating absent cells or fields as empty.  The generator's audit fields are
    not copied into TES3 records, but its declared category/reference counts
    are checked against the placement rows here so a stale generation output
    cannot become a valid-looking plugin.
    """

    root = _require_mapping(document, "document")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ScatterAuthorError(
            f"schema_version: expected integer {SCHEMA_VERSION}, got {root.get('schema_version')!r}"
        )

    scope = _require_mapping(root.get("scope"), "scope")
    bounds_value = scope.get("bounds_cells")
    if not isinstance(bounds_value, Sequence) or isinstance(bounds_value, (str, bytes)) or len(bounds_value) != 4:
        raise ScatterAuthorError("scope.bounds_cells: must be a four-item integer array")
    bounds = tuple(_require_int(value, f"scope.bounds_cells[{index}]") for index, value in enumerate(bounds_value))
    if bounds != FALKREATH_BOUNDS:
        raise ScatterAuthorError(f"scope.bounds_cells: expected {list(FALKREATH_BOUNDS)}, got {list(bounds)}")
    declared_cell_count = _require_int(scope.get("cell_count"), "scope.cell_count", minimum=0)
    if declared_cell_count != EXPECTED_CELL_COUNT:
        raise ScatterAuthorError(
            f"scope.cell_count: expected {EXPECTED_CELL_COUNT}, got {declared_cell_count}"
        )

    density = _require_mapping(root.get("density"), "density")
    raw_cells = density.get("cells")
    if not isinstance(raw_cells, list):
        raise ScatterAuthorError("density.cells: must be an array")
    if len(raw_cells) != EXPECTED_CELL_COUNT:
        raise ScatterAuthorError(
            f"density.cells: expected {EXPECTED_CELL_COUNT} cells, got {len(raw_cells)}"
        )

    expected_grids = _expected_grids()
    seen_grids: set[tuple[int, int]] = set()
    seen_ref_ids: set[str] = set()
    source_cells: list[SourceCell] = []
    all_references: list[SourceReference] = []
    mesh_values: list[str] = []
    category_counts: dict[str, int] = {category: 0 for category in sorted(_CATEGORIES)}

    for cell_index, raw_cell in enumerate(raw_cells):
        cell_path = f"density.cells[{cell_index}]"
        cell = _require_mapping(raw_cell, cell_path)
        grid = _require_grid(cell.get("grid"), f"{cell_path}.grid")
        if grid not in expected_grids:
            raise ScatterAuthorError(f"{cell_path}.grid: outside Falkreath bounds: {list(grid)}")
        if grid in seen_grids:
            raise ScatterAuthorError(f"{cell_path}.grid: duplicate grid {list(grid)}")
        seen_grids.add(grid)
        raw_refs = cell.get("refs")
        if not isinstance(raw_refs, list):
            raise ScatterAuthorError(f"{cell_path}.refs: missing or not an array")
        if not raw_refs:
            raise ScatterAuthorError(f"{cell_path}.refs: empty reference list")

        cell_references: list[SourceReference] = []
        for ref_index, raw_ref in enumerate(raw_refs):
            ref_path = f"{cell_path}.refs[{ref_index}]"
            ref = _require_mapping(raw_ref, ref_path)
            source_id = _require_nonempty_string(ref.get("ref_id"), f"{ref_path}.ref_id")
            if source_id != source_id.strip():
                raise ScatterAuthorError(f"{ref_path}.ref_id: surrounding whitespace is not allowed")
            if source_id in seen_ref_ids:
                raise ScatterAuthorError(f"{ref_path}.ref_id: duplicate source reference ID {source_id!r}")
            seen_ref_ids.add(source_id)

            source_grid = _require_grid(ref.get("cell"), f"{ref_path}.cell")
            if source_grid != grid:
                raise ScatterAuthorError(
                    f"{ref_path}.cell: {list(source_grid)} does not match containing grid {list(grid)}"
                )
            mesh = _require_nonempty_string(ref.get("mesh"), f"{ref_path}.mesh")
            category = _require_nonempty_string(ref.get("category"), f"{ref_path}.category")
            if category not in _CATEGORIES:
                raise ScatterAuthorError(f"{ref_path}.category: unsupported category {category!r}")
            translation = _require_vector(ref.get("position_gu"), f"{ref_path}.position_gu")
            rotation = _require_vector(ref.get("rotation_radians"), f"{ref_path}.rotation_radians")
            scale_value = ref.get("scale")
            if not _is_finite_number(scale_value):
                raise ScatterAuthorError(f"{ref_path}.scale: must be a finite number")
            scale = float(scale_value)
            if not 0.5 <= scale <= 2.0:
                raise ScatterAuthorError(f"{ref_path}.scale: must be in the TES3 range 0.5..2.0")

            # These optional fields are not consumed from the source document,
            # but rejecting bad supplied authoring indices prevents a caller
            # from mistaking them for a validated identity.
            if "refr_index" in ref:
                _require_int(ref["refr_index"], f"{ref_path}.refr_index", minimum=1, maximum=0xFFFFFF)
            if "mast_index" in ref and _require_int(ref["mast_index"], f"{ref_path}.mast_index", minimum=0, maximum=255) != 0:
                raise ScatterAuthorError(f"{ref_path}.mast_index: must be zero for new references")
            if "temporary" in ref and not isinstance(ref["temporary"], bool):
                raise ScatterAuthorError(f"{ref_path}.temporary: must be a boolean when supplied")

            source_reference = SourceReference(
                source_id=source_id,
                mesh=mesh,
                mesh_key=mesh.casefold(),
                category=category,
                grid=grid,
                translation=translation,
                rotation=rotation,
                scale=scale,
            )
            cell_references.append(source_reference)
            all_references.append(source_reference)
            mesh_values.append(mesh)
            category_counts[category] += 1
        source_cells.append(SourceCell(grid=grid, references=tuple(cell_references)))

    if seen_grids != expected_grids:
        missing = sorted(expected_grids - seen_grids, key=lambda grid: (grid[1], grid[0]))
        raise ScatterAuthorError(f"density.cells: missing grids {missing}")

    placement_stats = _require_mapping(root.get("placement_stats"), "placement_stats")
    declared_refs = _require_int(placement_stats.get("total_refs"), "placement_stats.total_refs", minimum=0)
    if declared_refs != len(all_references):
        raise ScatterAuthorError(
            f"placement_stats.total_refs: declared {declared_refs}, actual {len(all_references)}"
        )
    declared_unique_meshes = _require_int(
        placement_stats.get("unique_meshes"), "placement_stats.unique_meshes", minimum=0
    )
    canonical_meshes = _canonical_meshes(mesh_values)
    if declared_unique_meshes != len(canonical_meshes):
        raise ScatterAuthorError(
            "placement_stats.unique_meshes: "
            f"declared {declared_unique_meshes}, actual {len(canonical_meshes)}"
        )
    raw_declared_categories = _require_mapping(placement_stats.get("by_category"), "placement_stats.by_category")
    declared_categories: dict[str, int] = {}
    for category in sorted(_CATEGORIES):
        declared_categories[category] = _require_int(
            raw_declared_categories.get(category),
            f"placement_stats.by_category.{category}",
            minimum=0,
        )
        if declared_categories[category] != category_counts[category]:
            raise ScatterAuthorError(
                f"placement_stats.by_category.{category}: declared {declared_categories[category]}, "
                f"actual {category_counts[category]}"
            )

    sorted_cells = tuple(sorted(source_cells, key=lambda cell: (cell.grid[1], cell.grid[0])))
    return ValidatedScatter(
        cells=sorted_cells,
        references=tuple(all_references),
        canonical_meshes=canonical_meshes,
        declared_ref_count=declared_refs,
        declared_category_counts=declared_categories,
    )


def build_scatter_plugin(
    document: Mapping[str, Any],
) -> AuthoringResult:
    """Build a masterless tes3conv plugin from one validated v6 input.

    The generated plugin defines its local STAT/CELL records and declares no
    TES3 masters.  ``tamriel.esm`` is loaded separately by the user's load
    order; accepting a source path or file size here would reintroduce the
    launcher-blocking dependency this stage is specifically meant to avoid.
    """

    validated = validate_scatter_document(document)

    mesh_to_stat = allocate_stat_ids(mesh for _key, mesh in validated.canonical_meshes)
    # ``allocate_stat_ids`` keys by case-folded spelling.  Rebuild the explicit
    # map here so a case-only spelling variant in a future input resolves to the
    # same local STAT without adding a duplicate definition.
    stat_id_to_mesh: dict[str, str] = {}
    stat_manifest: list[Mapping[str, str]] = []
    for mesh_key, mesh in validated.canonical_meshes:
        stat_id = mesh_to_stat[mesh_key]
        stat_id_to_mesh[stat_id] = mesh
        stat_manifest.append({"mesh": mesh, "mesh_key": mesh_key, "stat_id": stat_id})

    records: PluginDoc = new_plugin(
        {
            "author": "ProcGen",
            "description": "Procedural Falkreath scatter v6",
            "file_type": "Esp",
            "num_objects": 0,  # replaced below after the complete record list exists
            "masters": [],
        }
    )
    records.extend(
        build_static(stat_id, stat_id_to_mesh[stat_id])
        for stat_id in (entry["stat_id"] for entry in stat_manifest)
    )

    for cell in validated.cells:
        authored_references: list[JsonObject] = []
        source_references = sorted(cell.references, key=lambda reference: reference.source_id)
        for refr_index, source_reference in enumerate(source_references, start=1):
            authored_references.append(
                build_reference(
                    mesh_to_stat[source_reference.mesh_key],
                    refr_index,
                    translation=source_reference.translation,
                    rotation=source_reference.rotation,
                    mast_index=0,
                    temporary=True,
                    scale=source_reference.scale,
                )
            )
        records.append(build_cell("", list(cell.grid), references=authored_references))

    # tes3conv recomputes this field when saving, but keeping the authoring JSON
    # accurate makes the pre-conversion document independently inspectable.
    records[0]["num_objects"] = len(records) - 1
    if records[0]["num_objects"] != len(stat_manifest) + len(validated.cells):
        raise ScatterAuthorError("Header.num_objects does not match authored records")

    return AuthoringResult(
        plugin=records,
        mesh_to_stat=mesh_to_stat,
        stat_id_to_mesh=stat_id_to_mesh,
        stat_manifest=tuple(stat_manifest),
        cell_grids=tuple(cell.grid for cell in validated.cells),
        reference_count=validated.declared_ref_count,
    )


# A short alias keeps the public authoring operation easy to discover in a
# REPL while the longer name documents the input family in call sites.
build_plugin = build_scatter_plugin


def _records(document: Sequence[Mapping[str, Any]], record_type: str) -> list[Mapping[str, Any]]:
    return [record for record in document if record.get("type") == record_type]


def _grid_from_plugin_cell(record: Mapping[str, Any], path: str) -> tuple[int, int] | None:
    data = record.get("data")
    if not isinstance(data, Mapping):
        return None
    grid = data.get("grid")
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or len(grid) != 2:
        return None
    if not all(_is_int(value) for value in grid):
        return None
    return (int(grid[0]), int(grid[1]))


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


@dataclass
class _NumericComparison:
    exact: int = 0
    converter_tolerance: int = 0
    mismatches: int = 0
    max_abs_delta: float = 0.0

    def add(self, expected: float, actual: float | None) -> str:
        if actual is None or not _is_finite_number(actual):
            self.mismatches += 1
            return "missing-or-nonfinite"
        actual_float = float(actual)
        delta = abs(actual_float - float(expected))
        self.max_abs_delta = max(self.max_abs_delta, delta)
        if actual_float == float(expected):
            self.exact += 1
            return "exact"
        # tes3conv writes a float32, then serializes it as a shortest decimal.
        # Compare the resulting JSON number by its re-quantized TES3 bit
        # pattern, not by the decimal spelling or by a free numerical epsilon.
        if _f32(actual_float) == _f32(float(expected)):
            self.converter_tolerance += 1
            return "float32"
        self.mismatches += 1
        return "different"

    def as_dict(self) -> dict[str, Any]:
        return {
            "exact": self.exact,
            "converter_tolerance": self.converter_tolerance,
            "mismatches": self.mismatches,
            "max_abs_delta": self.max_abs_delta,
        }


def _reference_key(reference: Mapping[str, Any]) -> tuple[int, int] | None:
    mast_index = reference.get("mast_index")
    refr_index = reference.get("refr_index")
    if not _is_int(mast_index) or not _is_int(refr_index):
        return None
    return (int(mast_index), int(refr_index))


def _compare_reference_sets(
    expected_cells: Mapping[tuple[int, int], Mapping[tuple[int, int], Mapping[str, Any]]],
    actual_cells: Mapping[tuple[int, int], Mapping[tuple[int, int], Mapping[str, Any]]],
    *,
    actual_temporary: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    """Compare all CELL reference transforms and TES3 identity fields."""

    issues: list[str] = []
    transform = {
        "translation": _NumericComparison(),
        "rotation": _NumericComparison(),
        "scale": _NumericComparison(),
    }
    scale_omitted_default = 0
    expected_ref_count = sum(len(refs) for refs in expected_cells.values())
    actual_ref_count = sum(len(refs) for refs in actual_cells.values())
    if expected_ref_count != actual_ref_count:
        issues.append(f"reference count expected {expected_ref_count}, got {actual_ref_count}")

    for grid in sorted(set(expected_cells) | set(actual_cells), key=lambda value: (value[1], value[0])):
        expected_refs = expected_cells.get(grid, {})
        actual_refs = actual_cells.get(grid, {})
        if grid not in expected_cells:
            issues.append(f"unexpected cell grid {list(grid)}")
        if grid not in actual_cells:
            issues.append(f"missing cell grid {list(grid)}")
        if set(expected_refs) != set(actual_refs):
            missing = sorted(set(expected_refs) - set(actual_refs))
            extra = sorted(set(actual_refs) - set(expected_refs))
            if missing:
                issues.append(f"cell {list(grid)} missing reference keys {missing[:5]}")
            if extra:
                issues.append(f"cell {list(grid)} has unexpected reference keys {extra[:5]}")
        for key in sorted(set(expected_refs) & set(actual_refs)):
            expected = expected_refs[key]
            actual = actual_refs[key]
            if expected.get("id") != actual.get("id"):
                issues.append(f"cell {list(grid)} ref {key}: STAT id differs")
            if expected.get("mast_index") != actual.get("mast_index"):
                issues.append(f"cell {list(grid)} ref {key}: mast_index differs")
            if not actual_temporary(actual):
                issues.append(f"cell {list(grid)} ref {key}: temporary flag is not true")

            expected_translation = expected.get("translation")
            actual_translation = actual.get("translation")
            expected_rotation = expected.get("rotation")
            actual_rotation = actual.get("rotation")
            if not isinstance(expected_translation, Sequence) or len(expected_translation) != 3:
                issues.append(f"cell {list(grid)} ref {key}: expected translation malformed")
            elif not isinstance(actual_translation, Sequence) or len(actual_translation) != 3:
                for _ in range(3):
                    transform["translation"].mismatches += 1
                issues.append(f"cell {list(grid)} ref {key}: translation missing or malformed")
            else:
                for expected_value, actual_value in zip(expected_translation, actual_translation):
                    transform["translation"].add(float(expected_value), actual_value)

            if not isinstance(expected_rotation, Sequence) or len(expected_rotation) != 3:
                issues.append(f"cell {list(grid)} ref {key}: expected rotation malformed")
            elif not isinstance(actual_rotation, Sequence) or len(actual_rotation) != 3:
                for _ in range(3):
                    transform["rotation"].mismatches += 1
                issues.append(f"cell {list(grid)} ref {key}: rotation missing or malformed")
            else:
                for expected_value, actual_value in zip(expected_rotation, actual_rotation):
                    transform["rotation"].add(float(expected_value), actual_value)

            expected_scale = expected.get("scale")
            actual_scale = actual.get("scale")
            if actual_scale is None and expected_scale == 1.0:
                # tes3conv intentionally drops XSCL=1.0 for a new mast-index-0
                # reference.  Treat the decoded semantic value as one while
                # recording the representation delta separately.
                scale_omitted_default += 1
                transform["scale"].converter_tolerance += 1
            elif _is_finite_number(expected_scale):
                transform["scale"].add(float(expected_scale), actual_scale)
            else:
                transform["scale"].mismatches += 1
                issues.append(f"cell {list(grid)} ref {key}: expected scale malformed")

    for label, comparison in transform.items():
        if comparison.mismatches:
            issues.append(f"{label} has {comparison.mismatches} mismatched scalar values")
    return {
        "expected_reference_count": expected_ref_count,
        "actual_reference_count": actual_ref_count,
        "temporary_true_count": sum(
            1 for refs in actual_cells.values() for reference in refs.values() if actual_temporary(reference)
        ),
        "transform_scale": {label: comparison.as_dict() for label, comparison in transform.items()},
        "scale_omitted_default_count": scale_omitted_default,
        "issues": issues,
        "passed": not issues,
    }


def _plugin_cell_reference_maps(
    document: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[int, int], Mapping[str, Any]], dict[tuple[int, int], dict[tuple[int, int], Mapping[str, Any]]], list[str]]:
    cells: dict[tuple[int, int], Mapping[str, Any]] = {}
    references: dict[tuple[int, int], dict[tuple[int, int], Mapping[str, Any]]] = {}
    issues: list[str] = []
    for index, cell in enumerate(_records(document, "Cell")):
        grid = _grid_from_plugin_cell(cell, f"Cell[{index}]")
        if grid is None:
            issues.append(f"Cell[{index}] has no valid exterior grid")
            continue
        if grid in cells:
            issues.append(f"duplicate plugin cell grid {list(grid)}")
        cells[grid] = cell
        raw_refs = cell.get("references")
        if not isinstance(raw_refs, list):
            issues.append(f"cell {list(grid)} has no reference array")
            continue
        indexed: dict[tuple[int, int], Mapping[str, Any]] = {}
        for ref_index, reference in enumerate(raw_refs):
            if not isinstance(reference, Mapping):
                issues.append(f"cell {list(grid)} reference {ref_index} is not an object")
                continue
            key = _reference_key(reference)
            if key is None:
                issues.append(f"cell {list(grid)} reference {ref_index} has invalid indices")
                continue
            if key in indexed:
                issues.append(f"cell {list(grid)} has duplicate reference key {key}")
            indexed[key] = reference
        references[grid] = indexed
    return cells, references, issues


def compare_roundtrip(
    authored: AuthoringResult,
    roundtrip: PluginDoc,
) -> dict[str, Any]:
    """Compare the authoring document with one ESP→JSON conversion result."""

    issues: list[str] = []
    if not isinstance(roundtrip, list):
        return {"passed": False, "issues": ["round-trip document is not a top-level array"]}
    expected_header = next((record for record in authored.plugin if record.get("type") == "Header"), None)
    actual_headers = _records(roundtrip, "Header")
    actual_header = actual_headers[0] if actual_headers else None
    expected_masters = expected_header.get("masters") if expected_header else None
    actual_masters = actual_header.get("masters") if actual_header else None
    master_match = bool(actual_header and actual_masters == expected_masters) if expected_header else False
    if not master_match:
        issues.append("Header.masters differs from the authored master list")
    if actual_masters != []:
        issues.append(f"Header.masters is not masterless: {actual_masters!r}")
    expected_num_objects = expected_header.get("num_objects") if expected_header else None
    if not actual_header or actual_header.get("num_objects") != expected_num_objects:
        issues.append("Header.num_objects differs")

    expected_stats = {
        str(record.get("id")): record.get("mesh")
        for record in _records(authored.plugin, "Static")
        if isinstance(record.get("id"), str)
    }
    actual_stats = {
        str(record.get("id")): record.get("mesh")
        for record in _records(roundtrip, "Static")
        if isinstance(record.get("id"), str)
    }
    stat_mapping_match = expected_stats == actual_stats
    if not stat_mapping_match:
        issues.append("STAT id-to-mesh mapping differs")

    expected_cells, expected_refs, expected_cell_issues = _plugin_cell_reference_maps(authored.plugin)
    actual_cells, actual_refs, actual_cell_issues = _plugin_cell_reference_maps(roundtrip)
    issues.extend(expected_cell_issues)
    issues.extend(actual_cell_issues)
    if set(expected_cells) != set(actual_cells):
        issues.append("CELL grid set differs")
    reference_comparison = _compare_reference_sets(
        expected_refs,
        actual_refs,
        actual_temporary=lambda reference: reference.get("temporary") is True,
    )
    issues.extend(reference_comparison["issues"])
    if len(_records(roundtrip, "Cell")) != len(expected_cells):
        issues.append("CELL record count differs")
    forbidden_types = sorted(
        record_type
        for record_type in ("Landscape", "LandscapeTexture", "Region")
        if _records(roundtrip, record_type)
    )
    if forbidden_types:
        issues.append(f"forbidden record types present: {forbidden_types}")

    return {
        "passed": not issues,
        "issues": issues,
        "master_match": master_match,
        "expected_masters": expected_masters,
        "actual_masters": actual_masters,
        "masterless": actual_masters == [],
        "expected_header_num_objects": expected_num_objects,
        "actual_header_num_objects": actual_header.get("num_objects") if actual_header else None,
        "stat_count": len(actual_stats),
        "stat_mapping_match": stat_mapping_match,
        "cell_count": len(actual_cells),
        "expected_grids": [list(grid) for grid in sorted(expected_cells, key=lambda value: (value[1], value[0]))],
        "actual_grids": [list(grid) for grid in sorted(actual_cells, key=lambda value: (value[1], value[0]))],
        "reference_comparison": reference_comparison,
        "forbidden_record_types": forbidden_types,
    }


def compare_binary_scan(
    authored: AuthoringResult,
    scan_result: Any,
) -> dict[str, Any]:
    """Compare one :func:`procgen.espscan.scan_file` result with the plugin."""

    issues: list[str] = []
    expected_stats = authored.stat_id_to_mesh
    actual_stats = getattr(scan_result, "stat_models", {})
    if dict(expected_stats) != dict(actual_stats):
        issues.append("binary STAT id-to-mesh mapping differs")

    record_counts = getattr(scan_result, "record_counts", {})
    expected_record_types = {"TES3": 1, "STAT": len(expected_stats), "CELL": EXPECTED_CELL_COUNT}
    for record_type, expected_count in expected_record_types.items():
        actual_count = int(record_counts.get(record_type, 0))
        if actual_count != expected_count:
            issues.append(f"binary {record_type} count expected {expected_count}, got {actual_count}")
    forbidden_types = sorted(record_type for record_type in ("LAND", "LTEX", "REGN") if record_counts.get(record_type, 0))
    if forbidden_types:
        issues.append(f"binary forbidden record types present: {forbidden_types}")

    expected_cells, expected_refs, cell_issues = _plugin_cell_reference_maps(authored.plugin)
    issues.extend(cell_issues)
    binary_cells: dict[tuple[int, int], dict[tuple[int, int], Mapping[str, Any]]] = {}
    for cell in getattr(scan_result, "cells", ()):
        grid = getattr(cell, "grid", None)
        if grid is None or getattr(cell, "is_interior", True):
            issues.append("binary CELL is interior or has no exterior grid")
            continue
        grid_key = (int(grid[0]), int(grid[1]))
        if grid_key in binary_cells:
            issues.append(f"duplicate binary cell grid {list(grid_key)}")
        refs: dict[tuple[int, int], Mapping[str, Any]] = {}
        for reference in getattr(cell, "references", ()):
            key = (int(reference.mast_index), int(reference.refr_index))
            if key in refs:
                issues.append(f"binary cell {list(grid_key)} has duplicate reference key {key}")
            refs[key] = {
                "mast_index": reference.mast_index,
                "refr_index": reference.refr_index,
                "id": reference.object_id,
                "temporary": reference.temporary,
                "translation": list(reference.position) if reference.position is not None else None,
                "rotation": list(reference.rotation) if reference.rotation is not None else None,
                "scale": reference.scale,
            }
            if reference.unresolved:
                issues.append(f"binary unresolved reference {reference.object_id!r} in {list(grid_key)}")
        binary_cells[grid_key] = refs

    if set(expected_cells) != set(binary_cells):
        issues.append("binary CELL grid set differs")
    binary_reference_comparison = _compare_reference_sets(
        expected_refs,
        binary_cells,
        actual_temporary=lambda reference: reference.get("temporary") is True,
    )
    issues.extend(binary_reference_comparison["issues"])
    unresolved = int(getattr(scan_result, "unresolved_reference_count", -1))
    if unresolved != 0:
        issues.append(f"binary unresolved reference count is {unresolved}, expected 0")
    malformed = int(getattr(scan_result, "malformed_records", -1))
    if malformed != 0:
        issues.append(f"binary malformed record count is {malformed}, expected 0")

    return {
        "passed": not issues,
        "issues": issues,
        "record_counts": dict(sorted(record_counts.items())),
        "stat_count": int(getattr(scan_result, "static_count", record_counts.get("STAT", 0))),
        "cell_count": int(getattr(scan_result, "cell_count", len(binary_cells))),
        "exterior_cells": int(getattr(scan_result, "exterior_cells", 0)),
        "interior_cells": int(getattr(scan_result, "interior_cells", 0)),
        "reference_count": int(getattr(scan_result, "reference_count", 0)),
        "resolved_mesh_reference_count": int(getattr(scan_result, "resolved_mesh_reference_count", 0)),
        "unresolved_reference_count": unresolved,
        "malformed_records": malformed,
        "forbidden_record_types": forbidden_types,
        "reference_comparison": binary_reference_comparison,
    }


__all__ = [
    "AuthoringResult",
    "EXPECTED_CELL_COUNT",
    "FALKREATH_BOUNDS",
    "SCHEMA_VERSION",
    "STAT_ID_PREFIX",
    "ScatterAuthorError",
    "SourceCell",
    "SourceReference",
    "ValidatedScatter",
    "allocate_stat_ids",
    "build_plugin",
    "build_scatter_plugin",
    "compare_binary_scan",
    "compare_roundtrip",
    "validate_scatter_document",
]
