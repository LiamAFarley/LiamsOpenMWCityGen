"""Explicit semantic road assignments shared by terrain and Cityforge stages.

Pipeline position
------------------
The landscape-remap stage emits a ``road_assignments`` table.  Cityscape,
scatter, groundcover, and site-audit stages consume that table instead of
inferring road meaning from a raw VTEX value or an LTEX record name.

Inputs are JSON-ready mappings containing either ``road_assignments`` or
``road_classes``.  A Cityforge palette may alternatively expose the same
assignments through ``semantic_surfaces.surfaces`` using an explicit
``road_class`` field.  Every assignment must declare its raw VTEX, local LTEX
index, record ID, and texture path; raw VTEX is never derived from an ordinal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class RoadSemanticsError(ValueError):
    """Raised when a region's explicit road assignment table is invalid."""


@dataclass(frozen=True)
class RoadAssignment:
    """One semantic road class and its output LAND/LTEX identity."""

    road_class: str
    raw_vtex: int
    ltex_index: int
    ltex_id: str
    file_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "road_class": self.road_class,
            "raw_vtex": self.raw_vtex,
            "ltex_index": self.ltex_index,
            "ltex_id": self.ltex_id,
            "file_name": self.file_name,
        }


def _assignment_rows(source: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Read one of the accepted explicit assignment representations."""

    raw = source.get("road_assignments")
    if not raw:
        raw = source.get("road_classes")
    if isinstance(raw, Mapping):
        rows: list[tuple[str, Mapping[str, Any]]] = []
        for road_class, value in raw.items():
            if not isinstance(value, Mapping):
                raise RoadSemanticsError(f"road assignment {road_class!r} must be an object")
            rows.append((str(road_class), value))
        return rows
    if isinstance(raw, list):
        rows = []
        for value in raw:
            if not isinstance(value, Mapping):
                raise RoadSemanticsError("road assignment list entries must be objects")
            road_class = value.get("road_class")
            if not isinstance(road_class, str) or not road_class:
                raise RoadSemanticsError("road assignment list entry needs road_class")
            rows.append((road_class, value))
        return rows

    surfaces = source.get("semantic_surfaces")
    surfaces = surfaces.get("surfaces") if isinstance(surfaces, Mapping) else None
    if isinstance(surfaces, list):
        rows = []
        for surface in surfaces:
            if not isinstance(surface, Mapping):
                continue
            # ``road`` is a persisted semantic surface name in the original
            # Cityforge palette.  It is retained as a class label only; its
            # VTEX/LTEX identity still comes entirely from the assignment.
            road_class = surface.get("road_class")
            if road_class is None and surface.get("surface") == "road":
                road_class = "road"
            if not isinstance(road_class, str) or not road_class:
                continue
            assignment = surface.get("planned_assignment")
            if not isinstance(assignment, Mapping):
                raise RoadSemanticsError(f"surface {road_class!r} has no planned_assignment")
            measured = surface.get("measured_identity")
            remap = measured.get("remap_identity") if isinstance(measured, Mapping) else None
            path = remap.get("texture_path") if isinstance(remap, Mapping) else None
            if not path:
                from .regionpalette import REMAP_LTEX_TABLE

                index = assignment.get("planned_ltex_index")
                expected = REMAP_LTEX_TABLE.get(int(index)) if index is not None else None
                if expected and expected[0] == assignment.get("planned_ltex_id"):
                    path = expected[1]
            rows.append((road_class, {
                "raw_vtex": assignment.get("planned_raw_vtex"),
                "ltex_index": assignment.get("planned_ltex_index"),
                "ltex_id": assignment.get("planned_ltex_id"),
                "file_name": path or surface.get("texture_path"),
            }))
        return rows
    return []


def load_road_assignments(source: Mapping[str, Any]) -> dict[str, RoadAssignment]:
    """Validate and return the source's explicit semantic road assignments."""

    rows = _assignment_rows(source)
    if not rows:
        raise RoadSemanticsError("active region has no explicit road_assignments")
    result: dict[str, RoadAssignment] = {}
    raw_seen: dict[int, str] = {}
    index_seen: dict[int, str] = {}
    for road_class, value in rows:
        if road_class in result:
            raise RoadSemanticsError(f"duplicate road class {road_class!r}")
        try:
            raw = int(value["raw_vtex"])
            index = int(value["ltex_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RoadSemanticsError(f"road class {road_class!r} has invalid raw/index") from exc
        ltex_id = value.get("ltex_id")
        file_name = value.get("file_name")
        if raw <= 0 or index < 0 or index != raw - 1:
            raise RoadSemanticsError(
                f"road class {road_class!r} requires positive raw_vtex and ltex_index=raw_vtex-1"
            )
        if not isinstance(ltex_id, str) or not ltex_id:
            raise RoadSemanticsError(f"road class {road_class!r} has no ltex_id")
        if not isinstance(file_name, str) or not file_name:
            raise RoadSemanticsError(f"road class {road_class!r} has no file_name")
        if raw in raw_seen:
            prior = result[raw_seen[raw]]
            if (prior.ltex_index, prior.ltex_id, prior.file_name) != (index, ltex_id, file_name):
                raise RoadSemanticsError(
                    f"raw VTEX {raw} is assigned to incompatible classes {raw_seen[raw]!r} and {road_class!r}"
                )
        if index in index_seen:
            prior = result[index_seen[index]]
            if (prior.raw_vtex, prior.ltex_id, prior.file_name) != (raw, ltex_id, file_name):
                raise RoadSemanticsError(
                    f"LTEX index {index} is assigned to incompatible classes {index_seen[index]!r} and {road_class!r}"
                )
        result[road_class] = RoadAssignment(road_class, raw, index, ltex_id, file_name)
        raw_seen[raw] = road_class
        index_seen[index] = road_class
    return result


def road_raw_values(source: Mapping[str, Any]) -> frozenset[int]:
    """Return configured road raw values, failing if the table is absent."""

    return frozenset(assignment.raw_vtex for assignment in load_road_assignments(source).values())


def road_class_for_plan_road(
    road: Mapping[str, Any],
    assignments: Mapping[str, RoadAssignment],
    hierarchy_map: Mapping[str, str] | None = None,
) -> RoadAssignment:
    """Resolve a plan road's explicit class without a default fallback."""

    road_class = road.get("road_class")
    if road_class is None:
        surface = road.get("surface")
        if isinstance(surface, str) and surface in assignments:
            road_class = surface
    if road_class is None and hierarchy_map is not None:
        hierarchy = road.get("hierarchy")
        if isinstance(hierarchy, str):
            road_class = hierarchy_map.get(hierarchy)
    if road_class is None and len(assignments) == 1 and "road" in assignments:
        # The corridor remap palette has one explicit road class.  In that
        # texture-agnostic mode every generated hierarchy intentionally uses
        # the same configured assignment until a hierarchy map is authored.
        road_class = "road"
    if not isinstance(road_class, str) or road_class not in assignments:
        raise RoadSemanticsError(
            f"plan road {road.get('road_id')!r} needs road_class matching an active assignment"
        )
    return assignments[road_class]


def assignments_as_mapping(assignments: Iterable[RoadAssignment]) -> dict[str, dict[str, Any]]:
    """Serialize assignments for remap reports and downstream documents."""

    return {assignment.road_class: assignment.to_dict() for assignment in assignments}
