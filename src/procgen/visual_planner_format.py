"""Cityforge visual-planning extension format (version 1).

Pipeline position
------------------
This module is the format boundary between a vision-capable settlement
designer and the deterministic visual renderer/analyser.  It is deliberately
*not* a replacement for the accepted T1.1 ``city_plan.json`` contract.  A
visual plan is a small, versioned sibling document which references a T1.1
plan id and adds planning-only records: aligned source-road ids, authored
streets and alleys, polygonal hardstanding, stamp lots, door intents, and
annotations.  T1.1 remains the validator for production plan semantics and
the visual document never changes TES3/LAND/VTEX meaning.

Inputs and outputs
------------------
``validate_extension`` checks the JSON shape and finite numeric values without
loading site or stamp products.  ``canonical_json_bytes`` provides the stable
serialization used by the fixture builder and audit hashes.  The renderer
and advisory analyser resolve the referenced stamp geometry and aligned roads
after this structural gate has passed.

Invariants
----------
* ``schema_version`` and ``kind`` are pinned to this module's version.
* Coordinates are accepted-site survey plan-frame GU (east/north), never
  absolute TES3 placement coordinates.
* Existing road records carry aligned edge ids only; source-space/XCF road
  coordinates do not have a field in this format.
* Extension records are advisory design inputs.  They do not author plugin
  records and do not mutate exact discrete VTEX data.
* Canonical output is deterministic and rejects NaN/Infinity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
KIND = "cityforge_visual_plan_extension"
COORDINATE_FRAME = "site_survey_plan_gu"


class VisualPlanFormatError(ValueError):
    """Raised when a visual-plan extension violates its structural contract."""


@dataclass(frozen=True)
class FormatIssue:
    """One deterministic structural issue."""

    path: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "code": self.code, "message": self.message}


def canonical_json_bytes(value: object) -> bytes:
    """Return stable, finite JSON bytes with a trailing newline."""

    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, indent=2) + "\n").encode("utf-8")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _point(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_finite(v) for v in value)


def _points(value: Any, minimum: int = 2) -> bool:
    return isinstance(value, list) and len(value) >= minimum and all(_point(v) for v in value)


def _required(record: Mapping[str, Any], keys: Iterable[str], path: str,
              issues: list[FormatIssue]) -> None:
    for key in keys:
        if key not in record:
            issues.append(FormatIssue(f"{path}.{key}", "missing_required",
                                      "required field is missing"))


def _unknown(record: Mapping[str, Any], allowed: set[str], path: str,
             issues: list[FormatIssue]) -> None:
    for key in sorted(set(record) - allowed):
        issues.append(FormatIssue(f"{path}.{key}", "unknown_key",
                                  "unknown key in strict visual-plan extension"))


def _check_point(record: Mapping[str, Any], key: str, path: str,
                 issues: list[FormatIssue]) -> None:
    value = record.get(key)
    if not _point(value):
        issues.append(FormatIssue(f"{path}.{key}", "invalid_point",
                                  "expected a finite [x, y] plan-GU pair"))


def _check_polyline(record: Mapping[str, Any], key: str, path: str,
                    issues: list[FormatIssue], minimum: int = 2) -> None:
    value = record.get(key)
    if not _points(value, minimum):
        issues.append(FormatIssue(f"{path}.{key}", "invalid_polyline",
                                  f"expected at least {minimum} finite [x, y] points"))


def _check_polygon(record: Mapping[str, Any], key: str, path: str,
                   issues: list[FormatIssue]) -> None:
    value = record.get(key)
    if not _points(value, 3):
        issues.append(FormatIssue(f"{path}.{key}", "invalid_polygon",
                                  "expected at least three finite [x, y] points"))


def _check_connection_targets(value: Any, path: str,
                              issues: list[FormatIssue]) -> None:
    if not isinstance(value, list):
        issues.append(FormatIssue(path, "invalid_connections",
                                  "connection_targets must be an array"))
        return
    for index, target in enumerate(value):
        target_path = f"{path}[{index}]"
        if not isinstance(target, dict):
            issues.append(FormatIssue(target_path, "invalid_connection",
                                      "connection target must be an object"))
            continue
        allowed = {"target_id", "at_plan_gu", "tolerance_gu", "reason"}
        _unknown(target, allowed, target_path, issues)
        _required(target, ("target_id", "at_plan_gu"), target_path, issues)
        if not isinstance(target.get("target_id"), str) or not target.get("target_id"):
            issues.append(FormatIssue(f"{target_path}.target_id", "invalid_id",
                                      "target_id must be a non-empty string"))
        _check_point(target, "at_plan_gu", target_path, issues)
        if "tolerance_gu" in target and not _finite(target["tolerance_gu"]):
            issues.append(FormatIssue(f"{target_path}.tolerance_gu", "not_finite",
                                      "tolerance_gu must be finite"))


def _check_extension_record(record: Any, index: int, section: str,
                            allowed: set[str], required: tuple[str, ...],
                            issues: list[FormatIssue]) -> None:
    path = f"$.{section}[{index}]"
    if not isinstance(record, dict):
        issues.append(FormatIssue(path, "wrong_type", "record must be an object"))
        return
    _unknown(record, allowed, path, issues)
    _required(record, required, path, issues)


def validate_extension(document: Any) -> list[FormatIssue]:
    """Return all deterministic structural issues in a visual-plan document.

    This is intentionally a structural gate.  Site masks, exact stamp hulls,
    road corridors, and advisory policy belong to
    :mod:`procgen.visual_planner_advisory`, so a format-only caller can inspect
    or edit a plan without loading the large world products.
    """

    issues: list[FormatIssue] = []
    if not isinstance(document, dict):
        return [FormatIssue("$", "wrong_type", "visual plan must be an object")]

    allowed_top = {
        "schema_version", "kind", "plan_id", "base_t1_1_plan_id", "seed",
        "coordinate_frame", "rectangle", "existing_source_roads",
        "authored_roads", "alleys", "road_surface_polygons", "shared_courts",
        "stamps", "districts", "annotations", "advisory_overrides",
        "render_options", "design_notes",
    }
    _unknown(document, allowed_top, "$", issues)
    _required(document, (
        "schema_version", "kind", "plan_id", "seed", "coordinate_frame",
        "rectangle", "existing_source_roads", "authored_roads", "alleys",
        "road_surface_polygons", "shared_courts", "stamps", "districts",
        "annotations", "advisory_overrides",
    ), "$", issues)

    if document.get("schema_version") != SCHEMA_VERSION:
        issues.append(FormatIssue("$.schema_version", "schema_version_mismatch",
                                  f"expected visual extension version {SCHEMA_VERSION}"))
    if document.get("kind") != KIND:
        issues.append(FormatIssue("$.kind", "kind_mismatch", f"expected {KIND!r}"))
    for key in ("plan_id", "coordinate_frame"):
        if not isinstance(document.get(key), str) or not document.get(key):
            issues.append(FormatIssue(f"$.{key}", "invalid_id", "must be a non-empty string"))
    if document.get("coordinate_frame") not in (None, COORDINATE_FRAME):
        issues.append(FormatIssue("$.coordinate_frame", "coordinate_frame_mismatch",
                                  f"expected {COORDINATE_FRAME!r}"))
    if not isinstance(document.get("seed"), int) or isinstance(document.get("seed"), bool):
        issues.append(FormatIssue("$.seed", "invalid_seed", "seed must be an integer"))

    rectangle = document.get("rectangle")
    if not isinstance(rectangle, dict):
        issues.append(FormatIssue("$.rectangle", "wrong_type", "rectangle must be an object"))
    else:
        allowed = {"cell_bounds", "context_margin_gu", "full_site_inset", "world_bounds_gu"}
        _unknown(rectangle, allowed, "$.rectangle", issues)
        _required(rectangle, ("cell_bounds", "context_margin_gu"), "$.rectangle", issues)
        cells = rectangle.get("cell_bounds")
        if not (isinstance(cells, list) and len(cells) == 4 and
                all(isinstance(v, int) and not isinstance(v, bool) for v in cells)):
            issues.append(FormatIssue("$.rectangle.cell_bounds", "invalid_cell_bounds",
                                      "expected [min_x, max_x, min_y, max_y] integers"))
        elif cells[0] > cells[1] or cells[2] > cells[3]:
            issues.append(FormatIssue("$.rectangle.cell_bounds", "invalid_cell_bounds",
                                      "minimum cell must not exceed maximum cell"))
        if not _finite(rectangle.get("context_margin_gu")) or \
                float(rectangle.get("context_margin_gu", -1)) < 0:
            issues.append(FormatIssue("$.rectangle.context_margin_gu", "invalid_margin",
                                      "context_margin_gu must be finite and non-negative"))
        if "full_site_inset" in rectangle and not isinstance(rectangle["full_site_inset"], bool):
            issues.append(FormatIssue("$.rectangle.full_site_inset", "wrong_type",
                                      "full_site_inset must be boolean"))
        if "world_bounds_gu" in rectangle and not _points(rectangle["world_bounds_gu"], 2):
            issues.append(FormatIssue("$.rectangle.world_bounds_gu", "invalid_bounds",
                                      "world_bounds_gu must contain finite points"))

    sections = {
        "existing_source_roads": (
            {"edge_id", "label", "hierarchy", "show_corridor", "corridor_margin_gu",
             "connection_points", "notes"}, ("edge_id",)),
        "authored_roads": (
            {"road_id", "class", "width_gu", "surface", "polyline_plan_gu",
             "connection_targets", "notes"},
            ("road_id", "class", "width_gu", "surface", "polyline_plan_gu",
             "connection_targets")),
        "alleys": (
            {"alley_id", "class", "width_gu", "surface", "polyline_plan_gu",
             "connection_targets", "notes"},
            ("alley_id", "width_gu", "polyline_plan_gu", "connection_targets")),
        "road_surface_polygons": (
            {"region_id", "kind", "surface", "polygon_plan_gu", "district_id", "notes"},
            ("region_id", "kind", "surface", "polygon_plan_gu")),
        "shared_courts": (
            {"court_id", "polygon_plan_gu", "surface", "connection_targets", "notes"},
            ("court_id", "polygon_plan_gu", "connection_targets")),
        "districts": (
            {"district_id", "label", "kind", "polygon_plan_gu", "notes"},
            ("district_id", "label", "polygon_plan_gu")),
        "annotations": (
            {"annotation_id", "kind", "text", "position_plan_gu", "target_id", "notes"},
            ("annotation_id", "kind", "text", "position_plan_gu")),
        "advisory_overrides": (
            {"code", "lot_id", "door_id", "reason", "scope"}, ("code", "reason")),
    }
    for section, (allowed, required) in sections.items():
        values = document.get(section)
        if not isinstance(values, list):
            issues.append(FormatIssue(f"$.{section}", "wrong_type", "section must be an array"))
            continue
        for index, record in enumerate(values):
            _check_extension_record(record, index, section, allowed, required, issues)
            if not isinstance(record, dict):
                continue
            path = f"$.{section}[{index}]"
            if section == "existing_source_roads":
                if not isinstance(record.get("edge_id"), str) or not record.get("edge_id"):
                    issues.append(FormatIssue(f"{path}.edge_id", "invalid_id",
                                              "edge_id must be a non-empty aligned edge id"))
                if "show_corridor" in record and not isinstance(record["show_corridor"], bool):
                    issues.append(FormatIssue(f"{path}.show_corridor", "wrong_type",
                                              "show_corridor must be boolean"))
                if "corridor_margin_gu" in record and not _finite(record["corridor_margin_gu"]):
                    issues.append(FormatIssue(f"{path}.corridor_margin_gu", "not_finite",
                                              "corridor_margin_gu must be finite"))
                if "connection_points" in record and not _points(record["connection_points"], 1):
                    issues.append(FormatIssue(f"{path}.connection_points", "invalid_points",
                                              "connection_points must contain finite points"))
            elif section in ("authored_roads", "alleys"):
                if "width_gu" in record and (not _finite(record["width_gu"]) or
                                              float(record["width_gu"]) <= 0):
                    issues.append(FormatIssue(f"{path}.width_gu", "invalid_width",
                                              "width_gu must be finite and positive"))
                _check_polyline(record, "polyline_plan_gu", path, issues)
                _check_connection_targets(record.get("connection_targets"),
                                          f"{path}.connection_targets", issues)
            elif section in ("road_surface_polygons", "shared_courts", "districts"):
                _check_polygon(record, "polygon_plan_gu", path, issues)
                if section == "road_surface_polygons" and not isinstance(record.get("surface"), str):
                    issues.append(FormatIssue(f"{path}.surface", "invalid_surface",
                                              "surface must be a string"))
                if section == "shared_courts":
                    _check_connection_targets(record.get("connection_targets"),
                                              f"{path}.connection_targets", issues)
            elif section == "annotations":
                _check_point(record, "position_plan_gu", path, issues)
                if not isinstance(record.get("text"), str):
                    issues.append(FormatIssue(f"{path}.text", "invalid_text",
                                              "annotation text must be a string"))

    stamps = document.get("stamps")
    if not isinstance(stamps, list):
        issues.append(FormatIssue("$.stamps", "wrong_type", "stamps must be an array"))
    else:
        stamp_allowed = {
            "lot_id", "stamp_id", "position_plan_gu", "yaw_deg", "district_id",
            "kit", "category", "label", "door_intents", "access_links",
            "show_source_terrain", "show_burial_envelope", "intentional_slope_capable",
            "road_overlap_intent", "anchor_z_gu", "terrain_evidence", "notes",
            "door_targets",
        }
        for index, stamp in enumerate(stamps):
            path = f"$.stamps[{index}]"
            if not isinstance(stamp, dict):
                issues.append(FormatIssue(path, "wrong_type", "stamp placement must be an object"))
                continue
            _unknown(stamp, stamp_allowed, path, issues)
            _required(stamp, ("lot_id", "stamp_id", "position_plan_gu", "yaw_deg",
                              "kit", "category", "door_intents"), path, issues)
            for key in ("lot_id", "stamp_id", "kit", "category"):
                if not isinstance(stamp.get(key), str) or not stamp.get(key):
                    issues.append(FormatIssue(f"{path}.{key}", "invalid_id",
                                              "must be a non-empty string"))
            _check_point(stamp, "position_plan_gu", path, issues)
            if not _finite(stamp.get("yaw_deg")):
                issues.append(FormatIssue(f"{path}.yaw_deg", "not_finite",
                                          "yaw_deg must be finite"))
            for key in ("show_source_terrain", "show_burial_envelope",
                        "intentional_slope_capable"):
                if key in stamp and not isinstance(stamp[key], bool):
                    issues.append(FormatIssue(f"{path}.{key}", "wrong_type",
                                              f"{key} must be boolean"))
            if "anchor_z_gu" in stamp and not _finite(stamp["anchor_z_gu"]):
                issues.append(FormatIssue(f"{path}.anchor_z_gu", "not_finite",
                                          "anchor_z_gu must be finite"))
            terrain_evidence = stamp.get("terrain_evidence")
            if terrain_evidence is not None:
                if not isinstance(terrain_evidence, dict):
                    issues.append(FormatIssue(f"{path}.terrain_evidence", "wrong_type",
                                              "terrain_evidence must be an object"))
                else:
                    allowed_evidence = {"observed_relief_gu", "observed_burial_depth_gu",
                                        "observed_slope_deg", "access_assembly_bounds_plan_gu",
                                        "reason"}
                    _unknown(terrain_evidence, allowed_evidence,
                             f"{path}.terrain_evidence", issues)
                    for key in ("observed_relief_gu", "observed_burial_depth_gu",
                                "observed_slope_deg"):
                        if key in terrain_evidence and not _finite(terrain_evidence[key]):
                            issues.append(FormatIssue(f"{path}.terrain_evidence.{key}",
                                                      "not_finite", "measurement must be finite"))
                    if "access_assembly_bounds_plan_gu" in terrain_evidence and not _points(
                            terrain_evidence["access_assembly_bounds_plan_gu"], 2):
                        issues.append(FormatIssue(
                            f"{path}.terrain_evidence.access_assembly_bounds_plan_gu",
                            "invalid_bounds", "access assembly bounds must contain finite points"))
            intents = stamp.get("door_intents")
            if not isinstance(intents, list) or not intents:
                issues.append(FormatIssue(f"{path}.door_intents", "invalid_door_intents",
                                          "door_intents must be a non-empty array"))
            else:
                intent_allowed = {"door_id", "intent", "target_id", "reason"}
                for didx, intent in enumerate(intents):
                    ipath = f"{path}.door_intents[{didx}]"
                    if not isinstance(intent, dict):
                        issues.append(FormatIssue(ipath, "wrong_type", "door intent must be an object"))
                        continue
                    _unknown(intent, intent_allowed, ipath, issues)
                    _required(intent, ("door_id", "intent"), ipath, issues)
                    if not isinstance(intent.get("door_id"), str) or not intent.get("door_id"):
                        issues.append(FormatIssue(f"{ipath}.door_id", "invalid_id",
                                                  "door_id must be a non-empty source door id"))
                    if intent.get("intent") not in ("public", "service", "private", "unused"):
                        issues.append(FormatIssue(f"{ipath}.intent", "invalid_intent",
                                                  "intent must be public, service, private, or unused"))
            door_targets = stamp.get("door_targets")
            if door_targets is not None:
                if not isinstance(door_targets, list):
                    issues.append(FormatIssue(f"{path}.door_targets", "wrong_type",
                                              "door_targets must be an array"))
                else:
                    target_allowed = {"door_id", "target_id", "intent"}
                    for didx, target in enumerate(door_targets):
                        tpath = f"{path}.door_targets[{didx}]"
                        if not isinstance(target, dict):
                            issues.append(FormatIssue(tpath, "wrong_type",
                                                      "door target must be an object"))
                            continue
                        _unknown(target, target_allowed, tpath, issues)
                        _required(target, ("door_id", "target_id", "intent"), tpath, issues)
                        if not isinstance(target.get("door_id"), str) or not target.get("door_id"):
                            issues.append(FormatIssue(f"{tpath}.door_id", "invalid_id",
                                                      "door_id must be a non-empty source door id"))
                        if not isinstance(target.get("target_id"), str) or not target.get("target_id"):
                            issues.append(FormatIssue(f"{tpath}.target_id", "invalid_id",
                                                      "target_id must be a non-empty target id"))
                        if target.get("intent") not in ("public", "service"):
                            issues.append(FormatIssue(f"{tpath}.intent", "invalid_intent",
                                                      "intent must be public or service"))
            links = stamp.get("access_links", [])
            if not isinstance(links, list):
                issues.append(FormatIssue(f"{path}.access_links", "wrong_type",
                                          "access_links must be an array"))
            else:
                link_allowed = {"door_id", "target_id", "polyline_plan_gu", "notes"}
                for lidx, link in enumerate(links):
                    lpath = f"{path}.access_links[{lidx}]"
                    if not isinstance(link, dict):
                        issues.append(FormatIssue(lpath, "wrong_type", "access link must be an object"))
                        continue
                    _unknown(link, link_allowed, lpath, issues)
                    _required(link, ("door_id", "target_id", "polyline_plan_gu"), lpath, issues)
                    _check_polyline(link, "polyline_plan_gu", lpath, issues)

    render_options = document.get("render_options")
    if render_options is not None:
        if not isinstance(render_options, dict):
            issues.append(FormatIssue("$.render_options", "wrong_type",
                                      "render_options must be an object"))
        else:
            allowed = {"map_width_px", "map_height_px", "show_contours", "show_slope",
                       "show_context_inset", "show_source_terrain", "show_burial_envelope",
                       "legend_title", "selected_lot_id"}
            _unknown(render_options, allowed, "$.render_options", issues)
            for key in ("show_contours", "show_slope", "show_context_inset",
                        "show_source_terrain", "show_burial_envelope"):
                if key in render_options and not isinstance(render_options[key], bool):
                    issues.append(FormatIssue(f"$.render_options.{key}", "wrong_type",
                                              f"{key} must be boolean"))
            for key in ("map_width_px", "map_height_px"):
                if key in render_options and (not isinstance(render_options[key], int) or
                                              render_options[key] < 400):
                    issues.append(FormatIssue(f"$.render_options.{key}", "invalid_resolution",
                                              f"{key} must be an integer >= 400"))
            if "selected_lot_id" in render_options and (
                    not isinstance(render_options["selected_lot_id"], str) or
                    not render_options["selected_lot_id"].strip()):
                issues.append(FormatIssue("$.render_options.selected_lot_id", "invalid_id",
                                          "selected_lot_id must be a non-empty string"))

    return sorted(issues, key=lambda issue: (issue.path, issue.code, issue.message))


def require_valid_extension(document: Mapping[str, Any]) -> None:
    """Raise :class:`VisualPlanFormatError` with all structural failures."""

    issues = validate_extension(document)
    if issues:
        detail = "; ".join(f"{i.path}: {i.message}" for i in issues[:8])
        if len(issues) > 8:
            detail += f"; and {len(issues) - 8} more"
        raise VisualPlanFormatError(detail)


def extension_schema() -> dict[str, Any]:
    """Return a compact machine-readable schema for documentation/tooling.

    The semantic shape is intentionally documented here rather than injected
    into T1.1's schema.  Consumers should run this structural gate first and
    then the advisory analyser with the accepted site/stamp/road products.
    """

    point = {"type": "array", "prefixItems": [{"type": "number"}, {"type": "number"}],
             "items": False, "minItems": 2, "maxItems": 2}
    points = {"type": "array", "items": point, "minItems": 2}
    polygon = {"type": "array", "items": point, "minItems": 3}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "cityforge/visual_plan_extension_schema_v1.json",
        "title": "Cityforge visual planning extension v1",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "kind", "plan_id", "seed", "coordinate_frame",
                      "rectangle", "existing_source_roads", "authored_roads", "alleys",
                      "road_surface_polygons", "shared_courts", "stamps", "districts",
                      "annotations", "advisory_overrides"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "kind": {"const": KIND},
            "plan_id": {"type": "string", "minLength": 1},
            "base_t1_1_plan_id": {"type": "string"},
            "seed": {"type": "integer"},
            "coordinate_frame": {"const": COORDINATE_FRAME},
            "rectangle": {"type": "object", "additionalProperties": False,
                          "required": ["cell_bounds", "context_margin_gu"],
                          "properties": {
                              "cell_bounds": {"type": "array", "minItems": 4, "maxItems": 4,
                                              "items": {"type": "integer"}},
                              "context_margin_gu": {"type": "number", "minimum": 0},
                              "full_site_inset": {"type": "boolean"},
                              "world_bounds_gu": points,
                          }},
            "existing_source_roads": {"type": "array"},
            "authored_roads": {"type": "array"},
            "alleys": {"type": "array"},
            "road_surface_polygons": {"type": "array"},
            "shared_courts": {"type": "array"},
            "stamps": {"type": "array"},
            "districts": {"type": "array"},
            "annotations": {"type": "array"},
            "advisory_overrides": {"type": "array"},
             "render_options": {"type": "object", "additionalProperties": False,
                                "properties": {
                                    "map_width_px": {"type": "integer", "minimum": 400},
                                    "map_height_px": {"type": "integer", "minimum": 400},
                                    "show_contours": {"type": "boolean"},
                                    "show_slope": {"type": "boolean"},
                                    "show_context_inset": {"type": "boolean"},
                                    "show_source_terrain": {"type": "boolean"},
                                    "show_burial_envelope": {"type": "boolean"},
                                    "legend_title": {"type": "string"},
                                    "selected_lot_id": {"type": "string", "minLength": 1},
                                }},
            "design_notes": {"type": "string"},
        },
        "definitions": {"point": point, "points": points, "polygon": polygon},
    }


__all__ = [
    "COORDINATE_FRAME", "FormatIssue", "KIND", "SCHEMA_VERSION",
    "VisualPlanFormatError", "canonical_json_bytes", "extension_schema",
    "require_valid_extension", "validate_extension",
]
