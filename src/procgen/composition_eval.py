"""Pure deterministic evaluation of authored Cityforge composition intent.

Purpose
-------
Measure the narrow v1 road and lot-group relationships that an authored intent
declares.  This module is observational: it never projects geometry, selects a
candidate, mutates a domain, or invents a road, lot, target, or stamp.

Inputs
------
``evaluate_composition`` consumes normalized intent data and the selected
world-GU fact rows materialized by the frontage-fit stage.  Each fact row must
use the exact contract documented by :func:`evaluate_composition` below.

Outputs
-------
The evaluator returns a canonical JSON-ready mapping with ``roads``, ``groups``,
and ``findings`` arrays.  Metrics and findings are sorted by authored ids and
contain only finite numbers, strings, booleans, ``None``, lists, and mappings.
The preference helper returns only the four fixed, deterministic v1 preference
components; it does not score marker displacement or assignment identity.

Invariants
----------
Facts are immutable inputs; every selected lot occurs at most once; target
lengths for one target agree within ``EPSILON_GU``; all authored group members
have exactly one selected fact; and every relationship measurement is computed
from supplied facts rather than inferred from centroid coordinates.

Pipeline position
------------------
This is the pure relational stage after frontage-fit candidate selection and
before reporting, rendering, or TES3 authoring.  It deliberately has no
project, renderer, terrain, filesystem, or search dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


EPSILON_GU = 1.0e-6
"""Comparison tolerance used by all v1 excess and consistency checks."""


class CompositionEvalError(ValueError):
    """Raised when intent or selected facts cannot be evaluated safely."""


_PURPOSES = frozenset(("urban_street", "service_lane", "connector"))
_CHARACTERS = frozenset((
    "compact_cluster",
    "irregular_two_sided",
    "formal_square",
    "gateway_cluster",
    "sparse_outskirts",
))
_COMPACT_PREFERENCE_CHARACTERS = frozenset((
    "compact_cluster",
    "formal_square",
    "gateway_cluster",
))
_FACT_KEYS = frozenset((
    "lot_id",
    "centroid",
    "intentional_outlier",
    "primary_target_id",
    "target_arc_gu",
    "target_length_gu",
    "frontage_side",
    "plaza_angle_deg",
))


def _error(path: str, message: str) -> CompositionEvalError:
    return CompositionEvalError(f"{path} {message}")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if not _finite_number(value):
        raise _error(path, "must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise _error(path, f"must be at least {minimum:g}")
    return result


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(path, "must be a non-empty string")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(path, "must be an array")
    return value


def _point(value: Any, path: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        raise _error(path, "must contain exactly two finite numbers")
    return (
        _number(value[0], f"{path}[0]"),
        _number(value[1], f"{path}[1]"),
    )


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _strictly_exceeds(value: float, limit: float) -> bool:
    return value - limit > EPSILON_GU


def _polyline_length(value: Any, path: str) -> float:
    points = _array(value, path)
    if len(points) < 2:
        raise _error(path, "must contain at least two points")
    parsed = [_point(point, f"{path}[{index}]") for index, point in enumerate(points)]
    return sum(_distance(a, b) for a, b in zip(parsed, parsed[1:]))


def _clamped_arc(row: Mapping[str, Any], target_lengths: Mapping[str, float]) -> float:
    length = target_lengths[row["primary_target_id"]]
    return max(0.0, min(float(row["target_arc_gu"]), length))


def _primary_target(lot: Mapping[str, Any], path: str) -> str | None:
    """Return a normalized lot's primary target when its frontage is present.

    Small unit fixtures may omit the non-composition parts of a normalized lot;
    in that case there is no cross-check to perform.  If a frontage declaration
    is present, however, a malformed or ambiguous primary is contradictory and
    is rejected rather than silently ignored.
    """

    if "frontages" not in lot:
        return None
    frontages = lot["frontages"]
    if not isinstance(frontages, list):
        raise _error(f"{path}.frontages", "must be an array")
    primary: list[str] = []
    for index, frontage in enumerate(frontages):
        if not isinstance(frontage, Mapping):
            raise _error(f"{path}.frontages[{index}]", "must be an object")
        if frontage.get("primary") is True:
            primary.append(_nonempty_string(
                frontage.get("target_id"),
                f"{path}.frontages[{index}].target_id",
            ))
    if len(primary) != 1:
        raise _error(path, "must declare exactly one primary frontage")
    return primary[0]


def _read_lots(intent: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if "lots" not in intent:
        return {}
    rows = _array(intent["lots"], "$.lots")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise _error(f"$.lots[{index}]", "must be an object")
        lot_id = _nonempty_string(row.get("id"), f"$.lots[{index}].id")
        if lot_id in result:
            raise _error(f"$.lots[{index}].id", f"duplicates lot {lot_id!r}")
        result[lot_id] = row
    return result


def _read_roads(intent: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "roads" not in intent:
        return []
    rows = _array(intent["roads"], "$.roads")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        path = f"$.roads[{index}]"
        if not isinstance(row, Mapping):
            raise _error(path, "must be an object")
        road_id = _nonempty_string(row.get("id"), f"{path}.id")
        if road_id in seen:
            raise _error(f"{path}.id", f"duplicates road {road_id!r}")
        seen.add(road_id)
        if "purpose" not in row:
            continue
        purpose = row["purpose"]
        if not isinstance(purpose, str) or purpose not in _PURPOSES:
            raise _error(f"{path}.purpose", "is not a supported composition purpose")
        maximum: float | None = None
        if "max_unsupported_frontage_gu" in row:
            if purpose not in ("urban_street", "service_lane"):
                raise _error(
                    f"{path}.max_unsupported_frontage_gu",
                    "requires purpose 'urban_street' or 'service_lane'",
                )
            maximum = _number(
                row["max_unsupported_frontage_gu"],
                f"{path}.max_unsupported_frontage_gu",
                minimum=0.0,
            )
        if "points" not in row:
            raise _error(f"{path}.points", "is required for a composition road")
        result.append({
            "id": road_id,
            "purpose": purpose,
            "max_unsupported_frontage_gu": maximum,
            "points": row["points"],
        })
    result.sort(key=lambda row: row["id"])
    return result


def _read_group_sectors(value: Any, path: str, character: str, has_shared_target: bool) -> list[dict[str, Any]]:
    if character != "formal_square":
        raise _error(path, "requires character 'formal_square'")
    if not has_shared_target:
        raise _error(path, "requires shared_target_id")
    sectors = _array(value, path)
    if not sectors:
        raise _error(path, "must be non-empty")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, sector in enumerate(sectors):
        sector_path = f"{path}[{index}]"
        if not isinstance(sector, Mapping):
            raise _error(sector_path, "must be an object")
        sector_id = _nonempty_string(sector.get("id"), f"{sector_path}.id")
        if sector_id in seen:
            raise _error(f"{sector_path}.id", f"duplicates sector {sector_id!r}")
        seen.add(sector_id)
        start = _number(sector.get("start_deg"), f"{sector_path}.start_deg")
        end = _number(sector.get("end_deg"), f"{sector_path}.end_deg")
        if not (0.0 <= start < end <= 360.0):
            raise _error(sector_path, "must satisfy 0 <= start_deg < end_deg <= 360")
        result.append({"id": sector_id, "start_deg": start, "end_deg": end})
    result.sort(key=lambda row: row["id"])
    return result


def _read_groups(intent: Mapping[str, Any], lot_ids: set[str]) -> list[dict[str, Any]]:
    if "lot_groups" not in intent:
        return []
    rows = _array(intent["lot_groups"], "$.lot_groups")
    result: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    assigned: set[str] = set()
    for index, row in enumerate(rows):
        path = f"$.lot_groups[{index}]"
        if not isinstance(row, Mapping):
            raise _error(path, "must be an object")
        group_id = _nonempty_string(row.get("id"), f"{path}.id")
        if group_id in seen_groups:
            raise _error(f"{path}.id", f"duplicates group {group_id!r}")
        seen_groups.add(group_id)
        character = row.get("character")
        if not isinstance(character, str) or character not in _CHARACTERS:
            raise _error(f"{path}.character", "is not a supported group character")
        raw_lot_ids = _array(row.get("lot_ids"), f"{path}.lot_ids")
        if not raw_lot_ids:
            raise _error(f"{path}.lot_ids", "must be non-empty")
        members: list[str] = []
        local: set[str] = set()
        for member_index, value in enumerate(raw_lot_ids):
            member = _nonempty_string(value, f"{path}.lot_ids[{member_index}]")
            if member in local:
                raise _error(f"{path}.lot_ids", f"duplicates lot {member!r}")
            if lot_ids and member not in lot_ids:
                raise _error(f"{path}.lot_ids", f"references undeclared lot {member!r}")
            if member in assigned:
                raise _error(f"{path}.lot_ids", f"overlaps another group on lot {member!r}")
            local.add(member)
            assigned.add(member)
            members.append(member)
        members.sort()

        shared_target: str | None = None
        if "shared_target_id" in row:
            shared_target = _nonempty_string(row["shared_target_id"], f"{path}.shared_target_id")

        bounds: dict[str, float | int | None] = {
            "max_span_gu": None,
            "max_consecutive_gap_gu": None,
            "max_non_outlier_distance_gu": None,
            "max_consecutive_same_side": None,
        }
        for key in (
            "max_span_gu",
            "max_consecutive_gap_gu",
            "max_non_outlier_distance_gu",
        ):
            if key in row:
                bounds[key] = _number(row[key], f"{path}.{key}", minimum=0.0)
        if "max_consecutive_same_side" in row:
            value = row["max_consecutive_same_side"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise _error(f"{path}.max_consecutive_same_side", "must be a positive integer")
            bounds["max_consecutive_same_side"] = value

        along_order: list[str] | None = None
        if "along_order" in row:
            raw_order = _array(row["along_order"], f"{path}.along_order")
            if not raw_order:
                raise _error(f"{path}.along_order", "must be non-empty")
            along_order = []
            seen_order: set[str] = set()
            for order_index, value in enumerate(raw_order):
                member = _nonempty_string(value, f"{path}.along_order[{order_index}]")
                if member in seen_order:
                    raise _error(f"{path}.along_order", f"duplicates lot {member!r}")
                seen_order.add(member)
                along_order.append(member)
            if set(along_order) != local:
                raise _error(f"{path}.along_order", "must be a permutation of lot_ids")

        sectors: list[dict[str, Any]] = []
        if "plaza_sectors" in row:
            sectors = _read_group_sectors(
                row["plaza_sectors"],
                f"{path}.plaza_sectors",
                character,
                shared_target is not None,
            )

        result.append({
            "id": group_id,
            "character": character,
            "lot_ids": members,
            "shared_target_id": shared_target,
            "max_span_gu": bounds["max_span_gu"],
            "max_consecutive_gap_gu": bounds["max_consecutive_gap_gu"],
            "max_non_outlier_distance_gu": bounds["max_non_outlier_distance_gu"],
            "max_consecutive_same_side": bounds["max_consecutive_same_side"],
            "along_order": along_order,
            "plaza_sectors": sectors,
        })
    result.sort(key=lambda row: row["id"])
    return result


def _read_facts(
    selected_facts: Sequence[Mapping[str, Any]],
    lots: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(selected_facts, (str, bytes, bytearray)) or not isinstance(selected_facts, Sequence):
        raise _error("selected_facts", "must be a sequence of objects")
    facts: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(selected_facts):
        path = f"$.selected_facts[{index}]"
        if not isinstance(raw, Mapping):
            raise _error(path, "must be an object")
        unknown = sorted(set(raw) - _FACT_KEYS)
        missing = sorted(_FACT_KEYS - set(raw))
        if unknown:
            raise _error(path, f"has unknown keys {unknown}")
        if missing:
            raise _error(path, f"is missing required keys {missing}")
        lot_id = _nonempty_string(raw["lot_id"], f"{path}.lot_id")
        if lot_id in facts:
            raise _error(f"{path}.lot_id", f"duplicates selected lot {lot_id!r}")
        if lots and lot_id not in lots:
            raise _error(f"{path}.lot_id", f"is not declared in intent lots")
        centroid = _point(raw["centroid"], f"{path}.centroid")
        outlier = raw["intentional_outlier"]
        if not isinstance(outlier, bool):
            raise _error(f"{path}.intentional_outlier", "must be boolean")
        target_id = _nonempty_string(raw["primary_target_id"], f"{path}.primary_target_id")
        target_arc = _number(
            raw["target_arc_gu"],
            f"{path}.target_arc_gu",
            minimum=0.0,
        )
        target_length = _number(
            raw["target_length_gu"],
            f"{path}.target_length_gu",
            minimum=0.0,
        )
        frontage_side = raw["frontage_side"]
        if frontage_side is not None and frontage_side not in ("left", "right"):
            raise _error(f"{path}.frontage_side", "must be 'left', 'right', or null")
        plaza_angle = raw["plaza_angle_deg"]
        if plaza_angle is not None:
            plaza_angle = _number(plaza_angle, f"{path}.plaza_angle_deg")
            if not 0.0 <= plaza_angle < 360.0:
                raise _error(f"{path}.plaza_angle_deg", "must be in [0, 360)")

        if lot_id in lots:
            lot = lots[lot_id]
            if "intentional_outlier" in lot:
                expected = lot["intentional_outlier"]
                if not isinstance(expected, bool):
                    raise _error(f"$.lots[{lot_id!r}].intentional_outlier", "must be boolean")
            else:
                expected = False
            if expected != outlier:
                raise _error(path, "intentional_outlier contradicts intent lot declaration")
            if "frontage_side" in lot and lot["frontage_side"] != frontage_side:
                raise _error(path, "frontage_side contradicts intent lot declaration")
            declared_target = _primary_target(lot, f"$.lots[{lot_id!r}]")
            if declared_target is not None and declared_target != target_id:
                raise _error(path, "primary_target_id contradicts intent frontage")

        facts[lot_id] = {
            "lot_id": lot_id,
            "centroid": [centroid[0], centroid[1]],
            "intentional_outlier": outlier,
            "primary_target_id": target_id,
            "target_arc_gu": target_arc,
            "target_length_gu": target_length,
            "frontage_side": frontage_side,
            "plaza_angle_deg": plaza_angle,
        }

    # A target has one physical length.  Use the lowest canonical value when
    # values differ only inside the permitted tolerance, so fact order cannot
    # change the emitted metric.
    by_target: dict[str, list[float]] = {}
    for fact in facts.values():
        by_target.setdefault(fact["primary_target_id"], []).append(fact["target_length_gu"])
    for target_id, values in by_target.items():
        if max(values) - min(values) > EPSILON_GU:
            raise _error(
                "$.selected_facts",
                f"target {target_id!r} has inconsistent target_length_gu values",
            )
    return facts


def _medoid(member_facts: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    best_id = ""
    best_sum = float("inf")
    for candidate in sorted(member_facts, key=lambda row: row["lot_id"]):
        total = sum(_distance(candidate["centroid"], other["centroid"]) for other in member_facts)
        if total < best_sum:
            best_id = candidate["lot_id"]
            best_sum = total
    return best_id, best_sum


def _unsupported_intervals(length: float, positions: Sequence[float]) -> tuple[list[list[float]], float]:
    if not positions:
        intervals = [[0.0, length]]
    else:
        intervals = [[0.0, positions[0]]]
        intervals.extend(
            [[previous, current] for previous, current in zip(positions, positions[1:])]
        )
        intervals.append([positions[-1], length])
    longest = max((interval[1] - interval[0] for interval in intervals), default=0.0)
    return intervals, longest


def _finding_sort_key(finding: Mapping[str, Any]) -> tuple[Any, ...]:
    if "road_id" in finding:
        scope = 0
        owner = finding["road_id"]
    else:
        scope = 1
        owner = finding.get("group_id", "")
    return (
        scope,
        str(owner),
        str(finding.get("code", "")),
        str(finding.get("sector_id", "")),
        tuple(str(value) for value in finding.get("lot_ids", [])),
    )


def evaluate_composition(
    intent: Mapping[str, Any],
    selected_facts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure normalized composition declarations against selected fact rows.

    The exact fact contract is eight required keys: ``lot_id`` (unique
    non-empty string), finite two-number ``centroid``, boolean
    ``intentional_outlier``, non-empty ``primary_target_id``, finite
    ``target_arc_gu``, finite non-negative ``target_length_gu``, nullable
    ``frontage_side`` (``left``/``right``), and nullable ``plaza_angle_deg`` in
    ``[0, 360)``.  Unknown or missing keys fail closed.  Group members must be
    represented exactly once; ungrouped selected lots are permitted for road
    measurements.

    ``target_arc_gu`` is already a canonical increasing-path projection.  The
    evaluator only clamps it to the supplied target length; it never projects a
    centroid or reconstructs a target from intent coordinates.
    """

    if not isinstance(intent, Mapping):
        raise _error("intent", "must be a mapping")
    lots = _read_lots(intent)
    roads = _read_roads(intent)
    groups = _read_groups(intent, set(lots))
    facts = _read_facts(selected_facts, lots)

    target_lengths: dict[str, float] = {}
    for fact in facts.values():
        target_id = fact["primary_target_id"]
        target_lengths[target_id] = min(
            target_lengths.get(target_id, fact["target_length_gu"]),
            fact["target_length_gu"],
        )

    findings: list[dict[str, Any]] = []
    road_metrics: list[dict[str, Any]] = []
    for road in roads:
        road_id = road["id"]
        serving = [fact for fact in facts.values() if fact["primary_target_id"] == road_id]
        serving.sort(key=lambda fact: fact["lot_id"])
        if serving:
            length = target_lengths[road_id]
        else:
            length = _polyline_length(road["points"], f"$.roads[{road_id!r}].points")
        positions = [
            {
                "lot_id": fact["lot_id"],
                "target_arc_gu": _clamped_arc(fact, target_lengths),
            }
            for fact in serving
        ]
        positions.sort(key=lambda row: (row["target_arc_gu"], row["lot_id"]))
        arcs = [row["target_arc_gu"] for row in positions]
        unsupported, longest = _unsupported_intervals(length, arcs)
        maximum = road["max_unsupported_frontage_gu"]
        excess: float | None = None
        if maximum is not None:
            excess = max(0.0, longest - maximum)
            if _strictly_exceeds(longest, maximum):
                findings.append({
                    "code": "road_unsupported_frontage_exceeded",
                    "road_id": road_id,
                    "longest_unsupported_interval_gu": longest,
                    "limit_gu": maximum,
                    "excess_gu": excess,
                })
        road_metrics.append({
            "id": road_id,
            "purpose": road["purpose"],
            "length_gu": length,
            "serving_lot_count": len(serving),
            "projected_frontage_positions": positions,
            "unsupported_intervals_gu": unsupported,
            "longest_unsupported_interval_gu": longest,
            "max_unsupported_frontage_gu": maximum,
            "unsupported_frontage_excess_gu": excess,
        })

    group_metrics: list[dict[str, Any]] = []
    for group in groups:
        group_id = group["id"]
        missing = [lot_id for lot_id in group["lot_ids"] if lot_id not in facts]
        if missing:
            raise _error(
                f"$.lot_groups[{group_id!r}].lot_ids",
                f"has no selected fact for lots {missing}",
            )
        member_facts = [facts[lot_id] for lot_id in group["lot_ids"]]
        member_facts.sort(key=lambda fact: fact["lot_id"])
        span = max(
            (_distance(left["centroid"], right["centroid"])
             for index, left in enumerate(member_facts)
             for right in member_facts[index + 1:]),
            default=0.0,
        )
        span_limit = group["max_span_gu"]
        span_excess: float | None = None
        if span_limit is not None:
            span_excess = max(0.0, span - span_limit)
            if _strictly_exceeds(span, span_limit):
                findings.append({
                    "code": "group_span_exceeded",
                    "group_id": group_id,
                    "measured_gu": span,
                    "limit_gu": span_limit,
                    "excess_gu": span_excess,
                })

        shared_target = group["shared_target_id"]
        if shared_target is not None:
            mismatched = sorted(
                fact["lot_id"]
                for fact in member_facts
                if fact["primary_target_id"] != shared_target
            )
            if mismatched:
                findings.append({
                    "code": "group_shared_target_mismatch",
                    "group_id": group_id,
                    "target_id": shared_target,
                    "lot_ids": mismatched,
                })

        actual_rows = [
            (fact, _clamped_arc(fact, target_lengths))
            for fact in member_facts
        ]
        actual_rows.sort(key=lambda row: (row[1], row[0]["lot_id"]))
        actual_order = [row[0]["lot_id"] for row in actual_rows]
        actual_arcs = [row[1] for row in actual_rows]
        all_same_target = len({fact["primary_target_id"] for fact in member_facts}) == 1
        can_measure_gaps = all_same_target and bool(actual_rows)
        gaps = (
            [current - previous for previous, current in zip(actual_arcs, actual_arcs[1:])]
            if can_measure_gaps else []
        )
        max_gap = max(gaps, default=0.0) if can_measure_gaps else None
        repeated_gap_pairs = (
            sum(
                1
                for previous, current in zip(gaps, gaps[1:])
                if abs(current - previous) <= EPSILON_GU
            )
            if len(member_facts) >= 3 and can_measure_gaps
            else 0
        )
        gap_limit = group["max_consecutive_gap_gu"]
        gap_excess: float | None = None
        if gap_limit is not None and max_gap is not None:
            gap_excess = max(0.0, max_gap - gap_limit)
            if _strictly_exceeds(max_gap, gap_limit):
                findings.append({
                    "code": "group_gap_exceeded",
                    "group_id": group_id,
                    "measured_gu": max_gap,
                    "limit_gu": gap_limit,
                    "excess_gu": gap_excess,
                })
        elif gap_limit is not None:
            target_facts = [
                {
                    "lot_id": fact["lot_id"],
                    "primary_target_id": fact["primary_target_id"],
                }
                for fact in member_facts
            ]
            target_facts.sort(key=lambda row: (row["primary_target_id"], row["lot_id"]))
            findings.append({
                "code": "group_gap_unmeasurable",
                "group_id": group_id,
                "lot_ids": sorted(fact["lot_id"] for fact in member_facts),
                "target_facts": target_facts,
            })

        authored_order = group["along_order"]
        order_violation = authored_order is not None and authored_order != actual_order
        if order_violation:
            findings.append({
                "code": "along_order_violation",
                "group_id": group_id,
                "expected_lot_ids": list(authored_order),
                "actual_lot_ids": actual_order,
            })
        side_order = authored_order if authored_order is not None else actual_order
        by_lot = {fact["lot_id"]: fact for fact in member_facts}
        current_side: str | None = None
        current_run = 0
        max_side_run = 0
        for lot_id in side_order:
            side = by_lot[lot_id]["frontage_side"]
            if side is None:
                current_side = None
                current_run = 0
            elif side == current_side:
                current_run += 1
            else:
                current_side = side
                current_run = 1
            max_side_run = max(max_side_run, current_run)
        side_limit = group["max_consecutive_same_side"]
        side_excess: int | None = None
        if side_limit is not None:
            side_excess = max(0, max_side_run - side_limit)
            if max_side_run - side_limit > 0:
                findings.append({
                    "code": "group_same_side_run_exceeded",
                    "group_id": group_id,
                    "measured": max_side_run,
                    "limit": side_limit,
                    "excess": side_excess,
                })

        occupancy: list[dict[str, Any]] = []
        for sector in group["plaza_sectors"]:
            occupants = sorted(
                fact["lot_id"]
                for fact in member_facts
                if shared_target is not None
                and fact["primary_target_id"] == shared_target
                and fact["plaza_angle_deg"] is not None
                and sector["start_deg"] <= fact["plaza_angle_deg"] < sector["end_deg"]
            )
            occupancy.append({"id": sector["id"], "lot_ids": occupants})
            if not occupants:
                findings.append({
                    "code": "plaza_sector_unoccupied",
                    "group_id": group_id,
                    "sector_id": sector["id"],
                })

        medoid_id, _ = _medoid(member_facts)
        medoid_fact = by_lot[medoid_id]
        non_outlier = [fact for fact in member_facts if not fact["intentional_outlier"]]
        non_outlier_max = max(
            (_distance(medoid_fact["centroid"], fact["centroid"]) for fact in non_outlier),
            default=0.0,
        )
        non_outlier_limit = group["max_non_outlier_distance_gu"]
        non_outlier_excess: float | None = None
        if non_outlier_limit is not None:
            non_outlier_excess = max(0.0, non_outlier_max - non_outlier_limit)
            if _strictly_exceeds(non_outlier_max, non_outlier_limit):
                findings.append({
                    "code": "group_non_outlier_distance_exceeded",
                    "group_id": group_id,
                    "measured_gu": non_outlier_max,
                    "limit_gu": non_outlier_limit,
                    "excess_gu": non_outlier_excess,
                })
        group_metrics.append({
            "id": group_id,
            "character": group["character"],
            "lot_ids": list(group["lot_ids"]),
            "shared_target_id": shared_target,
            "centroid_span_gu": span,
            "centroid_span_excess_gu": span_excess,
            "along_order_actual": actual_order,
            "along_gaps_gu": gaps,
            "max_consecutive_gap_gu_actual": max_gap,
            "repeated_consecutive_gap_pair_count": repeated_gap_pairs,
            "max_consecutive_gap_excess_gu": gap_excess,
            "along_order_violation": order_violation,
            "max_consecutive_same_side_actual": max_side_run,
            "max_consecutive_same_side_excess": side_excess,
            "plaza_sector_occupancy": occupancy,
            "medoid_lot_id": medoid_id,
            "non_outlier_max_distance_gu": non_outlier_max,
            "non_outlier_max_distance_excess_gu": non_outlier_excess,
        })

    findings.sort(key=_finding_sort_key)
    return {
        "roads": road_metrics,
        "groups": group_metrics,
        "findings": findings,
    }


def _composition_rows(
    composition: Mapping[str, Any],
    key: str,
    authored_ids: set[str],
) -> dict[str, Mapping[str, Any]]:
    """Read one evaluator output array and require exact authored-id coverage."""

    path = f"$.composition.{key}"
    if key not in composition:
        if authored_ids:
            raise _error(path, "is required for authored composition records")
        return {}
    rows = _array(composition[key], path)
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        row_path = f"{path}[{index}]"
        if not isinstance(row, Mapping):
            raise _error(row_path, "must be an object")
        row_id = _nonempty_string(row.get("id"), f"{row_path}.id")
        if row_id in result:
            raise _error(f"{row_path}.id", f"duplicates composition id {row_id!r}")
        result[row_id] = row

    missing = sorted(authored_ids - set(result))
    unknown = sorted(set(result) - authored_ids)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing authored ids {missing}")
        if unknown:
            details.append(f"unknown ids {unknown}")
        raise _error(path, "does not match authored composition ids (" + "; ".join(details) + ")")
    return result


def _preference_number(row: Mapping[str, Any], key: str, path: str) -> float:
    """Read a required non-negative evaluator metric canonically as a float."""

    value = _number(row.get(key), f"{path}.{key}", minimum=0.0)
    return 0.0 if value == 0.0 else value


def _preference_optional_number(
    row: Mapping[str, Any],
    key: str,
    path: str,
) -> float | None:
    """Read a nullable non-negative evaluator metric, preserving None as absent."""

    if key not in row:
        raise _error(f"{path}.{key}", "is required")
    if row[key] is None:
        return None
    return _preference_number(row, key, path)


def _preference_count(row: Mapping[str, Any], key: str, path: str) -> int:
    """Read a required JSON integer metric without accepting booleans or floats."""

    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _error(f"{path}.{key}", "must be a non-negative integer")
    return value


def composition_preference_components(
    intent: Mapping[str, Any],
    composition: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the fixed v1 preference components for an evaluated composition.

    The evaluator output is joined to authored roads and groups by id, never by
    array position.  Exact id coverage is required in both directions so a
    partial or foreign composition cannot silently improve an assignment.  The
    authored purpose/character controls inclusion; measured values come only
    from the supplied composition facts.
    """

    if not isinstance(intent, Mapping):
        raise _error("intent", "must be a mapping")
    if not isinstance(composition, Mapping):
        raise _error("composition", "must be a mapping")

    lots = _read_lots(intent)
    authored_roads = {
        row["id"]: row
        for row in _read_roads(intent)
    }
    authored_groups = {
        row["id"]: row
        for row in _read_groups(intent, set(lots))
    }
    roads = _composition_rows(composition, "roads", set(authored_roads))
    groups = _composition_rows(composition, "groups", set(authored_groups))

    urban_profile = [
        _preference_number(
            roads[road_id],
            "longest_unsupported_interval_gu",
            f"$.composition.roads[{road_id!r}]",
        )
        for road_id, authored in authored_roads.items()
        if authored["purpose"] in ("urban_street", "service_lane")
    ]
    urban_profile.sort(reverse=True)

    compact_span_profile: list[float] = []
    compact_gap_profile: list[float] = []
    irregular_repeated_gap_pairs = 0
    for group_id, authored in authored_groups.items():
        row = groups[group_id]
        path = f"$.composition.groups[{group_id!r}]"
        character = authored["character"]
        if character in _COMPACT_PREFERENCE_CHARACTERS:
            compact_span_profile.append(
                _preference_number(row, "centroid_span_gu", path)
            )
            gap = _preference_optional_number(
                row,
                "max_consecutive_gap_gu_actual",
                path,
            )
            if gap is not None:
                compact_gap_profile.append(gap)
        elif character == "irregular_two_sided":
            irregular_repeated_gap_pairs += _preference_count(
                row,
                "repeated_consecutive_gap_pair_count",
                path,
            )

    compact_span_profile.sort(reverse=True)
    compact_gap_profile.sort(reverse=True)
    return {
        "urban_unsupported_profile_gu": urban_profile,
        "compact_span_profile_gu": compact_span_profile,
        "compact_gap_profile_gu": compact_gap_profile,
        "irregular_repeated_gap_pairs": irregular_repeated_gap_pairs,
    }


__all__ = [
    "CompositionEvalError",
    "EPSILON_GU",
    "composition_preference_components",
    "evaluate_composition",
]
