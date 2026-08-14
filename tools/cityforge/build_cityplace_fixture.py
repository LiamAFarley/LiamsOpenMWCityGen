"""Build and prove the synthetic Cityforge T1.2 placement fixture.

Pipeline position
------------------
This proof builder is a deterministic harness around the production T1.1
validator and T1.2 solver.  It copies no source plugin and never authors a real
Falkreath design.  It derives a clearly labelled synthetic plan from the
accepted T1.1 synthetic template, validates it, runs the planned pass, creates
one synthetic final-field copy for the pad re-seat proof, and runs the final
pass into a separate subdirectory.

Outputs are restricted to
``output/cityforge/phase1/t1_2_placement_fixture/``: planned T1.2 products,
final-reseat products, a proof-case ledger, deterministic numerical audit, and
a top-down placement diagnostic.  The diagnostic contains no building mesh or
city render; it draws only lot hull/outcome symbols on a labelled plan frame.

The builder also exercises structured rejected cases (slope/relief, water,
invalid cases are proof inputs, not additions to the zero-error T1.1 plan.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
from numpy.lib import format as np_format

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen import cityplan, cityplace  # noqa: E402
from procgen.cityplace_contracts import PlacementConfig, TerrainField, load_json, sha256_file  # noqa: E402
from procgen.cityplace_output import build_manifest, write_products  # noqa: E402
from procgen.censusio import deterministic_dumps, write_deterministic  # noqa: E402


OUT_REL = Path("output/cityforge/phase1/t1_2_placement_fixture")
T1_1_TEMPLATE = Path(
    "output/cityforge/phase1/t1_1_validation_fixture/"
    "synthetic_not_a_falkreath_design.city_plan.json"
)
SURVEY = Path("output/cityforge/sites/falkreath_v1/site_survey.json")
BRIEF = Path("output/cityforge/briefs/falkreath_v1/kit_brief.json")
PALETTE = Path("output/cityforge/briefs/falkreath_v1/region_palette.json")
LIBRARIES = [
    Path("output/cityforge/stamps/karthgad_nord_v1.json"),
    Path("output/cityforge/stamps/markarth_side_stone_v1.json"),
]
CENTERLINES = Path(
    "output/mapdata/roads/tamriel_aligned_centerlines_v1/"
    "tamriel_aligned_centerlines_v1.json"
)
FIELD = Path("output/cityforge/sites/falkreath_v1/survey_fields.npz")

FARM_ID = "markarth_side_v1__u18_halgir_s_farm_wylc_s_hut"


def _human_report(result: Mapping[str, Any], plan_path: Path) -> str:
    """Write the small deterministic validation report needed by the fixture."""

    lines = [
        "Cityforge D-PLAN validation (T1.1)",
        f"plan: {plan_path}",
        f"plan_id: {result.get('plan_id')}",
        f"result: {'VALID' if result.get('valid') else 'INVALID'}",
        f"issues: {result.get('issue_count')} (errors {result.get('error_count')}, warnings {result.get('warning_count')})",
    ]
    for issue in result.get("issues", []):
        lines.append(f"  [{issue['severity']}] {issue['code']} {issue['path']}: {issue['message']}")
    lines.append("")
    lines.append("input hashes (sha256):")
    for name, digest in sorted(result.get("input_hashes", {}).items()):
        lines.append(f"  {name}: {digest}")
    return "\n".join(lines) + "\n"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _road_for_lot(
    lot_id: str,
    position: tuple[float, float],
    yaw_deg: float,
    stamp: Mapping[str, Any],
    *,
    exit_id: str,
) -> dict[str, Any]:
    """Create a synthetic access line outside the building's door direction."""

    heading = float(stamp["access_heading_rad"]) + math.radians(float(yaw_deg))
    ux, uy = math.cos(heading), math.sin(heading)
    # The line is perpendicular to the door-to-building axis and 1,800 GU away.
    center_x, center_y = position[0] + ux * 1800.0, position[1] + uy * 1800.0
    px, py = -uy, ux
    length = 1200.0
    points = []
    for sign in (-1.0, 1.0):
        x = min(max(center_x + sign * px * length, 0.0), 57343.0)
        y = min(max(center_y + sign * py * length, 0.0), 57343.0)
        points.append([round(x, 6), round(y, 6)])
    return {
        "road_id": f"proof_road_{lot_id}",
        "class": "street",
        "connects": [exit_id],
        "grade_policy": "conform",
        "polyline": points,
        "surface": "road",
        "width_gu": 256.0,
    }


def _lot(
    lot_id: str,
    *,
    x: float,
    y: float,
    yaw: float,
    stamp: Mapping[str, Any],
    explicit: bool,
    terrain_mode: str = "conform",
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "building_type": stamp["building_type"],
        "size_class": stamp["size_class"],
    }
    if explicit:
        request["stamp_id"] = stamp["stamp_id"]
    return {
        "lot_id": lot_id,
        "district": "core",
        "position": [float(x), float(y)],
        "yaw_deg": float(yaw),
        "request": request,
        "terrain_policy": {
            "mode": terrain_mode,
            "max_cut_fill_gu": 400.0,
        },
        "access": {"face_road": f"proof_road_{lot_id}"},
        "notes": "synthetic_not_a_falkreath_design placement proof lot",
    }


def make_plan(bundle: cityplan.Bundle) -> dict[str, Any]:
    """Derive a valid synthetic plan with explicit and selector lots."""

    template = _json(T1_1_TEMPLATE)
    farm = bundle.stamp_geometry[FARM_ID]
    relief_stamp = bundle.stamp_geometry["karthgad_v1__door_094670"]
    lots = [
        _lot("farm_explicit_yaw0", x=4352.0, y=4352.0, yaw=0.0,
             stamp=farm, explicit=True),
        _lot("farm_selector_yaw37", x=16640.0, y=4352.0, yaw=37.0,
             stamp=farm, explicit=False),
        _lot("farm_explicit_yaw_minus45", x=24832.0, y=4352.0, yaw=-45.0,
             stamp=farm, explicit=True),
        _lot("farm_selector_yaw90", x=28928.0, y=4352.0, yaw=90.0,
             stamp=farm, explicit=False),
        _lot("farm_explicit_full_turn", x=33024.0, y=4352.0, yaw=360.0,
             stamp=farm, explicit=True),
        _lot("farm_explicit_far_yaw0", x=49408.0, y=4352.0, yaw=0.0,
             stamp=farm, explicit=True),
        _lot("shop_legal_pad", x=10496.0, y=2304.0, yaw=0.0,
             stamp=relief_stamp, explicit=True, terrain_mode="flatten_pad"),
    ]
    roads = [
        _road_for_lot(
            str(lot["lot_id"]), tuple(lot["position"]), float(lot["yaw_deg"]),
            relief_stamp if lot["lot_id"] == "shop_legal_pad" else farm,
            exit_id=sorted(bundle.map_exits)[0],
        )
        for lot in lots
    ]
    template.update({
        "plan_id": "synthetic_not_a_falkreath_design_t1_2_v1",
        "design_notes": (
            "SYNTHETIC VALIDATION FIXTURE - NOT A FALKREATH DESIGN. "
            "T1.2 houses-only placement proof: explicit and shared-selector "
            "farm/tavern stamps, exact yaws, terrain pad, and structured rejects."
        ),
        "districts": [{
            "district_id": "core",
            "kind": "core",
            "polygon": [[0.0, 0.0], [57343.0, 0.0], [57343.0, 57343.0], [0.0, 57343.0]],
            "texture_zone": "dirt_core",
        }],
        "roads": roads,
        "lots": lots,
        "boundaries": [],
        "features": [],
        "terrain_edits": [],
        "texture_zones": template.get("texture_zones", []),
        "wilderness_hints": [],
    })
    return template


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a timestamp-free NPZ so final synthetic fields hash identically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            stream = io.BytesIO()
            np_format.write_array(stream, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _make_final_field(
    output_dir: Path,
    *,
    base_field: TerrainField,
    pad_request: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Create a synthetic flattened copy solely for the final-reseat proof."""

    values = np.array(base_field.values_gu, dtype=np.float64, copy=True)
    polygon = pad_request["pad_polygon"]
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    target = float(pad_request["target_height_gu"])
    spacing_x, spacing_y = base_field.spacing_gu
    # The extra half-cell border makes bilinear samples at hull vertices flat;
    # it is still a synthetic copy, never a source or T1.3 terrain edit.
    for iy in range(values.shape[0]):
        y = iy * spacing_y
        if y < min(ys) - spacing_y or y > max(ys) + spacing_y:
            continue
        for ix in range(values.shape[1]):
            x = ix * spacing_x
            if min(xs) - spacing_x <= x <= max(xs) + spacing_x:
                values[iy, ix] = target
    field_path = output_dir / "synthetic_final_fields.npz"
    _write_deterministic_npz(field_path, {"height_gu": values})
    metadata = {
        "schema_version": 1,
        "frame_origin_gu": list(base_field.origin_gu),
        "spacing_gu": list(base_field.spacing_gu),
        "shape": list(values.shape),
        "units": "game_units",
        "pass": "final",
        "provenance": "synthetic_not_a_falkreath_design final-reseat proof only",
        "planned_source_field_sha256": base_field.source_sha256,
    }
    metadata_path = output_dir / "synthetic_final_fields.metadata.json"
    write_deterministic(metadata_path, metadata)
    return field_path, metadata_path


def _case_record(
    case_id: str,
    expected_code: str,
    *,
    status: str,
    actual_codes: list[str],
    evidence: Mapping[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    result = {
        "case_id": case_id,
        "status": status,
        "expected_code": expected_code,
        "actual_codes": sorted(set(actual_codes)),
        "pass": expected_code in actual_codes and status == "rejected",
        "notes": notes,
    }
    if evidence is not None:
        result["measured"] = {
            "checks": evidence.get("checks", {}),
            "issues": evidence.get("issues", []),
        }
    return result


def build_proof_cases(
    *,
    bundle: cityplan.Bundle,
    plan: Mapping[str, Any],
    validation: Mapping[str, Any],
    field: TerrainField,
    workspace_root: Path,
    accepted_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run invalid diagnostic lots through the same production checks."""

    run = cityplace.SolverRun(
        plan=plan,
        plan_path=T1_1_TEMPLATE,
        validation=validation,
        bundle=bundle,
        field=field,
        terrain_pass="planned",
        workspace_root=workspace_root,
        config=PlacementConfig(),
    )
    farm = bundle.stamp_geometry[FARM_ID]
    relief_stamp = bundle.stamp_geometry["karthgad_v1__door_094670"]
    records: list[dict[str, Any]] = []
    # Terrain conform at the known pad lot deliberately rejects its measured
    # source-envelope violation instead of requesting a pad.
    run.plan = {"roads": [_road_for_lot("slope_relief", (4352.0, 2304.0), 0.0, relief_stamp,
                                         exit_id=sorted(bundle.map_exits)[0])]}
    lot = _lot("slope_relief", x=4352.0, y=2304.0, yaw=0.0, stamp=relief_stamp, explicit=True)
    try:
        status, evidence, _, _ = run._evaluate_lot(lot, stamp=relief_stamp, mode="explicit", existing=[])
        records.append(_case_record("slope_relief", "terrain_relief_exceeded", status=status,
                                    actual_codes=[i["code"] for i in evidence["issues"]], evidence=evidence))
    except Exception as exc:
        records.append(_case_record("slope_relief", "terrain_relief_exceeded", status="error",
                                    actual_codes=["case_execution_failed"], notes=str(exc)))

    # First surveyed water tile; the case is intentionally not a dock.
    water_tile = next(
        ( (tx, ty) for ty in range(112) for tx in range(112) if bundle.tile_water(tx, ty) ),
        None,
    )
    if water_tile is None:
        raise cityplace.CityPlaceError("synthetic water rejection case has no surveyed water tile")
    water_xy = (water_tile[0] * 512.0 + 256.0, water_tile[1] * 512.0 + 256.0)
    run.plan = {"roads": [_road_for_lot("water", water_xy, 0.0, farm, exit_id=sorted(bundle.map_exits)[0])]}
    lot = _lot("water", x=water_xy[0], y=water_xy[1], yaw=0.0, stamp=farm, explicit=True)
    try:
        status, evidence, _, _ = run._evaluate_lot(lot, stamp=farm, mode="explicit", existing=[])
        records.append(_case_record("water", "non_dock_water", status=status,
                                    actual_codes=[i["code"] for i in evidence["issues"]], evidence=evidence))
    except Exception as exc:
        records.append(_case_record("water", "non_dock_water", status="error",
                                    actual_codes=["case_execution_failed"], notes=str(exc)))

    out_lot = _lot("out_of_scope", x=-100.0, y=100.0, yaw=0.0, stamp=farm, explicit=True)
    out_evidence = cityplace.preflight_fixture_rejection(out_lot, bundle=bundle)
    if out_evidence is None:
        raise cityplace.CityPlaceError("out-of-scope proof preflight did not reject")
    records.append(_case_record(
        "out_of_scope", "footprint_out_of_scope", status=str(out_evidence["status"]),
        actual_codes=[i["code"] for i in out_evidence["issues"]], evidence=out_evidence,
        notes="invalid diagnostic anchor is outside the accepted [0,57344) plan frame; no terrain sample is claimed",
    ))

    # Collision against the first accepted placement's exact hull.
    first = next(item for item in accepted_evidence["city_placement"]["placements"])
    collision_xy = tuple(first["position_plan_gu"])
    run.plan = {"roads": [_road_for_lot("collision", collision_xy, 0.0, farm, exit_id=sorted(bundle.map_exits)[0])]}
    lot = _lot("collision", x=collision_xy[0], y=collision_xy[1], yaw=0.0, stamp=farm, explicit=True)
    try:
        status, evidence, _, _ = run._evaluate_lot(
            lot, stamp=farm, mode="explicit",
            existing=[(str(first["lot_id"]), [tuple(point) for point in first["footprint_hull_xy_plan_gu"]])],
        )
        records.append(_case_record("collision", "footprint_collision", status=status,
                                    actual_codes=[i["code"] for i in evidence["issues"]], evidence=evidence))
    except Exception as exc:
        records.append(_case_record("collision", "footprint_collision", status="error",
                                    actual_codes=["case_execution_failed"], notes=str(exc)))

    no_stamp_lot = {
        "lot_id": "no_stamp",
        "position": [4352.0, 4352.0],
        "yaw_deg": 0.0,
        "request": {"building_type": "lodge", "size_class": "medium"},
    }
    no_stamp_evidence = cityplace.preflight_fixture_rejection(no_stamp_lot, bundle=bundle)
    if no_stamp_evidence is None:
        raise cityplace.CityPlaceError("no-stamp proof preflight did not reject")
    records.append(_case_record(
        "no_stamp", "no_compatible_stamp", status=str(no_stamp_evidence["status"]),
        actual_codes=[i["code"] for i in no_stamp_evidence["issues"]], evidence=no_stamp_evidence,
        notes="lodge capability gap is resolved independently and never silently replaced by a manor",
    ))

    far_xy = (4352.0, 4352.0)
    run.plan = {"roads": [{
        "road_id": "far_road", "class": "street", "width_gu": 256.0,
        "polyline": [[0.0, 0.0], [0.0, 1024.0]],
    }]}
    lot = _lot("road_distance", x=far_xy[0], y=far_xy[1], yaw=0.0, stamp=farm, explicit=True)
    lot["access"] = {"face_road": "far_road"}
    try:
        status, evidence, _, _ = run._evaluate_lot(lot, stamp=farm, mode="explicit", existing=[])
        records.append(_case_record("road_distance", "road_distance_exceeded", status=status,
                                    actual_codes=[i["code"] for i in evidence["issues"]], evidence=evidence))
    except Exception as exc:
        records.append(_case_record("road_distance", "road_distance_exceeded", status="error",
                                    actual_codes=["case_execution_failed"], notes=str(exc)))

    if not all(record["pass"] for record in records):
        raise cityplace.CityPlaceError(
            "synthetic proof case failure: " + json.dumps(records, sort_keys=True)
        )
    return records


def draw_diagnostic(output_dir: Path, result: Mapping[str, Any], cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Draw and inspect a deterministic outcome map, not a city render."""

    size = 1024
    image = Image.new("RGB", (size, size), (18, 22, 28))
    draw = ImageDraw.Draw(image)
    scale = (size - 80.0) / 57344.0
    def px(point: Sequence[float]) -> tuple[int, int]:
        return (int(round(40.0 + float(point[0]) * scale)),
                int(round(size - 40.0 - float(point[1]) * scale)))
    draw.rectangle((40, 40, size - 40, size - 40), outline=(140, 150, 165), width=2)
    draw.text((48, 12), "SYNTHETIC_NOT_A_FALKREATH_DESIGN — T1.2 PLACEMENT DIAGNOSTIC", fill=(255, 220, 120))
    colors = {"accepted": (55, 220, 120), "provisional": (255, 170, 40), "rejected": (240, 70, 70)}
    rows = []
    for item in result["city_placement"]["placements"]:
        rows.append((item, "accepted"))
    for item in result["city_placement"]["provisional_pad_lots"]:
        rows.append((item, "provisional"))
    for item in result["city_placement"]["rejected_lots"]:
        rows.append((item, "rejected"))
    hull_hits = 0
    for label_index, (item, status) in enumerate(
        sorted(rows, key=lambda pair: str(pair[0].get("lot_id")))
    ):
        points = [px(point) for point in item.get("footprint_hull_xy_plan_gu", [])]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=colors[status], width=2)
            hull_hits += len(points)
        anchor = px(item.get("position_plan_gu", [0.0, 0.0]))
        draw.ellipse((anchor[0] - 4, anchor[1] - 4, anchor[0] + 4, anchor[1] + 4), fill=colors[status])
        # Four deterministic callout lanes keep the synthetic proof legible
        # when several lots intentionally share a diagnostic row.
        label_point = (anchor[0] + 8, anchor[1] - 8 - (label_index % 4) * 16)
        draw.line((anchor[0], anchor[1], label_point[0], label_point[1] + 6), fill=colors[status], width=1)
        draw.text(label_point, str(item.get("lot_id")), fill=colors[status])
    image_path = output_dir / "diagnostic_topdown.png"
    image.save(image_path, format="PNG", optimize=False)
    reopened = Image.open(image_path)
    pixels = np.asarray(reopened.convert("RGB"))
    color_hits = {name: int(np.count_nonzero(np.all(pixels == color, axis=2))) for name, color in colors.items()}
    audit = {
        "label": "synthetic_not_a_falkreath_design",
        "kind": "top_down_placement_outcome_diagnostic_not_city_render",
        "resolution": [size, size],
        "source_product": "city_placement.json",
        "hull_vertex_draw_hits": hull_hits,
        "outcome_color_pixel_hits": color_hits,
        "case_count": len(cases),
        "image_sha256": sha256_file(image_path),
    }
    if sum(color_hits.values()) <= 0 or hull_hits <= 0:
        raise cityplace.CityPlaceError("synthetic diagnostic inspection found no outcome geometry")
    write_deterministic(output_dir / "diagnostic_topdown_audit.json", audit)
    return audit


def run_fixture(root: Path) -> dict[str, Any]:
    """Build all synthetic products and return measured audit data."""

    output_dir = root / OUT_REL
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = cityplan.Bundle.from_paths(
        site_survey=root / SURVEY,
        kit_brief=root / BRIEF,
        region_palette=root / PALETTE,
        stamp_libraries=[root / path for path in LIBRARIES],
        centerlines=root / CENTERLINES,
    )
    plan = make_plan(bundle)
    plan_path = output_dir / "synthetic_not_a_falkreath_design.city_plan.json"
    write_deterministic(plan_path, plan)
    validation = cityplan.validate_plan_file(plan_path, bundle)
    validation_path = output_dir / "synthetic_not_a_falkreath_design.validation.json"
    write_deterministic(validation_path, validation)
    (output_dir / "synthetic_not_a_falkreath_design.validation_report.txt").write_text(
        _human_report(validation, plan_path), encoding="utf-8"
    )
    if not validation["valid"] or validation["error_count"] != 0:
        raise cityplace.CityPlaceError(
            f"synthetic T1.1 plan is not zero-error: {validation['error_count']} errors"
        )

    planned_result = cityplace.solve_city_plan(
        plan_path=plan_path,
        validation_path=validation_path,
        site_survey_path=root / SURVEY,
        kit_brief_path=root / BRIEF,
        region_palette_path=root / PALETTE,
        stamp_library_paths=[root / path for path in LIBRARIES],
        centerlines_path=root / CENTERLINES,
        terrain_field_path=root / FIELD,
        terrain_pass="planned",
        workspace_root=root,
    )
    base_field = TerrainField.from_npz(root / FIELD, survey=bundle.site_survey, field_pass="planned")
    cases = build_proof_cases(
        bundle=bundle, plan=plan, validation=validation, field=base_field,
        workspace_root=root, accepted_evidence=planned_result,
    )
    planned_result["solver_report"]["synthetic_fixture"] = {
        "label": "synthetic_not_a_falkreath_design",
        "case_outcomes": cases,
        "case_outcome_counts": {
            "total": len(cases),
            "passed": sum(1 for case in cases if case["pass"]),
            "rejected": sum(1 for case in cases if case["status"] == "rejected"),
        },
    }
    source_hashes = dict(planned_result["source_hashes"])
    source_hashes["t1_1_validation"] = sha256_file(validation_path)
    output_hashes = write_products(
        output_dir,
        city_placement=planned_result["city_placement"],
        land_edit_requests=planned_result["land_edit_requests"],
        solver_report=planned_result["solver_report"],
        source_hashes=source_hashes,
    )
    write_deterministic(output_dir / "proof_cases.json", {
        "label": "synthetic_not_a_falkreath_design",
        "cases": cases,
        "counts": {"total": len(cases), "passed": len(cases)},
    })
    diagnostic = draw_diagnostic(output_dir, planned_result, cases)
    output_hashes.update({
        name: sha256_file(output_dir / name)
        for name in (
            "synthetic_not_a_falkreath_design.city_plan.json",
            "synthetic_not_a_falkreath_design.validation.json",
            "synthetic_not_a_falkreath_design.validation_report.txt",
            "proof_cases.json",
            "diagnostic_topdown.png",
            "diagnostic_topdown_audit.json",
        )
    })
    planned_identity = cityplace.result_identity(planned_result)
    manifest = build_manifest(
        source_hashes=source_hashes, output_hashes=output_hashes,
        plan_id=plan["plan_id"], terrain_pass="planned", deterministic_identity=planned_identity,
    )
    write_deterministic(output_dir / "manifest.json", manifest)

    if not planned_result["land_edit_requests"]["requests"]:
        raise cityplace.CityPlaceError("synthetic fixture did not produce the required legal provisional pad request")
    final_field_path, final_metadata_path = _make_final_field(
        output_dir, base_field=base_field, pad_request=planned_result["land_edit_requests"]["requests"][0]
    )
    final_dir = output_dir / "final_reseat"
    final_result = cityplace.solve_city_plan(
        plan_path=plan_path,
        validation_path=validation_path,
        site_survey_path=root / SURVEY,
        kit_brief_path=root / BRIEF,
        region_palette_path=root / PALETTE,
        stamp_library_paths=[root / path for path in LIBRARIES],
        centerlines_path=root / CENTERLINES,
        terrain_field_path=final_field_path,
        terrain_metadata_path=final_metadata_path,
        terrain_pass="final",
        planned_placement_path=output_dir / "city_placement.json",
        workspace_root=root,
    )
    final_hashes = dict(final_result["source_hashes"])
    final_hashes["t1_1_validation"] = sha256_file(validation_path)
    final_output_hashes = write_products(
        final_dir,
        city_placement=final_result["city_placement"],
        land_edit_requests=final_result["land_edit_requests"],
        solver_report=final_result["solver_report"],
        source_hashes=final_hashes,
    )
    final_identity = cityplace.result_identity(final_result)
    final_output_hashes.update({
        name: sha256_file(final_dir / name)
        for name in ("city_placement.json", "land_edit_requests.json", "solver_report.json")
    })
    write_deterministic(final_dir / "manifest.json", build_manifest(
        source_hashes=final_hashes, output_hashes=final_output_hashes,
        plan_id=plan["plan_id"], terrain_pass="final", deterministic_identity=final_identity,
    ))
    return {
        "planned": planned_result,
        "final": final_result,
        "cases": cases,
        "diagnostic": diagnostic,
        "planned_identity": planned_identity,
        "final_identity": final_identity,
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the synthetic T1.2 Cityforge proof fixture")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args(argv)
    try:
        result = run_fixture(Path(args.workspace_root).resolve())
        print(
            f"cityplace fixture PASS accepted={result['planned']['city_placement']['counts']['accepted']} "
            f"provisional={result['planned']['city_placement']['counts']['provisional']} "
            f"rejected_cases={len(result['cases'])} "
            f"source_members={result['planned']['solver_report']['gates']['source_replay']['members_checked']} "
            f"oracle37={result['planned']['solver_report']['gates']['multi_axis_oracle_37deg']['checked_members']} "
            f"planned_identity={result['planned_identity']} final_identity={result['final_identity']}"
        )
        return 0
    except Exception as exc:
        print(f"FAILURE: cityplace {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
