"""Contracts and terrain-field sampling for the Cityforge T1.2 solver.

Pipeline position
------------------
This module is the small, dependency-light contract layer between the accepted
T1.1 plan validator and the houses-only placement engine.  It loads one dense
height field, verifies its frame/shape/unit/pass metadata, and exposes only
float64 bilinear samples to the placement and geometry stages.  It does not
select stamps, compose rotations, author TES3 records, or edit source terrain.

Inputs and outputs
------------------
``TerrainField.from_npz`` consumes the read-only ``survey_fields.npz`` (or a
T1.3 final-field copy) plus explicit field metadata.  The canonical T0.2 field
has its frame metadata in ``site_survey.json``; a synthetic/final field may
provide a sidecar JSON with the same contract.  Samples are returned in GU,
with a unit normal and slope in degrees.  ``PlacementConfig`` contains the
solver tolerances and is serialized into every deterministic solver report.

Invariants
----------
* The field is finite, two-dimensional, float64-compatible, and has at least
  a 2x2 bilinear cell.
* The frame origin, spacing, shape, and units are explicit and are checked
  against the accepted survey frame.  ``planned`` and ``final`` are distinct
  execution passes; a final run cannot be silently inferred from a planned
  field.
* Coordinates outside the field are rejected rather than clamped.  The last
  row/column is a valid exact sample, so boundary interpolation remains
  deterministic.
* No terrain value is synthesized when coverage is missing.  Callers receive
  ``FieldCoverageError`` and must reject the lot or stop the stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


WATER_LEVEL_GU = 0.0
WORLD_CELL_SIZE_GU = 8192.0
TILE_SIZE_GU = 512.0
FIELD_UNITS = "game_units"
FIELD_PASSES = ("planned", "final")


class CityPlaceInputError(ValueError):
    """Fatal input-contract error; the CLI reports this as ``FAILURE``."""


class FieldCoverageError(CityPlaceInputError):
    """A requested terrain sample lies outside the supplied field."""


def sha256_file(path: Path | str) -> str:
    """Hash one read-only input without loading it wholly into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path | str, label: str) -> dict[str, Any]:
    """Load a JSON object and fail closed on malformed/non-object input."""

    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CityPlaceInputError(f"cannot load {label} {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise CityPlaceInputError(f"{label} {target} is not a JSON object")
    return value


@dataclass(frozen=True)
class PlacementConfig:
    """Explicit T1.2 tolerances; none are hidden module-level placement rules.

    ``spacing_guidance`` is intentionally recorded but never used as a hard
    rejection threshold.  The only hard pairwise spacing rule is exact hull
    overlap/contact.
    """

    relief_tolerance_fraction: float = 0.50
    step_tolerance_fraction: float = 0.50
    step_zero_slack_gu: float = 1.0
    slope_slack_deg: float = 5.0
    burial_tolerance_fraction: float = 0.50
    burial_zero_slack_gu: float = 32.0
    bottom_clearance_min_gu: float = 0.0
    hard_road_distance_gu: float = 2500.0
    preferred_road_distance_gu: float = 1500.0
    hard_cross_slope_deg: float = 25.0
    pad_margin_gu: float = 256.0
    pad_falloff_gu: float = 512.0
    max_pad_cut_fill_gu: float = 400.0
    max_encoded_delta_gu: float = 1016.0
    contact_epsilon_gu: float = 0.25
    fine_collision_deferred: bool = True
    spacing_guidance_source: str = (
        "dispatch-5 inter_building_gap_gu; diagnostic only, not a minimum"
    )

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"placement config {name} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"placement config {name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Stable JSON form used by the solver report and manifest."""

        return {
            "relief_tolerance_fraction": float(self.relief_tolerance_fraction),
            "step_tolerance_fraction": float(self.step_tolerance_fraction),
            "step_zero_slack_gu": float(self.step_zero_slack_gu),
            "slope_slack_deg": float(self.slope_slack_deg),
            "burial_tolerance_fraction": float(self.burial_tolerance_fraction),
            "burial_zero_slack_gu": float(self.burial_zero_slack_gu),
            "bottom_clearance_min_gu": float(self.bottom_clearance_min_gu),
            "hard_road_distance_gu": float(self.hard_road_distance_gu),
            "preferred_road_distance_gu": float(self.preferred_road_distance_gu),
            "hard_cross_slope_deg": float(self.hard_cross_slope_deg),
            "pad_margin_gu": float(self.pad_margin_gu),
            "pad_falloff_gu": float(self.pad_falloff_gu),
            "max_pad_cut_fill_gu": float(self.max_pad_cut_fill_gu),
            "max_encoded_delta_gu": float(self.max_encoded_delta_gu),
            "contact_epsilon_gu": float(self.contact_epsilon_gu),
            "fine_collision_deferred": bool(self.fine_collision_deferred),
            "spacing_guidance_source": self.spacing_guidance_source,
        }


def _finite_pair(value: Sequence[Any], label: str) -> tuple[float, float]:
    if len(value) != 2:
        raise CityPlaceInputError(f"{label} must contain exactly two numbers")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(v) for v in result):
        raise CityPlaceInputError(f"{label} contains a non-finite number")
    return result


def _metadata_from_survey(
    survey: Mapping[str, Any], field_shape: tuple[int, int], field_pass: str
) -> dict[str, Any]:
    """Build explicit metadata for the canonical T0.2 field.

    T0.2 stores the dense-field frame in the survey JSON and the numerical
    values in NPZ.  This adapter makes that split explicit instead of copying
    a hidden coordinate constant into the solver.
    """

    frame = survey.get("frame")
    if not isinstance(frame, Mapping):
        raise CityPlaceInputError("site survey has no frame object")
    origin = frame.get("origin_gu")
    spacing = frame.get("field_spacing_gu")
    if not isinstance(origin, list) or len(origin) != 2:
        raise CityPlaceInputError("site survey frame origin_gu is not a pair")
    if not isinstance(spacing, (int, float)):
        raise CityPlaceInputError("site survey frame field_spacing_gu is missing")
    return {
        "schema_version": 1,
        "frame_origin_gu": [float(origin[0]), float(origin[1])],
        "spacing_gu": [float(spacing), float(spacing)],
        "shape": [int(field_shape[0]), int(field_shape[1])],
        "units": FIELD_UNITS,
        "pass": field_pass,
        "provenance": "site_survey.frame + survey_fields.npz",
    }


@dataclass(frozen=True)
class TerrainSample:
    """One bilinear field sample plus its analytic height-derived normal."""

    x_plan_gu: float
    y_plan_gu: float
    x_field: float
    y_field: float
    height_gu: float
    slope_deg: float
    normal: tuple[float, float, float]
    field_index: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_plan_gu": [self.x_plan_gu, self.y_plan_gu],
            "field_coordinate": [self.x_field, self.y_field],
            "height_gu": self.height_gu,
            "slope_deg": self.slope_deg,
            "normal": list(self.normal),
            "field_index": list(self.field_index),
        }


@dataclass(frozen=True)
class TerrainField:
    """Validated dense height field in absolute-world GU with plan sampling."""

    values_gu: np.ndarray
    origin_gu: tuple[float, float]
    spacing_gu: tuple[float, float]
    units: str
    field_pass: str
    source_path: Path
    source_sha256: str
    metadata: dict[str, Any]
    metadata_path: Path | None = None
    metadata_sha256: str | None = None

    @classmethod
    def from_npz(
        cls,
        path: Path | str,
        *,
        survey: Mapping[str, Any],
        field_pass: str,
        metadata_path: Path | str | None = None,
    ) -> "TerrainField":
        """Load and validate one planned/final NPZ field.

        The caller must pass the execution pass explicitly.  If no sidecar is
        supplied, only the canonical survey-derived metadata adapter is used;
        arbitrary fields therefore cannot masquerade as the accepted field.
        """

        if field_pass not in FIELD_PASSES:
            raise CityPlaceInputError(
                f"terrain pass must be one of {FIELD_PASSES}, got {field_pass!r}"
            )
        source = Path(path)
        if not source.is_file():
            raise CityPlaceInputError(f"terrain field is missing: {source}")
        try:
            with np.load(source, allow_pickle=False) as archive:
                if "height_gu" in archive.files:
                    values = np.asarray(archive["height_gu"], dtype=np.float64)
                elif "elevation_gu" in archive.files:
                    values = np.asarray(archive["elevation_gu"], dtype=np.float64)
                else:
                    raise CityPlaceInputError(
                        f"terrain field {source} has no height_gu/elevation_gu array"
                    )
        except (OSError, ValueError, KeyError) as exc:
            if isinstance(exc, CityPlaceInputError):
                raise
            raise CityPlaceInputError(f"cannot read terrain field {source}: {exc}") from exc
        if values.ndim != 2 or min(values.shape) < 2:
            raise CityPlaceInputError(
                f"terrain field must be a 2-D array with both axes >=2, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise CityPlaceInputError(f"terrain field contains NaN or infinity: {source}")
        values = np.ascontiguousarray(values, dtype=np.float64)

        meta_target = Path(metadata_path) if metadata_path is not None else None
        if meta_target is not None:
            metadata = load_json(meta_target, "terrain field metadata")
        else:
            metadata = _metadata_from_survey(survey, values.shape, field_pass)
        if metadata.get("schema_version") != 1:
            raise CityPlaceInputError("terrain field metadata schema_version must be 1")
        origin_value = metadata.get("frame_origin_gu", metadata.get("origin_gu"))
        spacing_value = metadata.get("spacing_gu", metadata.get("field_spacing_gu"))
        if not isinstance(origin_value, list) or len(origin_value) != 2:
            raise CityPlaceInputError("terrain field metadata frame_origin_gu is missing")
        if isinstance(spacing_value, (int, float)):
            spacing_value = [spacing_value, spacing_value]
        if not isinstance(spacing_value, list) or len(spacing_value) != 2:
            raise CityPlaceInputError("terrain field metadata spacing_gu is missing")
        origin = _finite_pair(origin_value, "terrain field origin")
        spacing = _finite_pair(spacing_value, "terrain field spacing")
        if min(spacing) <= 0.0:
            raise CityPlaceInputError("terrain field spacing must be positive")
        if metadata.get("units") != FIELD_UNITS:
            raise CityPlaceInputError(
                f"terrain field units must be {FIELD_UNITS!r}, got {metadata.get('units')!r}"
            )
        if metadata.get("pass") != field_pass:
            raise CityPlaceInputError(
                f"terrain field pass metadata {metadata.get('pass')!r} does not match "
                f"requested {field_pass!r}"
            )
        shape = metadata.get("shape")
        if shape != [int(values.shape[0]), int(values.shape[1])]:
            raise CityPlaceInputError(
                f"terrain field shape metadata {shape!r} does not match {list(values.shape)!r}"
            )

        # The accepted survey is the only valid site frame for this stage.
        frame = survey.get("frame")
        if not isinstance(frame, Mapping):
            raise CityPlaceInputError("site survey frame is missing")
        survey_origin = _finite_pair(frame.get("origin_gu"), "survey origin")
        survey_spacing = float(frame.get("field_spacing_gu"))
        if origin != survey_origin or spacing != (survey_spacing, survey_spacing):
            raise CityPlaceInputError(
                "terrain field frame does not exactly match the accepted site frame"
            )

        return cls(
            values_gu=values,
            origin_gu=origin,
            spacing_gu=spacing,
            units=FIELD_UNITS,
            field_pass=field_pass,
            source_path=source,
            source_sha256=sha256_file(source),
            metadata=metadata,
            metadata_path=meta_target,
            metadata_sha256=sha256_file(meta_target) if meta_target is not None else None,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.values_gu.shape[0]), int(self.values_gu.shape[1]))

    @property
    def extent_gu(self) -> tuple[float, float]:
        """Absolute field span from the first to the last sample point."""

        return (
            (self.shape[1] - 1) * self.spacing_gu[0],
            (self.shape[0] - 1) * self.spacing_gu[1],
        )

    def _coordinates(self, x_plan_gu: float, y_plan_gu: float) -> tuple[float, float]:
        if not math.isfinite(x_plan_gu) or not math.isfinite(y_plan_gu):
            raise FieldCoverageError("terrain sample coordinate is non-finite")
        abs_x = self.origin_gu[0] + float(x_plan_gu)
        abs_y = self.origin_gu[1] + float(y_plan_gu)
        fx = (abs_x - self.origin_gu[0]) / self.spacing_gu[0]
        fy = (abs_y - self.origin_gu[1]) / self.spacing_gu[1]
        max_x = self.shape[1] - 1
        max_y = self.shape[0] - 1
        if fx < 0.0 or fy < 0.0 or fx > max_x or fy > max_y:
            raise FieldCoverageError(
                f"terrain field does not cover plan point ({x_plan_gu}, {y_plan_gu})"
            )
        # A tiny clamp only removes arithmetic noise at the exact last sample;
        # it never turns an out-of-frame request into a covered one.
        return min(max(fx, 0.0), float(max_x)), min(max(fy, 0.0), float(max_y))

    def sample(self, x_plan_gu: float, y_plan_gu: float) -> TerrainSample:
        """Return a deterministic bilinear height, slope, and up normal."""

        fx, fy = self._coordinates(float(x_plan_gu), float(y_plan_gu))
        ix = min(int(math.floor(fx)), self.shape[1] - 2)
        iy = min(int(math.floor(fy)), self.shape[0] - 2)
        tx = fx - ix
        ty = fy - iy
        v00 = float(self.values_gu[iy, ix])
        v10 = float(self.values_gu[iy, ix + 1])
        v01 = float(self.values_gu[iy + 1, ix])
        v11 = float(self.values_gu[iy + 1, ix + 1])
        lower = v00 * (1.0 - tx) + v10 * tx
        upper = v01 * (1.0 - tx) + v11 * tx
        height = lower * (1.0 - ty) + upper * ty
        dzdx = ((v10 - v00) * (1.0 - ty) + (v11 - v01) * ty) / self.spacing_gu[0]
        dzdy = ((v01 - v00) * (1.0 - tx) + (v11 - v10) * tx) / self.spacing_gu[1]
        slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
        nx, ny, nz = -dzdx, -dzdy, 1.0
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        return TerrainSample(
            x_plan_gu=float(x_plan_gu),
            y_plan_gu=float(y_plan_gu),
            x_field=fx,
            y_field=fy,
            height_gu=height,
            slope_deg=slope,
            normal=(nx / norm, ny / norm, nz / norm),
            field_index=(int(math.floor(float(x_plan_gu) / self.spacing_gu[0])),
                         int(math.floor(float(y_plan_gu) / self.spacing_gu[1]))),
        )

    def contract_dict(self) -> dict[str, Any]:
        """Stable provenance record for solver manifests and audits."""

        return {
            "path": str(self.source_path),
            "sha256": self.source_sha256,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "metadata_sha256": self.metadata_sha256,
            "frame_origin_gu": list(self.origin_gu),
            "spacing_gu": list(self.spacing_gu),
            "shape": list(self.shape),
            "units": self.units,
            "pass": self.field_pass,
            "dtype": str(self.values_gu.dtype),
            "extent_gu": list(self.extent_gu),
        }


def field_index(x_plan_gu: float, y_plan_gu: float, spacing_gu: float) -> tuple[int, int]:
    """Mathematical-floor plan-frame field index, including negative inputs."""

    return (int(math.floor(float(x_plan_gu) / float(spacing_gu))),
            int(math.floor(float(y_plan_gu) / float(spacing_gu))))
