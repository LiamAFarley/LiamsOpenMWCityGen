"""Falkreath kit house assembly from native NIF bounds (no stamp examples).

Each shell has door and window slots on all four facades. Callers pick which
doors and which window faces to emit. Door-local ``-Y`` is outward.

Openings sit on the first-floor wall rectangle (roof overhang is not a wall).
A single AABB inset cannot work: gable and eave overhangs differ by face.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from shapely.geometry import MultiPoint

from . import engine_transform
from .kit_house_grammar import canonical_json_bytes

BOUNDS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "kits" / "falkreath" / "kit_bounds.json"
)
DORMER_BOUNDS_PATH = BOUNDS_PATH.with_name("dormer_bounds.json")
WALL_PROFILES_PATH = BOUNDS_PATH.with_name("wall_profiles.json")
REFERENCE_KIT_PATH = (
    Path(__file__).resolve().parents[2]
    / "output" / "cityforge" / "falkreath_kit_extraction_v1" / "kit.json"
)

# Outer face sits this far outside the first-floor wall AABB. Keep these small:
# a previous "anti-clip" change of ~100 GU put openings out at the eaves.
DOOR_PROTRUSION_GU = 24.0
# Pull door panel back into the frame recess; frame placement is unchanged.
DOOR_RECESS_IN_FRAME_GU = 18.0
WINDOW_PROTRUSION_GU = 30.0
STORY2_PROTRUSION_GU = 38.0
# Offset from door origin → frame origin in door-local coords (tuned on house 02).
# Applied with the door rotation so every facade gets the same relationship.
DOOR_FRAME_OFFSET_LOCAL: dict[tuple[str, str], tuple[float, float, float]] = {
    # Local +Y: align outward (min-Y) faces. Local Z: drop frame sill below door bottom.
    ("sky_ex_fk_door_01", "sky_FK_DFrame_01"): (0.0, 18.204, -100.0),
    ("sky_ex_fk_door_02", "sky_FK_DFrame_01"): (0.0, 19.81, -104.0),
}
WINDOW_LINTEL_BELOW_DOOR_GU = 8.0
STORY2_LIFT_GU = 270.0
CHIMNEY_RIDGE_SINK_GU = 180.0
CHIMNEY_WALL_EMBED_GU = 40.0
FACADE_IDS = ("neg_x", "pos_x", "neg_y", "pos_y")

# Texture variants (_a/_b/_c) share one family spec. Wall rectangles come from
# blender_fk_wall_slice.py on the _a mesh. Mix any WINDOW_MODELS onto any shell.
_FAMILIES: dict[str, dict[str, Any]] = {
    "01": {
        "variants": ("a", "b", "c"),
        "plinth_top_gu": -18.0,
        "wall_min_xy": (-239.117, -380.142),
        "wall_max_xy": (239.026, 380.142),
        "default_door_facades": ("neg_x",),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.86,
        "window_pitch_gu": 260.0,
        "window_bands": ("lintel",),
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 330.0,
        "chimney_across_gu": 0.0,
    },
    "02": {
        "variants": ("a", "b"),
        "plinth_top_gu": -18.0,
        "wall_min_xy": (-239.117, -380.142),
        "wall_max_xy": (239.026, 380.142),
        "default_door_facades": ("neg_x",),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.70,
        "window_pitch_gu": 260.0,
        "window_bands": ("lintel",),
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 400.0,
        "chimney_across_gu": 0.0,
    },
    "03": {
        "variants": ("a", "b"),
        "plinth_top_gu": -18.0,
        "wall_min_xy": (-239.117, -780.142),
        "wall_max_xy": (239.026, 780.142),
        "default_door_facades": ("neg_x",),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.70,
        "window_pitch_gu": 260.0,
        "window_bands": ("lintel",),
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 650.0,
        "chimney_across_gu": 0.0,
    },
    "04": {
        "variants": ("a", "b"),
        "plinth_top_gu": -18.0,
        "wall_min_xy": (-249.108, -780.143),
        "wall_max_xy": (1249.572, 735.651),
        "default_door_facades": ("inner_neg_y",),
        "default_window_facades": (
            "neg_x",
            "pos_x",
            "neg_y",
            "pos_y",
            "inner_pos_x",
            "inner_neg_y",
        ),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.70,
        "window_bands": ("lintel",),
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 400.0,
        "chimney_across_gu": 493.0,
        # First-floor wall planes from blender_fk_l_wings.py, not the AABB.
        "wall_faces": {
            "neg_x": {
                "outward": (-1.0, 0.0),
                "face": -249.108,
                "tangent_min": -780.143,
                "tangent_max": 735.651,
            },
            "pos_x": {
                "outward": (1.0, 0.0),
                "face": 1249.572,
                "tangent_min": 250.628,
                "tangent_max": 735.651,
            },
            "pos_y": {
                "outward": (0.0, 1.0),
                "face": 735.651,
                "tangent_min": -249.108,
                "tangent_max": 1249.572,
            },
            "neg_y": {
                "outward": (0.0, -1.0),
                "face": -780.143,
                "tangent_min": -249.108,
                "tangent_max": 239.043,
            },
            "inner_pos_x": {
                "outward": (1.0, 0.0),
                "face": 239.043,
                "tangent_min": -780.143,
                "tangent_max": 250.628,
            },
            "inner_neg_y": {
                "outward": (0.0, -1.0),
                "face": 250.628,
                "tangent_min": 239.043,
                "tangent_max": 1249.572,
            },
        },
    },
    "05": {
        "variants": ("a", "b", "c"),
        "plinth_top_gu": -8.0,
        "wall_min_xy": (-239.073, -380.142),
        "wall_max_xy": (239.07, 380.142),
        "default_door_facades": ("neg_x",),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.72,
        "window_pitch_gu": 260.0,
        "window_bands": ("lintel",),
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 24.0,
        "chimney_across_gu": 0.0,
    },
    "06": {
        "variants": ("a", "b", "c", "d"),
        "plinth_top_gu": -8.0,
        "wall_min_xy": (-224.064, -334.772),
        "wall_max_xy": (223.708, 338.821),
        "default_door_facades": ("neg_x",),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.88,
        "window_bands": ("lintel", "story2"),
        "story2_lift_gu": 270.0,
        "window_pitch_gu": 260.0,
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 280.0,
        "chimney_across_gu": 0.0,
    },
    "07": {
        "variants": ("a", "b"),
        "plinth_top_gu": -8.0,
        "door_recess_in_frame_gu": 6.0,
        "wall_min_xy": (-224.064, -334.772),
        "wall_max_xy": (434.674, 338.821),
        "default_door_facades": ("inner_pos_x",),
        "default_window_facades": ("neg_x", "neg_y", "pos_y", "inner_pos_x", "pos_x"),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.80,
        "window_bands": ("lintel", "story2"),
        "story2_lift_gu": 270.0,
        "window_pitch_gu": 260.0,
        "door_lateral_gu": -140.0,
        "window_facades_by_band": {
            "lintel": ("neg_x", "neg_y", "pos_y", "inner_pos_x"),
            "story2": ("neg_x", "neg_y", "pos_y", "pos_x"),
        },
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 280.0,
        "chimney_across_gu": -109.0,
        "wall_faces": {
            "neg_x": {
                "outward": (-1.0, 0.0),
                "face": -224.064,
                "tangent_min": -334.772,
                "tangent_max": 338.821,
            },
            "pos_y": {
                "outward": (0.0, 1.0),
                "face": 338.821,
                "tangent_min": -224.064,
                "tangent_max": 223.708,
            },
            "neg_y": {
                "outward": (0.0, -1.0),
                "face": -334.772,
                "tangent_min": -224.064,
                "tangent_max": 223.708,
            },
            "inner_pos_x": {
                "outward": (1.0, 0.0),
                "face": 223.708,
                "tangent_min": -180.0,
                "tangent_max": 180.0,
            },
            "pos_x": {
                "outward": (1.0, 0.0),
                "face": 434.674,
                "tangent_min": -334.772,
                "tangent_max": 338.821,
            },
        },
    },
    "08": {
        "variants": ("a", "b"),
        "plinth_top_gu": -8.0,
        "door_recess_in_frame_gu": 6.0,
        # Brick plane from inner clusters at z 160-320; outer ±239 is the log story.
        "wall_min_xy": (-218.838, -341.203),
        "wall_max_xy": (218.836, 343.274),
        "default_door_facades": ("neg_x",),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.90,
        "window_bands": ("lintel", "story2"),
        "story2_lift_gu": 270.0,
        "window_pitch_gu": 260.0,
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 24.0,
        "chimney_across_gu": 0.0,
    },
    "09": {
        "variants": ("a", "b"),
        "plinth_top_gu": -8.0,
        "door_recess_in_frame_gu": 6.0,
        "wall_min_xy": (-224.064, -464.594),
        "wall_max_xy": (223.708, 448.527),
        "default_door_facades": ("inner_neg_y",),
        "default_window_facades": ("neg_x", "pos_x", "pos_y", "inner_neg_y"),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.78,
        "window_pitch_gu": 200.0,
        "window_bands": ("lintel", "story2"),
        "story2_lift_gu": 270.0,
        "door_lateral_gu": 0.0,
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 400.0,
        "chimney_across_gu": 0.0,
        "wall_faces": {
            "neg_x": {
                "outward": (-1.0, 0.0),
                "face": -224.064,
                "tangent_min": -426.213,
                "tangent_max": 448.527,
            },
            "pos_x": {
                "outward": (1.0, 0.0),
                "face": 223.708,
                "tangent_min": -426.213,
                "tangent_max": 448.527,
            },
            "pos_y": {
                "outward": (0.0, 1.0),
                "face": 448.527,
                "tangent_min": -224.064,
                "tangent_max": 223.708,
            },
            "inner_neg_y": {
                "outward": (0.0, -1.0),
                "face": -390.0,
                "tangent_min": -182.818,
                "tangent_max": 182.818,
            },
        },
    },
    "10": {
        "variants": ("a", "b"),
        "plinth_top_gu": -34.0,
        "wall_min_xy": (-624.913, -642.217),
        "wall_max_xy": (602.083, 642.217),
        "default_door_facades": ("neg_y",),
        "door_model": "sky_ex_fk_door_02",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "six_thirds",
        "window_span_frac": 0.86,
        "window_bands": ("lintel", "story2"),
        "story2_lift_gu": 270.0,
        "chimney_model": "sky_FK_Chimney_01",
        "chimney_mode": "wall",
        "chimney_facade": "pos_y",
        "chimney_lateral_gu": 0.0,
    },
    "11": {
        "variants": ("a",),
        "plinth_top_gu": -18.0,
        "wall_min_xy": (-1413.654, -780.143),
        "wall_max_xy": (1413.653, 735.651),
        "default_door_facades": ("inner_neg_y", "pos_y"),
        "default_window_facades": (
            "neg_x",
            "pos_x",
            "pos_y",
            "neg_y",
            "neg_y_east",
            "inner_pos_x",
            "inner_neg_x",
            "inner_neg_y",
        ),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.88,
        "window_pitch_gu": 360.0,
        "window_bands": ("lintel",),
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 0.0,
        "chimney_across_gu": 508.0,
        "wall_faces": {
            "neg_x": {
                "outward": (-1.0, 0.0),
                "face": -1413.654,
                "tangent_min": -780.143,
                "tangent_max": 735.651,
            },
            "pos_x": {
                "outward": (1.0, 0.0),
                "face": 1413.653,
                "tangent_min": -780.143,
                "tangent_max": 735.651,
            },
            "pos_y": {
                "outward": (0.0, 1.0),
                "face": 735.651,
                "tangent_min": -1413.654,
                "tangent_max": 1413.653,
            },
            "neg_y": {
                "outward": (0.0, -1.0),
                "face": -780.143,
                "tangent_min": -1413.654,
                "tangent_max": -925.519,
            },
            "neg_y_east": {
                "outward": (0.0, -1.0),
                "face": -780.143,
                "tangent_min": 925.519,
                "tangent_max": 1413.653,
            },
            "inner_pos_x": {
                "outward": (1.0, 0.0),
                "face": -925.519,
                "tangent_min": -780.143,
                "tangent_max": 261.62,
            },
            "inner_neg_x": {
                "outward": (-1.0, 0.0),
                "face": 925.519,
                "tangent_min": -780.143,
                "tangent_max": 261.62,
            },
            "inner_neg_y": {
                "outward": (0.0, -1.0),
                "face": 261.62,
                "tangent_min": -925.519,
                "tangent_max": 925.519,
            },
        },
    },
    "12": {
        "variants": ("a",),
        "plinth_top_gu": -8.0,
        "wall_min_xy": (-265.129, -306.145),
        "wall_max_xy": (265.13, 306.145),
        "default_door_facades": ("south_west",),
        "door_model": "sky_ex_fk_door_01",
        "door_frame": "sky_FK_DFrame_01",
        "window_model": "sky_FK_Window_04a",
        "window_layout": "flank",
        "window_span_frac": 0.70,
        "window_bands": ("lintel", "story2"),
        "story2_lift_gu": 270.0,
        "chimney_model": "sky_FK_Chimney_02",
        "chimney_mode": "roof",
        "chimney_along_gu": 0.0,
        "chimney_across_gu": 0.0,
        "default_window_facades": (
            "west",
            "north_west",
            "north_east",
            "east",
            "south_east",
            "south_west",
        ),
        "window_laterals_by_facade": {
            "west": (0.0,),
            "north_west": (0.0,),
            "north_east": (0.0,),
            "east": (0.0,),
        },
        # Regular pointy-top hex: south/north are corners, so openings belong on
        # the six flats around them, not on the cardinal vertex directions.
        "wall_faces": {
            "west": {
                "outward": (-1.0, 0.0),
                "face": -265.129,
                "face_mode": "normal",
                "tangent_min": -153.073,
                "tangent_max": 153.073,
            },
            "north_west": {
                "outward": (-0.5, 0.8660254),
                "face": 265.129,
                "face_mode": "normal",
                "tangent_min": -153.073,
                "tangent_max": 153.073,
            },
            "north_east": {
                "outward": (0.5, 0.8660254),
                "face": 265.129,
                "face_mode": "normal",
                "tangent_min": -153.073,
                "tangent_max": 153.073,
            },
            "east": {
                "outward": (1.0, 0.0),
                "face": 265.13,
                "face_mode": "normal",
                "tangent_min": -153.073,
                "tangent_max": 153.073,
            },
            "south_east": {
                "outward": (0.5, -0.8660254),
                "face": 265.129,
                "face_mode": "normal",
                "tangent_min": -153.073,
                "tangent_max": 153.073,
            },
            "south_west": {
                "outward": (-0.5, -0.8660254),
                "face": 265.129,
                "face_mode": "normal",
                "tangent_min": -153.073,
                "tangent_max": 153.073,
            },
        },
    },
}


def _build_shell_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for family, spec in _FAMILIES.items():
        for variant in spec["variants"]:
            shell_id = f"sky_FK_house_{family}_{variant}"
            row = {key: value for key, value in spec.items() if key != "variants"}
            row["model_key"] = f"sky\\x\\{shell_id}.nif"
            row["family"] = family
            available_facades = tuple((spec.get("wall_faces") or {}).keys()) or FACADE_IDS
            primary_doors = tuple(str(value) for value in spec["default_door_facades"])
            row["secondary_door_facades"] = (
                ()
                if family == "12"
                else tuple(face for face in available_facades if face not in primary_doors)
            )
            specs[shell_id] = row
    return specs


SHELL_SPECS = _build_shell_specs()


def reference_shell_ids(path: Path | None = None) -> tuple[str, ...]:
    """Return observed shell variants from the extracted Falkreath reference."""
    source = path or REFERENCE_KIT_PATH
    try:
        import json
        with source.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError, TypeError):
        document = {}
    observed: set[str] = set()
    for row in document.get("models", []) if isinstance(document, dict) else ():
        if not isinstance(row, dict):
            continue
        family = str(row.get("family", ""))
        if family.startswith("house_"):
            observed.add("sky_FK_" + family)
    valid = sorted(shell_id for shell_id in observed if shell_id in SHELL_SPECS)
    if valid:
        return tuple(valid)
    return tuple(f"sky_FK_house_{family}_a" for family in sorted(_FAMILIES))

WINDOW_MODELS: dict[str, str] = {
    "sky_FK_Window_01a": "sky\\x\\sky_FK_Window_01a.nif",
    "sky_FK_Window_01b": "sky\\x\\sky_FK_Window_01b.nif",
    "sky_FK_Window_02a": "sky\\x\\sky_FK_Window_02a.nif",
    "sky_FK_Window_02b": "sky\\x\\sky_FK_Window_02b.nif",
    "sky_FK_Window_03a": "sky\\x\\sky_FK_Window_03a.nif",
    "sky_FK_Window_03b": "sky\\x\\sky_FK_Window_03b.nif",
    "sky_FK_Window_04a": "sky\\x\\sky_FK_Window_04a.nif",
    "sky_FK_Window_04b": "sky\\x\\sky_FK_Window_04b.nif",
    "sky_FK_Window_04c": "sky\\x\\sky_FK_Window_04c.nif",
    "sky_FK_Window_05a": "sky\\x\\sky_FK_Window_05a.nif",
    "sky_FK_Window_05b": "sky\\x\\sky_FK_Window_05b.nif",
    "sky_FK_Window_06a": "sky\\x\\sky_FK_Window_06a.nif",
    "sky_FK_Window_06b": "sky\\x\\sky_FK_Window_06b.nif",
}

DOOR_MODELS: dict[str, str] = {
    "sky_ex_fk_door_01": "sky\\d\\sky_ex_fk_door_01.nif",
    "sky_ex_fk_door_02": "sky\\d\\sky_ex_fk_door_02.nif",
    "sky_ex_fk_door_03": "sky\\d\\sky_ex_fk_door_03.nif",
}

FRAME_MODELS: dict[str, str] = {
    "sky_FK_DFrame_01": "sky\\x\\sky_FK_DFrame_01.nif",
    "sky_FK_DFrame_02": "sky\\x\\sky_FK_DFrame_02.nif",
    "sky_FK_DFrame_03": "sky\\x\\sky_FK_DFrame_03.nif",
}

PORCH_MODELS: dict[str, str] = {
    "sky_FK_Porch_01a": "sky\\x\\sky_FK_Porch_01a.nif",
    "sky_FK_Porch_01b": "sky\\x\\sky_FK_Porch_01b.nif",
    "sky_FK_Porch_01c": "sky\\x\\sky_FK_Porch_01c.nif",
    "sky_FK_Porch_02a": "sky\\x\\sky_FK_Porch_02a.nif",
    "sky_FK_Porch_02b": "sky\\x\\sky_FK_Porch_02b.nif",
}

# Deck/landing references are measured internal planes, not total-mesh minima:
# Porch 01 has support legs below its deck; Porch 02 has stone steps below its
# raised landing. Aligning the total AABB minimum made the porch float above
# the door.
PORCH_ACCESS_Z_LOCAL: dict[str, float] = {
    "sky_FK_Porch_01a": -1.392,
    "sky_FK_Porch_01b": -1.392,
    "sky_FK_Porch_01c": -1.392,
    "sky_FK_Porch_02a": 43.681,
    "sky_FK_Porch_02b": 40.966,
}

STAIR_MODELS: dict[str, str] = {
    "sky_ex_mk_str_02": "sky\\x\\sky_ex_mk_str_02.nif",
}

DORMER_MODELS: dict[str, str] = {
    "sky_FK_Dormer_01a": "sky\\x\\sky_FK_Dormer_01a.nif",
    "sky_FK_Dormer_01b": "sky\\x\\sky_FK_Dormer_01b.nif",
    "sky_FK_Dormer_01c": "sky\\x\\sky_FK_Dormer_01c.nif",
    "sky_FK_Dormer_02a": "sky\\x\\sky_FK_Dormer_02a.nif",
    "sky_FK_Dormer_02b": "sky\\x\\sky_FK_Dormer_02b.nif",
    "sky_FK_Dormer_02c": "sky\\x\\sky_FK_Dormer_02c.nif",
}

CHIMNEY_MODELS: dict[str, str] = {
    "sky_FK_Chimney_01": "sky\\x\\sky_FK_Chimney_01.nif",
    "sky_FK_Chimney_02": "sky\\x\\sky_FK_Chimney_02.nif",
}

KIT_MODELS: dict[str, str] = {
    **WINDOW_MODELS,
    **DOOR_MODELS,
    **FRAME_MODELS,
    **CHIMNEY_MODELS,
    **PORCH_MODELS,
    **STAIR_MODELS,
    **DORMER_MODELS,
}

_FACADES: dict[str, tuple[float, float]] = {
    "pos_x": (1.0, 0.0),
    "neg_x": (-1.0, 0.0),
    "pos_y": (0.0, 1.0),
    "neg_y": (0.0, -1.0),
}


def load_kit_bounds(path: Path | None = None) -> dict[str, dict[str, Any]]:
    import json

    payload = json.loads((path or BOUNDS_PATH).read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for row in payload["meshes"]:
        key = str(row["model_key"]).replace("/", "\\")
        index[key] = row["bounds_gu"]
        stem = key.rsplit("\\", 1)[-1]
        if stem.lower().endswith(".nif"):
            index[stem[:-4]] = row["bounds_gu"]
    dormer_path = DORMER_BOUNDS_PATH
    if dormer_path.exists() and (path is None or dormer_path != path):
        dormer_payload = json.loads(dormer_path.read_text(encoding="utf-8"))
        for row in dormer_payload.get("meshes", []):
            key = str(row["model_key"]).replace("/", "\\")
            index[key] = row["bounds_gu"]
            stem = key.rsplit("\\", 1)[-1]
            if stem.lower().endswith(".nif"):
                index[stem[:-4]] = row["bounds_gu"]
    for shell_id, spec in SHELL_SPECS.items():
        key = str(spec["model_key"]).replace("/", "\\")
        if key in index:
            continue
        alias = f"sky\\x\\sky_FK_house_{spec['family']}_a.nif"
        if alias in index:
            index[key] = index[alias]
            index[shell_id] = index[alias]
    return index


def load_wall_profiles(path: Path | None = None) -> dict[str, Any]:
    profile_path = path or WALL_PROFILES_PATH
    if not profile_path.exists():
        raise ValueError(f"missing Falkreath wall profiles: {profile_path}")
    import json
    return json.loads(profile_path.read_text(encoding="utf-8"))


def fk_secondary_door_facades(shell_id: str) -> tuple[str, ...]:
    """Return family-valid side/rear door facades for city-layout routing."""
    spec = SHELL_SPECS.get(shell_id)
    if spec is None:
        raise ValueError(f"unknown Falkreath shell {shell_id!r}")
    return tuple(str(value) for value in spec.get("secondary_door_facades") or ())


def rotz_for_outward(ox: float, oy: float) -> float:
    return math.atan2(-ox, -oy)


def outward_heading_deg(ox: float, oy: float) -> float:
    return (math.degrees(math.atan2(oy, ox)) + 360.0) % 360.0


def parse_facade_list(
    value: str | Sequence[str] | None,
    *,
    default: Sequence[str],
    allowed: Mapping[str, Any] | None = None,
) -> list[str]:
    catalog = allowed if allowed is not None else _FACADES
    if value is None:
        return [str(item) for item in default]
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    else:
        parts = [str(item).strip() for item in value if str(item).strip()]
    unknown = [part for part in parts if part not in catalog]
    if unknown:
        raise ValueError(f"unknown facade id(s) {unknown}; use {list(catalog)}")
    return parts


def _allowed_facades(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    faces = spec.get("wall_faces")
    if faces:
        return faces
    return _FACADES


def _facade_outward(spec: Mapping[str, Any], facade_id: str) -> tuple[float, float]:
    faces = spec.get("wall_faces") or {}
    if facade_id in faces:
        ox, oy = faces[facade_id]["outward"]
        return float(ox), float(oy)
    return _FACADES[facade_id]


def _wall_rect(spec: Mapping[str, Any]) -> tuple[float, float, float, float]:
    wall_min = spec["wall_min_xy"]
    wall_max = spec["wall_max_xy"]
    return (float(wall_min[0]), float(wall_min[1]), float(wall_max[0]), float(wall_max[1]))


def _wall_face_and_tangent(spec: Mapping[str, Any], facade_id: str) -> tuple[float, float, float]:
    """Return (normal_face, tangent_center, tangent_span) for a facade."""
    faces = spec.get("wall_faces") or {}
    if facade_id in faces:
        row = faces[facade_id]
        t0 = float(row["tangent_min"])
        t1 = float(row["tangent_max"])
        return float(row["face"]), 0.5 * (t0 + t1), t1 - t0
    x0, y0, x1, y1 = _wall_rect(spec)
    ox, oy = _facade_outward(spec, facade_id)
    if abs(ox) >= abs(oy):
        face = x0 if ox < 0 else x1
        return face, 0.5 * (y0 + y1), y1 - y0
    face = y0 if oy < 0 else y1
    return face, 0.5 * (x0 + x1), x1 - x0


def _profile_edge(
    spec: Mapping[str, Any],
    facade_id: str,
    pivot_z: float,
    profiles: Mapping[str, Any],
    allow_nearest: bool = False,
) -> dict[str, Any] | None:
    """Resolve a measured edge for a pilot opening placement.

    Semantic facade names remain owned by the family spec. The measured
    profile is only queried by expected normal and the legacy face/tangent
    hint; it never invents a semantic facade name. Preview generation may use
    ``needs_review`` profiles, but production generation rejects them before
    reaching this helper.
    """
    shell_key = str(spec["model_key"]).replace("/", "\\")
    profile = profiles.get("shells", {}).get(Path(shell_key).stem)
    if profile is None:
        raise ValueError(f"no wall profile for {shell_key}")
    expected = _facade_outward(spec, facade_id)
    face_hint, tangent_hint, _span = _wall_face_and_tangent(spec, facade_id)
    face_row = (spec.get("wall_faces") or {}).get(facade_id) or {}
    # `_wall_face_and_tangent` returns an axis coordinate for cardinal faces,
    # while measured edges are scored in outward-normal projection space.
    # Convert the legacy coordinate before comparing; otherwise neg_x could
    # select the +X wall because both planes had the same absolute distance.
    if face_row.get("face_mode") != "normal":
        axis_sign = expected[0] if abs(expected[0]) >= abs(expected[1]) else expected[1]
        face_hint *= axis_sign
    candidates: list[tuple[float, dict[str, Any]]] = []
    for band in profile.get("bands", []):
        if float(band["z0"]) - 8.0 <= pivot_z <= float(band["z1"]) + 8.0:
            for edge in band.get("candidate_edges", []):
                ox, oy = map(float, edge["outward"])
                normal_error = 1.0 - (ox * expected[0] + oy * expected[1])
                a, b = edge["a"], edge["b"]
                mid = ((float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0)
                face = mid[0] * expected[0] + mid[1] * expected[1]
                tangent = (-expected[1], expected[0])
                tangent_mid = mid[0] * tangent[0] + mid[1] * tangent[1]
                candidates.append((normal_error * 100000.0 + abs(face - face_hint) + abs(tangent_mid - tangent_hint) * 0.01, edge))
    if not candidates and allow_nearest:
        for band in profile.get("bands", []):
            for edge in band.get("candidate_edges", []):
                ox, oy = map(float, edge["outward"])
                normal_error = 1.0 - (ox * expected[0] + oy * expected[1])
                a, b = edge["a"], edge["b"]
                mid = ((float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0)
                candidates.append((normal_error * 100000.0 + abs(mid[0] * expected[0] + mid[1] * expected[1] - face_hint) + abs(mid[0] * (-expected[1]) + mid[1] * expected[0] - tangent_hint) * 0.01 + abs((float(band["z0"]) + float(band["z1"])) / 2.0 - pivot_z), edge))
    if not candidates:
        raise ValueError(f"no measured wall edge supports {spec['family']}:{facade_id} at z={pivot_z:.1f}")
    selected = min(candidates, key=lambda item: item[0])[1]
    selected_outward = tuple(float(value) for value in selected["outward"])
    normal_dot = selected_outward[0] * expected[0] + selected_outward[1] * expected[1]
    if normal_dot < 0.85:
        # The measured stack has no supported edge for this semantic wall at
        # the requested height (common on gable ends). Let the semantic wall
        # binding place it on its declared plane instead of silently rotating
        # the opening onto a different wall.
        return None
    return selected


def _six_slot_laterals(length: float, skip_third: int | None) -> tuple[float, ...]:
    slots = tuple(((i + 0.5) / 6.0 - 0.5) * length for i in range(6))
    if skip_third is None:
        return slots
    skip = {skip_third * 2, skip_third * 2 + 1}
    return tuple(value for index, value in enumerate(slots) if index not in skip)


def _window_laterals(
    spec: Mapping[str, Any],
    facade_id: str,
    *,
    has_door: bool,
    band: str,
    skip_center: bool = False,
    profile_edge: Mapping[str, Any] | None = None,
) -> tuple[float, ...]:
    band_override = spec.get("window_laterals_by_band_facade") or {}
    if band in band_override and facade_id in band_override[band]:
        return tuple(float(value) for value in band_override[band][facade_id])
    _face, _center, span = _wall_face_and_tangent(spec, facade_id)
    override = spec.get("window_laterals_by_facade") or {}
    if profile_edge is None and facade_id in override:
        return tuple(float(value) for value in override[facade_id])
    if profile_edge is not None:
        a, b = profile_edge["a"], profile_edge["b"]
        span = math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    usable = span * float(spec.get("window_span_frac") or 0.72)
    layout = str(spec.get("window_layout") or "flank")
    if layout == "six_thirds":
        skip = 1 if (has_door and band == "lintel") or skip_center else None
        return _six_slot_laterals(usable, skip)
    pitch = max(120.0, float(spec.get("window_pitch_gu") or 260.0))
    count = max(1, int(usable // pitch))
    if count == 1:
        return (0.0,)
    step = usable / float(count)
    return tuple((index + 0.5 - count / 2.0) * step for index in range(count))


def _place_on_facade(
    *,
    spec: Mapping[str, Any],
    piece_bounds: Mapping[str, Any],
    facade_id: str,
    lateral_gu: float,
    pivot_z: float,
    protrusion_gu: float,
    profile_edge: Mapping[str, Any] | None = None,
    rotation_offset_rad: float = 0.0,
) -> tuple[list[float], list[float]]:
    ox, oy = _facade_outward(spec, facade_id)
    rotz = rotz_for_outward(ox, oy) + float(rotation_offset_rad)
    rotation = [0.0, 0.0, rotz]
    local_y_axis_world = engine_transform.rotate_reference_point((0.0, 1.0, 0.0), rotation)
    local_y_outward = local_y_axis_world[0] * ox + local_y_axis_world[1] * oy
    outer_y = float(piece_bounds["max"][1]) if local_y_outward > 0.0 else float(piece_bounds["min"][1])
    outward_rel = engine_transform.rotate_reference_point((0.0, outer_y, 0.0), rotation)
    wall_face, tangent_center, _span = _wall_face_and_tangent(spec, facade_id)
    face_row = (spec.get("wall_faces") or {}).get(facade_id) or {}
    if profile_edge is not None:
        a, b = profile_edge["a"], profile_edge["b"]
        ox, oy = map(float, profile_edge["outward"])
        rotation = [0.0, 0.0, rotz_for_outward(ox, oy) + float(rotation_offset_rad)]
        local_y_axis_world = engine_transform.rotate_reference_point((0.0, 1.0, 0.0), rotation)
        local_y_outward = local_y_axis_world[0] * ox + local_y_axis_world[1] * oy
        outer_y = float(piece_bounds["max"][1]) if local_y_outward > 0.0 else float(piece_bounds["min"][1])
        outward_rel = engine_transform.rotate_reference_point((0.0, outer_y, 0.0), rotation)
        tangent_x, tangent_y = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
        edge_len = math.hypot(tangent_x, tangent_y)
        if edge_len <= 1e-6:
            raise ValueError(f"degenerate measured edge {profile_edge}")
        tangent_x, tangent_y = tangent_x / edge_len, tangent_y / edge_len
        mid_x, mid_y = (float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0
        target_normal = mid_x * ox + mid_y * oy + protrusion_gu
        target_tangent = mid_x * tangent_x + mid_y * tangent_y + lateral_gu
        rel_normal = outward_rel[0] * ox + outward_rel[1] * oy
        rel_tangent = outward_rel[0] * tangent_x + outward_rel[1] * tangent_y
        return [ox * (target_normal - rel_normal) + tangent_x * (target_tangent - rel_tangent), oy * (target_normal - rel_normal) + tangent_y * (target_tangent - rel_tangent), float(pivot_z)], rotation
    if face_row.get("face_mode") == "normal":
        tangent = (-oy, ox)
        target_normal = wall_face + protrusion_gu
        target_tangent = tangent_center + lateral_gu
        rel_normal = outward_rel[0] * ox + outward_rel[1] * oy
        rel_tangent = outward_rel[0] * tangent[0] + outward_rel[1] * tangent[1]
        pos = [
            ox * (target_normal - rel_normal) + tangent[0] * (target_tangent - rel_tangent),
            oy * (target_normal - rel_normal) + tangent[1] * (target_tangent - rel_tangent),
            float(pivot_z),
        ]
        return pos, rotation
    pos = [0.0, 0.0, float(pivot_z)]
    if abs(ox) >= abs(oy):
        target = wall_face + ox * protrusion_gu
        pos[0] = target - outward_rel[0]
        pos[1] = tangent_center + lateral_gu
    else:
        target = wall_face + oy * protrusion_gu
        pos[1] = target - outward_rel[1]
        pos[0] = tangent_center + lateral_gu
    return pos, rotation


def _place_frame_on_door(
    *,
    door_id: str,
    frame_id: str,
    door_pos: Sequence[float],
    door_rotation: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Seat the frame at a fixed door-local offset (same on every house/facade)."""
    key = (door_id, frame_id)
    local = DOOR_FRAME_OFFSET_LOCAL.get(key)
    if local is None:
        raise ValueError(
            f"no door→frame offset for {key!r}; add to DOOR_FRAME_OFFSET_LOCAL"
        )
    delta = engine_transform.rotate_reference_point(local, door_rotation)
    pos = [
        float(door_pos[0]) + delta[0],
        float(door_pos[1]) + delta[1],
        float(door_pos[2]) + delta[2],
    ]
    return pos, [float(door_rotation[0]), float(door_rotation[1]), float(door_rotation[2])]


def _place_porch_at_door(
    *,
    door_pos: Sequence[float],
    door_rotation: Sequence[float],
    door_bounds: Mapping[str, Any],
    porch_bounds: Mapping[str, Any],
    porch_access_z_local: float,
) -> list[float]:
    """Seat a porch's rear/top edge at the door's outward edge.

    The measured porch meshes use native origins and extend outward along
    local Y. Matching porch max-Y to the door min-Y keeps the porch attached
    across every facade after applying the door's already-validated rotation;
    matching minimum Z to the door sill puts the steps on the same plinth.
    """
    local_delta = (
        0.0,
        float(door_bounds["min"][1]) - float(porch_bounds["max"][1]),
        float(door_bounds["min"][2]) - float(porch_access_z_local),
    )
    delta = engine_transform.rotate_reference_point(local_delta, door_rotation)
    return [
        float(door_pos[0]) + delta[0],
        float(door_pos[1]) + delta[1],
        float(door_pos[2]) + delta[2],
    ]


def _place_stair_at_door(
    *,
    door_pos: Sequence[float],
    door_rotation: Sequence[float],
    door_bounds: Mapping[str, Any],
    stair_bounds: Mapping[str, Any],
) -> list[float]:
    """Seat an exterior stair's upper landing at the door threshold."""
    local_delta = (
        0.0,
        float(door_bounds["min"][1]) - float(stair_bounds["max"][1]),
        float(door_bounds["min"][2]) - float(stair_bounds["max"][2]),
    )
    delta = engine_transform.rotate_reference_point(local_delta, door_rotation)
    return [
        float(door_pos[0]) + delta[0],
        float(door_pos[1]) + delta[1],
        float(door_pos[2]) + delta[2],
    ]


def _overlaps_door_opening(
    *,
    facade_id: str,
    window_pos: Sequence[float],
    window_rotation: Sequence[float],
    window_bounds: Mapping[str, Any],
    door_obstacles: Sequence[Mapping[str, Any]],
) -> bool:
    """Reject only windows whose actual footprint overlaps a door opening.

    Facade membership alone is not a blocker: a second-story window, or a
    laterally spaced window, may legitimately share a facade with a door.
    Compare the two placements in the door's tangent/vertical plane so the
    rule follows measured mesh bounds and works for both profile and legacy
    placement paths.
    """
    window_min = window_bounds["min"]
    window_max = window_bounds["max"]
    for obstacle in door_obstacles:
        if obstacle["facade"] != facade_id:
            continue
        door_rotation = obstacle["rotation"]
        tangent = engine_transform.rotate_reference_point((1.0, 0.0, 0.0), door_rotation)
        door_pos = obstacle["position"]
        door_bounds = obstacle["bounds"]
        door_min = door_bounds["min"]
        door_max = door_bounds["max"]
        door_center_t = sum(float(door_pos[i]) * float(tangent[i]) for i in range(3))
        window_center_t = sum(float(window_pos[i]) * float(tangent[i]) for i in range(3))
        door_t0 = door_center_t + float(door_min[0])
        door_t1 = door_center_t + float(door_max[0])
        window_t0 = window_center_t - 0.5 * float(window_max[0] - window_min[0])
        window_t1 = window_center_t + 0.5 * float(window_max[0] - window_min[0])
        door_z0 = float(door_pos[2]) + float(door_min[2])
        door_z1 = float(door_pos[2]) + float(door_max[2])
        window_z0 = float(window_pos[2]) + float(window_min[2])
        window_z1 = float(window_pos[2]) + float(window_max[2])
        if min(door_t1, window_t1) > max(door_t0, window_t0) and min(door_z1, window_z1) > max(door_z0, window_z0):
            return True
    return False


def _place_wall_chimney(
    *,
    spec: Mapping[str, Any],
    chimney_bounds: Mapping[str, Any],
    facade_id: str,
    lateral_gu: float,
    pivot_z: float,
) -> tuple[list[float], list[float]]:
    """Seat Chimney_01 with local +X against the wall; local -X sticks out."""
    ox, oy = _facade_outward(spec, facade_id)
    rotz = math.atan2(oy, -ox)
    rotation = [0.0, 0.0, rotz]
    back_rel = engine_transform.rotate_reference_point(
        (float(chimney_bounds["max"][0]), 0.0, 0.0), rotation
    )
    wall_face, tangent_center, _span = _wall_face_and_tangent(spec, facade_id)
    pos = [0.0, 0.0, float(pivot_z)]
    if abs(ox) >= abs(oy):
        wall_plane = wall_face - ox * CHIMNEY_WALL_EMBED_GU
        pos[0] = wall_plane - back_rel[0]
        pos[1] = tangent_center + lateral_gu
    else:
        wall_plane = wall_face - oy * CHIMNEY_WALL_EMBED_GU
        pos[1] = wall_plane - back_rel[1]
        pos[0] = tangent_center + lateral_gu
    return pos, rotation


def _member(
    *,
    source_id: str,
    object_id: str,
    model_key: str,
    offset_gu: Sequence[float] | list[float],
    rotation: Sequence[float],
    is_door: bool = False,
    structural_role: str | None = None,
    outward_heading_deg_value: float | None = None,
) -> dict[str, Any]:
    member: dict[str, Any] = {
        "source_id": source_id,
        "object_id": object_id,
        "model_key": model_key,
        "record_type": "DOOR" if is_door else "STAT",
        "category": "door" if is_door else "exterior",
        "is_door": is_door,
        "offset_gu": [round(float(v), 3) for v in offset_gu],
        "rotation": [round(float(v), 9) for v in rotation],
        "scale": 1.0,
        "structural_role": structural_role,
    }
    if outward_heading_deg_value is not None:
        member["outward_heading_deg"] = round(float(outward_heading_deg_value), 3)
    return member


def _window_pivot_z(
    band: str,
    *,
    plinth: float,
    door_bounds: Mapping[str, Any],
    window_bounds: Mapping[str, Any],
    story2_lift_gu: float,
) -> float:
    target_top = plinth + float(door_bounds["span"][2])
    lintel_z = target_top - WINDOW_LINTEL_BELOW_DOOR_GU - float(window_bounds["max"][2])
    if band == "story2":
        return lintel_z + story2_lift_gu
    return lintel_z


def _require_catalog_id(piece_id: str, catalog: Mapping[str, str], kind: str) -> str:
    if piece_id not in catalog:
        raise ValueError(f"unknown {kind} {piece_id!r}; known: {sorted(catalog)}")
    return piece_id


def generate_fk_house(
    shell_id: str,
    *,
    bounds_index: Mapping[str, Mapping[str, Any]] | None = None,
    generated_id: str | None = None,
    door_facades: Sequence[str] | None = None,
    secondary_door_facades: Sequence[str] | None = None,
    window_facades: Sequence[str] | None = None,
    window_model: str | None = None,
    door_model: str | None = None,
    door_frame: str | None = None,
    porch_facades: Sequence[str] | None = None,
    porch_model: str | None = None,
    stair_facades: Sequence[str] | None = None,
    stair_model: str | None = None,
    dormer_attachments: Sequence[Mapping[str, Any]] | None = None,
    use_wall_profiles: bool = False,
    allow_review_profiles: bool = False,
) -> dict[str, Any]:
    spec = SHELL_SPECS.get(shell_id)
    if spec is None:
        raise ValueError(f"unknown Falkreath shell {shell_id!r}; known: {sorted(SHELL_SPECS)}")
    index = dict(bounds_index or load_kit_bounds())
    profiles = None
    if use_wall_profiles:
        profiles = load_wall_profiles()
        profile_row = profiles.get("shells", {}).get(Path(str(spec["model_key"])).stem)
        if profile_row is None:
            raise ValueError(f"no wall profile for {spec['model_key']}")
        if profile_row.get("validation_state") != "accepted" and not allow_review_profiles:
            raise ValueError(
                f"wall profile {Path(str(spec['model_key'])).stem} is "
                f"{profile_row.get('validation_state')}; review it before production generation"
            )
    shell_key = spec["model_key"]
    if shell_key not in index:
        raise ValueError(f"no AABB for {shell_key}; measure the family _a mesh or alias it")
    shell_bounds = index[shell_key]
    door_id = _require_catalog_id(door_model or spec["door_model"], DOOR_MODELS, "door")
    window_id = _require_catalog_id(window_model or spec["window_model"], WINDOW_MODELS, "window")
    chimney_id = _require_catalog_id(spec["chimney_model"], CHIMNEY_MODELS, "chimney")
    door_bounds = index[door_id]
    window_bounds = index[window_id]
    chimney_bounds = index[chimney_id]
    allowed = _allowed_facades(spec)
    if door_facades is not None and secondary_door_facades is not None:
        raise ValueError("pass door_facades or secondary_door_facades, not both")
    if secondary_door_facades is not None:
        candidates = set(spec.get("secondary_door_facades") or ())
        requested_secondary = parse_facade_list(secondary_door_facades, default=(), allowed=allowed)
        invalid_secondary = [facade for facade in requested_secondary if facade not in candidates]
        if invalid_secondary:
            raise ValueError(
                f"secondary door facade(s) {invalid_secondary} are not valid for {shell_id}; "
                f"choose from {sorted(candidates)}"
            )
        door_facade_values = list(spec["default_door_facades"]) + requested_secondary
    else:
        door_facade_values = door_facades
    doors = parse_facade_list(door_facade_values, default=spec["default_door_facades"], allowed=allowed)
    if not doors:
        raise ValueError("at least one door facade is required")
    default_windows = spec.get("default_window_facades") or FACADE_IDS
    windows = parse_facade_list(window_facades, default=default_windows, allowed=allowed)
    door_set = set(doors)
    plinth = float(spec["plinth_top_gu"])
    chimney_facade = str(spec["chimney_facade"]) if spec.get("chimney_mode") == "wall" else ""
    member_n = 1
    first_door_pos: list[float] | None = None
    first_door_heading = 0.0
    first_door_source_id: str | None = None
    door_obstacles: list[dict[str, Any]] = []
    door_records: list[dict[str, Any]] = []
    member_bounds: list[dict[str, Any]] = []
    step_by_facade: dict[str, float] = {}

    def next_id(prefix: str) -> str:
        nonlocal member_n
        member_n += 1
        return f"fk_{prefix}_{member_n:04d}"

    members: list[dict[str, Any]] = [
        _member(
            source_id="fk_shell_0001",
            object_id=shell_id,
            model_key=shell_key,
            offset_gu=[0.0, 0.0, 0.0],
            rotation=[0.0, 0.0, 0.0],
            structural_role="shell",
        )
    ]
    member_bounds.append(shell_bounds)
    door_z = plinth - float(door_bounds["min"][2])
    frame_id = door_frame if door_frame is not None else spec.get("door_frame")
    if frame_id:
        frame_id = _require_catalog_id(str(frame_id), FRAME_MODELS, "door frame")
    frame_bounds = index[frame_id] if frame_id else None
    door_lateral = float(spec.get("door_lateral_gu") or 0.0)
    for facade in doors:
        ox, oy = _facade_outward(spec, facade)
        heading = outward_heading_deg(ox, oy)
        profile_edge = _profile_edge(spec, facade, door_z, profiles, allow_nearest=allow_review_profiles) if profiles is not None else None
        door_pos, door_rot = _place_on_facade(
            spec=spec,
            piece_bounds=door_bounds,
            facade_id=facade,
            lateral_gu=door_lateral,
            pivot_z=door_z,
            protrusion_gu=DOOR_PROTRUSION_GU,
            profile_edge=profile_edge,
            rotation_offset_rad=float(spec.get("door_rotation_offset_rad") or 0.0),
        )
        if frame_bounds is not None:
            frame_pos, frame_rot = _place_frame_on_door(
                door_id=door_id,
                frame_id=str(frame_id),
                door_pos=door_pos,
                door_rotation=door_rot,
            )
        recess = float(spec.get("door_recess_in_frame_gu") or DOOR_RECESS_IN_FRAME_GU)
        door_pos = [
            float(door_pos[0]) - float(ox) * recess,
            float(door_pos[1]) - float(oy) * recess,
            float(door_pos[2]),
        ]
        door_source_id = next_id("door")
        members.append(
            _member(
                source_id=door_source_id,
                object_id=door_id,
                model_key=KIT_MODELS[door_id],
                offset_gu=door_pos,
                rotation=door_rot,
                is_door=True,
                outward_heading_deg_value=heading,
            )
        )
        member_bounds.append(door_bounds)
        step_by_facade[facade] = -float(door_bounds["min"][2])
        if frame_bounds is not None:
            frame_source_id = next_id("frame")
            members.append(
                _member(
                    source_id=frame_source_id,
                    object_id=str(frame_id),
                    model_key=KIT_MODELS[str(frame_id)],
                    offset_gu=frame_pos,
                    rotation=frame_rot,
                )
            )
            member_bounds.append(frame_bounds)
        if first_door_pos is None:
            first_door_pos = door_pos
            first_door_heading = heading
            first_door_source_id = door_source_id
        door_obstacles.append(
            {
                "facade": facade,
                "position": door_pos,
                "rotation": door_rot,
                "bounds": door_bounds,
            }
        )
        door_records.append({"facade": facade, "position": door_pos, "rotation": door_rot, "source_id": door_source_id})

    if porch_facades is not None and porch_model is None:
        porch_model = "sky_FK_Porch_01a"
    porch_rows: list[dict[str, Any]] = []
    if porch_model is not None:
        porch_id = _require_catalog_id(str(porch_model), PORCH_MODELS, "porch")
        porch_bounds = index.get(PORCH_MODELS[porch_id])
        if porch_bounds is None:
            raise ValueError(f"no AABB for {PORCH_MODELS[porch_id]}; measure the porch mesh")
        selected_porches = parse_facade_list(
            porch_facades,
            default=tuple(spec["default_door_facades"]),
            allowed=allowed,
        )
        door_by_facade = {row["facade"]: row for row in door_records}
        missing_doors = [facade for facade in selected_porches if facade not in door_by_facade]
        if missing_doors:
            raise ValueError(
                f"porch facade(s) {missing_doors} have no generated door on {shell_id}; "
                "add the facade to door_facades or secondary_door_facades"
            )
        for facade in selected_porches:
            door_row = door_by_facade[facade]
            porch_pos = _place_porch_at_door(
                door_pos=door_row["position"],
                door_rotation=door_row["rotation"],
                door_bounds=door_bounds,
                porch_bounds=porch_bounds,
                porch_access_z_local=PORCH_ACCESS_Z_LOCAL[porch_id],
            )
            members.append(
                _member(
                    source_id=next_id("porch"),
                    object_id=porch_id,
                    model_key=PORCH_MODELS[porch_id],
                    offset_gu=porch_pos,
                    rotation=door_row["rotation"],
                    structural_role="porch",
                )
            )
            member_bounds.append(porch_bounds)
            step_by_facade[facade] = float(door_row["position"][2]) - (float(porch_pos[2]) + float(porch_bounds["min"][2]))
            porch_rows.append({"facade": facade, "model": porch_id})

    if stair_facades is not None and stair_model is None:
        stair_model = "sky_ex_mk_str_02"
    stair_rows: list[dict[str, Any]] = []
    if stair_model is not None:
        stair_id = _require_catalog_id(str(stair_model), STAIR_MODELS, "stair")
        stair_bounds = index.get(STAIR_MODELS[stair_id])
        if stair_bounds is None:
            raise ValueError(f"no AABB for {STAIR_MODELS[stair_id]}; measure the stair mesh")
        selected_stairs = parse_facade_list(
            stair_facades,
            default=tuple(spec["default_door_facades"]),
            allowed=allowed,
        )
        door_by_facade = {row["facade"]: row for row in door_records}
        missing_doors = [facade for facade in selected_stairs if facade not in door_by_facade]
        if missing_doors:
            raise ValueError(
                f"stair facade(s) {missing_doors} have no generated door on {shell_id}; "
                "add the facade to door_facades or secondary_door_facades"
            )
        integrated_porches = {row["facade"] for row in porch_rows}
        duplicate_access = [facade for facade in selected_stairs if facade in integrated_porches]
        if duplicate_access:
            raise ValueError(
                f"stair facade(s) {duplicate_access} already use a porch with integrated steps; "
                "omit the standalone stair or place it on another door"
            )
        for facade in selected_stairs:
            door_row = door_by_facade[facade]
            stair_pos = _place_stair_at_door(
                door_pos=door_row["position"],
                door_rotation=door_row["rotation"],
                door_bounds=door_bounds,
                stair_bounds=stair_bounds,
            )
            members.append(
                _member(
                    source_id=next_id("stair"),
                    object_id=stair_id,
                    model_key=STAIR_MODELS[stair_id],
                    offset_gu=stair_pos,
                    rotation=door_row["rotation"],
                    structural_role="stair",
                )
            )
            member_bounds.append(stair_bounds)
            step_by_facade[facade] = float(door_row["position"][2]) - (float(stair_pos[2]) + float(stair_bounds["min"][2]))
            stair_rows.append({"facade": facade, "model": stair_id})

    story2_lift = float(spec.get("story2_lift_gu") or STORY2_LIFT_GU)
    bands = tuple(spec.get("window_bands") or ("lintel",))
    band_facades = spec.get("window_facades_by_band") or {}
    for facade in windows:
        for band in bands:
            if band in band_facades and facade not in band_facades[band]:
                continue
            pivot_z = _window_pivot_z(
                str(band),
                plinth=plinth,
                door_bounds=door_bounds,
                window_bounds=window_bounds,
                story2_lift_gu=story2_lift,
            )
            profile_edge = _profile_edge(spec, facade, pivot_z, profiles, allow_nearest=allow_review_profiles) if profiles is not None else None
            laterals = _window_laterals(
                spec,
                facade,
                has_door=facade in door_set,
                band=str(band),
                skip_center=facade == chimney_facade,
                profile_edge=profile_edge,
            )
            for lateral in laterals:
                win_pos, win_rot = _place_on_facade(
                    spec=spec,
                    piece_bounds=window_bounds,
                    facade_id=facade,
                    lateral_gu=float(lateral),
                    pivot_z=pivot_z,
                    protrusion_gu=(
                        STORY2_PROTRUSION_GU if str(band) == "story2" else WINDOW_PROTRUSION_GU
                    ),
                    profile_edge=profile_edge,
                    rotation_offset_rad=float(
                        (spec.get("window_rotation_offset_by_band") or {}).get(str(band), 0.0)
                    ),
                )
                if _overlaps_door_opening(
                    facade_id=facade,
                    window_pos=win_pos,
                    window_rotation=win_rot,
                    window_bounds=window_bounds,
                    door_obstacles=door_obstacles,
                ):
                    continue
                members.append(
                    _member(
                        source_id=next_id("window"),
                        object_id=window_id,
                        model_key=KIT_MODELS[window_id],
                        offset_gu=win_pos,
                        rotation=win_rot,
                    )
                )
                member_bounds.append(window_bounds)

    dormer_rows: list[dict[str, Any]] = []
    for attachment_index, row in enumerate(dormer_attachments or ()):
        mode = str(row.get("mode") or "").strip()
        if mode not in {"roof_slope", "wall_bay"}:
            raise ValueError(
                f"dormer attachment {attachment_index} has unsupported mode {mode!r}; "
                "use 'roof_slope' or 'wall_bay'"
            )
        dormer_id = _require_catalog_id(str(row.get("model") or ""), DORMER_MODELS, "dormer")
        dormer_bounds = index.get(DORMER_MODELS[dormer_id])
        if dormer_bounds is None:
            raise ValueError(f"no AABB for {DORMER_MODELS[dormer_id]}; measure the dormer mesh")
        position = row.get("offset_gu")
        rotation = row.get("rotation")
        if not isinstance(position, Sequence) or len(position) != 3:
            raise ValueError(f"dormer attachment {attachment_index} requires offset_gu=[x,y,z]")
        if not isinstance(rotation, Sequence) or len(rotation) != 3:
            raise ValueError(f"dormer attachment {attachment_index} requires rotation=[rx,ry,rz]")
        dormer_pos = [float(value) for value in position]
        dormer_rot = [float(value) for value in rotation]
        members.append(
            _member(
                source_id=next_id("dormer"),
                object_id=dormer_id,
                model_key=DORMER_MODELS[dormer_id],
                offset_gu=dormer_pos,
                rotation=dormer_rot,
                structural_role="dormer",
            )
        )
        member_bounds.append(dormer_bounds)
        paired_window_id = row.get("window_model")
        if paired_window_id is not None:
            paired_window_id = _require_catalog_id(str(paired_window_id), KIT_MODELS, "window")
            paired_window_bounds = index.get(KIT_MODELS[paired_window_id])
            if paired_window_bounds is None:
                raise ValueError(f"no AABB for {KIT_MODELS[paired_window_id]}; measure the window mesh")
            paired_offset = row.get("window_offset_gu")
            paired_rotation = row.get("window_rotation")
            if not isinstance(paired_offset, Sequence) or len(paired_offset) != 3:
                raise ValueError(
                    f"dormer attachment {attachment_index} window requires window_offset_gu=[x,y,z]"
                )
            if not isinstance(paired_rotation, Sequence) or len(paired_rotation) != 3:
                raise ValueError(
                    f"dormer attachment {attachment_index} window requires window_rotation=[rx,ry,rz]"
                )
            window_delta = engine_transform.rotate_reference_point(
                tuple(float(value) for value in paired_offset), dormer_rot
            )
            paired_window_pos = [dormer_pos[i] + window_delta[i] for i in range(3)]
            # Window orientation is authored in the dormer's local frame, just
            # like window_offset_gu. Compose the dormer yaw so a roof-facing
            # dormer cannot leave its paired window looking through the back.
            paired_window_rot = [
                float(paired_rotation[0]),
                float(paired_rotation[1]),
                float(paired_rotation[2]) + dormer_rot[2],
            ]
            members.append(
                _member(
                    source_id=next_id("dormer_window"),
                    object_id=paired_window_id,
                    model_key=KIT_MODELS[paired_window_id],
                    offset_gu=paired_window_pos,
                    rotation=paired_window_rot,
                    structural_role="dormer_window",
                )
            )
            member_bounds.append(paired_window_bounds)
        dormer_rows.append(
            {
                "mode": mode,
                "model": dormer_id,
                "offset_gu": [round(value, 3) for value in dormer_pos],
                "rotation": [round(value, 6) for value in dormer_rot],
                "window_model": paired_window_id,
            }
        )

    span_x = float(shell_bounds["span"][0])
    span_y = float(shell_bounds["span"][1])
    if spec.get("chimney_mode") == "wall":
        chimney_z = float(shell_bounds["min"][2]) - float(chimney_bounds["min"][2])
        chimney_pos, chimney_rot = _place_wall_chimney(
            spec=spec,
            chimney_bounds=chimney_bounds,
            facade_id=str(spec["chimney_facade"]),
            lateral_gu=float(spec["chimney_lateral_gu"]),
            pivot_z=chimney_z,
        )
    else:
        along_axis = 0 if span_x >= span_y else 1
        across_axis = 1 - along_axis
        chimney_rotz = 0.0 if along_axis == 1 else math.pi / 2.0
        ridge_z = float(shell_bounds["max"][2])
        chimney_pos = [float(shell_bounds["center"][0]), float(shell_bounds["center"][1]), 0.0]
        chimney_pos[along_axis] = float(shell_bounds["center"][along_axis]) + float(spec["chimney_along_gu"])
        chimney_pos[across_axis] = float(shell_bounds["center"][across_axis]) + float(
            spec.get("chimney_across_gu") or 0.0
        )
        sink = float(spec.get("chimney_sink_gu") or CHIMNEY_RIDGE_SINK_GU)
        chimney_pos[2] = ridge_z - sink - float(chimney_bounds["min"][2])
        chimney_rot = [0.0, 0.0, chimney_rotz]
    members.append(
        _member(
            source_id=next_id("chimney"),
            object_id=chimney_id,
            model_key=KIT_MODELS[chimney_id],
            offset_gu=chimney_pos,
            rotation=chimney_rot,
        )
    )
    member_bounds.append(chimney_bounds)

    if first_door_pos is None:
        raise ValueError("generated house has no door")
    if first_door_source_id is None:
        raise ValueError("generated house has no anchor door source_id")
    if len(members) != len(member_bounds):
        raise ValueError(f"member/bounds length mismatch: {len(members)} vs {len(member_bounds)}")
    anchor = first_door_pos
    anchored = []
    for member in members:
        copied = dict(member)
        copied["offset_gu"] = [
            round(float(member["offset_gu"][0]) - anchor[0], 3),
            round(float(member["offset_gu"][1]) - anchor[1], 3),
            round(float(member["offset_gu"][2]) - anchor[2], 3),
        ]
        anchored.append(copied)
    # Real bounds via 8 transformed AABB corners per anchored member.
    world_corners: list[list[float]] = []
    xy_points: list[tuple[float, float]] = []
    rmin = [float("inf"), float("inf"), float("inf")]
    rmax = [float("-inf"), float("-inf"), float("-inf")]
    for member, b in zip(anchored, member_bounds):
        b_min = b["min"]
        b_max = b["max"]
        scale = float(member.get("scale", 1.0))
        rotation = member.get("rotation") or [0.0, 0.0, 0.0]
        offset = member["offset_gu"]
        for x in (float(b_min[0]), float(b_max[0])):
            for y in (float(b_min[1]), float(b_max[1])):
                for z in (float(b_min[2]), float(b_max[2])):
                    sx = x * scale
                    sy = y * scale
                    sz = z * scale
                    rx, ry, rz = engine_transform.rotate_reference_point((sx, sy, sz), rotation)
                    wx = rx + float(offset[0])
                    wy = ry + float(offset[1])
                    wz = rz + float(offset[2])
                    world_corners.append([wx, wy, wz])
                    xy_points.append((wx, wy))
                    if wx < rmin[0]:
                        rmin[0] = wx
                    if wy < rmin[1]:
                        rmin[1] = wy
                    if wz < rmin[2]:
                        rmin[2] = wz
                    if wx > rmax[0]:
                        rmax[0] = wx
                    if wy > rmax[1]:
                        rmax[1] = wy
                    if wz > rmax[2]:
                        rmax[2] = wz
    real_bounds = {
        "min": [float(rmin[0]), float(rmin[1]), float(rmin[2])],
        "max": [float(rmax[0]), float(rmax[1]), float(rmax[2])],
        "span": [float(rmax[0] - rmin[0]), float(rmax[1] - rmin[1]), float(rmax[2] - rmin[2])],
    }
    # Convex CCW XY hull of all corners.
    hull_poly = MultiPoint(xy_points).convex_hull
    if hull_poly.is_empty or hull_poly.area <= 0:
        raise ValueError(f"generated hull is empty or degenerate area={getattr(hull_poly, 'area', 0)}")
    exterior = list(hull_poly.exterior.coords)
    # Ensure CCW.
    if not hull_poly.exterior.is_ccw:
        exterior = list(reversed(exterior))
    # Drop duplicated closing point.
    if len(exterior) >= 2 and exterior[0] == exterior[-1]:
        exterior = exterior[:-1]
    hull_xy_rel: list[list[float]] = [[round(float(x), 3), round(float(y), 3)] for x, y in exterior]
    if len(hull_xy_rel) < 3:
        raise ValueError(f"generated hull has <3 points: {hull_xy_rel}")
    if hull_poly.area <= 0:
        raise ValueError(f"generated hull area <=0: {hull_poly.area}")
    # Sort anchored members (doors first, then source_id) and derive per-door step heights.
    sorted_members = sorted(anchored, key=lambda member: (0 if member.get("is_door") else 1, member["source_id"]))
    door_id_to_facade = {row["source_id"]: row["facade"] for row in door_records}
    door_step_heights: list[float] = []
    for mem in sorted_members:
        if not mem.get("is_door"):
            continue
        sid = str(mem.get("source_id"))
        facade = door_id_to_facade.get(sid)
        if facade is None:
            raise ValueError(f"door member {sid} has no facade mapping")
        step = step_by_facade.get(facade)
        if step is None:
            raise ValueError(f"no step height for facade {facade!r} (door {sid})")
        step_f = float(step)
        if not math.isfinite(step_f) or step_f < 0:
            raise ValueError(f"invalid step height {step_f} for facade {facade!r}")
        door_step_heights.append(step_f)
    if not door_step_heights:
        raise ValueError("no door step heights derived")
    burial_depth = max(0.0, -float(real_bounds["min"][2]))
    stamp_id = generated_id or f"generated__{shell_id}"
    # access_heading_rad: radial from hull centroid to seed door (0,0) — stamp contract.
    _c = hull_poly.centroid
    access_heading_rad = math.atan2(0.0 - _c.y, 0.0 - _c.x)
    return {
        "stamp_id": stamp_id,
        "source": {
            "kind": "generated_fk_kit",
            "shell_id": shell_id,
            "family": spec["family"],
            "plinth_top_gu": plinth,
            "door_model": door_id,
            "window_model": window_id,
            "door_facades": list(doors),
            "secondary_door_facades": [
                facade for facade in doors if facade not in spec["default_door_facades"]
            ],
            "window_facades": list(windows),
            "porch_model": porch_model,
            "porch_facades": [row["facade"] for row in porch_rows],
            "porch_count": len(porch_rows),
            "stair_model": stair_model,
            "stair_facades": [row["facade"] for row in stair_rows],
            "stair_count": len(stair_rows),
            "dormers": dormer_rows,
            "dormer_count": len(dormer_rows),
            "chimney_mode": spec.get("chimney_mode"),
            "wall_profile_mode": "review_preview" if profiles is not None and allow_review_profiles else ("accepted" if profiles is not None else "legacy"),
            "seed_door": first_door_source_id,
        },
        "building_type": "house",
        "size_class": "large" if span_x >= 1200 or span_y >= 1200 else "medium",
        "door_count": len(doors),
        "multi_shell": False,
        "anchor": {"kind": "seed_door", "source_position_gu": [0.0, 0.0, 0.0]},
        "access_heading_rad": access_heading_rad,
        "members": sorted_members,
        "footprint": {"aabb_rel": real_bounds, "hull_xy_rel": hull_xy_rel},
        "bounds_rel_gu": real_bounds,
        "terrain_envelope": {
            "door_step_heights_gu": door_step_heights,
            "footprint_relief_gu": burial_depth,
            "footprint_slope_deg": None,
            "burial_depth_gu": burial_depth,
        },
    }


__all__ = [
    "BOUNDS_PATH",
    "DORMER_BOUNDS_PATH",
    "REFERENCE_KIT_PATH",
    "DOOR_MODELS",
    "FACADE_IDS",
    "SHELL_SPECS",
    "WINDOW_MODELS",
    "PORCH_MODELS",
    "STAIR_MODELS",
    "DORMER_MODELS",
    "reference_shell_ids",
    "canonical_json_bytes",
    "fk_secondary_door_facades",
    "generate_fk_house",
    "load_kit_bounds",
    "parse_facade_list",
]
