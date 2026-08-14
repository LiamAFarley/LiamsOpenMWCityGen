"""Build one Cityforge T1.3 hard landscape product set.

Pipeline position
------------------
This is the production-shaped command wrapper for
``procgen.cityscape.build_cityscape``.  It consumes explicit accepted T1.1 /
T1.2 products, real source LAND, and the accepted remap ESP, then writes only
T1.3 host-side fields, diagnostics, and tes3conv JSON records.  It does not
author or copy an ESP and does not modify any original mod file.

Usage
-----
The canonical synthetic proof is normally run through
``build_cityscape_fixture.py``.  This command is useful for a caller that
wants a single output directory or explicit input paths; all defaults are
root-relative and point at the accepted Falkreath/T1.2 fixture.

Failure behavior
----------------
Any missing input, source seam/payload disagreement, VNML root failure, illegal
THU delta, VTEX/LTEX gate failure, tes3json mismatch, or T1.2 final-reseat
failure exits with ``FAILURE: cityscape ...``.  There is no degraded fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.cityscape import CityscapeError, CityscapePaths, build_cityscape, default_paths  # noqa: E402


def _paths(args: argparse.Namespace) -> CityscapePaths:
    defaults = default_paths(args.workspace_root, args.output_dir)
    replacements = {}
    for field in (
        "survey", "palette", "plan", "validation", "t12_placement", "t12_land_edits",
        "source_land", "effective_remap", "kit_brief", "centerlines",
    ):
        value = getattr(args, field)
        if value is not None:
            replacements[field] = Path(value).resolve()
    if args.stamp_library:
        replacements["stamp_libraries"] = tuple(Path(value).resolve() for value in args.stamp_library)
    return CityscapePaths(**{**defaults.__dict__, **replacements})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Cityforge Dispatch 6 T1.3 landscape records")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-dir", default=None)
    for name in (
        "survey", "palette", "plan", "validation", "t12-placement", "t12-land-edits",
        "source-land", "effective-remap", "kit-brief", "centerlines",
    ):
        parser.add_argument("--" + name, dest=name.replace("-", "_"), default=None)
    parser.add_argument("--stamp-library", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        paths = _paths(args)
        result = build_cityscape(paths)
        final = result["terrain"]["final"]
        paint = result["paint"].paint_ledger
        print(
            "cityscape PASS "
            f"cells={result['validation']['gates']['source_stitch']['cell_count']} "
            f"field={result['validation']['gates']['source_stitch']['field_shape']} "
            f"height_changed={int(np.count_nonzero(final.quantized_values_gu - result['terrain']['planned'].quantized_values_gu))} "
            f"max_delta_thu={final.final_encoding['max_abs_encoded_delta_thu']} "
            f"painted_tiles={paint['support_tile_count']} "
            f"t12_final={result['t12_evidence']['status']} "
            f"manifest={result['manifest_sha256']}"
        )
        return 0
    except Exception as exc:
        print(f"FAILURE: cityscape {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
