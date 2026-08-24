"""Finalize Stage 06 from the frozen Stage 05 road-network JSON.

Inputs: ``streets.json`` from Stage 05 and a site survey for the diagnostic
background.  Output: a masterless Stage 06 JSON product plus a diagnostic PNG.
The input is never rewritten; fortification only adds the wall/gate/strip
contract and preserves all Stage 05 road arrays byte-for-byte in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.site_context import build_site_context, resolve_topdown_png  # noqa: E402
from procgen.townlayout.validate import TownLayoutError, validate_fortification_product  # noqa: E402
from procgen.townlayout.walls import build_walls_and_gates, write_walls_diagnostic  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 06 fortification")
    parser.add_argument("--input", required=True)
    parser.add_argument("--survey", required=True)
    parser.add_argument("--fields", required=True)
    parser.add_argument("--census", required=True)
    parser.add_argument("--brief", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        print("FAILURE: fortification out-dir not empty", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        product = json.loads(Path(args.input).read_text(encoding="utf-8"))
        brief = json.loads(Path(args.brief).read_text(encoding="utf-8"))
        ctx = build_site_context(survey_json=Path(args.survey), fields_npz=Path(args.fields),
                                census_json=Path(args.census), town_brief=brief)
        result = build_walls_and_gates(ctx, product, brief,
                                       approaches=product.get("approaches"),
                                       candidate_id=product.get("candidate_id", "c00"))
        _doc, validation_issues = validate_fortification_product(result)
        errors = [i for i in validation_issues if i.get("severity") == "error"]
        if errors:
            raise TownLayoutError(f"validation: {errors[0]['code']} {errors[0]['message']}")
        (out / "fortification.json").write_text(json.dumps(result, allow_nan=False) + "\n",
                                                  encoding="utf-8")
        topdown = resolve_topdown_png(Path(args.survey))
        if topdown is not None:
            survey = json.loads(Path(args.survey).read_text(encoding="utf-8"))
            write_walls_diagnostic(ctx, result, topdown_path=topdown, survey=survey,
                                   out_png=out / "fortification_diagnostic.png")
    except (TownLayoutError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAILURE: fortification {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
