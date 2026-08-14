"""Cityforge T1.1 D-PLAN validation CLI - strict city-plan gate.

Pipeline position
------------------
First executable gate of the Cityforge plan chain (after ``build_city_brief.py``
produces the accepted vocabulary bundle).  Consumes one declarative
``city_plan.json`` plus the accepted planner-input bundle and emits a
deterministic structured issue list.  Nothing here authors a plan, renders
an overlay, or runs placement - those are ``render_plan.py`` (T1.1) and the
T1.2+ solvers.

Usage
-----
    python tools/cityforge/validate_city_plan.py --plan <city_plan.json> [options]

Options (all default to the canonical accepted bundle):
    --site-survey       site_survey.json (canonical default)
    --kit-brief         kit_brief.json (canonical default)
    --region-palette    region_palette.json (canonical default)
    --stamp-libraries   one or more D-STAMP library JSONs (defaults to the
                        two canonical libraries)
    --centerlines       tamriel_aligned_centerlines_v1.json (canonical default;
                        loaded through procgen.aligned_roads; the source-space
                        bundle is refused)
    --out               validation result JSON (default: next to the plan,
                        named ``<plan>.validation.json``)
    --report            human-readable report path (optional; printed to
                        stdout when omitted)
    --emit-schema PATH  write the machine-readable JSON Schema (draft
                        2020-12) derived from the strict structural spec and
                        exit (no plan needed)

Exit codes
----------
0 = plan valid (no errors; warnings allowed)
1 = plan invalid (errors present) - no trusted validated-plan artifact
2 = configuration/bundle failure (missing input, corrupt accepted file)

Determinism
-----------
The validation JSON is byte-deterministic for identical inputs (canonical
serialization via ``censusio.write_deterministic``, issues sorted by
path/code/message).  Run twice and diff to verify.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen import cityplan  # noqa: E402
from procgen.censusio import write_deterministic  # noqa: E402

CANONICAL_SURVEY = "output/cityforge/sites/falkreath_v1/site_survey.json"
CANONICAL_BRIEF = "output/cityforge/briefs/falkreath_v1/kit_brief.json"
CANONICAL_PALETTE = "output/cityforge/briefs/falkreath_v1/region_palette.json"
CANONICAL_LIBRARIES = (
    "output/cityforge/stamps/karthgad_nord_v1.json",
    "output/cityforge/stamps/markarth_side_stone_v1.json",
)
CANONICAL_CENTERLINES = ("output/mapdata/roads/tamriel_aligned_centerlines_v1/"
                         "tamriel_aligned_centerlines_v1.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict Cityforge D-PLAN city_plan.json validator (T1.1)")
    parser.add_argument("--plan", required=False,
                        help="path to city_plan.json (not needed with --emit-schema)")
    parser.add_argument("--site-survey", default=CANONICAL_SURVEY)
    parser.add_argument("--kit-brief", default=CANONICAL_BRIEF)
    parser.add_argument("--region-palette", default=CANONICAL_PALETTE)
    parser.add_argument("--stamp-libraries", nargs="+", default=list(CANONICAL_LIBRARIES))
    parser.add_argument("--centerlines", default=CANONICAL_CENTERLINES)
    parser.add_argument("--out", default=None,
                        help="validation result JSON path (default: "
                             "<plan>.validation.json next to the plan)")
    parser.add_argument("--report", default=None,
                        help="human-readable report path (default: stdout)")
    parser.add_argument("--emit-schema", default=None, metavar="PATH",
                        help="write the derived JSON Schema and exit")
    return parser


def human_report(result: dict, plan_path: Path) -> str:
    lines = [
        f"Cityforge D-PLAN validation (T1.1)",
        f"plan: {plan_path}",
        f"plan_id: {result.get('plan_id')}",
        f"result: {'VALID' if result['valid'] else 'INVALID'}",
        f"issues: {result['issue_count']} "
        f"(errors {result['error_count']}, warnings {result['warning_count']})",
    ]
    summary = result.get("summary", {})
    if summary:
        sections = summary.get("sections", {})
        lines.append("sections: " + ", ".join(f"{k}={v}" for k, v in sections.items()))
        resolutions = summary.get("lot_resolution", [])
        if resolutions:
            by_mode: dict = {}
            for r in resolutions:
                by_mode.setdefault(r["mode"], 0)
                by_mode[r["mode"]] += 1
            lines.append("lot resolution: " + ", ".join(
                f"{k}={v}" for k, v in sorted(by_mode.items())))
        ext = summary.get("external_references", {})
        lines.append("external network: " + ", ".join(
            f"{k}={v}" for k, v in ext.items()))
        lines.append("warning codes: " + ", ".join(summary.get("warning_codes", [])))
    if result.get("issues"):
        lines.append("")
        lines.append("issues (sorted by path/code):")
        for issue in result["issues"]:
            lines.append(f"  [{issue['severity']}] {issue['code']} "
                         f"{issue['path']}: {issue['message']}")
    lines.append("")
    lines.append("input hashes (sha256):")
    for name, digest in sorted(result.get("input_hashes", {}).items()):
        lines.append(f"  {name}: {digest}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.emit_schema:
        schema = cityplan.emit_json_schema()
        out = Path(args.emit_schema)
        out.parent.mkdir(parents=True, exist_ok=True)
        digest = write_deterministic(out, schema)
        print(f"wrote schema {out} sha256={digest}")
        return 0

    if not args.plan:
        parser.error("--plan is required (or use --emit-schema)")
        return 2

    try:
        bundle = cityplan.Bundle.from_paths(
            site_survey=args.site_survey,
            kit_brief=args.kit_brief,
            region_palette=args.region_palette,
            stamp_libraries=args.stamp_libraries,
            centerlines=args.centerlines,
        )
        result = cityplan.validate_plan_file(args.plan, bundle)
    except cityplan.BundleError as exc:
        print(f"configuration failure: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"cannot read plan: {exc}", file=sys.stderr)
        return 2

    plan_path = Path(args.plan)
    out_path = Path(args.out) if args.out else \
        plan_path.with_name(plan_path.stem + ".validation.json")
    write_deterministic(out_path, result)

    report_text = human_report(result, plan_path)
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(report_text, encoding="utf-8")
    else:
        print(report_text, end="")

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
