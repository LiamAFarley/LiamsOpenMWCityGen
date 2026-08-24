"""Build R6 courtyard/access reservations from wall-aware front rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.checkpoint import read_checkpoint, write_checkpoint  # noqa: E402
from procgen.townlayout.place import write_placement_diagnostic  # noqa: E402
from procgen.townlayout.row_access import reserve_row_access  # noqa: E402
from procgen.townlayout.site_context import build_site_context, resolve_topdown_png  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reserve R6 courtyard access")
    parser.add_argument("--input", required=True)
    parser.add_argument("--brief", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        print("FAILURE: R6 output directory is not empty", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = read_checkpoint(args.input, expected_stages=("r5_wall_front_rows",))
        product = reserve_row_access(source)
        product["preceding_checkpoint"] = str(Path(args.input).resolve())
        write_checkpoint(product, out_dir / "rows_access.json")
        survey_path = Path(source["identities"]["survey"]["path"])
        fields_path = Path(source["identities"]["fields"]["path"])
        census_path = Path(source["identities"]["census"]["path"])
        brief = json.loads(Path(args.brief).read_text(encoding="utf-8"))
        ctx = build_site_context(survey_json=survey_path, fields_npz=fields_path,
                                 census_json=census_path, town_brief=brief)
        topdown = resolve_topdown_png(survey_path)
        if topdown is None:
            raise RuntimeError("R6 terrain render source is missing")
        write_placement_diagnostic(
            ctx, product, topdown_path=topdown,
            survey=json.loads(survey_path.read_text(encoding="utf-8")),
            out_png=out_dir / "rows_access_terrain.png")
    except Exception as exc:
        print(f"FAILURE: R6 {exc}", file=sys.stderr)
        return 1
    print(json.dumps(product["row_access_metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

