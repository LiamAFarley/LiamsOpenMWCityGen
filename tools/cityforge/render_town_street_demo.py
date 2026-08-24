"""Render the Phase 21 pre-R3 street demo from the R2 ports checkpoint.

Preview only: arterials plus selective seam promotion over the fine R1 cells,
for visual review of block-size variety before R3 is specified.  The output
directory must be empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.checkpoint import read_checkpoint  # noqa: E402
from procgen.townlayout.street_demo import build_street_demo, render_street_demo  # noqa: E402
from procgen.townlayout.validate import TownLayoutError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the pre-R3 street demo")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print("FAILURE: street-demo output directory is not empty", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        product = read_checkpoint(args.input, expected_stages=("r1", "r2"))
        demo = build_street_demo(product)
        (out / "street_demo.json").write_text(
            json.dumps(demo, indent=1, sort_keys=True), encoding="utf-8")
        render_street_demo(product, demo, product["identities"]["survey"]["path"],
                           out / "street_demo.png")
    except (TownLayoutError, OSError, ValueError, KeyError) as exc:
        print(f"FAILURE: street demo {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
