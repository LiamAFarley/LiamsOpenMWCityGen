"""Build Stage-B arterial-safe blocks from an accepted Stage-A checkpoint.

Input: one ``r2a_arterials`` JSON.  Output: deterministic
``road_blocks.json`` and same-extent topology/terrain visual-review PNGs.
The output directory must be empty; this CLI never rebuilds upstream stages.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from procgen.townlayout.checkpoint import read_checkpoint, write_checkpoint  # noqa: E402
from procgen.townlayout.road_blocks import build_road_blocks  # noqa: E402
from procgen.townlayout.road_review import render_road_blocks  # noqa: E402
from procgen.townlayout.validate import TownLayoutError  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build Stage-B town road blocks")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print("FAILURE: B output directory is not empty", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        source = read_checkpoint(args.input, expected_stages=("r2a_arterials",))
        product = build_road_blocks(source)
        product["preceding_checkpoint"] = str(Path(args.input).resolve())
        write_checkpoint(product, out / "road_blocks.json")
        survey_path = (product.get("identities") or {}).get("survey", {}).get("path")
        if not survey_path:
            raise TownLayoutError("B render: survey identity missing")
        render_road_blocks(product, survey_path,
                           out / "road_blocks_topology.png",
                           out / "road_blocks_terrain.png")
    except (TownLayoutError, OSError, ValueError, KeyError) as exc:
        print(f"FAILURE: B {exc}", file=sys.stderr)
        return 1
    metrics = product["metrics"]
    print(f"r2b_road_blocks: blocks={metrics['block_count']} "
          f"p10={metrics['p10_lot_equivalents']:.2f} "
          f"p50={metrics['p50_lot_equivalents']:.2f} "
          f"p90={metrics['p90_lot_equivalents']:.2f} "
          f"exceptions={metrics['exception_count']} runtime={metrics['runtime_s']:.2f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
