#!/usr/bin/env python3
"""Generate a stamp-compatible exterior house from a kit grammar.

Usage::

    python tools/cityforge/generate_house.py \\
        --grammar configs/kits/stone/house_grammar_v1.json \\
        --library output/cityforge/stamps/markarth_side_stone_v2.json \\
        --shell mk_h_m_02 \\
        --seed 1 \\
        --out output/cityforge/stamps/generated/stone/house_seed0001.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.kit_house_grammar import canonical_json_bytes, generate_house  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate house stamp from grammar")
    parser.add_argument("--grammar", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--shell", default=None, help="shell_id from grammar")
    parser.add_argument("--stamp-template", default=None, help="stamp_template_id for multi-shell replay")
    parser.add_argument("--block-pattern", default=None, help="deprecated alias for --stamp-template")
    parser.add_argument("--door-slots", nargs="*", default=None, help="door slot ids (default first slot)")
    parser.add_argument("--no-windows", action="store_true")
    parser.add_argument("--no-chimney", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stairs", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("--terrain-at-door", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("--access-placement", choices=["synthetic", "mined"], default="synthetic")
    parser.add_argument(
        "--window-facades",
        default=None,
        help="comma-separated facade ids: pos_x,neg_x,pos_y,neg_y",
    )
    parser.add_argument("--stamp-id", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    grammar = json.loads(args.grammar.read_text(encoding="utf-8"))
    library = json.loads(args.library.read_text(encoding="utf-8"))
    shell_id = args.shell
    if shell_id is None and args.block_pattern is None:
        shell_id = grammar["shells"][0]["shell_id"]
    stamp = generate_house(
        grammar,
        library,
        shell_id=shell_id or grammar["shells"][0]["shell_id"],
        door_slot_ids=args.door_slots,
        include_windows=not args.no_windows,
        include_chimney=not args.no_chimney,
        stamp_template_id=args.stamp_template or args.block_pattern,
        block_pattern_id=args.block_pattern,
        generated_id=args.stamp_id,
        seed=args.seed,
        access_policy={
            "stairs": args.stairs,
            "terrain_at_door": args.terrain_at_door,
            "access_placement": args.access_placement,
        },
        window_facade_ids=(
            [part.strip() for part in args.window_facades.split(",") if part.strip()]
            if args.window_facades
            else None
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json_bytes(stamp))
    print(f"wrote {args.out} members={len(stamp['members'])} doors={stamp['door_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
