#!/usr/bin/env python3
"""Mine a kit house grammar from a D-STAMP library.

Usage::

    python tools/cityforge/mine_house_grammar.py \\
        --library output/cityforge/stamps/markarth_side_stone_v2.json \\
        --kit-id stone \\
        --out configs/kits/stone/house_grammar_v1.json \\
        --date 2026-08-18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.kit_house_grammar import canonical_json_bytes, mine_grammar_from_library  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine house grammar from stamp library")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--kit-id", required=True)
    parser.add_argument("--grammar-id", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD provenance date")
    args = parser.parse_args()

    grammar = mine_grammar_from_library(
        args.library,
        kit_id=args.kit_id,
        grammar_id=args.grammar_id,
        mined_at=args.date,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json_bytes(grammar))
    print(f"wrote {args.out}")
    print(
        f"shells={grammar['stats']['shell_count']} "
        f"templates={grammar['stats']['stamp_template_count']} "
        f"eligible_templates={grammar['stats']['eligible_template_count']} "
        f"single_shell={grammar['stats']['single_shell_stamps']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
