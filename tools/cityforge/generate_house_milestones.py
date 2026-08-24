#!/usr/bin/env python3
"""Generate the Phase-4 milestone batch of houses from a kit grammar."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.kit_house_grammar import canonical_json_bytes, generate_house  # noqa: E402

GRAMMAR = WORKSPACE / "configs" / "kits" / "stone" / "house_grammar_v1.json"
LIBRARY = WORKSPACE / "output" / "cityforge" / "stamps" / "markarth_side_stone_v2.json"
OUT_DIR = WORKSPACE / "output" / "cityforge" / "stamps" / "generated" / "stone" / "milestones"


def main() -> int:
    grammar = json.loads(GRAMMAR.read_text(encoding="utf-8"))
    library = json.loads(LIBRARY.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = [
        {"name": "m1_minimal", "shell_id": "sky_ex_mk_h_m_02", "seed": 1, "include_windows": False, "include_chimney": False, "access_policy": {"stairs": "on"}},
        {"name": "m2_full", "shell_id": "sky_ex_mk_h_m_02", "seed": 2, "include_windows": True, "include_chimney": True, "access_policy": {"stairs": "on"}},
        {
            "name": "m3_multi_door",
            "shell_id": "sky_ex_mk_h_m_02",
            "seed": 3,
            "door_slot_ids": ["door_0", "door_1"],
            "include_windows": True,
            "include_chimney": True,
            "access_policy": {"stairs": "on"},
        },
        {
            "name": "m4_block",
            "stamp_template_id": "template_1",
            "shell_id": "sky_ex_mk_h_m_05",
            "seed": 4,
            "access_policy": {"stairs": "on"},
        },
        {"name": "m5_large", "shell_id": "sky_ex_mk_h_m_05", "seed": 5, "door_slot_ids": ["door_0"], "include_windows": True, "include_chimney": True, "access_policy": {"stairs": "on"}},
        {"name": "m6_farm", "shell_id": "sky_ex_farm_h_02", "seed": 6, "include_windows": True, "include_chimney": True},
        {"name": "m7_reach", "shell_id": "sky_ex_rm_h_02", "seed": 7, "include_windows": True, "include_chimney": False},
        {"name": "m8_tavern", "shell_id": "sky_ex_mk_tv_m_01", "seed": 8, "include_windows": True, "include_chimney": True},
    ]

    manifest = []
    for spec in specs:
        stamp = generate_house(
            grammar,
            library,
            shell_id=spec["shell_id"],
            door_slot_ids=spec.get("door_slot_ids"),
            include_windows=spec.get("include_windows", True),
            include_chimney=spec.get("include_chimney", True),
            block_pattern_id=spec.get("block_pattern_id"),
            stamp_template_id=spec.get("stamp_template_id"),
            generated_id=f"generated__{spec['name']}",
            seed=spec["seed"],
            access_policy=spec.get("access_policy"),
            window_facade_ids=spec.get("window_facade_ids"),
        )
        stamp_path = OUT_DIR / f"{spec['name']}.json"
        stamp_path.write_bytes(canonical_json_bytes(stamp))
        sheet_path = OUT_DIR / f"{spec['name']}_sheet_2x3.png"
        render_cmd = [
            sys.executable,
            str(WORKSPACE / "tools" / "cityforge" / "render_generated_house.py"),
            "--stamp",
            str(stamp_path),
            "--out",
            str(sheet_path),
        ]
        result = subprocess.run(render_cmd, cwd=WORKSPACE, check=False)
        manifest.append(
            {
                "name": spec["name"],
                "stamp": stamp_path.as_posix(),
                "sheet": sheet_path.as_posix() if result.returncode == 0 else None,
                "render_exit": result.returncode,
                "members": len(stamp["members"]),
                "doors": stamp["door_count"],
            }
        )
        print(f"{spec['name']}: members={len(stamp['members'])} doors={stamp['door_count']} render={result.returncode}")

    (OUT_DIR / "manifest.json").write_bytes(canonical_json_bytes({"milestones": manifest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
