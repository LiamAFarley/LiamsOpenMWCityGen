"""Build the multi-source terrain corpus for the Tamriel Reworked pipeline.

Purpose
    One-shot Stage A CLI: stream every landmass plugin listed in the config
    and emit the canonical corpus npz + human-readable manifest used by all
    later stages (seam atlas, solve, authoring verification).

Inputs
    --config JSON (default configs/tamriel_reworked_v1.json): ordered source
    specs (exactly one role=base), output paths, expected_counts_v1 baseline.

Outputs
    paths.corpus_npz, paths.corpus_manifest (relative paths resolve against
    the workspace root). Prints a per-source summary table and drift warnings
    if owner mods changed since the recorded baseline.

Pipeline position
    First stage of tamriel-reworked-heightmap; nothing else may run until its
    counts match the session-verified scan (see runs/ request.md).

Invariants
    Read-only over plugins; grid bounds derived from scans; fail-closed on
    missing sources or ambiguous roles.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen.terrainfield import (  # noqa: E402
    TerrainFieldError,
    build_corpus,
    load_config,
    save_corpus,
)


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tamriel_reworked_v1.json"))
    args = ap.parse_args()

    cfg_path = _resolve(ROOT, args.config)
    cfg = load_config(cfg_path)

    t0 = time.time()
    try:
        arrays, manifest = build_corpus(cfg)
    except TerrainFieldError as exc:
        print(f"FAILURE: corpus build refused: {exc}")
        return 1

    npz_path = _resolve(ROOT, cfg["paths"]["corpus_npz"])
    man_path = _resolve(ROOT, cfg["paths"]["corpus_manifest"])
    save_corpus(arrays, manifest, npz_path, man_path)

    print(f"grid cells x={manifest['grid']['cells_x']} y={manifest['grid']['cells_y']}")
    for name, info in manifest["sources"].items():
        peak = f"{info['peak_gu']:.0f}" if info["peak_gu"] is not None else "-"
        print(f"  {name:>4} ({info['role']:>5}): lands={info['lands']:>6} "
              f"bbox={info['bbox_xyxy']} peak={peak} GU @ {info['peak_cell']}")
    print(f"retained={manifest['retained_cells']} deleted={manifest['deleted_cells']} "
          f"seam_edges={manifest['seam_edges']} seam_tam_cells={manifest['seam_tam_cells']}")
    dup = manifest["duplicate_vertex_audit"]
    for key in ("tam_h", "oth_h"):
        s = dup.get(key, {})
        print(f"duplicate-vertex audit [{key}]: conflicts>0.5GU={s.get('conflicts')} "
              f"max_delta_gu={s.get('max_delta_gu', 0.0):.1f}")
    print(f"wrote {npz_path}")
    print(f"wrote {man_path}")
    print(f"elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
