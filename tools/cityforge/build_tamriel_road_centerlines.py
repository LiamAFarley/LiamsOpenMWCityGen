#!/usr/bin/env python3
"""Build the deterministic full-map Tamriel road-centerline bundle.

Pipeline position
------------------
This is the production CLI driver for the accepted road task::

    gimpformats XCF extraction + parity gate
        -> measured source endpoint profile and local repair
        -> repaired-mask skeleton graph
        -> bounded smooth vectors and TES3-GU transform
        -> canonical JSON, audit, GeoJSON/SVG, masks, and review images

The command reads only the supplied XCF/BMP/palette and writes only a new,
empty output directory.  It always runs the full 4992x3040 canvas; the
Falkreath products are a diagnostic crop, never a substitute for full-map
generation.  There is no random sampling: the ``--seed`` value is retained in
metadata as zero for reproducibility but is not consumed.

Example::

    python tools/cityforge/build_tamriel_road_centerlines.py

For a deterministic rerun, provide a different empty ``--output-dir`` and
byte-compare ``tamriel_road_centerlines_v1.json`` and ``audit.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.road_graph import build_skeleton_graph  # noqa: E402
from procgen.road_outputs import write_bundle  # noqa: E402
from procgen.road_repair import RepairSettings, repair_source_mask  # noqa: E402
from procgen.road_source import (  # noqa: E402
    compare_corrected_effective_png,
    extract_road_source,
    parse_ltex_palette,
)
from procgen.road_vectors import RoadTransform, VectorSettings, vectorize_graph  # noqa: E402


DEFAULT_XCF = ROOT / "Extra Reference Mods" / "Source Files-58155-3-0-1779931459" / "opensource" / "tesannwyn-vtex3.xcf"
DEFAULT_BMP = ROOT / "Extra Reference Mods" / "Source Files-58155-3-0-1779931459" / "opensource" / "tesannwyn-vtex3.bmp"
DEFAULT_PALETTE = ROOT / "Extra Reference Mods" / "Source Files-58155-3-0-1779931459" / "opensource" / "tes3ltex.txt"
DEFAULT_PARITY = Path(r"C:\Users\LiamF\AppData\Local\Temp\opencode\roads-from-xcf-corrected\road_network_effective_full.png")
DEFAULT_OUTPUT = ROOT / "output" / "mapdata" / "roads" / "tamriel_source_centerlines_v1"


def _parser() -> argparse.ArgumentParser:
    """Create the strict full-map CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-xcf", type=Path, default=DEFAULT_XCF)
    parser.add_argument("--source-bmp", type=Path, default=DEFAULT_BMP)
    parser.add_argument("--source-palette", type=Path, default=DEFAULT_PALETTE)
    parser.add_argument("--corrected-parity-png", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def build(args: argparse.Namespace) -> dict[str, object]:
    """Run every essential stage and return measured completion statistics."""

    for path, label in (
        (args.source_xcf, "source XCF"),
        (args.source_bmp, "source VTEX BMP"),
        (args.source_palette, "source LTEX palette"),
        (args.corrected_parity_png, "corrected parity PNG"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")

    started = time.perf_counter()
    source = extract_road_source(args.source_xcf)
    parity = compare_corrected_effective_png(source.effective_alpha, args.corrected_parity_png)
    # Parse the palette as a required source-evidence gate.  The renderer uses
    # stable class colors, while canonical source metadata preserves the exact
    # file hash and the semantic raw-1/raw-78 distinction.
    palette = parse_ltex_palette(args.source_palette)
    if len(palette) < 141:
        raise ValueError(f"LTEX palette has {len(palette)} rows; expected at least 141")
    source.metadata["palette_entry_count"] = len(palette)
    source.metadata["raw_vtex_labels"] = {
        "1": palette.get(1, {}),
        "78": palette.get(78, {}),
    }

    repair = repair_source_mask(source.binary_mask, RepairSettings())
    accepted = repair.bridge_ledger["accepted"]
    graph = build_skeleton_graph(
        repair.repaired_mask,
        bridge_owner=repair.bridge_owner,
        accepted_bridges=accepted,
    )
    transform = RoadTransform(
        canvas_width_px=source.width,
        canvas_height_px=source.height,
    )
    vectors = vectorize_graph(
        graph,
        repair.repaired_mask,
        transform=transform,
        settings=VectorSettings(),
    )
    bundle = write_bundle(
        args.output_dir,
        source=source,
        repair=repair,
        graph=graph,
        vectors=vectors,
        source_bmp=args.source_bmp,
        source_palette=args.source_palette,
        corrected_parity=parity,
    )
    elapsed = time.perf_counter() - started
    return {
        "output_dir": str(bundle.output_dir),
        "canonical_path": str(bundle.canonical_path),
        "audit_path": str(bundle.audit_path),
        "canonical_sha256": bundle.canonical_sha256,
        "audit_sha256": bundle.audit_sha256,
        "elapsed_seconds": round(elapsed, 3),
        "source_occupancy": int(bundle.statistics["source"]["effective_occupancy_px"]),
        "raw_components": int(bundle.statistics["source"]["component_count"]),
        "repaired_occupancy": int(bundle.statistics["repair"]["repaired_occupancy_px"]),
        "repaired_components": int(bundle.statistics["repair"]["component_count"]),
        "accepted_bridges": int(bundle.statistics["repair"]["accepted_bridge_count"]),
        "rejected_candidates": int(bundle.statistics["repair"]["rejected_candidate_count"]),
        "selected_max_gap_px": float(bundle.statistics["repair"]["selected_max_gap_px"]),
        "skeleton_pixels": int(bundle.statistics["graph"]["skeleton_pixels"]),
        "nodes": int(bundle.statistics["graph"]["node_count"]),
        "edges": int(bundle.statistics["graph"]["edge_count"]),
        "loops": int(bundle.statistics["graph"]["loop_edge_count"]),
        "falkreath_raw_components": int(bundle.statistics["falkreath"]["raw_local_component_count"]),
        "falkreath_repaired_components": int(bundle.statistics["falkreath"]["repaired_local_component_count"]),
        "max_deviation_px": float(bundle.statistics["vectors"]["max_edge_deviation_px"]),
        "max_symmetric_hausdorff_px": float(bundle.statistics["vectors"]["maximum_symmetric_hausdorff_px"]),
        "max_endpoint_displacement_gu": float(bundle.statistics["vectors"]["maximum_endpoint_displacement_gu"]),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; an essential stage failure is non-zero."""

    args = _parser().parse_args(argv)
    try:
        result = build(args)
    except Exception as exc:  # noqa: BLE001 - CLI must surface the exact failed stage.
        print(f"FAILURE: full-map road-centerline build {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
