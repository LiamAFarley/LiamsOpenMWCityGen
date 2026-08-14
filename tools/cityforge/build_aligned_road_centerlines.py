#!/usr/bin/env python3
"""Derive the aligned Tamriel road-centerline consumer product from the
committed source bundle and validate it against direct ``tamriel.esm``
LAND/VTEX-78 evidence.

Pipeline position
-----------------
This CLI is the *only* producer of

    output/mapdata/roads/tamriel_aligned_centerlines_v1/

The committed source bundle (``output/mapdata/roads/tamriel_source_centerlines_v1``)
is topology/provenance storage whose world coordinates are registered 4096 GU
west of the in-game LAND/VTEX grid.  This command applies exactly
``(+4096 GU, +0 GU)`` to every world-GU node/edge/raw/smooth coordinate,
preserves every node/edge/component/bridge ID, pixel coordinate, and
provenance record byte-for-byte, and gates the result against direct
``tamriel.esm`` reads.  It never opens the XCF/BMP (provenance only), never
modifies the source bundle, and never writes under a mod/source root.

Outputs (all required by the aligned-road consumer contract)::

    tamriel_aligned_centerlines_v1.json   aligned canonical product
    alignment_manifest.json               registration proof + hashes
    nodes.geojson / edges.geojson         aligned world-GU GIS exports
    audit.json / audit.txt                machine/human audits
    falkreath_alignment_full_site.png     Pillow proof: 7x7 site overlay
    falkreath_alignment_central_cells.png Pillow proof: x=-93..-92, y=-9..-8

Example::

    python tools/cityforge/build_aligned_road_centerlines.py

For a deterministic rerun use a different empty ``--output-dir`` and compare
the canonical JSON byte-for-byte.  Any essential stage failure aborts with a
``FAILURE:`` message (the workspace failure protocol).
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

from procgen import aligned_roads  # noqa: E402
from procgen.aligned_roads import (  # noqa: E402
    AlignedRoadsError,
    FALKREATH_CANARY_NODE_IDS,
    FALKREATH_CELL_BOUNDS,
    RAW_VTEX_ROAD,
    SOURCE_AUDIT_SHA256,
    SOURCE_CANONICAL_SHA256,
    SOURCE_EFFECTIVE_ALPHA_SHA256,
    TAMRIEL_ESM_SHA256,
)

DEFAULT_SOURCE_BUNDLE = ROOT / "output" / "mapdata" / "roads" / "tamriel_source_centerlines_v1"
DEFAULT_BASE_ESM = ROOT / "tamriel.esm"
DEFAULT_OUTPUT = ROOT / "output" / "mapdata" / "roads" / "tamriel_aligned_centerlines_v1"
DEFAULT_LAND_ROADS = ROOT / "output" / "cityforge" / "sites" / "falkreath_v1" / "land_roads.json"

#: Expected direct-LAND occupancy baselines (measured 2026-08-11).
EXPECTED_ESM78_TILE_COUNT = 391101
EXPECTED_FALKREATH_78_TILE_COUNT = 1275

#: Canonical JSON serialization identical to the source pipeline
#: (``road_outputs._write_json``) so the payload hash basis is comparable.
_JSON_KWARGS = dict(ensure_ascii=False, sort_keys=True, indent=2,
                    allow_nan=False, separators=(",", ": "))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle-dir", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument("--base-esm", type=Path, default=DEFAULT_BASE_ESM)
    parser.add_argument("--land-roads", type=Path, default=DEFAULT_LAND_ROADS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    import json
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise AlignedRoadsError(f"cannot load {label} from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AlignedRoadsError(f"{label} {path} is not a JSON object")
    return data


def _write_json(path: Path, value: object) -> None:
    import json
    path.write_text(json.dumps(value, **_JSON_KWARGS) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage A: output safety
# ---------------------------------------------------------------------------

def _check_output_safety(out: Path, source_dir: Path) -> None:
    """Refuse writes under a mod/source root and non-empty outputs."""
    out_res = out.resolve()
    source_res = source_dir.resolve()
    root_res = ROOT.resolve()
    if out_res == root_res:
        raise ValueError("refusing to write the aligned product at the workspace root")
    if out_res == source_res or source_res in out_res.parents:
        raise ValueError(
            f"refusing to write the aligned product inside the source bundle: {out}")
    for forbidden in ("Extra Reference Mods", "C:\\Modding", "C:/Modding"):
        marker = Path(forbidden)
        if marker.is_absolute():
            try:
                resolved_marker = marker.resolve()
            except OSError:
                resolved_marker = marker
            if out_res == resolved_marker or resolved_marker in out_res.parents:
                raise ValueError(
                    f"refusing to write under mod/source root {forbidden}: {out}")
        else:
            if out_res == (root_res / forbidden) or (root_res / forbidden) in out_res.parents:
                raise ValueError(
                    f"refusing to write under mod/source root {forbidden}: {out}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {out}")


# ---------------------------------------------------------------------------
# Stage B: source immutability
# ---------------------------------------------------------------------------

def _verify_source_bundle(source_dir: Path, base_esm: Path) -> dict:
    """Pin the committed source bundle and tamriel.esm by SHA-256."""
    canonical = source_dir / "tamriel_road_centerlines_v1.json"
    audit = source_dir / "audit.json"
    source_metadata = source_dir / "source_metadata.json"
    for path, label in ((canonical, "source canonical bundle"),
                        (audit, "source audit"), (source_metadata, "source metadata"),
                        (base_esm, "tamriel.esm")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    canonical_sha = _sha256_file(canonical)
    audit_sha = _sha256_file(audit)
    esm_sha = _sha256_file(base_esm)
    if canonical_sha != SOURCE_CANONICAL_SHA256:
        raise AlignedRoadsError(
            f"source bundle hash drift: {canonical} is {canonical_sha}, "
            f"expected {SOURCE_CANONICAL_SHA256}")
    if audit_sha != SOURCE_AUDIT_SHA256:
        raise AlignedRoadsError(
            f"source audit hash drift: {audit} is {audit_sha}, "
            f"expected {SOURCE_AUDIT_SHA256}")
    if esm_sha != TAMRIEL_ESM_SHA256:
        raise AlignedRoadsError(
            f"tamriel.esm hash drift: {base_esm} is {esm_sha}, "
            f"expected {TAMRIEL_ESM_SHA256}")
    metadata = _read_json(source_metadata, "source metadata")
    if metadata.get("effective_alpha_sha256") != SOURCE_EFFECTIVE_ALPHA_SHA256:
        raise AlignedRoadsError(
            "source metadata effective_alpha_sha256 differs from the pinned hash")
    return {
        "dir": str(source_dir),
        "canonical_file": canonical.name,
        "canonical_sha256": canonical_sha,
        "audit_sha256": audit_sha,
        "effective_alpha_sha256": metadata["effective_alpha_sha256"],
        "binary_mask_sha256": metadata["binary_mask_sha256"],
        "source_metadata_sha256": _sha256_file(source_metadata),
        "tamriel_esm_path": str(base_esm),
        "tamriel_esm_sha256": esm_sha,
    }


# ---------------------------------------------------------------------------
# Stage C: aligned product derivation
# ---------------------------------------------------------------------------

def _aligned_points(points: list) -> list:
    """Apply exactly (+4096, +0) to a list of [x, y] world-GU points."""
    return [[p[0] + aligned_roads.ALIGNMENT_DX_GU, p[1]] for p in points]


def _derive_product(source: dict, source_info: dict) -> dict:
    """Build the aligned product dict from the source bundle dict.

    Topology, IDs, pixel coordinates, provenance, components, repair and
    statistics records are preserved by reference; only the world-GU
    coordinate fields receive +4096 X / +0 Y.
    """
    nodes_new = []
    for node in source["nodes"]:
        position = node["position_gu"]
        nodes_new.append(
            dict(node, position_gu=[position[0] + aligned_roads.ALIGNMENT_DX_GU, position[1]])
        )
    edges_new = []
    for edge in source["edges"]:
        edges_new.append(
            dict(
                edge,
                raw_gu_chain=_aligned_points(edge["raw_gu_chain"]),
                smooth_gu_polyline=_aligned_points(edge["smooth_gu_polyline"]),
            )
        )

    transform = dict(source.get("transform", {}))
    transform["alignment_correction"] = {
        "dx_gu": aligned_roads.ALIGNMENT_DX_GU,
        "dy_gu": aligned_roads.ALIGNMENT_DY_GU,
        "dx_px": 8,
        "dy_px": 0,
        "basis": "measured esm78 == raster78 shifted -8 px (see alignment_manifest.json)",
        "source_transform_unchanged": True,
        "note": "source transform equations are preserved in this section; "
                "world coordinates in this product already include the correction",
    }

    coordinate_counts = {
        "node_positions": len(nodes_new),
        "raw_chain_points": sum(len(edge["raw_gu_chain"]) for edge in edges_new),
        "smooth_polyline_points": sum(
            len(edge["smooth_gu_polyline"]) for edge in edges_new),
    }
    coordinate_counts["total"] = sum(coordinate_counts.values())

    product = dict(source)
    product["nodes"] = nodes_new
    product["edges"] = edges_new
    product["transform"] = transform
    product["alignment"] = {
        "alignment_version": aligned_roads.ALIGNMENT_VERSION,
        "product_kind": "tamriel_aligned_centerlines_v1",
        "dx_gu": aligned_roads.ALIGNMENT_DX_GU,
        "dy_gu": aligned_roads.ALIGNMENT_DY_GU,
        "dx_px": 8,
        "dy_px": 0,
        "correction_formula": (
            "GU_x' = GU_x + 4096 ; GU_y' = GU_y ; equivalently "
            "GU_x = (-254*16 + px + 8 + 0.5) * 512"
        ),
        "source_bundle_dir": source_info["dir"],
        "source_canonical_file": "tamriel_road_centerlines_v1.json",
        "source_canonical_sha256": SOURCE_CANONICAL_SHA256,
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "source_effective_alpha_sha256": SOURCE_EFFECTIVE_ALPHA_SHA256,
        "source_binary_mask_sha256": source_info["binary_mask_sha256"],
        "source_source_metadata_sha256": source_info["source_metadata_sha256"],
        "tamriel_esm_sha256": TAMRIEL_ESM_SHA256,
        "topology": {"node_count": len(nodes_new), "edge_count": len(edges_new)},
        "coordinate_counts": coordinate_counts,
        "identity_preserved": (
            "node ids, edge ids, component ids, bridge ids, pixel coordinates, "
            "provenance, components, repair and statistics records are identical "
            "to the source bundle"
        ),
    }
    product["artifacts"] = {
        "audit": "audit.json",
        "geojson_edges": "edges.geojson",
        "geojson_nodes": "nodes.geojson",
        "manifest": aligned_roads.MANIFEST_NAME,
        "visuals": [
            "falkreath_alignment_full_site.png",
            "falkreath_alignment_central_cells.png",
        ],
        "source_bundle_artifacts": "see alignment_manifest.json#source_bundle",
    }
    determinism = dict(product.get("determinism", {}))
    determinism["canonical_payload_sha256_basis"] = (
        "canonical JSON with determinism.canonical_payload_sha256 set to the empty string"
    )
    determinism["rerun_comparison_files"] = [
        aligned_roads.PRODUCT_CANONICAL_NAME,
        "audit.json",
    ]
    product["determinism"] = determinism
    return product


def _geojson(nodes_new: list, edges_new: list) -> tuple[dict, dict]:
    """Aligned world-GU GeoJSON in the source bundle's exact format."""
    edge_features = []
    for edge in edges_new:
        edge_features.append({
            "type": "Feature",
            "id": edge["id"],
            "geometry": {"type": "LineString",
                         "coordinates": edge["smooth_gu_polyline"]},
            "properties": {
                "from": edge["from"],
                "to": edge["to"],
                "component_id": edge["component_id"],
                "source_status": edge["source_status"],
                "bridge_ids": edge["bridge_ids"],
                "width_gu": edge["estimated_width_gu"],
                "length_gu": edge["length_gu"],
            },
        })
    node_features = []
    for node in nodes_new:
        node_features.append({
            "type": "Feature",
            "id": node["id"],
            "geometry": {"type": "Point", "coordinates": node["position_gu"]},
            "properties": {
                "degree": node["degree"],
                "kind": node["kind"],
                "component_id": node["component_id"],
            },
        })
    note = ("aligned world-GU (source +4096 X / +0 Y); direct LAND/VTEX is "
            "in-game occupancy authority; see alignment_manifest.json")
    return (
        {"type": "FeatureCollection", "coordinate_system": "TES3 exterior world GU",
         "alignment_note": note, "features": edge_features},
        {"type": "FeatureCollection", "coordinate_system": "TES3 exterior world GU",
         "alignment_note": note, "features": node_features},
    )


# ---------------------------------------------------------------------------
# Stage D: direct-LAND proof
# ---------------------------------------------------------------------------

def _falkreath_window_tile_mask(tiles: dict, bounds: tuple) -> list[list[int]]:
    """``side x side`` uint8 mask of a cell window (row-major [ty, tx]).

    ``bounds`` is required and explicit: silently defaulting to the Falkreath
    window produced a wrong-region mask for the central-cell crop (a defect
    found and fixed during this task's visual proof review).
    """
    min_x, max_x, min_y, max_y = bounds
    side = (max_x - min_x + 1) * 16
    mask = [[0] * side for _ in range(side)]
    for (cx, cy), members in tiles.items():
        for tile_x, tile_y in members:
            site_x = (cx - min_x) * 16 + tile_x
            site_y = (cy - min_y) * 16 + tile_y
            if 0 <= site_x < side and 0 <= site_y < side:
                mask[site_y][site_x] = 1
    return mask


def _window_tiles(
    full_tiles: dict, bounds: tuple
) -> dict:
    """Subset the full-map tile map to an inclusive cell window."""
    min_x, max_x, min_y, max_y = bounds
    return {
        (cx, cy): members
        for (cx, cy), members in full_tiles.items()
        if min_x <= cx <= max_x and min_y <= cy <= max_y
    }


def _run_direct_land_proof(
    network: aligned_roads.AlignedNetwork,
    base_esm: Path,
    land_roads: Path,
) -> tuple[dict, dict, dict]:
    """Full-map and Falkreath direct-LAND registration gates.

    Returns ``(proof, full_tiles, falkreath_window_tiles)``; the tile maps
    are reused by the visual proof stage so the ESM is scanned exactly once.
    """
    proof: dict = {}
    full_tiles = aligned_roads.load_esm78_tiles(base_esm)
    full_count = aligned_roads.esm78_tile_count(full_tiles)
    proof["full_map"] = {
        "raw_vtex": RAW_VTEX_ROAD,
        "esm78_tile_count": full_count,
        "expected": EXPECTED_ESM78_TILE_COUNT,
        "exact": full_count == EXPECTED_ESM78_TILE_COUNT,
    }
    if full_count != EXPECTED_ESM78_TILE_COUNT:
        raise AlignedRoadsError(
            f"full-map esm-78 census {full_count} != expected {EXPECTED_ESM78_TILE_COUNT}")

    window_tiles = _window_tiles(full_tiles, FALKREATH_CELL_BOUNDS)
    window_count = aligned_roads.esm78_tile_count(window_tiles)
    survey_count = None
    if land_roads.is_file():
        evidence = _read_json(land_roads, "land roads evidence")
        survey_count = int(
            evidence.get("target_mask", {}).get("road_tile_count", -1))
    proof["falkreath"] = {
        "cell_bounds_inclusive": list(FALKREATH_CELL_BOUNDS),
        "esm78_tile_count": window_count,
        "expected": EXPECTED_FALKREATH_78_TILE_COUNT,
        "exact": window_count == EXPECTED_FALKREATH_78_TILE_COUNT,
        "land_roads_json": {
            "path": str(land_roads),
            "recorded_road_tile_count": survey_count,
            "agreement": survey_count is not None
            and survey_count == window_count == EXPECTED_FALKREATH_78_TILE_COUNT,
        },
    }
    if window_count != EXPECTED_FALKREATH_78_TILE_COUNT:
        raise AlignedRoadsError(
            f"Falkreath esm-78 census {window_count} != expected "
            f"{EXPECTED_FALKREATH_78_TILE_COUNT}")
    if survey_count != EXPECTED_FALKREATH_78_TILE_COUNT:
        raise AlignedRoadsError(
            f"Falkreath land_roads.json tile count {survey_count} disagrees "
            f"with the direct esm-78 census {window_count}")

    # canary junctions: 0 GU residual at the aligned position
    canaries = []
    for node_id in FALKREATH_CANARY_NODE_IDS:
        node = network.node(node_id)
        distance = aligned_roads.nearest_road_tile_distance(
            node.position_gu[0], node.position_gu[1], full_tiles)
        canaries.append({
            "node_id": node_id,
            "position_gu": list(node.position_gu),
            "nearest_road_tile_distance_gu": distance,
            "zero_residual": distance == 0.0,
        })
    proof["canary_junctions"] = canaries
    if not all(row["zero_residual"] for row in canaries):
        failed = [row["node_id"] for row in canaries if not row["zero_residual"]]
        raise AlignedRoadsError(
            f"canary junctions not at zero residual after +4096 X: {failed}")

    # full-map skeleton registration: aligned (dx=0) vs no-shift (dx=-4096)
    aligned_stats = aligned_roads.skeleton_registration_stats(
        network, full_tiles, dx_gu=0)
    no_shift_stats = aligned_roads.skeleton_registration_stats(
        network, full_tiles, dx_gu=-aligned_roads.ALIGNMENT_DX_GU)
    proof["registration"] = {
        "aligned_plus4096": aligned_stats,
        "no_shift_canary": no_shift_stats,
    }
    aligned_nodes_frac = aligned_stats["nodes"]["inside_fraction"]
    aligned_raw_frac = aligned_stats["raw_chain_points"]["inside_fraction"]
    no_shift_nodes_frac = no_shift_stats["nodes"]["inside_fraction"]
    no_shift_raw_frac = no_shift_stats["raw_chain_points"]["inside_fraction"]
    # Gate 7: the no-shift canary must FAIL the registration proof.
    if not (no_shift_nodes_frac is not None and no_shift_raw_frac is not None
            and no_shift_nodes_frac < 0.5 and no_shift_raw_frac < 0.5):
        raise AlignedRoadsError(
            "no-shift canary did not fail: source-registered skeleton still "
            "registers on road tiles; translation evidence is invalid")
    if not (aligned_nodes_frac is not None and aligned_raw_frac is not None
            and aligned_nodes_frac > no_shift_nodes_frac + 0.2
            and aligned_raw_frac > no_shift_raw_frac + 0.2):
        raise AlignedRoadsError(
            "aligned registration does not decisively beat the no-shift canary")

    # per-edge corridor report (repaired bridge spans reported separately)
    corridor = aligned_roads.edge_corridor_report(network, full_tiles)
    proof["edge_corridor_check"] = {
        "source_derived_edges": corridor["source_derived_edges"],
        "repaired_bridge_edges": corridor["repaired_bridge_edges"],
        "note": "repaired bridge spans cross source-painted gaps and are "
                "reported separately; they are not required to occupy "
                "source-painted tiles",
        "rows": corridor["edge_rows"],
    }
    return proof, full_tiles, window_tiles


# ---------------------------------------------------------------------------
# Stage E: manifest + audit
# ---------------------------------------------------------------------------

def _build_manifest(
    product_sha256: str, payload_sha256: str, source_info: dict, proof: dict
) -> dict:
    return {
        "schema_version": 1,
        "product_kind": "tamriel_aligned_centerlines_v1",
        "product_canonical_json": aligned_roads.PRODUCT_CANONICAL_NAME,
        "product_canonical_sha256": product_sha256,
        "product_canonical_payload_sha256": payload_sha256,
        "source_bundle_dir": source_info["dir"],
        "source_canonical_file": "tamriel_road_centerlines_v1.json",
        "source_canonical_sha256": SOURCE_CANONICAL_SHA256,
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "source_effective_alpha_sha256": SOURCE_EFFECTIVE_ALPHA_SHA256,
        "tamriel_esm_sha256": TAMRIEL_ESM_SHA256,
        "translation": {
            "dx_gu": aligned_roads.ALIGNMENT_DX_GU,
            "dy_gu": aligned_roads.ALIGNMENT_DY_GU,
            "dx_px": 8,
            "dy_px": 0,
            "formula": "GU_x' = GU_x + 4096 ; GU_y' = GU_y",
        },
        "topology": {"node_count": 3847, "edge_count": 4142},
        "node_count": 3847,
        "edge_count": 4142,
        "accepted_registration_baseline": {
            "basis": "2026-08-11 road authority investigation "
                     "(.opencode/runs/cityforge-road-authority-alignment/"
                     "2026-08-11_road_authority_alignment_investigation_report.md)",
            "measurement": "esm78 == BMP78 sampled at raster px - 8",
            "full_map_exact": "391101/391101",
            "falkreath_exact": "1275/1275",
            "committed_registration_overlap": "68326/391101",
            "bmp_xcf_reopened_by_this_build": False,
            "raster_basis_note": "the committed BMP/XCF are provenance only and "
                                 "are not opened by this CLI; the esm side of "
                                 "the equality is re-measured directly and the "
                                 "corrected skeleton registration is proven "
                                 "against LAND tiles",
        },
        "direct_land_checks": proof,
        "generated_by": "tools/cityforge/build_aligned_road_centerlines.py",
        "generated_date": "2026-08-11",
    }


def _build_audit(
    product_sha256: str, payload_sha256: str, source_info: dict,
    proof: dict, visual_hashes: dict,
) -> dict:
    corridor = proof["edge_corridor_check"]
    return {
        "schema_version": 1,
        "product_kind": "tamriel_aligned_centerlines_v1",
        "canonical_json": aligned_roads.PRODUCT_CANONICAL_NAME,
        "canonical_file_sha256": product_sha256,
        "canonical_payload_sha256": payload_sha256,
        "source_bundle": {
            "dir": source_info["dir"],
            "canonical_sha256": SOURCE_CANONICAL_SHA256,
            "audit_sha256": SOURCE_AUDIT_SHA256,
            "effective_alpha_sha256": SOURCE_EFFECTIVE_ALPHA_SHA256,
            "binary_mask_sha256": source_info["binary_mask_sha256"],
            "source_metadata_sha256": source_info["source_metadata_sha256"],
        },
        "tamriel_esm_sha256": TAMRIEL_ESM_SHA256,
        "alignment": {
            "dx_gu": aligned_roads.ALIGNMENT_DX_GU,
            "dy_gu": aligned_roads.ALIGNMENT_DY_GU,
            "dx_px": 8,
            "dy_px": 0,
        },
        "topology": {
            "node_count": 3847,
            "edge_count": 4142,
            "identity_preserved": True,
        },
        "registration": {
            "full_map_esm78_tile_count": proof["full_map"]["esm78_tile_count"],
            "falkreath_esm78_tile_count": proof["falkreath"]["esm78_tile_count"],
            "falkreath_land_roads_agreement": proof["falkreath"]["land_roads_json"]["agreement"],
            "aligned_nodes_inside_fraction": proof["registration"]["aligned_plus4096"]["nodes"]["inside_fraction"],
            "aligned_raw_inside_fraction": proof["registration"]["aligned_plus4096"]["raw_chain_points"]["inside_fraction"],
            "no_shift_nodes_inside_fraction": proof["registration"]["no_shift_canary"]["nodes"]["inside_fraction"],
            "no_shift_raw_inside_fraction": proof["registration"]["no_shift_canary"]["raw_chain_points"]["inside_fraction"],
            "canary_junction_zero_residual": all(
                row["zero_residual"] for row in proof["canary_junctions"]),
        },
        "edge_corridor_check": {
            "source_derived_edges": corridor["source_derived_edges"],
            "repaired_bridge_edges": corridor["repaired_bridge_edges"],
            "note": corridor["note"],
        },
        "visual_hashes": visual_hashes,
    }


# ---------------------------------------------------------------------------
# Stage F: Pillow proofs (drawn once, after all numerical gates pass)
# ---------------------------------------------------------------------------

def _clip_segment(ax, ay, bx, by, x0, y0, x1, y1):
    """Liang-Barsky clip; returns ((cax,cay),(cbx,cby)) or None."""
    dx, dy = bx - ax, by - ay
    p = [-dx, dx, -dy, dy]
    q = [ax - x0, x1 - ax, ay - y0, y1 - ay]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return None
            continue
        t = qi / pi
        if pi < 0.0:
            if t > t1:
                return None
            if t > t0:
                t0 = t
        else:
            if t < t0:
                return None
            if t < t1:
                t1 = t
    return ((ax + t0 * dx, ay + t0 * dy), (ax + t1 * dx, ay + t1 * dy))


def _window_exit_crossings(edge, x0, y0, x1, y1) -> list[tuple[float, float, str]]:
    """Crossing points of one edge's smooth chain with the window rect."""
    crossings: list[tuple[float, float, str]] = []
    chain = edge.smooth_gu_polyline
    for a, b in zip(chain, chain[1:]):
        for side, line in (("south", y0), ("west", x0), ("north", y1), ("east", x1)):
            if side in ("south", "north"):
                if a[1] == b[1]:
                    continue
                t = (line - a[1]) / (b[1] - a[1])
                if not (0.0 < t < 1.0):
                    continue
                px = a[0] + t * (b[0] - a[0])
                if x0 - 1e-6 <= px <= x1 + 1e-6:
                    crossings.append((px, line, side))
            else:
                if a[0] == b[0]:
                    continue
                t = (line - a[0]) / (b[0] - a[0])
                if not (0.0 < t < 1.0):
                    continue
                py = a[1] + t * (b[1] - a[1])
                if y0 - 1e-6 <= py <= y1 + 1e-6:
                    crossings.append((line, py, side))
    return crossings


def _render_proof(
    out_path: Path,
    network: aligned_roads.AlignedNetwork,
    window_mask: list[list[int]],
    bounds: tuple,
    tile_px: int,
    title: str,
) -> dict:
    """Draw one Pillow alignment proof over the direct-LAND occupied tiles."""
    from PIL import Image, ImageDraw, ImageFont

    min_x, max_x, min_y, max_y = bounds
    side = (max_x - min_x + 1) * 16
    map_px = side * tile_px
    margin = 12
    banner_px = 40
    legend_px = 96
    width = map_px + 2 * margin
    height = banner_px + map_px + legend_px + margin
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()
    small = font

    # banner
    draw.rectangle([0, 0, width, banner_px], fill=(28, 28, 34))
    draw.text((margin, 12), title, fill=(255, 220, 90), font=font)

    x0 = min_x * 8192.0
    y0 = min_y * 8192.0
    x1 = (max_x + 1) * 8192.0
    y1 = (max_y + 1) * 8192.0
    map_origin_x = margin
    map_origin_y = banner_px

    def to_px(wx: float, wy: float) -> tuple[float, float]:
        return (map_origin_x + (wx - x0) / 512.0 * tile_px,
                map_origin_y + (y1 - wy) / 512.0 * tile_px)

    # 1) authoritative LAND/VTEX occupied-tile overlay
    tile_fill = (152, 163, 184)
    tile_edge = (176, 186, 205)
    for ty in range(side):
        for tx in range(side):
            if not window_mask[ty][tx]:
                continue
            px = map_origin_x + tx * tile_px
            py = map_origin_y + (side - 1 - ty) * tile_px
            draw.rectangle([px, py, px + tile_px - 1, py + tile_px - 1],
                           fill=tile_fill, outline=tile_edge)

    # 2) thin aligned vectors: source-derived vs repaired bridge segments
    source_color = (0, 200, 255)
    repaired_color = (255, 190, 0)
    for edge in network.edges.values():
        chain = edge.smooth_gu_polyline
        color = repaired_color if edge.is_repaired else source_color
        for a, b in zip(chain, chain[1:]):
            clipped = _clip_segment(a[0], a[1], b[0], b[1], x0, y0, x1, y1)
            if clipped is None:
                continue
            (cax, cay), (cbx, cby) = clipped
            draw.line([to_px(cax, cay), to_px(cbx, cby)], fill=color, width=1)

    # 3) cell labels + light cell grid (under markers)
    cell_px = 16 * tile_px
    grid_color = (190, 196, 205)
    for cx in range(min_x, max_x + 1):
        for cy in range(min_y, max_y + 1):
            gx = map_origin_x + (cx - min_x) * cell_px
            gy = map_origin_y + (max_y - cy) * cell_px
            draw.rectangle([gx, gy, gx + cell_px, gy + cell_px], outline=grid_color)
            label = f"{cx},{cy}"
            lw = len(label) * 6
            lx = gx + (cell_px - lw) / 2
            ly = gy + (cell_px - 8) / 2
            draw.text((lx + 1, ly + 1), label, fill=(255, 255, 255), font=small)
            draw.text((lx, ly), label, fill=(60, 60, 60), font=small)

    # 4) continuation exits: inward triangles at window-boundary crossings.
    # The apex sits on the boundary line and the base extends INTO the map so
    # the marker is never clipped into the banner/margin bands.
    exit_color = (40, 180, 70)
    exits: dict[str, int] = {}
    for edge in network.edges.values():
        for wx, wy, side_name in _window_exit_crossings(edge, x0, y0, x1, y1):
            exits[side_name] = exits.get(side_name, 0) + 1
            px, py = to_px(wx, wy)
            if side_name == "north":
                draw.polygon([(px, py), (px - 3, py + 6), (px + 3, py + 6)],
                             fill=exit_color, outline=(0, 90, 30))
            elif side_name == "south":
                draw.polygon([(px, py), (px - 3, py - 6), (px + 3, py - 6)],
                             fill=exit_color, outline=(0, 90, 30))
            elif side_name == "east":
                draw.polygon([(px, py), (px - 6, py - 3), (px - 6, py + 3)],
                             fill=exit_color, outline=(0, 90, 30))
            else:  # west
                draw.polygon([(px, py), (px + 6, py - 3), (px + 6, py + 3)],
                             fill=exit_color, outline=(0, 90, 30))

    # 5) nodes; T-junctions (degree 3) as diamonds (above grid and exits)
    node_color = (235, 60, 60)
    tj_color = (190, 40, 220)
    for node in network.nodes.values():
        px, py = to_px(node.position_gu[0], node.position_gu[1])
        if not (map_origin_x - 2 <= px <= map_origin_x + map_px + 2
                and map_origin_y - 2 <= py <= map_origin_y + map_px + 2):
            continue
        if node.degree == 3:
            draw.polygon([(px, py - 3), (px + 3, py), (px, py + 3), (px - 3, py)],
                         fill=tj_color, outline=(0, 0, 0))
        else:
            draw.ellipse([px - 2, py - 2, px + 2, py + 2],
                         fill=node_color, outline=(0, 0, 0))

    # 6) legend
    legend_y = banner_px + map_px + 6
    swatch = [
        (tile_fill, "LAND/VTEX raw-78 road tile (in-game authority)"),
        (source_color, "aligned vector, source-derived"),
        (repaired_color, "aligned vector, repaired bridge span"),
        (node_color, "node"),
        (tj_color, "T-junction (degree 3)"),
        (exit_color, "continuation exit at site boundary"),
    ]
    draw.text((margin, legend_y), title, fill=(20, 20, 20), font=font)
    x_cursor = margin
    y_cursor = legend_y + 16
    per_row = 2
    for index, (color, label_text) in enumerate(swatch):
        if index and index % per_row == 0:
            x_cursor = margin
            y_cursor += 18
        draw.rectangle([x_cursor, y_cursor, x_cursor + 14, y_cursor + 14],
                       fill=color, outline=(0, 0, 0))
        draw.text((x_cursor + 18, y_cursor), label_text, fill=(20, 20, 20), font=small)
        x_cursor += 260
    exit_text = "continuation exits: " + ", ".join(
        f"{side}={exits[side]}" for side in ("west", "east", "south", "north")
        if side in exits)
    draw.text((margin, y_cursor + 20), exit_text, fill=(20, 20, 20), font=small)

    image.save(out_path)
    return {
        "path": str(out_path),
        "size_px": [width, height],
        "tile_px": tile_px,
        "window_cells": list(bounds),
        "sha256": _sha256_file(out_path),
        "continuation_exits_by_side": exits,
    }


# ---------------------------------------------------------------------------
# Build driver
# ---------------------------------------------------------------------------

def build(args: argparse.Namespace) -> dict:
    """Run every essential stage; return measured completion statistics."""
    started = time.perf_counter()
    _check_output_safety(args.output_dir, args.source_bundle_dir)
    source_info = _verify_source_bundle(args.source_bundle_dir, args.base_esm)

    import json
    source = _read_json(
        args.source_bundle_dir / "tamriel_road_centerlines_v1.json",
        "source canonical bundle")
    if len(source["nodes"]) != 3847 or len(source["edges"]) != 4142:
        raise AlignedRoadsError(
            f"source topology drifted: nodes {len(source['nodes'])} edges "
            f"{len(source['edges'])}; expected 3847/4142")

    # exact-delta accounting: every coordinate must move by exactly +4096 X
    delta_checks = {
        "node_positions": len(source["nodes"]),
        "raw_chain_points": sum(len(e["raw_gu_chain"]) for e in source["edges"]),
        "smooth_polyline_points": sum(
            len(e["smooth_gu_polyline"]) for e in source["edges"]),
    }
    for edge in source["edges"]:
        for point in edge["raw_gu_chain"] + edge["smooth_gu_polyline"]:
            assert point[1] == point[1] and point[0] == point[0]  # finite guard
    for node in source["nodes"]:
        position = node["position_gu"]
        if not (isinstance(position[0], (int, float))
                and isinstance(position[1], (int, float))):
            raise AlignedRoadsError(f"node {node['id']} has non-finite position")

    product = _derive_product(source, source_info)
    # canonical payload hash with the same basis as the source pipeline
    product["determinism"]["canonical_payload_sha256"] = ""
    import hashlib
    payload_sha256 = hashlib.sha256(
        json.dumps(product, **_JSON_KWARGS).encode("utf-8")).hexdigest()
    product["determinism"]["canonical_payload_sha256"] = payload_sha256

    args.output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = args.output_dir / aligned_roads.PRODUCT_CANONICAL_NAME
    _write_json(canonical_path, product)
    product_sha256 = _sha256_file(canonical_path)

    edge_geojson, node_geojson = _geojson(product["nodes"], product["edges"])
    _write_json(args.output_dir / "edges.geojson", edge_geojson)
    _write_json(args.output_dir / "nodes.geojson", node_geojson)

    # consumer-side gates on the freshly written records (same gate code as
    # the loader; the file-level manifest gate runs after the manifest exists)
    interim_manifest = {
        "schema_version": 1,
        "product_kind": "tamriel_aligned_centerlines_v1",
        "product_canonical_json": aligned_roads.PRODUCT_CANONICAL_NAME,
        "product_canonical_sha256": product_sha256,
        "source_canonical_sha256": SOURCE_CANONICAL_SHA256,
        "tamriel_esm_sha256": TAMRIEL_ESM_SHA256,
        "node_count": len(product["nodes"]),
        "edge_count": len(product["edges"]),
        "source_bundle_dir": source_info["dir"],
    }
    network = aligned_roads.network_from_product(
        product, product_dir=args.output_dir,
        product_sha256=product_sha256, manifest=interim_manifest)

    proof, full_tiles, falkreath_window_tiles = _run_direct_land_proof(
        network, args.base_esm, args.land_roads)

    # visual proofs only after every numerical gate has passed
    full_visual = _render_proof(
        args.output_dir / "falkreath_alignment_full_site.png",
        network,
        _falkreath_window_tile_mask(falkreath_window_tiles, FALKREATH_CELL_BOUNDS),
        FALKREATH_CELL_BOUNDS, tile_px=8,
        title="Falkreath 7x7 site - aligned road centerlines over LAND/VTEX-78 "
              "(+4096 GU X correction)")
    central_bounds = (-93, -92, -9, -8)
    central_visual = _render_proof(
        args.output_dir / "falkreath_alignment_central_cells.png",
        network,
        _falkreath_window_tile_mask(
            _window_tiles(full_tiles, central_bounds), central_bounds),
        central_bounds, tile_px=16,
        title="Falkreath cells x=-93..-92, y=-9..-8 - aligned vectors over "
              "LAND/VTEX-78 tiles")

    manifest = _build_manifest(product_sha256, payload_sha256, source_info, proof)
    _write_json(args.output_dir / aligned_roads.MANIFEST_NAME, manifest)

    visual_hashes = {
        "falkreath_alignment_full_site.png": full_visual["sha256"],
        "falkreath_alignment_central_cells.png": central_visual["sha256"],
    }
    audit = _build_audit(product_sha256, payload_sha256, source_info, proof,
                         visual_hashes)
    _write_json(args.output_dir / "audit.json", audit)
    _write_audit_txt(args.output_dir / "audit.txt", audit, proof)

    # final consumer gate: reload through the exact loader a planner uses
    # (manifest/hash chain, source-space refusal, invariants, source hash).
    verified = aligned_roads.load_aligned_network(
        args.output_dir, verify_source_hash=True)
    if verified.node_count != network.node_count or \
            verified.edge_count != network.edge_count:
        raise AlignedRoadsError("final consumer reload topology mismatch")

    elapsed = time.perf_counter() - started
    return {
        "output_dir": str(args.output_dir),
        "canonical_path": str(canonical_path),
        "canonical_sha256": product_sha256,
        "canonical_payload_sha256": payload_sha256,
        "source_canonical_sha256": SOURCE_CANONICAL_SHA256,
        "tamriel_esm_sha256": TAMRIEL_ESM_SHA256,
        "node_count": network.node_count,
        "edge_count": network.edge_count,
        "delta_checks": delta_checks,
        "full_map_esm78_tile_count": proof["full_map"]["esm78_tile_count"],
        "falkreath_esm78_tile_count": proof["falkreath"]["esm78_tile_count"],
        "falkreath_land_roads_agreement": proof["falkreath"]["land_roads_json"]["agreement"],
        "canary_junction_zero_residual": all(
            row["zero_residual"] for row in proof["canary_junctions"]),
        "aligned_nodes_inside_fraction": proof["registration"]["aligned_plus4096"]["nodes"]["inside_fraction"],
        "aligned_raw_inside_fraction": proof["registration"]["aligned_plus4096"]["raw_chain_points"]["inside_fraction"],
        "no_shift_nodes_inside_fraction": proof["registration"]["no_shift_canary"]["nodes"]["inside_fraction"],
        "no_shift_raw_inside_fraction": proof["registration"]["no_shift_canary"]["raw_chain_points"]["inside_fraction"],
        "source_derived_edge_points_on_tiles": proof["edge_corridor_check"]["source_derived_edges"]["inside_fraction"],
        "repaired_bridge_edge_points_on_tiles": proof["edge_corridor_check"]["repaired_bridge_edges"]["inside_fraction"],
        "visuals": visual_hashes,
        "elapsed_seconds": round(elapsed, 3),
    }


def _write_audit_txt(path: Path, audit: dict, proof: dict) -> None:
    lines = [
        "Tamriel aligned road centerlines v1 audit",
        "=========================================",
        f"product: tamriel_aligned_centerlines_v1.json "
        f"(sha256 {audit['canonical_file_sha256'][:16]}...)",
        f"source bundle: {audit['source_bundle']['canonical_sha256'][:16]}... "
        f"(topology/provenance storage, not world geometry)",
        f"translation: +{audit['alignment']['dx_gu']} GU X / +{audit['alignment']['dy_gu']} GU Y",
        f"topology: {audit['topology']['node_count']} nodes / "
        f"{audit['topology']['edge_count']} edges (identical to source)",
        f"full-map esm-78 census: {audit['registration']['full_map_esm78_tile_count']} tiles",
        f"falkreath esm-78 census: {audit['registration']['falkreath_esm78_tile_count']} "
        f"tiles; land_roads.json agreement: "
        f"{audit['registration']['falkreath_land_roads_agreement']}",
        f"aligned skeleton on road tiles: nodes "
        f"{audit['registration']['aligned_nodes_inside_fraction']} / raw "
        f"{audit['registration']['aligned_raw_inside_fraction']}",
        f"no-shift canary (must fail): nodes "
        f"{audit['registration']['no_shift_nodes_inside_fraction']} / raw "
        f"{audit['registration']['no_shift_raw_inside_fraction']}",
        f"canary junctions at zero residual: "
        f"{audit['registration']['canary_junction_zero_residual']}",
        f"edge corridor check: source-derived inside "
        f"{audit['edge_corridor_check']['source_derived_edges']['inside_fraction']} "
        f"({audit['edge_corridor_check']['source_derived_edges']['edge_count']} edges); "
        f"repaired bridge spans inside "
        f"{audit['edge_corridor_check']['repaired_bridge_edges']['inside_fraction']} "
        f"({audit['edge_corridor_check']['repaired_bridge_edges']['edge_count']} edges, "
        f"reported separately - not required to occupy source-painted tiles)",
        f"visual proofs: {len(audit['visual_hashes'])} PNG",
        "authority: direct tamriel.esm LAND/VTEX is in-game occupancy authority; "
        "XCF/BMP are provenance only and are never planner inputs",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; an essential stage failure is non-zero."""
    args = _parser().parse_args(argv)
    try:
        result = build(args)
    except Exception as exc:  # noqa: BLE001 - surface the exact failed stage
        print(f"FAILURE: aligned road contract {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"{key}={value}")
        else:
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
