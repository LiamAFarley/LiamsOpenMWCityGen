"""Write canonical road-graph JSON, audits, map exports, and review images.

Pipeline position
------------------
This is the output/validation-facing stage after source extraction, measured
repair, graph tracing, and vector fitting::

    RoadSource + RepairResult + SkeletonGraph + VectorResult
        -> immutable masks and bridge ledger
        -> canonical versioned graph JSON
        -> GeoJSON/SVG/audit products
        -> full-map and Falkreath visual review images

The canonical JSON is the project-native world-coordinate graph.  All arrays
are sorted before serialization, IDs are content-derived upstream, and JSON is
written with one fixed formatting policy so a clean rerun can be byte-compared.
Rendering is intentionally separate from topology: colors and line widths do
not alter masks or graph records.  No source XCF/BMP/palette or existing output
is edited; only the caller-selected bundle directory is authored.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .road_graph import SkeletonGraph
from .road_repair import RepairResult, component_labels
from .road_source import RoadSource, read_vtex_canvas, sha256_bytes, sha256_file
from .road_vectors import RoadTransform, VectorResult


FALKREATH_CELL_BOUNDS = (-95, -89, -11, -5)
FALKREATH_WINDOW = {"x0": 2544, "y0": 1024, "width": 112, "height": 112}


@dataclass(frozen=True)
class BundleResult:
    """Paths and canonical/audit data written by :func:`write_bundle`."""

    output_dir: Path
    canonical_path: Path
    audit_path: Path
    canonical_sha256: str
    audit_sha256: str
    statistics: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    """Serialize JSON with the fixed canonical formatting contract."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    """Write a deterministic JSON document and return its file digest."""

    data = _json_bytes(value)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_png(path: Path, array: np.ndarray, mode: str) -> None:
    """Write a deterministic PIL image from a copied NumPy array."""

    Image.fromarray(np.ascontiguousarray(array), mode=mode).save(path)


def _upscale_nearest(image: Image.Image, scale: int = 8) -> Image.Image:
    """Nearest-neighbour scale used for pixel-faithful Falkreath crops."""

    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _draw_polyline(draw: ImageDraw.ImageDraw, points: Sequence[Sequence[float]], *, fill: Any, width: int = 1, offset: tuple[float, float] = (0.0, 0.0)) -> None:
    """Draw a finite coordinate polyline with a stable integer rasterization."""

    if len(points) < 2:
        return
    ox, oy = offset
    coords = [(int(round(float(point[0]) - ox)), int(round(float(point[1]) - oy))) for point in points]
    draw.line(coords, fill=fill, width=width, joint="curve")


def _class_color(raw_value: int) -> tuple[int, int, int]:
    """Stable review color for a raw VTEX class, keeping Sand distinct."""

    fixed = {
        0: (28, 47, 77),
        1: (194, 178, 128),  # Sand; never used as road authority.
        3: (150, 110, 70),  # Road Dirt, mostly absent from the source road layer.
        33: (82, 155, 84),
        78: (200, 124, 43),  # MA_sulphur_rock02 source road paint.
        92: (112, 142, 92),
        94: (57, 86, 137),
        128: (192, 214, 235),
        137: (211, 226, 244),
    }
    if raw_value in fixed:
        return fixed[raw_value]
    digest = hashlib.sha256(str(int(raw_value)).encode("ascii")).digest()
    return (80 + digest[0] % 130, 80 + digest[1] % 130, 80 + digest[2] % 130)


def _texture_rgb(vtex: np.ndarray) -> np.ndarray:
    """Convert a raw VTEX canvas into a readable diagnostic RGB raster."""

    result = np.zeros((*vtex.shape, 3), dtype=np.uint8)
    for raw_value in np.unique(vtex):
        result[vtex == raw_value] = _class_color(int(raw_value))
    return result


def _falkreath_crop(array: np.ndarray) -> np.ndarray:
    """Extract the fixed full 7x7-cell Falkreath pixel window."""

    window = FALKREATH_WINDOW
    return np.asarray(
        array[
            window["y0"] : window["y0"] + window["height"],
            window["x0"] : window["x0"] + window["width"],
        ]
    )


def _component_window_statistics(repair: RepairResult) -> dict[str, Any]:
    """Measure raw/repaired local topology without relabeling global IDs away."""

    x0, y0 = FALKREATH_WINDOW["x0"], FALKREATH_WINDOW["y0"]
    x1, y1 = x0 + FALKREATH_WINDOW["width"], y0 + FALKREATH_WINDOW["height"]
    raw_crop = repair.source_mask[y0:y1, x0:x1] > 0
    repaired_crop = repair.repaired_mask[y0:y1, x0:x1] > 0
    raw_local_labels, raw_local_count = component_labels(raw_crop, connectivity=8)
    repaired_local_labels, repaired_local_count = component_labels(repaired_crop, connectivity=8)
    raw_global = sorted({int(value) for value in repair.source_component_labels[y0:y1, x0:x1].ravel() if value})
    repaired_global = sorted({int(value) for value in repair.repaired_component_labels[y0:y1, x0:x1].ravel() if value})
    raw_id_map = repair.metadata.get("source_component_id_by_label", {})
    repaired_id_map = repair.metadata.get("repaired_component_id_by_label", {})
    return {
        "cell_bounds_inclusive": {
            "min_x": FALKREATH_CELL_BOUNDS[0],
            "max_x": FALKREATH_CELL_BOUNDS[1],
            "min_y": FALKREATH_CELL_BOUNDS[2],
            "max_y": FALKREATH_CELL_BOUNDS[3],
        },
        "canvas_window_px": dict(FALKREATH_WINDOW),
        "raw_road_pixels": int(np.count_nonzero(raw_crop)),
        "repaired_road_pixels": int(np.count_nonzero(repaired_crop)),
        "bridge_pixels": int(np.count_nonzero(repair.bridge_mask[y0:y1, x0:x1])),
        "raw_local_component_count": int(raw_local_count),
        "repaired_local_component_count": int(repaired_local_count),
        "raw_global_component_ids_intersecting": [str(raw_id_map.get(str(value), f"raw_label_{value}")) for value in raw_global],
        "repaired_global_component_ids_intersecting": [
            str(repaired_id_map.get(str(value), f"raw_label_{value}")) for value in repaired_global
        ],
        "raw_local_component_sizes": [
            int(np.count_nonzero(raw_local_labels == value)) for value in range(1, int(raw_local_count) + 1)
        ],
        "repaired_local_component_sizes": [
            int(np.count_nonzero(repaired_local_labels == value)) for value in range(1, int(repaired_local_count) + 1)
        ],
    }


def _render_images(
    output_dir: Path,
    source: RoadSource,
    repair: RepairResult,
    graph: SkeletonGraph,
    vectors: VectorResult,
    vtex: np.ndarray,
) -> dict[str, str]:
    """Render all required full-map and Falkreath visual review products."""

    height, width = source.binary_mask.shape
    source_alpha = np.asarray(source.effective_alpha, dtype=np.uint8)
    repaired_binary = (np.asarray(repair.repaired_mask) > 0).astype(np.uint8) * 255
    bridge_binary = (np.asarray(repair.bridge_mask) > 0).astype(np.uint8) * 255
    _write_png(output_dir / "full_source_mask.png", source_alpha, "L")
    _write_png(output_dir / "full_repaired_mask.png", repaired_binary, "L")
    _write_png(output_dir / "full_bridge_mask.png", bridge_binary, "L")

    center = Image.new("RGB", (width, height), (8, 12, 20))
    center_draw = ImageDraw.Draw(center)
    overlay = Image.fromarray(np.repeat(source_alpha[:, :, None], 3, axis=2), mode="RGB")
    overlay_draw = ImageDraw.Draw(overlay)
    for edge in vectors.edges:
        color = (250, 223, 70) if edge.get("bridge_ids") else (70, 218, 255)
        _draw_polyline(center_draw, edge["smooth_pixel_polyline"], fill=color, width=2)
        _draw_polyline(overlay_draw, edge["smooth_pixel_polyline"], fill=(245, 75, 25), width=2)
    center.save(output_dir / "full_centerlines.png")
    overlay.save(output_dir / "full_centerlines_over_source.png")

    window = FALKREATH_WINDOW
    sx, sy = window["x0"], window["y0"]
    ex, ey = sx + window["width"], sy + window["height"]
    source_crop = source_alpha[sy:ey, sx:ex]
    source_image = _upscale_nearest(Image.fromarray(source_crop, mode="L"))
    source_image.save(output_dir / "falkreath_source_mask_8x.png")

    # Repair bridge view: the original source is gray, the repaired corridor
    # is a restrained light-gray fill, and bridge pixels are unmistakable red.
    repair_rgb = np.zeros((window["height"], window["width"], 3), dtype=np.uint8)
    repair_rgb[:] = (16, 19, 27)
    local_source = repair.source_mask[sy:ey, sx:ex] > 0
    local_repaired = repair.repaired_mask[sy:ey, sx:ex] > 0
    local_bridge = repair.bridge_mask[sy:ey, sx:ex] > 0
    repair_rgb[local_repaired] = (132, 142, 157)
    repair_rgb[local_source] = (215, 219, 225)
    repair_rgb[local_bridge] = (244, 56, 62)
    _upscale_nearest(Image.fromarray(repair_rgb, mode="RGB")).save(output_dir / "falkreath_repair_bridges_8x.png")

    # Centerline review images are rendered directly at 8x rather than
    # drawing at 112x112 and nearest-upscaling.  That keeps the measured raster
    # products pixel-faithful while letting the vector products show genuinely
    # smooth bends instead of an enlarged staircase.
    falk_center = Image.new("RGB", (window["width"] * 8, window["height"] * 8), (9, 12, 20))
    falk_draw = ImageDraw.Draw(falk_center)
    for edge in vectors.edges:
        points = [
            (float(point[0] - sx) * 8.0 + 4.0, float(point[1] - sy) * 8.0 + 4.0)
            for point in edge["smooth_pixel_polyline"]
        ]
        if len(points) >= 2:
            falk_draw.line(
                points,
                fill=(250, 222, 71) if edge.get("bridge_ids") else (58, 219, 255),
                width=5,
                joint="curve",
            )
    for node in vectors.nodes:
        nx, ny = node["position_px"]
        if sx <= nx < ex and sy <= ny < ey:
            color = (255, 75, 75) if node["kind"] == "junction" else (255, 235, 80)
            px, py = (nx - sx) * 8 + 4, (ny - sy) * 8 + 4
            falk_draw.ellipse((px - 10, py - 10, px + 10, py + 10), fill=color)
    falk_center.save(output_dir / "falkreath_centerlines_8x.png")

    texture_rgb = _texture_rgb(vtex)
    texture_crop = Image.fromarray(texture_rgb[sy:ey, sx:ex], mode="RGB").convert("RGBA").resize(
        (window["width"] * 8, window["height"] * 8), Image.Resampling.NEAREST
    )
    texture_draw = ImageDraw.Draw(texture_crop, "RGBA")
    # A translucent centerline preserves the sulphur-rock and Sand classes
    # beneath it; the source corridor itself is not reclassified here.
    for edge in vectors.edges:
        points = [
            (float(point[0] - sx) * 8.0 + 4.0, float(point[1] - sy) * 8.0 + 4.0)
            for point in edge["smooth_pixel_polyline"]
        ]
        if len(points) >= 2:
            texture_draw.line(points, fill=(250, 250, 250, 225), width=5, joint="curve")
    texture_crop.convert("RGB").save(output_dir / "falkreath_centerlines_over_texture_8x.png")

    # Component/junction diagnostic with an actual text legend.  Component
    # colors are derived from IDs so reruns never reshuffle the palette.
    diagnostic = Image.new("RGB", (window["width"] * 8, window["height"] * 8 + 132), (12, 14, 20))
    diagnostic_draw = ImageDraw.Draw(diagnostic)
    labels = repair.repaired_component_labels[sy:ey, sx:ex]
    for label_value in sorted(int(value) for value in np.unique(labels) if value):
        digest = hashlib.sha256(str(label_value).encode("ascii")).digest()
        color = (60 + digest[0] % 150, 60 + digest[1] % 150, 60 + digest[2] % 150)
        mask = labels == label_value
        yy, xx = np.nonzero(mask)
        for x, y in zip(xx, yy):
            diagnostic_draw.rectangle((int(x) * 8, int(y) * 8, int(x) * 8 + 7, int(y) * 8 + 7), fill=color)
    # Draw geometry directly at 8x so component blocks and topology witnesses
    # are not obscured by a second, unscaled raster pass.
    for edge in vectors.edges:
        points = [
            (int(round(float(point[0]) - sx)) * 8 + 4, int(round(float(point[1]) - sy)) * 8 + 4)
            for point in edge["smooth_pixel_polyline"]
        ]
        if len(points) >= 2:
            diagnostic_draw.line(points, fill=(255, 66, 66) if edge.get("bridge_ids") else (245, 245, 245), width=3, joint="curve")
    node_index = 0
    for node in vectors.nodes:
        nx, ny = node["position_px"]
        if sx <= nx < ex and sy <= ny < ey:
            node_index += 1
            px, py = (nx - sx) * 8 + 4, (ny - sy) * 8 + 4
            color = (255, 54, 54) if node["kind"] == "junction" else (255, 224, 50)
            diagnostic_draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=color, outline=(10, 10, 10), width=1)
            diagnostic_draw.text((px + 8, py - 8), f"N{node_index}", fill=(255, 255, 255))
    legend_y = window["height"] * 8 + 8
    diagnostic_draw.rectangle((0, legend_y, diagnostic.width, diagnostic.height), fill=(20, 22, 30))
    legend_lines = [
        "Falkreath road topology diagnostic",
        "colored fill = repaired component; white = source-derived centerline; red = bridge-influenced edge",
        "red node = junction; yellow node = endpoint/loop anchor; source Sand is not road authority",
        f"window cells x={FALKREATH_CELL_BOUNDS[0]}..{FALKREATH_CELL_BOUNDS[1]}, y={FALKREATH_CELL_BOUNDS[2]}..{FALKREATH_CELL_BOUNDS[3]}",
    ]
    for index, line in enumerate(legend_lines):
        diagnostic_draw.text((8, legend_y + 8 + index * 27), line, fill=(245, 245, 245))
    diagnostic.save(output_dir / "falkreath_junction_component_diagnostic.png")

    names = [
        "full_source_mask.png",
        "full_repaired_mask.png",
        "full_centerlines.png",
        "full_centerlines_over_source.png",
        "falkreath_source_mask_8x.png",
        "falkreath_repair_bridges_8x.png",
        "falkreath_centerlines_8x.png",
        "falkreath_centerlines_over_texture_8x.png",
        "falkreath_junction_component_diagnostic.png",
    ]
    return {name: sha256_file(output_dir / name) for name in names}


def _geojson(vectors: VectorResult) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build edge and node GeoJSON without replacing the canonical graph."""

    edge_features = []
    for edge in vectors.edges:
        edge_features.append(
            {
                "type": "Feature",
                "id": edge["id"],
                "geometry": {"type": "LineString", "coordinates": edge["smooth_gu_polyline"]},
                "properties": {
                    "from": edge["from"],
                    "to": edge["to"],
                    "component_id": edge["component_id"],
                    "source_status": edge["source_status"],
                    "bridge_ids": edge["bridge_ids"],
                    "width_gu": edge["estimated_width_gu"],
                    "length_gu": edge["length_gu"],
                },
            }
        )
    node_features = []
    for node in vectors.nodes:
        node_features.append(
            {
                "type": "Feature",
                "id": node["id"],
                "geometry": {"type": "Point", "coordinates": node["position_gu"]},
                "properties": {
                    "degree": node["degree"],
                    "kind": node["kind"],
                    "component_id": node["component_id"],
                },
            }
        )
    return (
        {"type": "FeatureCollection", "coordinate_system": "TES3 exterior world GU", "features": edge_features},
        {"type": "FeatureCollection", "coordinate_system": "TES3 exterior world GU", "features": node_features},
    )


def _svg(vectors: VectorResult, width: int, height: int) -> str:
    """Build a lightweight full-map SVG in source image coordinates."""

    paths: list[str] = []
    for edge in vectors.edges:
        coords = edge["smooth_pixel_polyline"]
        if len(coords) < 2:
            continue
        commands = [f"M {float(coords[0][0]):.3f},{float(coords[0][1]):.3f}"]
        commands.extend(f"L {float(point[0]):.3f},{float(point[1]):.3f}" for point in coords[1:])
        color = "#fadd47" if edge.get("bridge_ids") else "#45d9ff"
        paths.append(f'<path id="{edge["id"]}" d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}"><title>Tamriel road centerlines v1</title>'
        + "".join(paths)
        + "</svg>\n"
    )


def write_bundle(
    output_dir: str | Path,
    *,
    source: RoadSource,
    repair: RepairResult,
    graph: SkeletonGraph,
    vectors: VectorResult,
    source_bmp: str | Path,
    source_palette: str | Path,
    corrected_parity: Mapping[str, Any] | None = None,
) -> BundleResult:
    """Write the complete versioned road-centerline bundle."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite a non-empty road bundle: {out}")

    vtex = read_vtex_canvas(source_bmp, width=source.width, height=source.height)
    # Persist exact source/repaired arrays as sidecars in addition to PNGs;
    # PNG is for review, while these arrays are the lossless pipeline handoff.
    np.save(out / "source_effective_alpha.npy", source.effective_alpha, allow_pickle=False)
    np.save(out / "source_binary_mask.npy", source.binary_mask, allow_pickle=False)
    np.save(out / "repaired_mask.npy", repair.repaired_mask, allow_pickle=False)
    np.save(out / "bridge_mask.npy", repair.bridge_mask, allow_pickle=False)
    np.save(out / "final_skeleton.npy", graph.skeleton, allow_pickle=False)
    np.save(out / "repaired_component_labels.npy", repair.repaired_component_labels, allow_pickle=False)

    visual_hashes = _render_images(out, source, repair, graph, vectors, vtex)
    edge_geojson, node_geojson = _geojson(vectors)
    _write_json(out / "edges.geojson", edge_geojson)
    _write_json(out / "nodes.geojson", node_geojson)
    (out / "centerlines.svg").write_text(_svg(vectors, source.width, source.height), encoding="utf-8")
    _write_json(out / "bridge_ledger.json", repair.bridge_ledger)

    window_stats = _component_window_statistics(repair)
    window_nodes = [
        node
        for node in vectors.nodes
        if FALKREATH_WINDOW["x0"] <= int(node["position_px"][0]) < FALKREATH_WINDOW["x0"] + FALKREATH_WINDOW["width"]
        and FALKREATH_WINDOW["y0"] <= int(node["position_px"][1]) < FALKREATH_WINDOW["y0"] + FALKREATH_WINDOW["height"]
    ]
    window_edges = [
        edge
        for edge in vectors.edges
        if any(
            FALKREATH_WINDOW["x0"] <= int(round(float(point[0]))) < FALKREATH_WINDOW["x0"] + FALKREATH_WINDOW["width"]
            and FALKREATH_WINDOW["y0"] <= int(round(float(point[1]))) < FALKREATH_WINDOW["y0"] + FALKREATH_WINDOW["height"]
            for point in edge["smooth_pixel_polyline"]
        )
    ]
    statistics: dict[str, Any] = {
        "source": {
            "effective_occupancy_px": int(np.count_nonzero(source.effective_alpha)),
            "binary_occupancy_px": int(np.count_nonzero(source.binary_mask)),
            "effective_alpha_sha256": source.metadata["effective_alpha_sha256"],
            "binary_mask_sha256": source.metadata["binary_mask_sha256"],
            "component_count": int(repair.metadata["source_component_count"]),
        },
        "repair": {
            "repaired_occupancy_px": int(np.count_nonzero(repair.repaired_mask)),
            "bridge_occupancy_px": int(np.count_nonzero(repair.bridge_mask)),
            "component_count": int(repair.metadata["repaired_component_count"]),
            "accepted_bridge_count": int(repair.metadata["accepted_bridge_count"]),
            "rejected_candidate_count": int(repair.metadata["rejected_candidate_count"]),
            "accepted_bridge_counts": dict(repair.metadata["accepted_bridge_counts"]),
            "rejected_candidate_counts": dict(repair.metadata["rejected_candidate_counts"]),
            "union_component_count": int(repair.metadata["union_component_count"]),
            "union_matches_repaired_component_count": bool(
                repair.metadata["union_matches_repaired_component_count"]
            ),
            "accepted_touched_source_component_count_histogram": dict(
                repair.metadata["accepted_touched_source_component_count_histogram"]
            ),
            "selected_max_gap_px": repair.metadata["selected_max_gap_px"],
            "profile": repair.bridge_ledger["profile"],
        },
        "graph": graph.validation["statistics"],
        "graph_validation": graph.validation,
        "vectors": vectors.metrics,
        "falkreath": {
            **window_stats,
            "node_count_in_window": len(window_nodes),
            "edge_count_in_window": len(window_edges),
            "junction_count_in_window": sum(1 for node in window_nodes if node["kind"] == "junction"),
            "loop_count_in_window": sum(1 for edge in window_edges if edge["from"] == edge["to"]),
        },
        "visual_hashes": visual_hashes,
    }

    source_metadata = dict(source.metadata)
    source_metadata.update(
        {
            "source_bmp_path": str(source_bmp),
            "source_bmp_sha256": sha256_file(source_bmp),
            "source_palette_path": str(source_palette),
            "source_palette_sha256": sha256_file(source_palette),
            "raw_vtex_road_correlation": {
                "raw_1_semantics": "Sand; excluded from geometry authority",
                "raw_78_semantics": "LTEX[77] MA_sulphur_rock02; source-correlation evidence only",
                "geometry_authority": "effective XCF road network layer",
            },
        }
    )
    if corrected_parity is not None:
        source_metadata["corrected_parity_evidence"] = dict(corrected_parity)
    _write_json(out / "source_metadata.json", source_metadata)
    _write_json(out / "audit.json", {"schema_version": 1, "statistics": statistics})
    (out / "audit.txt").write_text(_human_audit(statistics), encoding="utf-8")

    canonical: dict[str, Any] = {
        "schema_version": 1,
        "graph_kind": "tamriel_road_centerlines",
        "graph_version": "v1",
        "source": source_metadata,
        "transform": vectors.metrics["transform"],
        "algorithm": {
            "version": "tamriel_road_centerlines_v1",
            "seed": 0,
            "randomness": "none; all ordering and curve sampling are deterministic",
            "repair_settings": repair.bridge_ledger["settings"],
            "vector_settings": vectors.metrics["settings"],
            "skeleton_connectivity": 8,
            "source_mask_immutable": True,
        },
        "repair": {
            "selected_max_gap_px": repair.bridge_ledger["selected_max_gap_px"],
            "threshold_basis": repair.bridge_ledger["threshold_basis"],
            "bridge_ledger": repair.bridge_ledger,
            "source_component_count": repair.metadata["source_component_count"],
            "repaired_component_count": repair.metadata["repaired_component_count"],
        },
        "components": graph.components,
        "nodes": vectors.nodes,
        "edges": vectors.edges,
        "statistics": statistics,
        "artifacts": {
            "source_effective_alpha": "source_effective_alpha.npy",
            "source_binary_mask": "source_binary_mask.npy",
            "repaired_mask": "repaired_mask.npy",
            "bridge_mask": "bridge_mask.npy",
            "final_skeleton": "final_skeleton.npy",
            "bridge_ledger": "bridge_ledger.json",
            "audit": "audit.json",
            "geojson_edges": "edges.geojson",
            "geojson_nodes": "nodes.geojson",
            "svg": "centerlines.svg",
            "visuals": sorted(visual_hashes),
        },
        "ordering": {
            "nodes": "ascending content-derived node id",
            "edges": "ascending content-derived edge id",
            "bridge_ledger": "accepted/rejected bridge id, each candidate retained",
            "raw_pixel_chain": "trace direction normalized by node ids; loops choose lexicographically smaller direction",
        },
        "determinism": {
            "canonical_payload_sha256_basis": "canonical JSON with determinism.canonical_payload_sha256 set to the empty string",
            "canonical_payload_sha256": "",
            "rerun_comparison_files": ["tamriel_road_centerlines_v1.json", "audit.json"],
        },
    }
    basis = _json_bytes(canonical)
    canonical["determinism"]["canonical_payload_sha256"] = hashlib.sha256(basis).hexdigest()
    canonical_path = out / "tamriel_road_centerlines_v1.json"
    canonical_sha = _write_json(canonical_path, canonical)
    audit_payload = {
        "schema_version": 1,
        "canonical_json": "tamriel_road_centerlines_v1.json",
        "canonical_payload_sha256": canonical["determinism"]["canonical_payload_sha256"],
        "canonical_file_sha256": canonical_sha,
        "source_effective_alpha_sha256": source.metadata["effective_alpha_sha256"],
        "source_binary_mask_sha256": source.metadata["binary_mask_sha256"],
        "statistics": statistics,
        "visual_hashes": visual_hashes,
    }
    audit_path = out / "audit.json"
    audit_sha = _write_json(audit_path, audit_payload)
    (out / "audit.txt").write_text(_human_audit({**statistics, "audit_sha256": audit_sha, "canonical_sha256": canonical_sha}), encoding="utf-8")
    return BundleResult(
        output_dir=out,
        canonical_path=canonical_path,
        audit_path=audit_path,
        canonical_sha256=canonical_sha,
        audit_sha256=audit_sha,
        statistics=statistics,
    )


def _human_audit(statistics: Mapping[str, Any]) -> str:
    """Make a concise deterministic text audit for reviewers."""

    source = statistics["source"]
    repair = statistics["repair"]
    graph = statistics["graph"]
    vectors = statistics["vectors"]
    falkreath = statistics["falkreath"]
    lines = [
        "Tamriel road centerlines v1 audit",
        "=================================",
        f"source occupancy: {source['effective_occupancy_px']} px; source components: {source['component_count']}",
        f"repair occupancy: {repair['repaired_occupancy_px']} px; repaired components: {repair['component_count']}",
        f"bridges accepted/rejected: {repair['accepted_bridge_count']}/{repair['rejected_candidate_count']}; selected max gap: {repair['selected_max_gap_px']:.3f} px",
        f"bridge families accepted endpoint-endpoint/endpoint-corridor: {repair['accepted_bridge_counts']['endpoint_endpoint']}/{repair['accepted_bridge_counts']['endpoint_to_corridor']}",
        f"bridge families rejected endpoint-endpoint/endpoint-corridor: {repair['rejected_candidate_counts']['endpoint_endpoint']}/{repair['rejected_candidate_counts']['endpoint_to_corridor']}",
        f"union components/repaired components: {repair['union_component_count']}/{repair['component_count']}; touched-source histogram: {repair['accepted_touched_source_component_count_histogram']}",
        f"skeleton pixels/nodes/edges/loops: {graph['skeleton_pixels']}/{graph['node_count']}/{graph['edge_count']}/{graph['loop_edge_count']}",
        f"max smooth deviation: {vectors['max_edge_deviation_px']:.3f} px; max endpoint displacement: {vectors['maximum_endpoint_displacement_gu']:.3f} GU",
        f"max symmetric sampled Hausdorff: {vectors['maximum_symmetric_hausdorff_px']:.3f} px",
        f"centerline raster inside repaired corridor: {vectors['centerline_raster_inside_repaired_fraction']:.6f}",
        f"high-frequency turn reduction: {vectors['high_frequency_turn_reduction_count']} ({vectors['high_frequency_turn_reduction_fraction']:.6f}); zigzag reduction: {vectors['high_frequency_zigzag_reduction_count']} ({vectors['high_frequency_zigzag_reduction_fraction']:.6f})",
        f"smoothing methods: {vectors['smoothing_method_counts']}; raw fallback edges: {vectors['raw_fallback_edge_count']}",
        f"Falkreath raw/repaired local components: {falkreath['raw_local_component_count']}/{falkreath['repaired_local_component_count']}; bridges: {falkreath['bridge_pixels']}",
    ]
    if "canonical_sha256" in statistics:
        lines.append(f"canonical file sha256: {statistics['canonical_sha256']}")
    if "audit_sha256" in statistics:
        lines.append(f"audit file sha256: {statistics['audit_sha256']}")
    return "\n".join(lines) + "\n"


__all__ = ["BundleResult", "FALKREATH_CELL_BOUNDS", "FALKREATH_WINDOW", "write_bundle"]
