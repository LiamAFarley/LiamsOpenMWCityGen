"""Build one per-site Cityforge "planning bundle" for a visual design agent.

Pipeline position
------------------
This CLI sits between the accepted T0.x products (site survey, aligned
roads, D-STAMP libraries, Markarth palette, stamp catalog) and the
vision-capable town designer.  It precompiles everything that designer
needs for ONE town in ONE short session into a self-contained directory:
the planning canvas PNG with labelled GU graticule, the eligible stamp
catalog (stamps.json), one contact sheet of eligible stamp previews, the
site rectangle + clipped source roads (site.json), the design rules text
(rules.md), and a tool-written manifest (bundle_manifest.json).  The
design agent reads ONLY this directory; it never opens catalogs,
libraries, or preview trees.

The eligibility policy (``src/procgen/visual_planner_eligibility.py``) is
applied fail-closed: quarantined stamps (e.g. ``markarth_side_v1__u114_
castle_barracks``, ``markarth_side_v1__conn_114_1``) never appear in
``stamps.json`` or ``stamps_sheet.png``.

Inputs
------
``--survey-dir``       D-SITE directory containing ``site_survey.json`` +
                       ``survey_fields.npz``.
``--roads-dir``        aligned road product directory (loaded through
                       ``src/procgen/aligned_roads.py``; source-space
                       bundles are refused by that loader).
``--stamp-libraries``  D-STAMP library JSON files (1..n).
``--palette``          accepted Markarth stamp palette ``catalog.json``.
``--catalog-index``    ``output/cityforge/stamps/catalog_v1/index.json``
                       mapping stamp_id -> verified preview PNG.
``--cells``            ``x0..x1,y0..y1`` inclusive TES3 cell range.
``--margin-gu``        context margin around the cell rectangle (default
                       2048).
``--out``              fresh output directory (refused when non-empty or
                       under a protected data root).

Outputs (all six, in one deterministic run)
-------------------------------------------
``canvas.png``         planning canvas (see ``planning_canvas.py``).
``stamps.json``        one entry per ELIGIBLE stamp: id, kit,
                        building_type, size_class, footprint dims, doors
                        (stable source ids, relative offsets + heading), terrain
                        envelope, and style tags.
``stamps_sheet.png``   contact sheet of eligible stamp thumbnails from the
                       catalog previews (a second ``stamps_sheet_2.png``
                       only when one sheet would exceed 4 MB).
``site.json``          site name, cells, world-GU rectangle, clipped +
                       simplified source-road chains.
``rules.md``           the fixed design-rules text (site name/cells
                       substituted).
``bundle_manifest.json`` tool-written: UTC timestamp, input/output
                       sha256, eligible/quarantined counts.

Invariants
----------
* All outputs are deterministic for identical inputs.
* Every number written to ``stamps.json`` is measured library data; no
  metric is invented (burial min/max both equal the library's single
  ``burial_depth_gu`` because that is all the accepted library records).
* ``stamps.json`` contains ONLY eligible stamps; quarantined stamps are
  counted and listed in the manifest.
* The rectangle is cell bounds expanded by ``--margin-gu`` in TES3 world
  GU; canvas coordinates are world GU (x east, y north).
* This tool never modifies an original file and never authors TES3.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image, ImageFont  # noqa: E402

from procgen import aligned_roads  # noqa: E402
from procgen.planning_canvas import (  # noqa: E402
    CELL_SIZE_GU,
    clip_polyline,
    douglas_peucker,
    render_planning_canvas,
)
from procgen.visual_planner_eligibility import build_eligibility_policy  # noqa: E402
from procgen.visual_planner_terrain import TerrainBundle  # noqa: E402

#: Canonical library-level keys of the accepted D-STAMP format.
_LIBRARY_ID_KEY = "library_id"
_STAMP_ID_KEY = "stamp_id"

#: Deterministic size-class sort order (unknown classes sort last).
_SIZE_RANK = {"small": 0, "medium": 1, "large": 2}

#: Contact-sheet tuning: (thumbnail px, label px, columns).  The first
#: configuration is tried; on >4 MB the sheet is rebuilt smaller, and if
#: still too large it is split into two files.  Label bands are 48 px so an
#: id wrapped to two lines plus the footprint/door line always fit.
_SHEET_CONFIGS = ((224, 48, 6), (192, 44, 7))
_MAX_SHEET_BYTES = 4 * 1024 * 1024

#: Font sizes tried (largest first) when fitting a tile label to its width.
_ID_FONT_SIZES = (13, 12, 11, 10)
_META_FONT_SIZES = (12, 11, 10)

#: World-GU simplification tolerance for site.json road chains (about 10 px
#: on a 1600 px canvas, small enough to keep bends that matter visually).
_SIMPLIFY_EPSILON_GU = 128.0

RULES_TEMPLATE = """# {SITE_NAME} — city design rules

You are designing the town plan for {SITE_NAME} (cells {CELLS}). You have exactly
four inputs: canvas.png (terrain + existing roads + GU coordinate grid),
stamps_sheet.png (every building you may use), stamps.json (footprint sizes and
door positions), and this file. Nothing else exists. Do not read other files.

## What you author

One file, sketch.json, with exactly these top-level keys (no others):

```json
{{
  "site": "{SITE_NAME}",
  "roads":  [{{"id": "street_main", "kind": "street", "width_gu": 640,
              "points": [[x, y], [x, y]]}}],
  "spaces": [{{"id": "plaza_market", "kind": "plaza",
              "polygon": [[x, y], [x, y], [x, y]]}}],
  "lots":   [{{"id": "lot_example", "stamp": "kit__stamp_id",
              "x": -758784, "y": -62464, "yaw_deg": 90, "note": "optional"}}],
  "notes":  "your site reading and design rationale, at most 20 lines"
}}
```

- "site" must be exactly "{SITE_NAME}".
- roads: polylines (>=2 points), kind "street" or "alley", width_gu
  (street ~512-768, alley ~256-384)
- spaces: polygons (>=3 points), kind "plaza" or "court"
- lots: one per building — stamp id copied VERBATIM from stamps.json (ids in
  this bundle start with {ID_PREFIXES}; the sheet labels are shortened), center
  x,y in GU (read them off the canvas grid), yaw_deg (0 = stamp's source
  orientation; doors rotate with the stamp)

Everything else (door links, districts, validation, exports) is derived by
tools after you finish. You never write code, manifests, inventories, or reports.

## Design semantics (definitions, not metrics)

- The cyan existing roads are the primary structure. Compose the town around
  them. Never replace, move, or ignore them; new streets must connect to them
  or to another authored street. **A door close to a cyan path (within a few
  meters of the drawn band) counts as connected to that road.** Facing
  buildings toward the cyan roads is the DEFAULT — most doors should engage
  a cyan corridor or a space bounded by buildings; authored streets/alleys
  are only for doors the cyan roads cannot serve.
- Street: a circulation spine lined with building frontage, or a clear
  connection between meaningful destinations.
- Alley: a narrow route BETWEEN or BEHIND buildings serving rear/side doors.
  An alley drawn around the outside of a building group is not an alley.
- Court: an open room substantially enclosed by surrounding building faces,
  with one or two legible entrances. A rectangle drawn around unrelated
  buildings is not a court.
- Plaza: a deliberate public space bounded by active frontage or civic
  anchors; roads meet it at its edges.
- If a door has no road, alley, court, plaza, or facing-building relationship
  within a few meters, the building is misplaced or misrotated — fix the
  building, not by adding more circulation.

## Composition rules

- Clearance: no part of any building footprint may touch or cross a cyan
  source-road PATH band or an authored street/alley band. The cyan band is
  drawn at the PRACTICAL PATH width — in game the road texture blends wider
  than the band, and buildings sitting on that extra texture are FINE; only
  the drawn path must stay clear. Frontage means the footprint edge sits
  just OUTSIDE the band — adjacent, never overlapping. Leave a visible gap
  (a few hundred GU) between footprint and band edge.
- Density: build a real town center — a compact cluster around the main
  public space — then let buildings thin out with distance: a few sparse
  lots along the roads outward, isolated outliers near the edges. Do not
  fill the map; empty green between clusters is correct. Prefer occupying
  the natural room between existing roads over packing one corner.
- Buildings mostly face roads or face each other across lanes/courts.
  Never stack same-orientation lots directly behind each other; a rear
  building needs a real alley or court.
- Avoid clustering many identical or similar houses next to each other; vary
  stamp, yaw, and setback. Use a wide VARIETY of stamps — there are 50+
  eligible; repeats of the same stamp should be the exception, not the
  pattern (check `stamp_usage` in checks.json). Stamps with "hut" in the
  name belong at the outskirts only.
- Markarth stone stamps form the compact civic/richer core; Karthgad timber
  stamps are the surrounding town fabric.
- Keep circulation minimal and organic: as few streets/alleys as the
  frontage logic needs, with irregular shapes. No diamond or mirror-
  symmetric plazas, and no framing streets on every side. A plaza is sized
  by the buildings that actually bound it.
- Dry slopes are allowed; steep spots favor stamps whose stamps.json terrain
  envelope lists higher source slope/relief.
- Water and areas outside the canvas rectangle are forbidden.
- No palisades, gates, castles, docks, or decoration — buildings and
  circulation only.

## Working loop

1. Study canvas.png: terrain, water, the existing road approaches, where a
   center wants to be.
2. Study stamps_sheet.png; shortlist anchors (civic/stone) and fabric (timber).
3. Write sketch.json: circulation first (1-3 streets + a plaza/court), then
   anchor lots on frontage, then fabric lots. Roads/alleys may FORK a cyan
   road anywhere along it (T-junctions are fine) — the endpoint just has to
   geometrically reach the corridor.
4. Run the render command given in your task. Open the image. Judge it against
   these rules BY EYE. Name specific weak lots/clusters in your notes.
   Hard-error overlaps paint the exact intersecting region in red; doors that
   reach no circulation are deep blue.
5. Iterate in SMALL steps — move or rotate a few lots per render rather than
   rewriting everything at once. Rotate lots freely (any angle, not just
   0/90/180/270) so doors face their circulation; if rotating by hand is
   awkward, the `--auto-face` flag on the render command rotates every lot
   so its main door faces the nearest road/space — use it, then hand-adjust.
   Keep going until it looks right (your task states the render budget).
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} is not a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    """Compact deterministic JSON (bundle outputs stay small; sort_keys
    keeps byte-identical reruns)."""
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")


def parse_cells(raw: str) -> tuple[int, int, int, int]:
    """Parse ``x0..x1,y0..y1`` into ``(min_x, max_x, min_y, max_y)``."""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("--cells must look like 'x0..x1,y0..y1'")
    values: list[int] = []
    for part in parts:
        match = re.fullmatch(r"(-?\d+)\.\.(-?\d+)", part)
        if not match:
            raise ValueError(f"--cells segment {part!r} is not 'a..b'")
        values.extend((int(match.group(1)), int(match.group(2))))
    min_x, max_x, min_y, max_y = values
    if min_x > max_x or min_y > max_y:
        raise ValueError("--cells minimum exceeds maximum")
    return min_x, max_x, min_y, max_y


def protected_roots() -> list[Path]:
    """Data roots from configs/procgen.json plus the fixed C:\\Modding root."""
    roots = [Path(r"C:\Modding").resolve()]
    config = _load_json(ROOT / "configs/procgen.json", "procgen config")
    paths = config.get("paths")
    if isinstance(paths, dict):
        for raw in paths.get("data_roots", []):
            if isinstance(raw, str) and raw.strip():
                roots.append(Path(raw).resolve())
    return roots


def refuse_unless_fresh(out_dir: Path) -> None:
    """Standard fresh-output-dir + protected-root refusal."""
    resolved = out_dir.resolve()
    for root in protected_roots():
        if resolved == root or root in resolved.parents:
            raise ValueError(f"refusing output under protected data root: {out_dir}")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {out_dir}")
    resolved.mkdir(parents=True, exist_ok=True)


def load_stamp_geometry(library_paths: list[Path]) -> tuple[dict[str, tuple[str, dict[str, Any]]], dict[str, str]]:
    """Deterministic stamp-id -> ``(kit, record)`` map plus library id hashes.

    Each library is read exactly once; the returned kit mapping is the
    library's own ``library_id`` string (e.g. ``karthgad_nord_v1``).
    """
    geometry: dict[str, tuple[str, dict[str, Any]]] = {}
    kits: dict[str, str] = {}
    for path in library_paths:
        library = _load_json(path, "stamp library")
        kit = library.get(_LIBRARY_ID_KEY)
        if not isinstance(kit, str) or not kit:
            raise ValueError(f"stamp library has no library_id: {path}")
        kits[str(path)] = kit
        for stamp in library.get("stamps", []):
            if not isinstance(stamp, dict):
                raise ValueError(f"stamp record is not an object in {path}")
            stamp_id = stamp.get(_STAMP_ID_KEY)
            if not isinstance(stamp_id, str) or not stamp_id:
                raise ValueError(f"stamp without stable id in {path}")
            if stamp_id in geometry:
                raise ValueError(f"duplicate stamp id {stamp_id}")
            geometry[stamp_id] = (kit, stamp)
    return dict(sorted(geometry.items())), kits


def _door_members(stamp: dict[str, Any]) -> list[dict[str, Any]]:
    doors = [m for m in stamp.get("members", []) if isinstance(m, dict) and m.get("is_door")]
    doors.sort(key=lambda m: (float(m.get("offset_gu", [0, 0, 0])[0]),
                              float(m.get("offset_gu", [0, 0, 0])[1]),
                              str(m.get("source_id", ""))))
    return doors


def stamp_entry(stamp_id: str, stamp: dict[str, Any], kit: str) -> dict[str, Any]:
    """One stamps.json row, all numbers measured from the library record."""
    bounds = stamp.get("bounds_rel_gu")
    span = bounds.get("span") if isinstance(bounds, dict) else None
    if not isinstance(span, (list, tuple)) or len(span) < 2:
        raise ValueError(f"stamp {stamp_id} has no usable bounds_rel_gu.span")
    envelope = stamp.get("terrain_envelope")
    if not isinstance(envelope, dict):
        raise ValueError(f"stamp {stamp_id} has no terrain_envelope")
    burial = float(envelope.get("burial_depth_gu", 0.0))
    doors: list[dict[str, Any]] = []
    for member in _door_members(stamp):
        offset = member.get("offset_gu", [0, 0, 0])
        # Prefer the geometric outward heading (thin-axis wall normal, sign
        # away from the body centroid; derived in the v2 library build).  The
        # raw TES3 door rotz is only a fallback: mesh forward axes differ
        # per model family, so rotz is not a reliable facing.
        outward = member.get("outward_heading_deg")
        if outward is None:
            rotation = member.get("rotation", [0, 0, 0])
            outward = math.degrees(float(rotation[2])) % 360.0
        door_id = member.get("source_id")
        if not isinstance(door_id, str) or not door_id:
            raise ValueError(f"stamp {stamp_id} has a door without stable source_id")
        doors.append({
            "door_id": door_id,
            "dx_gu": round(float(offset[0]), 1),
            "dy_gu": round(float(offset[1]), 1),
            "heading_deg": round(float(outward) % 360.0, 1),
        })
    declared_doors = int(stamp.get("door_count", -1))
    if declared_doors != len(doors):
        raise ValueError(
            f"stamp {stamp_id} door_count {declared_doors} != door members {len(doors)}")
    return {
        "id": stamp_id,
        "kit": kit,
        "building_type": stamp.get("building_type"),
        "size_class": stamp.get("size_class"),
        "footprint_w_gu": round(float(span[0]), 1),
        "footprint_d_gu": round(float(span[1]), 1),
        "door_count": declared_doors,
        "doors": doors,
        "terrain_envelope": {
            "source_slope_deg": round(float(envelope.get("footprint_slope_deg", 0.0)), 2),
            "relief_gu": round(float(envelope.get("footprint_relief_gu", 0.0)), 1),
            # The accepted library records one burial depth per stamp; both
            # bounds carry it so consumers can use the same envelope shape.
            "burial_min_gu": round(burial, 1),
            "burial_max_gu": round(burial, 1),
        },
        "style_tags": list(stamp.get("style_tags", [])),
    }


def _load_catalog(index_path: Path) -> dict[str, str]:
    index = _load_json(index_path, "stamp catalog index")
    mapping: dict[str, str] = {}
    for row in index.get("stamps", []):
        if not isinstance(row, dict):
            continue
        stamp_id = row.get("stamp_id")
        preview = row.get("preview_copy")
        if isinstance(stamp_id, str) and isinstance(preview, str):
            mapping[stamp_id] = preview
    return mapping


def _text_width(draw: Any, text: str, font: Any) -> float:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def fit_label_lines(
    draw: Any, text: str, max_width_px: float, sizes: tuple[int, ...] = _ID_FONT_SIZES,
) -> tuple[list[str], int]:
    """Fit a tile label to ``max_width_px``: shrink font, then wrap.

    Returns ``(lines, font_size)`` where every line fits within
    ``max_width_px`` -- the guarantee the contact sheet relies on, so a
    label can never overflow its tile or bleed into a neighbour.  Wrapping
    prefers a balanced split at an underscore; the midpoint hard split is
    the deterministic last resort (labels are ids, so a cut never happens
    inside a letter).
    """
    for size in sizes:
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)
        except OSError:
            font = ImageFont.load_default()
        if _text_width(draw, text, font) <= max_width_px:
            return [text], size
        candidates: list[tuple[int, int]] = []
        for index in range(1, len(text)):
            if text[index] == "_":
                left, right = text[:index], text[index + 1:]
                if (left and right and
                        _text_width(draw, left, font) <= max_width_px and
                        _text_width(draw, right, font) <= max_width_px):
                    candidates.append((abs(len(left) - len(right)), index))
        if candidates:
            _, index = min(candidates)
            return [text[:index], text[index + 1:]], size
        midpoint = len(text) // 2
        return [text[:midpoint], text[midpoint:]], size
    return [text], sizes[0]


def _sheet_font(size: int) -> Any:
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _write_sheet(
    entries: list[dict[str, Any]], preview_paths: dict[str, Path], out_path: Path,
    *, thumb: int, label_h: int, columns: int,
) -> None:
    """One contact-sheet PNG: thumbnails + width-fitted id/door labels."""
    from PIL import ImageDraw as _ImageDraw  # noqa: PLC0415
    rows = math.ceil(len(entries) / columns)
    step = thumb + label_h
    sheet = Image.new("RGB", (columns * step + 8, rows * step + 8), (30, 30, 30))
    draw = _ImageDraw.Draw(sheet)
    for index, entry in enumerate(entries):
        col, row = index % columns, index // columns
        x0 = col * step + 4
        y0 = row * step + 4
        preview = preview_paths[entry["id"]]
        try:
            thumb_img = Image.open(preview).convert("RGB")
        except OSError as exc:
            raise ValueError(f"cannot open stamp preview {preview}: {exc}")
        thumb_img.thumbnail((thumb, thumb), Image.LANCZOS)
        sheet.paste(thumb_img, (x0 + (thumb - thumb_img.width) // 2,
                                y0 + (thumb - thumb_img.height) // 2))
        # Labels never exceed the tile width: the id is the key the design
        # agent must match against stamps.json, so it is printed in full
        # (on two lines when needed), then the footprint/door meta line.
        max_width = thumb - 8
        short = entry["id"].split("__", 1)[-1]
        id_lines, id_size = fit_label_lines(draw, short, max_width)
        meta = f"{entry['footprint_w_gu']:.0f}x{entry['footprint_d_gu']:.0f}  {entry['door_count']}d"
        meta_lines, meta_size = fit_label_lines(draw, meta, max_width, sizes=_META_FONT_SIZES)
        # Three 13-14 px lines must fit the label band even in the smaller
        # sheet configuration, so line height derives from the band size.
        line_height = (label_h - 4) // 3
        text_y = y0 + thumb + 4
        for line in [*id_lines, *meta_lines]:
            font = _sheet_font(id_size if line in id_lines else meta_size)
            box = draw.textbbox((0, 0), line, font=font)
            draw.text((x0 + (thumb - (box[2] - box[0])) / 2.0, text_y), line,
                      font=font, fill=(235, 235, 235))
            text_y += line_height
    sheet.save(out_path, format="PNG", compress_level=6)


def build_stamps_sheet(
    entries: list[dict[str, Any]],
    preview_paths: dict[str, Path],
    out_path: Path,
    out_path_second: Path | None = None,
) -> list[Path]:
    """One contact sheet; a second file is written only when a single sheet
    exceeds the 4 MB budget at the smallest tile configuration."""
    for thumb, label_h, columns in _SHEET_CONFIGS:
        _write_sheet(entries, preview_paths, out_path,
                     thumb=thumb, label_h=label_h, columns=columns)
        if out_path.stat().st_size <= _MAX_SHEET_BYTES:
            return [out_path]
    if out_path_second is None:
        raise ValueError("contact sheet oversized and no second sheet path was provided")
    split = (len(entries) + 1) // 2
    for path, group in ((out_path, entries[:split]), (out_path_second, entries[split:])):
        _write_sheet(group, preview_paths, path, thumb=192, label_h=24, columns=7)
    return [out_path, out_path_second]


def source_road_rows(network: Any, rect: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    """Clipped + simplified world-GU chains of every edge crossing the rect."""
    x0, y0, x1, y1 = rect
    rows: list[dict[str, Any]] = []
    for edge in sorted(network.edges_in_rect(x0, y0, x1, y1), key=lambda e: e.id):
        spans = clip_polyline(edge.smooth_gu_polyline, rect)
        chain: list[tuple[float, float]] = []
        for span in spans:
            if not chain or chain[-1] != span[0]:
                chain.append(span[0])
            chain.append(span[1])
        simplified = douglas_peucker(chain, _SIMPLIFY_EPSILON_GU)
        rows.append({
            "edge_id": edge.id,
            "points_gu": [[int(round(p[0])), int(round(p[1]))] for p in simplified],
        })
    return rows


def build_bundle(
    *,
    site_name: str,
    survey_dir: Path,
    roads_dir: Path,
    stamp_libraries: list[Path],
    palette_path: Path,
    catalog_index: Path,
    cells: tuple[int, int, int, int],
    margin_gu: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Run the full bundle build; returns the summary dict printed by main."""
    refuse_unless_fresh(out_dir)
    if margin_gu < 0:
        raise ValueError("--margin-gu must be non-negative")

    survey_path = survey_dir / "site_survey.json"
    fields_path = survey_dir / "survey_fields.npz"
    for path in (survey_path, fields_path, palette_path, catalog_index):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")
    for path in stamp_libraries:
        if not path.is_file():
            raise ValueError(f"stamp library is missing: {path}")

    terrain = TerrainBundle.from_paths(survey_path, fields_path)
    network = aligned_roads.load_aligned_network(roads_dir)
    policy = build_eligibility_policy(stamp_libraries, palette_path=palette_path)
    geometry, kits = load_stamp_geometry(stamp_libraries)
    preview_map = _load_catalog(catalog_index)

    eligible_ids = sorted(s for s in policy.accepted_stamp_ids
                          if policy.is_eligible(s) and s in geometry)
    quarantined = sorted(s for s in policy.rejected_stamp_ids)
    if len(eligible_ids) != len(policy.accepted_stamp_ids):
        raise ValueError("accepted stamp inventory contains ids absent from the libraries")

    min_x, max_x, min_y, max_y = cells
    world = (
        min_x * CELL_SIZE_GU - margin_gu,
        min_y * CELL_SIZE_GU - margin_gu,
        (max_x + 1) * CELL_SIZE_GU + margin_gu,
        (max_y + 1) * CELL_SIZE_GU + margin_gu,
    )
    cells_label = f"x={min_x}..{max_x}, y={min_y}..{max_y}"

    # 1. canvas.png ---------------------------------------------------------
    canvas_path = out_dir / "canvas.png"
    image, projection = render_planning_canvas(
        terrain, world, network, site_name=site_name,
        title=f"{site_name} — cells {cells_label} — GU [{world[0]:.0f},{world[1]:.0f}].."
              f"[{world[2]:.0f},{world[3]:.0f}] (margin {margin_gu} GU)")
    image.save(canvas_path, format="PNG", compress_level=6)

    # 2. stamps.json --------------------------------------------------------
    entries: list[dict[str, Any]] = []
    for stamp_id in eligible_ids:
        kit, stamp = geometry[stamp_id]
        entries.append(stamp_entry(stamp_id, stamp, kit))
    entries.sort(key=lambda e: (str(e["kit"]),
                                _SIZE_RANK.get(str(e["size_class"]), 9),
                                str(e["id"])))
    stamps_path = out_dir / "stamps.json"
    _write_json(stamps_path, {"site_name": site_name, "stamp_count": len(entries),
                              "stamps": entries})

    # 3. stamps_sheet.png ---------------------------------------------------
    preview_paths: dict[str, Path] = {}
    for stamp_id in eligible_ids:
        relative = preview_map.get(stamp_id)
        if not relative:
            raise ValueError(f"eligible stamp {stamp_id} has no catalog preview entry")
        resolved = (ROOT / relative).resolve()
        if not resolved.is_file():
            raise ValueError(f"catalog preview is missing for {stamp_id}: {resolved}")
        preview_paths[stamp_id] = resolved
    sheet_path = out_dir / "stamps_sheet.png"
    sheet_second = out_dir / "stamps_sheet_2.png"
    sheet_files = build_stamps_sheet(entries, preview_paths, sheet_path, sheet_second)

    # 4. site.json ----------------------------------------------------------
    site = {
        "site_name": site_name,
        "cells": [min_x, max_x, min_y, max_y],
        "rectangle_gu": [int(v) for v in world],
        "margin_gu": margin_gu,
        "coordinate_frame": "tes3_world_gu_x_east_y_north",
        "source_roads": source_road_rows(network, world),
        "notes": (f"Planning bundle for {site_name}. Terrain: {survey_dir}. "
                  f"Roads: aligned centerlines (edges listed above). World GU, "
                  f"x east, y north; canvas graticule is every 1024 GU."),
    }
    site_path = out_dir / "site.json"
    _write_json(site_path, site)

    # 5. rules.md -----------------------------------------------------------
    # The strict sketch schema rejects unknown stamp ids, so the rules must
    # state the actual id prefixes present in this bundle's stamps.json.
    id_prefixes = ", ".join(
        f"`{entry['id'].split('__')[0]}__`" for entry in entries if "__" in entry["id"]
    )
    rules_path = out_dir / "rules.md"
    rules_path.write_text(
        RULES_TEMPLATE.format(SITE_NAME=site_name, CELLS=cells_label,
                              ID_PREFIXES=id_prefixes or "`(see stamps.json)`"),
        encoding="utf-8")

    # 6. bundle_manifest.json ----------------------------------------------
    input_paths = [survey_path, fields_path,
                   roads_dir / aligned_roads.PRODUCT_CANONICAL_NAME,
                   roads_dir / aligned_roads.MANIFEST_NAME,
                   *stamp_libraries, palette_path, catalog_index]
    output_paths = [canvas_path, stamps_path, *sheet_files, site_path, rules_path]
    manifest = {
        "schema_version": 1,
        "kind": "cityforge_planning_bundle",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "site_name": site_name,
        "cells": [min_x, max_x, min_y, max_y],
        "rectangle_gu": [int(v) for v in world],
        "margin_gu": margin_gu,
        "canvas": {"size_px": list(image.size),
                   "map_size_px": [projection.map_width_px, projection.map_height_px],
                   "projection": {
                       "world_bounds_gu": list(projection.world_bounds_gu),
                       "title_band_px": projection.title_band_px,
                       "mapping": ("px = (x-x0)/(x1-x0)*map_w; "
                                   "py = title_band + (y1-y)/(y1-y0)*map_h")}},
        "inputs": {str(path): _sha256(path) for path in input_paths},
        "outputs": {str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
                    for path in output_paths},
        "eligibility": {
            "eligible_stamp_count": len(eligible_ids),
            "quarantined_stamp_count": len(quarantined),
            "quarantined_stamp_ids": quarantined,
            "metadata_hashes": dict(policy.metadata_hashes),
            "fail_closed": True,
        },
        "road_edges_drawn": sorted(
            e.id for e in network.edges_in_rect(*world)),
    }
    manifest_path = out_dir / "bundle_manifest.json"
    _write_json(manifest_path, manifest)

    summary_outputs = {path.name: path.stat().st_size
                       for path in [*output_paths, manifest_path]}
    return {
        "out_dir": out_dir,
        "site_name": site_name,
        "cells": cells_label,
        "rectangle_gu": [int(v) for v in world],
        "eligible_count": len(eligible_ids),
        "quarantined_ids": quarantined,
        "outputs": summary_outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-name", required=True)
    parser.add_argument("--survey-dir", type=Path, required=True)
    parser.add_argument("--roads-dir", type=Path, required=True)
    parser.add_argument("--stamp-libraries", type=Path, nargs="+", required=True)
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--catalog-index", type=Path, required=True)
    parser.add_argument("--cells", required=True,
                        help="inclusive cell range x0..x1,y0..y1")
    parser.add_argument("--margin-gu", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cells = parse_cells(args.cells)
        summary = build_bundle(
            site_name=args.site_name,
            survey_dir=args.survey_dir,
            roads_dir=args.roads_dir,
            stamp_libraries=args.stamp_libraries,
            palette_path=args.palette,
            catalog_index=args.catalog_index,
            cells=cells,
            margin_gu=args.margin_gu,
            out_dir=args.out,
        )
        print(f"site: {summary['site_name']}  cells: {summary['cells']}")
        print(f"rectangle_gu: {summary['rectangle_gu']}")
        print(f"eligible stamps: {summary['eligible_count']}  "
              f"quarantined: {summary['quarantined_ids']}")
        for name, size in summary["outputs"].items():
            print(f"  {summary['out_dir'] / name}  ({size} bytes)")
        return 0
    except Exception as exc:  # noqa: BLE001 - exact failure is the CLI contract
        print(f"FAILURE: planning bundle {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
