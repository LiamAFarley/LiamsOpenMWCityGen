"""Deterministic thumbnail generation for the stamp palette (Cityforge T0.5).

Purpose
-------
Generate compact, lossless PNG thumbnails from the **actual final** 2304×1536
terrain-backed source sheets (and the supporting overview/textured-map
images) so the stamp palette cards load fast and reliably.  Thumbnails are
derived at build time from the read-only library PNGs - never from stale
previews or prior split-render files - and every thumbnail record carries the
SHA-256 of its source file for traceability.

Inputs
------
* ``library_root`` - read-only final extraction library (source PNGs).
* ``thumb_dir`` - output directory for the thumbnails (canonical location:
  ``<library>/stamp_palette_v1/thumbnails/``; tests use temp dirs).
* ``plan`` - pure dict from :func:`procgen.stamp_palette.thumbnail_plan`:
  ``{rel_path: {"source_file", "kind", "width", "height", "source_dims"}}``
  (rel_path like ``thumbnails/02_s_house_sheet_2x3.png``).
* ``source_sha256_by_file`` - manifest SHA-256 per source file (provenance).

Outputs
-------
A list of result records (one per thumbnail), each with the thumbnail file
name, dimensions, aspect, its own SHA-256, the source file + source SHA-256,
and nonblank validation metrics.  Raises :class:`ThumbnailError` if any
thumbnail fails validation (wrong dims/aspect, blank or near-uniform pixels),
so black placeholders can never be presented as success.

Invariants
----------
* Deterministic: same source bytes + same plan -> byte-identical PNGs
  (LANCZOS downscale, PNG lossless, no metadata/timestamps, fixed zlib
  settings).  Pillow version is the only environment factor, documented.
* Lossless PNG output: full color/terrain fidelity, whole sheet preserved,
  aspect ratio preserved - only resolution is reduced (no crop, no JPEG
  artifacts).
* Every thumbnail is validated nonblank (sampled distinct-color buckets and
  mean luminance above documented thresholds).

Pipeline position
-----------------
* Feeds: the accepted final Markarth extraction library (read-only).
* Consumed by: ``tools/cityforge/stamp_palette.py`` (writes thumbnails into
  the palette output, attaches metadata via
  ``procgen.stamp_palette.attach_thumbnail_metadata``, includes thumbnail
  bytes in the determinism proof); ``tests/test_stamp_palette.py``.
* The card browser uses ``links.thumb`` for card images and keeps
  ``links.sheet`` (the original PNG) for the lightbox and "Open original
  PNG" links.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

from procgen.stamp_palette import sha256_file

# Nonblank validation thresholds (documented, deterministic):
# sampled colors are quantized to /8 buckets; a real terrain render scores
# hundreds of buckets and a mean luminance far above these floors.
MIN_SAMPLED_COLOR_BUCKETS = 24
MIN_MEAN_LUMA = 12  # 0..255


class ThumbnailError(Exception):
    """One or more thumbnails failed generation/validation."""


def _png_dimensions(path: Path):
    """Minimal PNG IHDR read (same logic as the palette engine)."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ThumbnailError(f"not a valid PNG: {path}")
    return (
        int.from_bytes(head[16:20], "big"),
        int.from_bytes(head[20:24], "big"),
    )


def sample_stats(path: Path):
    """Sample a PNG grid and return ``(quantized_distinct_buckets, mean_luma)``.

    Steps every ~1/48 of the smaller dimension across the whole image so the
    sample always covers the entire sheet (no crop-based blind spot)."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    step = max(1, min(w, h) // 48)
    px = im.load()
    colors = set()
    total = 0.0
    n = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            r, g, b = px[x, y]
            colors.add((r // 8, g // 8, b // 8))
            total += 0.299 * r + 0.587 * g + 0.114 * b
            n += 1
    return len(colors), (total / n if n else 0.0)


def write_thumbnails(
    library_root: Path,
    thumb_dir: Path,
    plan: Dict[str, Dict[str, Any]],
    source_sha256_by_file: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Generate + validate all thumbnails in ``plan`` under ``thumb_dir``.

    Returns one result record per thumbnail (deterministic order = plan
    order).  Raises :class:`ThumbnailError` listing every failing thumbnail
    (blank output, wrong dimensions, aspect mismatch, missing source).
    """
    from PIL import Image

    resample = getattr(Image, "Resampling", Image).LANCZOS
    thumb_dir = Path(thumb_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    for rel, spec in plan.items():
        src = Path(library_root) / spec["source_file"]
        out = Path(thumb_dir) / Path(rel).name
        if not src.is_file():
            failures.append(f"{rel}: source missing {src}")
            continue
        try:
            im = Image.open(src)
            im.load()
        except Exception as exc:  # noqa: BLE001 - any decode failure is fatal
            failures.append(f"{rel}: decode failed {exc}")
            continue
        sw, sh = im.size
        if (sw, sh) != tuple(spec["source_dims"]):
            failures.append(
                f"{rel}: source dims {im.size} != plan {spec['source_dims']}"
            )
            continue
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        resized = im.resize((spec["width"], spec["height"]), resample)
        resized.save(out, format="PNG", optimize=True)
        # validation: dimensions + aspect (sub-pixel tolerance: the output
        # height is the rounded ideal height, so a half-pixel rounding
        # error is expected for non-3:2 sources like overview pages) +
        # nonblank
        tw, th = _png_dimensions(out)
        if (tw, th) != (spec["width"], spec["height"]):
            failures.append(
                f"{rel}: wrote dims {(tw, th)} != plan {(spec['width'], spec['height'])}"
            )
            continue
        ideal_h = sh * tw / sw
        if abs(th - ideal_h) > 0.75:
            failures.append(f"{rel}: aspect changed {sw}/{sh} -> {tw}/{th}")
            continue
        buckets, mean_luma = sample_stats(out)
        if buckets < MIN_SAMPLED_COLOR_BUCKETS or mean_luma < MIN_MEAN_LUMA:
            failures.append(
                f"{rel}: nonblank check failed "
                f"(buckets={buckets} < {MIN_SAMPLED_COLOR_BUCKETS}, "
                f"mean_luma={mean_luma:.0f} < {MIN_MEAN_LUMA})"
            )
            continue
        sha = sha256_file(out)
        source_sha = source_sha256_by_file.get(spec["source_file"], "")
        results.append(
            {
                "file": rel,
                "source_file": spec["source_file"],
                "source_sha256": source_sha,
                "width": tw,
                "height": th,
                "aspect": round(tw / th, 3),
                "sha256": sha,
                "sha256_short": sha[:8],
                "nonblank": True,
                "sampled_color_buckets": buckets,
                "mean_luma": round(mean_luma),
            }
        )

    if failures:
        shown = failures[:30]
        more = f" (+{len(failures) - len(shown)} more)" if len(failures) > 30 else ""
        raise ThumbnailError(
            "thumbnail generation failed for {} file(s): {}{}".format(
                len(failures), "; ".join(shown), more
            )
        )
    return results


def verify_thumbnails(
    thumb_dir: Path,
    results: Sequence[Dict[str, Any]],
) -> List[str]:
    """Independent re-verification of written thumbnails.

    Re-checks every thumbnail file's existence, dimensions, aspect, own
    SHA-256 and nonblank sample.  Returns a list of problems (empty == all
    good).  Used post-write so the canonical state is proven, not assumed.
    """
    thumb_dir = Path(thumb_dir)
    problems: List[str] = []
    for r in results:
        p = thumb_dir / Path(r["file"]).name
        if not p.is_file():
            problems.append(f"{r['file']}: missing on disk")
            continue
        try:
            tw, th = _png_dimensions(p)
        except ThumbnailError as exc:
            problems.append(str(exc))
            continue
        if (tw, th) != (r["width"], r["height"]):
            problems.append(
                f"{r['file']}: dims {(tw, th)} != recorded {(r['width'], r['height'])}"
            )
        if round(tw / th, 3) != r["aspect"]:
            problems.append(f"{r['file']}: aspect mismatch")
        if sha256_file(p) != r["sha256"]:
            problems.append(f"{r['file']}: sha256 mismatch")
        buckets, mean_luma = sample_stats(p)
        if buckets < MIN_SAMPLED_COLOR_BUCKETS or mean_luma < MIN_MEAN_LUMA:
            problems.append(f"{r['file']}: nonblank re-check failed")
    return problems
