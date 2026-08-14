"""Markarth stamp-palette catalog CLI (Cityforge T0.5).

Purpose
-------
Build the deterministic, self-contained static stamp palette for the
accepted final Markarth Side v2 extraction library:

* verifies every manifest asset on disk (existence, SHA-256, dimensions),
* generates compact lossless thumbnails from the actual final source PNGs
  (with source-SHA provenance and nonblank validation),
* builds the canonical ``catalog.json`` + single-file ``index.html``,
* proves byte-determinism by generating twice into fresh temp directories
  and comparing every output byte (catalog, HTML, all thumbnails) before
  the canonical write,
* writes only into ``<library>/stamp_palette_v1/`` (the read-only library
  root itself is never touched).

Usage
-----
::

    python tools/cityforge/stamp_palette.py --date 2026-08-10
    python tools/cityforge/stamp_palette.py --library <root> --out <dir> --date 2026-08-10

Exit codes: 0 = success (canonical output written and verified), nonzero =
any essential stage failure.  The script never writes canonical output
unless the full source verification, the thumbnail validation, and the
double temp-run determinism check pass.

Core logic lives in ``src/procgen/stamp_palette.py`` +
``src/procgen/stamp_thumbnails.py``; this file is the thin CLI driver.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from procgen import stamp_palette, stamp_thumbnails  # noqa: E402
from procgen.stamp_palette import (  # noqa: E402
    CatalogError,
    SourceVerificationError,
    attach_thumbnail_metadata,
    build_catalog,
    canonical_json_bytes,
    check_relative_links,
    load_manifest,
    render_html,
    thumbnail_plan,
    write_catalog,
)
from procgen.stamp_thumbnails import ThumbnailError, verify_thumbnails

DEFAULT_LIBRARY = (
    ROOT
    / "output"
    / "settlement-splits"
    / "markarth-side-v2"
    / "final-markarth-extraction-2026-08-10-library"
)
DEFAULT_OUT_NAME = "stamp_palette_v1"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the deterministic Markarth stamp palette (catalog.json + "
            "thumbnails + index.html) from the final extraction library manifest."
        )
    )
    p.add_argument("--library", type=Path, default=DEFAULT_LIBRARY,
                   help=f"final extraction library root (default: {DEFAULT_LIBRARY})")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: <library>/stamp_palette_v1)")
    p.add_argument("--date", required=True, metavar="YYYY-MM-DD",
                   help="explicit catalog provenance date (keeps output deterministic)")
    p.add_argument("--manifest", default=stamp_palette.MANIFEST_FILE,
                   help="manifest file name inside the library root")
    p.add_argument("--skip-augment", action="store_true",
                   help="do not augment entries from artifacts/*/building_spec.json")
    return p.parse_args(argv)


def _build_into(out_dir: Path, library: Path, catalog: dict,
                source_sha_by_file: dict) -> dict:
    """One full generation into ``out_dir``: thumbnails -> attach -> html ->
    catalog write.  Returns ``{file: sha256}`` for every written file
    including all thumbnails."""
    plan = thumbnail_plan(catalog)
    thumb_dir = out_dir / stamp_palette.THUMBNAIL_DIR
    if thumb_dir.exists():
        shutil.rmtree(thumb_dir)  # stale thumbnails must never linger
    results = stamp_thumbnails.write_thumbnails(
        library, thumb_dir, plan, source_sha_by_file
    )
    catalog_thumb = attach_thumbnail_metadata(catalog, results)
    json_bytes = canonical_json_bytes(catalog_thumb)
    html = render_html(catalog_thumb, json_bytes)
    written = write_catalog(out_dir, catalog_thumb, html)
    for r in results:
        written[r["file"]] = r["sha256"]
    return written


def main(argv=None) -> int:
    args = parse_args(argv)
    library = args.library.resolve()
    out_dir = (args.out or library / DEFAULT_OUT_NAME).resolve()

    # Stage 1 -- manifest + full source verification (fail loudly).
    manifest_path = library / args.manifest
    manifest, manifest_sha = load_manifest(manifest_path)
    try:
        results = stamp_palette.verify_asset_sources(library, manifest)
    except SourceVerificationError as exc:
        print(f"FAILURE: source verification {exc}", file=sys.stderr)
        return 2
    ok_count = sum(1 for r in results if r["ok"])
    print(f"source verification: {ok_count}/{len(results)} assets ok "
          f"(manifest sha256 {manifest_sha})")

    # Stage 2 -- build catalog (+ pure thumbnail plan).
    catalog = build_catalog(
        library,
        manifest,
        date=args.date,
        verify_sources=False,  # already verified above
        augment=not args.skip_augment,
        manifest_sha256=manifest_sha,
    )
    json_bytes = canonical_json_bytes(catalog)
    c = catalog["counts"]
    print(
        f"catalog: {c['standard_sheets']} standard sheets "
        f"({c['eligible']} eligible, {c['excluded']} excluded) "
        f"| by_category {c['by_category']} "
        f"| supporting {c['supporting_overview_pages']} overviews, "
        f"{c['supporting_textured_maps']} textured maps"
    )
    source_sha_by_file = {a["file"]: a["sha256"] for a in manifest["assets"]}
    plan = thumbnail_plan(catalog)
    print(f"thumbnail plan: {len(plan)} thumbnails "
          f"(sheets 360w, overviews/maps 320w, lossless PNG, LANCZOS)")

    # Stage 3 -- determinism proof: two fresh temp builds (thumbnails +
    # catalog + html), byte compare of every file.
    with tempfile.TemporaryDirectory(prefix="stamp_palette_det_") as tmp1, \
         tempfile.TemporaryDirectory(prefix="stamp_palette_det_") as tmp2:
        try:
            wrote1 = _build_into(Path(tmp1), library, catalog, source_sha_by_file)
            wrote2 = _build_into(Path(tmp2), library, catalog, source_sha_by_file)
        except ThumbnailError as exc:
            print(f"FAILURE: thumbnail generation {exc}", file=sys.stderr)
            return 3
        if wrote1 != wrote2:
            diffs = [f for f in wrote1 if wrote1.get(f) != wrote2.get(f)]
            print(f"FAILURE: determinism check {diffs} differ between temp runs",
                  file=sys.stderr)
            return 4
        print(f"determinism check: two fresh temp builds byte-identical "
              f"({len(wrote1)} files incl. {len(plan)} thumbnails; "
              f"catalog.json {wrote1['catalog.json']}, "
              f"index.html {wrote1['index.html']})")

    # Stage 4 -- canonical write (same full generation into the real out dir).
    try:
        written = _build_into(out_dir, library, catalog, source_sha_by_file)
    except (ThumbnailError, OSError) as exc:
        print(f"FAILURE: canonical write {exc}", file=sys.stderr)
        return 5
    print(f"canonical output written: {out_dir}")

    # Stage 5 -- post-write verification: thumbnails re-checked on disk,
    # links (incl. thumbnails), counts, embedded JSON.
    with open(out_dir / "catalog.json", "r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    if on_disk["counts"]["standard_sheets"] != c["standard_sheets"]:
        print("FAILURE: canonical catalog counts disagree", file=sys.stderr)
        return 6
    thumb_results = []
    for e in on_disk["entries"]:
        thumb_results.append(e["thumbnail"])
    for group in ("overview_pages", "textured_maps"):
        for item in on_disk["supporting"][group]:
            thumb_results.append(item["thumbnail"])
    problems = verify_thumbnails(out_dir / stamp_palette.THUMBNAIL_DIR,
                                 thumb_results)
    if problems:
        print(f"FAILURE: thumbnail re-verification {len(problems)} problem(s)",
              file=sys.stderr)
        for p in problems[:20]:
            print("  " + p, file=sys.stderr)
        return 7
    print(f"thumbnail re-verification: {len(thumb_results)}/{len(thumb_results)} ok "
          "(dims, aspect, sha256, nonblank)")
    problems = check_relative_links(on_disk, library, out_dir)
    if problems:
        print(f"FAILURE: link check {len(problems)} problem(s)", file=sys.stderr)
        for p in problems[:20]:
            print("  " + p, file=sys.stderr)
        return 8
    print("link check: all relative links resolve inside the library root "
          "(cards, originals, thumbnails, supporting)")
    print("canonical catalog.json parses; counts consistent")

    print("output inventory:")
    for f, h in written.items():
        print(f"  {out_dir / f}  {h}")
    nonblank = [r for r in thumb_results if r.get("nonblank")]
    print(f"thumbnails: {len(nonblank)} nonblank validated, "
          f"mean_luma range {min(r['mean_luma'] for r in thumb_results)}-"
          f"{max(r['mean_luma'] for r in thumb_results)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CatalogError as exc:
        print(f"FAILURE: catalog build {exc}", file=sys.stderr)
        sys.exit(1)
