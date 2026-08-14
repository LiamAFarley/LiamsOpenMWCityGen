"""Markarth stamp-palette catalog builder (Cityforge T0.5).

Purpose
-------
Deterministically turn a final split-render library's authoritative
``render_library_manifest.json`` into a browsable, self-contained static
stamp palette:

* ``catalog.json``  - canonical machine-readable catalog (one record per
  manifest ``standard_sheet``, never any other asset kind, never any
  image bytes; each record carries its compact-thumbnail provenance).
* ``thumbnails/``    - deterministic lossless PNG thumbnails derived from the
  actual final source sheets (see :mod:`procgen.stamp_thumbnails`), used by
  the browser cards for fast reliable loading.
* ``index.html``    - single-file, dependency-free browser over the catalog
  that works from ``file:///`` (embedded canonical JSON, local CSS/JS,
  category tabs, live search, status filters, full-sheet lightbox,
  supporting overview/map links, explicit red excluded state).  Default
  view is the eligible Building Units category (per task contract).

The module treats the manifest as the single inventory/provenance authority:
every catalogued image must exist, must match its manifest SHA-256 and
dimensions, and loose files absent from the manifest are never added.  The
two user-reported defective Castle Barracks sheets are quarantined by exact
file name into a ``Needs Repair / Excluded`` status with the exact recorded
reason; this module never reinterprets that user assessment from the render
or the manifest.

Inputs
------
* ``render_library_manifest.json`` (read-only, from
  ``output/settlement-splits/markarth-side-v2/final-markarth-extraction-2026-08-10-library``)
* The library root directory holding the flat PNG assets it describes
  (read-only).
* Optional augmentation: ``artifacts/<label>/building_spec.json`` files
  inside the library (read-only) for marker-edge metadata.

Outputs
-------
* A catalog ``dict`` (serialized by :func:`canonical_json_bytes`) and an
  ``index.html`` string; written by the CLI
  ``tools/cityforge/stamp_palette.py`` into ``<library>/stamp_palette_v1/``.
* Never writes into the read-only library root itself.

Invariants
----------
* Byte-deterministic: same inputs + same ``date`` argument -> byte-identical
  JSON and HTML.  No wall clock, no randomness, no filesystem ordering.
* Exactly the manifest ``standard_sheet`` records appear as entries: 105
  total, 103 eligible, 2 excluded (only the two Castle Barracks sheets).
* No overview/map/rendered-asset image is ever classified as a stamp.
* All relative links stay inside the library root (``../<file>.png``).
* Every output contains no source image bytes.

Pipeline position
-----------------
* Feeds: the accepted final Markarth Side v2 extraction library
  (``package_split_render_library.py`` output), read-only.
* Consumed by: ``tools/cityforge/stamp_palette.py`` CLI and
  ``tests/test_stamp_palette.py`` (+ integration test).
* Does NOT re-render, edit, move, rename, recompress, or re-hash any source
  image; does not touch split outputs or the earlier D-STAMP v1 catalog
  (``output/cityforge/stamps/``).
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
CATALOG_ID = "markarth-stamp-palette-v1"

MANIFEST_FILE = "render_library_manifest.json"
STANDARD_SHEET_DIMENSIONS = (2304, 1536)

# User-directed quarantine: exact file names that must never be eligible.
# Reason text is the exact user authority wording; do not reinterpret.
EXCLUSIONS = {
    "castle_barracks_sheet_2x3.png": "user-reported defective Castle Barracks extraction",
    "castle_barracks__elfstone_keep__connection_sheet_2x3.png": "user-reported defective Castle Barracks extraction",
}

CATEGORY_ORDER = ("building_unit", "connection", "residual", "fused", "excluded")
CATEGORY_TITLES = {
    "building_unit": "Building Units",
    "connection": "Connections",
    "residual": "Residual/Unassigned",
    "fused": "Fused/Special",
    "excluded": "Needs Repair / Excluded",
}

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_REF_CELL_RE = re.compile(r"^(-?\d+)_(-?\d+)_ref_")
_CONN_SUFFIX_RE = re.compile(r"__connection(?:_c(\d+)(?:_(\d+))?)?$")
_FUSED_SUFFIX_RE = re.compile(r"__fused(?:_(\d+))?$")
_RESIDUAL_RE = re.compile(r"^residual_(c\d+)_(\d+)$")
_LEADING_UNDERSCORES_RE = re.compile(r"^_+")
_HUMAN_POSSESSIVE_RE = re.compile(r"_s_")


class CatalogError(Exception):
    """Raised for any catalog-contract violation (bad manifest, missing
    source, hash/dimension mismatch, classification gap)."""


class SourceVerificationError(CatalogError):
    """One or more source assets failed existence/hash/dimension checks."""


# ---------------------------------------------------------------------------
# Filesystem helpers (stdlib only, deterministic)
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, streamed in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def png_dimensions(path: Path) -> Tuple[int, int]:
    """Return ``(width, height)`` of a PNG by parsing its IHDR chunk.

    Reads only the fixed 24-byte header (signature 8 + length 4 + 'IHDR' 4
    + width 4 + height 4); no image decoding, no third-party dependency.
    """
    with open(path, "rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != _PNG_SIGNATURE:
        raise CatalogError(f"not a PNG (bad signature): {path}")
    if head[12:16] != b"IHDR":
        raise CatalogError(f"PNG missing IHDR chunk: {path}")
    width = int.from_bytes(head[16:20], "big")
    height = int.from_bytes(head[20:24], "big")
    return width, height


# ---------------------------------------------------------------------------
# Manifest loading and source verification
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> Tuple[Dict[str, Any], str]:
    """Load the manifest dict and return ``(manifest, sha256_of_file_bytes)``."""
    if not manifest_path.is_file():
        raise CatalogError(f"manifest not found: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"manifest unreadable/invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or "assets" not in manifest:
        raise CatalogError("manifest missing top-level 'assets' list")
    return manifest, manifest_sha


def verify_asset_sources(
    library_root: Path, manifest: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Verify every manifest asset on disk (existence, SHA-256, dimensions).

    Returns one result dict per manifest asset:
    ``{"file", "kind", "ok", "errors": [...]}``.  Raises
    :class:`SourceVerificationError` if any asset fails, listing every
    failing asset (capped) so the caller never proceeds on partial truth.
    """
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise CatalogError("manifest 'assets' is not a list")
    results: List[Dict[str, Any]] = []
    failures: List[str] = []
    for asset in assets:
        if not isinstance(asset, dict) or "file" not in asset:
            raise CatalogError("manifest contains a malformed asset record")
        file = asset["file"]
        kind = asset.get("kind", "?")
        path = library_root / file
        errors: List[str] = []
        if not path.is_file():
            errors.append("missing file")
        else:
            if asset.get("sha256") != sha256_file(path):
                errors.append("sha256 mismatch")
            try:
                actual = png_dimensions(path)
            except CatalogError as exc:
                errors.append(str(exc))
            else:
                expected = tuple(asset.get("dimensions", []))
                if len(expected) != 2 or actual != expected:
                    errors.append(
                        f"dimensions {actual} != manifest {expected}"
                    )
        ok = not errors
        results.append({"file": file, "kind": kind, "ok": ok, "errors": errors})
        if not ok:
            failures.append(f"{file}: {', '.join(errors)}")
    if failures:
        shown = failures[:40]
        more = f" (+{len(failures) - len(shown)} more)" if len(failures) > 40 else ""
        raise SourceVerificationError(
            "source verification failed for {} asset(s): {}{}".format(
                len(failures), "; ".join(shown), more
            )
        )
    return results


# ---------------------------------------------------------------------------
# Classification, exclusion and naming (deterministic)
# ---------------------------------------------------------------------------


def classify_kind(split_record_id: str) -> str:
    """Fallback kind inference from a split record id prefix.

    Used only when the manifest ``buildings`` list has no entry for the
    record (a contradiction that :func:`build_catalog` fails on before this
    fallback can matter); kept as an explicit documented rule.
    """
    if re.match(r"^u\d+_", split_record_id):
        return "unit"
    if split_record_id.startswith("conn_"):
        return "connection"
    if split_record_id.startswith("residual_"):
        return "residual"
    if split_record_id.startswith("fused_"):
        return "fused_review"
    raise CatalogError(f"cannot classify split record id: {split_record_id!r}")


KIND_TO_CATEGORY = {
    "unit": "building_unit",
    "connection": "connection",
    "residual": "residual",
    "fused_review": "fused",
}


def cells_from_refs(refs: Sequence[str]) -> List[str]:
    """Sorted unique ``"x,y"`` cell coordinates extracted from ref ids of
    the form ``-100_20_ref_028292`` (component/cell provenance for search)."""
    cells = set()
    for ref in refs:
        m = _REF_CELL_RE.match(ref)
        if m:
            cells.add(f"{m.group(1)},{m.group(2)}")
    return sorted(cells)


# Common English title particles lowered mid-name ("College of the Voice",
# "Grumm the Muttering's House") but never when they start the name.
_TITLE_PARTICLES = frozenset(
    {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to"}
)


def title_case_human(text: str) -> str:
    """Title-case a human name while respecting possessives and hyphens.

    Capitalizes only the first character of every space- or hyphen-separated
    token when it is alphabetic and leaves the remainder untouched, so
    ``"water's edge tavern"`` -> ``"Water's Edge Tavern"``,
    ``"torentius's house"`` -> ``"Torentius's House"`` and
    ``"02's house"`` -> ``"02's House"`` (the possessive ``'s`` after a
    leading numeral is never capitalized).  Common title particles
    (``of``, ``the``, ...) are kept lowercase mid-name but capitalized at
    the start: ``"college of the voice"`` -> ``"College of the Voice"``.
    """
    words = text.split(" ")
    out_words = []
    for idx, word in enumerate(words):
        if not word:
            continue
        parts = word.split("-")
        parts = [p[:1].upper() + p[1:] if p[:1].isalpha() else p for p in parts]
        titled = "-".join(parts)
        if idx > 0 and word.lower() in _TITLE_PARTICLES:
            titled = word.lower()
        out_words.append(titled)
    return " ".join(out_words)


_UNIT_PREFIX_RE = re.compile(r"^(?:unit_)?u\d+_")


def humanize_label(label: str) -> str:
    """Deterministic slug-to-display fallback for labels without a manifest
    ``building_keys`` entry: strip prefixes, convert ``_s_`` to a
    possessive apostrophe, underscores to spaces, then title-case.

    ``unit_u13_02_s_house`` -> ``"02's House"``
    ``residual_c31_001``   -> ``"Residual C31 001"``
    """
    text = _LEADING_UNDERSCORES_RE.sub("", label)
    text = _UNIT_PREFIX_RE.sub("", text)
    text = text.removesuffix("_fused").removesuffix("_connection")
    text = _HUMAN_POSSESSIVE_RE.sub("'s ", text).rstrip()
    text = text.replace("_", " ").strip()
    return title_case_human(text)


def connection_suffix(label: str) -> Optional[str]:
    """Extract a human connection-sheet variant suffix from a label tail.

    ``...__connection``        -> None
    ``...__connection_c49``    -> "C49"
    ``...__connection_c31_2``  -> "C31-2"
    """
    m = _CONN_SUFFIX_RE.search(label)
    if not m or (m.group(1) is None):
        return None
    suffix = f"C{m.group(1)}"
    if m.group(2) is not None:
        suffix += f"-{m.group(2)}"
    return suffix


def display_name_for(
    kind: str, label: str, building_keys: Sequence[str]
) -> Tuple[str, List[str], Optional[str]]:
    """Return ``(display_name, participants, connection_suffix)``.

    Units use the manifest's source-plugin human name (``building_keys[0]``)
    with leading underscores stripped (``___02's house`` -> ``02's House``).
    Connections/fused joins render their double-underscore participants
    clearly with `` × `` separators and a ``— Connection [suffix]`` /
    ``— Fused`` tail.  Residuals keep their stable ``Residual C31 001``
    form (zero-padded so plain string sorting stays numeric).  When
    ``building_keys`` is missing the name is derived from the label.
    """
    keys = [k for k in (building_keys or []) if k]
    if kind == "unit":
        base = keys[0] if keys else humanize_label(label)
        base = _LEADING_UNDERSCORES_RE.sub("", base)
        return title_case_human(base), [], None
    if kind in ("connection", "fused_review"):
        if keys:
            participants = [title_case_human(_LEADING_UNDERSCORES_RE.sub("", k)) for k in keys]
        else:
            body = label
            if kind == "connection":
                body = _CONN_SUFFIX_RE.sub("", label)
            elif _FUSED_SUFFIX_RE.search(label):
                body = _FUSED_SUFFIX_RE.sub("", label)
            participants = [humanize_label(p) for p in body.split("__")]
        tail = "Fused" if kind == "fused_review" else "Connection"
        suffix = None if kind == "fused_review" else connection_suffix(label)
        name = " × ".join(participants) + f" — {tail}"
        if suffix:
            name += f" {suffix}"
        return name, participants, suffix
    if kind == "residual":
        m = _RESIDUAL_RE.match(label)
        if m:
            return f"Residual {m.group(1).upper()} {m.group(2)}", [], None
        return humanize_label(label), [], None
    raise CatalogError(f"unknown building kind: {kind!r}")


def normalized_sort_name(name: str) -> str:
    """Lowercase, whitespace-collapsed sort key for a display name."""
    return " ".join(name.lower().split())


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------


def artifact_dir_for(file: str) -> str:
    """Artifact directory name for a sheet file: file stem minus the
    ``_sheet_2x3`` suffix (e.g. ``water_s_edge_tavern_sheet_2x3.png`` ->
    ``water_s_edge_tavern``)."""
    if not file.endswith("_sheet_2x3.png"):
        raise CatalogError(f"unexpected standard sheet name: {file!r}")
    return file[: -len("_sheet_2x3.png")]


def build_catalog(
    library_root: Path,
    manifest: Dict[str, Any],
    date: str,
    verify_sources: bool = True,
    augment: bool = True,
    manifest_sha256: str = "",
) -> Dict[str, Any]:
    """Build the complete catalog dict from a loaded manifest.

    ``verify_sources=True`` (default) runs the full existence/SHA/dimension
    gate over all manifest assets and raises :class:`SourceVerificationError`
    on any failure.  ``date`` is an explicit provenance label (YYYY-MM-DD)
    that keeps output byte-deterministic; no wall clock is ever read.
    ``manifest_sha256`` is recorded in ``source.manifest_sha256`` for
    provenance (the CLI passes the hash of the manifest file bytes).
    """
    root = Path(library_root)
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise CatalogError("manifest 'assets' is not a list")
    buildings = manifest.get("buildings")
    if not isinstance(buildings, list):
        raise CatalogError("manifest 'buildings' is not a list")

    if verify_sources:
        verify_asset_sources(root, manifest)

    # Cross-check the manifest's own internal bookkeeping before trusting it.
    sheets = [a for a in assets if a.get("kind") == "standard_sheet"]
    declared = manifest.get("sheet_count")
    if not isinstance(declared, int) or declared != len(sheets):
        raise CatalogError(
            f"manifest contradiction: sheet_count {declared} != standard_sheet "
            f"records {len(sheets)}"
        )
    buildings_with_sheets = sum(1 for b in buildings if b.get("has_standard_sheet"))
    if buildings_with_sheets != len(sheets):
        raise CatalogError(
            f"manifest contradiction: buildings with standard sheets "
            f"{buildings_with_sheets} != standard_sheet records {len(sheets)}"
        )
    file_seen = {}
    for a in sheets:
        f = a["file"]
        if f in file_seen:
            raise CatalogError(f"duplicate standard sheet file in manifest: {f}")
        file_seen[f] = a
        if tuple(a.get("dimensions", [])) != STANDARD_SHEET_DIMENSIONS:
            raise CatalogError(
                f"manifest contradiction: {f} dimensions {a.get('dimensions')} "
                f"!= expected {STANDARD_SHEET_DIMENSIONS}"
            )

    buildings_by_id = {}
    for b in buildings:
        rid = b.get("split_record_id")
        if not rid:
            raise CatalogError("manifest 'buildings' entry missing split_record_id")
        if rid in buildings_by_id:
            raise CatalogError(f"duplicate buildings entry: {rid}")
        buildings_by_id[rid] = b
        if b.get("has_standard_sheet") and b.get("sheet") not in file_seen:
            raise CatalogError(
                f"manifest contradiction: buildings entry {rid} references "
                f"sheet {b.get('sheet')} absent from standard sheets"
            )

    # Optional building_spec augmentation (read-only; absence is never a
    # reason to omit an asset).
    augmented = {}
    if augment:
        for f in sorted(file_seen):
            spec = root / "artifacts" / artifact_dir_for(f) / "building_spec.json"
            if spec.is_file():
                try:
                    with open(spec, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    data = None
                if isinstance(data, dict):
                    augmented[f] = {
                        "marker_edge_count": data.get("marker_edge_count"),
                        "marker_edge_pair_count": (
                            len(data.get("marker_edges"))
                            if isinstance(data.get("marker_edges"), list)
                            else None
                        ),
                        "building_spec_member_count": data.get("member_count"),
                    }

    entries: List[Dict[str, Any]] = []
    for a in sheets:
        f = a["file"]
        rid = a["split_record_id"]
        building = buildings_by_id.get(rid)
        if building is None:
            raise CatalogError(
                f"manifest contradiction: standard sheet {f} has no buildings entry for {rid}"
            )
        kind = building.get("kind")
        if kind not in KIND_TO_CATEGORY:
            raise CatalogError(f"unknown buildings kind {kind!r} for {f}")
        category = KIND_TO_CATEGORY[kind]

        status = "excluded" if f in EXCLUSIONS else "eligible"
        reason = EXCLUSIONS.get(f)
        # Category drives the palette tabs: excluded entries live only under
        # "Needs Repair / Excluded" (per user authority) while the underlying
        # classification is preserved as original_category for provenance.
        effective_category = "excluded" if status == "excluded" else category

        label = building.get("label") or a.get("source_slug") or f
        display_name, participants, suffix = display_name_for(
            kind, label, building.get("building_keys", [])
        )

        member_refs = list(a.get("member_refs") or [])
        context_refs = list(a.get("context_refs") or [])
        source_refs = list(a.get("source_refs") or [])
        spec = augmented.get(f) or {}

        entries.append(
            {
                "file": f,
                "kind": "standard_sheet",
                "category": effective_category,
                "original_category": category,
                "status": status,
                "excluded_reason": reason,
                "display_name": display_name,
                "participants": participants,
                "connection_suffix": suffix,
                "slug": a.get("source_slug"),
                "label": label,
                "split_record_id": rid,
                "component_id": building.get("component_id"),
                "cells": cells_from_refs(member_refs + source_refs + context_refs),
                "member_count": len(member_refs),
                "context_ref_count": len(context_refs),
                "source_ref_count": len(source_refs),
                "member_refs": member_refs,
                "context_refs": context_refs,
                "source_refs": source_refs,
                "dimensions": list(a.get("dimensions")),
                "sha256": a.get("sha256"),
                "sha256_short": (a.get("sha256") or "")[:8],
                "marker_edge_count": spec.get("marker_edge_count"),
                "marker_edge_pair_count": spec.get("marker_edge_pair_count"),
                "augmented_from_building_spec": bool(spec),
                "links": {"sheet": "../" + f},
            }
        )

    # Deterministic sort: category order, normalized display name, record id, file.
    entries.sort(
        key=lambda e: (
            CATEGORY_ORDER.index(e["category"]),
            normalized_sort_name(e["display_name"]),
            e["split_record_id"],
            e["file"],
        )
    )

    overview_pages = []
    textured_maps = []
    for a in assets:
        if a.get("kind") == "overview":
            overview_pages.append(
                {
                    "file": a["file"],
                    "dimensions": list(a.get("dimensions")),
                    "sha256_short": (a.get("sha256") or "")[:8],
                    "slot_count": len(a.get("slots") or []),
                    "links": {"image": "../" + a["file"]},
                }
            )
        elif a.get("kind") == "map" and a["file"].endswith("_textured.png"):
            textured_maps.append(
                {
                    "file": a["file"],
                    "dimensions": list(a.get("dimensions")),
                    "sha256_short": (a.get("sha256") or "")[:8],
                    "links": {"image": "../" + a["file"]},
                }
            )
    overview_pages.sort(key=lambda x: x["file"])
    textured_maps.sort(key=lambda x: x["file"])

    by_category = {c: 0 for c in ("building_unit", "connection", "residual", "fused")}
    for e in entries:
        if e["status"] == "eligible":
            by_category[e["category"]] += 1

    catalog: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "catalog_id": CATALOG_ID,
        "date": date,
        "source": {
            "library_root": str(root.resolve()),
            "manifest": MANIFEST_FILE,
            "manifest_sha256": manifest_sha256,
            "input_hashes": manifest.get("input_hashes", {}),
        },
        "rules": {
            "classification": (
                "category comes from manifest buildings[].kind mapped by split_record_id "
                "(unit -> building_unit, connection -> connection, residual -> residual, "
                "fused_review -> fused); fallback inference from split_record_id prefixes "
                "u<digits>_ / conn_ / residual_ / fused_ is defined but never reached when "
                "the manifest is consistent"
            ),
            "status": (
                "every manifest standard_sheet starts eligible; the two user-directed "
                "Castle Barracks exclusions are applied by exact file name before any "
                "eligible counting and can never become eligible"
            ),
            "exclusion_precedence": (
                "explicit exclusion file-name match overrides classification-derived "
                "category; excluded records appear only under 'Needs Repair / Excluded' "
                "with the exact recorded reason"
            ),
            "sort": (
                "category order (building_unit, connection, residual, fused, excluded), "
                "then lowercase normalized display name, then split_record_id, then file"
            ),
            "generated_files": (
                "index.html entry hashes the exact HTML file bytes; catalog.json entry "
                "hashes the canonical catalog payload excluding the self-referential "
                "generated_files field (a file cannot contain an exact hash of itself); "
                "the CLI run report records the on-disk file hashes too"
            ),
            "thumbnails": (
                "card images are compact lossless PNG thumbnails generated at build "
                "time from the actual final source PNGs (LANCZOS downscale, no crop, "
                "aspect and full sheet preserved; 360px wide for sheets, 320px for "
                "overviews/maps); each thumbnail record carries its own sha256 plus "
                "source_file/source_sha256 provenance and nonblank validation metrics "
                "(sampled color buckets >= 24, mean luma >= 12); the original PNG "
                "remains the lightbox/original-link target"
            ),
        },
        "exclusions": [
            {"file": f, "reason": EXCLUSIONS[f]} for f in sorted(EXCLUSIONS)
        ],
        "counts": {
            "total_manifest_assets": len(assets),
            "standard_sheets": len(sheets),
            "eligible": sum(1 for e in entries if e["status"] == "eligible"),
            "excluded": sum(1 for e in entries if e["status"] == "excluded"),
            "by_category": by_category,
            "supporting_overview_pages": len(overview_pages),
            "supporting_textured_maps": len(textured_maps),
        },
        "supporting": {
            "overview_pages": overview_pages,
            "textured_maps": textured_maps,
        },
        "entries": entries,
    }
    return catalog


def canonical_json_bytes(catalog: Dict[str, Any]) -> bytes:
    """Serialize the catalog to canonical bytes: sorted keys, compact
    separators, ASCII-escaped non-ASCII (byte-stable across platforms)."""
    return json.dumps(
        catalog, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Link safety and output writing
# ---------------------------------------------------------------------------


# Thumbnail sizing (documented in rules.thumbnails): cards show the whole
# 2x3 sheet at 360px wide (3:2 -> 360x240), supporting renders at 320px
# wide; height is derived from the source aspect so nothing is cropped.
THUMBNAIL_WIDTHS = {"standard_sheet": 360, "overview": 320, "map": 320}
THUMBNAIL_DIR = "thumbnails"


def thumbnail_plan(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Pure plan of thumbnails to generate: rel path -> spec.

    One entry per catalogued standard sheet and per supporting
    overview/textured-map item; ``source_dims`` come from the catalog
    records so the generator can verify it decodes exactly the right file.
    """
    plan: Dict[str, Dict[str, Any]] = {}
    for e in catalog["entries"]:
        width = THUMBNAIL_WIDTHS[e["kind"]]
        sw, sh = e["dimensions"]
        height = max(1, round(sh * width / sw))
        plan[f"{THUMBNAIL_DIR}/{e['file']}"] = {
            "source_file": e["file"],
            "kind": e["kind"],
            "width": width,
            "height": height,
            "source_dims": [sw, sh],
        }
    for group, kind in (("overview_pages", "overview"), ("textured_maps", "map")):
        for item in catalog["supporting"][group]:
            width = THUMBNAIL_WIDTHS[kind]
            sw, sh = item["dimensions"]
            height = max(1, round(sh * width / sw))
            plan[f"{THUMBNAIL_DIR}/{item['file']}"] = {
                "source_file": item["file"],
                "kind": kind,
                "width": width,
                "height": height,
                "source_dims": [sw, sh],
            }
    return plan


_THUMB_RECORD_KEYS = (
    "file", "source_file", "source_sha256", "width", "height", "aspect",
    "sha256", "sha256_short", "nonblank", "sampled_color_buckets", "mean_luma",
)


def attach_thumbnail_metadata(
    catalog: Dict[str, Any], thumb_results: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a copy of the catalog with thumbnail provenance merged in.

    Every entry/supporting item gains ``thumbnail`` (file, dims, aspect,
    own sha256, source sha256, nonblank metrics) and ``links.thumb``.
    ``counts.thumbnails_generated`` records how many were produced.  Pure
    and deterministic: identical inputs -> identical output dict.
    """
    by_source = {r["source_file"]: r for r in thumb_results}
    out = json.loads(json.dumps(catalog))  # deep copy, keeps sorted order
    for e in out["entries"]:
        r = by_source[e["file"]]
        e["thumbnail"] = {k: r[k] for k in _THUMB_RECORD_KEYS if k in r}
        e["links"]["thumb"] = r["file"]
    for group in ("overview_pages", "textured_maps"):
        for item in out["supporting"][group]:
            r = by_source[item["file"]]
            item["thumbnail"] = {k: r[k] for k in _THUMB_RECORD_KEYS if k in r}
            item["links"]["thumb"] = r["file"]
    counts = dict(out["counts"])
    counts["thumbnails_generated"] = len(thumb_results)
    out["counts"] = counts
    return out


def check_relative_links(
    catalog: Dict[str, Any], library_root: Path, out_dir: Optional[Path] = None
) -> List[str]:
    """Resolve every relative link in the catalog and verify it exists and
    stays inside the library root.

    Links are written for the palette living one level below the library
    root (``<library>/stamp_palette_v1/``), so ``../<file>.png`` is the only
    legal shape; ``out_dir`` (default ``<root>/stamp_palette_v1``) is the
    base the browser would resolve them from.  Returns a list of problem
    descriptions (empty == all links resolve and stay inside the root).
    """
    root = Path(library_root).resolve()
    base = (Path(out_dir) if out_dir is not None else root / "stamp_palette_v1").resolve()
    problems: List[str] = []
    for entry in catalog["entries"]:
        for name, uri in entry.get("links", {}).items():
            p = (base / uri).resolve()
            if not p.is_file():
                problems.append(f"missing target for {entry['file']} link {name}: {uri}")
            if root not in p.parents and p != root:
                problems.append(f"link escapes library root: {uri}")
    for group in ("overview_pages", "textured_maps"):
        for item in catalog["supporting"][group]:
            for name, uri in item.get("links", {}).items():
                p = (base / uri).resolve()
                if not p.is_file():
                    problems.append(f"missing supporting target {name}: {uri}")
                if root not in p.parents and p != root:
                    problems.append(f"supporting link escapes library root: {uri}")
    return problems


def _escape_script_json(raw: str) -> str:
    """Escape a JSON text for embedding inside a <script> element.

    ``</`` is the only sequence that can terminate the script block; the
    JSON-legal ``\/`` escape neutralizes it without changing meaning.
    """
    return raw.replace("</", "<\\/")


def render_html(catalog: Dict[str, Any], canonical_json: bytes) -> str:
    """Render the single-file ``index.html`` string.

    The canonical JSON bytes are embedded verbatim (with only the
    script-safe ``</`` -> ``<\\/`` escape) in a ``<script type="application/json">``
    block; all cards, tabs, filters and lightboxes are rendered client-side
    from that embedded data, so the HTML can never drift from the canonical
    JSON.  No external assets, no network.
    """
    data = _escape_script_json(canonical_json.decode("utf-8"))
    manifest_sha = catalog["source"]["manifest_sha256"]
    counts = catalog["counts"]
    count_chips = (
        f'<span class="chip">{counts["standard_sheets"]} standard sheets</span>'
        f'<span class="chip">{counts["eligible"]} eligible</span>'
        f'<span class="chip">{counts["excluded"]} excluded</span>'
        f'<span class="chip">{counts["supporting_overview_pages"]} overview pages</span>'
        f'<span class="chip">{counts["supporting_textured_maps"]} textured maps</span>'
    )
    return _HTML_TEMPLATE.replace("__CATALOG_JSON__", data).replace(
        "__COUNT_CHIPS__", count_chips
    ).replace("__MANIFEST_SHA__", html.escape(manifest_sha))


def attach_inventory(
    catalog: Dict[str, Any], html_bytes: bytes
) -> Tuple[Dict[str, Any], bytes]:
    """Attach the ``generated_files`` inventory and return the final catalog
    plus its canonical serialized bytes.

    A file cannot carry an exact SHA-256 of itself inside its own bytes, so
    the convention is documented and exact: the ``catalog.json`` inventory
    entry is the SHA-256 of the canonical catalog payload *without* the
    self-referential ``generated_files`` field (a well-defined, deterministic
    digest of everything else the file contains), while the ``index.html``
    entry is the SHA-256 of the exact HTML file bytes.  The CLI additionally
    reports the on-disk file hashes after writing.
    """
    payload = canonical_json_bytes(catalog)
    inventory = [
        {"file": "catalog.json", "sha256": hashlib.sha256(payload).hexdigest()},
        {"file": "index.html", "sha256": hashlib.sha256(html_bytes).hexdigest()},
    ]
    final = dict(catalog)
    final["generated_files"] = inventory
    return final, canonical_json_bytes(final)


def write_catalog(
    out_dir: Path, catalog: Dict[str, Any], html: str
) -> Dict[str, str]:
    """Write ``catalog.json`` + ``index.html`` into ``out_dir`` (created if
    needed) and return ``{file: sha256}`` of the exact written bytes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, json_bytes = attach_inventory(catalog, html.encode("utf-8"))
    paths = {
        "catalog.json": out_dir / "catalog.json",
        "index.html": out_dir / "index.html",
    }
    for file, p in paths.items():
        p.write_bytes(json_bytes if file == "catalog.json" else html.encode("utf-8"))
    return {file: sha256_file(p) for file, p in paths.items()}


# ---------------------------------------------------------------------------
# HTML template (self-contained; local CSS/JS only)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Markarth Side — Final Stamp Palette</title>
<style>
  :root {
    --bg: #10151c; --panel: #18202b; --panel2: #1e2836; --ink: #e8ecf2;
    --muted: #93a1b5; --line: #2c3a4d; --accent: #7ab3ff;
    --ok: #6fd18b; --bad: #ff6b6b; --badbg: #3a1418; --badline: #a33;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f4f6f9; --panel: #ffffff; --panel2: #eef2f7; --ink: #1c2733;
      --muted: #5a6a7d; --line: #d3dce6; --accent: #1a5fb4;
      --ok: #1d7a3d; --bad: #b3261e; --badbg: #fdecea; --badline: #d33;
    }
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--ink);
    font: 15px/1.45 "Segoe UI", system-ui, sans-serif;
  }
  header { padding: 18px 20px 6px; }
  header h1 { margin: 0 0 4px; font-size: 22px; }
  .subtitle { margin: 0 0 8px; color: var(--muted); }
  .srcinfo { margin: 0; font-size: 12.5px; color: var(--muted); word-break: break-all; }
  .srcinfo code { color: var(--accent); }
  .chips { margin-top: 8px; }
  .chip {
    display: inline-block; margin: 2px 6px 2px 0; padding: 2px 9px;
    border: 1px solid var(--line); border-radius: 12px; font-size: 12px;
    color: var(--muted); background: var(--panel);
  }
  .controls { position: sticky; top: 0; z-index: 5; padding: 10px 20px;
    background: color-mix(in srgb, var(--bg) 88%, transparent);
    backdrop-filter: blur(4px); border-bottom: 1px solid var(--line); }
  #search { width: 100%; max-width: 620px; padding: 8px 12px; font-size: 14px;
    border: 1px solid var(--line); border-radius: 8px; background: var(--panel);
    color: var(--ink); }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
  .tab, .fbtn {
    padding: 5px 12px; border: 1px solid var(--line); border-radius: 14px;
    background: var(--panel); color: var(--muted); cursor: pointer; font-size: 13px;
  }
  .tab.active, .fbtn.active { background: var(--accent); color: #fff;
    border-color: var(--accent); }
  .tab .n { font-weight: 700; }
  #countline { margin: 8px 0 0; font-size: 12.5px; color: var(--muted); }
  main.grid {
    display: grid; gap: 14px; padding: 16px 20px;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  }
  article.card {
    display: flex; flex-direction: column; min-width: 0;
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    overflow: hidden; transition: border-color .12s;
  }
  article.card:hover { border-color: var(--accent); }
  article.card.excluded {
    background: var(--badbg); border-color: var(--badline);
  }
  article.card.excluded:hover { border-color: var(--bad); }
  .thumb { display: block; aspect-ratio: 3/2; background: #000; cursor: zoom-in; }
  .thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .badge-row { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 10px 0; }
  .badge {
    font-size: 11px; padding: 1px 8px; border-radius: 9px;
    border: 1px solid var(--line); color: var(--muted); background: var(--panel2);
  }
  .badge.cat { color: var(--accent); }
  .badge.status-eligible { color: var(--ok); }
  .badge.status-excluded { color: var(--bad); border-color: var(--badline); }
  .title { margin: 6px 10px 0; font-size: 15.5px; line-height: 1.25; cursor: zoom-in; }
  article.card.excluded .title { color: var(--bad); }
  .raw { margin: 3px 10px 0; font-size: 11.5px; color: var(--muted);
    word-break: break-all; }
  .meta { margin: 6px 10px 0; font-size: 11.5px; color: var(--muted); }
  .meta b { color: var(--ink); font-weight: 600; }
  .cardfoot { margin-top: auto; padding: 8px 10px 10px; }
  .orig {
    display: inline-block; font-size: 12px; color: var(--accent);
    text-decoration: none; border: 1px solid var(--line); border-radius: 6px;
    padding: 3px 9px; background: var(--panel2);
  }
  .orig:hover { border-color: var(--accent); }
  .exreason { margin: 6px 10px 0; font-size: 11.5px; color: var(--bad); }
  section.supporting { padding: 6px 20px 20px; }
  section.supporting h2 { font-size: 17px; margin: 18px 0 4px; }
  section.supporting h3 { font-size: 13.5px; color: var(--muted); margin: 12px 0 6px; }
  .ovgrid {
    display: grid; gap: 10px;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  }
  .ovitem { background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; padding: 8px; min-width: 0; }
  .ovitem img { width: 100%; border-radius: 4px; display: block; }
  .ovitem .cap { font-size: 11.5px; color: var(--muted); margin-top: 5px;
    word-break: break-all; }
  footer { padding: 10px 20px 24px; font-size: 11.5px; color: var(--muted); }
  .lightbox {
    position: fixed; inset: 0; z-index: 50; display: flex; flex-direction: column;
    align-items: center; justify-content: center; background: rgba(5,8,12,.92);
    padding: 20px;
  }
  .lightbox[hidden] { display: none; }
  .lightbox img { max-width: 96vw; max-height: 78vh; object-fit: contain;
    background: #000; border-radius: 6px; }
  .lb-cap { color: #dfe6ef; margin-top: 10px; max-width: 900px; text-align: center; }
  .lb-cap .lb-raw { color: #8fa0b3; font-size: 12px; word-break: break-all; }
  .lb-close {
    position: absolute; top: 14px; right: 18px; font-size: 26px; color: #dfe6ef;
    background: none; border: none; cursor: pointer; padding: 6px 12px;
  }
  .lb-actions { margin-top: 8px; }
  .lb-actions a {
    color: var(--accent); border: 1px solid #445; border-radius: 6px;
    padding: 4px 12px; text-decoration: none; font-size: 13px; background: #18202b;
  }
  .empty { padding: 30px 20px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Markarth Side — Final Stamp Palette</h1>
  <p class="subtitle">Terrain-backed 2×3 building-unit sheets from the accepted final Markarth Side v2 extraction library</p>
  <p class="srcinfo">Authoritative manifest: <code>render_library_manifest.json</code> · sha256 <code>__MANIFEST_SHA__</code> · generated by <code>tools/cityforge/stamp_palette.py</code></p>
  <p class="chips">__COUNT_CHIPS__</p>
</header>
<nav class="controls">
  <input id="search" type="search" placeholder="Search display name, slug, split record id, component/cell, source ref id…" autocomplete="off">
  <div class="row" id="tabs"></div>
  <div class="row" id="statusfilters"></div>
  <p id="countline"></p>
</nav>
<main id="grid" class="grid"></main>
<section class="supporting">
  <h2>Supporting renders — not stamps</h2>
  <p class="subtitle">Overview sheets and textured component maps from the same library, linked directly for navigation.</p>
  <h3 id="ovh">Overview pages</h3>
  <div id="overviews" class="ovgrid"></div>
  <h3 id="maph">Textured component maps</h3>
  <div id="maps" class="ovgrid"></div>
</section>
<footer>
  Deterministic catalog generated from <code>render_library_manifest.json</code> (manifest sha256 <code>__MANIFEST_SHA__</code>).
  Source images are read-only; this palette adds no image bytes. Castle Barracks sheets are quarantined per user report.
</footer>
<div id="lightbox" class="lightbox" hidden>
  <button id="lb-close" class="lb-close" aria-label="Close">×</button>
  <img id="lb-img" alt="Full sheet">
  <p class="lb-cap"><span id="lb-title"></span><br><span class="lb-raw" id="lb-raw"></span></p>
  <p class="lb-actions"><a id="lb-orig" href="#" target="_blank" rel="noopener">Open original PNG</a></p>
</div>
<script type="application/json" id="catalog-data">__CATALOG_JSON__</script>
<script>
(function () {
  "use strict";
  var catalog = JSON.parse(document.getElementById("catalog-data").textContent);
  var entries = catalog.entries;
  var CATS = ["building_unit", "connection", "residual", "fused", "excluded"];
  var CAT_TITLES = {
    building_unit: "Building Units", connection: "Connections",
    residual: "Residual/Unassigned", fused: "Fused/Special",
    excluded: "Needs Repair / Excluded"
  };
  // Default view per the task contract: eligible named Building Units.
  var state = { q: "", cat: "building_unit", status: "eligible" };

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }
  function norm(s) { return String(s || "").toLowerCase().replace(/\\s+/g, " "); }

  function matches(e) {
    if (state.status === "eligible" && e.status !== "eligible") return false;
    if (state.status === "excluded" && e.status !== "excluded") return false;
    if (state.cat !== "all" && e.category !== state.cat) return false;
    var q = state.q;
    if (!q) return true;
    var hay = [
      e.display_name, e.slug, e.split_record_id, e.label, e.category,
      String(e.component_id == null ? "" : e.component_id),
      (e.cells || []).join(" "), (e.participants || []).join(" "),
      (e.member_refs || []).join(" "), (e.context_refs || []).join(" "),
      (e.source_refs || []).join(" "), (e.member_count || ""),
      (e.context_ref_count || ""), (e.source_ref_count || ""),
      (e.connection_suffix || ""), (e.excluded_reason || "")
    ].join(" | ");
    return norm(hay).indexOf(norm(q)) !== -1;
  }

  function cardHTML(e) {
    var badge = esc(CAT_TITLES[e.category]);
    var status = e.status === "excluded"
      ? '<span class="badge status-excluded">Needs Repair / Excluded</span>'
      : '<span class="badge status-eligible">Eligible</span>';
    var reason = e.status === "excluded" && e.excluded_reason
      ? '<p class="exreason">' + esc(e.excluded_reason) + "</p>" : "";
    var comp = e.component_id == null ? "—" : esc(String(e.component_id));
    var suffix = e.connection_suffix ? " · " + esc(e.connection_suffix) : "";
    return (
      '<article class="card ' + e.status + '" data-category="' + esc(e.category) +
      '" data-status="' + esc(e.status) + '" data-file="' + esc(e.file) + '">' +
      '<a class="thumb" data-lb="' + esc(e.file) + '" href="#" aria-label="Open ' +
      esc(e.display_name) + '"><img src="' + esc(e.links.thumb) + '" alt="' +
      esc(e.display_name) + ' sheet" decoding="async"></a>' +
      '<div class="badge-row"><span class="badge cat">' + badge + "</span>" + status +
      "<span class=\\"badge\\">" + esc(e.member_count) + " members</span>" +
      '<span class="badge">comp ' + comp + suffix + "</span></div>" +
      '<h2 class="title" data-lb="' + esc(e.file) + '">' + esc(e.display_name) + "</h2>" +
      '<p class="raw">' + esc(e.slug) + " · " + esc(e.split_record_id) + "</p>" +
      '<p class="meta">' + esc(e.member_count) + " members · " + esc(e.context_ref_count) +
      " context · " + esc(e.source_ref_count) + " source refs · " +
      esc(e.dimensions.join("×")) + " · sha " + esc(e.sha256_short) +
      (e.cells && e.cells.length ? " · cells " + esc(e.cells.join(", ")) : "") + "</p>" +
      reason +
      '<div class="cardfoot"><a class="orig" href="' + esc(e.links.sheet) +
      '" target="_blank" rel="noopener">Open original PNG</a></div>' +
      "</article>"
    );
  }

  function render() {
    var grid = document.getElementById("grid");
    var vis = entries.filter(matches);
    if (!vis.length) {
      grid.innerHTML = '<p class="empty">No sheets match the current search/filters.</p>';
    } else {
      grid.innerHTML = vis.map(cardHTML).join("");
    }
    var byCat = { building_unit: 0, connection: 0, residual: 0, fused: 0, excluded: 0 };
    entries.forEach(function (e) { byCat[e.category] += 1; });
    var tabs = document.getElementById("tabs");
    tabs.innerHTML = '<button class="tab' + (state.cat === "all" ? " active" : "") +
      '" data-cat="all">All <span class="n">' + entries.length + "</span></button>" +
      CATS.map(function (c) {
        return '<button class="tab' + (state.cat === c ? " active" : "") +
          '" data-cat="' + c + '">' + esc(CAT_TITLES[c]) + ' <span class="n">' +
          byCat[c] + "</span></button>";
      }).join("");
    var eligible = entries.filter(function (e) { return e.status === "eligible"; }).length;
    var excluded = entries.length - eligible;
    var f = document.getElementById("statusfilters");
    f.innerHTML = ["all", "eligible", "excluded"].map(function (s) {
      var n = s === "all" ? entries.length : (s === "eligible" ? eligible : excluded);
      var label = s === "all" ? "All statuses" : (s === "eligible" ? "Eligible" : "Excluded");
      return '<button class="fbtn' + (state.status === s ? " active" : "") +
        '" data-status="' + s + '">' + label + ' <span class="n">' + n + "</span></button>";
    }).join("");
    document.getElementById("countline").textContent =
      "Showing " + vis.length + " of " + entries.length + " standard sheets" +
      (state.q ? " (search: “" + state.q + "”)" : "");
    renderSupporting();
  }

  function renderSupporting() {
    var ov = document.getElementById("overviews");
    ov.innerHTML = catalog.supporting.overview_pages.map(function (p) {
      return '<div class="ovitem"><a href="' + esc(p.links.image) +
        '" target="_blank" rel="noopener"><img src="' + esc(p.links.thumb) +
        '" alt="' + esc(p.file) + '" decoding="async"></a><div class="cap">' +
        esc(p.file) + " · " + esc(p.dimensions.join("×")) + " · " +
        esc(p.slot_count) + " slots</div></div>";
    }).join("");
    document.getElementById("ovh").textContent =
      "Overview pages (" + catalog.supporting.overview_pages.length + ")";
    var mp = document.getElementById("maps");
    mp.innerHTML = catalog.supporting.textured_maps.map(function (p) {
      return '<div class="ovitem"><a href="' + esc(p.links.image) +
        '" target="_blank" rel="noopener"><img src="' + esc(p.links.thumb) +
        '" alt="' + esc(p.file) + '" decoding="async"></a><div class="cap">' +
        esc(p.file) + " · " + esc(p.dimensions.join("×")) + "</div></div>";
    }).join("");
    document.getElementById("maph").textContent =
      "Textured component maps (" + catalog.supporting.textured_maps.length + ")";
  }

  var lb = document.getElementById("lightbox");
  function openLb(file) {
    var e = entries.filter(function (x) { return x.file === file; })[0];
    if (!e) return;
    document.getElementById("lb-img").src = e.links.sheet;
    document.getElementById("lb-title").textContent = e.display_name;
    document.getElementById("lb-raw").textContent =
      e.slug + " · " + e.split_record_id + (e.status === "excluded" ? " · EXCLUDED — " + e.excluded_reason : "");
    document.getElementById("lb-orig").href = e.links.sheet;
    lb.hidden = false;
  }
  function closeLb() { lb.hidden = true; document.getElementById("lb-img").src = ""; }
  document.getElementById("grid").addEventListener("click", function (ev) {
    var t = ev.target.closest("[data-lb]");
    if (t) { ev.preventDefault(); openLb(t.getAttribute("data-lb")); }
  });
  document.getElementById("lb-close").addEventListener("click", closeLb);
  lb.addEventListener("click", function (ev) { if (ev.target === lb) closeLb(); });
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape" && !lb.hidden) closeLb();
  });
  document.getElementById("search").addEventListener("input", function (ev) {
    state.q = ev.target.value.trim();
    render();
  });
  document.getElementById("tabs").addEventListener("click", function (ev) {
    var b = ev.target.closest("[data-cat]");
    if (b) {
      state.cat = b.getAttribute("data-cat");
      // Needs Repair / Excluded must show both excluded records immediately,
      // without requiring a second status toggle.
      if (state.cat === "excluded") state.status = "all";
      render();
    }
  });
  document.getElementById("statusfilters").addEventListener("click", function (ev) {
    var b = ev.target.closest("[data-status]");
    if (b) { state.status = b.getAttribute("data-status"); render(); }
  });

  render();
})();
</script>
</body>
</html>
"""
