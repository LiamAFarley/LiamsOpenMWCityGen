# 2026-08-10 — Markarth Stamp Palette Tool Family (Cityforge T0.5)

Date: 2026-08-10 · Task: bounded non-rendering catalog task
(`.opencode/runs/markarth-stamp-palette-v1/`)

## What this family does

Builds the deterministic, self-contained **stamp palette** for the accepted
final Markarth Side v2 extraction library: a browsable static catalog of the
105 terrain-backed 2×3 building-unit sheets, with the two user-reported
defective Castle Barracks sheets quarantined into a red
**Needs Repair / Excluded** state.

| Tool | Purpose |
|---|---|
| `src/procgen/stamp_palette.py` | Core engine: manifest loading + full source verification (existence, SHA-256, PNG dimensions for all 152 assets), classification, exclusion precedence, deterministic human naming, canonical JSON serialization, single-file HTML rendering, relative-link safety, generated-file inventory. |
| `tools/cityforge/stamp_palette.py` | CLI driver: verifies sources, builds the catalog, proves byte-determinism with two fresh temp runs, writes `<library>/stamp_palette_v1/`, re-verifies the written output. |
| `tests/test_stamp_palette.py` | 22 focused unit tests on synthetic fixtures (classification, errors, naming, exclusion precedence, determinism, escaping, link safety, inventory semantics). |
| `tests/test_stamp_palette_integration.py` | 9 integration tests against the real library + canonical output (counts, exclusions, naming spot-checks, link resolution, embedded JSON, structural re-derivation equality). |

## Outputs

- `.../final-markarth-extraction-2026-08-10-library/stamp_palette_v1/catalog.json`
  — canonical machine-readable catalog (105 records, no image bytes; every
  record carries thumbnail provenance: own sha256 + source file/sha256 +
  nonblank metrics).
- `.../final-markarth-extraction-2026-08-10-library/stamp_palette_v1/thumbnails/`
  — 138 deterministic lossless PNG thumbnails (105 sheets at 360×240 + 27
  overviews + 6 textured maps at 320px wide) derived from the actual final
  source PNGs (LANCZOS, no crop, aspect preserved); cards use these, the
  original PNGs stay for the lightbox/"Open original PNG".
- `.../final-markarth-extraction-2026-08-10-library/stamp_palette_v1/index.html`
  — single-file browser (embedded canonical JSON, local CSS/JS, works from
  `file:///`, dark/light themes, tabs, live search, lightbox, supporting
  overview/map links, red excluded state). Default view = eligible Building
  Units (56); Needs Repair / Excluded tab shows both excluded records
  directly.

## Commands

```powershell
python tools/cityforge/stamp_palette.py --date 2026-08-10
python -m unittest tests.test_stamp_palette tests.test_stamp_palette_integration
```

## Key facts (2026-08-10 run, after lead visual-review fixes)

- 152/152 manifest assets verified (hash + dimensions); manifest sha256
  `87ce3869b693ede135ae12e67bee1e1815eaa0a72e584943aac1215a4eca7ac0`.
- 105 standard sheets: 103 eligible (56 units, 27 connections, 19 residuals,
  1 fused) + 2 excluded (`castle_barracks_sheet_2x3.png`,
  `castle_barracks__elfstone_keep__connection_sheet_2x3.png`, reason
  `user-reported defective Castle Barracks extraction`).
- 138/138 thumbnails generated, validated nonblank (mean luma 130-192,
  sampled color buckets 24+), and re-verified on disk after the canonical
  write (dims, aspect, sha256, nonblank).
- Two fresh temp runs byte-identical across **all 140 output files**
  (catalog.json, index.html, 138 thumbnails); two separate CLI runs produced
  identical canonical hashes.
- Browser evidence (headless Edge + CDP, `--allow-file-access-from-files` for
  canvas sampling): every visible card image verified `complete`, nonzero
  natural size, and nonblank canvas-sampled pixels (mean luma 137-175 across
  all category views) BEFORE each screenshot; default view = 56 eligible
  Building Units; excluded tab = exactly the 2 Castle Barracks cards with
  their sheets visible in-card; lightbox shows the original 2304×1536 sheet.
  Screenshots + per-card evidence JSON are in this run directory.

Full guide: `Documentation/guides/markarth_stamp_palette.md`.
