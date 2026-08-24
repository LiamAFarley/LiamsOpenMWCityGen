# 2026-08-20 Falkreath authoritative kit extraction

`falkreath_kit_extract.py` mirrors the dependency-aware Markarth source scan for
the five-layer `Sky_xFa` Falkreath reference set. It loads `Sky_Main.esm` and
`Tamriel_Data.esm` definitions, scans the five read-only xFa ESPs, selects
exterior cells whose names contain `Falkreath`, and records construction refs
with source layer, cell/grid, object id, model, category, and transform.

The output is an evidence bundle under
`output/cityforge/falkreath_kit_extraction_v1/`. It is input evidence for the
Falkreath house grammar; it does not modify the reference plugins or claim that
every selected reference is one generated house.
