# 2026-08-20 Falkreath structural preview

`render_structural_components.py` reads `structural_library.json`, selects the
most-used extracted wall, gate, and fence component for a readable palette
layout, and sends the resulting D-STAMP through the existing Blender sheet
renderer. It is a visual inspection tool, not a wall-chain composer: source
world placements remain in the structural library and are not replaced by the
fixed preview spacing.
