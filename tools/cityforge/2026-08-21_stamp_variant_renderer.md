# 2026-08-21 Stamp Variant Renderer

`tools/cityforge/render_stamp_variants.py` renders controlled source-stamp
variants from a JSON config. It removes explicitly listed `source_id` members,
filters the corresponding contact edges, writes each filtered stamp JSON, and
invokes `render_generated_house.py` for a six-view Blender sheet.

The tool is diagnostic only: it never edits the source stamp or source ESP.
Use `input_stamp_id` when `input_stamp` is a library document, and define each
variant with `variant_id` plus `exclude_source_ids`. The output summary records
retained members, shell refs, and contact-edge counts.
