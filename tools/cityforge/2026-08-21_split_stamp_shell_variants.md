# 2026-08-21 Split Stamp Shell Variants

`tools/cityforge/split_stamp_shell_variants.py` reconstructs a multi-shell
source stamp from complementary diagnostic JSONs and emits one standalone
variant per shell. Every variant retains every non-shell member and filters
contact evidence to the retained set. When `attachment_host_shell_id` is
provided, copied attachments are rebased through the host shell's local frame
onto each target shell; contact evidence is intentionally cleared because the
new geometry must be visually reviewed. It is a visual diagnostic and does not
edit source ESPs or canonical libraries.
