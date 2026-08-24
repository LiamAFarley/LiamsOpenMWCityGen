# 2026-08-19 Falkreath profile-driven house pilot

`generate_fk_house.py --all-pilots --render` now exercises measured wall
profiles through `src/procgen/fk_house.py`. The CLI pilot path is intentionally
review-preview mode: it can read `needs_review` profiles and use nearest-edge
fallback for openings whose complete support envelope is not yet represented.

Generated stamps are tagged with `source.wall_profile_mode: review_preview`.
Production callers that request profile mode without explicit review-preview
permission still fail closed on `needs_review` profiles.
