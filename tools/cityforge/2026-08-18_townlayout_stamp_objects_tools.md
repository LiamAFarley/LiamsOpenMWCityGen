# 2026-08-18 townlayout stamp-object realization tools

`realize_townlayout_stamps.py` consumes the accepted R13
`city_layout.json` directly and expands each occupied placement through the
canonical D-STAMP v2 libraries. It writes a deterministic
`townlayout_stamp_objects_v1` JSON product with stable placement/member
identities, world transforms, and exterior-cell buckets.

Terrain seating, circulation surfaces, scatter clearing, and TES3/ESP writing
are later stages. The default anchor Z is `0.0`; callers may provide a terrain
stage's seated anchor Z with `--anchor-z-gu`.

`seat_townlayout_stamps.py` is the next stage. It uses the accepted survey
height field and each stamp's measured primary-door step height to seat every
expanded stamp while checking door clearance and bottom protrusion.
