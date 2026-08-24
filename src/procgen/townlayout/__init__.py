"""V2 townlayout package: contracts, geometry kernel, and generators.

Purpose
-------
Public entry for TownBrief / CityLayout / MacroLayoutCandidate validators
plus Phase 2 ``stage_rng`` and planar helpers.

Inputs / outputs
----------------
See ``validate``, ``candidate``, ``rng.stage_rng``, and ``geometry``.

Pipeline position
-----------------
V2 townlayout Phase 5 interface; generators live in submodules.
"""

from .candidate import require_macro_layout, validate_macro_layout
from .geometry import (
    VERTEX_EPS_GU,
    destructive_difference,
    normalize_ring,
    polygon_from_ring,
)
from .rng import stage_rng
from .validate import (
    TownLayoutError,
    validate_city_layout,
    validate_site_context,
    validate_town_brief,
)

__all__ = [
    "validate_city_layout",
    "validate_town_brief",
    "validate_site_context",
    "validate_macro_layout",
    "require_macro_layout",
    "TownLayoutError",
    "stage_rng",
    "VERTEX_EPS_GU",
    "normalize_ring",
    "polygon_from_ring",
    "destructive_difference",
]
