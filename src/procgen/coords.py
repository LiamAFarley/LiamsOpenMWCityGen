"""Canonical Tamriel grid and heightmap coordinate conversions.

The composite heightmap uses a south-west origin: pixel ``(0, 0)`` is the
south-west point of cell ``(-251, -122)`` and y increases northward.  A cell
occupies 64 by 64 height samples.  Pixel coordinates are zero based and refer
to sample points, not cell centers.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence, TypeAlias

MIN_X = -251
MAX_X = 57
MIN_Y = -122
MAX_Y = 59
CELL_POINTS = 64

GRID_WIDTH = MAX_X - MIN_X + 1
GRID_HEIGHT = MAX_Y - MIN_Y + 1
PIXEL_WIDTH = GRID_WIDTH * CELL_POINTS
PIXEL_HEIGHT = GRID_HEIGHT * CELL_POINTS

Cell: TypeAlias = tuple[int, int]
Pixel: TypeAlias = tuple[int, int]


def _is_int(value: object) -> bool:
    """Return true for integer-like values, excluding booleans."""

    return isinstance(value, Integral) and not isinstance(value, bool)


def in_grid(cell: Sequence[object]) -> bool:
    """Return whether ``cell`` is an integer exterior cell in the game grid."""

    if len(cell) != 2 or not all(_is_int(value) for value in cell):
        return False
    x, y = int(cell[0]), int(cell[1])
    return MIN_X <= x <= MAX_X and MIN_Y <= y <= MAX_Y


def require_in_grid(cell: Sequence[object]) -> Cell:
    """Validate and normalize a placement cell.

    Out-of-grid planned coordinates are provenance only and must not reach a
    placement stage.  ``ValueError`` is deliberately raised here rather than
    silently clipping the coordinate.
    """

    if not in_grid(cell):
        raise ValueError(
            f"cell {tuple(cell)!r} is outside the Tamriel grid "
            f"x=[{MIN_X},{MAX_X}], y=[{MIN_Y},{MAX_Y}]"
        )
    return int(cell[0]), int(cell[1])


def cell_to_pixel(cell: Sequence[object], *, center: bool = False) -> tuple[float, float]:
    """Convert a cell to its south-west pixel sample, or its center.

    ``center=False`` returns integer-valued floats for the first sample in the
    64x64 block.  ``center=True`` returns the geometric center in pixel
    coordinates (``+31.5``), useful for map overlays.
    """

    x, y = require_in_grid(cell)
    offset = (CELL_POINTS - 1) / 2 if center else 0
    return (
        (x - MIN_X) * CELL_POINTS + offset,
        (y - MIN_Y) * CELL_POINTS + offset,
    )


def pixel_to_cell(pixel: Sequence[Real]) -> Cell:
    """Convert a heightmap sample coordinate to its containing cell.

    The last valid sample is ``(PIXEL_WIDTH-1, PIXEL_HEIGHT-1)``.  Fractional
    pixel coordinates are accepted and are floored like the raw block mapping.
    """

    if len(pixel) != 2:
        raise ValueError(f"pixel must contain two coordinates, got {pixel!r}")
    px, py = float(pixel[0]), float(pixel[1])
    if not (math.isfinite(px) and math.isfinite(py)):
        raise ValueError(f"pixel must be finite, got {pixel!r}")
    if px < 0 or py < 0 or px >= PIXEL_WIDTH or py >= PIXEL_HEIGHT:
        raise ValueError(
            f"pixel {pixel!r} is outside heightmap bounds "
            f"x=[0,{PIXEL_WIDTH - 1}], y=[0,{PIXEL_HEIGHT - 1}]"
        )
    return MIN_X + math.floor(px / CELL_POINTS), MIN_Y + math.floor(py / CELL_POINTS)
