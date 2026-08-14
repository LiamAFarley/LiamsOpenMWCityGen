"""Small, deterministic foundations for the Procedural Tamriel pipeline.

The package is intentionally kept independent of the TES3 exporter.  From a
workspace checkout, either install the package later or put ``src`` on
``PYTHONPATH`` (for example, ``$env:PYTHONPATH='src'; python -c "import
procgen"``).
"""

__version__ = "0.1.0"

from .coords import (
    CELL_POINTS,
    GRID_HEIGHT,
    GRID_WIDTH,
    MAX_X,
    MAX_Y,
    MIN_X,
    MIN_Y,
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    Cell,
    cell_to_pixel,
    in_grid,
    pixel_to_cell,
    require_in_grid,
)
from .ledger import ProjectLedger, record, tail
from .provenance import sha256_file, stamp
from .seeds import derive_seed
from .worldcontext import WorldContext

__all__ = [
    "__version__",
    "CELL_POINTS",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "MAX_X",
    "MAX_Y",
    "MIN_X",
    "MIN_Y",
    "PIXEL_HEIGHT",
    "PIXEL_WIDTH",
    "Cell",
    "ProjectLedger",
    "WorldContext",
    "cell_to_pixel",
    "derive_seed",
    "in_grid",
    "pixel_to_cell",
    "record",
    "require_in_grid",
    "sha256_file",
    "stamp",
    "tail",
]
