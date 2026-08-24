"""Stage-isolated RNG for V2 townlayout.

Purpose
-------
Return a stdlib ``random.Random`` seeded by ``procgen.seeds.derive_seed``
so each (master_seed, candidate, stage, object) stream is independent.

Inputs
------
``master_seed`` int, ``candidate_id`` str, ``stage_name`` str,
``object_id`` str (default empty).

Outputs
-------
A new ``random.Random`` instance.  Never uses ``numpy.random``.

Pipeline position
-----------------
V2 townlayout Phase 2 geometry/RNG; no generation.
"""

from __future__ import annotations

import random

from procgen.seeds import derive_seed


def stage_rng(master_seed: int, candidate_id: str, stage_name: str,
              object_id: str = "") -> random.Random:
    seed = derive_seed(int(master_seed), "townlayout", str(candidate_id),
                       str(stage_name), str(object_id))
    return random.Random(seed)
