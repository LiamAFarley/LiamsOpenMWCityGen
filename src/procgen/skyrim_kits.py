"""Pure helpers for the Sky_Main region-kit analysis.

The Skyrim investigation deliberately keeps its source-specific policy here,
separate from the TES3 binary reader.  :mod:`procgen.espscan` resolves the
records and references; this module decides which resolved references are
construction pieces and provides the deterministic mesh-vector distance used
by the output driver.

No NIF geometry is inspected.  The exclusion rules are conservative lexical
rules over the already classified MODL path and TES3 object id.  They are
intended to be reused by the later render pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

from .espscan import CellReference


SKYRIM_FILTER_VERSION = "1.1"
DEFAULT_PREVALENCE_FLOOR = 0.75
DEFAULT_REGION_DISTANCE_THRESHOLD = 0.62
DEFAULT_JACCARD_WEIGHT = 0.5

# Sky_Main contains one inherited Morrowind region label on a cell which is
# not part of the Skyrim built-world analysis.  Keep the list explicit so a
# new RGNN is not silently discarded merely because its name is unfamiliar.
VANILLA_MORROWIND_REGIONS = frozenset(
    {
        "ascadian isles",
        "ascadian isles region",
        "ashlands",
        "ashlands region",
        "azura's coast",
        "azura's coast region",
        "bitter coast",
        "bitter coast region",
        "grazelands",
        "grazelands region",
        "molag amur",
        "molag amur region",
        "red mountain",
        "red mountain region",
        "sheogorad",
        "sheogorad region",
        "west gash",
        "west gash region",
    }
)


@dataclass(frozen=True)
class FilterDecision:
    """Result of the reusable Skyrim building-piece filter."""

    included: bool
    reason: str


def _normalise_text(value: str | None) -> str:
    return " ".join((value or "").replace("/", "\\").casefold().split())


def _tokens(value: str | None) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", (value or "").casefold()) if token}


def is_skyrim_region(region: str | None) -> bool:
    """Return whether an RGNN label belongs to the Skyrim-only pass."""

    return bool(_normalise_text(region)) and _normalise_text(region) not in VANILLA_MORROWIND_REGIONS


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def _standalone_wall(model: str, object_id: str) -> bool:
    """Recognise wall barriers without rejecting house-wall construction.

    The object ids in Sky_Main distinguish ``HouseWall``/``HouseWallRuin``
    from ``StoneWall`` and ``FortWall``.  For legacy records with no useful id,
    a bare ``_WL_`` model token is treated as a standalone wall unless the
    model also carries a house/ruin marker.
    """

    combined = f"{model} {object_id}".casefold()
    if _contains_any(combined, ("stonewall", "fortwall", "wallsegment", "wall_section")):
        return True
    id_tokens = _tokens(object_id)
    if "wall" in id_tokens and not id_tokens.intersection(
        {"house", "housewall", "hut", "ruin", "ruinwall", "barrow"}
    ):
        return True
    model_tokens = _tokens(model)
    if "wl" in model_tokens and not _contains_any(combined, ("house", "hut", "ruin")):
        return True
    return False


def classify_profile_reference(ref: CellReference) -> FilterDecision:
    """Apply the profile-tier filter used by this task's JSON outputs.

    Profile vocabulary intentionally retains structural boundary pieces:
    fences, palisades, and standalone/non-building walls are strong tileset
    discriminators.  Within exterior construction candidates, this tier only
    removes flora, rocks, clutter/furniture, and non-profile categories.  The
    later render tier adds the house-sheet exclusions separately.
    """

    model = ref.model or ""
    object_id = ref.object_id or ""
    if not model:
        return FilterDecision(False, "unresolved: missing MODL/object definition")
    if ref.position is None:
        return FilterDecision(False, "missing_transform: kept mesh has no DATA position")

    category = (ref.category or "unknown").casefold()
    if category == "flora":
        return FilterDecision(False, "flora: classified flora")
    if category == "rocks":
        return FilterDecision(False, "rocks: classified rocks")
    if category == "clutter":
        return FilterDecision(False, "clutter/furniture: classified clutter or furniture")
    if category == "terrain":
        return FilterDecision(False, "terrain: shared terrain/entrance asset")
    if category == "interior":
        return FilterDecision(False, "interior: interior construction excluded from exterior kits")
    if category not in {"exterior", "door"}:
        return FilterDecision(False, "other: not an exterior construction or door category")

    combined = f"{model} {object_id}".casefold()
    # A small set of exterior-folder pieces are explicitly loose scenery or
    # debris rather than construction. They remain profile clutter even
    # though the inventory folder is ``x``.
    if _contains_any(combined, ("rubble", "scrapwood", "woodpile", "looseplank", "skeletoncattlebone")):
        return FilterDecision(False, "clutter: loose debris/scenery despite exterior folder")

    return FilterDecision(True, "included_profile: exterior construction, structural boundary, or door")


def classify_render_reference(ref: CellReference) -> FilterDecision:
    """Apply the later house-render filter on top of profile eligibility.

    This function is deliberately *not* used by ``tools/skyrim_region_kits``.
    It exists so the render pipeline can reuse the exact user-requested
    exclusions without removing structural vocabulary from profile outputs.
    """

    profile = classify_profile_reference(ref)
    if not profile.included:
        return profile
    model = ref.model or ""
    object_id = ref.object_id or ""
    combined = f"{model} {object_id}".casefold()

    # Render sheets omit boundaries; profile prevalence and ground rules do
    # not. Check object ids as well as paths because Sky records often use
    # abbreviated MODLs.
    if _contains_any(combined, ("palisade", "fence", "fencepost", "railing", "rail")):
        reason = "palisade: boundary/palisade element" if "palisade" in combined else "fence: boundary fence/railing element"
        return FilterDecision(False, reason)
    if _standalone_wall(model, object_id):
        return FilterDecision(False, "standalone_wall: non-building stone/fort wall element")

    return FilterDecision(True, "included_render: profile piece retained for house sheet")


def classify_building_reference(ref: CellReference) -> FilterDecision:
    """Backward-compatible alias for the profile-tier classifier."""

    return classify_profile_reference(ref)


def _presence_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _log_frequency_cosine(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    left_values = [math.log1p(max(0, int(left.get(key, 0)))) for key in keys]
    right_values = [math.log1p(max(0, int(right.get(key, 0)))) for key in keys]
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left_values, right_values)) / (left_norm * right_norm)


def region_mesh_distance(
    left: Mapping[str, int],
    right: Mapping[str, int],
    *,
    jaccard_weight: float = DEFAULT_JACCARD_WEIGHT,
) -> float:
    """Return the documented mixed presence/log-frequency distance."""

    if not 0.0 <= jaccard_weight <= 1.0:
        raise ValueError("jaccard_weight must be in [0, 1]")
    left_presence = {key for key, value in left.items() if int(value) > 0}
    right_presence = {key for key, value in right.items() if int(value) > 0}
    jaccard_distance = 1.0 - _presence_jaccard(left_presence, right_presence)
    cosine_distance = 1.0 - _log_frequency_cosine(left, right)
    return jaccard_weight * jaccard_distance + (1.0 - jaccard_weight) * cosine_distance


def cluster_region_meshes(
    region_mesh_counts: Mapping[str, Mapping[str, int]],
    *,
    distance_threshold: float = DEFAULT_REGION_DISTANCE_THRESHOLD,
    jaccard_weight: float = DEFAULT_JACCARD_WEIGHT,
) -> tuple[list[list[str]], dict[tuple[str, str], float]]:
    """Deterministically complete-link cluster region mesh vectors.

    Starting with one region per cluster, the closest pair is merged only if
    its complete-link distance is at most ``distance_threshold``.  Pair and
    member ordering are lexical, so no process hash order can change a group.
    The returned matrix contains the original region-to-region distances.
    """

    if distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative")
    regions = sorted(str(region) for region in region_mesh_counts)
    matrix: dict[tuple[str, str], float] = {}
    for index, left_name in enumerate(regions):
        for right_name in regions[index + 1 :]:
            matrix[(left_name, right_name)] = region_mesh_distance(
                region_mesh_counts[left_name],
                region_mesh_counts[right_name],
                jaccard_weight=jaccard_weight,
            )

    clusters: list[list[str]] = [[region] for region in regions]

    def pair_distance(left: Sequence[str], right: Sequence[str]) -> float:
        values = []
        for left_name in left:
            for right_name in right:
                key = tuple(sorted((left_name, right_name)))
                values.append(matrix.get(key, 0.0))
        return max(values, default=0.0)

    while len(clusters) > 1:
        candidates: list[tuple[float, tuple[str, ...], tuple[str, ...], int, int]] = []
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                candidates.append(
                    (
                        pair_distance(left, right),
                        tuple(left),
                        tuple(right),
                        left_index,
                        right_index,
                    )
                )
        distance, _left_key, _right_key, left_index, right_index = min(candidates)
        if distance > distance_threshold:
            break
        merged = sorted(clusters[left_index] + clusters[right_index])
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: tuple(cluster))

    clusters.sort(key=lambda cluster: tuple(cluster))
    return clusters, matrix


def conserved_mesh_vocabulary(
    members: Sequence[str],
    region_mesh_counts: Mapping[str, Mapping[str, int]],
    *,
    prevalence_floor: float = DEFAULT_PREVALENCE_FLOOR,
) -> list[dict[str, object]]:
    """List meshes present in at least the requested fraction of members."""

    if not members:
        return []
    if not 0.0 < prevalence_floor <= 1.0:
        raise ValueError("prevalence_floor must be in (0, 1]")
    vocabulary = sorted(
        {
            mesh
            for region in members
            for mesh, count in region_mesh_counts[region].items()
            if int(count) > 0
        }
    )
    rows = []
    for mesh in vocabulary:
        present = sum(1 for region in members if int(region_mesh_counts[region].get(mesh, 0)) > 0)
        prevalence = present / len(members)
        if prevalence + 1e-12 < prevalence_floor:
            continue
        rows.append(
            {
                "mesh": mesh,
                "member_presence_count": present,
                "prevalence": round(prevalence, 6),
                "counts_by_region": {
                    region: int(region_mesh_counts[region].get(mesh, 0))
                    for region in members
                },
            }
        )
    rows.sort(key=lambda row: (-float(row["prevalence"]), str(row["mesh"])))
    return rows


__all__ = [
    "DEFAULT_JACCARD_WEIGHT",
    "DEFAULT_PREVALENCE_FLOOR",
    "DEFAULT_REGION_DISTANCE_THRESHOLD",
    "FilterDecision",
    "SKYRIM_FILTER_VERSION",
    "VANILLA_MORROWIND_REGIONS",
    "classify_building_reference",
    "classify_profile_reference",
    "classify_render_reference",
    "cluster_region_meshes",
    "conserved_mesh_vocabulary",
    "is_skyrim_region",
    "region_mesh_distance",
]
