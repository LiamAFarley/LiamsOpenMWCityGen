"""Stage 04 contiguous ward blobs grown by deterministic round-robin BFS."""

from __future__ import annotations

import math
from typing import Any

from .geometry import polygon_from_ring
from .validate import TownLayoutError

WARD_DPLAN = {"market": "market", "craft": "craft", "residential": "residential",
              "outskirts": "outskirts", "keep": "keep"}
WARD_ORDER = ("craft", "residential", "outskirts")


def _quotas(mix: dict, n: int) -> dict[str, int]:
    """Largest-remainder quota allocation with stable kind tie-breaking."""
    raw = {k: max(0.0, float(mix.get(k, 0.0))) * n for k in WARD_ORDER}
    counts = {k: int(math.floor(raw[k])) for k in WARD_ORDER}
    for k in sorted(WARD_ORDER, key=lambda k: (-(raw[k] - counts[k]), k))[:n - sum(counts.values())]:
        counts[k] += 1
    return counts


def _patches(candidate: dict) -> dict[str, dict]:
    return {p["patch_id"]: p for p in candidate.get("patches", [])
            if p.get("inside_city")}


def _shared(a: dict, b: dict) -> float:
    return polygon_from_ring(a["polygon"]).boundary.intersection(
        polygon_from_ring(b["polygon"]).boundary).length


def _bfs_component(seed: str, members: set[str], by_id: dict[str, dict]) -> set[str]:
    seen = {seed}; todo = [seed]
    while todo:
        pid = todo.pop()
        for nid in by_id[pid].get("neighbour_patch_ids", []):
            if nid in members and nid not in seen:
                seen.add(nid); todo.append(nid)
    return seen


def assign_wards(candidate: dict, town_brief: dict, *, candidate_id: str = "c00") -> dict[str, Any]:
    by_id = _patches(candidate)
    if not by_id:
        raise TownLayoutError("disconnected_ward: no selected domain patches")
    anchors = candidate.get("anchors") or []
    market = next((a for a in anchors if a.get("kind") == "market"), None)
    if market is None or market["patch_id"] not in by_id:
        raise TownLayoutError("missing_anchor: market ward missing")
    assigned: dict[str, str] = {market["patch_id"]: "market"}
    for anchor in anchors:
        if anchor.get("kind") == "keep" and anchor["patch_id"] in by_id:
            assigned[anchor["patch_id"]] = "keep"

    remaining = sorted(set(by_id) - set(assigned))
    forced_out = {pid for pid in remaining if by_id[pid].get("morphology_region") == "outskirts"}
    quotas = _quotas(town_brief["ward_mix"], len(remaining))
    if not forced_out:
        # A quota cannot manufacture an outskirts patch.  Keep outskirts
        # fringe-only and transfer its integer slots to residential.
        quotas["residential"] += quotas["outskirts"]
        quotas["outskirts"] = 0
    else:
        quotas["outskirts"] = max(quotas["outskirts"], len(forced_out))
    # Quotas are counts for unreserved patches; force fringe patches into the
    # outskirts quota and let the remaining slots be filled by adjacency.
    if sum(quotas.values()) > len(remaining):
        quotas["residential"] = max(0, len(remaining) - quotas["craft"] - quotas["outskirts"])

    seeds: dict[str, str] = {}
    market_id = market["patch_id"]
    fabric = [pid for pid in remaining if pid not in forced_out]
    craft_candidates = [pid for pid in fabric if market_id in by_id[pid].get("neighbour_patch_ids", [])]
    if quotas["craft"]:
        if not craft_candidates:
            raise TownLayoutError("disconnected_ward: craft has no market-adjacent seed")
        seeds["craft"] = max(craft_candidates, key=lambda pid: (_shared(by_id[pid], by_id[market_id]),
                                                                  -pid.count("_"), pid))
    residential_candidates = [pid for pid in fabric if pid not in seeds and
                               (market_id in by_id[pid].get("neighbour_patch_ids", []) or
                                any(seeds.get("craft") == n for n in by_id[pid].get("neighbour_patch_ids", [])))]
    if quotas["residential"]:
        if not residential_candidates:
            raise TownLayoutError("disconnected_ward: residential has no adjacent seed")
        seeds["residential"] = sorted(residential_candidates)[0]
    if quotas["outskirts"]:
        forced_available = sorted(pid for pid in forced_out if pid not in seeds)
        if not forced_available:
            quotas["residential"] += quotas["outskirts"]
            quotas["outskirts"] = 0
        if not forced_out:
            raise TownLayoutError("disconnected_ward: outskirts is not fringe-forced")
        elif quotas["outskirts"]:
            seeds["outskirts"] = forced_available[0]

    # Remove seeds from the pool and grow each ward only from its own frontier.
    for kind, pid in seeds.items():
        assigned[pid] = kind
    members = {k: {p for p, w in assigned.items() if w == k} for k in WARD_ORDER}
    counts = {k: len(members[k]) for k in WARD_ORDER}
    quota_shortfall: dict[str, int] = {}
    active = True
    while active:
        active = False
        for kind in WARD_ORDER:
            target = quotas[kind]
            if counts[kind] >= target:
                continue
            active = True
            frontier = [pid for pid in remaining if pid not in assigned and any(
                n in members[kind] for n in by_id[pid].get("neighbour_patch_ids", []))]
            if kind == "outskirts":
                frontier = [pid for pid in frontier if by_id[pid].get("morphology_region") == "outskirts"
                            or not by_id[pid].get("inside_wall", False)]
            if not frontier:
                # A legacy small fixture can request more members than exist
                # in that ward's connected component.  Freeze its connected
                # blob and record the shortfall; never bridge through another
                # ward (the Stage 04 adversarial gate checks this invariant).
                quota_shortfall[kind] = target - counts[kind]
                quotas[kind] = counts[kind]
                continue
            seed_poly = by_id[seeds[kind]]
            def key(pid: str):
                same = sum(_shared(by_id[pid], by_id[n]) for n in by_id[pid].get("neighbour_patch_ids", []) if n in members[kind])
                c = polygon_from_ring(by_id[pid]["polygon"]).centroid
                sc = polygon_from_ring(seed_poly["polygon"]).centroid
                dist = math.hypot(c.x - sc.x, c.y - sc.y)
                suit = 1.0 - float(by_id[pid].get("terrain_summary", {}).get("mean_slope_deg", 0.0)) / 25.0
                return (-same, dist, -suit, pid)
            chosen = sorted(frontier, key=key)[0]
            assigned[chosen] = kind; members[kind].add(chosen); counts[kind] += 1

    # Deterministic articulation-safe boundary repair permitted by Stage 04.
    # It only moves a patch when the source ward remains one BFS component.
    for target, missing in sorted(quota_shortfall.items()):
        for _ in range(missing):
            options = []
            for source in WARD_ORDER:
                if source == target:
                    continue
                for pid in sorted(members[source]):
                    if not any(n in members[target] for n in by_id[pid].get("neighbour_patch_ids", [])):
                        continue
                    remaining_source = members[source] - {pid}
                    if remaining_source and len(remaining_source) != len(_bfs_component(min(remaining_source), remaining_source, by_id)):
                        continue
                    if source == "outskirts" and by_id[pid].get("morphology_region") == "outskirts":
                        continue
                    options.append((pid, source))
            if not options:
                break
            pid, source = options[0]
            members[source].remove(pid); members[target].add(pid); assigned[pid] = target
            quota_shortfall[target] -= 1

    # Any zero-quota category aside, all selected patches must be assigned.
    unassigned = set(by_id) - set(assigned)
    while unassigned:
      progressed = False
      for pid in sorted(unassigned):
        # A compact connected domain can leave a final boundary patch adjacent
        # only to the reserved market seed.  It remains a valid market fringe;
        # fail-closed here only when it has no selected ward frontier at all.
        ward_kinds = ("market", *WARD_ORDER)
        ward_members = {"market": {market_id}, **members}
        candidates = [(kind, -max(_shared(by_id[pid], by_id[mid])
                                  for mid in sorted(ward_members[kind])))
                      for kind in ward_kinds if ward_members[kind] and
                      any(n in ward_members[kind] for n in by_id[pid].get("neighbour_patch_ids", []))]
        if not candidates:
            continue
        kind = sorted(candidates, key=lambda x: (x[1], x[0]))[0][0]
        assigned[pid] = kind
        if kind != "market":
            members[kind].add(pid)
        unassigned.remove(pid)
        progressed = True
      if not progressed:
        raise TownLayoutError("disconnected_ward: selected patch has no ward frontier")

    wards = [{"ward_id": f"ward_{candidate_id}_{i:04d}", "ward_type": kind,
              "patch_ids": sorted(pids), "dplan_kind": WARD_DPLAN[kind],
              "score_evidence": {}}
             for i, kind in enumerate(("market", "keep", *WARD_ORDER))
             if (pids := {pid for pid, value in assigned.items() if value == kind})]
    components = {}
    for ward in wards:
        ids = set(ward["patch_ids"])
        components[ward["ward_type"]] = len(ids - _bfs_component(next(iter(ids)), ids, by_id)) + 1
        if components[ward["ward_type"]] != 1:
            raise TownLayoutError(f"disconnected_ward: {ward['ward_type']} BFS component count")
    reports = list(candidate.get("reports") or [])
    reports.append({"stage": "wards", "status": "ok", "message":
                    f"components={components} quota_shortfall={quota_shortfall} " +
                    " ".join(f"{w['ward_type']}={len(w['patch_ids'])}" for w in wards)})
    out = dict(candidate); out["wards"] = wards; out["reports"] = reports
    out["ward_metrics"] = {"component_counts": components, "quota_shortfall": quota_shortfall,
                            "ward_patch_counts": {w["ward_type"]: len(w["patch_ids"]) for w in wards}}
    return out
