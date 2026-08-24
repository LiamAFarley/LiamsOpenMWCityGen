"""Phase 2 member/role inventory builder for the building generation rule kit.

Pipeline position: Phase 2 of the xFa building rule kit (spec:
``.opencode/runs/2026-08-21-building-generation-rule-kit/2026-08-22_phase2_implementation_spec.md``,
plan: ``2026-08-21_building_generation_rule_kit_plan_v2.md`` section 7.1).

Builds one inventory row per distinct model from actual stamp members and
their source ``structural_role`` values. No filename-substring classification
anywhere: observed roles, sites, scales, contacts, and relation references come
only from the site stamp libraries and the Phase 1 derived connections.

The driver (``tools/cityforge/build_model_profiles.py``) merges Blender
profile rows into this inventory and decides eligibility.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from procgen.building_gen.normalize import canonicalize

INVENTORY_SCHEMA_VERSION = 1

# Trailing single lowercase letter after a digit, covering both "house_01_a"
# and "window_05a" / "dormer_01b" naming forms.
_ALIAS_SUFFIX_RE = re.compile(r"^(.*\d)_?[a-z]$")


def canonical_model_key(model_key: str) -> str:
    return str(model_key).replace("/", "\\").casefold()


def alias_family_stem(model_key: str) -> str | None:
    """Return the candidate alias family stem, or None when no suffix form."""
    stem = str(model_key).replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".nif").casefold()
    match = _ALIAS_SUFFIX_RE.match(stem)
    return match.group(1) if match else None


def _member_rows(stamp: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    members = stamp.get("members", [])
    return [m for m in members if isinstance(m, Mapping)]


def build_inventory(
    site_stamp_libraries: Mapping[str, Mapping[str, Any]],
    site_connections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-model evidence rows across every configured site."""
    rows: dict[str, dict[str, Any]] = {}

    def row_for(member: Mapping[str, Any]) -> dict[str, Any]:
        key = canonical_model_key(str(member["model_key"]))
        row = rows.get(key)
        if row is None:
            row = {
                "model_key": key,
                "display_key": str(member["model_key"]),
                "sites": set(),
                "stamp_ids": set(),
                "member_ref_count": 0,
                "observed_roles": set(),
                "record_types": set(),
                "scales_observed": set(),
                "direct_contact_neighbors": set(),
                "relation_rule_ids": set(),
            }
            rows[key] = row
        return row

    for site_id in sorted(site_stamp_libraries):
        library = site_stamp_libraries[site_id]
        for stamp in library.get("stamps", []):
            members = _member_rows(stamp)
            ref_to_key = {str(m["source_id"]): canonical_model_key(str(m["model_key"])) for m in members}
            for member in members:
                row = row_for(member)
                row["sites"].add(site_id)
                row["stamp_ids"].add(str(stamp["stamp_id"]))
                row["member_ref_count"] += 1
                role = member.get("structural_role")
                if role is not None:
                    row["observed_roles"].add(str(role))
                row["record_types"].add(str(member["record_type"]))
                row["scales_observed"].add(round(float(member["scale"]), 3))
            for edge_key in ("shell_attachment_edges", "internal_edges", "touching_pairs"):
                for edge in stamp.get(edge_key, []) or []:
                    if isinstance(edge, Mapping):
                        a, b = edge.get("ref_a"), edge.get("ref_b")
                    else:
                        a, b = (list(edge) + [None, None])[:2]
                    ka, kb = ref_to_key.get(str(a)), ref_to_key.get(str(b))
                    if ka and kb and ka != kb:
                        rows[ka]["direct_contact_neighbors"].add(kb)
                        rows[kb]["direct_contact_neighbors"].add(ka)

    for site_id in sorted(site_connections):
        document = site_connections[site_id]
        for rule in document.get("rules", []):
            rule_id = f"{rule['model_a']}|{rule['model_b']}"
            for key in (canonical_model_key(str(rule["model_a"])), canonical_model_key(str(rule["model_b"]))):
                row = rows.get(key)
                if row is not None:
                    row["relation_rule_ids"].add(rule_id)

    out_rows = []
    for key in sorted(rows):
        row = rows[key]
        roles = sorted(row["observed_roles"])
        if not roles:
            authority = "unresolved"
        elif len(roles) == 1:
            authority = "source_role"
        else:
            authority = "source_role_mixed"
        out_rows.append({
            "model_key": row["model_key"],
            "display_key": row["display_key"],
            "sites": sorted(row["sites"]),
            "stamp_ids": sorted(row["stamp_ids"]),
            "member_ref_count": row["member_ref_count"],
            "observed_roles": roles,
            "record_types": sorted(row["record_types"]),
            "scales_observed": sorted(row["scales_observed"]),
            "direct_contact_neighbors": sorted(row["direct_contact_neighbors"]),
            "relation_rule_ids": sorted(row["relation_rule_ids"]),
            "classification_authority": authority,
            "profile_eligible": None,
            "rejection_reason": None,
            "resolved_path": None,
        })
    return canonicalize({
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "origin": "stamp_members_and_source_roles",
        "model_count": len(out_rows),
        "models": out_rows,
    })


def alias_families(inventory: Mapping[str, Any]) -> dict[str, list[str]]:
    """Group observed models into candidate suffix families for digest evidence."""
    families: dict[str, list[str]] = {}
    for row in inventory.get("models", []):
        stem = alias_family_stem(str(row["model_key"]))
        if stem is None:
            continue
        families.setdefault(stem, []).append(str(row["model_key"]))
    return {stem: sorted(keys) for stem, keys in sorted(families.items()) if len(keys) > 1}
