"""Strict district policy resolution and deterministic weighted choices.

Pipeline position: Phase 5 policy loading, after measured rule-kit evidence
and before the Phase 6 composer.  This module never discovers profiles or
selects an ineligible ID; the compiler supplies the selectable domains.
Settlement defaults are complete palettes. District and parcel documents are
explicit recursive overlays, so an omitted field can never acquire a hidden
Python default.
"""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Mapping
from typing import Any

from .contracts import validate_palette
from .normalize import canonicalize


def _fail(code: str, detail: str) -> None:
    raise ValueError(f"{code}: {detail}")


def _merge(base: Mapping[str, Any], override: Mapping[str, Any], path: str = "palette") -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key not in result:
            _fail("palette_override_unknown_key", f"{path}.{key} is not present in the settlement default")
        if isinstance(value, Mapping):
            if not isinstance(result[key], Mapping):
                _fail("palette_override_shape", f"{path}.{key} cannot replace a scalar with an object")
            result[key] = _merge(result[key], value, f"{path}.{key}")
        else:
            result[key] = copy.deepcopy(value)
    return result


def _nonzero_domain(values: Mapping[str, Any], field: str) -> None:
    if not values or not any(float(value) > 0.0 for value in values.values()):
        _fail("palette_empty_choice_domain", f"{field} has no positive weight")


def validate_resolved_palette(palette: Mapping[str, Any]) -> None:
    """Run the strict contract plus domain checks used by the compiler."""

    validate_palette(palette)
    shells = palette["shells"]
    attachments = palette["attachments"]
    _nonzero_domain(shells["weights"], "shells.weights")
    _nonzero_domain(shells["size_weights"], "shells.size_weights")
    if attachments["window_mode"] != "none":
        if attachments["window_rates"]:
            _nonzero_domain(attachments["window_rates"], "attachments.window_rates")
        elif attachments["window_mode"] == "observed_slots":
            _fail("palette_empty_choice_domain", "attachments.window_rates is required for observed_slots")


def load_policy_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the inheritance carrier without resolving a request."""

    if document.get("schema_version") != 1:
        _fail("palette_policy_invalid", "schema_version must be 1")
    allowed = {"schema_version", "settlement_defaults", "district_overrides", "parcel_overrides"}
    unknown = sorted(set(document) - allowed)
    if unknown:
        _fail("palette_policy_unknown_key", "unknown key(s): " + ", ".join(unknown))
    default = document.get("settlement_defaults")
    if not isinstance(default, Mapping):
        _fail("palette_policy_invalid", "settlement_defaults must be an object")
    validate_resolved_palette(default)
    for field in ("district_overrides", "parcel_overrides"):
        value = document.get(field, {})
        if not isinstance(value, Mapping):
            _fail("palette_policy_invalid", f"{field} must be an object")
        for key, override in value.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(override, Mapping):
                _fail("palette_policy_invalid", f"{field} entries must map non-empty IDs to objects")
            # Validate shape and references even for an override that is not
            # used by the current request set.
            validate_resolved_palette(_merge(default, override))
    return canonicalize(dict(document))


def resolve_policy(
    document: Mapping[str, Any],
    district_id: str,
    parcel_id: str | None = None,
) -> dict[str, Any]:
    """Resolve settlement -> district -> parcel policy for one request."""

    policy = copy.deepcopy(dict(document["settlement_defaults"]))
    district = document.get("district_overrides", {}).get(district_id)
    if district is not None:
        policy = _merge(policy, district, "district")
    if parcel_id is not None:
        parcel = document.get("parcel_overrides", {}).get(f"{district_id}:{parcel_id}")
        if parcel is not None:
            policy = _merge(policy, parcel, "parcel")
    validate_resolved_palette(policy)
    return canonicalize(policy)


def _seed(master_seed: int | float, request_id: str, palette_id: str, domain: str) -> tuple[str, int]:
    material = f"{int(master_seed)}|{request_id}|{palette_id}|{domain}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return digest, int(digest[:16], 16)


def weighted_choice(
    weights: Mapping[str, float],
    *,
    master_seed: int | float,
    request_id: str,
    palette_id: str,
    domain: str,
) -> tuple[str, str]:
    """Return one sorted weighted key and its deterministic subseed."""

    positive = [(str(key), float(value)) for key, value in sorted(weights.items()) if float(value) > 0.0]
    if not positive:
        _fail("palette_empty_choice_domain", f"{domain} has no positive weight")
    digest, seed = _seed(master_seed, request_id, palette_id, domain)
    picker = random.Random(seed)
    target = picker.random() * sum(weight for _key, weight in positive)
    cumulative = 0.0
    for key, weight in positive:
        cumulative += weight
        if target < cumulative:
            return key, digest
    return positive[-1][0], digest


def bernoulli(
    rate: float,
    *,
    master_seed: int | float,
    request_id: str,
    palette_id: str,
    domain: str,
) -> tuple[bool, str]:
    digest, seed = _seed(master_seed, request_id, palette_id, domain)
    return random.Random(seed).random() < float(rate), digest


def resolve_selection(
    policy: Mapping[str, Any],
    *,
    request_id: str,
    master_seed: int | float,
    requested_size: str,
) -> dict[str, Any]:
    """Resolve policy choices without composing or mutating a building."""

    validate_resolved_palette(policy)
    subseeds: dict[str, str] = {}
    shell_id, subseeds["shell"] = weighted_choice(
        policy["shells"]["weights"], master_seed=master_seed, request_id=request_id,
        palette_id=policy["palette_id"], domain="shell",
    )
    size_weights = policy["shells"]["size_weights"]
    if requested_size in size_weights and float(size_weights[requested_size]) > 0.0:
        size_id = requested_size
        subseeds["size"] = _seed(master_seed, request_id, policy["palette_id"], "size")[0]
    else:
        size_id, subseeds["size"] = weighted_choice(
            size_weights, master_seed=master_seed, request_id=request_id,
            palette_id=policy["palette_id"], domain="size",
        )
    access_id, subseeds["primary_access"] = weighted_choice(
        policy["access"]["primary_bundle_weights"], master_seed=master_seed,
        request_id=request_id, palette_id=policy["palette_id"], domain="primary_access",
    )
    attachments = policy["attachments"]
    window_id = None
    if attachments["window_mode"] != "none" and attachments["window_rates"]:
        window_id, subseeds["window"] = weighted_choice(
            attachments["window_rates"], master_seed=master_seed, request_id=request_id,
            palette_id=policy["palette_id"], domain="window",
        )
    observed, subseeds["observed_template"] = bernoulli(
        policy["shells"]["observed_template_rate"], master_seed=master_seed,
        request_id=request_id, palette_id=policy["palette_id"], domain="observed_template",
    )
    multi_shell, subseeds["multi_shell"] = bernoulli(
        policy["shells"]["multi_shell_rate"], master_seed=master_seed,
        request_id=request_id, palette_id=policy["palette_id"], domain="multi_shell",
    )
    secondary, subseeds["secondary_door"] = bernoulli(
        policy["access"]["secondary_door_rate"], master_seed=master_seed,
        request_id=request_id, palette_id=policy["palette_id"], domain="secondary_door",
    )
    return canonicalize({
        "palette_id": policy["palette_id"],
        "request_id": request_id,
        "master_seed": int(master_seed),
        "selected_shell_profile_id": shell_id,
        "selected_size_class": size_id,
        "selected_primary_access_id": access_id,
        "selected_window_profile_id": window_id,
        "observed_template": observed,
        "multi_shell": multi_shell,
        "secondary_door": secondary,
        "selection_subseeds": subseeds,
    })


__all__ = [
    "bernoulli",
    "load_policy_document",
    "resolve_policy",
    "resolve_selection",
    "validate_resolved_palette",
    "weighted_choice",
]
