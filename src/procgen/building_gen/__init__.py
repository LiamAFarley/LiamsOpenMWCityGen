"""Audited building-generation evidence and relation rebuilding primitives.

This package is intentionally host-side and deterministic.  It validates the
Phase 0 contracts and converts immutable xFa source-world transforms into
template-local and A-local relation products for later composer phases.
"""

from .contracts import (
    validate_access_bundle,
    validate_building_extension_request,
    validate_building_request,
    validate_connection_sample,
    validate_generated_building,
    validate_model_profile,
    validate_normalized_member,
    validate_palette,
    validate_source_member,
)
from .palette import load_policy_document, resolve_policy, resolve_selection, validate_resolved_palette

__all__ = [
    "validate_access_bundle",
    "validate_building_extension_request",
    "validate_building_request",
    "validate_connection_sample",
    "validate_generated_building",
    "validate_model_profile",
    "validate_normalized_member",
    "validate_palette",
    "validate_source_member",
    "load_policy_document",
    "resolve_policy",
    "resolve_selection",
    "validate_resolved_palette",
]
