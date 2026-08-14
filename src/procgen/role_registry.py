#!/usr/bin/env python3
"""Strict, versioned, deterministic kit/role registry (schema v1, Phase 0).

Purpose
-------
Replace scattered mesh-name token constants (e.g.
``tools/karthgad_connectivity/component_manifests.py::SHELL_MODEL_TOKENS``)
with one validated, versioned JSON document and a deterministic matcher.
Phase 0 scope (accepted plan ``.opencode/runs/kit-role-registry/``): the
registry exists with built-in rules equivalent to the current component
shell tokens only; NO consumer imports it yet and no behavior changes.

Inputs
------
- A registry document (``configs/kit_role_registry_v1.json``,
  ``schema_version`` 1): rules, enums, explicit defaults, provenance, and a
  self-describing ``registry_sha256`` (canonical SHA-256 of the document
  with the ``registry_sha256`` field removed; the loader verifies it).

Outputs
-------
- ``Registry`` -- a loaded, fully validated document. ``Registry.verdict()``
  returns a ``Verdict``: merged exclusive effect fields, additive tags,
  matched rule IDs, per-field source rule IDs, and per-field losers.
- ``validate_document(doc)`` -- the complete list of validation errors
  (used by the CLI ``tools/kit_roles/validate_registry.py`` and the tests).

Invariants (fail-closed, deterministic)
---------------------------------------
- Rejected at load, before any matching: unknown top-level/rule/match/
  effect keys, unknown enum values, duplicate ``rule_id``, malformed
  patterns, patterns that normalize to an empty string, empty
  ``match.any``, an empty ``rules`` list, negative/missing priority,
  ``schema_version`` that is not exactly the integer 1 (bools/floats
  rejected), missing/malformed/mismatched ``registry_sha256``,
  ``structural_role: null`` inside a rule effect, and same-priority
  identical-predicate conflicts on exclusive fields.
- Verdicts depend only on the normalized query and the document; rule
  order never decides an exclusive field. Highest ``priority`` wins;
  equal-highest-priority disagreement on an exclusive field is a FATAL
  runtime conflict (``RegistryConflictError``). ``(priority, rule_id)``
  ordering exists only for deterministic iteration and reporting.
- Matching uses Python ``fnmatch.fnmatchcase`` exactly: ``*`` matches any
  characters INCLUDING ``/`` (plain fnmatch semantics; ``*`` is documented
  to cross path separators), ``?`` matches one character, ``[...]`` a
  character class. No regex, no invented slash-aware behavior.

Normalization (accepted plan item 2)
------------------------------------
- Model keys: Unicode casefold, ``\\`` -> ``/``, repeated slashes collapsed,
  a single leading ``meshes/`` stripped; separators are retained.
- Object IDs: Unicode casefold only.
- Patterns are normalized identically to the keys of their channel at
  load time, so equality of a normalized key and normalized pattern is the
  matching test.

Schema (v1) highlights
----------------------
- ``match.any``: non-empty list of predicates; each predicate selects
  exactly ONE channel (``model_key`` or ``object_id``) with one or more
  glob patterns; ANY predicate may match (OR across predicates, OR across
  patterns within a predicate).
- ``match.when`` (optional): ``kit`` / ``record_type`` / ``category``
  conditions, ANDed across keys, OR within each list. A condition requires
  the query to supply a non-None value present in the list (exact
  equality; these vocabularies are already canonical).
- ``effect``: exclusive fields ``structural_role`` (enum, never null),
  ``shell_eligible``, ``contact_exclude``, ``render_exclude``,
  ``profile_exclude`` (bools) plus additive ``tags`` (string union).
  ``defaults.unmatched_verdict`` supplies explicit values when no rule
  sets a field (``structural_role: null`` is legal there and only there).
- ``enabled: false`` rules are schema-validated but inert (never match,
  never conflict; retained for audit).

Pipeline position
-----------------
Standalone library. Phase 1 migrates ``component_manifests._shell_id`` to
``Registry.verdict`` (strict mode). Equivalence test
``tests/test_role_registry_equivalence.py`` proves the current verdicts are
reproduced for every unique model key in current settlement outputs.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

# The three enums the document must declare (exact match required).
STRUCTURAL_ROLES = ("shell", "connector", "detail", "access", "boundary")
CATEGORIES = ("exterior", "interior", "door", "terrain", "flora", "rocks", "clutter", "other")
RECORD_TYPES = ("STAT", "DOOR", "CONT", "LIGH", "ACTI", "FURN", "MISC")

# Exclusive per-consumer effect fields (accepted plan item 5): highest
# priority wins; tags are the only additive field.
EXCLUSIVE_EFFECT_FIELDS = (
    "structural_role",
    "shell_eligible",
    "contact_exclude",
    "render_exclude",
    "profile_exclude",
)
TAGS_FIELD = "tags"
EFFECT_FIELDS = EXCLUSIVE_EFFECT_FIELDS + (TAGS_FIELD,)

CHANNELS = ("model_key", "object_id")
WHEN_CONDITIONS = ("kit", "record_type", "category")

_RULE_ID_RE = re.compile(r"^[a-z0-9._-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLASH_RUN_RE = re.compile(r"/{2,}")
_MESHES_PREFIX = "meshes/"

# ---------------------------------------------------------------------------
# Normalization (accepted plan item 2)
# ---------------------------------------------------------------------------


def normalize_model_key(key: object) -> str:
    """Normalize a model key for identity tests.

    casefold -> ``\\`` to ``/`` -> collapse repeated slashes -> strip one
    leading ``meshes/``.  Path separators are retained (matching the
    consumer ``component_manifests._normalize_model_key`` for real keys,
    plus the plan's collapse/strip rules).  ``None``/non-string inputs are
    stringified via ``str()`` so callers can pass raw JSON values.
    """
    text = str(key).casefold()
    text = text.replace("\\", "/")
    text = _SLASH_RUN_RE.sub("/", text)
    if text.startswith(_MESHES_PREFIX):
        text = text[len(_MESHES_PREFIX):]
    return text


def normalize_object_id(key: object) -> str:
    """Normalize an object ID: Unicode casefold only (accepted plan item 2)."""
    return str(key).casefold()


def normalize_pattern(channel: str, pattern: object) -> str:
    """Normalize a glob pattern for ``channel`` exactly like its keys."""
    if channel == "model_key":
        return normalize_model_key(pattern)
    return normalize_object_id(pattern)


# ---------------------------------------------------------------------------
# Canonical serialization / hash
# ---------------------------------------------------------------------------


def canonical_json(doc: Mapping[str, Any]) -> str:
    """Deterministic JSON text: sorted keys, compact separators, ascii."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(doc: Mapping[str, Any]) -> str:
    """SHA-256 (hex) of the canonical JSON of ``doc`` minus ``registry_sha256``.

    The hash field describes the rest of the document, so it is excluded
    from the hashed bytes (otherwise the hash could never be self-consistent).
    """
    body = {key: value for key, value in doc.items() if key != "registry_sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RegistryError(RuntimeError):
    """Base class for registry failures (validation and runtime conflicts)."""


class RegistryValidationError(RegistryError):
    """Document failed validation; ``errors`` holds every detected problem."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        joined = "; ".join(self.errors)
        super().__init__(f"registry validation failed ({len(self.errors)} error(s)): {joined}")


class RegistryConflictError(RegistryError):
    """Runtime exclusive-field conflict at the winning priority.

    Raised only when two or more matching rules at the SAME highest
    priority specify different values for one exclusive effect field
    (accepted plan item 6: this is a fatal runtime conflict, never an
    accidental ``rule_id`` tie-break).
    """

    def __init__(self, model_key: Optional[str], object_id: Optional[str], field: str, rule_ids: Sequence[str]) -> None:
        self.model_key = model_key
        self.object_id = object_id
        self.field = field
        self.rule_ids = tuple(sorted(rule_ids))
        query = f"model_key={model_key!r}" if model_key is not None else f"object_id={object_id!r}"
        super().__init__(
            f"registry runtime conflict: {query} exclusive field {field!r} is set to "
            f"different values by equal-priority rules {self.rule_ids!r}"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _err(errors: list[str], where: str, message: str) -> None:
    errors.append(f"{where}: {message}")


def _require_str_list(value: Any, where: str, errors: list[str], *, nonempty: bool) -> None:
    if not isinstance(value, list):
        _err(errors, where, "expected a list of strings")
        return
    if nonempty and not value:
        _err(errors, where, "list must be non-empty")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            _err(errors, where, f"entry {index} must be a non-empty string")


def _pattern_is_malformed(pattern: str) -> bool:
    """True for a glob that fnmatch semantics cannot express cleanly.

    ``*``/``?`` are always legal; a ``[`` that never closes its character
    class is malformed (load-time failure per the accepted plan item 8).
    A ``]`` outside a class is a literal in fnmatch and stays legal.
    """
    depth = 0
    for char in pattern:
        if char == "[":
            depth += 1
        elif char == "]":
            if depth > 0:
                depth -= 1
    return depth > 0


def _check_enum(value: Any, where: str, allowed: Sequence[str], errors: list[str]) -> None:
    if value not in allowed:
        _err(errors, where, f"unknown enum value {value!r} (expected one of {list(allowed)})")


def _normalized_pattern_tuple(channel: str, patterns: Sequence[str]) -> tuple[str, ...]:
    """Sorted, deduplicated, normalized patterns for one channel.

    Sorting makes predicate identity (static conflict check) and verdict
    iteration deterministic; pattern order is semantically irrelevant (OR).
    """
    return tuple(sorted({normalize_pattern(channel, p) for p in patterns}))


def _validate_any_predicates(match_any: Any, rule_id: str, errors: list[str]) -> None:
    where = f"rule {rule_id!r}: match.any"
    if not isinstance(match_any, list):
        _err(errors, where, "must be a non-empty list of predicates")
        return
    if not match_any:
        _err(errors, where, "must be a non-empty list of predicates (empty match)")
        return
    for index, predicate in enumerate(match_any):
        pwhere = f"{where}[{index}]"
        if not isinstance(predicate, dict):
            _err(errors, pwhere, "each predicate must be an object")
            continue
        unknown = set(predicate) - set(CHANNELS)
        for key in sorted(unknown):
            _err(errors, pwhere, f"unknown channel {key!r} (predicate selects exactly one channel)")
        channels = [key for key in CHANNELS if key in predicate]
        if len(channels) != 1:
            _err(errors, pwhere, "predicate must select exactly ONE channel (model_key or object_id)")
            continue
        patterns = predicate[channels[0]]
        _require_str_list(patterns, f"{pwhere}.{channels[0]}", errors, nonempty=True)
        if isinstance(patterns, list):
            for pindex, pattern in enumerate(patterns):
                if isinstance(pattern, str) and pattern:
                    normalized = normalize_pattern(channels[0], pattern)
                    if not normalized:
                        _err(errors, f"{pwhere}.{channels[0]}[{pindex}]",
                             f"pattern {pattern!r} normalizes to an empty string")
                    elif _pattern_is_malformed(normalized):
                        _err(errors, f"{pwhere}.{channels[0]}[{pindex}]",
                             f"malformed glob pattern {pattern!r} (unbalanced '[')")


def _validate_when(when: Any, rule_id: str, errors: list[str]) -> None:
    where = f"rule {rule_id!r}: match.when"
    if when is None:
        return
    if not isinstance(when, dict):
        _err(errors, where, "must be an object")
        return
    unknown = set(when) - set(WHEN_CONDITIONS)
    for key in sorted(unknown):
        _err(errors, where, f"unknown condition {key!r} (expected kit/record_type/category)")
    for key in WHEN_CONDITIONS:
        if key not in when:
            continue
        _require_str_list(when[key], f"{where}.{key}", errors, nonempty=True)
        if key == "record_type":
            for index, value in enumerate(when[key]):
                if isinstance(value, str):
                    _check_enum(value, f"{where}.{key}[{index}]", RECORD_TYPES, errors)
        elif key == "category":
            for index, value in enumerate(when[key]):
                if isinstance(value, str):
                    _check_enum(value, f"{where}.{key}[{index}]", CATEGORIES, errors)


def _validate_effect(effect: Any, rule_id: str, errors: list[str]) -> None:
    where = f"rule {rule_id!r}: effect"
    if not isinstance(effect, dict):
        _err(errors, where, "must be an object")
        return
    unknown = set(effect) - set(EFFECT_FIELDS)
    for key in sorted(unknown):
        _err(errors, where, f"unknown effect field {key!r}")
    if not any(key in effect for key in EFFECT_FIELDS):
        _err(errors, where, "effect must declare at least one field")
    if "structural_role" in effect:
        # A rule effect declares a ROLE; null is only legal as the explicit
        # unmatched default (defaults.unmatched_verdict), never in a rule.
        _check_enum(effect["structural_role"], f"{where}.structural_role", STRUCTURAL_ROLES, errors)
    for key in ("shell_eligible", "contact_exclude", "render_exclude", "profile_exclude"):
        if key in effect and not isinstance(effect[key], bool):
            _err(errors, f"{where}.{key}", f"must be a boolean, got {type(effect[key]).__name__}")
    if TAGS_FIELD in effect:
        _require_str_list(effect[TAGS_FIELD], f"{where}.{TAGS_FIELD}", errors, nonempty=False)


def _validate_defaults(defaults: Any, errors: list[str]) -> None:
    where = "defaults"
    if not isinstance(defaults, dict):
        _err(errors, where, "must be an object")
        return
    unknown = set(defaults) - {"unmatched_verdict"}
    for key in sorted(unknown):
        _err(errors, where, f"unknown defaults key {key!r}")
    if "unmatched_verdict" not in defaults:
        _err(errors, where, "missing required 'unmatched_verdict'")
        return
    verdict = defaults["unmatched_verdict"]
    vwhere = f"{where}.unmatched_verdict"
    if not isinstance(verdict, dict):
        _err(errors, vwhere, "must be an object")
        return
    unknown = set(verdict) - set(EFFECT_FIELDS)
    for key in sorted(unknown):
        _err(errors, vwhere, f"unknown default field {key!r}")
    for key in EXCLUSIVE_EFFECT_FIELDS:
        if key not in verdict:
            _err(errors, vwhere, f"missing explicit default for {key!r} (defaults are explicit)")
    if "structural_role" in verdict:
        value = verdict["structural_role"]
        if value is not None:
            _check_enum(value, f"{vwhere}.structural_role", STRUCTURAL_ROLES, errors)
    for key in ("shell_eligible", "contact_exclude", "render_exclude", "profile_exclude"):
        if key in verdict and not isinstance(verdict[key], bool):
            _err(errors, f"{vwhere}.{key}", f"must be a boolean, got {type(verdict[key]).__name__}")
    if TAGS_FIELD in verdict:
        _require_str_list(verdict[TAGS_FIELD], f"{vwhere}.{TAGS_FIELD}", errors, nonempty=False)


def _validate_enums(enums: Any, errors: list[str]) -> None:
    where = "enums"
    if not isinstance(enums, dict):
        _err(errors, where, "must be an object")
        return
    expected = {
        "structural_roles": STRUCTURAL_ROLES,
        "categories": CATEGORIES,
        "record_types": RECORD_TYPES,
    }
    unknown = set(enums) - set(expected)
    for key in sorted(unknown):
        _err(errors, where, f"unknown enum key {key!r}")
    for key, builtin in expected.items():
        if key not in enums:
            _err(errors, where, f"missing enum {key!r}")
            continue
        value = enums[key]
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            _err(errors, f"{where}.{key}", "must be a list of strings")
            continue
        if sorted(value) != sorted(builtin):
            _err(errors, f"{where}.{key}",
                 f"must equal the loader enum {sorted(builtin)!r}, got {sorted(value)!r}")


def _validate_provenance(provenance: Any, errors: list[str]) -> None:
    where = "provenance"
    if provenance is None:
        return
    if not isinstance(provenance, dict):
        _err(errors, where, "must be an object")
        return
    allowed = {"source_files", "constants", "report_ref"}
    unknown = set(provenance) - allowed
    for key in sorted(unknown):
        _err(errors, where, f"unknown provenance key {key!r}")
    for key in ("source_files", "constants"):
        if key in provenance:
            _require_str_list(provenance[key], f"{where}.{key}", errors, nonempty=False)
    if "report_ref" in provenance and not isinstance(provenance["report_ref"], str):
        _err(errors, f"{where}.report_ref", "must be a string")


def validate_document(doc: Any) -> list[str]:
    """Return every validation error in ``doc`` (empty list == valid).

    Per the accepted plan, ALL failures are detected before any matching:
    unknown keys/enums, duplicate rule ids, malformed patterns/types,
    empty matches, negative priorities, schema mismatch, canonical-hash
    mismatch, and same-priority identical-predicate exclusive conflicts.
    """
    errors: list[str] = []

    if not isinstance(doc, dict):
        return ["document: must be a JSON object"]

    allowed_top = {
        "schema_version", "registry_id", "registry_version", "registry_sha256",
        "provenance", "enums", "defaults", "rules",
    }
    unknown = set(doc) - allowed_top
    for key in sorted(unknown):
        _err(errors, "document", f"unknown top-level key {key!r}")

    # schema_version: must be the integer 1 exactly -- bools and floats are
    # rejected (True == 1 and 1.0 == 1 in Python, so a plain != check would
    # admit them; type() is exact).
    schema_version = doc.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        _err(errors, "document.schema_version",
             f"must be the integer {SCHEMA_VERSION} (schema mismatch), got {schema_version!r}")

    for key, type_name in (("registry_id", "string"), ("registry_version", "string")):
        if key in doc and (not isinstance(doc[key], str) or not doc[key]):
            _err(errors, f"document.{key}", f"must be a non-empty {type_name}")

    # Canonical hash: REQUIRED, lowercase 64-hex, recomputed and compared
    # (fail-closed on missing field, wrong format, or drift).
    if "registry_sha256" not in doc:
        _err(errors, "document.registry_sha256",
             "required (canonical SHA-256 of the document, 64 lowercase hex chars)")
    else:
        stored = doc["registry_sha256"]
        if not isinstance(stored, str) or not _SHA256_RE.match(stored):
            _err(errors, "document.registry_sha256",
                 f"must be a 64-char lowercase hex SHA-256 string, got {stored!r}")
        else:
            recomputed = canonical_hash(doc)
            if stored != recomputed:
                _err(errors, "document.registry_sha256",
                     f"mismatch: stored {stored}, recomputed {recomputed} "
                     "(edit invalidates the canonical hash; run tools/kit_roles/validate_registry.py --write-hash)")

    if "provenance" in doc:
        _validate_provenance(doc["provenance"], errors)
    if "enums" in doc:
        _validate_enums(doc["enums"], errors)
    if "defaults" in doc:
        _validate_defaults(doc["defaults"], errors)

    # Rules.
    rules = doc.get("rules")
    if rules is None:
        _err(errors, "document", "missing required 'rules' list")
        return errors
    if not isinstance(rules, list):
        _err(errors, "document.rules", "must be a list")
        return errors
    if not rules:
        _err(errors, "document.rules", "must be a non-empty list (a registry needs at least one rule)")
        return errors

    seen_ids: set[str] = set()
    parsed_rules: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        where = f"rules[{index}]"
        if not isinstance(rule, dict):
            _err(errors, where, "each rule must be an object")
            continue
        allowed_rule = {"rule_id", "priority", "match", "effect", "rationale", "source", "enabled", "deprecated_by"}
        unknown = set(rule) - allowed_rule
        for key in sorted(unknown):
            _err(errors, where, f"unknown rule field {key!r}")

        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or not _RULE_ID_RE.match(rule_id or ""):
            _err(errors, where, f"rule_id must match {_RULE_ID_RE.pattern!r}, got {rule_id!r}")
        elif rule_id in seen_ids:
            _err(errors, where, f"duplicate rule_id {rule_id!r}")
        else:
            seen_ids.add(rule_id)

        priority = rule.get("priority")
        if isinstance(priority, bool) or not isinstance(priority, int):
            _err(errors, where, f"priority must be a non-negative integer, got {priority!r}")
        elif priority < 0:
            _err(errors, where, f"priority must be non-negative, got {priority}")

        match = rule.get("match")
        mwhere = f"{where} (rule {rule_id!r})" if isinstance(rule_id, str) else where
        if not isinstance(match, dict):
            _err(errors, mwhere, "missing/invalid 'match' object")
        else:
            unknown = set(match) - {"any", "when"}
            for key in sorted(unknown):
                _err(errors, mwhere, f"unknown match key {key!r}")
            if "any" not in match:
                _err(errors, mwhere, "missing required 'match.any' (non-empty predicate list)")
            else:
                _validate_any_predicates(match["any"], rule_id if isinstance(rule_id, str) else f"[{index}]", errors)
            _validate_when(match.get("when"), rule_id if isinstance(rule_id, str) else f"[{index}]", errors)

        if "effect" not in rule:
            _err(errors, mwhere, "missing required 'effect'")
        else:
            _validate_effect(rule["effect"], rule_id if isinstance(rule_id, str) else f"[{index}]", errors)

        for key in ("rationale", "source", "deprecated_by"):
            if key in rule and not isinstance(rule[key], str):
                _err(errors, f"{where}.{key}", "must be a string")
        if "enabled" in rule and not isinstance(rule["enabled"], bool):
            _err(errors, f"{where}.enabled", "must be a boolean")

        parsed_rules.append(rule)

    # Static same-priority identical-predicate conflict detection (plan
    # item 6 + review B4/B5): only provably identical predicates count
    # (same normalized patterns on every channel, same when conditions);
    # arbitrary glob overlap is left to the exact runtime audit.
    for i in range(len(parsed_rules)):
        for j in range(i + 1, len(parsed_rules)):
            left, right = parsed_rules[i], parsed_rules[j]
            if left.get("enabled", True) is False or right.get("enabled", True) is False:
                continue
            if left.get("priority") != right.get("priority"):
                continue
            if not isinstance(left.get("priority"), int) or isinstance(left.get("priority"), bool):
                continue
            if not isinstance(left.get("match"), dict) or not isinstance(right.get("match"), dict):
                continue
            if _predicate_identity(left) != _predicate_identity(right):
                continue
            conflicts = _exclusive_conflicts(left, right)
            if conflicts:
                _err(errors, f"rules {left.get('rule_id')!r} and {right.get('rule_id')!r}",
                     f"same-priority identical-predicate conflict on exclusive field(s) {conflicts}")

    return errors


def _predicate_identity(rule: Mapping[str, Any]) -> tuple[Any, ...]:
    """Deterministic identity of a rule's predicate (channels + conditions)."""
    match = rule.get("match") or {}
    channel_patterns: list[tuple[str, tuple[str, ...]]] = []
    for predicate in match.get("any") or []:
        if not isinstance(predicate, dict):
            continue
        for channel in CHANNELS:
            if channel in predicate and isinstance(predicate[channel], list):
                channel_patterns.append((channel, _normalized_pattern_tuple(channel, predicate[channel])))
    when = match.get("when") or {}
    when_sorted = tuple(
        (key, tuple(sorted(when[key]))) for key in sorted(when) if isinstance(when.get(key), list)
    )
    return (tuple(sorted(channel_patterns)), when_sorted)


def _exclusive_conflicts(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    """Exclusive effect fields both rules set to DIFFERENT values."""
    left_effect = left.get("effect") if isinstance(left.get("effect"), dict) else {}
    right_effect = right.get("effect") if isinstance(right.get("effect"), dict) else {}
    conflicts = []
    for field in EXCLUSIVE_EFFECT_FIELDS:
        if field in left_effect and field in right_effect and left_effect[field] != right_effect[field]:
            conflicts.append(field)
    return conflicts


# ---------------------------------------------------------------------------
# Loaded rule / registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One validated, normalized rule."""

    rule_id: str
    priority: int
    predicates: tuple[tuple[str, tuple[str, ...]], ...]  # (channel, patterns) per match.any entry
    when: Mapping[str, tuple[str, ...]]  # kit / record_type / category -> OR list
    effect: Mapping[str, Any]
    rationale: str
    source: str
    enabled: bool
    deprecated_by: str

    def matches(self, model_key: Optional[str], object_id: Optional[str],
                kit: Optional[str], record_type: Optional[str], category: Optional[str]) -> bool:
        """True iff this rule matches the query (conditions ANDed, OR within lists)."""
        for channel, patterns in self.predicates:
            query = model_key if channel == "model_key" else object_id
            if query is None:
                continue
            if any(fnmatch.fnmatchcase(query, pattern) for pattern in patterns):
                # At least one predicate matched; now conditions must hold.
                return self._conditions_hold(kit, record_type, category)
        return False

    def _conditions_hold(self, kit: Optional[str], record_type: Optional[str],
                         category: Optional[str]) -> bool:
        for key, allowed in self.when.items():
            query = {"kit": kit, "record_type": record_type, "category": category}[key]
            if query is None or query not in allowed:
                return False
        return True

    def sets(self, field: str) -> bool:
        return field in self.effect

    def effect_value(self, field: str) -> Any:
        return self.effect.get(field)


class Registry:
    """A validated kit/role registry document, ready for verdicts."""

    def __init__(self, doc: Mapping[str, Any], *, path: Optional[Path] = None) -> None:
        errors = validate_document(doc)
        if errors:
            raise RegistryValidationError(errors)
        self._doc = dict(doc)
        self._path = path
        self._defaults = dict(doc["defaults"]["unmatched_verdict"])
        rules = []
        disabled = []
        for raw in doc["rules"]:
            rule = self._build_rule(raw)
            if rule.enabled:
                rules.append(rule)
            else:
                disabled.append(rule)
        # Deterministic iteration order only; exclusive fields never depend on it.
        self._rules = tuple(sorted(rules, key=lambda r: (-r.priority, r.rule_id)))
        self._disabled = tuple(sorted(disabled, key=lambda r: (-r.priority, r.rule_id)))

    @staticmethod
    def _build_rule(raw: Mapping[str, Any]) -> Rule:
        match = raw["match"]
        predicates = tuple(
            (channel, _normalized_pattern_tuple(channel, predicate[channel]))
            for predicate in match["any"]
            for channel in CHANNELS
            if channel in predicate
        )
        when = {
            key: tuple(sorted(match["when"][key]))
            for key in WHEN_CONDITIONS
            if match.get("when") is not None and key in match["when"]
        }
        effect = dict(raw["effect"])
        return Rule(
            rule_id=raw["rule_id"],
            priority=raw["priority"],
            predicates=predicates,
            when=when,
            effect=effect,
            rationale=raw.get("rationale", ""),
            source=raw.get("source", ""),
            enabled=raw.get("enabled", True),
            deprecated_by=raw.get("deprecated_by", ""),
        )

    # -- constructors -------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        """Load, validate, and hash-verify the document at ``path``."""
        path = Path(path)
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        return cls(doc, path=path)

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "Registry":
        return cls(doc)

    @classmethod
    def loads(cls, text: str) -> "Registry":
        return cls(json.loads(text))

    # -- document facts -----------------------------------------------------

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def schema_version(self) -> int:
        return self._doc["schema_version"]

    @property
    def registry_id(self) -> str:
        return self._doc["registry_id"]

    @property
    def registry_version(self) -> str:
        return self._doc.get("registry_version", "")

    @property
    def registry_sha256(self) -> str:
        return self._doc["registry_sha256"]

    @property
    def canonical_hash(self) -> str:
        return canonical_hash(self._doc)

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    @property
    def disabled_rules(self) -> tuple[Rule, ...]:
        return self._disabled

    @property
    def defaults(self) -> Mapping[str, Any]:
        return dict(self._defaults)

    def rule_ids(self) -> tuple[str, ...]:
        return tuple(rule.rule_id for rule in self._rules)

    # -- matching -----------------------------------------------------------

    def verdict(self, model_key: object = None, object_id: object = None, *,
                kit: Optional[str] = None, record_type: Optional[str] = None,
                category: Optional[str] = None) -> "Verdict":
        """Deterministic merged verdict for one key/context.

        At least one of ``model_key`` / ``object_id`` must be provided
        (a key-less query is a caller error, fail-closed).

        Exclusive fields: highest priority wins; equal-highest-priority
        disagreement raises ``RegistryConflictError``; losing (lower
        priority) rule ids are recorded per field.  ``tags`` is the union
        over all matching rules.  Unmatched keys return the explicit
        ``defaults.unmatched_verdict`` values.
        """
        if model_key is None and object_id is None:
            raise ValueError("verdict() requires at least one of model_key / object_id")
        norm_model = normalize_model_key(model_key) if model_key is not None else None
        norm_object = normalize_object_id(object_id) if object_id is not None else None

        matched = [
            rule for rule in self._rules
            if rule.matches(norm_model, norm_object, kit, record_type, category)
        ]
        if not matched:
            return Verdict(
                matched=False,
                model_key=norm_model,
                object_id=norm_object,
                structural_role=self._defaults.get("structural_role"),
                shell_eligible=bool(self._defaults.get("shell_eligible", False)),
                contact_exclude=bool(self._defaults.get("contact_exclude", False)),
                render_exclude=bool(self._defaults.get("render_exclude", False)),
                profile_exclude=bool(self._defaults.get("profile_exclude", False)),
                tags=tuple(sorted(set(self._defaults.get("tags", [])))),
                matched_rule_ids=(),
                field_sources={},
                losers={},
            )

        matched_ids = tuple(sorted(rule.rule_id for rule in matched))
        field_sources: dict[str, tuple[str, ...]] = {}
        losers: dict[str, tuple[str, ...]] = {}
        merged: dict[str, Any] = {}

        for field in EXCLUSIVE_EFFECT_FIELDS:
            setters = [rule for rule in matched if rule.sets(field)]
            if not setters:
                merged[field] = self._defaults.get(field)
                continue
            winning_priority = max(rule.priority for rule in setters)
            contenders = [rule for rule in setters if rule.priority == winning_priority]
            values = {rule.effect_value(field) for rule in contenders}
            if len(values) > 1:
                raise RegistryConflictError(
                    norm_model, norm_object, field, [rule.rule_id for rule in contenders]
                )
            merged[field] = next(iter(values))
            field_sources[field] = tuple(sorted(rule.rule_id for rule in contenders))
            field_losers = [rule for rule in setters if rule.priority < winning_priority]
            if field_losers:
                losers[field] = tuple(
                    rule.rule_id for rule in sorted(field_losers, key=lambda r: (-r.priority, r.rule_id))
                )

        tag_set: set[str] = set()
        for rule in matched:
            tag_set.update(rule.effect.get(TAGS_FIELD, []))
        tags = tuple(sorted(tag_set))

        return Verdict(
            matched=True,
            model_key=norm_model,
            object_id=norm_object,
            structural_role=merged["structural_role"],
            shell_eligible=bool(merged["shell_eligible"]),
            contact_exclude=bool(merged["contact_exclude"]),
            render_exclude=bool(merged["render_exclude"]),
            profile_exclude=bool(merged["profile_exclude"]),
            tags=tags,
            matched_rule_ids=matched_ids,
            field_sources=field_sources,
            losers=losers,
        )

    def __repr__(self) -> str:
        return (
            f"Registry(id={self.registry_id!r}, version={self.registry_version!r}, "
            f"rules={len(self._rules)}, disabled={len(self._disabled)}, "
            f"sha256={self.registry_sha256[:12]}...)"
        )


@dataclass(frozen=True)
class Verdict:
    """Merged result of ``Registry.verdict`` for one key/context.

    ``matched`` distinguishes a rule hit from the explicit unmatched
    default.  ``field_sources`` maps each exclusive field to the rule
    id(s) that set the winning value; ``losers`` maps fields to lower-
    priority rule ids that also set the field (the runtime conflict audit).
    """

    matched: bool
    model_key: Optional[str]
    object_id: Optional[str]
    structural_role: Optional[str]
    shell_eligible: bool
    contact_exclude: bool
    render_exclude: bool
    profile_exclude: bool
    tags: tuple[str, ...]
    matched_rule_ids: tuple[str, ...]
    field_sources: Mapping[str, tuple[str, ...]]
    losers: Mapping[str, tuple[str, ...]]

    @property
    def is_shell(self) -> bool:
        """Convenience: shell-count evidence (the Phase-1 consumer meaning)."""
        return self.shell_eligible

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "model_key": self.model_key,
            "object_id": self.object_id,
            "structural_role": self.structural_role,
            "shell_eligible": self.shell_eligible,
            "contact_exclude": self.contact_exclude,
            "render_exclude": self.render_exclude,
            "profile_exclude": self.profile_exclude,
            "tags": list(self.tags),
            "matched_rule_ids": list(self.matched_rule_ids),
            "field_sources": {key: list(value) for key, value in self.field_sources.items()},
            "losers": {key: list(value) for key, value in self.losers.items()},
        }
