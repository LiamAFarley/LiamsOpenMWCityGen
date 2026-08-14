"""Fail-closed stamp eligibility for the Cityforge visual planner.

Pipeline position
------------------
This module is the metadata gate between accepted D-STAMP geometry libraries
and the visual-plan structural/render stages.  It reads the accepted Markarth
stamp palette plus each library's own exclusion metadata; it never edits those
read-only products and never decides whether a stamp is aesthetically useful.
The renderer therefore cannot accidentally turn a quarantined source record
into a planning example merely because its geometry is present in a library.

Inputs and outputs
------------------
``build_eligibility_policy`` consumes the D-STAMP library JSON paths and, for
the Markarth library, the accepted final palette ``catalog.json``.  It returns
an immutable policy containing accepted IDs and deterministic rejection
reasons.  ``validate_document`` checks every placement in a visual-plan
document and returns JSON-ready gate findings; callers use
``require_document`` to fail before advisory analysis or Pillow rendering.

Invariants
----------
* Missing accepted metadata is an error, never an implicit allow.
* A palette entry whose status is not ``eligible`` can never be selected.
* Library exclusion rows are applied even when a corresponding geometry stamp
  was emitted in the library's main ``stamps`` array.
* Matching uses stable source/unit/stamp tokens, not display names alone.
* No random choices or filesystem-order dependence are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class StampEligibilityError(ValueError):
    """Raised when accepted stamp metadata cannot establish eligibility."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StampEligibilityError(f"eligibility metadata is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StampEligibilityError(f"eligibility metadata is not an object: {path}")
    return value


def _values(value: Any) -> Iterable[str]:
    """Yield non-empty scalar identifiers from a metadata value."""

    if isinstance(value, str) and value.strip():
        yield value.strip()
    elif isinstance(value, Mapping):
        for key in ("stamp_id", "unit_id", "split_record_id", "candidate_id", "slug", "file"):
            yield from _values(value.get(key))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _values(item)


def _identifier_tokens(values: Iterable[Any]) -> frozenset[str]:
    """Build normalized exact-match tokens for source and library IDs.

    Markarth palette records use ``unit_u31_name`` while D-STAMP source rows
    use ``u31_name``.  Both spellings are deliberately retained, together
    with the original ID, so the gate is strict without depending on a single
    producer's prefix convention.
    """

    tokens: set[str] = set()
    for raw in _values(list(values)):
        token = raw.replace("\\", "/").strip().casefold()
        if not token:
            continue
        token = token.rsplit("/", 1)[-1]
        if token.endswith(".png"):
            token = token[:-4]
        if token.endswith("_sheet_2x3"):
            token = token[:-len("_sheet_2x3")]
        tokens.add(token)
        if token.startswith("unit_"):
            tokens.add(token[len("unit_"):])
        elif token.startswith("u") and "_" in token:
            tokens.add("unit_" + token)
    return frozenset(tokens)


def _stamp_tokens(stamp_id: str, stamp: Mapping[str, Any]) -> frozenset[str]:
    source = stamp.get("source") if isinstance(stamp.get("source"), Mapping) else {}
    values = [
        stamp_id,
        stamp.get("unit_id"),
        stamp.get("slug"),
        source.get("unit_id"),
        source.get("slug"),
        source.get("split_record_id"),
    ]
    return _identifier_tokens(values)


def _metadata_tokens(record: Mapping[str, Any]) -> frozenset[str]:
    values = [
        record.get("stamp_id"), record.get("unit_id"), record.get("slug"),
        record.get("split_record_id"), record.get("candidate_id"), record.get("file"),
    ]
    return _identifier_tokens(values)


def _library_exclusion_rows(library: Mapping[str, Any]) -> list[tuple[frozenset[str], str]]:
    """Collect every exclusion form emitted by the accepted library builders."""

    stats = library.get("stats") if isinstance(library.get("stats"), Mapping) else {}
    rows: list[tuple[frozenset[str], str]] = []

    for section_name in ("excluded", "source_recorded_exclusions"):
        section = stats.get(section_name, [])
        if not isinstance(section, list):
            continue
        for row in section:
            if not isinstance(row, Mapping):
                continue
            tokens = _metadata_tokens(row)
            if tokens:
                reason = str(row.get("reason") or row.get("audit_reason") or section_name)
                rows.append((tokens, f"library metadata exclusion ({reason})"))

    audit = stats.get("audit_exclusions") if isinstance(stats.get("audit_exclusions"), Mapping) else {}
    decisions = audit.get("decisions", [])
    if isinstance(decisions, list):
        for row in decisions:
            if not isinstance(row, Mapping):
                continue
            tokens = _metadata_tokens(row)
            if tokens:
                reason = str(row.get("reason") or "audited library exclusion")
                rows.append((tokens, f"library audit exclusion ({reason})"))

    excluded_ids = audit.get("excluded_unit_ids", [])
    for value in _values(excluded_ids):
        rows.append((_identifier_tokens([value]), "library audit exclusion (excluded_unit_ids)"))

    top_level = library.get("excluded", [])
    if isinstance(top_level, list):
        for row in top_level:
            if not isinstance(row, Mapping):
                continue
            tokens = _metadata_tokens(row)
            if tokens:
                rows.append((tokens, f"library exclusion ({row.get('reason', 'excluded')})"))
    return rows


@dataclass(frozen=True)
class StampEligibilityPolicy:
    """Immutable accepted/rejected stamp inventory for one input bundle."""

    accepted_stamp_ids: frozenset[str]
    rejected_stamp_ids: Mapping[str, str]
    rejected_metadata_tokens: frozenset[str]
    metadata_hashes: Mapping[str, str]
    palette_path: str | None

    def is_eligible(self, stamp_id: str) -> bool:
        """Return whether a stable D-STAMP ID passed all metadata gates."""

        return stamp_id in self.accepted_stamp_ids and stamp_id not in self.rejected_stamp_ids

    def rejection_reason(self, stamp_id: str) -> str:
        if stamp_id in self.rejected_stamp_ids:
            return str(self.rejected_stamp_ids[stamp_id])
        if _identifier_tokens([stamp_id]).intersection(self.rejected_metadata_tokens):
            return "accepted palette exclusion/quarantine metadata"
        return "stamp is absent from the accepted eligibility inventory"

    def validate_document(self, document: Mapping[str, Any],
                          stamp_geometry: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
        """Return deterministic placement-level failures without rendering."""

        issues: list[dict[str, str]] = []
        placements = document.get("stamps", [])
        if not isinstance(placements, list):
            return issues
        for index, placement in enumerate(placements):
            if not isinstance(placement, Mapping):
                continue
            stamp_id = str(placement.get("stamp_id", ""))
            if self.is_eligible(stamp_id):
                continue
            issues.append({
                "path": f"$.stamps[{index}].stamp_id",
                "stamp_id": stamp_id,
                "lot_id": str(placement.get("lot_id", "")),
                "reason": self.rejection_reason(stamp_id),
            })
        return issues

    def require_document(self, document: Mapping[str, Any],
                         stamp_geometry: Mapping[str, Mapping[str, Any]]) -> None:
        """Raise before analysis/rendering when any known stamp is ineligible."""

        issues = self.validate_document(document, stamp_geometry)
        if not issues:
            return
        detail = "; ".join(
            f"{row['lot_id']}:{row['stamp_id']} ({row['reason']})" for row in issues
        )
        raise StampEligibilityError(f"stamp eligibility gate rejected {len(issues)} placement(s): {detail}")


def _palette_records(path: Path) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    catalog = _load_json(path)
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        raise StampEligibilityError(f"accepted palette has no entries list: {path}")
    return [row for row in entries if isinstance(row, Mapping)], catalog


def build_eligibility_policy(
    library_paths: Sequence[Path], *, palette_path: Path | None = None,
) -> StampEligibilityPolicy:
    """Load accepted metadata and return a fail-closed stamp policy.

    Markarth requires the accepted final palette.  Karthgad has no separate
    palette in this workspace, so its D-STAMP inventory plus all emitted
    exclusion sections are authoritative.  A library with missing metadata or
    a stamp that cannot be matched to its accepted palette is rejected rather
    than silently allowed.
    """

    accepted: set[str] = set()
    rejected: dict[str, str] = {}
    rejected_tokens: set[str] = set()
    hashes: dict[str, str] = {}
    palette_records: list[Mapping[str, Any]] = []
    palette_catalog: Mapping[str, Any] = {}

    paths = tuple(sorted((Path(path) for path in library_paths), key=lambda item: str(item)))
    if not paths:
        raise StampEligibilityError("no stamp library metadata paths supplied")

    has_markarth = False
    loaded_libraries: list[tuple[Path, Mapping[str, Any]]] = []
    for path in paths:
        if not path.is_file():
            raise StampEligibilityError(f"stamp library metadata is missing: {path}")
        library = _load_json(path)
        library_id = str(library.get("library_id", ""))
        if not library_id:
            raise StampEligibilityError(f"stamp library has no library_id: {path}")
        stamps = library.get("stamps")
        if not isinstance(stamps, list):
            raise StampEligibilityError(f"stamp library has no stamps list: {path}")
        has_markarth = has_markarth or "markarth" in library_id.casefold()
        hashes[str(path)] = _sha256(path)
        loaded_libraries.append((path, library))

    if has_markarth:
        if palette_path is None or not Path(palette_path).is_file():
            raise StampEligibilityError(
                f"accepted Markarth palette metadata is missing: {palette_path}"
            )
        palette_path = Path(palette_path)
        palette_records, palette_catalog = _palette_records(palette_path)
        hashes[str(palette_path)] = _sha256(palette_path)
        if palette_catalog.get("catalog_id") != "markarth-stamp-palette-v1":
            raise StampEligibilityError(f"unexpected accepted palette catalog: {palette_path}")

    palette_by_token: dict[str, list[Mapping[str, Any]]] = {}
    for entry in palette_records:
        if entry.get("status") != "eligible":
            rejected_tokens.update(_metadata_tokens(entry))
            # The palette may quarantine a connection sheet that is not a
            # D-STAMP building geometry record.  Preserve a stable visual
            # planner alias for that second Castle Barracks record so it is
            # still explicitly rejected instead of disappearing at the
            # library boundary.
            for token in _metadata_tokens(entry):
                if token.startswith("unit_"):
                    rejected.setdefault(
                        "markarth_side_v1__" + token[len("unit_"):],
                        f"accepted Markarth palette exclusion: {entry.get('excluded_reason') or 'not eligible'}",
                    )
                elif token.startswith("conn_"):
                    rejected.setdefault(
                        "markarth_side_v1__" + token,
                        f"accepted Markarth palette exclusion: {entry.get('excluded_reason') or 'not eligible'}",
                    )
        for token in _metadata_tokens(entry):
            palette_by_token.setdefault(token, []).append(entry)

    for path, library in loaded_libraries:
        library_id = str(library.get("library_id", ""))
        library_exclusions = _library_exclusion_rows(library)
        for stamp in library.get("stamps", []):
            if not isinstance(stamp, Mapping):
                continue
            stamp_id = str(stamp.get("stamp_id", ""))
            if not stamp_id:
                raise StampEligibilityError(f"stamp library contains a stamp without stamp_id: {path}")
            tokens = _stamp_tokens(stamp_id, stamp)
            matching_exclusions = [reason for excluded, reason in library_exclusions
                                   if tokens.intersection(excluded)]
            if matching_exclusions:
                rejected[stamp_id] = sorted(matching_exclusions)[0]
                continue

            if "markarth" in library_id.casefold():
                matches: list[Mapping[str, Any]] = []
                seen: set[int] = set()
                for token in sorted(tokens):
                    for entry in palette_by_token.get(token, []):
                        marker = id(entry)
                        if marker not in seen:
                            seen.add(marker)
                            matches.append(entry)
                if not matches:
                    rejected[stamp_id] = "no matching entry in accepted Markarth palette"
                    continue
                statuses = sorted({str(entry.get("status", "")) for entry in matches})
                eligible_matches = [entry for entry in matches if entry.get("status") == "eligible"]
                if not eligible_matches:
                    reason = str(matches[0].get("excluded_reason") or "palette status is not eligible")
                    rejected[stamp_id] = f"accepted Markarth palette exclusion: {reason}"
                    continue
                if len(matches) != len(eligible_matches):
                    rejected[stamp_id] = (
                        "accepted Markarth palette has conflicting eligibility records "
                        f"({', '.join(statuses)})"
                    )
                    continue

            accepted.add(stamp_id)

    return StampEligibilityPolicy(
        accepted_stamp_ids=frozenset(sorted(accepted)),
        rejected_stamp_ids=dict(sorted(rejected.items())),
        rejected_metadata_tokens=frozenset(sorted(rejected_tokens)),
        metadata_hashes=dict(sorted(hashes.items())),
        palette_path=str(palette_path) if palette_path is not None else None,
    )


__all__ = ["StampEligibilityError", "StampEligibilityPolicy", "build_eligibility_policy"]
