#!/usr/bin/env python3
"""Cityforge T0.4 proof: real TES3 DOOR records through tes3conv.

Purpose
-------
Prove, reproducibly, that a masterless scratch plugin containing a DOOR base
record, one exterior and one interior CELL, a forward door reference (DODT +
non-empty DNAM destination cell name), and an interior -> exterior return door
reference (DODT + empty DNAM) survives JSON -> ESP -> JSON through
``tes3conv-master/tes3conv.exe`` exactly as authored.  This is the de-risking
proof for all later Cityforge door work (master plan T0.4); it is a proof
artifact, not production city authoring.

Inputs
------
- ``src/procgen/tes3json.py`` builders (Header/DOOR/Cell/reference).
- ``src/procgen/espscan.py`` for the independent binary scan evidence.
- ``tes3conv-master/tes3conv.exe`` (pinned by hash in the report).

Outputs (written to ``output/cityforge/proofs/door_tes3conv_v1/``)
------------------------------------------------------------------
- ``fixture.json``        source authoring document (indent 2).
- ``authored.esp``        tes3conv JSON -> ESP output.
- ``roundtrip.json``      tes3conv ESP -> JSON output.
- ``verification.json``   machine-readable evidence: commands, tool hashes,
                          artifact sha256 hashes, every assertion with
                          pass/fail, espscan summary, binary subrecord audit,
                          and the observed empty-DNAM serialization.
- ``artifacts.sha256``    sha256 of every artifact above (written last).

Invariants / safety
-------------------
- Never touches original plugins; writes only under the proof output path
  plus the tes3conv working directory it creates there.
- Deterministic fixture constants (ids, grids, transforms) chosen to permit
  exact assertions; no randomness.
- Exits nonzero on any failed assertion or any essential stage failure
  (authoring, binary scanning, round-trip) with no degraded substitute.
- Empty-DNAM expectation (evidence-driven, 2026-08-10): the tes3 crate at
  rev 51fae82b79838d76a39d0d1d0d472d7f48e8577f models door destinations as
  ``TravelDestination { translation, rotation, cell: String }`` and its
  ``Save`` emits DNAM only when ``cell`` is non-empty; a DODT without DNAM is
  the empty-DNAM return door.  ``"cell": ""`` is therefore the correct JSON
  form, and espscan observes it through the unchanged reference fields
  ``has_dodt=True, destination_cell=None``; DNAM presence/size is proven by
  the driver-local raw byte audit below (espscan is deliberately untouched).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import struct
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
TES3CONV = WORKSPACE / "tes3conv-master" / "tes3conv.exe"
OUT_DIR = WORKSPACE / "output" / "cityforge" / "proofs" / "door_tes3conv_v1"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procgen.espscan import scan_file  # noqa: E402
from procgen.tes3json import (  # noqa: E402
    build_cell,
    build_door,
    build_reference,
    new_plugin,
    read_json,
    validate,
    write_json,
)

# ---------------------------------------------------------------------------
# Deterministic fixture constants (chosen to permit exact assertions).
# ---------------------------------------------------------------------------

DOOR_ID = "cf_t04_door_01"
DOOR_MESH = r"x\ex_door_01.nif"
EXTERIOR_NAME = "cf_t04_exterior"
EXTERIOR_GRID = [-95, -11]
INTERIOR_NAME = "cf_t04_interior"
INTERIOR_GRID = [0, 0]

FORWARD_DOOR = {
    "refr_index": 1,
    "translation": [128.0, 256.0, 384.5],
    "rotation": [0.5, -1.5, 2.0],
    "destination": {
        "translation": [512.0, 640.0, 96.0],
        "rotation": [0.25, -0.5, 1.75],
        "cell": INTERIOR_NAME,
    },
}

RETURN_DOOR = {
    "refr_index": 2,
    "translation": [512.0, 640.0, 96.0],
    "rotation": [0.25, -0.5, 1.75],
    "destination": {
        "translation": [128.0, 256.0, 384.5],
        "rotation": [0.5, -1.5, 2.0],
        "cell": "",  # empty DNAM: return door, engine resolves by position
    },
}


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Proof:
    """Collects assertions and evidence, then renders the verification JSON."""

    assertions: list[dict[str, Any]] = field(default_factory=list)

    def check(self, assertion_id: str, description: str, passed: bool, detail: Any = None) -> bool:
        self.assertions.append(
            {
                "id": assertion_id,
                "description": description,
                "passed": bool(passed),
                "detail": detail,
            }
        )
        return bool(passed)


def build_fixture() -> list[dict[str, Any]]:
    """Assemble the deterministic proof document (see module docstring)."""
    document = new_plugin(
        {
            "author": "Cityforge T0.4 proof",
            "description": "DOOR records + linked exterior/interior pair through tes3conv, empty-DNAM return door",
            "masters": [],
        }
    )
    document.append(build_door(DOOR_ID, name="", mesh=DOOR_MESH, flags="PERSISTENT"))
    document.append(
        build_cell(
            EXTERIOR_NAME,
            EXTERIOR_GRID,
            references=[build_reference(DOOR_ID, **FORWARD_DOOR)],
        )
    )
    document.append(
        build_cell(
            INTERIOR_NAME,
            INTERIOR_GRID,
            interior=True,
            references=[build_reference(DOOR_ID, **RETURN_DOOR)],
        )
    )
    return document


def run_tes3conv(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(TES3CONV), "-o", "-c", *arguments],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=150,
        check=False,
    )


def decode_text(payload: bytes) -> str:
    return payload.split(b"\0", 1)[0].decode("cp1252", errors="replace")


def walk_subrecords(body: bytes) -> list[tuple[bytes, int, bytes]]:
    """Yield ``(tag, size, payload)`` triples for a TES3 record body."""
    result = []
    pos = 0
    while pos + 8 <= len(body):
        tag = body[pos : pos + 4]
        (size,) = struct.unpack_from("<I", body, pos + 4)
        payload = body[pos + 8 : pos + 8 + size]
        result.append((tag, size, payload))
        pos += 8 + size
    if pos != len(body):
        raise ValueError("trailing bytes in record body")
    return result


def parse_hedr_masters(body: bytes) -> tuple[float, int, list[str]]:
    """Read the TES3 header record: version, raw file-type integer, master names.

    Per the tes3 crate (rev 51fae82): HEDR is a fixed 300-byte subrecord
    (version f32, FileType repr(u32) where Esp=0, author FixedString<32>,
    description FixedString<256>, num_objects u32); masters are *separate*
    MAST + DATA(8) subrecord pairs following HEDR, so a masterless plugin has
    zero MAST subrecords.
    """
    subrecords = walk_subrecords(body)
    hedr = dict((tag, data) for tag, _size, data in subrecords)[b"HEDR"]
    version = struct.unpack_from("<f", hedr, 0)[0]
    (file_type,) = struct.unpack_from("<I", hedr, 4)
    names = []
    for tag, _size, payload in subrecords:
        if tag == b"MAST":
            names.append(payload.split(b"\0", 1)[0].decode("cp1252"))
    return float(version), int(file_type), names


def audit_reference_groups(cell_body: bytes) -> list[dict[str, Any]]:
    """Walk the inline FRMR groups of a CELL body.

    Returns one record per reference: packed index, object id, DODT size, and
    DNAM size (``None`` when the reference carries no DNAM subrecord).
    """
    references = []
    subrecords = walk_subrecords(cell_body)
    index = 0
    while index < len(subrecords):
        tag, _size, _payload = subrecords[index]
        if tag != b"FRMR":
            index += 1
            continue
        (packed,) = struct.unpack_from("<I", subrecords[index][2])
        object_id = None
        dodt_size = None
        dnam_size = None
        index += 1
        while index < len(subrecords) and subrecords[index][0] != b"FRMR":
            inner_tag, inner_size, inner_payload = subrecords[index]
            if inner_tag == b"NAME":
                object_id = decode_text(inner_payload)
            elif inner_tag == b"DODT":
                dodt_size = inner_size
            elif inner_tag == b"DNAM":
                dnam_size = inner_size
            index += 1
        references.append(
            {
                "packed": packed,
                "mast_index": (packed >> 24) & 0xFF,
                "refr_index": packed & 0xFFFFFF,
                "id": object_id,
                "dodt_size": dodt_size,
                "dnam_size": dnam_size,
            }
        )
    return references


def binary_audit(path: Path) -> dict[str, Any]:
    """Independent byte-level audit of the authored ESP.

    Walks record bodies directly (no tes3conv, no JSON): header master count,
    DOOR record presence, and per-CELL door-reference DODT/DNAM subrecord
    sizes.  This is the evidence that JSON round-trip alone did not produce.
    """
    data = path.read_bytes()
    audit: dict[str, Any] = {"records": [], "cells": {}}
    pos = 0
    while pos + 16 <= len(data):
        tag = data[pos : pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        body = data[pos + 16 : pos + 16 + size]
        if tag == b"TES3":
            version, file_type, masters = parse_hedr_masters(body)
            audit["header"] = {
                "version": version,
                "file_type": file_type,
                "master_count": len(masters),
                "masters": masters,
                "hedr_size": size,
            }
        elif tag == b"DOOR":
            audit["records"].append(
                {
                    "type": "DOOR",
                    "body_size": size,
                    "subrecords": [(s.decode("latin1"), ss) for s, ss, _ in walk_subrecords(body)],
                }
            )
        elif tag == b"CELL":
            cell = {"body_size": size, "references": audit_reference_groups(body)}
            cell_subrecords = walk_subrecords(body)
            name = next((decode_text(p) for s, _ss, p in cell_subrecords if s == b"NAME"), None)
            data_payload = next((p for s, _ss, p in cell_subrecords if s == b"DATA"), None)
            if data_payload is not None and len(data_payload) >= 12:
                flags, grid_x, grid_y = struct.unpack_from("<Iii", data_payload)
            else:
                flags, grid_x, grid_y = None, None, None
            cell["name"] = name
            cell["data_flags"] = flags
            cell["grid"] = [grid_x, grid_y] if grid_x is not None else None
            audit["cells"][str(name)] = cell
        pos += 16 + size
    return audit


def assert_destination(ref: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Exact assertion on a round-tripped door reference destination."""
    destination = ref.get("destination")
    if not isinstance(destination, Mapping):
        return False
    for key in ("translation", "rotation", "cell"):
        actual = destination.get(key)
        expected_value = expected[key]
        if key == "cell":
            if actual != expected_value:
                return False
        else:
            if not isinstance(actual, list) or len(actual) != 3:
                return False
            for actual_value, wanted in zip(actual, expected_value):
                if not isinstance(actual_value, (int, float)) or not math.isclose(
                    float(actual_value), float(wanted), rel_tol=1e-9, abs_tol=1e-9
                ):
                    return False
    return True


def find_ref(roundtrip: list[dict[str, Any]], cell_name: str, refr_index: int) -> Mapping[str, Any]:
    for record in roundtrip:
        if record.get("type") == "Cell" and record.get("name") == cell_name:
            for reference in record.get("references", []):
                if reference.get("refr_index") == refr_index:
                    return reference
    raise AssertionError(f"reference {refr_index} not found in cell {cell_name!r}")


def main() -> int:
    proof = Proof()
    failures: list[str] = []

    def require(assertion_id: str, description: str, passed: bool, detail: Any = None) -> None:
        if not proof.check(assertion_id, description, passed, detail):
            failures.append(assertion_id)

    # --- 1. Author the fixture document --------------------------------
    document = build_fixture()

    issues = validate(document)
    require(
        "A1-validator-clean",
        "tes3json.validate() returns no issues for the fixture (incl. empty-DNAM return door)",
        not issues,
        [str(issue) for issue in issues],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = OUT_DIR / "fixture.json"
    esp_path = OUT_DIR / "authored.esp"
    roundtrip_path = OUT_DIR / "roundtrip.json"
    verification_path = OUT_DIR / "verification.json"
    manifest_path = OUT_DIR / "artifacts.sha256"
    for old in OUT_DIR.iterdir():
        if old.name in {"fixture.json", "authored.esp", "roundtrip.json", "verification.json", "artifacts.sha256"}:
            old.unlink()

    write_json(document, fixture_path, indent=2)

    # --- 2. Author the ESP and round-trip it ---------------------------
    author = run_tes3conv([str(fixture_path), str(esp_path)], OUT_DIR)
    require(
        "A2-author-exit-zero",
        "tes3conv JSON->ESP exits 0",
        author.returncode == 0,
        {"exit": author.returncode, "stdout": author.stdout, "stderr": author.stderr},
    )
    esp_exists = esp_path.is_file() and esp_path.stat().st_size > 0
    require("A3-author-esp-nonempty", "authored.esp exists and is non-empty", esp_exists, {"size": esp_path.stat().st_size if esp_path.is_file() else None})
    if author.returncode != 0 or not esp_exists:
        print("FAILURE: author <tes3conv JSON->ESP failed>")
        return 1

    roundtrip_proc = run_tes3conv([str(esp_path), str(roundtrip_path)], OUT_DIR)
    require(
        "A4-roundtrip-exit-zero",
        "tes3conv ESP->JSON exits 0",
        roundtrip_proc.returncode == 0,
        {"exit": roundtrip_proc.returncode, "stdout": roundtrip_proc.stdout, "stderr": roundtrip_proc.stderr},
    )
    if roundtrip_proc.returncode != 0 or not roundtrip_path.is_file():
        print("FAILURE: roundtrip <tes3conv ESP->JSON failed>")
        return 1
    roundtrip = read_json(roundtrip_path)

    # --- 3. Round-trip JSON assertions ---------------------------------
    header = roundtrip[0]
    require(
        "A5-header-masterless",
        "round-tripped Header masters == []",
        header.get("type") == "Header" and header.get("masters") == [],
        {"masters": header.get("masters")},
    )
    require(
        "A6-header-file-type-esp",
        "round-tripped Header file_type == Esp",
        header.get("file_type") == "Esp",
        {"file_type": header.get("file_type")},
    )

    doors = [record for record in roundtrip if record.get("type") == "Door"]
    require(
        "A7-door-base-record",
        "round-tripped document contains the DOOR base record with id and mesh",
        len(doors) == 1 and doors[0].get("id") == DOOR_ID and doors[0].get("mesh") == DOOR_MESH,
        {"doors": doors},
    )

    exterior = next((r for r in roundtrip if r.get("type") == "Cell" and r.get("name") == EXTERIOR_NAME), None)
    interior = next((r for r in roundtrip if r.get("type") == "Cell" and r.get("name") == INTERIOR_NAME), None)
    require(
        "A8-exterior-cell",
        "exterior CELL present with authored grid and no interior flag",
        exterior is not None
        and exterior.get("data", {}).get("grid") == EXTERIOR_GRID
        and exterior.get("data", {}).get("flags") != "IS_INTERIOR",
        {"cell": exterior},
    )
    require(
        "A9-interior-cell",
        "interior CELL present with IS_INTERIOR flag",
        interior is not None
        and "IS_INTERIOR" in interior.get("data", {}).get("flags", "")
        and len(interior.get("references", [])) == 1,
        {"cell": interior},
    )

    forward = find_ref(roundtrip, EXTERIOR_NAME, FORWARD_DOOR["refr_index"])
    return_ref = find_ref(roundtrip, INTERIOR_NAME, RETURN_DOOR["refr_index"])
    require(
        "A10-forward-ref-index",
        "forward door reference keeps refr_index 1",
        forward.get("refr_index") == 1,
        {"refr_index": forward.get("refr_index")},
    )
    require(
        "A11-return-ref-index",
        "return door reference keeps refr_index 2",
        return_ref.get("refr_index") == 2,
        {"refr_index": return_ref.get("refr_index")},
    )
    require(
        "A12-forward-ref-transform",
        "forward door reference translation/rotation round-trip exactly",
        forward.get("translation") == FORWARD_DOOR["translation"] and forward.get("rotation") == FORWARD_DOOR["rotation"],
        {"translation": forward.get("translation"), "rotation": forward.get("rotation")},
    )
    require(
        "A13-forward-ref-destination",
        "forward door destination DODT values round-trip exactly with cell name",
        assert_destination(forward, FORWARD_DOOR["destination"]),
        {"destination": forward.get("destination")},
    )
    require(
        "A14-return-ref-destination",
        "return door destination DODT values round-trip exactly",
        assert_destination(return_ref, RETURN_DOOR["destination"]),
        {"destination": return_ref.get("destination")},
    )
    require(
        "A15-empty-dnam-serialized",
        "return door destination cell is the empty string (empty-DNAM serialization observed)",
        return_ref.get("destination", {}).get("cell") == "",
        {"destination.cell": return_ref.get("destination", {}).get("cell")},
    )
    require(
        "A16-refs-persistent",
        "both door references are persistent (temporary false)",
        forward.get("temporary") is False and return_ref.get("temporary") is False,
        {"forward.temporary": forward.get("temporary"), "return.temporary": return_ref.get("temporary")},
    )

    # --- 4. Independent binary evidence --------------------------------
    scan = scan_file(esp_path, source_kit="vanilla", collect_cells=True)
    scan_summary = scan.to_dict()
    cells = {cell.name: cell for cell in scan.cells}
    require(
        "A17-scan-record-counts",
        "binary scan finds exactly one DOOR record and two CELL records",
        scan.record_counts.get("DOOR") == 1 and scan.record_counts.get("CELL") == 2,
        {"record_counts": dict(scan.record_counts)},
    )
    require(
        "A18-scan-cell-kinds",
        "binary scan classifies one exterior and one interior cell",
        scan.exterior_cells == 1 and scan.interior_cells == 1,
        {"exterior_cells": scan.exterior_cells, "interior_cells": scan.interior_cells},
    )
    exterior_cell = cells.get(EXTERIOR_NAME)
    interior_cell = cells.get(INTERIOR_NAME)
    forward_scan_ref = exterior_cell.references[0] if exterior_cell and exterior_cell.references else None
    return_scan_ref = interior_cell.references[0] if interior_cell and interior_cell.references else None
    require(
        "A19-scan-forward-door",
        "scan sees forward door ref with DODT and destination cell name",
        forward_scan_ref is not None
        and forward_scan_ref.has_dodt
        and forward_scan_ref.destination_cell == INTERIOR_NAME
        and forward_scan_ref.destination_position == tuple(FORWARD_DOOR["destination"]["translation"]),
        {
            "has_dodt": forward_scan_ref.has_dodt if forward_scan_ref else None,
            "destination_cell": forward_scan_ref.destination_cell if forward_scan_ref else None,
            "destination_position": forward_scan_ref.destination_position if forward_scan_ref else None,
        },
    )
    require(
        "A20-scan-return-door",
        "scan sees return door ref with DODT but no destination cell (empty-DNAM return door)",
        return_scan_ref is not None
        and return_scan_ref.has_dodt
        and return_scan_ref.destination_cell is None
        and return_scan_ref.destination_position == tuple(RETURN_DOOR["destination"]["translation"]),
        {
            "has_dodt": return_scan_ref.has_dodt if return_scan_ref else None,
            "destination_cell": return_scan_ref.destination_cell if return_scan_ref else None,
            "destination_position": return_scan_ref.destination_position if return_scan_ref else None,
        },
    )

    audit = binary_audit(esp_path)
    header_audit = audit.get("header", {})
    require(
        "A21-binary-header-masterless",
        "raw header shows zero MAST subrecords (masterless) and Esp file type (u32 0)",
        header_audit.get("master_count") == 0 and header_audit.get("file_type") == 0,
        header_audit,
    )
    require(
        "A22-binary-door-record",
        "raw ESP contains the DOOR record with NAME+MODL subrecords",
        any(entry["type"] == "DOOR" for entry in audit["records"]),
        audit["records"],
    )
    forward_bin = audit["cells"].get(EXTERIOR_NAME, {}).get("references", [None])[0]
    return_bin = audit["cells"].get(INTERIOR_NAME, {}).get("references", [None])[0]
    require(
        "A23-binary-forward-subrecords",
        "forward door ref bytes: DODT size 24 and DNAM size 16",
        bool(forward_bin) and forward_bin.get("dodt_size") == 24 and forward_bin.get("dnam_size") == len(INTERIOR_NAME) + 1,
        forward_bin,
    )
    require(
        "A24-binary-return-subrecords",
        "return door ref bytes: DODT size 24 and NO DNAM subrecord",
        bool(return_bin) and return_bin.get("dodt_size") == 24 and return_bin.get("dnam_size") is None,
        return_bin,
    )

    # --- 5. Machine-readable evidence ----------------------------------
    driver_text = Path(__file__).read_text(encoding="utf-8")
    scan_summary = scan.to_dict()
    # Timing is not part of the evidence contract and would make the JSON
    # non-reproducible; drop it so verification.json is byte-stable across
    # re-runs.
    scan_summary.pop("elapsed_seconds", None)
    evidence = {
        "proof": "cityforge-t04-door-tes3conv-v1",
        "date": "2026-08-10",
        "fixture": {
            "door_id": DOOR_ID,
            "door_mesh": DOOR_MESH,
            "exterior_name": EXTERIOR_NAME,
            "exterior_grid": EXTERIOR_GRID,
            "interior_name": INTERIOR_NAME,
            "interior_grid": INTERIOR_GRID,
            "forward_door": FORWARD_DOOR,
            "return_door": RETURN_DOOR,
        },
        "commands": [
            {"step": "author", "argv": [str(TES3CONV), "-o", "-c", str(fixture_path), str(esp_path)], "exit_code": author.returncode},
            {"step": "roundtrip", "argv": [str(TES3CONV), "-o", "-c", str(esp_path), str(roundtrip_path)], "exit_code": roundtrip_proc.returncode},
        ],
        "tool_hashes": {
            "tes3conv.exe": sha256_file(TES3CONV),
            "door_tes3conv_proof.py": sha256_text(driver_text),
            "tes3json.py": sha256_text(Path(SRC / "procgen" / "tes3json.py").read_text(encoding="utf-8")),
            "espscan.py": sha256_text(Path(SRC / "procgen" / "espscan.py").read_text(encoding="utf-8")),
        },
        "artifact_hashes": {
            "fixture.json": sha256_file(fixture_path),
            "authored.esp": sha256_file(esp_path),
            "roundtrip.json": sha256_file(roundtrip_path),
        },
        "observed_empty_dnam": {
            "roundtrip_destination_cell": return_ref.get("destination", {}).get("cell"),
            "binary_dnam_size": return_bin.get("dnam_size") if return_bin else None,
            "binary_dodt_size": return_bin.get("dodt_size") if return_bin else None,
            "espscan_destination_cell": return_scan_ref.destination_cell if return_scan_ref else None,
            "forward_binary_dnam_size": forward_bin.get("dnam_size") if forward_bin else None,
        },
        "espscan": scan_summary,
        "espscan_door_refs": {
            "forward": {
                "has_dodt": forward_scan_ref.has_dodt if forward_scan_ref else None,
                "destination_cell": forward_scan_ref.destination_cell if forward_scan_ref else None,
                "destination_position": list(forward_scan_ref.destination_position) if forward_scan_ref else None,
                "destination_rotation": list(forward_scan_ref.destination_rotation) if forward_scan_ref else None,
            },
            "return": {
                "has_dodt": return_scan_ref.has_dodt if return_scan_ref else None,
                "destination_cell": return_scan_ref.destination_cell if return_scan_ref else None,
                "destination_position": list(return_scan_ref.destination_position) if return_scan_ref else None,
                "destination_rotation": list(return_scan_ref.destination_rotation) if return_scan_ref else None,
            },
        },
        "binary_audit": audit,
        "roundtrip_observations": {
            "header": header,
            "door": doors[0] if doors else None,
            "exterior_cell": exterior,
            "interior_cell": interior,
            "forward_ref": forward,
            "return_ref": return_ref,
        },
        "assertions": proof.assertions,
        "passed": not failures,
        "failure_ids": failures,
    }
    verification_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")

    manifest_lines = []
    for name in ("fixture.json", "authored.esp", "roundtrip.json", "verification.json"):
        manifest_lines.append(f"{sha256_file(OUT_DIR / name)}  {name}")
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8", newline="\n")

    # --- 6. Human-readable summary -------------------------------------
    print("Cityforge T0.4 door-through-tes3conv proof")
    print(f"outputs: {OUT_DIR}")
    print(f"tes3conv author exit={author.returncode}, roundtrip exit={roundtrip_proc.returncode}")
    print(f"authored.esp size={esp_path.stat().st_size} bytes")
    passed_count = sum(1 for assertion in proof.assertions if assertion["passed"])
    print(f"assertions: {passed_count}/{len(proof.assertions)} passed")
    for assertion in proof.assertions:
        marker = "PASS" if assertion["passed"] else "FAIL"
        print(f"  [{marker}] {assertion['id']} {assertion['description']}")
    if failures:
        print("FAILED ASSERTIONS: " + ", ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
