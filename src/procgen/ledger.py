"""Append-only project ledger with artifact hashes."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, TypeAlias

from .provenance import PathLike, sha256_file

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parents[2] / ".opencode" / "ledger.jsonl"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _artifact_record(path: PathLike) -> dict[str, str]:
    resolved = Path(path)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


class ProjectLedger:
    """A minimal JSON-lines ledger; records are never rewritten in place."""

    def __init__(self, path: PathLike = DEFAULT_LEDGER_PATH) -> None:
        self.path = Path(path)

    def record(
        self,
        task_slug: str,
        status: str,
        artifacts: Iterable[PathLike],
        notes: str,
    ) -> dict[str, object]:
        """Append one timestamped entry and return the serialized object."""

        if not task_slug or not status:
            raise ValueError("task_slug and status must be non-empty")
        artifact_records = [_artifact_record(path) for path in artifacts]
        entry: dict[str, object] = {
            "timestamp_utc": _timestamp(),
            "task_slug": task_slug,
            "status": status,
            "artifacts": artifact_records,
            "notes": notes,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        return entry

    def tail(self, n: int) -> list[dict[str, object]]:
        """Read at most the last ``n`` valid JSON-lines entries."""

        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0 or not self.path.exists():
            return []
        records: deque[dict[str, object]] = deque(maxlen=n)
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid ledger JSON at {self.path}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"ledger entry at {self.path}:{line_number} is not an object"
                    )
                records.append(value)
        return list(records)


def record(
    task_slug: str,
    status: str,
    artifacts: Iterable[PathLike],
    notes: str,
    *,
    ledger_path: PathLike = DEFAULT_LEDGER_PATH,
) -> dict[str, object]:
    """Functional convenience wrapper around :class:`ProjectLedger`."""

    return ProjectLedger(ledger_path).record(task_slug, status, artifacts, notes)


def tail(
    n: int,
    *,
    ledger_path: PathLike = DEFAULT_LEDGER_PATH,
) -> list[dict[str, object]]:
    """Functional convenience wrapper around :class:`ProjectLedger.tail`."""

    return ProjectLedger(ledger_path).tail(n)
