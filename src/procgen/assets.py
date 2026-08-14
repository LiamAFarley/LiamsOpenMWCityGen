"""Read-only Tamriel_Data NIF asset catalog and kit queries.

The registry deliberately treats NIFs as opaque files.  It records only the
TES3 mesh path, file size, and the path-based classification needed by the
generation stages.  In particular, this module must not be tempted to inspect
NIF geometry: the source data under ``C:\\Modding`` is a read-only dependency.

The path rules here follow the tileset inventory report.  ``b`` is body parts,
not buildings; construction exteriors are in ``x`` and interiors in ``i``.
The distinction is represented explicitly in each entry so later palette
builders cannot accidentally use body meshes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import json
import os
from pathlib import Path
import random
import re
from typing import Iterable, Mapping, Sequence, TypeAlias


PathLike: TypeAlias = str | Path

_WORKSPACE = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _WORKSPACE / "configs" / "procgen.json"
DEFAULT_REGISTRY_PATH = _WORKSPACE / "output" / "asset_registry.json"
DEFAULT_PROFILES_PATH = _WORKSPACE / "output" / "asset_kit_profiles.json"

REGISTRY_SCHEMA_VERSION = 1
REGISTRY_TOOL_VERSION = "1.0"

# These are the province/asset namespaces observed in the inventory.  The
# smaller provinces are retained even though the profile output focuses on
# tr/sky/pc/oaab; silently folding them into ``other`` loses useful fallback
# information for settlement generation.
KNOWN_KITS = frozenset(
    {"tr", "sky", "pc", "va", "els", "sum", "oaab", "hf", "hr", "pi", "epos", "bm"}
)
PROFILE_KITS = ("tr", "sky", "pc", "oaab")

_INVENTORY_FOLDERS = frozenset(
    {
        "a",
        "b",
        "c",
        "cr",
        "d",
        "env",
        "f",
        "grass",
        "i",
        "l",
        "m",
        "n",
        "o",
        "r",
        "td",
        "tdg",
        "w",
        "x",
    }
)

CATEGORIES = (
    "exterior",
    "interior",
    "door",
    "terrain",
    "flora",
    "rocks",
    "clutter",
    "other",
)

# The first matching rule wins.  These are intentionally conservative: a
# folder with a documented semantic beats a filename guess, and ambiguous
# content goes to ``other``.
_FLORA_WORDS = (
    "flora",
    "grass",
    "tree",
    "sapling",
    "fern",
    "flower",
    "plant",
    "reed",
    "moss",
    "mold",
    "mushroom",
    "bush",
    "shrub",
    "weed",
    "root",
    "stalk",
    "anemone",
    "sugarcane",
    "oleander",
    "hibiscus",
    "cactus",
)
_ROCK_WORDS = (
    "rock",
    "rocks",
    "stone",
    "cliff",
    "boulder",
    "pebble",
    "mtridge",
    "mtpeak",
)
_SCATTER_WORDS = _FLORA_WORDS + _ROCK_WORDS + (
    "terr",
    "terrain",
    "natural",
    "rubble",
    "branch",
    "log",
)

_WORD_RE = re.compile(r"[^a-z0-9]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _key(value: str) -> str:
    """Return the case-folded TES3 path key used by queries."""

    return value.replace("/", "\\").strip("\\").casefold()


def _normalise_path(value: str | os.PathLike[str]) -> str:
    """Normalize a relative mesh path without changing filename case."""

    raw = os.fspath(value).replace("/", "\\").strip()
    parts = [part for part in raw.split("\\") if part not in {"", "."}]
    if parts and parts[0].casefold() == "meshes":
        parts = parts[1:]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"invalid relative asset path: {value!r}")
    return "\\".join(parts)


def _stem_tokens(filename: str) -> set[str]:
    return {token for token in _WORD_RE.split(filename.casefold()) if token}


def _has_word_hint(filename: str, words: Sequence[str]) -> bool:
    tokens = _stem_tokens(filename)
    lowered = filename.casefold()
    return any(word in tokens or word in lowered for word in words)


def _classify(folder: str, filename: str) -> tuple[str, bool, str]:
    """Classify one inventory folder and return category, construction flag, rule."""

    top = folder.casefold()
    if top == "b":
        # This is the critical inventory correction.  Keep it in ``other``
        # (the public category vocabulary) and mark it unavailable to all
        # construction palettes.
        return "other", False, "folder b is body parts; explicit construction exclusion"
    if top == "x":
        return "exterior", True, "folder x is exterior construction"
    if top == "i":
        return "interior", True, "folder i is interior shells and kit pieces"
    if top == "d":
        return "door", True, "folder d is doors"
    if top == "tdg":
        return "terrain", False, "folder tdg is shared terrain and entrances"
    if top == "grass":
        return "flora", False, "flat grass folder is shared flora"
    if top == "env":
        return "flora", False, "folder env is environment flora"

    # Furniture/misc folders contain a small but useful set of named plants
    # and rocks (for example VA_TerrXM_Rock_01).  Name classification is only
    # used in those documented mixed folders; unknown names remain clutter or
    # other instead of being guessed into construction.
    if top in {"f", "c", "o", "m"}:
        if _has_word_hint(filename, _FLORA_WORDS):
            return "flora", False, f"folder {top} with documented flora filename hint"
        if _has_word_hint(filename, _ROCK_WORDS):
            return "rocks", False, f"folder {top} with documented rock filename hint"
    if top in {"f", "c"}:
        return "clutter", False, f"folder {top} is furniture/clutter"
    return "other", False, f"folder {top or '<root>'} has no unambiguous inventory category"


@dataclass(frozen=True)
class Asset:
    """One opaque NIF file in the merged TES3 mesh namespace."""

    path: str
    size: int
    kit: str
    category: str
    source_root: str
    source_subfolder: str
    construction_eligible: bool
    classification_rule: str

    @property
    def relative_path(self) -> str:
        """Alias used by callers that call mesh paths ``relative_path``."""

        return self.path

    @property
    def rel_path(self) -> str:
        """Short alias for integrations that use ``rel_path`` terminology."""

        return self.path

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "relative_path": self.path,
            "size": self.size,
            "kit": self.kit,
            "category": self.category,
            "source_root": self.source_root,
            "source_subfolder": self.source_subfolder,
            "construction_eligible": self.construction_eligible,
            "classification_rule": self.classification_rule,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Asset":
        path = str(value.get("path", value.get("relative_path", "")))
        return cls(
            path=_normalise_path(path),
            size=int(value.get("size", 0)),
            kit=str(value.get("kit", "other")).casefold(),
            category=str(value.get("category", "other")),
            source_root=str(value.get("source_root", "")),
            source_subfolder=str(value.get("source_subfolder", "")),
            construction_eligible=bool(value.get("construction_eligible", False)),
            classification_rule=str(value.get("classification_rule", "cache entry")),
        )


@dataclass(frozen=True)
class MeshClassification:
    """Path-only classification for a mesh referenced by a TES3 record.

    Asset-registry entries have an explicit province directory (for example
    ``tr\\x\\...``), while vanilla and older TES3 records usually store only
    ``x\\...``.  The ESP scanner uses this small shared rule set with a
    ``source_kit`` fallback for the latter form; it never opens a NIF.
    """

    kit: str
    category: str
    source_folder: str
    construction_eligible: bool
    classification_rule: str


def classify_mesh_path(
    mesh_path: str | os.PathLike[str],
    *,
    source_kit: str | None = None,
    record_type: str | None = None,
) -> MeshClassification:
    """Classify one TES3 ``MODL`` path using inventory semantics.

    ``source_kit`` is used only when a legacy model path has no explicit
    province namespace.  The scanner passes ``vanilla``, ``tr``, ``sky`` or
    ``pc`` for the corresponding reference ESM.  Explicit namespaces always
    win, which matters for TR/SHOTN/PC records that reuse another province's
    assets.  The function is deliberately conservative for paths without an
    inventory folder.
    """

    raw = os.fspath(mesh_path).replace("/", "\\").strip()
    if not raw:
        return MeshClassification(
            kit="unknown",
            category="other",
            source_folder="",
            construction_eligible=False,
            classification_rule="empty MODL path",
        )
    try:
        normalized = _normalise_path(raw)
    except ValueError:
        return MeshClassification(
            kit="unknown",
            category="other",
            source_folder="",
            construction_eligible=False,
            classification_rule="invalid MODL path",
        )

    parts = normalized.split("\\")
    first = parts[0].casefold() if parts else ""
    supplied = str(source_kit or "").casefold()
    if first in KNOWN_KITS:
        kit = first
        folder = parts[1] if len(parts) > 1 else ""
        rule_prefix = "explicit province namespace"
    elif first in _INVENTORY_FOLDERS:
        kit = supplied if supplied in KNOWN_KITS else ("vanilla" if supplied == "vanilla" else "unknown")
        folder = parts[0]
        rule_prefix = "legacy folder path plus source ESM"
    else:
        kit = supplied if supplied in KNOWN_KITS else ("vanilla" if supplied == "vanilla" else "unknown")
        folder = ""
        rule_prefix = "source ESM fallback; no inventory folder"

    filename = parts[-1] if parts else ""
    category, construction, rule = _classify(folder, filename)
    # A TES3 DOOR record is semantically a door even when its model lives in
    # an older generic folder.  Likewise, CONT/LIGH/ACTI/FURN are clutter
    # unless the path itself proves construction geometry.
    if str(record_type or "").upper() == "DOOR":
        category, construction, rule = "door", True, "TES3 DOOR record"
    elif str(record_type or "").upper() in {"CONT", "LIGH", "ACTI", "FURN", "MISC"}:
        if category not in {"exterior", "interior", "door"}:
            category, construction = "clutter", False
            rule = f"TES3 {str(record_type).upper()} record; {rule}"

    return MeshClassification(
        kit=kit,
        category=category,
        source_folder=folder,
        construction_eligible=construction,
        classification_rule=f"{rule_prefix}; {rule}",
    )


@dataclass(frozen=True)
class _RootSpec:
    path: Path
    hint: str | None = None


def _infer_hint(path: Path, supplied: str | None) -> str | None:
    if supplied and supplied.casefold() in KNOWN_KITS:
        return supplied.casefold()
    text = str(path).casefold().replace("_", " ").replace("-", " ")
    if "oaab" in text:
        return "oaab"
    if "shotn" in text or "skyrim" in text:
        return "sky"
    if "projectcyrodil" in text or "projectcyrodiil" in text or "cyrodi" in text:
        return "pc"
    if "tamrielrebuild" in text or "\\tr\\" in text:
        return "tr"
    return supplied.casefold() if supplied else None


def _as_root_specs(
    roots: Iterable[PathLike] | Mapping[str, PathLike] | PathLike | None,
    config_path: PathLike,
) -> list[_RootSpec]:
    if roots is None:
        roots = configured_asset_roots(config_path)
    if isinstance(roots, Mapping):
        values = [_RootSpec(Path(value), str(name)) for name, value in roots.items()]
    elif isinstance(roots, (str, Path)):
        values = [_RootSpec(Path(roots))]
    else:
        values = [_RootSpec(Path(value)) for value in roots]
    return [
        _RootSpec(path=_resolved_path(item.path), hint=_infer_hint(item.path, item.hint))
        for item in values
    ]


def _resolved_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def configured_asset_roots(config_path: PathLike = _DEFAULT_CONFIG) -> list[Path]:
    """Load configured read-only asset roots, retaining missing paths.

    ``meshcheck.configured_data_roots`` has the same config contract.  This
    local loader is kept independent so the registry can also accept the
    project's future ``asset_roots`` alias without changing mesh validation.
    """

    config = Path(config_path)
    with config.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    paths = document.get("paths", {}) if isinstance(document, Mapping) else {}
    roots_value = paths.get("asset_roots") if isinstance(paths, Mapping) else None
    if roots_value is None and isinstance(paths, Mapping):
        roots_value = paths.get("data_roots")
    if roots_value is None and isinstance(document, Mapping):
        roots_value = document.get("data_roots")
    if isinstance(roots_value, Mapping):
        values = list(roots_value.values())
    elif isinstance(roots_value, list):
        values = roots_value
    else:
        values = []
    workspace = config.resolve().parents[1]
    result: list[Path] = []
    for value in values:
        path = Path(str(value))
        result.append(path if path.is_absolute() else workspace / path)
    return result


def _root_signature(spec: _RootSpec) -> dict[str, object]:
    try:
        stat = spec.path.stat()
    except OSError as exc:
        return {
            "path": str(spec.path),
            "hint": spec.hint or "",
            "exists": False,
            "mtime_ns": None,
            "error": str(exc),
        }
    if not spec.path.is_dir():
        return {
            "path": str(spec.path),
            "hint": spec.hint or "",
            "exists": False,
            "mtime_ns": stat.st_mtime_ns,
            "error": "asset root is not a directory",
        }
    return {
        "path": str(spec.path),
        "hint": spec.hint or "",
        "exists": True,
        "mtime_ns": stat.st_mtime_ns,
    }


def _mesh_roots(spec: _RootSpec) -> list[tuple[Path, str | None]]:
    """Find one or more ``Meshes`` trees without opening any NIF files."""

    root = spec.path
    if not root.is_dir():
        return []
    if root.name.casefold() == "meshes":
        return [(root, spec.hint)]
    direct = [child for child in root.iterdir() if child.is_dir() and child.name.casefold() == "meshes"]
    if direct:
        return [(child, spec.hint) for child in sorted(direct, key=lambda p: str(p).casefold())]

    # OAAB is distributed as a collection of numbered packages, each with its
    # own Meshes directory.  Discover those directories deterministically.
    discovered: list[Path] = []
    try:
        for directory, dirnames, _filenames in os.walk(root, followlinks=False):
            dirnames.sort(key=str.casefold)
            if Path(directory).name.casefold() == "meshes":
                discovered.append(Path(directory))
                dirnames[:] = []
    except OSError:
        return []
    return [(path, spec.hint) for path in sorted(discovered, key=lambda p: str(p).casefold())]


def _canonical_parts(relative: Path, hint: str | None) -> tuple[str, ...]:
    parts = list(relative.parts)
    if not parts:
        return ()
    first = parts[0].casefold()
    if first in KNOWN_KITS:
        parts[0] = first
    elif hint in KNOWN_KITS and first in _INVENTORY_FOLDERS:
        parts.insert(0, hint)
    return tuple(parts)


def _scan_root(spec: _RootSpec, entries_by_key: dict[str, Asset]) -> tuple[int, str | None]:
    """Scan a root and merge unique TES3 paths into ``entries_by_key``."""

    if not spec.path.is_dir():
        return 0, "asset root is not a directory"
    physical_count = 0
    try:
        mesh_roots = _mesh_roots(spec)
        for mesh_root, mesh_hint in mesh_roots:
            try:
                mesh_root_relative = mesh_root.relative_to(spec.path)
            except ValueError:
                mesh_root_relative = Path(mesh_root.name)
            # If a source root is a direct ``Meshes`` directory, its basename
            # is deliberately not part of a TES3 path.
            del mesh_root_relative
            for directory, dirnames, filenames in os.walk(mesh_root, followlinks=False):
                dirnames.sort(key=str.casefold)
                for filename in sorted(filenames, key=str.casefold):
                    if not filename.casefold().endswith(".nif"):
                        continue
                    file_path = Path(directory) / filename
                    try:
                        size = file_path.stat().st_size
                        relative = Path(os.path.relpath(file_path, mesh_root))
                    except OSError:
                        # A file disappearing during a read-only walk should
                        # not make the complete catalog fail.
                        continue
                    parts = _canonical_parts(relative, mesh_hint)
                    if len(parts) < 2:
                        # A flat NIF below Meshes is legal, but it is not
                        # possible to infer a province/category from it.
                        parts = tuple(parts)
                    path = _normalise_path("\\".join(parts))
                    key = _key(path)
                    physical_count += 1
                    if key in entries_by_key:
                        continue
                    has_kit = bool(parts) and parts[0].casefold() in KNOWN_KITS
                    folder_index = 1 if has_kit else 0
                    folder = parts[folder_index] if len(parts) > folder_index else ""
                    category, construction, rule = _classify(folder, filename)
                    kit = parts[0].casefold() if has_kit else "other"
                    entries_by_key[key] = Asset(
                        path=path,
                        size=size,
                        kit=kit,
                        category=category,
                        source_root=str(spec.path),
                        source_subfolder=folder,
                        construction_eligible=construction,
                        classification_rule=rule,
                    )
    except OSError as exc:
        return physical_count, str(exc)
    return physical_count, None


def _cache_roots(document: Mapping[str, object]) -> object:
    cache = document.get("cache")
    if isinstance(cache, Mapping):
        return cache.get("roots")
    return None


class Registry:
    """Immutable-in-practice catalog with indexed path/kit/category queries."""

    def __init__(
        self,
        entries: Iterable[Asset],
        *,
        roots: Iterable[Mapping[str, object]] = (),
        meta: Mapping[str, object] | None = None,
        cache_hit: bool = False,
    ) -> None:
        self.entries = tuple(entries)
        self.roots = tuple(dict(root) for root in roots)
        self.meta = dict(meta or {})
        self.cache_hit = cache_hit
        self._path_index = {_key(asset.path): asset for asset in self.entries}
        self._kit_index: dict[str, list[Asset]] = {}
        self._category_index: dict[str, list[Asset]] = {}
        for asset in self.entries:
            self._kit_index.setdefault(asset.kit.casefold(), []).append(asset)
            self._category_index.setdefault(asset.category.casefold(), []).append(asset)

    @property
    def assets(self) -> tuple[Asset, ...]:
        return self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def by_kit(self, kit: str) -> list[Asset]:
        return list(self._kit_index.get(str(kit).casefold(), ()))

    def by_category(self, category: str) -> list[Asset]:
        return list(self._category_index.get(str(category).casefold(), ()))

    def find(self, pattern: str) -> list[Asset]:
        """Find paths by case-insensitive glob, or substring without metacharacters."""

        query = _key(str(pattern))
        has_glob = any(char in query for char in "*?[")
        if has_glob:
            return [asset for asset in self.entries if fnmatch.fnmatchcase(_key(asset.path), query)]
        return [asset for asset in self.entries if query in _key(asset.path)]

    def exists(self, rel_path: str) -> bool:
        try:
            return _key(_normalise_path(rel_path)) in self._path_index
        except ValueError:
            return False

    def stats(self) -> dict[str, object]:
        kits = Counter(asset.kit for asset in self.entries)
        categories = Counter(asset.category for asset in self.entries)
        return {
            "total": len(self.entries),
            "total_nifs": len(self.entries),
            "by_kit": dict(sorted(kits.items())),
            "by_category": dict(sorted(categories.items())),
            "construction_eligible": sum(asset.construction_eligible for asset in self.entries),
            "missing_roots": [
                root.get("path")
                for root in self.roots
                if not bool(root.get("exists", False))
            ],
        }

    def to_document(self) -> dict[str, object]:
        stats = self.stats()
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "meta": self.meta,
            "roots": list(self.roots),
            "stats": stats,
            "classification_rules": list(CLASSIFICATION_RULES),
            "entries": [asset.to_dict() for asset in self.entries],
            "cache": {"roots": list(self.meta.get("cache_roots", ()))},
        }

    def save(self, path: PathLike = DEFAULT_REGISTRY_PATH) -> Path:
        destination = Path(path)
        _write_json(destination, self.to_document())
        return destination

    @classmethod
    def from_document(cls, document: Mapping[str, object], *, cache_hit: bool = False) -> "Registry":
        raw_entries = document.get("entries", [])
        entries = [Asset.from_dict(value) for value in raw_entries if isinstance(value, Mapping)]
        raw_roots = document.get("roots", [])
        roots = [value for value in raw_roots if isinstance(value, Mapping)]
        meta = document.get("meta") if isinstance(document.get("meta"), Mapping) else {}
        return cls(entries, roots=roots, meta=meta, cache_hit=cache_hit)

    @classmethod
    def load(
        cls,
        path: PathLike = DEFAULT_REGISTRY_PATH,
        *,
        roots: Iterable[PathLike] | Mapping[str, PathLike] | PathLike | None = None,
        config_path: PathLike = _DEFAULT_CONFIG,
    ) -> "Registry":
        """Load a current cached registry, rebuilding it when roots are stale."""

        return load_registry(path, roots=roots, config_path=config_path)


CLASSIFICATION_RULES = (
    "x -> exterior; x is the inventory's exterior construction folder",
    "i -> interior; i contains room shells and modular interior pieces",
    "d -> door; d contains exterior and interior doors",
    "tdg -> terrain; tdg is shared terrain and entrance content",
    "grass/env -> flora",
    "mixed f/c/o/m folders use explicit flora/rock filename hints only; otherwise clutter/other",
    "b -> other with construction_eligible=false; b is body parts, never buildings",
    "all ambiguous folders and names -> other; no NIF geometry parsing or guessing",
)


def _same_root_signature(left: object, right: object) -> bool:
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    # JSON dictionaries are ordered in practice but equality is semantic.  Do
    # not include a previous scan's error text in the cache key: a transient
    # missing root should still become valid when its mtime/existence changes.
    def compact(value: object) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for item in value:
            if isinstance(item, Mapping):
                result.append(
                    {
                        "path": item.get("path"),
                        "hint": item.get("hint", ""),
                        "exists": item.get("exists", False),
                        "mtime_ns": item.get("mtime_ns"),
                    }
                )
        return result

    return compact(left) == compact(right)


def _load_cached(path: Path, signatures: list[dict[str, object]]) -> Registry | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, Mapping):
        return None
    if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        return None
    if not _same_root_signature(_cache_roots(document), signatures):
        return None
    return Registry.from_document(document, cache_hit=True)


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def build_registry(
    roots: Iterable[PathLike] | Mapping[str, PathLike] | PathLike | None = None,
    *,
    cache_path: PathLike = DEFAULT_REGISTRY_PATH,
    use_cache: bool = True,
    config_path: PathLike = _DEFAULT_CONFIG,
) -> Registry:
    """Build or reload the catalog for configured read-only roots.

    Missing roots are represented in ``Registry.roots`` and contribute no
    entries.  A cache hit checks only the configured root path, hint,
    existence, and directory mtime; this keeps the fast path a small JSON
    read while matching the task's explicit root-mtime cache contract.
    """

    specs = _as_root_specs(roots, config_path)
    signatures = [_root_signature(spec) for spec in specs]
    cache = Path(cache_path)
    if use_cache and cache.is_file():
        cached = _load_cached(cache, signatures)
        if cached is not None:
            return cached

    entries_by_key: dict[str, Asset] = {}
    root_records: list[dict[str, object]] = []
    for spec, signature in zip(specs, signatures):
        record = dict(signature)
        if not bool(signature.get("exists", False)):
            record["nif_count"] = 0
            record["error"] = signature.get("error", "asset root is not a directory")
        else:
            count, error = _scan_root(spec, entries_by_key)
            record["nif_count"] = count
            if error:
                record["error"] = error
        root_records.append(record)

    entries = sorted(entries_by_key.values(), key=lambda asset: _key(asset.path))
    meta: dict[str, object] = {
        "tool": "procgen.assets",
        "version": REGISTRY_TOOL_VERSION,
        "generated_utc": _utc_now(),
        "provenance": {
            "tool": "procgen.assets",
            "version": REGISTRY_TOOL_VERSION,
            "generated_utc": _utc_now(),
            "inputs": {
                str(index): {
                    "path": str(signature.get("path", "")),
                    "exists": bool(signature.get("exists", False)),
                    "mtime_ns": signature.get("mtime_ns"),
                }
                for index, signature in enumerate(signatures)
            },
        },
        "cache_roots": signatures,
    }
    registry = Registry(entries, roots=root_records, meta=meta)
    registry.save(cache)
    return registry


def _palette_candidate(asset: Asset) -> bool:
    if not asset.construction_eligible and asset.category == "other":
        # This includes b/body parts and every unknown bucket.
        return False
    if asset.category in {"flora", "rocks", "terrain"}:
        return True
    if asset.category == "exterior":
        filename = asset.path.rsplit("\\", 1)[-1]
        return _has_word_hint(filename, _SCATTER_WORDS)
    return False


def build_kit_profiles(
    registry: Registry,
    *,
    kits: Sequence[str] = PROFILE_KITS,
    palette_size: int = 64,
    seed: int = 20260801,
) -> dict[str, object]:
    """Build deterministic kit summaries and scatter starter palettes."""

    if palette_size < 0:
        raise ValueError("palette_size must be non-negative")
    profiles: dict[str, object] = {}
    for offset, kit_value in enumerate(kits):
        kit = str(kit_value).casefold()
        assets = registry.by_kit(kit)
        categories = Counter(asset.category for asset in assets)
        subfolders = Counter(asset.source_subfolder for asset in assets)
        candidates = [asset for asset in assets if _palette_candidate(asset)]
        # Sampling from a sorted list and a local seeded RNG makes output
        # stable across process runs and independent of query order.
        candidates.sort(key=lambda asset: _key(asset.path))
        rng = random.Random(seed + offset * 1009 + sum(ord(char) for char in kit))
        if len(candidates) > palette_size:
            candidates = rng.sample(candidates, palette_size)
            candidates.sort(key=lambda asset: _key(asset.path))
        palette = [asset.path for asset in candidates]
        profiles[kit] = {
            "asset_count": len(assets),
            "counts_by_category": {
                category: categories.get(category, 0) for category in CATEGORIES
            },
            "top_level_subfolders": dict(sorted(subfolders.items())),
            "starter_palette": palette,
            "starter_palette_size": len(palette),
            "palette_rule": (
                "flora/rocks/terrain plus filename-confirmed terrain-like exterior NIFs; "
                "body parts and ambiguous assets excluded"
            ),
        }
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "meta": {
            "tool": "procgen.assets",
            "version": REGISTRY_TOOL_VERSION,
            "generated_utc": _utc_now(),
            "registry_total": len(registry.entries),
            "seed": seed,
            "palette_size_limit": palette_size,
        },
        "kits": profiles,
    }


def write_kit_profiles(
    registry: Registry,
    path: PathLike = DEFAULT_PROFILES_PATH,
    *,
    kits: Sequence[str] = PROFILE_KITS,
    palette_size: int = 64,
    seed: int = 20260801,
) -> Path:
    document = build_kit_profiles(
        registry, kits=kits, palette_size=palette_size, seed=seed
    )
    destination = Path(path)
    _write_json(destination, document)
    return destination


def load_registry(
    path: PathLike = DEFAULT_REGISTRY_PATH,
    *,
    roots: Iterable[PathLike] | Mapping[str, PathLike] | PathLike | None = None,
    config_path: PathLike = _DEFAULT_CONFIG,
) -> Registry:
    """Load a registry and reject stale cache metadata by rebuilding it."""

    if roots is None:
        roots = configured_asset_roots(config_path)
    specs = _as_root_specs(roots, config_path)
    signatures = [_root_signature(spec) for spec in specs]
    destination = Path(path)
    loaded = _load_cached(destination, signatures)
    if loaded is not None:
        return loaded
    return build_registry(
        roots,
        cache_path=destination,
        use_cache=False,
        config_path=config_path,
    )


__all__ = [
    "Asset",
    "CATEGORIES",
    "DEFAULT_PROFILES_PATH",
    "DEFAULT_REGISTRY_PATH",
    "KNOWN_KITS",
    "MeshClassification",
    "PROFILE_KITS",
    "Registry",
    "build_kit_profiles",
    "build_registry",
    "classify_mesh_path",
    "configured_asset_roots",
    "load_registry",
    "write_kit_profiles",
]
