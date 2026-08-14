"""Read-only mesh and texture path checks for tes3conv JSON documents.

TES3 paths are relative to ``Meshes`` or ``Textures`` and are commonly written
with Windows separators.  The checker indexes configured data roots using a
case-folded, separator-normalized key, so it catches missing or incorrectly
cased assets without ever opening or modifying a plugin or source mod.

``AssetResolver`` is the public prepared resolver: it walks every configured
root exactly once and resolves any number of assets with pure dictionary
lookups.  ``check_refs`` and the single-shot ``resolve_asset`` keep their
legacy signatures; bulk same-process loops should construct one
``AssetResolver`` and reuse it (``karthgad_rebuild_inventory.build_model_rows``
and ``blender_flat_render.load_procgen_meshcheck`` do this).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .tes3json import Issue, PluginDoc


_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "procgen.json"


def _key(value: str) -> str:
    return value.replace("/", "\\").strip("\\").casefold()


def _relative_asset(value: str, kind: str) -> str | None:
    """Normalize a relative TES3 asset path, rejecting absolute/traversal paths."""

    normalized = value.replace("/", "\\").strip()
    if not normalized or "\x00" in normalized:
        return None
    if normalized.startswith("\\") or (len(normalized) >= 2 and normalized[1] == ":"):
        return None
    parts = [part for part in normalized.split("\\") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        return None
    if parts and parts[0].casefold() in {"meshes", "textures"}:
        parts = parts[1:]
    if not parts:
        return None
    return _key("\\".join(parts))


def _relative_asset_candidates(value: str, kind: str) -> tuple[str, ...]:
    """Return safe lookup names, including the vanilla TGA-to-DDS bridge.

    Morrowind's original ``tamriel.esm`` stores several LTEX filenames as
    ``.tga``.  The read-only extracted BSA used by this workspace contains the
    same vanilla assets as ``.dds``.  TES3 lookup is extension-flexible in the
    target runtime, so the renderer/checker must model that compatibility
    rather than silently creating a fallback material.  The declared name is
    always tried first; no arbitrary basename substitution is performed.
    """

    relative = _relative_asset(value, kind)
    if relative is None:
        return ()
    candidates = [relative]
    if kind == "texture" and relative.casefold().endswith(".tga"):
        candidates.append(relative[:-4] + ".dds")
    return tuple(candidates)


def configured_data_roots(config_path: str | Path = _DEFAULT_CONFIG) -> list[Path]:
    """Load read-only data roots from the project config.

    Both the current ``paths.data_roots`` list and a top-level ``data_roots``
    list are accepted to keep config migration harmless.  Relative roots are
    resolved against the workspace (the directory containing ``configs``).
    Missing roots are retained in the result.  ``check_refs`` reports a data
    root problem when all configured roots are unavailable, while allowing
    optional TR/SHOTN/PC roots to be absent when another root is usable.
    """

    config = Path(config_path)
    with config.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    paths = document.get("paths", {}) if isinstance(document, Mapping) else {}
    roots_value = paths.get("data_roots") if isinstance(paths, Mapping) else None
    if roots_value is None and isinstance(document, Mapping):
        roots_value = document.get("data_roots")
    roots: list[str] = []
    if isinstance(roots_value, Mapping):
        roots.extend(str(value) for value in roots_value.values())
    elif isinstance(roots_value, list):
        roots.extend(str(value) for value in roots_value)
    workspace = config.resolve().parents[1]
    result: list[Path] = []
    for value in roots:
        path = Path(value)
        result.append(path if path.is_absolute() else workspace / path)
    return result


@dataclass
class _RootIndex:
    root: Path
    meshes: set[str]
    textures: set[str]
    error: str | None = None
    mesh_files: dict[str, Path] | None = None
    texture_files: dict[str, Path] | None = None

    @classmethod
    def build(cls, root: Path) -> "_RootIndex":
        meshes: set[str] = set()
        textures: set[str] = set()
        mesh_files: dict[str, Path] = {}
        texture_files: dict[str, Path] = {}
        resolved = root.expanduser()
        if not resolved.is_dir():
            return cls(resolved, meshes, textures, "data root is not a directory")
        try:
            for directory, _subdirectories, filenames in os.walk(resolved, followlinks=False):
                directory_path = Path(directory)
                try:
                    relative = directory_path.relative_to(resolved)
                except ValueError:
                    continue
                relative_parts = [part.casefold() for part in relative.parts]
                category: str | None = None
                category_parts: list[str] = []
                if relative_parts and relative_parts[0] in {"meshes", "textures"}:
                    category = relative_parts[0]
                    category_parts = relative_parts[1:]
                elif resolved.name.casefold() in {"meshes", "textures"}:
                    category = resolved.name.casefold()
                    category_parts = relative_parts
                if category is None:
                    continue
                destination = meshes if category == "meshes" else textures
                destination_files = mesh_files if category == "meshes" else texture_files
                prefix = "\\".join(category_parts)
                for filename in filenames:
                    asset = "\\".join(part for part in (prefix, filename) if part)
                    asset_key = _key(asset)
                    destination.add(asset_key)
                    destination_files.setdefault(asset_key, directory_path / filename)
        except OSError as exc:
            return cls(resolved, meshes, textures, str(exc))
        return cls(resolved, meshes, textures, None, mesh_files, texture_files)

    def contains(self, kind: str, relative: str) -> bool:
        values = self.meshes if kind == "mesh" else self.textures
        return _key(relative) in values

    def resolve(self, kind: str, relative: str) -> Path | None:
        values = self.mesh_files if kind == "mesh" else self.texture_files
        return values.get(_key(relative)) if values is not None else None


def _as_roots(roots: Iterable[str | Path] | Mapping[str, str | Path] | None) -> list[Path]:
    if roots is None:
        return configured_data_roots()
    if isinstance(roots, Mapping):
        return [Path(value) for value in roots.values()]
    return [Path(value) for value in roots]


def _roots_tuple(
    roots: Iterable[str | Path] | Mapping[str, str | Path] | None,
) -> tuple[Path, ...]:
    """Return the ordered, typed root spelling used for resolver identity."""
    return tuple(_as_roots(roots))


class AssetResolver:
    """Immutable, process-local asset resolver for a fixed root order.

    Construction walks every configured data root exactly once and builds the
    ordered first-match index.  ``resolve`` then performs pure dictionary
    lookups, so a hot loop that resolves many assets against the same roots
    must construct one resolver instead of calling the module-level
    ``resolve_asset`` (which rebuilds every root index per call).

    Semantics are identical to ``resolve_asset``:

    - Roots are tried in configured order; the first root whose filesystem
      contains the asset wins (ordered first-match).
    - Lookup is case-insensitive and separator-normalized.
    - Texture lookup keeps the vanilla TGA-to-DDS bridge: a declared
      ``.tga`` is tried first, then the same basename with ``.dds``.
    - A root that is missing or fails to walk is recorded in ``root_indexes``
      with an ``error`` and is skipped, exactly like the legacy per-call
      builder; other roots remain usable.

    Instances are process-local and stateless after construction: there is no
    mutable global cache authority, so concurrent callers with different root
    orders cannot interfere with each other.

    The instance is also callable with the legacy ``resolve_asset`` signature
    ``(value, kind, roots=None)``.  When ``roots`` is omitted or equals the
    prepared root order, the prepared indexes are used (no rescan); any other
    root order falls back to the module-level ``resolve_asset`` so callers
    that pass a different order keep exact legacy behavior.
    """

    def __init__(
        self,
        roots: Iterable[str | Path] | Mapping[str, str | Path] | None = None,
        *,
        config_path: str | Path = _DEFAULT_CONFIG,
    ) -> None:
        if roots is None:
            self._roots: tuple[Path, ...] = tuple(configured_data_roots(config_path))
        else:
            self._roots = _roots_tuple(roots)
        # Build every root index exactly once, in configured order.  A failed
        # root carries its error inside the index and is skipped at resolve
        # time (legacy behavior), never fatal to the whole resolver.
        self._indexes: tuple[_RootIndex, ...] = tuple(
            _RootIndex.build(path) for path in self._roots
        )

    @property
    def roots(self) -> tuple[Path, ...]:
        """The immutable root order this resolver was built for."""
        return self._roots

    @property
    def root_indexes(self) -> tuple[_RootIndex, ...]:
        """The per-root indexes, one per configured root, in root order.

        An index whose ``error`` is not ``None`` represents an unavailable
        root (missing directory or walk failure) and is skipped during
        resolution.  Exposed for inspection and tests; the index type itself
        is private.
        """
        return self._indexes

    def resolve(self, value: str, kind: str) -> Path | None:
        """Return the first match for one relative asset path, or ``None``."""
        relatives = _relative_asset_candidates(value, kind)
        if not relatives:
            return None
        for index in self._indexes:
            if index.error:
                continue
            for relative in relatives:
                resolved = index.resolve(kind, relative)
                if resolved is not None:
                    return resolved
        return None

    def __call__(
        self,
        value: str,
        kind: str,
        roots: Iterable[str | Path] | Mapping[str, str | Path] | None = None,
    ) -> Path | None:
        """Legacy-compatible callable bound to this prepared resolver.

        ``resolve_asset(value, kind, roots=roots)`` and
        ``resolver(value, kind, roots=roots)`` return identical paths.  The
        prepared indexes are used when ``roots`` is omitted or matches the
        prepared order (common case: callers pass back the exact tuple they
        received); a different root order delegates to the module-level
        function so its per-call rebuild semantics are preserved exactly.
        """
        if roots is None:
            return self.resolve(value, kind)
        try:
            matches = _roots_tuple(roots) == self._roots
        except TypeError:  # pragma: no cover - non-path iterable element
            matches = False
        if matches:
            return self.resolve(value, kind)
        return resolve_asset(value, kind, roots=roots)


def check_refs(
    doc: PluginDoc,
    roots: Iterable[str | Path] | Mapping[str, str | Path] | None = None,
    *,
    config_path: str | Path = _DEFAULT_CONFIG,
) -> list[Issue]:
    """Check every non-empty MODL and texture path in a plugin document.

    ``roots`` is primarily for tests and isolated generation runs.  When it is
    omitted, configured roots are read from ``configs/procgen.json``.  The
    returned ``Issue`` objects use the same shape as the structural validator.
    """

    root_paths = _as_roots(roots) if roots is not None else configured_data_roots(config_path)
    indexes = [_RootIndex.build(path) for path in root_paths]
    issues: list[Issue] = []
    if not indexes:
        return [Issue("document", "no read-only data roots are configured", "no-roots")]
    # Optional TR/SHOTN/PC roots may not be installed on every generation
    # machine.  They are still indexed when present, but an absent optional
    # root must not make a valid Tamriel_Data asset fail.  If *all* configured
    # roots are absent, report that installation problem below.
    if indexes and all(index.error for index in indexes):
        issues.extend(
            Issue("data_roots", f"{index.root}: {index.error}", "data-root")
            for index in indexes
        )

    def check(path: str, value: object, kind: str) -> None:
        if not isinstance(value, str):
            issues.append(Issue(path, "asset path must be a string", "type"))
            return
        relatives = _relative_asset_candidates(value, kind)
        if not relatives:
            issues.append(Issue(path, "asset path is empty, absolute, or traverses its data root", "asset-path"))
            return
        if not any(index.contains(kind, relative) for index in indexes for relative in relatives):
            category = "mesh" if kind == "mesh" else "texture"
            issues.append(Issue(path, f"{category} does not exist under configured data roots: {value!r}", "missing-asset"))

    if not isinstance(doc, list):
        return [Issue("document", "must be a top-level JSON array", "type")]
    for record_index, record in enumerate(doc):
        if not isinstance(record, Mapping):
            continue
        record_path = f"records[{record_index}]"
        record_type = record.get("type")
        if "mesh" in record:
            mesh = record.get("mesh")
            # NPC records may intentionally carry an empty MODL and use body
            # parts instead.  An empty MODL on a renderable object is exactly
            # the PTR bug class this gate is intended to expose.
            if mesh == "" and record_type != "Npc":
                issues.append(Issue(f"{record_path}.mesh", "renderable record has an empty MODL path", "empty-mesh"))
            elif mesh != "":
                check(f"{record_path}.mesh", mesh, "mesh")
        if record_type == "LandscapeTexture" and "file_name" in record:
            check(f"{record_path}.file_name", record.get("file_name"), "texture")
        # Future record builders may expose a direct texture path.  Do not
        # mistake LAND's compressed texture_indices field for a filesystem
        # path; only string values are asset paths.
        for field, value in record.items():
            if field.casefold() in {"texture", "texture_path", "texture_file"} and isinstance(value, str):
                check(f"{record_path}.{field}", value, "texture")
    return issues


def resolve_asset(
    value: str,
    kind: str,
    roots: Iterable[str | Path] | Mapping[str, str | Path] | None = None,
) -> Path | None:
    """Return the first case-insensitive match for one relative asset path.

    Backward-compatible single-shot helper: it constructs a fresh
    ``AssetResolver`` (which walks every configured data root once) and
    resolves through it.  Bulk loops that resolve many assets against the
    same roots should construct one ``AssetResolver`` and call
    ``resolver.resolve`` instead, which performs no rescans.
    """

    return AssetResolver(roots=roots).resolve(value, kind)


__all__ = ["AssetResolver", "check_refs", "configured_data_roots", "resolve_asset"]
