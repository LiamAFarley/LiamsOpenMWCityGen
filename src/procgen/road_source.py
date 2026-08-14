"""Read the authoritative XCF road layer and preserve its effective mask.

Pipeline position
------------------
This module is the source-extraction stage of the road-centerline pipeline::

    read-only XCF + optional VTEX BMP/palette
        -> pinned ``road network.png`` metadata
        -> immutable effective-alpha mask
        -> source evidence for repair, graphing, rendering, and audit

The production reader deliberately uses the installed :mod:`gimpformats`
package rather than the rejected temporary interleaved-RLE decoder.  The
installed package predates one property id present in this v011 file, so a
small scoped compatibility context extends its enum and skips *only* the
known, size-framed property 42.  The context is restored before returning;
the installed package is never modified on disk.

Inputs are read-only.  Outputs are NumPy arrays and metadata dictionaries;
callers own artifact paths.  The source array is canvas-oriented (row 0 is
north, matching the corrected investigation evidence), contains the exact
effective alpha values after the layer mask and layer offset are applied, and
is never repaired in this module.  The pinned contract rejects a different
canvas, layer, mask, offset, opacity, mode, channel count, paint colour, or
known occupancy.  This fail-closed behavior prevents a format or source
change from silently becoming a different road network.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import enum
import hashlib
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
from PIL import Image


CANVAS_WIDTH = 4992
CANVAS_HEIGHT = 3040
ROAD_LAYER_NAME = "road network.png"
ROAD_LAYER_OFFSET = (-8, 0)
ROAD_LAYER_SIZE = (4992, 3040)
ROAD_LAYER_PAINT_RGBA = (0, 8, 112, 255)
EXPECTED_LAYER_PAINT_PIXELS = 477_009
EXPECTED_SOURCE_OCCUPANCY = 399_600
EXPECTED_XCF_VERSION = 11
EXPECTED_BASE_COLOR_MODE = 0  # RGB in the XCF's GIMP enum.
EXPECTED_PRECISION_BITS = 16
EXPECTED_PRECISION_GAMMA = True
EXPECTED_PRECISION_INTEGER = True
KNOWN_NEWER_PROPERTY_ID = 42


def sha256_bytes(data: bytes) -> str:
    """Return the lower-case SHA-256 digest for ``data``."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a read-only source file without changing it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RoadSource:
    """Immutable source evidence returned by :func:`extract_road_source`.

    ``effective_alpha`` is ``uint8`` and remains the exact mask-derived alpha
    on the XCF canvas.  ``binary_mask`` is a separate ``uint8`` 0/1 view used
    by topology code; neither array is modified by repair functions.
    """

    effective_alpha: np.ndarray
    binary_mask: np.ndarray
    metadata: dict[str, Any]

    @property
    def width(self) -> int:
        """Canvas width in source pixels."""

        return int(self.effective_alpha.shape[1])

    @property
    def height(self) -> int:
        """Canvas height in source pixels."""

        return int(self.effective_alpha.shape[0])


@contextmanager
def _gimpformats_compatibility() -> Iterator[None]:
    """Temporarily make the installed reader tolerate XCF property 42.

    ``gimpformats`` consumes property payloads before dispatching them, so an
    unknown property can be skipped safely only after its framed bytes have
    been handed to ``_propertyDecode``.  The library's enum lookup otherwise
    raises before reaching its unknown-property branch.  We extend the enum
    in memory and replace the branch with a strict wrapper that accepts only
    the pinned property id.  Any other unknown id still fails closed.
    """

    try:
        import gimpformats.GimpIOBase as giob
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific
        raise RuntimeError("gimpformats is required for production XCF extraction") from exc

    original_properties = giob.ImageProperties
    original_prop_cmp = giob._prop_cmp
    original_decode = giob.GimpIOBase._propertyDecode

    members = [(member.name, member.value) for member in original_properties]
    existing_values = {value for _name, value in members}
    for value in range(max(existing_values, default=-1) + 1, KNOWN_NEWER_PROPERTY_ID + 1):
        members.append((f"PROCGEN_UNKNOWN_{value}", value))
    # Enum member names must be unique.  The source package currently ends at
    # PROP_NUM_PROPS=40, but this also handles a future package that already
    # knows some of the newer values.
    extended_properties = enum.Enum("ImageProperties", members)
    giob.ImageProperties = extended_properties

    def safe_prop_cmp(value: int, prop: Any) -> bool:
        if isinstance(prop, list):
            return any(safe_prop_cmp(value, item) for item in prop)
        all_members = list(giob.ImageProperties)
        if value < 0 or value >= len(all_members):
            return False
        return all_members[value] == prop

    def patched_property_decode(instance: Any, prop: int, payload: bytearray) -> int:
        try:
            return original_decode(instance, prop, payload)
        except RuntimeError as exc:
            if prop == KNOWN_NEWER_PROPERTY_ID and str(exc) == (
                f"Unknown property id {KNOWN_NEWER_PROPERTY_ID}"
            ):
                # _propertiesDecode has already consumed this property's
                # declared payload, so returning zero is the correct no-op.
                return 0
            raise

    giob._prop_cmp = safe_prop_cmp
    giob.GimpIOBase._propertyDecode = patched_property_decode
    try:
        yield
    finally:
        giob.ImageProperties = original_properties
        giob._prop_cmp = original_prop_cmp
        giob.GimpIOBase._propertyDecode = original_decode


def _opacity_is_opaque(value: Any) -> bool:
    """Accept the two representations used by gimpformats for opacity 255."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return abs(numeric - 1.0) < 1e-6 or abs(numeric - 255.0) < 1e-6


def _blend_mode_is_normal(value: Any) -> bool:
    """Return whether a gimpformats blend-mode value is ordinary Normal."""

    mode_value = getattr(value, "value", value)
    return str(mode_value).strip().lower() in {"normal", "normal (legacy)"}


def _precision_metadata(document: Any) -> dict[str, Any]:
    """Normalize gimpformats' precision object for contract checking."""

    precision = getattr(document, "precision", None)
    return {
        "bits": getattr(precision, "bits", None),
        "gamma": getattr(precision, "gamma", None),
        "integer": getattr(precision, "numberFormat", None) is int,
    }


def _require(condition: bool, message: str) -> None:
    """Raise a descriptive source-contract failure when ``condition`` is false."""

    if not condition:
        raise ValueError(f"XCF road-source contract failure: {message}")


def _place_layer_canvas(layer_array: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    """Place a layer-sized 2-D array onto the pinned canvas with clipping.

    XCF offsets are layer-origin coordinates in canvas space.  For the road
    layer ``(-8, 0)`` means layer column 8 is canvas column 0; the eight right
    columns therefore clip outside the canvas and remain zero.
    """

    layer = np.asarray(layer_array)
    _require(layer.ndim == 2, f"expected a 2-D mask array, got shape {layer.shape}")
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=layer.dtype)
    offset_x, offset_y = offset
    canvas_x0 = max(0, offset_x)
    canvas_y0 = max(0, offset_y)
    layer_x0 = max(0, -offset_x)
    layer_y0 = max(0, -offset_y)
    width = min(CANVAS_WIDTH - canvas_x0, layer.shape[1] - layer_x0)
    height = min(CANVAS_HEIGHT - canvas_y0, layer.shape[0] - layer_y0)
    _require(width >= 0 and height >= 0, "layer offset does not overlap the canvas")
    if width and height:
        canvas[canvas_y0 : canvas_y0 + height, canvas_x0 : canvas_x0 + width] = layer[
            layer_y0 : layer_y0 + height, layer_x0 : layer_x0 + width
        ]
    return canvas


def _load_document_and_extract(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the named layer through gimpformats and return alpha/mask arrays."""

    with _gimpformats_compatibility():
        from gimpformats.gimpXcfDocument import GimpDocument

        document = GimpDocument(str(path))
        _require(document.version == EXPECTED_XCF_VERSION, f"version {document.version} != 11")
        _require(
            (document.width, document.height) == (CANVAS_WIDTH, CANVAS_HEIGHT),
            f"canvas {(document.width, document.height)} != {(CANVAS_WIDTH, CANVAS_HEIGHT)}",
        )
        _require(
            document.baseColorMode == EXPECTED_BASE_COLOR_MODE,
            f"base color mode {document.baseColorMode} != {EXPECTED_BASE_COLOR_MODE}",
        )
        precision = _precision_metadata(document)
        _require(precision["bits"] == EXPECTED_PRECISION_BITS, f"precision {precision}")
        _require(precision["gamma"] is EXPECTED_PRECISION_GAMMA, f"precision gamma {precision}")
        _require(precision["integer"] is EXPECTED_PRECISION_INTEGER, f"precision format {precision}")

        layers = [layer for layer in document.raw_layers if str(layer.name) == ROAD_LAYER_NAME]
        _require(len(layers) == 1, f"found {len(layers)} exact '{ROAD_LAYER_NAME}' layers")
        layer = layers[0]
        _require(
            (layer.width, layer.height) == ROAD_LAYER_SIZE,
            f"road layer size {(layer.width, layer.height)} != {ROAD_LAYER_SIZE}",
        )
        _require((layer.xOffset, layer.yOffset) == ROAD_LAYER_OFFSET, f"offset {(layer.xOffset, layer.yOffset)}")
        _require(bool(layer.visible), "road layer is not visible")
        _require(_opacity_is_opaque(layer.opacity), f"road layer opacity {layer.opacity!r} is not opaque")
        _require(_blend_mode_is_normal(layer.blendMode), f"road layer mode {layer.blendMode!r} is not Normal")
        _require(bool(layer.applyMask), "road layer does not apply its layer mask")
        _require(layer.mask is not None, "road layer has no layer mask")
        _require(
            (layer.mask.width, layer.mask.height) == ROAD_LAYER_SIZE,
            f"mask size {(layer.mask.width, layer.mask.height)} != {ROAD_LAYER_SIZE}",
        )
        _require(
            (getattr(layer.mask, "xOffset", 0), getattr(layer.mask, "yOffset", 0)) == (0, 0),
            "road layer mask has a non-zero offset",
        )

        pixels = np.asarray(layer.image)
        mask = np.asarray(layer.mask.image)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        _require(pixels.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 4), f"road pixels shape {pixels.shape}")
        _require(mask.shape == (CANVAS_HEIGHT, CANVAS_WIDTH), f"road mask shape {mask.shape}")
        _require(pixels.dtype == np.uint8 and mask.dtype == np.uint8, "XCF arrays are not uint8")

        alpha = pixels[:, :, 3]
        nonzero_pixels = pixels[alpha > 0]
        _require(
            int(nonzero_pixels.shape[0]) == EXPECTED_LAYER_PAINT_PIXELS,
            f"road paint occupancy {nonzero_pixels.shape[0]} != {EXPECTED_LAYER_PAINT_PIXELS}",
        )
        _require(
            nonzero_pixels.size == 0
            or np.all(np.all(nonzero_pixels == np.asarray(ROAD_LAYER_PAINT_RGBA, dtype=np.uint8), axis=1)),
            "road layer has a non-contract nonzero RGBA paint value",
        )

        # GIMP's normal mask modulation is alpha * mask / 255.  The road
        # layer is opaque, so the effective alpha is numerically the mask,
        # including the small intermediate values retained by the source.
        effective_layer = (alpha.astype(np.uint16) * mask.astype(np.uint16) // 255).astype(np.uint8)
        effective_canvas = _place_layer_canvas(effective_layer, ROAD_LAYER_OFFSET)
        binary_canvas = (effective_canvas > 0).astype(np.uint8)
        _require(
            int(np.count_nonzero(effective_canvas)) == EXPECTED_SOURCE_OCCUPANCY,
            f"effective occupancy {np.count_nonzero(effective_canvas)} != {EXPECTED_SOURCE_OCCUPANCY}",
        )
        metadata: dict[str, Any] = {
            "xcf_version": int(document.version),
            "canvas_size_px": [CANVAS_WIDTH, CANVAS_HEIGHT],
            "base_color_mode": int(document.baseColorMode),
            "precision": precision,
            "layer_name": ROAD_LAYER_NAME,
            "layer_index_top_first": int(document.raw_layers.index(layer)),
            "layer_size_px": [int(layer.width), int(layer.height)],
            "layer_offset_px": [int(layer.xOffset), int(layer.yOffset)],
            "layer_visible": bool(layer.visible),
            "layer_opacity": float(layer.opacity),
            "layer_blend_mode": str(getattr(layer.blendMode, "value", layer.blendMode)),
            "layer_apply_mask": bool(layer.applyMask),
            "mask_size_px": [int(layer.mask.width), int(layer.mask.height)],
            "mask_offset_px": [
                int(getattr(layer.mask, "xOffset", 0)),
                int(getattr(layer.mask, "yOffset", 0)),
            ],
            "layer_channel_count": int(pixels.shape[2]),
            "layer_nonzero_rgba": list(ROAD_LAYER_PAINT_RGBA),
            "layer_nonzero_pixel_count": int(np.count_nonzero(alpha)),
            "mask_nonzero_pixel_count": int(np.count_nonzero(mask)),
            "effective_alpha_values": [int(value) for value in np.unique(effective_canvas)],
            "effective_occupancy": int(np.count_nonzero(effective_canvas)),
            "effective_alpha_sha256": sha256_bytes(np.ascontiguousarray(effective_canvas).tobytes()),
            "binary_mask_sha256": sha256_bytes(np.ascontiguousarray(binary_canvas).tobytes()),
            "layer_alpha_sha256": sha256_bytes(np.ascontiguousarray(alpha).tobytes()),
            "layer_mask_sha256": sha256_bytes(np.ascontiguousarray(mask).tobytes()),
        }
        return effective_canvas, binary_canvas, metadata


def extract_road_source(xcf_path: str | Path) -> RoadSource:
    """Extract the pinned visible road layer from ``xcf_path``.

    The returned source hash is over the canvas-oriented effective alpha bytes,
    not over a repaired or thresholded mask.  This distinction is carried into
    the canonical graph so source fidelity can be audited independently from
    all later topology decisions.
    """

    path = Path(xcf_path)
    if not path.is_file():
        raise FileNotFoundError(f"authoritative XCF does not exist: {path}")
    effective_alpha, binary_mask, metadata = _load_document_and_extract(path)
    metadata = dict(metadata)
    metadata["xcf_path"] = str(path)
    metadata["xcf_sha256"] = sha256_file(path)
    metadata["binary_occupancy"] = int(np.count_nonzero(binary_mask))
    return RoadSource(
        effective_alpha=np.ascontiguousarray(effective_alpha, dtype=np.uint8),
        binary_mask=np.ascontiguousarray(binary_mask, dtype=np.uint8),
        metadata=metadata,
    )


def compare_corrected_effective_png(effective_alpha: np.ndarray, evidence_png: str | Path) -> dict[str, Any]:
    """Compare a source array with the corrected parity PNG, if supplied.

    The parity image was written as an 8-bit grayscale canvas image by the
    corrected investigation.  Comparing decoded pixels as well as file hash
    avoids treating a different PNG encoder as a source mismatch while still
    recording the evidence artifact's exact digest.
    """

    path = Path(evidence_png)
    if not path.is_file():
        raise FileNotFoundError(f"corrected parity evidence image does not exist: {path}")
    with Image.open(path) as image:
        evidence = np.asarray(image.convert("L"))
    actual = np.asarray(effective_alpha, dtype=np.uint8)
    _require(evidence.shape == actual.shape, f"parity PNG shape {evidence.shape} != {actual.shape}")
    equal = bool(np.array_equal(evidence, actual))
    _require(equal, "effective mask pixels differ from corrected parity PNG")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "pixel_identical": equal,
        "occupancy": int(np.count_nonzero(evidence)),
        "size_px": [int(evidence.shape[1]), int(evidence.shape[0])],
    }


def read_vtex_canvas(bmp_path: str | Path, *, width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT) -> np.ndarray:
    """Read the 16-bit TESAnnwyn VTEX BMP into north-up canvas orientation.

    TESAnnwyn's BMP rows are bottom-up (row zero is the southern edge), while
    the XCF canvas is north-up.  The returned array is a copied little-endian
    ``uint16`` grid with row zero north, suitable for direct source overlays.
    """

    path = Path(bmp_path)
    with path.open("rb") as handle:
        header = handle.read(70)
        if len(header) < 70 or header[:2] != b"BM":
            raise ValueError(f"not a BMP: {path}")
        data_offset = int.from_bytes(header[10:14], "little")
        bmp_width = int.from_bytes(header[18:22], "little", signed=True)
        bmp_height = int.from_bytes(header[22:26], "little", signed=True)
        bpp = int.from_bytes(header[28:30], "little")
        compression = int.from_bytes(header[30:34], "little")
        if (bmp_width, bmp_height) != (width, height):
            raise ValueError(f"VTEX BMP dimensions {(bmp_width, bmp_height)} != {(width, height)}")
        if bpp != 16 or compression not in (0, 3):
            raise ValueError(f"VTEX BMP requires 16-bit BI_RGB/BI_BITFIELDS; got bpp={bpp}, compression={compression}")
        if compression == 3:
            # The supplied TESAnnwyn file advertises BI_BITFIELDS and stores
            # the standard 5-6-5 masks in the 12 bytes between the DIB header
            # and pixel data.  The road audit consumes the raw little-endian
            # 16-bit index values, not decoded RGB channels, so the masks are
            # metadata only but are pinned to prevent accepting a different
            # packed format by accident.
            masks = tuple(int.from_bytes(header[offset : offset + 4], "little") for offset in (54, 58, 62))
            if masks != (0x0000F800, 0x000007E0, 0x0000001F):
                raise ValueError(f"unexpected 16-bit VTEX bit masks: {masks!r}")
        row_bytes = ((width * 2 + 3) // 4) * 4
        handle.seek(data_offset)
        raw = np.frombuffer(handle.read(row_bytes * height), dtype="<u2")
    if raw.size != width * height:
        raise ValueError(f"VTEX BMP pixel payload has {raw.size} values, expected {width * height}")
    return np.flipud(raw.reshape(height, width)).copy()


def parse_ltex_palette(path: str | Path) -> dict[int, dict[str, str]]:
    """Parse the supplied ``tes3ltex.txt`` into raw-VTEX labels.

    Its first field is zero-based LTEX order; raw VTEX value is therefore
    ``field + 1``.  Raw value 1 remains the Sand record and is never folded
    into the road mask merely because it has a terrain-like colour.
    """

    labels: dict[int, dict[str, str]] = {}
    with Path(path).open("r", encoding="latin-1") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(",")
            if len(fields) < 3:
                continue
            try:
                ltex_index = int(fields[0])
            except ValueError:
                continue
            labels[ltex_index + 1] = {
                "ltex_index": str(ltex_index),
                "name": fields[1],
                "texture": fields[2],
                "line": str(line_number),
            }
    return labels


__all__ = [
    "CANVAS_HEIGHT",
    "CANVAS_WIDTH",
    "EXPECTED_SOURCE_OCCUPANCY",
    "ROAD_LAYER_NAME",
    "RoadSource",
    "compare_corrected_effective_png",
    "extract_road_source",
    "parse_ltex_palette",
    "read_vtex_canvas",
    "sha256_bytes",
    "sha256_file",
]
