from __future__ import annotations

from dataclasses import dataclass
from glob import has_magic
from os import PathLike, fspath
from pathlib import Path
from typing import Literal

import numpy as np

from ._variopinta import decode_image as _decode_image
from ._variopinta import encode_image as _encode_image
from ._variopinta import read_image as _read_image
from ._variopinta import write_image as _write_image

ImageFormat = Literal["jpeg", "png"]
ImageMode = Literal["unchanged", "gray", "rgb", "rgba"]
DEFAULT_MAX_PIXELS = 100_000_000


@dataclass(frozen=True, slots=True)
class ArrayInput:
    pass


@dataclass(frozen=True, slots=True)
class EncodedInput:
    max_pixels: int | None = DEFAULT_MAX_PIXELS
    max_encoded_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_limit("max_pixels", self.max_pixels)
        _validate_limit("max_encoded_bytes", self.max_encoded_bytes)


@dataclass(frozen=True, slots=True)
class PathInput:
    max_pixels: int | None = DEFAULT_MAX_PIXELS
    max_encoded_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_limit("max_pixels", self.max_pixels)
        _validate_limit("max_encoded_bytes", self.max_encoded_bytes)


@dataclass(frozen=True, slots=True)
class ReturnOutput:
    pass


@dataclass(frozen=True, slots=True)
class EncodedOutput:
    format: ImageFormat
    quality: int | None = None
    compression: int | None = None

    def __post_init__(self) -> None:
        image_format = _normalize_format(self.format)
        quality, compression = _validate_encode_options(
            image_format, self.quality, self.compression
        )
        object.__setattr__(self, "format", image_format)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "compression", compression)


@dataclass(frozen=True, slots=True)
class PathOutput:
    format: ImageFormat
    quality: int | None = None
    compression: int | None = None

    def __post_init__(self) -> None:
        image_format = _normalize_format(self.format)
        quality, compression = _validate_encode_options(
            image_format, self.quality, self.compression
        )
        object.__setattr__(self, "format", image_format)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "compression", compression)


def read_image(
    path: str | PathLike[str],
    *,
    mode: ImageMode = "rgb",
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
    max_encoded_bytes: int | None = None,
) -> np.ndarray:
    """Decode a JPEG or PNG file into an owned NumPy array."""
    return _read_image(
        _path(path, "path"),
        mode,
        _validate_limit("max_pixels", max_pixels),
        _validate_limit("max_encoded_bytes", max_encoded_bytes),
    )


def decode_image(
    data: bytes | bytearray | memoryview,
    *,
    mode: ImageMode = "rgb",
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
    max_encoded_bytes: int | None = None,
) -> np.ndarray:
    """Decode JPEG or PNG bytes into an owned NumPy array."""
    return _decode_image(
        data,
        mode,
        _validate_limit("max_pixels", max_pixels),
        _validate_limit("max_encoded_bytes", max_encoded_bytes),
    )


def encode_image(
    image: np.ndarray,
    *,
    format: ImageFormat,
    quality: int | None = None,
    compression: int | None = None,
) -> bytes:
    """Encode a NumPy array as JPEG or PNG bytes."""
    image_format = _normalize_format(format)
    quality, compression = _validate_encode_options(image_format, quality, compression)
    return _encode_image(_prepare_image(image), image_format, quality, compression)


def write_image(
    path: str | PathLike[str],
    image: np.ndarray,
    *,
    format: ImageFormat | None = None,
    quality: int | None = None,
    compression: int | None = None,
) -> None:
    """Encode a NumPy array and write it to a JPEG or PNG file."""
    path = _path(path, "path")
    inferred = _format_from_path(path)
    image_format = inferred if format is None else _normalize_format(format)
    if image_format is None:
        raise ValueError("format is required when the path has no JPEG or PNG extension")
    if inferred is not None and inferred != image_format:
        raise ValueError("format conflicts with the path extension")
    quality, compression = _validate_encode_options(image_format, quality, compression)
    _write_image(path, _prepare_image(image), image_format, quality, compression)


def _prepare_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise ValueError("image dtype must be uint8 or uint16")
    return np.ascontiguousarray(image)


def _normalize_format(value: str) -> ImageFormat:
    if not isinstance(value, str):
        raise TypeError("format must be a string")
    normalized = value.lower().removeprefix(".")
    if normalized in ("jpg", "jpeg"):
        return "jpeg"
    if normalized == "png":
        return "png"
    raise ValueError("format must be 'jpeg' or 'png'")


def _format_from_path(path: str | PathLike[str]) -> ImageFormat | None:
    suffix = Path(path).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "jpeg"
    if suffix == ".png":
        return "png"
    return None


def _validate_encode_options(
    image_format: ImageFormat,
    quality: int | None,
    compression: int | None,
) -> tuple[int | None, int | None]:
    if image_format == "jpeg":
        if compression is not None:
            raise TypeError("compression is only valid for PNG")
        return _integer_option("quality", 95 if quality is None else quality, 1, 100), None
    if quality is not None:
        raise TypeError("quality is only valid for JPEG")
    return None, _integer_option("compression", 6 if compression is None else compression, 0, 9)


def _integer_option(name: str, value: int, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _validate_limit(name: str, value: int | None) -> int | None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{name} must be a positive integer or None")
    return value


def _path(value: object, name: str) -> Path:
    if isinstance(value, bytes):
        raise TypeError(f"{name} must be str or os.PathLike[str]")
    try:
        normalized = fspath(value)
    except TypeError as error:
        raise TypeError(f"{name} must be str or os.PathLike[str]") from error
    if not isinstance(normalized, str):
        raise TypeError(f"{name} must be str or os.PathLike[str]")
    if "://" in normalized or normalized.startswith("file:") or has_magic(normalized):
        raise TypeError(f"{name} must be a local path, not a URL or glob")
    return Path(normalized)
