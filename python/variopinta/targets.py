from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Generic, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor as _TorchTensor
else:
    _TorchTensor = object

from .io import (
    DEFAULT_MAX_PIXELS,
    ImageFormat,
    _format_from_path,
    _normalize_format,
    _path,
    _validate_encode_options,
    _validate_limit,
)

_Result = TypeVar("_Result")
_BIND_TOKEN = object()
_RESERVED_NAMES = frozenset({"key"})


def _validate_name(name: object, level: str) -> None:
    if name is None:
        return
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"{level} name must be a public Python identifier or None")
    if name in _RESERVED_NAMES or __import__("keyword").iskeyword(name):
        raise ValueError(f"{level} name {name!r} is reserved")


@dataclass(frozen=True, slots=True)
class Array:
    pass


@dataclass(frozen=True, slots=True)
class Encoded:
    max_pixels: int | None = DEFAULT_MAX_PIXELS
    max_encoded_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_limit("max_pixels", self.max_pixels)
        _validate_limit("max_encoded_bytes", self.max_encoded_bytes)


@dataclass(frozen=True, slots=True)
class Path:
    max_pixels: int | None = DEFAULT_MAX_PIXELS
    max_encoded_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_limit("max_pixels", self.max_pixels)
        _validate_limit("max_encoded_bytes", self.max_encoded_bytes)


class OutputPort(Generic[_Result]):
    __slots__ = ()

    name: str | None

    def __new__(cls, *args: object, **kwargs: object) -> OutputPort[object]:
        if cls is OutputPort:
            raise TypeError("OutputPort is an abstract output port")
        return super().__new__(cls)


@dataclass(frozen=True, slots=True, eq=False)
class ReturnArray(OutputPort[np.ndarray]):
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, "output")


@dataclass(frozen=True, slots=True, eq=False)
class ReturnTensor(OutputPort[_TorchTensor]):
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, "output")


@dataclass(frozen=True, slots=True, eq=False)
class Encode(OutputPort[bytes]):
    format: ImageFormat
    quality: int | None = None
    compression: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, "output")
        image_format = _normalize_format(self.format)
        quality, compression = _validate_encode_options(
            image_format, self.quality, self.compression
        )
        object.__setattr__(self, "format", image_format)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "compression", compression)


@dataclass(frozen=True, slots=True, eq=False)
class Write(OutputPort[_Path]):
    format: ImageFormat | None = None
    quality: int | None = None
    compression: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, "output")
        if self.format is None:
            if self.quality is not None and self.compression is not None:
                raise TypeError("quality and compression require different output formats")
            if self.quality is not None:
                _validate_encode_options("jpeg", self.quality, None)
            if self.compression is not None:
                _validate_encode_options("png", None, self.compression)
            return
        image_format = _normalize_format(self.format)
        quality, compression = _validate_encode_options(
            image_format, self.quality, self.compression
        )
        object.__setattr__(self, "format", image_format)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "compression", compression)

    def bind(self, destination: str | PathLike[str]) -> WriteBinding:
        normalized = _path(destination, _output_label(self, "destination"))
        inferred = _format_from_path(normalized)
        image_format = self.format or inferred
        if image_format is None:
            raise ValueError(
                f"format is required for {_output_label(self, 'destination')} without a JPEG or PNG suffix"
            )
        if inferred is not None and inferred != image_format:
            raise ValueError("output format conflicts with the destination suffix")
        _validate_encode_options(image_format, self.quality, self.compression)
        return WriteBinding(self, normalized, _token=_BIND_TOKEN)


Carrier = Array | Encoded | Path
Output = ReturnArray | ReturnTensor | Encode | Write


@dataclass(frozen=True, slots=True, eq=False, init=False)
class WriteBinding:
    _output: Write
    _destination: _Path

    def __init__(
        self,
        output: Write,
        destination: _Path,
        *,
        _token: object = None,
    ) -> None:
        if _token is not _BIND_TOKEN:
            raise TypeError("WriteBinding values are created by Write.bind()")
        object.__setattr__(self, "_output", output)
        object.__setattr__(self, "_destination", destination)

    @property
    def output(self) -> Write:
        return self._output

    @property
    def destination(self) -> _Path:
        return self._destination

    def __repr__(self) -> str:
        return f"WriteBinding(output={_port_repr(self._output)})"


@dataclass(frozen=True, slots=True, eq=False, init=False)
class BoundTarget(Generic[_Result]):
    _target: Image | Mask
    _source: object
    _write_bindings: tuple[WriteBinding, ...]

    def __init__(
        self,
        target: Image | Mask,
        source: object,
        write_bindings: tuple[WriteBinding, ...],
        *,
        _token: object = None,
    ) -> None:
        if _token is not _BIND_TOKEN:
            raise TypeError("BoundTarget values are created by target.bind()")
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_write_bindings", write_bindings)

    @property
    def target(self) -> Image | Mask:
        return self._target

    def __repr__(self) -> str:
        return (
            f"BoundTarget(target={_target_repr(self._target)}, "
            f"carrier={type(self._target.carrier).__name__}, "
            f"outputs={len(self._target.outputs)})"
        )


@dataclass(frozen=True, slots=True, eq=False)
class Image:
    carrier: Carrier = Array()
    outputs: Sequence[OutputPort[object]] = (ReturnArray(),)
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", _validate_target(self.carrier, self.outputs, self.name))

    @overload
    def bind(self, source: np.ndarray, *write_bindings: WriteBinding) -> BoundTarget[object]: ...

    @overload
    def bind(
        self,
        source: bytes | bytearray | memoryview | str | PathLike[str],
        *write_bindings: WriteBinding,
    ) -> BoundTarget[object]: ...

    def bind(self, source: object, *write_bindings: WriteBinding) -> BoundTarget[object]:
        return _bind(self, source, write_bindings)


@dataclass(frozen=True, slots=True, eq=False)
class Mask:
    carrier: Carrier = Array()
    outputs: Sequence[OutputPort[object]] = (ReturnArray(),)
    fill: int = 0
    name: str | None = None

    def __post_init__(self) -> None:
        outputs = _validate_target(self.carrier, self.outputs, self.name)
        object.__setattr__(self, "outputs", outputs)
        if type(self.fill) is not int or not 0 <= self.fill <= 255:
            raise ValueError("fill must be an integer in [0, 255]")
        for output in outputs:
            if isinstance(output, Encode) and output.format != "png":
                raise ValueError("Mask Encode output must use PNG")
            if isinstance(output, Write) and output.format not in (None, "png"):
                raise ValueError("Mask Write output must use PNG")
            if isinstance(output, Write) and output.quality is not None:
                raise TypeError("quality is not valid for Mask Write output")

    @overload
    def bind(self, source: np.ndarray, *write_bindings: WriteBinding) -> BoundTarget[object]: ...

    @overload
    def bind(
        self,
        source: bytes | bytearray | memoryview | str | PathLike[str],
        *write_bindings: WriteBinding,
    ) -> BoundTarget[object]: ...

    def bind(self, source: object, *write_bindings: WriteBinding) -> BoundTarget[object]:
        return _bind(self, source, write_bindings)


Target = Image | Mask


def _validate_target(
    carrier: object,
    outputs: object,
    name: object,
) -> tuple[OutputPort[object], ...]:
    if not isinstance(carrier, Array | Encoded | Path):
        raise TypeError("carrier must be Array, Encoded, or Path")
    _validate_name(name, "target")
    if isinstance(outputs, str | bytes) or not isinstance(outputs, Sequence):
        raise TypeError("outputs must be a sequence of output ports")
    normalized = tuple(outputs)
    if not normalized:
        raise ValueError("outputs must contain at least one output port")
    if not all(type(output) in (ReturnArray, ReturnTensor, Encode, Write) for output in normalized):
        raise TypeError("outputs must contain only built-in output ports")
    if len({id(output) for output in normalized}) != len(normalized):
        raise ValueError("the same output port cannot appear more than once")
    names = [output.name for output in normalized if output.name is not None]
    if len(set(names)) != len(names):
        raise ValueError("output names must be unique within a target")
    return normalized


def _bind(
    target: Target,
    source: object,
    write_bindings: tuple[WriteBinding, ...],
) -> BoundTarget[object]:
    if isinstance(target.carrier, Array):
        if not isinstance(source, np.ndarray):
            raise TypeError(f"{_label(target)} source must be a NumPy array for Array")
        normalized_source = source
    elif isinstance(target.carrier, Encoded):
        if not isinstance(source, bytes | bytearray | memoryview):
            raise TypeError(
                f"{_label(target)} source must be bytes, bytearray, or memoryview for Encoded"
            )
        normalized_source = source
    else:
        normalized_source = _path(source, f"{_label(target)} source")

    if not all(isinstance(binding, WriteBinding) for binding in write_bindings):
        raise TypeError("target.bind() accepts only Write.bind() values after the source")
    expected = tuple(output for output in target.outputs if isinstance(output, Write))
    seen: set[int] = set()
    by_output: dict[int, WriteBinding] = {}
    for binding in write_bindings:
        identity = id(binding.output)
        if identity in seen:
            raise ValueError("a Write output cannot be bound more than once")
        if not any(binding.output is output for output in expected):
            raise ValueError("Write binding belongs to a different target or output port")
        seen.add(identity)
        by_output[identity] = binding
    missing = [output for output in expected if id(output) not in seen]
    if missing:
        names = ", ".join(output.name or "<unnamed>" for output in missing)
        raise TypeError(f"missing Write bindings: {names}")
    ordered = tuple(by_output[id(output)] for output in expected)
    if isinstance(target, Mask):
        for binding in ordered:
            inferred = _format_from_path(binding.destination)
            image_format = binding.output.format or inferred
            if image_format != "png":
                raise ValueError("Mask Write output must use a PNG destination")
    return BoundTarget(target, normalized_source, ordered, _token=_BIND_TOKEN)


def _label(target: Target) -> str:
    return type(target).__name__ if target.name is None else target.name


def _target_repr(target: Target) -> str:
    return (
        type(target).__name__
        if target.name is None
        else f"{type(target).__name__}({target.name!r})"
    )


def _port_repr(output: OutputPort[object]) -> str:
    return (
        type(output).__name__
        if output.name is None
        else f"{type(output).__name__}({output.name!r})"
    )


def _output_label(output: OutputPort[object], suffix: str) -> str:
    return f"{_port_repr(output)} {suffix}"


def _route(target: Target) -> dict[str, object]:
    carrier = target.carrier
    return {
        "role": "image" if isinstance(target, Image) else "mask",
        "fill": target.fill if isinstance(target, Mask) else None,
        "name": target.name,
        "carrier": (
            "array"
            if isinstance(carrier, Array)
            else "encoded"
            if isinstance(carrier, Encoded)
            else "path"
        ),
        "max_pixels": carrier.max_pixels if isinstance(carrier, Encoded | Path) else None,
        "max_encoded_bytes": (
            carrier.max_encoded_bytes if isinstance(carrier, Encoded | Path) else None
        ),
        "outputs": [
            {
                "name": output.name,
                "type": (
                    "return_array"
                    if isinstance(output, ReturnArray)
                    else "return_tensor"
                    if isinstance(output, ReturnTensor)
                    else "encode"
                    if isinstance(output, Encode)
                    else "write"
                ),
                "format": output.format if isinstance(output, Encode | Write) else None,
                "quality": output.quality if isinstance(output, Encode | Write) else None,
                "compression": output.compression if isinstance(output, Encode | Write) else None,
            }
            for output in target.outputs
        ],
    }
