from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, overload

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
_OutputValue = TypeVar("_OutputValue", covariant=True)
_BIND_TOKEN = object()
_RESERVED_NAMES = frozenset({"key"})


def _validate_name(name: object, level: str) -> None:
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"{level} name must be a public Python identifier")
    if name in _RESERVED_NAMES or __import__("keyword").iskeyword(name):
        raise ValueError(f"{level} name {name!r} is reserved")


@dataclass(frozen=True, slots=True)
class Array:
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Encoded:
    max_pixels: int | None = DEFAULT_MAX_PIXELS
    max_encoded_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_limit("max_pixels", self.max_pixels)
        _validate_limit("max_encoded_bytes", self.max_encoded_bytes)


@dataclass(frozen=True, slots=True, kw_only=True)
class Path:
    max_pixels: int | None = DEFAULT_MAX_PIXELS
    max_encoded_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_limit("max_pixels", self.max_pixels)
        _validate_limit("max_encoded_bytes", self.max_encoded_bytes)


class OutputPort(Generic[_OutputValue]):
    __slots__ = ()

    name: str

    def __new__(cls, *args: object, **kwargs: object) -> OutputPort[object]:
        if cls is OutputPort:
            raise TypeError("OutputPort is an abstract output port")
        return super().__new__(cls)


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class ReturnArray(OutputPort[np.ndarray]):
    name: str

    def __post_init__(self) -> None:
        _validate_name(self.name, "output")


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class ReturnTensor(OutputPort[_TorchTensor]):
    name: str

    def __post_init__(self) -> None:
        _validate_name(self.name, "output")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Encode(OutputPort[bytes]):
    format: ImageFormat
    name: str
    quality: int | None
    compression_level: int | None

    def __init__(
        self,
        format: str,
        *,
        name: str,
        quality: int | None = None,
        compression_level: int | None = None,
    ) -> None:
        object.__setattr__(self, "format", format)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "compression_level", compression_level)
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_name(self.name, "output")
        image_format = _normalize_format(self.format)
        quality, compression_level = _validate_encode_options(
            image_format, self.quality, self.compression_level
        )
        object.__setattr__(self, "format", image_format)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "compression_level", compression_level)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Write(OutputPort[_Path]):
    format: ImageFormat | None
    name: str
    quality: int | None
    compression_level: int | None

    def __init__(
        self,
        format: str | None = None,
        *,
        name: str,
        quality: int | None = None,
        compression_level: int | None = None,
    ) -> None:
        object.__setattr__(self, "format", format)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "compression_level", compression_level)
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_name(self.name, "output")
        if self.format is None:
            if self.quality is not None and self.compression_level is not None:
                raise TypeError("quality and compression_level require different output formats")
            if self.quality is not None:
                _validate_encode_options("jpeg", self.quality, None)
            if self.compression_level is not None:
                _validate_encode_options("png", None, self.compression_level)
            return
        image_format = _normalize_format(self.format)
        quality, compression_level = _validate_encode_options(
            image_format, self.quality, self.compression_level
        )
        object.__setattr__(self, "format", image_format)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "compression_level", compression_level)

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
        _validate_encode_options(image_format, self.quality, self.compression_level)
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
            f"input_spec={type(self._target.input_spec).__name__}, "
            f"output_specs={len(self._target.output_specs)})"
        )


@dataclass(frozen=True, slots=True, eq=False)
class Image:
    input_spec: Carrier = Array()
    name: str = field(kw_only=True)
    output_specs: Output | Sequence[OutputPort[object]] = field(
        default_factory=lambda: (ReturnArray(name="array"),), kw_only=True
    )
    decode_mode: Literal["rgb", "gray"] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.decode_mode not in (None, "rgb", "gray"):
            raise ValueError("decode_mode must be 'rgb', 'gray', or None")
        if isinstance(self.input_spec, Array) and self.decode_mode is not None:
            raise ValueError("Array images infer channels; decode_mode must be None")
        object.__setattr__(
            self,
            "output_specs",
            _validate_target(self.input_spec, self.output_specs, self.name),
        )

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
    input_spec: Carrier = Array()
    name: str = field(kw_only=True)
    output_specs: Output | Sequence[OutputPort[object]] = field(
        default_factory=lambda: (ReturnArray(name="array"),), kw_only=True
    )
    fill: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        output_specs = _validate_target(self.input_spec, self.output_specs, self.name)
        object.__setattr__(self, "output_specs", output_specs)
        if type(self.fill) is not int or not 0 <= self.fill <= 255:
            raise ValueError("fill must be an integer in [0, 255]")
        for output in output_specs:
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
    input_spec: object,
    output_specs: object,
    name: object,
) -> tuple[OutputPort[object], ...]:
    if not isinstance(input_spec, Array | Encoded | Path):
        raise TypeError("input_spec must be Array, Encoded, or Path")
    _validate_name(name, "target")
    if isinstance(output_specs, OutputPort):
        normalized = (output_specs,)
    else:
        if isinstance(output_specs, str | bytes) or not isinstance(output_specs, Sequence):
            raise TypeError("output_specs must be an output port or a sequence of output ports")
        normalized = tuple(output_specs)
    if not normalized:
        raise ValueError("output_specs must contain at least one output port")
    if not all(type(output) in (ReturnArray, ReturnTensor, Encode, Write) for output in normalized):
        raise TypeError("output_specs must contain only built-in output ports")
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
    if isinstance(target.input_spec, Array):
        if not isinstance(source, np.ndarray):
            raise TypeError(f"{_label(target)} source must be a NumPy array for Array")
        normalized_source = source
    elif isinstance(target.input_spec, Encoded):
        if not isinstance(source, bytes | bytearray | memoryview):
            raise TypeError(
                f"{_label(target)} source must be bytes, bytearray, or memoryview for Encoded"
            )
        normalized_source = source
    else:
        normalized_source = _path(source, f"{_label(target)} source")

    if not all(isinstance(binding, WriteBinding) for binding in write_bindings):
        raise TypeError("target.bind() accepts only Write.bind() values after the source")
    expected = tuple(output for output in target.output_specs if isinstance(output, Write))
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
    input_spec = target.input_spec
    return {
        "role": "image" if isinstance(target, Image) else "mask",
        "decode_mode": target.decode_mode if isinstance(target, Image) else None,
        "fill": target.fill if isinstance(target, Mask) else None,
        "name": target.name,
        "carrier": (
            "array"
            if isinstance(input_spec, Array)
            else "encoded"
            if isinstance(input_spec, Encoded)
            else "path"
        ),
        "max_pixels": input_spec.max_pixels if isinstance(input_spec, Encoded | Path) else None,
        "max_encoded_bytes": (
            input_spec.max_encoded_bytes if isinstance(input_spec, Encoded | Path) else None
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
                "compression": (
                    output.compression_level if isinstance(output, Encode | Write) else None
                ),
            }
            for output in target.output_specs
        ],
    }


def _implicit_image() -> Image:
    output = object.__new__(ReturnArray)
    object.__setattr__(output, "name", None)
    target = object.__new__(Image)
    object.__setattr__(target, "input_spec", Array())
    object.__setattr__(target, "name", None)
    object.__setattr__(target, "output_specs", (output,))
    object.__setattr__(target, "decode_mode", None)
    return target
