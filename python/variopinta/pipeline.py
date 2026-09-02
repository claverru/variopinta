from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from ._validation import _image, _key, _seed, _torch_module
from ._variopinta import Pipeline as _NativePipeline
from .io import (
    ArrayInput,
    EncodedInput,
    EncodedOutput,
    PathInput,
    PathOutput,
    ReturnOutput,
    _format_from_path,
    _path,
)
from .transforms import _TRANSFORM_TYPES, Normalize, ToTorch, Transform

if TYPE_CHECKING:
    import torch

    _PipelineOutput: TypeAlias = np.ndarray | torch.Tensor | bytes | None
else:
    _PipelineOutput: TypeAlias = object

_PipelineInput: TypeAlias = np.ndarray | bytes | bytearray | memoryview | str | PathLike[str]
_InputConfiguration: TypeAlias = ArrayInput | EncodedInput | PathInput
_OutputConfiguration: TypeAlias = ReturnOutput | EncodedOutput | PathOutput

_DEFAULT_INPUT = ArrayInput()
_DEFAULT_OUTPUT = ReturnOutput()


class Compose:
    __slots__ = ("_input", "_output", "_pipeline", "_seed", "_specs", "_transforms")

    def __init__(
        self,
        transforms: Sequence[Transform],
        seed: int | None = None,
        *,
        input: _InputConfiguration = _DEFAULT_INPUT,
        output: _OutputConfiguration = _DEFAULT_OUTPUT,
    ) -> None:
        self._transforms = tuple(transforms)
        if not all(isinstance(transform, _TRANSFORM_TYPES) for transform in self._transforms):
            raise TypeError("Compose only accepts built-in transforms")
        self._input = _input_configuration(input)
        self._output = _output_configuration(output)
        _validate_encodable_output(self._transforms, self._output)
        self._seed = _seed(seed)
        self._specs = [transform._spec() for transform in self._transforms]
        self._pipeline = _native_pipeline(
            self._specs, self._seed, "reference", self._input, self._output
        )

    @property
    def transforms(self) -> tuple[Transform, ...]:
        return self._transforms

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def input(self) -> _InputConfiguration:
        return self._input

    @property
    def output(self) -> _OutputConfiguration:
        return self._output

    def __call__(
        self,
        image: _PipelineInput,
        *,
        destination: str | PathLike[str] | None = None,
        key: int | None = None,
    ) -> _PipelineOutput:
        return _apply(self, image, destination, key)

    def compile(self) -> CompiledCompose:
        return CompiledCompose(self._transforms, self._specs, self._seed, self._input, self._output)

    def explain(self) -> dict[str, object]:
        return self._pipeline.explain()


class CompiledCompose:
    __slots__ = ("_input", "_output", "_pipeline", "_seed", "_transforms")

    def __init__(
        self,
        transforms: tuple[Transform, ...],
        specs: list[dict[str, object]],
        seed: int,
        input: _InputConfiguration,
        output: _OutputConfiguration,
    ) -> None:
        self._transforms = transforms
        self._seed = seed
        self._input = input
        self._output = output
        self._pipeline = _native_pipeline(specs, seed, "compiled", input, output)

    @property
    def transforms(self) -> tuple[Transform, ...]:
        return self._transforms

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def input(self) -> _InputConfiguration:
        return self._input

    @property
    def output(self) -> _OutputConfiguration:
        return self._output

    def __call__(
        self,
        image: _PipelineInput,
        *,
        destination: str | PathLike[str] | None = None,
        key: int | None = None,
    ) -> _PipelineOutput:
        return _apply(self, image, destination, key)

    def explain(self) -> dict[str, object]:
        return self._pipeline.explain()


def _apply(
    pipeline: Compose | CompiledCompose,
    image: _PipelineInput,
    destination: str | PathLike[str] | None,
    key: int | None,
) -> _PipelineOutput:
    normalized_destination = _destination(pipeline.output, destination)
    if isinstance(pipeline.input, ArrayInput):
        prepared = _image(image)
        output = pipeline._pipeline.apply(prepared, _key(key), normalized_destination)
    elif isinstance(pipeline.input, EncodedInput):
        if not isinstance(image, bytes | bytearray | memoryview):
            raise TypeError("image must be bytes, bytearray, or memoryview for EncodedInput")
        output = pipeline._pipeline.apply_encoded(image, _key(key), normalized_destination)
    else:
        source = _path(image, "image")
        output = pipeline._pipeline.apply_path(source, _key(key), normalized_destination)

    torch = (
        _torch_module()
        if isinstance(pipeline.output, ReturnOutput)
        and pipeline.transforms
        and isinstance(pipeline.transforms[-1], ToTorch)
        else None
    )
    return output if torch is None else torch.from_numpy(output)


def _input_configuration(value: object) -> _InputConfiguration:
    if not isinstance(value, ArrayInput | EncodedInput | PathInput):
        raise TypeError("input must be ArrayInput, EncodedInput, or PathInput")
    return value


def _output_configuration(value: object) -> _OutputConfiguration:
    if not isinstance(value, ReturnOutput | EncodedOutput | PathOutput):
        raise TypeError("output must be ReturnOutput, EncodedOutput, or PathOutput")
    return value


def _validate_encodable_output(
    transforms: tuple[Transform, ...], output: _OutputConfiguration
) -> None:
    if isinstance(output, ReturnOutput):
        return
    if any(isinstance(transform, ToTorch) for transform in transforms) or any(
        isinstance(transform, Normalize) and transform.p != 0.0 for transform in transforms
    ):
        raise ValueError("encoded pipeline output requires an always-HWC RGB uint8 result")


def _destination(
    output: _OutputConfiguration, destination: str | PathLike[str] | None
) -> Path | None:
    if not isinstance(output, PathOutput):
        if destination is not None:
            raise TypeError("destination is only valid for PathOutput")
        return None
    if destination is None:
        raise TypeError("destination is required for PathOutput")
    path = _path(destination, "destination")
    inferred = _format_from_path(path)
    if inferred is not None and inferred != output.format:
        raise ValueError("output format conflicts with the destination extension")
    return path


def _native_pipeline(
    specs: list[dict[str, object]],
    seed: int,
    mode: str,
    input: _InputConfiguration,
    output: _OutputConfiguration,
) -> _NativePipeline:
    source_kind = "array"
    max_pixels = None
    max_encoded_bytes = None
    if isinstance(input, EncodedInput | PathInput):
        source_kind = "encoded" if isinstance(input, EncodedInput) else "path"
        max_pixels = input.max_pixels
        max_encoded_bytes = input.max_encoded_bytes

    sink_kind = "return"
    image_format = None
    quality = None
    compression = None
    if isinstance(output, EncodedOutput | PathOutput):
        sink_kind = "encoded" if isinstance(output, EncodedOutput) else "path"
        image_format = output.format
        quality = output.quality
        compression = output.compression
    return _NativePipeline(
        specs,
        seed,
        mode,
        source_kind,
        max_pixels,
        max_encoded_bytes,
        sink_kind,
        image_format,
        quality,
        compression,
    )
