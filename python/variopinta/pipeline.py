from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, TypeVar, overload

import numpy as np

from ._validation import _key, _seed, _torch_module
from ._variopinta import Pipeline as _NativePipeline
from .targets import (
    BoundTarget,
    Image,
    Mask,
    OutputPort,
    ReturnTensor,
    Target,
    Write,
    _route,
)
from .transforms import _TRANSFORM_TYPES, Transform

_Result: TypeAlias = object
_Value = TypeVar("_Value")


class TargetResult:
    __slots__ = ("_by_identity", "_names", "_outputs", "_values", "_locked")

    def __init__(self, outputs: tuple[OutputPort[object], ...], values: tuple[object, ...]) -> None:
        if len(outputs) != len(values):
            raise RuntimeError("native output does not match the target signature")
        object.__setattr__(self, "_outputs", outputs)
        object.__setattr__(self, "_values", values)
        object.__setattr__(
            self,
            "_by_identity",
            MappingProxyType(
                {id(port): value for port, value in zip(outputs, values, strict=True)}
            ),
        )
        object.__setattr__(
            self,
            "_names",
            MappingProxyType(
                {port.name: value for port, value in zip(outputs, values, strict=True)}
            ),
        )
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("TargetResult is immutable")
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> object:
        try:
            return self._names[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @overload
    def __getitem__(self, output: OutputPort[_Value]) -> _Value: ...

    @overload
    def __getitem__(self, output: object) -> object: ...

    def __getitem__(self, output: object) -> object:
        if not isinstance(output, OutputPort):
            raise TypeError("TargetResult indices must be output ports")
        try:
            return self._by_identity[id(output)]
        except KeyError as error:
            raise KeyError("output port does not belong to this result") from error

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._names))

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{port.name}={_value_summary(value)}"
            for port, value in zip(self._outputs, self._values, strict=True)
        )
        return f"TargetResult({fields})"


class PipelineResult:
    __slots__ = ("_by_identity", "_names", "_results", "_targets", "_locked")

    def __init__(self, targets: tuple[Target, ...], results: tuple[TargetResult, ...]) -> None:
        if len(targets) != len(results):
            raise RuntimeError("native output does not match the pipeline signature")
        object.__setattr__(self, "_targets", targets)
        object.__setattr__(self, "_results", results)
        object.__setattr__(
            self,
            "_by_identity",
            MappingProxyType(
                {id(target): result for target, result in zip(targets, results, strict=True)}
            ),
        )
        object.__setattr__(
            self,
            "_names",
            MappingProxyType(
                {target.name: result for target, result in zip(targets, results, strict=True)}
            ),
        )
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("PipelineResult is immutable")
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> TargetResult:
        try:
            return self._names[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __getitem__(self, target: Target) -> TargetResult:
        if not isinstance(target, Image | Mask):
            raise TypeError("PipelineResult indices must be target ports")
        try:
            return self._by_identity[id(target)]
        except KeyError as error:
            raise KeyError("target port does not belong to this result") from error

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._names))

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{target.name}={result!r}"
            for target, result in zip(self._targets, self._results, strict=True)
        )
        return f"PipelineResult({fields})"


class Pipeline:
    __slots__ = ("_explicit_targets", "_pipeline", "_seed", "_specs", "_targets", "_transforms")

    def __init__(
        self,
        transforms: Sequence[Transform],
        seed: int | None = None,
        *,
        targets: Sequence[Target] | None = None,
    ) -> None:
        normalized_transforms = tuple(transforms)
        if not all(isinstance(transform, _TRANSFORM_TYPES) for transform in normalized_transforms):
            raise TypeError("Pipeline only accepts built-in transforms")
        explicit_targets = targets is not None
        normalized_targets = (Image(),) if targets is None else tuple(targets)
        if not normalized_targets:
            raise ValueError("targets must contain at least one target port")
        if not all(isinstance(target, Image | Mask) for target in normalized_targets):
            raise TypeError("targets must contain only Image or Mask ports")
        if len({id(target) for target in normalized_targets}) != len(normalized_targets):
            raise ValueError("the same target port cannot appear more than once")
        if explicit_targets:
            _validate_explicit_signature(normalized_targets)

        self._transforms = normalized_transforms
        self._targets = normalized_targets
        self._explicit_targets = explicit_targets
        self._seed = _seed(seed)
        self._specs = [transform._spec() for transform in normalized_transforms]
        self._pipeline = _native_pipeline(self._specs, self._seed, "reference", self._targets)

    @property
    def transforms(self) -> tuple[Transform, ...]:
        return self._transforms

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def targets(self) -> tuple[Target, ...]:
        return self._targets

    def __call__(
        self,
        *values: object,
        key: int | None = None,
        **bindings: BoundTarget[object],
    ) -> _Result:
        normalized = _normalize_call(self, values, bindings)
        torch = _torch_for_presentation(self)
        output = self._pipeline.apply_targets(normalized, _key(key))
        return _present(self, normalized, output, torch)

    def compile(self) -> CompiledPipeline:
        return CompiledPipeline(
            self._transforms,
            self._specs,
            self._seed,
            self._targets,
            self._explicit_targets,
        )

    def explain(self) -> dict[str, object]:
        return self._pipeline.explain()


class CompiledPipeline:
    __slots__ = ("_explicit_targets", "_pipeline", "_seed", "_targets", "_transforms")

    def __init__(
        self,
        transforms: tuple[Transform, ...],
        specs: list[dict[str, object]],
        seed: int,
        targets: tuple[Target, ...],
        explicit_targets: bool,
    ) -> None:
        self._transforms = transforms
        self._seed = seed
        self._targets = targets
        self._explicit_targets = explicit_targets
        self._pipeline = _native_pipeline(specs, seed, "compiled", targets)

    @property
    def transforms(self) -> tuple[Transform, ...]:
        return self._transforms

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def targets(self) -> tuple[Target, ...]:
        return self._targets

    def __call__(
        self,
        *values: object,
        key: int | None = None,
        **bindings: BoundTarget[object],
    ) -> _Result:
        normalized = _normalize_call(self, values, bindings)
        torch = _torch_for_presentation(self)
        output = self._pipeline.apply_targets(normalized, _key(key))
        return _present(self, normalized, output, torch)

    def explain(self) -> dict[str, object]:
        return self._pipeline.explain()


def _validate_explicit_signature(targets: tuple[Target, ...]) -> None:
    missing_targets = [index for index, target in enumerate(targets) if target.name is None]
    if missing_targets:
        raise ValueError("every explicit target must have a name")
    names = [target.name for target in targets]
    if len(set(names)) != len(names):
        raise ValueError("target names must be unique within a pipeline")
    for target in targets:
        if any(output.name is None for output in target.outputs):
            raise ValueError(f"every output of target {target.name!r} must have a name")


def _normalize_call(
    pipeline: Pipeline | CompiledPipeline,
    values: tuple[object, ...],
    bindings: dict[str, BoundTarget[object]],
) -> tuple[BoundTarget[object], ...] | tuple[np.ndarray]:
    if not pipeline._explicit_targets:
        if bindings or len(values) != 1 or not isinstance(values[0], np.ndarray):
            raise TypeError("the implicit image target requires exactly one positional NumPy array")
        return (values[0],)

    if values:
        raise TypeError("explicit target signatures accept bindings only by keyword")
    expected = {target.name for target in pipeline.targets}
    actual = set(bindings)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise TypeError(
            "target bindings do not match the pipeline signature (" + "; ".join(details) + ")"
        )
    ordered = tuple(bindings[target.name] for target in pipeline.targets)
    if not all(isinstance(binding, BoundTarget) for binding in ordered):
        raise TypeError("explicit target signatures accept only BoundTarget values")
    for target, binding in zip(pipeline.targets, ordered, strict=True):
        if binding.target is not target:
            raise ValueError(f"binding for target {target.name!r} belongs to a different port")
    return ordered


def _present(
    pipeline: Pipeline | CompiledPipeline,
    bindings: tuple[BoundTarget[object], ...] | tuple[np.ndarray],
    output: object,
    torch: Any | None,
) -> object:
    if not pipeline._explicit_targets:
        if not isinstance(output, tuple) or len(output) != 1:
            raise RuntimeError("native target output does not match the implicit signature")
        target_values = output[0]
        if not isinstance(target_values, tuple) or len(target_values) != 1:
            raise RuntimeError("native output does not match the implicit target")
        return target_values[0]

    if not isinstance(output, tuple) or len(output) != len(pipeline.targets):
        raise RuntimeError("native target output does not match the pipeline signature")
    target_results: list[TargetResult] = []
    for target, binding, values in zip(pipeline.targets, bindings, output, strict=True):
        if not isinstance(binding, BoundTarget) or not isinstance(values, tuple):
            raise RuntimeError("native target output does not match the pipeline signature")
        if len(values) != len(target.outputs):
            raise RuntimeError("native output does not match the target signature")
        destinations = {id(item.output): item.destination for item in binding._write_bindings}
        presented: list[object] = []
        for port, value in zip(target.outputs, values, strict=True):
            if isinstance(port, ReturnTensor):
                value = torch.from_numpy(value)
            elif isinstance(port, Write):
                value = destinations[id(port)]
            presented.append(value)
        target_results.append(TargetResult(target.outputs, tuple(presented)))
    return PipelineResult(pipeline.targets, tuple(target_results))


def _torch_for_presentation(pipeline: Pipeline | CompiledPipeline) -> Any | None:
    return (
        _torch_module()
        if any(
            isinstance(output, ReturnTensor)
            for target in pipeline.targets
            for output in target.outputs
        )
        else None
    )


def _native_pipeline(
    specs: list[dict[str, object]],
    seed: int,
    mode: str,
    targets: tuple[Target, ...],
) -> _NativePipeline:
    return _NativePipeline(specs, seed, mode, [_route(target) for target in targets])


def _value_summary(value: object) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return f"{type(value).__name__}(shape={tuple(shape)!r}, dtype={dtype})"
    if isinstance(value, bytes):
        return f"bytes(len={len(value)})"
    if isinstance(value, Path):
        return f"{type(value).__name__}(...)"
    return repr(value)
