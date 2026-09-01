from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from ._validation import _image, _key, _seed, _torch_module
from ._variopinta import Pipeline as _NativePipeline
from .transforms import _TRANSFORM_TYPES, ToTorch, Transform

if TYPE_CHECKING:
    import torch

    _PipelineOutput: TypeAlias = np.ndarray | torch.Tensor
else:
    _PipelineOutput: TypeAlias = object


class Compose:
    __slots__ = ("_pipeline", "_seed", "_specs", "_transforms")

    def __init__(self, transforms: Sequence[Transform], seed: int | None = None) -> None:
        self._transforms = tuple(transforms)
        if not all(isinstance(transform, _TRANSFORM_TYPES) for transform in self._transforms):
            raise TypeError("Compose only accepts built-in transforms")
        self._seed = _seed(seed)
        self._specs = [transform._spec() for transform in self._transforms]
        self._pipeline = _NativePipeline(self._specs, self._seed, "reference")

    @property
    def transforms(self) -> tuple[Transform, ...]:
        return self._transforms

    @property
    def seed(self) -> int:
        return self._seed

    def __call__(self, image: np.ndarray, *, key: int | None = None) -> _PipelineOutput:
        torch = (
            _torch_module()
            if self._transforms and isinstance(self._transforms[-1], ToTorch)
            else None
        )
        output = self._pipeline.apply(_image(image), _key(key))
        return output if torch is None else torch.from_numpy(output)

    def compile(self) -> CompiledCompose:
        return CompiledCompose(self._transforms, self._specs, self._seed)

    def explain(self) -> dict[str, object]:
        return self._pipeline.explain()


class CompiledCompose:
    __slots__ = ("_pipeline", "_seed", "_transforms")

    def __init__(
        self,
        transforms: tuple[Transform, ...],
        specs: list[dict[str, object]],
        seed: int,
    ) -> None:
        self._transforms = transforms
        self._seed = seed
        self._pipeline = _NativePipeline(specs, seed, "compiled")

    @property
    def transforms(self) -> tuple[Transform, ...]:
        return self._transforms

    @property
    def seed(self) -> int:
        return self._seed

    def __call__(self, image: np.ndarray, *, key: int | None = None) -> _PipelineOutput:
        torch = (
            _torch_module()
            if self._transforms and isinstance(self._transforms[-1], ToTorch)
            else None
        )
        output = self._pipeline.apply(_image(image), _key(key))
        return output if torch is None else torch.from_numpy(output)

    def explain(self) -> dict[str, object]:
        return self._pipeline.explain()
