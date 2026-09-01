from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from ._validation import (
    _affine_degrees,
    _affine_scale,
    _affine_shear,
    _affine_translate,
    _color_factor_range,
    _dropout_size_range,
    _f32,
    _fill,
    _hue_range,
    _positive_integer,
    _positive_range,
    _probability,
    _sigma_range,
    _symmetric_limit_range,
    _triplet,
    _value_range,
)


class Interpolation(str, Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"


class BorderMode(str, Enum):
    CONSTANT = "constant"
    REFLECT101 = "reflect101"


class PadPosition(str, Enum):
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    RANDOM = "random"


@dataclass(frozen=True, slots=True)
class Resize:
    height: int
    width: int
    p: float = 1.0
    interpolation: Interpolation = Interpolation.BILINEAR
    antialias: bool = False

    def __post_init__(self) -> None:
        _positive_integer("height", self.height)
        _positive_integer("width", self.width)
        object.__setattr__(self, "p", _probability(self.p))
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation value")
        if not isinstance(self.antialias, bool):
            raise TypeError("antialias must be a bool")

    def _spec(self) -> dict[str, object]:
        return {
            "type": "Resize",
            "height": self.height,
            "width": self.width,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "antialias": self.antialias,
        }


@dataclass(frozen=True, slots=True)
class RandomCrop:
    height: int
    width: int
    p: float = 1.0

    def __post_init__(self) -> None:
        _positive_integer("height", self.height)
        _positive_integer("width", self.width)
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "RandomCrop", "height": self.height, "width": self.width, "p": self.p}


@dataclass(frozen=True, slots=True)
class RandomResizedCrop:
    height: int
    width: int
    scale: tuple[float, float] = (0.08, 1.0)
    ratio: tuple[float, float] = (0.75, 4.0 / 3.0)
    p: float = 1.0
    interpolation: Interpolation = Interpolation.BILINEAR
    antialias: bool = False

    def __post_init__(self) -> None:
        _positive_integer("height", self.height)
        _positive_integer("width", self.width)
        object.__setattr__(self, "scale", _positive_range("scale", self.scale, maximum=1.0))
        object.__setattr__(self, "ratio", _positive_range("ratio", self.ratio))
        object.__setattr__(self, "p", _probability(self.p))
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation value")
        if not isinstance(self.antialias, bool):
            raise TypeError("antialias must be a bool")

    def _spec(self) -> dict[str, object]:
        return {
            "type": "RandomResizedCrop",
            "height": self.height,
            "width": self.width,
            "scale": self.scale,
            "ratio": self.ratio,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "antialias": self.antialias,
        }


@dataclass(frozen=True, slots=True)
class CenterCrop:
    height: int
    width: int
    p: float = 1.0

    def __post_init__(self) -> None:
        _positive_integer("height", self.height)
        _positive_integer("width", self.width)
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "CenterCrop", "height": self.height, "width": self.width, "p": self.p}


@dataclass(frozen=True, slots=True)
class PadIfNeeded:
    min_height: int | None = None
    min_width: int | None = None
    pad_height_divisor: int | None = None
    pad_width_divisor: int | None = None
    position: PadPosition = PadPosition.CENTER
    p: float = 1.0
    border_mode: BorderMode = BorderMode.CONSTANT
    fill: int | tuple[int, int, int] = 0

    def __post_init__(self) -> None:
        for axis, minimum, divisor in (
            ("height", self.min_height, self.pad_height_divisor),
            ("width", self.min_width, self.pad_width_divisor),
        ):
            if (minimum is None) == (divisor is None):
                raise ValueError(f"exactly one of min_{axis} and pad_{axis}_divisor is required")
            if minimum is not None:
                _positive_integer(f"min_{axis}", minimum)
            elif divisor is not None:
                _positive_integer(f"pad_{axis}_divisor", divisor)
        if not isinstance(self.position, PadPosition):
            raise TypeError("position must be a PadPosition value")
        if not isinstance(self.border_mode, BorderMode):
            raise TypeError("border_mode must be a BorderMode value")
        object.__setattr__(self, "fill", _fill(self.fill))
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "PadIfNeeded",
            "min_height": self.min_height,
            "min_width": self.min_width,
            "pad_height_divisor": self.pad_height_divisor,
            "pad_width_divisor": self.pad_width_divisor,
            "position": self.position.value,
            "border_mode": self.border_mode.value,
            "fill": self.fill,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True)
class CoarseDropout:
    num_holes_range: tuple[int, int] = (1, 2)
    hole_height_range: tuple[int, int] | tuple[float, float] = (0.1, 0.2)
    hole_width_range: tuple[int, int] | tuple[float, float] = (0.1, 0.2)
    fill: int | tuple[int, int, int] = 0
    p: float = 0.5
    _hole_height_unit: str = field(init=False, repr=False, compare=False, default="fraction")
    _hole_width_unit: str = field(init=False, repr=False, compare=False, default="fraction")

    def __post_init__(self) -> None:
        if (
            not isinstance(self.num_holes_range, tuple)
            or len(self.num_holes_range) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.num_holes_range
            )
            or self.num_holes_range[0] > self.num_holes_range[1]
        ):
            raise ValueError("num_holes_range must contain two ordered positive integers")
        height_range, height_unit = _dropout_size_range("hole_height_range", self.hole_height_range)
        width_range, width_unit = _dropout_size_range("hole_width_range", self.hole_width_range)
        object.__setattr__(self, "hole_height_range", height_range)
        object.__setattr__(self, "hole_width_range", width_range)
        object.__setattr__(self, "_hole_height_unit", height_unit)
        object.__setattr__(self, "_hole_width_unit", width_unit)
        object.__setattr__(self, "fill", _fill(self.fill))
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "CoarseDropout",
            "num_holes_range": self.num_holes_range,
            "hole_height_range": self.hole_height_range,
            "hole_height_unit": self._hole_height_unit,
            "hole_width_range": self.hole_width_range,
            "hole_width_unit": self._hole_width_unit,
            "fill": self.fill,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True)
class HorizontalFlip:
    p: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "HorizontalFlip", "p": self.p}


@dataclass(frozen=True, slots=True)
class VerticalFlip:
    p: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "VerticalFlip", "p": self.p}


@dataclass(frozen=True, slots=True)
class ColorJitter:
    brightness: float | tuple[float, float] = 0.2
    contrast: float | tuple[float, float] = 0.2
    saturation: float | tuple[float, float] = 0.2
    hue: float | tuple[float, float] = 0.0
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "brightness", _color_factor_range("brightness", self.brightness))
        object.__setattr__(self, "contrast", _color_factor_range("contrast", self.contrast))
        object.__setattr__(self, "saturation", _color_factor_range("saturation", self.saturation))
        object.__setattr__(self, "hue", _hue_range(self.hue))
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "ColorJitter",
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "hue": self.hue,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True)
class Affine:
    degrees: float | tuple[float, float] = 10.0
    translate: tuple[float, float] = (0.0, 0.0)
    scale: float | tuple[float, float] = 1.0
    shear: float | tuple[float, float] | tuple[float, float, float, float] = 0.0
    p: float = 1.0
    interpolation: Interpolation = Interpolation.BILINEAR
    border_mode: BorderMode = BorderMode.CONSTANT
    fill: int | tuple[int, int, int] = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "degrees", _affine_degrees(self.degrees))
        object.__setattr__(self, "translate", _affine_translate(self.translate))
        object.__setattr__(self, "scale", _affine_scale(self.scale))
        object.__setattr__(self, "shear", _affine_shear(self.shear))
        object.__setattr__(self, "p", _probability(self.p))
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation value")
        if not isinstance(self.border_mode, BorderMode):
            raise TypeError("border_mode must be a BorderMode value")
        object.__setattr__(self, "fill", _fill(self.fill))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "Affine",
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "border_mode": self.border_mode.value,
            "fill": self.fill,
        }


@dataclass(frozen=True, slots=True)
class RandomRotation:
    degrees: float | tuple[float, float] = 10.0
    p: float = 1.0
    interpolation: Interpolation = Interpolation.BILINEAR
    border_mode: BorderMode = BorderMode.CONSTANT
    fill: int | tuple[int, int, int] = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "degrees", _affine_degrees(self.degrees))
        object.__setattr__(self, "p", _probability(self.p))
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation value")
        if not isinstance(self.border_mode, BorderMode):
            raise TypeError("border_mode must be a BorderMode value")
        object.__setattr__(self, "fill", _fill(self.fill))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "RandomRotation",
            "degrees": self.degrees,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "border_mode": self.border_mode.value,
            "fill": self.fill,
        }


@dataclass(frozen=True, slots=True)
class GaussianNoise:
    mean: float | tuple[float, float] = 0.0
    std: float | tuple[float, float] = 10.0
    per_channel: bool = True
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _value_range("mean", self.mean))
        object.__setattr__(self, "std", _value_range("std", self.std, non_negative=True))
        if not isinstance(self.per_channel, bool):
            raise TypeError("per_channel must be a bool")
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "GaussianNoise",
            "mean": self.mean,
            "std": self.std,
            "per_channel": self.per_channel,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True)
class Sharpen:
    alpha: float | tuple[float, float] = 0.5
    lightness: float | tuple[float, float] = 1.0
    p: float = 1.0

    def __post_init__(self) -> None:
        alpha = _value_range("alpha", self.alpha, non_negative=True)
        if alpha[1] > 1.0:
            raise ValueError("alpha values must be in [0, 1]")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(
            self, "lightness", _value_range("lightness", self.lightness, non_negative=True)
        )
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "Sharpen",
            "alpha": self.alpha,
            "lightness": self.lightness,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True)
class Perspective:
    scale: float | tuple[float, float] = 0.05
    p: float = 1.0
    interpolation: Interpolation = Interpolation.BILINEAR
    border_mode: BorderMode = BorderMode.CONSTANT
    fill: int | tuple[int, int, int] = 0

    def __post_init__(self) -> None:
        scale = _value_range("scale", self.scale, non_negative=True)
        if scale[1] >= 0.5:
            raise ValueError("scale values must be in [0, 0.5)")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "p", _probability(self.p))
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation value")
        if not isinstance(self.border_mode, BorderMode):
            raise TypeError("border_mode must be a BorderMode value")
        object.__setattr__(self, "fill", _fill(self.fill))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "Perspective",
            "scale": self.scale,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "border_mode": self.border_mode.value,
            "fill": self.fill,
        }


@dataclass(frozen=True, slots=True)
class GridDistortion:
    num_steps: int = 5
    distort_limit: float | tuple[float, float] = 0.3
    p: float = 1.0
    interpolation: Interpolation = Interpolation.BILINEAR
    border_mode: BorderMode = BorderMode.CONSTANT
    fill: int | tuple[int, int, int] = 0

    def __post_init__(self) -> None:
        _positive_integer("num_steps", self.num_steps)
        object.__setattr__(
            self,
            "distort_limit",
            _symmetric_limit_range("distort_limit", self.distort_limit, maximum=1.0),
        )
        object.__setattr__(self, "p", _probability(self.p))
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation value")
        if not isinstance(self.border_mode, BorderMode):
            raise TypeError("border_mode must be a BorderMode value")
        object.__setattr__(self, "fill", _fill(self.fill))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "GridDistortion",
            "num_steps": self.num_steps,
            "distort_limit": self.distort_limit,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "border_mode": self.border_mode.value,
            "fill": self.fill,
        }


@dataclass(frozen=True, slots=True)
class GaussianBlur:
    kernel_size: int = 5
    sigma: float | tuple[float, float] = 1.1
    p: float = 1.0

    def __post_init__(self) -> None:
        _positive_integer("kernel_size", self.kernel_size)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        object.__setattr__(self, "sigma", _sigma_range(self.sigma))
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "GaussianBlur",
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True)
class Grayscale:
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "Grayscale", "p": self.p}


@dataclass(frozen=True, slots=True)
class Invert:
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "Invert", "p": self.p}


@dataclass(frozen=True, slots=True)
class Solarize:
    threshold: int = 128
    p: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, int)
            or not 0 <= self.threshold <= 255
        ):
            raise ValueError("threshold must be an integer in [0, 255]")
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "Solarize", "threshold": self.threshold, "p": self.p}


@dataclass(frozen=True, slots=True)
class Posterize:
    bits: int = 4
    p: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or not 1 <= self.bits <= 8:
            raise ValueError("bits must be an integer in [1, 8]")
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "Posterize", "bits": self.bits, "p": self.p}


@dataclass(frozen=True, slots=True)
class Normalize:
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    max_pixel_value: float = 255.0
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _triplet("mean", self.mean, positive=False))
        object.__setattr__(self, "std", _triplet("std", self.std, positive=True))
        max_pixel_value = _f32("max_pixel_value", self.max_pixel_value)
        if max_pixel_value <= 0.0:
            raise ValueError("max_pixel_value must be finite and positive")
        object.__setattr__(self, "max_pixel_value", max_pixel_value)
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "Normalize",
            "mean": self.mean,
            "std": self.std,
            "max_pixel_value": self.max_pixel_value,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True)
class ToTorch:
    def _spec(self) -> dict[str, object]:
        return {"type": "ToTorch"}


Transform: TypeAlias = (
    Resize
    | RandomCrop
    | RandomResizedCrop
    | CenterCrop
    | PadIfNeeded
    | CoarseDropout
    | HorizontalFlip
    | VerticalFlip
    | ColorJitter
    | Affine
    | RandomRotation
    | GaussianNoise
    | Sharpen
    | Perspective
    | GridDistortion
    | GaussianBlur
    | Grayscale
    | Invert
    | Solarize
    | Posterize
    | Normalize
    | ToTorch
)
_TRANSFORM_CATALOG = {
    "Resize": (Resize, lambda p: Resize(1, 1, p=p)),
    "RandomCrop": (RandomCrop, lambda p: RandomCrop(1, 1, p=p)),
    "RandomResizedCrop": (
        RandomResizedCrop,
        lambda p: RandomResizedCrop(1, 1, p=p),
    ),
    "HorizontalFlip": (HorizontalFlip, lambda p: HorizontalFlip(p)),
    "VerticalFlip": (VerticalFlip, lambda p: VerticalFlip(p)),
    "CenterCrop": (CenterCrop, lambda p: CenterCrop(1, 1, p=p)),
    "PadIfNeeded": (
        PadIfNeeded,
        lambda p: PadIfNeeded(min_height=1, min_width=1, p=p),
    ),
    "CoarseDropout": (CoarseDropout, lambda p: CoarseDropout(p=p)),
    "ColorJitter": (ColorJitter, lambda p: ColorJitter(p=p)),
    "Affine": (Affine, lambda p: Affine(p=p)),
    "RandomRotation": (RandomRotation, lambda p: RandomRotation(p=p)),
    "GaussianNoise": (GaussianNoise, lambda p: GaussianNoise(p=p)),
    "Sharpen": (Sharpen, lambda p: Sharpen(p=p)),
    "Perspective": (Perspective, lambda p: Perspective(p=p)),
    "GridDistortion": (GridDistortion, lambda p: GridDistortion(p=p)),
    "GaussianBlur": (GaussianBlur, lambda p: GaussianBlur(p=p)),
    "Grayscale": (Grayscale, lambda p: Grayscale(p)),
    "Invert": (Invert, lambda p: Invert(p)),
    "Solarize": (Solarize, lambda p: Solarize(p=p)),
    "Posterize": (Posterize, lambda p: Posterize(p=p)),
    "Normalize": (Normalize, lambda p: Normalize(p=p)),
    "ToTorch": (ToTorch, lambda _p: ToTorch()),
}
_TRANSFORM_TYPES = tuple(entry[0] for entry in _TRANSFORM_CATALOG.values())
