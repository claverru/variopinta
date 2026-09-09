from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TypeAlias

from ._validation import (
    _affine_degrees,
    _affine_scale,
    _affine_shear,
    _affine_translate,
    _channel_stats,
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
    interpolation: Interpolation = field(default=Interpolation.BILINEAR, kw_only=True)
    antialias: bool = field(default=False, kw_only=True)
    p: float = field(default=1.0, kw_only=True)

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
class LongestMaxSize:
    max_size: int
    interpolation: Interpolation = field(default=Interpolation.BILINEAR, kw_only=True)
    antialias: bool = field(default=False, kw_only=True)
    p: float = field(default=1.0, kw_only=True)

    def __post_init__(self) -> None:
        _positive_integer("max_size", self.max_size)
        object.__setattr__(self, "p", _probability(self.p))
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation value")
        if not isinstance(self.antialias, bool):
            raise TypeError("antialias must be a bool")

    def _spec(self) -> dict[str, object]:
        return {
            "type": "LongestMaxSize",
            "max_size": self.max_size,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "antialias": self.antialias,
        }


@dataclass(frozen=True, slots=True)
class RandomCrop:
    height: int
    width: int
    p: float = field(default=1.0, kw_only=True)

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
    area_range: tuple[float, float] = field(default=(0.08, 1.0), kw_only=True)
    aspect_ratio_range: tuple[float, float] = field(default=(0.75, 4.0 / 3.0), kw_only=True)
    interpolation: Interpolation = field(default=Interpolation.BILINEAR, kw_only=True)
    antialias: bool = field(default=False, kw_only=True)
    p: float = field(default=1.0, kw_only=True)

    def __post_init__(self) -> None:
        _positive_integer("height", self.height)
        _positive_integer("width", self.width)
        object.__setattr__(
            self,
            "area_range",
            _positive_range("area_range", self.area_range, maximum=1.0),
        )
        object.__setattr__(
            self,
            "aspect_ratio_range",
            _positive_range("aspect_ratio_range", self.aspect_ratio_range),
        )
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
            "scale": self.area_range,
            "ratio": self.aspect_ratio_range,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "antialias": self.antialias,
        }


@dataclass(frozen=True, slots=True)
class CenterCrop:
    height: int
    width: int
    p: float = field(default=1.0, kw_only=True)

    def __post_init__(self) -> None:
        _positive_integer("height", self.height)
        _positive_integer("width", self.width)
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "CenterCrop", "height": self.height, "width": self.width, "p": self.p}


@dataclass(frozen=True, slots=True, kw_only=True)
class PadIfNeeded:
    min_height: int | None = None
    min_width: int | None = None
    height_divisor: int | None = None
    width_divisor: int | None = None
    position: PadPosition = PadPosition.CENTER
    border_mode: BorderMode = BorderMode.CONSTANT
    fill: int | Sequence[int] = 0
    p: float = 1.0

    def __post_init__(self) -> None:
        for axis, minimum, divisor in (
            ("height", self.min_height, self.height_divisor),
            ("width", self.min_width, self.width_divisor),
        ):
            if (minimum is None) == (divisor is None):
                raise ValueError(f"exactly one of min_{axis} and {axis}_divisor is required")
            if minimum is not None:
                _positive_integer(f"min_{axis}", minimum)
            elif divisor is not None:
                _positive_integer(f"{axis}_divisor", divisor)
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
            "pad_height_divisor": self.height_divisor,
            "pad_width_divisor": self.width_divisor,
            "position": self.position.value,
            "border_mode": self.border_mode.value,
            "fill": self.fill,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CoarseDropout:
    num_holes_range: tuple[int, int] = (1, 2)
    hole_height_range: tuple[int | float, int | float] = (0.1, 0.2)
    hole_height_unit: Literal["pixels", "fraction"] = "fraction"
    hole_width_range: tuple[int | float, int | float] = (0.1, 0.2)
    hole_width_unit: Literal["pixels", "fraction"] = "fraction"
    fill: int | Sequence[int] = 0
    p: float = 0.5

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
        height_range = _dropout_size_range(
            "hole_height_range", self.hole_height_range, self.hole_height_unit
        )
        width_range = _dropout_size_range(
            "hole_width_range", self.hole_width_range, self.hole_width_unit
        )
        object.__setattr__(self, "hole_height_range", height_range)
        object.__setattr__(self, "hole_width_range", width_range)
        object.__setattr__(self, "fill", _fill(self.fill))
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "CoarseDropout",
            "num_holes_range": self.num_holes_range,
            "hole_height_range": self.hole_height_range,
            "hole_height_unit": self.hole_height_unit,
            "hole_width_range": self.hole_width_range,
            "hole_width_unit": self.hole_width_unit,
            "fill": self.fill,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class HorizontalFlip:
    p: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "HorizontalFlip", "p": self.p}


@dataclass(frozen=True, slots=True, kw_only=True)
class VerticalFlip:
    p: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "VerticalFlip", "p": self.p}


@dataclass(frozen=True, slots=True, kw_only=True)
class ColorJitter:
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.8, 1.2)
    hue_range: tuple[float, float] = (0.0, 0.0)
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "brightness_range",
            _color_factor_range("brightness_range", self.brightness_range),
        )
        object.__setattr__(
            self,
            "contrast_range",
            _color_factor_range("contrast_range", self.contrast_range),
        )
        object.__setattr__(
            self,
            "saturation_range",
            _color_factor_range("saturation_range", self.saturation_range),
        )
        object.__setattr__(self, "hue_range", _hue_range(self.hue_range))
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "ColorJitter",
            "brightness": self.brightness_range,
            "contrast": self.contrast_range,
            "saturation": self.saturation_range,
            "hue": self.hue_range,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Affine:
    degrees_range: tuple[float, float] = (-10.0, 10.0)
    translate_max_fraction: tuple[float, float] = (0.0, 0.0)
    scale_range: tuple[float, float] = (1.0, 1.0)
    shear_x_range: tuple[float, float] = (0.0, 0.0)
    shear_y_range: tuple[float, float] = (0.0, 0.0)
    interpolation: Interpolation = Interpolation.BILINEAR
    border_mode: BorderMode = BorderMode.CONSTANT
    fill: int | Sequence[int] = 0
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "degrees_range", _affine_degrees(self.degrees_range))
        object.__setattr__(
            self,
            "translate_max_fraction",
            _affine_translate(self.translate_max_fraction),
        )
        object.__setattr__(self, "scale_range", _affine_scale(self.scale_range))
        object.__setattr__(
            self,
            "shear_x_range",
            _affine_shear("shear_x_range", self.shear_x_range),
        )
        object.__setattr__(
            self,
            "shear_y_range",
            _affine_shear("shear_y_range", self.shear_y_range),
        )
        object.__setattr__(self, "p", _probability(self.p))
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation value")
        if not isinstance(self.border_mode, BorderMode):
            raise TypeError("border_mode must be a BorderMode value")
        object.__setattr__(self, "fill", _fill(self.fill))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "Affine",
            "degrees": self.degrees_range,
            "translate": self.translate_max_fraction,
            "scale": self.scale_range,
            "shear": (*self.shear_x_range, *self.shear_y_range),
            "p": self.p,
            "interpolation": self.interpolation.value,
            "border_mode": self.border_mode.value,
            "fill": self.fill,
        }


@dataclass(frozen=True, slots=True)
class RandomRotation:
    degrees_range: tuple[float, float] = (-10.0, 10.0)
    interpolation: Interpolation = field(default=Interpolation.BILINEAR, kw_only=True)
    border_mode: BorderMode = field(default=BorderMode.CONSTANT, kw_only=True)
    fill: int | Sequence[int] = field(default=0, kw_only=True)
    p: float = field(default=1.0, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "degrees_range", _affine_degrees(self.degrees_range))
        object.__setattr__(self, "p", _probability(self.p))
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation value")
        if not isinstance(self.border_mode, BorderMode):
            raise TypeError("border_mode must be a BorderMode value")
        object.__setattr__(self, "fill", _fill(self.fill))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "RandomRotation",
            "degrees": self.degrees_range,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "border_mode": self.border_mode.value,
            "fill": self.fill,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class GaussianNoise:
    mean_range: tuple[float, float] = (0.0, 0.0)
    std_range: tuple[float, float] = (10.0, 10.0)
    per_channel: bool = True
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean_range", _value_range("mean_range", self.mean_range))
        object.__setattr__(
            self,
            "std_range",
            _value_range("std_range", self.std_range, non_negative=True),
        )
        if not isinstance(self.per_channel, bool):
            raise TypeError("per_channel must be a bool")
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "GaussianNoise",
            "mean": self.mean_range,
            "std": self.std_range,
            "per_channel": self.per_channel,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Sharpen:
    blend_weight_range: tuple[float, float] = (0.5, 0.5)
    strength_range: tuple[float, float] = (1.0, 1.0)
    p: float = 1.0

    def __post_init__(self) -> None:
        blend_weight_range = _value_range(
            "blend_weight_range", self.blend_weight_range, non_negative=True
        )
        if blend_weight_range[1] > 1.0:
            raise ValueError("blend_weight_range values must be in [0, 1]")
        object.__setattr__(self, "blend_weight_range", blend_weight_range)
        object.__setattr__(
            self,
            "strength_range",
            _value_range("strength_range", self.strength_range, non_negative=True),
        )
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "Sharpen",
            "alpha": self.blend_weight_range,
            "lightness": self.strength_range,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Perspective:
    distortion_scale_range: tuple[float, float] = (0.05, 0.05)
    interpolation: Interpolation = Interpolation.BILINEAR
    border_mode: BorderMode = BorderMode.CONSTANT
    fill: int | Sequence[int] = 0
    p: float = 1.0

    def __post_init__(self) -> None:
        distortion_scale_range = _value_range(
            "distortion_scale_range", self.distortion_scale_range, non_negative=True
        )
        if distortion_scale_range[1] >= 0.5:
            raise ValueError("distortion_scale_range values must be in [0, 0.5)")
        object.__setattr__(self, "distortion_scale_range", distortion_scale_range)
        object.__setattr__(self, "p", _probability(self.p))
        if not isinstance(self.interpolation, Interpolation):
            raise TypeError("interpolation must be an Interpolation value")
        if not isinstance(self.border_mode, BorderMode):
            raise TypeError("border_mode must be a BorderMode value")
        object.__setattr__(self, "fill", _fill(self.fill))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "Perspective",
            "scale": self.distortion_scale_range,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "border_mode": self.border_mode.value,
            "fill": self.fill,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class GridDistortion:
    num_steps: int = 5
    distortion_range: tuple[float, float] = (-0.3, 0.3)
    interpolation: Interpolation = Interpolation.BILINEAR
    border_mode: BorderMode = BorderMode.CONSTANT
    fill: int | Sequence[int] = 0
    p: float = 1.0

    def __post_init__(self) -> None:
        _positive_integer("num_steps", self.num_steps)
        object.__setattr__(
            self,
            "distortion_range",
            _symmetric_limit_range("distortion_range", self.distortion_range, maximum=1.0),
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
            "distort_limit": self.distortion_range,
            "p": self.p,
            "interpolation": self.interpolation.value,
            "border_mode": self.border_mode.value,
            "fill": self.fill,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class GaussianBlur:
    kernel_size: int = 5
    sigma_range: tuple[float, float] = (1.1, 1.1)
    p: float = 1.0

    def __post_init__(self) -> None:
        _positive_integer("kernel_size", self.kernel_size)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        object.__setattr__(self, "sigma_range", _sigma_range(self.sigma_range))
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {
            "type": "GaussianBlur",
            "kernel_size": self.kernel_size,
            "sigma": self.sigma_range,
            "p": self.p,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Grayscale:
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "Grayscale", "p": self.p}


@dataclass(frozen=True, slots=True, kw_only=True)
class Invert:
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "Invert", "p": self.p}


@dataclass(frozen=True, slots=True)
class Solarize:
    threshold: int = 128
    p: float = field(default=1.0, kw_only=True)

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
    p: float = field(default=1.0, kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or not 1 <= self.bits <= 8:
            raise ValueError("bits must be an integer in [1, 8]")
        object.__setattr__(self, "p", _probability(self.p))

    def _spec(self) -> dict[str, object]:
        return {"type": "Posterize", "bits": self.bits, "p": self.p}


@dataclass(frozen=True, slots=True, kw_only=True)
class Normalize:
    mean: float | Sequence[float] = (0.485, 0.456, 0.406)
    std: float | Sequence[float] = (0.229, 0.224, 0.225)
    max_pixel_value: float = 255.0
    p: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _channel_stats("mean", self.mean, positive=False))
        object.__setattr__(self, "std", _channel_stats("std", self.std, positive=True))
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


Transform: TypeAlias = (
    Resize
    | LongestMaxSize
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
)
_TRANSFORM_CATALOG = {
    "Resize": (Resize, lambda p: Resize(1, 1, p=p)),
    "LongestMaxSize": (LongestMaxSize, lambda p: LongestMaxSize(1, p=p)),
    "RandomCrop": (RandomCrop, lambda p: RandomCrop(1, 1, p=p)),
    "RandomResizedCrop": (
        RandomResizedCrop,
        lambda p: RandomResizedCrop(1, 1, p=p),
    ),
    "HorizontalFlip": (HorizontalFlip, lambda p: HorizontalFlip(p=p)),
    "VerticalFlip": (VerticalFlip, lambda p: VerticalFlip(p=p)),
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
    "Grayscale": (Grayscale, lambda p: Grayscale(p=p)),
    "Invert": (Invert, lambda p: Invert(p=p)),
    "Solarize": (Solarize, lambda p: Solarize(p=p)),
    "Posterize": (Posterize, lambda p: Posterize(p=p)),
    "Normalize": (Normalize, lambda p: Normalize(p=p)),
}
_TRANSFORM_TYPES = tuple(entry[0] for entry in _TRANSFORM_CATALOG.values())
