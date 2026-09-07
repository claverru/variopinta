from __future__ import annotations

import math
import secrets
import struct

import numpy as np


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _f32(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    try:
        result = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except OverflowError as error:
        raise ValueError(f"{name} exceeds the finite float32 range") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds the finite float32 range")
    return 0.0 if result == 0.0 else result


def _probability(value: float) -> float:
    value = _f32("p", value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("p must be in [0, 1]")
    return value


def _non_negative(name: str, value: float) -> float:
    value = _f32(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _positive_range(
    name: str, values: tuple[float, float], *, maximum: float | None = None
) -> tuple[float, float]:
    if (
        not isinstance(values, tuple)
        or len(values) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            for value in values
        )
    ):
        raise ValueError(f"{name} must contain two finite positive values")
    result = (_f32(name, values[0]), _f32(name, values[1]))
    if result[0] <= 0.0:
        raise ValueError(f"{name} must contain two finite positive values")
    if result[0] > result[1]:
        raise ValueError(f"{name} must be ordered")
    if maximum is not None and result[1] > maximum:
        raise ValueError(f"{name} values must be at most {maximum}")
    return result


def _channel_stats(name: str, values: object, *, positive: bool) -> tuple[float, ...]:
    from collections.abc import Sequence

    if isinstance(values, int | float) and not isinstance(values, bool):
        values = (values,)
    if (
        isinstance(values, str | bytes)
        or not isinstance(values, Sequence)
        or len(values) not in (1, 3)
    ):
        raise ValueError(f"{name} must be a scalar or a one-/three-element numeric sequence")
    result = tuple(_f32(name, value) for value in values)
    if positive and any(value <= 0.0 for value in result):
        raise ValueError(f"{name} values must be finite and positive")
    return result


def _fill(value: object) -> tuple[int, ...]:
    from collections.abc import Sequence

    values = (value,) if isinstance(value, int) and not isinstance(value, bool) else value
    if (
        isinstance(values, str | bytes)
        or not isinstance(values, Sequence)
        or len(values) not in (1, 3)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
            for item in values
        )
    ):
        raise ValueError(
            "fill must be an integer or a one-/three-element integer sequence in [0, 255]"
        )
    return tuple(values)


def _dropout_size_range(
    name: str,
    values: tuple[int | float, int | float],
    unit: str,
) -> tuple[int, int] | tuple[float, float]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    if unit not in ("pixels", "fraction"):
        raise ValueError(f"{name.removesuffix('_range')}_unit must be 'pixels' or 'fraction'")
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        raise TypeError(f"{name} must contain two integers or floats")
    if unit == "pixels":
        if any(not math.isfinite(value) or not float(value).is_integer() for value in values):
            raise ValueError(f"{name} pixel values must be finite whole numbers")
        result = (int(values[0]), int(values[1]))
        if result[0] <= 0 or result[0] > result[1]:
            raise ValueError(f"{name} pixel values must be ordered and positive")
        return result
    result = (_f32(name, values[0]), _f32(name, values[1]))
    if any(not 0.0 < value <= 1.0 for value in result) or result[0] > result[1]:
        raise ValueError(f"{name} fraction values must be ordered and in (0, 1]")
    return result


def _finite_pair(name: str, value: object) -> tuple[float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item)
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain two finite values")
    result = (_f32(name, value[0]), _f32(name, value[1]))
    if result[0] > result[1]:
        raise ValueError(f"{name} must be ordered")
    return result


def _value_range(
    name: str, value: tuple[float, float], *, non_negative: bool = False
) -> tuple[float, float]:
    result = _finite_pair(name, value)
    if non_negative and result[0] < 0.0:
        raise ValueError(f"{name} values must be non-negative")
    return result


def _symmetric_limit_range(
    name: str, value: tuple[float, float], *, maximum: float
) -> tuple[float, float]:
    result = _finite_pair(name, value)
    if result[0] <= -maximum or result[1] >= maximum:
        raise ValueError(f"{name} values must be strictly within (-{maximum}, {maximum})")
    return result


def _affine_degrees(value: tuple[float, float]) -> tuple[float, float]:
    return _finite_pair("degrees_range", value)


def _affine_translate(value: tuple[float, float]) -> tuple[float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item)
            for item in value
        )
    ):
        raise ValueError("translate_max_fraction must contain two finite values")
    result = (
        _f32("translate_max_fraction", value[0]),
        _f32("translate_max_fraction", value[1]),
    )
    if any(not 0.0 <= item <= 1.0 for item in result):
        raise ValueError("translate_max_fraction values must be in [0, 1]")
    return result


def _affine_scale(value: tuple[float, float]) -> tuple[float, float]:
    result = _finite_pair("scale_range", value)
    if result[0] <= 0.0:
        raise ValueError("scale_range values must be positive")
    return result


def _affine_shear(name: str, value: tuple[float, float]) -> tuple[float, float]:
    result = _finite_pair(name, value)
    if any(abs(item) >= 90.0 for item in result):
        raise ValueError(f"{name} values must be strictly between -90 and 90 degrees")
    return result


def _color_factor_range(name: str, value: tuple[float, float]) -> tuple[float, float]:
    result = _finite_pair(name, value)
    if result[0] < 0.0:
        raise ValueError(f"{name} range values must be non-negative")
    return result


def _hue_range(value: tuple[float, float]) -> tuple[float, float]:
    result = _finite_pair("hue_range", value)
    if result[0] < -0.5 or result[1] > 0.5:
        raise ValueError("hue range values must be in [-0.5, 0.5]")
    return result


def _sigma_range(value: tuple[float, float]) -> tuple[float, float]:
    result = _finite_pair("sigma_range", value)
    if result[0] <= 0.0:
        raise ValueError("sigma_range values must be positive")
    return result


def _seed(value: int | None) -> int:
    value = secrets.randbits(64) if value is None else value
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return value


def _image(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise TypeError("expected an HWC RGB uint8 NumPy array")
    if value.ndim != 3 or value.shape[2] != 3:
        raise TypeError("expected an HWC RGB uint8 NumPy array")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("image dimensions must be positive")
    return np.ascontiguousarray(value)


def _mask(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise TypeError("expected an HW uint8 NumPy mask")
    if value.ndim != 2:
        raise TypeError("expected an HW uint8 NumPy mask")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("mask dimensions must be positive")
    return np.ascontiguousarray(value)


def _key(value: int | None) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64
    ):
        raise ValueError("key must be an unsigned 64-bit integer")
    return value


def _torch_module():
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise ImportError(
            "ReturnTensor requires PyTorch; install the appropriate torch build for your platform"
        ) from error
    return torch
