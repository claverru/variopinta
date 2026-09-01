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


def _triplet(
    name: str, values: tuple[float, float, float], *, positive: bool
) -> tuple[float, float, float]:
    if (
        not isinstance(values, tuple)
        or len(values) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            for value in values
        )
    ):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} values must be {qualifier}")
    result = tuple(_f32(name, value) for value in values)
    if positive and any(value <= 0.0 for value in result):
        raise ValueError(f"{name} values must be finite and positive")
    return result


def _fill(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    values = (
        (value, value, value) if isinstance(value, int) and not isinstance(value, bool) else value
    )
    if (
        not isinstance(values, tuple)
        or len(values) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
            for item in values
        )
    ):
        raise ValueError("fill must be an integer or an RGB tuple with values in [0, 255]")
    return values


def _dropout_size_range(
    name: str, values: tuple[int, int] | tuple[float, float]
) -> tuple[tuple[int, int] | tuple[float, float], str]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        if values[0] <= 0 or values[0] > values[1]:
            raise ValueError(f"{name} pixel values must be ordered and positive")
        return values, "pixels"
    if all(isinstance(value, float) for value in values):
        result = (_f32(name, values[0]), _f32(name, values[1]))
        if any(not 0.0 < value <= 1.0 for value in result) or result[0] > result[1]:
            raise ValueError(f"{name} fraction values must be ordered and in (0, 1]")
        return result, "fraction"
    raise TypeError(f"{name} must contain either two integers or two floats")


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
    name: str, value: float | tuple[float, float], *, non_negative: bool = False
) -> tuple[float, float]:
    if isinstance(value, bool) or not isinstance(value, int | float):
        result = _finite_pair(name, value)
    else:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        canonical = _f32(name, value)
        result = (canonical, canonical)
    if non_negative and result[0] < 0.0:
        raise ValueError(f"{name} values must be non-negative")
    return result


def _symmetric_limit_range(
    name: str, value: float | tuple[float, float], *, maximum: float
) -> tuple[float, float]:
    if isinstance(value, bool) or not isinstance(value, int | float):
        result = _finite_pair(name, value)
    else:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
        canonical = _f32(name, value)
        if canonical < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        result = (-canonical, canonical)
    if result[0] <= -maximum or result[1] >= maximum:
        raise ValueError(f"{name} values must be strictly within (-{maximum}, {maximum})")
    return result


def _affine_degrees(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _finite_pair("degrees", value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("degrees must be finite and non-negative")
    canonical = _f32("degrees", value)
    return (-canonical, canonical)


def _affine_translate(value: tuple[float, float]) -> tuple[float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item)
            for item in value
        )
    ):
        raise ValueError("translate must contain two finite values")
    result = (_f32("translate", value[0]), _f32("translate", value[1]))
    if any(not 0.0 <= item <= 1.0 for item in result):
        raise ValueError("translate values must be in [0, 1]")
    return result


def _affine_scale(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, bool) or not isinstance(value, int | float):
        result = _finite_pair("scale", value)
    else:
        if not math.isfinite(value):
            raise ValueError("scale must be finite and positive")
        canonical = _f32("scale", value)
        result = (canonical, canonical)
    if result[0] <= 0.0:
        raise ValueError("scale values must be positive")
    return result


def _affine_shear(
    value: float | tuple[float, float] | tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if isinstance(value, int | float) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("shear must be finite and non-negative")
        canonical = _f32("shear", value)
        if canonical < 0.0:
            raise ValueError("shear must be finite and non-negative")
        result = (-canonical, canonical, 0.0, 0.0)
    elif isinstance(value, tuple) and len(value) in (2, 4):
        x_range = _finite_pair("shear", value[:2])
        y_range = _finite_pair("shear", value[2:]) if len(value) == 4 else (0.0, 0.0)
        result = (*x_range, *y_range)
    else:
        raise ValueError("shear must be a number or a tuple of two or four finite values")
    if any(abs(item) >= 90.0 for item in result):
        raise ValueError("shear values must be strictly between -90 and 90 degrees")
    return result


def _color_factor_range(name: str, value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, bool) or not isinstance(value, int | float):
        result = _finite_pair(name, value)
    else:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
        canonical = _f32(name, value)
        if canonical < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        result = (_f32(name, max(0.0, 1.0 - canonical)), _f32(name, 1.0 + canonical))
    if result[0] < 0.0:
        raise ValueError(f"{name} range values must be non-negative")
    return result


def _hue_range(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, bool) or not isinstance(value, int | float):
        result = _finite_pair("hue", value)
    else:
        if not math.isfinite(value):
            raise ValueError("hue must be finite and in [0, 0.5]")
        canonical = _f32("hue", value)
        if not 0.0 <= canonical <= 0.5:
            raise ValueError("hue must be finite and in [0, 0.5]")
        result = (-canonical, canonical)
    if result[0] < -0.5 or result[1] > 0.5:
        raise ValueError("hue range values must be in [-0.5, 0.5]")
    return result


def _sigma_range(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, bool) or not isinstance(value, int | float):
        result = _finite_pair("sigma", value)
    else:
        if not math.isfinite(value):
            raise ValueError("sigma must be finite and positive")
        canonical = _f32("sigma", value)
        result = (canonical, canonical)
    if result[0] <= 0.0:
        raise ValueError("sigma values must be positive")
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
            "ToTorch requires PyTorch; install the appropriate torch build for your platform"
        ) from error
    return torch
