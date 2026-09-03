from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from adapters import Adapter
from common import (
    MEAN,
    SEED,
    STD,
    make_images,
    output_facts,
    time_calls_adaptive,
)


def _materialize(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.contiguous()
    return np.ascontiguousarray(value)


def _rust_config(name: str, size: int) -> dict[str, object]:
    out = max(32, size * 3 // 4)
    configs: dict[str, dict[str, object]] = {
        "Resize": {
            "type": "Resize",
            "height": out,
            "width": out,
            "interpolation": "bilinear",
            "antialias": False,
        },
        "RandomCrop": {"type": "RandomCrop", "height": out, "width": out},
        "CenterCrop": {"type": "CenterCrop", "height": out, "width": out},
        "HorizontalFlip": {"type": "HorizontalFlip", "p": 1.0},
        "VerticalFlip": {"type": "VerticalFlip", "p": 1.0},
        "ColorJitter": {
            "type": "ColorJitter",
            "brightness": (0.8, 1.2),
            "contrast": (0.8, 1.2),
            "saturation": (0.8, 1.2),
            "hue": (0.0, 0.0),
        },
        "Affine": {
            "type": "Affine",
            "degrees": (-10.0, 10.0),
            "translate": (0.0, 0.0),
            "scale": (1.0, 1.0),
            "shear": (0.0, 0.0, 0.0, 0.0),
            "interpolation": "bilinear",
            "border_mode": "constant",
            "fill": (0, 0, 0),
        },
        "GaussianBlur": {"type": "GaussianBlur", "kernel_size": 5, "sigma": (1.1, 1.1)},
        "Grayscale": {"type": "Grayscale", "p": 1.0},
        "Invert": {"type": "Invert", "p": 1.0},
        "Solarize": {"type": "Solarize", "threshold": 128, "p": 1.0},
        "Posterize": {"type": "Posterize", "bits": 4, "p": 1.0},
        "Normalize": {"type": "Normalize", "mean": MEAN, "std": STD},
    }
    return configs[name]


def _focused_rust_config(case: str, size: int) -> dict[str, object]:
    configs: dict[str, dict[str, object]] = {
        "affine-reflect101": {
            "type": "Affine",
            "degrees": (-10.0, 10.0),
            "translate": (0.0, 0.0),
            "scale": (1.0, 1.0),
            "shear": (0.0, 0.0, 0.0, 0.0),
            "p": 1.0,
            "interpolation": "bilinear",
            "border_mode": "reflect101",
            "fill": (0, 0, 0),
        },
        "rotation-reflect101": {
            "type": "RandomRotation",
            "degrees": (-10.0, 10.0),
            "p": 1.0,
            "interpolation": "bilinear",
            "border_mode": "reflect101",
            "fill": (0, 0, 0),
        },
        "gaussian-noise-independent": {
            "type": "GaussianNoise",
            "mean": (0.0, 0.0),
            "std": (10.0, 10.0),
            "per_channel": True,
            "p": 1.0,
        },
        "gaussian-noise-shared": {
            "type": "GaussianNoise",
            "mean": (0.0, 0.0),
            "std": (10.0, 10.0),
            "per_channel": False,
            "p": 1.0,
        },
        "color-jitter-hue": {
            "type": "ColorJitter",
            "brightness": (1.0, 1.0),
            "contrast": (1.0, 1.0),
            "saturation": (1.0, 1.0),
            "hue": (-0.1, 0.1),
            "p": 1.0,
        },
        "sharpen-cross": {
            "type": "Sharpen",
            "alpha": (0.5, 0.5),
            "lightness": (1.0, 1.0),
            "p": 1.0,
        },
        "perspective-bilinear": {
            "type": "Perspective",
            "scale": (0.05, 0.05),
            "p": 1.0,
            "interpolation": "bilinear",
            "border_mode": "constant",
            "fill": (0, 0, 0),
        },
        "perspective-bilinear-reflect101": {
            "type": "Perspective",
            "scale": (0.05, 0.05),
            "p": 1.0,
            "interpolation": "bilinear",
            "border_mode": "reflect101",
            "fill": (0, 0, 0),
        },
        "perspective-nearest-constant": {
            "type": "Perspective",
            "scale": (0.05, 0.05),
            "p": 1.0,
            "interpolation": "nearest",
            "border_mode": "constant",
            "fill": (0, 0, 0),
        },
        "perspective-nearest-reflect101": {
            "type": "Perspective",
            "scale": (0.05, 0.05),
            "p": 1.0,
            "interpolation": "nearest",
            "border_mode": "reflect101",
            "fill": (0, 0, 0),
        },
        "grid-distortion-bilinear": {
            "type": "GridDistortion",
            "num_steps": 5,
            "distort_limit": (-0.3, 0.3),
            "p": 1.0,
            "interpolation": "bilinear",
            "border_mode": "constant",
            "fill": (0, 0, 0),
        },
        "grid-distortion-bilinear-reflect101": {
            "type": "GridDistortion",
            "num_steps": 5,
            "distort_limit": (-0.3, 0.3),
            "p": 1.0,
            "interpolation": "bilinear",
            "border_mode": "reflect101",
            "fill": (0, 0, 0),
        },
        "grid-distortion-nearest-constant": {
            "type": "GridDistortion",
            "num_steps": 5,
            "distort_limit": (-0.3, 0.3),
            "p": 1.0,
            "interpolation": "nearest",
            "border_mode": "constant",
            "fill": (0, 0, 0),
        },
        "grid-distortion-nearest-reflect101": {
            "type": "GridDistortion",
            "num_steps": 5,
            "distort_limit": (-0.3, 0.3),
            "p": 1.0,
            "interpolation": "nearest",
            "border_mode": "reflect101",
            "fill": (0, 0, 0),
        },
        "pad-constant": {
            "type": "PadIfNeeded",
            "min_height": size + 8,
            "min_width": size + 8,
            "pad_height_divisor": None,
            "pad_width_divisor": None,
            "position": "center",
            "p": 1.0,
            "border_mode": "constant",
            "fill": (17, 17, 17),
        },
        "pad-reflect101": {
            "type": "PadIfNeeded",
            "min_height": size + 8,
            "min_width": size + 8,
            "pad_height_divisor": None,
            "pad_width_divisor": None,
            "position": "center",
            "p": 1.0,
            "border_mode": "reflect101",
            "fill": (0, 0, 0),
        },
        "to-torch": {"type": "ToTorch"},
    }
    return configs[case]


def _rust_apply(configs: list[dict[str, object]], mode: str) -> Callable[[np.ndarray], Any]:
    from variopinta._variopinta import Pipeline

    pipeline = Pipeline(configs, SEED, mode)

    def apply(image: np.ndarray) -> Any:
        return np.ascontiguousarray(pipeline.apply(image))

    apply.explanation = pipeline.explain()  # type: ignore[attr-defined]
    return apply


def _seed_albu(transform: Any) -> Any:
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(SEED)
    return transform


def _focused_torch(case: str, size: int) -> Callable[[Any], Any] | None:
    import torch
    from torchvision.transforms import InterpolationMode, v2

    torch.manual_seed(SEED)
    transforms: dict[str, Any] = {
        "gaussian-noise-independent": v2.GaussianNoise(mean=0.0, sigma=10.0 / 255.0, clip=True),
        "color-jitter-hue": v2.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.1),
        "perspective-bilinear": v2.RandomPerspective(
            distortion_scale=0.1,
            p=1.0,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        "perspective-nearest-constant": v2.RandomPerspective(
            distortion_scale=0.1,
            p=1.0,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        ),
        "pad-constant": v2.Pad(4, fill=17, padding_mode="constant"),
        "pad-reflect101": v2.Pad(4, padding_mode="reflect"),
        "to-torch": v2.ToImage(),
    }
    transform = transforms.get(case)
    if transform is None:
        return None
    return lambda image: _materialize(transform(image))


def _focused_albu(case: str, size: int) -> Callable[[Any], Any]:
    import albumentations as A
    import cv2

    if case == "affine-reflect101":
        transform = A.Affine(
            scale=(1.0, 1.0),
            translate_percent=(0.0, 0.0),
            rotate=(-10.0, 10.0),
            shear=(0.0, 0.0),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            fill=0,
            p=1,
        )
    elif case == "rotation-reflect101":
        transform = A.Rotate(
            angle_range=(-10.0, 10.0),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            fill=0,
            p=1,
        )
    elif case in {"gaussian-noise-independent", "gaussian-noise-shared"}:
        noise: dict[str, Any] = {
            "std_range": (10.0 / 255.0, 10.0 / 255.0),
            "mean_range": (0.0, 0.0),
            "per_channel": case == "gaussian-noise-independent",
            "p": 1,
        }
        transform = A.GaussNoise(**noise)
    elif case == "color-jitter-hue":
        names = {
            "brightness_range": (1.0, 1.0),
            "contrast_range": (1.0, 1.0),
            "saturation_range": (1.0, 1.0),
            "hue_range": (-0.1, 0.1),
        }
        transform = A.ColorJitter(**names, p=1)
    elif case == "sharpen-cross":
        strength = {"alpha_range": (0.5, 0.5), "lightness_range": (1.0, 1.0)}
        transform = A.Sharpen(**strength, method="kernel", p=1)
    elif case.startswith("perspective-"):
        transform = A.Perspective(
            scale=(0.05, 0.05),
            keep_size=True,
            fit_output=False,
            interpolation=(cv2.INTER_NEAREST if "-nearest-" in case else cv2.INTER_LINEAR),
            border_mode=(
                cv2.BORDER_REFLECT_101 if case.endswith("reflect101") else cv2.BORDER_CONSTANT
            ),
            fill=0,
            p=1,
        )
    elif case.startswith("grid-distortion-"):
        transform = A.GridDistortion(
            distort_range=(-0.3, 0.3),
            num_steps=5,
            interpolation=(cv2.INTER_NEAREST if "-nearest-" in case else cv2.INTER_LINEAR),
            normalized=True,
            border_mode=(
                cv2.BORDER_REFLECT_101 if case.endswith("reflect101") else cv2.BORDER_CONSTANT
            ),
            fill=0,
            p=1,
        )
    elif case in {"pad-constant", "pad-reflect101"}:
        transform = A.PadIfNeeded(
            min_height=size + 8,
            min_width=size + 8,
            position="center",
            border_mode=(
                cv2.BORDER_REFLECT_101 if case == "pad-reflect101" else cv2.BORDER_CONSTANT
            ),
            fill=17,
            p=1,
        )
    elif case == "to-torch":
        from albumentations.pytorch import ToTensorV2

        transform = ToTensorV2()
    else:
        raise ValueError(case)
    composed = _seed_albu(A.Compose([transform]))
    return lambda image: _materialize(composed(image=image)["image"])


def _opencv_cross_sharpen() -> Callable[[np.ndarray], Any]:
    import cv2

    kernel = np.array(
        [[0.0, -0.5, 0.0], [-0.5, 3.0, -0.5], [0.0, -0.5, 0.0]],
        dtype=np.float32,
    )
    return lambda image: np.ascontiguousarray(
        cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    )


def _focused_transform(
    backend: str, case: str, size: int, rust_mode: str = "compiled"
) -> Callable[[Any], Any] | None:
    if backend == "rust":
        transform = _rust_apply([_focused_rust_config(case, size)], rust_mode)
        if case != "to-torch":
            return transform
        import torch

        def to_torch(image: np.ndarray) -> Any:
            return torch.from_numpy(transform(image))

        to_torch.explanation = transform.explanation  # type: ignore[attr-defined]
        return to_torch
    if backend == "torchvision":
        return _focused_torch(case, size)
    return _focused_albu(case, size)


def _pipeline_configs(name: str, size: int) -> list[dict[str, object]]:
    crop = max(224, size * 7 // 8)
    classic: list[dict[str, object]] = [
        {"type": "RandomCrop", "height": crop, "width": crop},
        {
            "type": "Resize",
            "height": 224,
            "width": 224,
            "interpolation": "bilinear",
            "antialias": False,
        },
        {"type": "HorizontalFlip", "p": 0.5},
        {
            "type": "ColorJitter",
            "brightness": (0.8, 1.2),
            "contrast": (0.8, 1.2),
            "saturation": (0.8, 1.2),
            "hue": (0.0, 0.0),
        },
        _rust_config("Affine", size),
        _rust_config("GaussianBlur", size),
    ]
    pipelines = {
        "classic": classic,
        "extended": [
            {"type": "RandomCrop", "height": crop, "width": crop},
            {
                "type": "Resize",
                "height": 224,
                "width": 224,
                "interpolation": "bilinear",
                "antialias": False,
            },
            {"type": "HorizontalFlip", "p": 0.5},
            {"type": "VerticalFlip", "p": 0.2},
            {
                "type": "ColorJitter",
                "brightness": (0.8, 1.2),
                "contrast": (0.8, 1.2),
                "saturation": (0.8, 1.2),
                "hue": (0.0, 0.0),
            },
            _rust_config("Affine", size),
            _rust_config("GaussianBlur", size),
            {"type": "Grayscale", "p": 0.1},
            {"type": "Solarize", "threshold": 128, "p": 0.2},
            {"type": "Posterize", "bits": 4, "p": 0.2},
        ],
        "pixel_policy": [
            {"type": "CenterCrop", "height": crop, "width": crop},
            {
                "type": "Resize",
                "height": 224,
                "width": 224,
                "interpolation": "bilinear",
                "antialias": False,
            },
            {"type": "Grayscale", "p": 0.2},
            {"type": "Invert", "p": 0.1},
            {"type": "Solarize", "threshold": 128, "p": 0.2},
            {"type": "Posterize", "bits": 4, "p": 0.2},
        ],
    }
    return [*pipelines[name], {"type": "Normalize", "mean": MEAN, "std": STD}]


def _quick_policy(policy: dict[str, Any], quick: bool) -> dict[str, Any]:
    if not quick:
        return policy
    value = dict(policy)
    value.update(
        {
            "budget_ms": min(float(value["budget_ms"]), 10.0),
            "warmup_calls": min(int(value["warmup_calls"]), 2),
            "min_samples": 3,
            "max_calls": 64,
        }
    )
    if value.get("block_size") is not None:
        value["block_size"] = min(int(value["block_size"]), 4)
    return value


def _planned_transform(
    factory: str, participant: str, variant: str, size: int
) -> tuple[Callable[[Any], Any], list[Any]]:
    backend = "rust" if participant == "variopinta" else participant
    images = make_images(size)
    if participant == "opencv":
        return _opencv_cross_sharpen(), images
    adapter = Adapter(backend)
    native = adapter.native_inputs(images)
    kind, name = factory.split(":", 1)
    if kind == "transform":
        function = (
            _rust_apply([_rust_config(name, size)], variant)
            if backend == "rust"
            else adapter.build_micro(name, size)
        )
    elif kind == "transform-antialias":
        function = adapter.build_antialiased_resize(size)
        if function is None:
            raise ValueError(f"{participant} does not provide {factory}")
    elif kind == "focused":
        function = _focused_transform(backend, name, size, variant)
        if function is None:
            raise ValueError(f"{participant} does not provide {factory}")
        if name == "to-torch" and backend != "torchvision":
            native = images
    elif kind == "pipeline":
        function = (
            _rust_apply(_pipeline_configs(name, size), variant)
            if backend == "rust"
            else adapter.build_pipeline(size, name)
        )
    else:
        raise ValueError(factory)
    return function, native


def _planned_output_valid(factory: str, participant: str, size: int, facts: dict[str, Any]) -> bool:
    kind, name = factory.split(":", 1)
    chw = participant == "torchvision" or name == "to-torch"
    if kind == "pipeline":
        return (
            facts["shape"] == ([3, 224, 224] if chw else [224, 224, 3])
            and facts["dtype"] == "float32"
            and facts["finite"]
            and facts["c_contiguous"]
        )
    if kind == "focused" and name.startswith("pad-"):
        expected_size = size + 8
    elif kind in {"transform", "transform-antialias"} and name in {
        "Resize",
        "RandomCrop",
        "CenterCrop",
    }:
        expected_size = max(32, size * 3 // 4)
    else:
        expected_size = size
    dtype = "float32" if name == "Normalize" else "uint8"
    return (
        facts["shape"]
        == ([3, expected_size, expected_size] if chw else [expected_size, expected_size, 3])
        and facts["dtype"] == dtype
        and facts["finite"]
        and facts["c_contiguous"]
    )


def run_planned(items: list[dict[str, Any]], quick: bool, repetition: int) -> list[dict[str, Any]]:
    rows = []
    for order, item in enumerate(items, start=1):
        route = item["route"]
        policy = _quick_policy(item["timing"], quick)
        for size in item["sizes"]:
            function, inputs = _planned_transform(
                item["factory"], route["participant"], route["variant"], size
            )
            timing, output = time_calls_adaptive(function, inputs, **policy)
            facts = output_facts(output)
            row = {
                "case_id": item["case_id"],
                "route_id": route["id"],
                "participant": route["participant"],
                "variant": route["variant"],
                "role": route["role"],
                "size": size,
                "repetition": repetition,
                "case_order": order,
                **timing,
                "validation": facts,
                "valid": _planned_output_valid(item["factory"], route["participant"], size, facts),
            }
            explanation = getattr(function, "explanation", None)
            if explanation is not None:
                row["explanation"] = explanation
            rows.append(row)
    return rows
