from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from adapters import Adapter
from common import (
    MEAN,
    SEED,
    STD,
    TRANSFORMS,
    control_cpu,
    iterations_for,
    make_images,
    metadata,
    output_facts,
    time_calls,
    write_json,
)

SIZES = (224, 512, 1024)
RUST_MODES = ("staged-fresh", "staged-reuse", "compiled")


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


def _albu_parts(backend: str) -> tuple[Any, Any, Any, Any, Any]:
    import albumentations as A
    import cv2

    if backend == "albumentationsx":
        jitter = A.ColorJitter(
            brightness_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2),
            hue_range=(0.0, 0.0),
            p=1,
        )
        affine = A.Affine(
            scale=(1.0, 1.0),
            translate_percent=(0.0, 0.0),
            rotate=(-10.0, 10.0),
            shear=(0.0, 0.0),
            interpolation=cv2.INTER_LINEAR,
            fill=0,
            p=1,
        )
        blur = A.GaussianBlur(blur_range=(5, 5), sigma_range=(1.1, 1.1), p=1)
    else:
        jitter = A.ColorJitter(0.2, 0.2, 0.2, 0.0, p=1)
        affine = A.Affine(
            scale=1.0,
            translate_percent=0.0,
            rotate=(-10.0, 10.0),
            shear=0.0,
            interpolation=cv2.INTER_LINEAR,
            fill=0,
            p=1,
        )
        blur = A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.1, 1.1), p=1)
    return A, cv2, jitter, affine, blur


def _competitor_case(backend: str, case: str, size: int, variant: str) -> Callable[[Any], Any]:
    crop = max(32, size * 7 // 8)
    area = (crop / size) ** 2
    if backend == "torchvision":
        import torch
        from torchvision.transforms import InterpolationMode, v2

        torch.manual_seed(SEED)
        if case == "crop_resize":
            transforms = (
                [
                    v2.RandomResizedCrop(
                        (224, 224),
                        scale=(area, area),
                        ratio=(1.0, 1.0),
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=False,
                    )
                ]
                if variant == "best-official"
                else [
                    v2.RandomCrop((crop, crop)),
                    v2.Resize(
                        (224, 224), interpolation=InterpolationMode.BILINEAR, antialias=False
                    ),
                ]
            )
        elif case == "color_jitter":
            transforms = [v2.ColorJitter(0.2, 0.2, 0.2, 0.0)]
        elif case == "normalize":
            transforms = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
        elif case == "full":
            prefix = (
                [
                    v2.RandomResizedCrop(
                        (224, 224),
                        scale=(area, area),
                        ratio=(1.0, 1.0),
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=False,
                    )
                ]
                if variant == "best-official"
                else [
                    v2.RandomCrop((crop, crop)),
                    v2.Resize(
                        (224, 224), interpolation=InterpolationMode.BILINEAR, antialias=False
                    ),
                ]
            )
            transforms = prefix + [
                v2.RandomHorizontalFlip(0.5),
                v2.ColorJitter(0.2, 0.2, 0.2, 0.0),
                v2.RandomAffine(10.0, interpolation=InterpolationMode.BILINEAR, fill=0),
                v2.GaussianBlur(5, (1.1, 1.1)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(MEAN, STD),
            ]
        else:
            raise ValueError(case)
        transform = v2.Compose(transforms)
        return lambda image: _materialize(transform(image))

    A, cv2, jitter, affine, blur = _albu_parts(backend)
    if case == "crop_resize":
        transforms = (
            [
                A.RandomResizedCrop(
                    size=(224, 224),
                    scale=(area, area),
                    ratio=(1.0, 1.0),
                    interpolation=cv2.INTER_LINEAR,
                    p=1,
                )
            ]
            if variant == "best-official"
            else [
                A.RandomCrop(crop, crop, p=1),
                A.Resize(224, 224, interpolation=cv2.INTER_LINEAR, p=1),
            ]
        )
    elif case == "color_jitter":
        transforms = [jitter]
    elif case == "normalize":
        transforms = [A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, p=1)]
    elif case == "full":
        prefix = (
            [
                A.RandomResizedCrop(
                    size=(224, 224),
                    scale=(area, area),
                    ratio=(1.0, 1.0),
                    interpolation=cv2.INTER_LINEAR,
                    p=1,
                )
            ]
            if variant == "best-official"
            else [
                A.RandomCrop(crop, crop, p=1),
                A.Resize(224, 224, interpolation=cv2.INTER_LINEAR, p=1),
            ]
        )
        transforms = prefix + [
            A.HorizontalFlip(p=0.5),
            jitter,
            affine,
            blur,
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, p=1),
        ]
    else:
        raise ValueError(case)
    transform = _seed_albu(A.Compose(transforms))
    return lambda image: _materialize(transform(image=image)["image"])


def _rust_case(case: str, size: int, mode: str) -> Callable[[np.ndarray], Any]:
    crop = max(32, size * 7 // 8)
    configs: dict[str, list[dict[str, object]]] = {
        "crop_resize": [
            {"type": "RandomCrop", "height": crop, "width": crop},
            {
                "type": "Resize",
                "height": 224,
                "width": 224,
                "interpolation": "bilinear",
                "antialias": False,
            },
        ],
        "color_jitter": [
            {
                "type": "ColorJitter",
                "brightness": (0.8, 1.2),
                "contrast": (0.8, 1.2),
                "saturation": (0.8, 1.2),
                "hue": (0.0, 0.0),
            },
        ],
        "normalize": [{"type": "Normalize", "mean": MEAN, "std": STD}],
        "full": [
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
            {
                "type": "Affine",
                "degrees": (-10.0, 10.0),
                "translate": (0.0, 0.0),
                "scale": (1.0, 1.0),
                "shear": (0.0, 0.0, 0.0, 0.0),
                "interpolation": "bilinear",
                "border_mode": "constant",
                "fill": (0, 0, 0),
            },
            {"type": "GaussianBlur", "kernel_size": 5, "sigma": (1.1, 1.1)},
            {"type": "Normalize", "mean": MEAN, "std": STD},
        ],
    }
    return _rust_apply(configs[case], mode)


def _valid_output(case: str, facts: dict[str, Any]) -> bool:
    if case in {"normalize", "full"}:
        expected = ([224, 224, 3], [3, 224, 224]) if case == "full" else None
        return (
            facts["container"] in {"numpy.ndarray", "torch.Tensor"}
            and (expected is None or facts["shape"] in expected)
            and facts["dtype"] == "float32"
            and facts["finite"]
            and facts["c_contiguous"]
        )
    expected = [224, 224, 3] if case == "crop_resize" else None
    return (
        (expected is None or facts["shape"] == expected or facts["shape"] == [3, 224, 224])
        and facts["dtype"] == "uint8"
        and facts["c_contiguous"]
    )


def run(backend: str, quick: bool, repetition: int) -> list[dict[str, Any]]:
    adapter = Adapter(backend)
    rows: list[dict[str, Any]] = []
    for size in SIZES:
        images = make_images(size)
        native = adapter.native_inputs(images)
        warmup, iterations = iterations_for(size, quick)
        for name in TRANSFORMS:
            variants = ("staged-fresh", "compiled") if backend == "rust" else ("stock",)
            for variant in variants:
                transform = (
                    _rust_apply([_rust_config(name, size)], variant)
                    if backend == "rust"
                    else adapter.build_micro(name, size)
                )
                timing, output = time_calls(transform, native, warmup, iterations)
                facts = output_facts(output)
                expected_size = (
                    max(32, size * 3 // 4)
                    if name in {"Resize", "RandomCrop", "CenterCrop"}
                    else size
                )
                expected_shape = (
                    [3, expected_size, expected_size]
                    if backend == "torchvision"
                    else [expected_size, expected_size, 3]
                )
                row = {
                    "kind": "layer1_transform",
                    "backend": backend,
                    "variant": variant,
                    "transform": name,
                    "size": size,
                    "repetition": repetition,
                    **timing,
                    "validation": facts,
                    "valid": facts["shape"] == expected_shape
                    and facts["finite"]
                    and facts["c_contiguous"],
                }
                if backend == "rust":
                    row["explanation"] = transform.explanation  # type: ignore[attr-defined]
                rows.append(row)

        for case in ("crop_resize", "color_jitter", "normalize"):
            variants = (
                RUST_MODES
                if backend == "rust"
                else ("stock", "best-official")
                if case == "crop_resize"
                else ("stock",)
            )
            for variant in variants:
                transform = (
                    _rust_case(case, size, variant)
                    if backend == "rust"
                    else _competitor_case(backend, case, size, variant)
                )
                timing, output = time_calls(transform, native, warmup, iterations)
                facts = output_facts(output)
                row = {
                    "kind": "layer2_case",
                    "backend": backend,
                    "variant": variant,
                    "case": case,
                    "size": size,
                    "repetition": repetition,
                    **timing,
                    "validation": facts,
                    "valid": _valid_output(case, facts),
                }
                if backend == "rust":
                    row["explanation"] = transform.explanation  # type: ignore[attr-defined]
                rows.append(row)

        if size == 512:
            variants = RUST_MODES if backend == "rust" else ("stock", "best-official")
            for variant in variants:
                transform = (
                    _rust_case("full", size, variant)
                    if backend == "rust"
                    else _competitor_case(backend, "full", size, variant)
                )
                timing, output = time_calls(transform, native, warmup, iterations)
                facts = output_facts(output)
                row = {
                    "kind": "layer2_case",
                    "backend": backend,
                    "variant": variant,
                    "case": "full",
                    "size": size,
                    "repetition": repetition,
                    **timing,
                    "validation": facts,
                    "valid": _valid_output("full", facts),
                }
                if backend == "rust":
                    row["explanation"] = transform.explanation  # type: ignore[attr-defined]
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=("torchvision", "albumentations", "albumentationsx", "rust"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    cpu = control_cpu(pin_process=True)
    write_json(
        args.output,
        {
            "metadata": metadata(args.backend, cpu),
            "rows": run(args.backend, args.quick, args.repetition),
        },
    )


if __name__ == "__main__":
    main()
