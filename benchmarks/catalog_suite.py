from __future__ import annotations

from typing import Any

import numpy as np
from common import (
    MEAN,
    SEED,
    STD,
    make_images,
    output_facts,
    time_calls_adaptive,
)


def catalog_cases(size: int) -> list[tuple[str, str, list[Any]]]:
    import variopinta as R

    out = max(32, size * 3 // 4)
    area = (out / size) ** 2
    pad = size + max(1, size // 8)
    return [
        ("Resize", "bilinear", [R.Resize(out, out)]),
        ("Resize", "bilinear-antialias", [R.Resize(out, out, antialias=True)]),
        ("LongestMaxSize", "bilinear", [R.LongestMaxSize(out)]),
        (
            "LongestMaxSize",
            "bilinear-antialias",
            [R.LongestMaxSize(out, antialias=True)],
        ),
        (
            "LongestMaxSize",
            "bilinear-upscale",
            [R.LongestMaxSize(size + max(1, size // 4))],
        ),
        (
            "LongestMaxSize+PadIfNeeded",
            "bilinear-centered-square",
            [R.LongestMaxSize(out), R.PadIfNeeded(min_height=out, min_width=out)],
        ),
        ("RandomCrop", "default", [R.RandomCrop(out, out)]),
        (
            "RandomResizedCrop",
            "bilinear",
            [
                R.RandomResizedCrop(
                    out,
                    out,
                    area_range=(area, area),
                    aspect_ratio_range=(1.0, 1.0),
                )
            ],
        ),
        (
            "RandomResizedCrop",
            "bilinear-antialias",
            [
                R.RandomResizedCrop(
                    out,
                    out,
                    area_range=(area, area),
                    aspect_ratio_range=(1.0, 1.0),
                    antialias=True,
                )
            ],
        ),
        ("CenterCrop", "default", [R.CenterCrop(out, out)]),
        (
            "PadIfNeeded",
            "constant",
            [R.PadIfNeeded(min_height=pad, min_width=pad)],
        ),
        (
            "PadIfNeeded",
            "reflect101",
            [
                R.PadIfNeeded(
                    min_height=pad,
                    min_width=pad,
                    border_mode=R.BorderMode.REFLECT101,
                )
            ],
        ),
        (
            "CoarseDropout",
            "eight-holes",
            [
                R.CoarseDropout(
                    num_holes_range=(8, 8),
                    hole_height_range=(0.05, 0.10),
                    hole_width_range=(0.05, 0.10),
                    p=1.0,
                )
            ],
        ),
        ("HorizontalFlip", "default", [R.HorizontalFlip(p=1.0)]),
        ("VerticalFlip", "default", [R.VerticalFlip(p=1.0)]),
        ("ColorJitter", "matrix", [R.ColorJitter()]),
        ("ColorJitter", "hue", [R.ColorJitter(hue_range=(-0.1, 0.1))]),
        ("Affine", "constant", [R.Affine(degrees_range=(-10.0, 10.0))]),
        (
            "Affine",
            "reflect101",
            [R.Affine(degrees_range=(-10.0, 10.0), border_mode=R.BorderMode.REFLECT101)],
        ),
        ("RandomRotation", "constant", [R.RandomRotation((-10.0, 10.0))]),
        (
            "RandomRotation",
            "reflect101",
            [R.RandomRotation((-10.0, 10.0), border_mode=R.BorderMode.REFLECT101)],
        ),
        ("GaussianNoise", "independent-rgb", [R.GaussianNoise(std_range=(10.0, 10.0))]),
        (
            "GaussianNoise",
            "shared-rgb",
            [R.GaussianNoise(std_range=(10.0, 10.0), per_channel=False)],
        ),
        ("Sharpen", "cross-3x3", [R.Sharpen()]),
        (
            "Perspective",
            "bilinear",
            [R.Perspective(distortion_scale_range=(0.05, 0.05))],
        ),
        (
            "Perspective",
            "nearest-reflect101",
            [
                R.Perspective(
                    distortion_scale_range=(0.05, 0.05),
                    interpolation=R.Interpolation.NEAREST,
                    border_mode=R.BorderMode.REFLECT101,
                )
            ],
        ),
        ("GridDistortion", "bilinear", [R.GridDistortion(num_steps=5)]),
        (
            "GridDistortion",
            "nearest-reflect101",
            [
                R.GridDistortion(
                    num_steps=5,
                    interpolation=R.Interpolation.NEAREST,
                    border_mode=R.BorderMode.REFLECT101,
                )
            ],
        ),
        (
            "GaussianBlur",
            "fixed-sigma",
            [R.GaussianBlur(kernel_size=5, sigma_range=(1.1, 1.1))],
        ),
        (
            "GaussianBlur",
            "sampled-sigma",
            [R.GaussianBlur(kernel_size=5, sigma_range=(0.8, 1.4))],
        ),
        ("Grayscale", "default", [R.Grayscale(p=1.0)]),
        ("Invert", "default", [R.Invert(p=1.0)]),
        ("Solarize", "default", [R.Solarize(128, p=1.0)]),
        ("Posterize", "default", [R.Posterize(4, p=1.0)]),
        ("Normalize", "default", [R.Normalize(mean=MEAN, std=STD)]),
    ]


def as_array(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def validate_catalog_coverage(cases: list[tuple[str, str, list[Any]]]) -> None:
    from variopinta import _variopinta

    covered = {name for name, _, _ in cases if "+" not in name}
    registered = set(_variopinta.registered_transform_names())
    if covered != registered:
        missing = sorted(registered - covered)
        extra = sorted(covered - registered)
        raise RuntimeError(f"catalog coverage mismatch: missing={missing}, extra={extra}")


def validate_output(transform: str, value: Any) -> tuple[dict[str, Any], bool]:
    facts = output_facts(value)
    valid = (
        facts["finite"]
        and facts["c_contiguous"]
        and facts["container"] == "numpy.ndarray"
        and facts["shape"][-1] == 3
        and facts["dtype"] == ("float32" if "Normalize" in transform else "uint8")
    )
    return facts, valid


def _case_transforms(factory: str, size: int) -> tuple[str, str, list[Any]]:
    expected_name, expected_variant = factory.split("|", 1)
    for name, variant, transforms in catalog_cases(size):
        if name == expected_name and variant == expected_variant:
            return name, variant, transforms
    raise ValueError(f"unknown catalog factory: {factory}")


def _case_images(transform: str, size: int, count: int = 8) -> list[np.ndarray]:
    images = make_images(size, count=count)
    if transform == "LongestMaxSize+PadIfNeeded":
        height = max(1, size * 2 // 3)
        return [np.ascontiguousarray(image[:height]) for image in images]
    return images


def run_planned(
    items: list[dict[str, Any]], quick: bool, repetition: int, *, validate_only: bool = False
) -> list[dict[str, Any]]:
    import variopinta as R

    validate_catalog_coverage(catalog_cases(224))
    rows = []
    for order, item in enumerate(items, start=1):
        route = item["route"]
        for size in item["sizes"]:
            transform, policy, transforms = _case_transforms(item["factory"], size)
            reference = R.Pipeline(transforms, seed=SEED)
            compiled = reference.compile()
            source = _case_images(transform, size, count=1)[0]
            reference_output = reference(source, key=SEED)
            compiled_output = compiled(source, key=SEED)
            exact = bool(np.array_equal(as_array(reference_output), as_array(compiled_output)))
            pipeline = reference if route["variant"] == "reference" else compiled
            output = reference_output if route["variant"] == "reference" else compiled_output
            facts, output_valid = validate_output(transform, output)
            row = {
                "case_id": item["case_id"],
                "route_id": route["id"],
                "participant": route["participant"],
                "variant": route["variant"],
                "role": route["role"],
                "size": size,
                "repetition": repetition,
                "case_order": order,
                "reference_exact": exact,
                "validation": facts,
                "valid": exact and output_valid,
                "explanation": pipeline.explain(),
                "catalog_policy": policy,
            }
            if not validate_only:
                timing_policy = dict(item["timing"])
                if quick:
                    timing_policy.update(
                        {
                            "budget_ms": min(float(timing_policy["budget_ms"]), 10.0),
                            "warmup_calls": min(int(timing_policy["warmup_calls"]), 2),
                            "min_samples": 3,
                            "max_calls": 64,
                        }
                    )
                images = _case_images(transform, size)
                timing, measured_output = time_calls_adaptive(
                    lambda image, selected=pipeline: selected(image, key=SEED),
                    images,
                    **timing_policy,
                )
                measured_facts, measured_valid = validate_output(transform, measured_output)
                row.update(timing)
                row["validation"] = measured_facts
                row["valid"] = row["valid"] and measured_valid
            rows.append(row)
    return rows
