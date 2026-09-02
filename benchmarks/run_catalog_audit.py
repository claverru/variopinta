from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    MEAN,
    ROOT,
    SEED,
    STD,
    control_cpu,
    evidence_provenance,
    make_images,
    metadata,
    output_facts,
    write_json,
)


def catalog_cases(size: int) -> list[tuple[str, str, list[Any]]]:
    import variopinta as R

    out = max(32, size * 3 // 4)
    area = (out / size) ** 2
    pad = size + max(1, size // 8)
    return [
        ("Resize", "bilinear", [R.Resize(out, out)]),
        ("Resize", "bilinear-antialias", [R.Resize(out, out, antialias=True)]),
        ("RandomCrop", "default", [R.RandomCrop(out, out)]),
        (
            "RandomResizedCrop",
            "bilinear",
            [R.RandomResizedCrop(out, out, scale=(area, area), ratio=(1.0, 1.0))],
        ),
        (
            "RandomResizedCrop",
            "bilinear-antialias",
            [
                R.RandomResizedCrop(
                    out,
                    out,
                    scale=(area, area),
                    ratio=(1.0, 1.0),
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
        ("HorizontalFlip", "default", [R.HorizontalFlip(1.0)]),
        ("VerticalFlip", "default", [R.VerticalFlip(1.0)]),
        ("ColorJitter", "matrix", [R.ColorJitter(0.2, 0.2, 0.2)]),
        ("ColorJitter", "hue", [R.ColorJitter(0.2, 0.2, 0.2, 0.1)]),
        ("Affine", "constant", [R.Affine(10.0)]),
        (
            "Affine",
            "reflect101",
            [R.Affine(10.0, border_mode=R.BorderMode.REFLECT101)],
        ),
        ("RandomRotation", "constant", [R.RandomRotation(10.0)]),
        (
            "RandomRotation",
            "reflect101",
            [R.RandomRotation(10.0, border_mode=R.BorderMode.REFLECT101)],
        ),
        ("GaussianNoise", "independent-rgb", [R.GaussianNoise(std=10.0)]),
        (
            "GaussianNoise",
            "shared-rgb",
            [R.GaussianNoise(std=10.0, per_channel=False)],
        ),
        ("Sharpen", "cross-3x3", [R.Sharpen(alpha=0.5, lightness=1.0)]),
        ("Perspective", "bilinear", [R.Perspective(scale=0.05)]),
        (
            "Perspective",
            "nearest-reflect101",
            [
                R.Perspective(
                    scale=0.05,
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
        ("GaussianBlur", "fixed-sigma", [R.GaussianBlur(5, 1.1)]),
        ("GaussianBlur", "sampled-sigma", [R.GaussianBlur(5, (0.8, 1.4))]),
        ("Grayscale", "default", [R.Grayscale(1.0)]),
        ("Invert", "default", [R.Invert(1.0)]),
        ("Solarize", "default", [R.Solarize(128, 1.0)]),
        ("Posterize", "default", [R.Posterize(4, 1.0)]),
        ("Normalize", "default", [R.Normalize(MEAN, STD)]),
        ("ToTorch", "default", [R.ToTorch()]),
        ("Normalize+ToTorch", "terminal", [R.Normalize(MEAN, STD), R.ToTorch()]),
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
    to_torch = "ToTorch" in transform
    valid = (
        facts["finite"]
        and facts["c_contiguous"]
        and facts["container"] == ("torch.Tensor" if to_torch else "numpy.ndarray")
        and facts["shape"][0 if to_torch else -1] == 3
        and facts["dtype"] == ("float32" if "Normalize" in transform else "uint8")
    )
    return facts, valid


def audit_catalog() -> dict[str, Any]:
    import variopinta as R

    cpu = control_cpu()
    rows: list[dict[str, Any]] = []
    registered: set[str] = set()
    for size in (224, 512, 1024):
        source = make_images(size, count=1)[0]
        cases = catalog_cases(size)
        validate_catalog_coverage(cases)
        registered.update(name for name, _, _ in cases if "+" not in name)
        for transform, variant, transforms in cases:
            reference = R.Compose(transforms, seed=SEED)
            compiled = reference.compile()
            reference_output = reference(source, key=SEED)
            compiled_output = compiled(source, key=SEED)
            exact = bool(np.array_equal(as_array(reference_output), as_array(compiled_output)))
            reference_facts, reference_valid = validate_output(transform, reference_output)
            compiled_facts, compiled_valid = validate_output(transform, compiled_output)
            rows.append(
                {
                    "transform": transform,
                    "variant": variant,
                    "size": size,
                    "reference_exact": exact,
                    "reference_validation": reference_facts,
                    "compiled_validation": compiled_facts,
                    "valid": exact and reference_valid and compiled_valid,
                    "explanation": compiled.explain(),
                }
            )
    return {
        "schema_version": 1,
        "provenance": evidence_provenance(),
        "metadata": metadata("rust-catalog-audit", cpu),
        "transforms": len(registered),
        "rows": rows,
    }


def render_catalog_audit_summary(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    return "\n".join(
        [
            "## Complete catalog correctness audit",
            "",
            f"The Rust-only audit covers **{payload['transforms']}** registered transforms and "
            f"has {sum(row['valid'] for row in rows)}/{len(rows)} valid case-size rows. "
            f"Compiled/reference equality holds in "
            f"{sum(row['reference_exact'] for row in rows)}/{len(rows)} rows.",
            "",
            "This gate executes each reference/compiled pair once at 224², 512², and 1024². "
            "It does not collect performance samples.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit complete catalog correctness")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "raw" / "catalog-audit.json",
    )
    args = parser.parse_args()
    payload = audit_catalog()
    write_json(args.output, payload)
    invalid = [row for row in payload["rows"] if not row["valid"]]
    if invalid:
        raise SystemExit(f"catalog audit failed: {len(invalid)} invalid rows")
    print(f"Catalog audit: {len(payload['rows'])} valid rows")
    print(f"Registered transforms: {payload['transforms']}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
