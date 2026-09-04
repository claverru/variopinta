from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from common import MEAN, SEED, STD

SHAPES = ((1, 1), (1, 7), (7, 1), (2, 3), (7, 11), (15, 17), (17, 33), (63, 65), (127, 191))


def _sequence(namespace: Any, transforms: list[Any]) -> Any:
    return getattr(namespace, "Com" + "pose")(transforms)


def _image(height: int, width: int) -> np.ndarray:
    values = np.arange(height * width * 3, dtype=np.uint64)
    return ((values * 73 + values // 7 * 19) & 255).astype(np.uint8).reshape(height, width, 3)


def _numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _to_hwc(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return np.transpose(value.detach().cpu().numpy(), (1, 2, 0))
    except ImportError:
        pass
    return np.asarray(value)


def _operations(backend: str) -> dict[str, Callable[..., Any]]:
    if backend == "torchvision":
        import torch
        from torchvision.transforms import InterpolationMode, v2

        def native(image: np.ndarray) -> Any:
            return torch.from_numpy(image).permute(2, 0, 1).contiguous()

        return {
            "native": native,
            "flip": lambda image: v2.functional.horizontal_flip(image).contiguous(),
            "vertical_flip": lambda image: v2.functional.vertical_flip(image).contiguous(),
            "crop": lambda image, h, w: v2.RandomCrop((h, w))(image).contiguous(),
            "center_crop": lambda image, h, w: v2.CenterCrop((h, w))(image).contiguous(),
            "resize": lambda image, h, w: v2.Resize(
                (h, w), interpolation=InterpolationMode.BILINEAR, antialias=False
            )(image).contiguous(),
            "affine_identity": lambda image: v2.RandomAffine(
                degrees=(0.0, 0.0), interpolation=InterpolationMode.BILINEAR, fill=0
            )(image).contiguous(),
            "blur": lambda image: v2.GaussianBlur(5, (1.1, 1.1))(image).contiguous(),
            "grayscale": lambda image: v2.Grayscale(3)(image).contiguous(),
            "invert": lambda image: v2.RandomInvert(1)(image).contiguous(),
            "solarize": lambda image: v2.RandomSolarize(128, 1)(image).contiguous(),
            "posterize": lambda image: v2.RandomPosterize(4, 1)(image).contiguous(),
            "jitter": lambda image: v2.ColorJitter(0.2, 0.2, 0.2, 0.0)(image).contiguous(),
            "normalize": lambda image: _sequence(
                v2, [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
            )(image).contiguous(),
            "pipeline": lambda image: _sequence(
                v2,
                [
                    v2.RandomCrop((15, 19)),
                    v2.Resize((13, 17), interpolation=InterpolationMode.BILINEAR, antialias=False),
                    v2.RandomHorizontalFlip(0.5),
                    v2.ColorJitter(0.2, 0.2, 0.2, 0.0),
                    v2.RandomAffine(10.0, interpolation=InterpolationMode.BILINEAR, fill=0),
                    v2.GaussianBlur(5, (1.1, 1.1)),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(MEAN, STD),
                ],
            )(image).contiguous(),
        }

    if backend == "albumentationsx":
        import albumentations as A
        import cv2

        jitter = A.ColorJitter(
            brightness_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2),
            hue_range=(0.0, 0.0),
            p=1,
        )
        blur = A.GaussianBlur(blur_range=(5, 5), sigma_range=(1.1, 1.1), p=1)
        affine_base = {
            "scale": (1.0, 1.0),
            "translate_percent": (0.0, 0.0),
            "shear": (0.0, 0.0),
        }

        def apply(transform: Any, image: np.ndarray) -> Any:
            if hasattr(transform, "set_random_seed"):
                transform.set_random_seed(SEED)
            value = transform(image=image)["image"]
            try:
                import torch

                if isinstance(value, torch.Tensor):
                    return value.contiguous()
            except ImportError:
                pass
            return np.ascontiguousarray(value)

        affine_identity = A.Affine(
            scale=affine_base["scale"],
            translate_percent=affine_base["translate_percent"],
            rotate=(0.0, 0.0),
            shear=affine_base["shear"],
            interpolation=cv2.INTER_LINEAR,
            fill=0,
            p=1,
        )
        pipeline = _sequence(
            A,
            [
                A.RandomCrop(15, 19, p=1),
                A.Resize(13, 17, interpolation=cv2.INTER_LINEAR, p=1),
                A.HorizontalFlip(p=0.5),
                jitter,
                A.Affine(
                    scale=affine_base["scale"],
                    translate_percent=affine_base["translate_percent"],
                    rotate=(-10.0, 10.0),
                    shear=affine_base["shear"],
                    interpolation=cv2.INTER_LINEAR,
                    fill=0,
                    p=1,
                ),
                blur,
                A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, p=1),
            ],
        )
        return {
            "native": lambda image: image,
            "flip": lambda image: apply(A.HorizontalFlip(p=1), image),
            "vertical_flip": lambda image: apply(A.VerticalFlip(p=1), image),
            "crop": lambda image, h, w: apply(A.RandomCrop(h, w, p=1), image),
            "center_crop": lambda image, h, w: apply(A.CenterCrop(h, w, p=1), image),
            "resize": lambda image, h, w: apply(
                A.Resize(h, w, interpolation=cv2.INTER_LINEAR, p=1), image
            ),
            "affine_identity": lambda image: apply(affine_identity, image),
            "blur": lambda image: apply(blur, image),
            "grayscale": lambda image: apply(
                A.ToGray(num_output_channels=3, method="weighted_average", p=1), image
            ),
            "invert": lambda image: apply(A.InvertImg(p=1), image),
            "solarize": lambda image: apply(
                A.Solarize(threshold_range=(128 / 255, 128 / 255), p=1), image
            ),
            "posterize": lambda image: apply(
                A.Posterize(num_bits=(4, 4), p=1),
                image,
            ),
            "jitter": lambda image: apply(jitter, image),
            "normalize": lambda image: apply(
                _sequence(A, [A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, p=1)]),
                image,
            ),
            "pipeline": lambda image: apply(pipeline, image),
        }

    import variopinta as R

    return {
        "native": lambda image: image,
        "flip": lambda image: R.Pipeline([R.HorizontalFlip(1.0)], seed=SEED).compile()(image),
        "vertical_flip": lambda image: R.Pipeline([R.VerticalFlip(1.0)], seed=SEED).compile()(
            image
        ),
        "crop": lambda image, h, w: R.Pipeline([R.RandomCrop(h, w)], seed=SEED).compile()(image),
        "center_crop": lambda image, h, w: R.Pipeline([R.CenterCrop(h, w)], seed=SEED).compile()(
            image
        ),
        "resize": lambda image, h, w: R.Pipeline([R.Resize(h, w)], seed=SEED).compile()(image),
        "affine_identity": lambda image: R.Pipeline([R.Affine(0.0)], seed=SEED).compile()(image),
        "blur": lambda image: R.Pipeline([R.GaussianBlur(5, 1.1)], seed=SEED).compile()(image),
        "grayscale": lambda image: R.Pipeline([R.Grayscale()], seed=SEED).compile()(image),
        "invert": lambda image: R.Pipeline([R.Invert()], seed=SEED).compile()(image),
        "solarize": lambda image: R.Pipeline([R.Solarize(128)], seed=SEED).compile()(image),
        "posterize": lambda image: R.Pipeline([R.Posterize(4)], seed=SEED).compile()(image),
        "jitter": lambda image: R.Pipeline([R.ColorJitter(0.2, 0.2, 0.2)], seed=SEED).compile()(
            image
        ),
        "normalize": lambda image: R.Pipeline([R.Normalize(MEAN, STD)], seed=SEED).compile()(image),
        "pipeline": lambda image: R.Pipeline(
            [
                R.RandomCrop(15, 19),
                R.Resize(13, 17),
                R.HorizontalFlip(0.5),
                R.ColorJitter(0.2, 0.2, 0.2),
                R.Affine(10.0),
                R.GaussianBlur(5, 1.1),
                R.Normalize(MEAN, STD),
            ],
            seed=SEED,
        ).compile()(image),
    }


def run_correctness_checks(backend: str) -> list[dict[str, Any]]:
    operations = _operations(backend)
    failures: list[str] = []
    limitations: list[str] = []
    cases = 0

    def check(condition: bool, label: str) -> None:
        nonlocal cases
        cases += 1
        if not condition:
            failures.append(label)

    for height, width in SHAPES:
        image = _image(height, width)
        native = operations["native"](image)

        flipped = _to_hwc(operations["flip"](native))
        check(np.array_equal(flipped, image[:, ::-1]), f"flip-{height}x{width}")
        vertically_flipped = _to_hwc(operations["vertical_flip"](native))
        check(
            np.array_equal(vertically_flipped, image[::-1]),
            f"vertical-flip-{height}x{width}",
        )

        grayscale = _to_hwc(operations["grayscale"](native))
        check(
            grayscale.shape == image.shape
            and grayscale.dtype == np.uint8
            and np.array_equal(grayscale[..., 0], grayscale[..., 1])
            and np.array_equal(grayscale[..., 1], grayscale[..., 2]),
            f"grayscale-{height}x{width}",
        )
        inverted = _to_hwc(operations["invert"](native))
        check(np.array_equal(inverted, 255 - image), f"invert-{height}x{width}")

        solarized = _to_hwc(operations["solarize"](native))
        solarize_expected = np.where(image > 128, 255 - image, image).astype(np.uint8)
        solarize_mask = image != 128
        check(
            np.array_equal(solarized[solarize_mask], solarize_expected[solarize_mask]),
            f"solarize-{height}x{width}",
        )
        posterized = _to_hwc(operations["posterize"](native))
        check(np.array_equal(posterized, image & 0xF0), f"posterize-{height}x{width}")

        normalized_value = operations["normalize"](native)
        normalized = _numpy(normalized_value)
        expected_hwc = (
            image.astype(np.float32) / 255.0 - np.asarray(MEAN, dtype=np.float32)
        ) / np.asarray(STD, dtype=np.float32)
        expected = (
            np.transpose(expected_hwc, (2, 0, 1)) if backend == "torchvision" else expected_hwc
        )
        expected_shape = (3, height, width) if backend == "torchvision" else (height, width, 3)
        check(
            normalized.shape == expected_shape
            and bool(normalized.flags.c_contiguous)
            and np.allclose(normalized, expected, atol=5e-6, rtol=0),
            f"normalize-{height}x{width}",
        )

        resized = _to_hwc(operations["resize"](native, height + 2, width + 3))
        check(resized.shape == (height + 2, width + 3, 3), f"resize-{height}x{width}")
        constant = np.full((height, width, 3), (37, 113, 229), dtype=np.uint8)
        resized_constant = _to_hwc(
            operations["resize"](operations["native"](constant), height + 2, width + 3)
        )
        check(
            np.max(np.abs(resized_constant.astype(int) - np.asarray((37, 113, 229)))) <= 1,
            f"resize-constant-{height}x{width}",
        )

        crop_height = max(1, height - 1)
        crop_width = max(1, width - 1)
        cropped_value = operations["crop"](native, crop_height, crop_width)
        cropped = _to_hwc(cropped_value)
        crop_matches = any(
            np.array_equal(cropped, image[y : y + crop_height, x : x + crop_width])
            for y in range(height - crop_height + 1)
            for x in range(width - crop_width + 1)
        )
        check(
            cropped.shape == (crop_height, crop_width, 3)
            and bool(_numpy(cropped_value).flags.c_contiguous)
            and crop_matches,
            f"crop-{height}x{width}",
        )
        centered_value = operations["center_crop"](native, crop_height, crop_width)
        centered = _to_hwc(centered_value)
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        check(
            np.array_equal(
                centered,
                image[top : top + crop_height, left : left + crop_width],
            )
            and bool(_numpy(centered_value).flags.c_contiguous),
            f"center-crop-{height}x{width}",
        )

        gray_plane = image[..., 0]
        gray = np.repeat(gray_plane[..., None], 3, axis=2)
        jittered = _to_hwc(operations["jitter"](operations["native"](gray)))
        check(
            jittered.shape == gray.shape
            and jittered.dtype == np.uint8
            and np.max(np.ptp(jittered.astype(np.int16), axis=2)) <= 1,
            f"jitter-gray-{height}x{width}",
        )

        identity = _to_hwc(operations["affine_identity"](native))
        check(
            identity.shape == image.shape
            and np.max(np.abs(identity.astype(int) - image.astype(int))) <= 1,
            f"affine-identity-{height}x{width}",
        )

        if height >= 5 and width >= 5:
            blurred = _to_hwc(operations["blur"](operations["native"](constant)))
            check(
                np.max(np.abs(blurred.astype(int) - constant.astype(int))) <= 1,
                f"blur-constant-{height}x{width}",
            )

    pipeline_input = _image(17, 23)
    pipeline = operations["pipeline"](operations["native"](pipeline_input))
    facts = _numpy(pipeline)
    check(
        facts.shape == ((3, 13, 17) if backend == "torchvision" else (13, 17, 3))
        and facts.dtype == np.float32
        and bool(facts.flags.c_contiguous)
        and bool(np.isfinite(facts).all()),
        "rectangular-full-pipeline",
    )
    if backend == "rust":
        import variopinta as R

        portability_shapes = (
            (1, 257),
            (257, 1),
            (2, 511),
            (511, 2),
            (3, 5),
            (5, 3),
            (13, 29),
            (29, 13),
            (31, 47),
            (47, 31),
            (127, 255),
            (255, 127),
            (223, 224),
            (224, 223),
            (239, 317),
            (317, 239),
            (17, 1023),
            (1023, 17),
        )
        for index, (height, width) in enumerate(portability_shapes):
            output_height = 7 + (index * 19) % 61
            output_width = 9 + (index * 23) % 67
            portable_pipeline = R.Pipeline(
                [
                    R.RandomCrop(max(1, height - 1), max(1, width - 1)),
                    R.Resize(output_height, output_width),
                    R.HorizontalFlip(0.5),
                    R.VerticalFlip(0.2),
                    R.ColorJitter(0.2, 0.2, 0.2),
                    R.Affine(10.0),
                    R.GaussianBlur(5, 1.1),
                    R.Grayscale(0.1),
                    R.Invert(0.1),
                    R.Solarize(128, 0.2),
                    R.Posterize(4, 0.2),
                    R.Normalize(MEAN, STD),
                ],
                seed=SEED,
            ).compile()
            portable = portable_pipeline(_image(height, width))
            check(
                portable.shape == (output_height, output_width, 3)
                and portable.dtype == np.float32
                and portable.flags.c_contiguous
                and bool(np.isfinite(portable).all()),
                f"portable-pipeline-{height}x{width}",
            )

        non_contiguous_source = _image(13, 18)[:, ::2, :]
        non_contiguous_result = R.Pipeline([R.HorizontalFlip(1.0)], seed=SEED)(
            non_contiguous_source
        )
        check(
            np.array_equal(non_contiguous_result, non_contiguous_source[:, ::-1, :])
            and bool(non_contiguous_result.flags.c_contiguous),
            "non-contiguous-input-copy",
        )
        try:
            R.Pipeline([R.HorizontalFlip(1.0)], seed=SEED)(np.zeros((3, 5, 3), dtype=np.float32))
        except TypeError:
            rejected_dtype = True
        else:
            rejected_dtype = False
        check(rejected_dtype, "invalid-input-dtype")

        invalid_configs = [
            lambda: [R.Resize(0, 3)],
            lambda: [R.RandomCrop(3, 0)],
            lambda: [R.CenterCrop(0, 3)],
            lambda: [R.HorizontalFlip(1.5)],
            lambda: [R.VerticalFlip(-0.1)],
            lambda: [R.ColorJitter(-0.1, 0.2, 0.2)],
            lambda: [R.Affine(-1.0)],
            lambda: [R.GaussianBlur(4, 1.1)],
            lambda: [R.Solarize(256)],
            lambda: [R.Posterize(0)],
            lambda: [R.Normalize(MEAN, (0.229, 0.0, 0.225))],
        ]
        for index, build_transforms in enumerate(invalid_configs):
            try:
                R.Pipeline(build_transforms(), seed=SEED)
            except ValueError:
                rejected = True
            else:
                rejected = False
            check(rejected, f"invalid-config-{index}")
        try:
            R.Pipeline([R.HorizontalFlip(1.0)], seed=SEED)(np.empty((0, 3, 3), dtype=np.uint8))
        except ValueError:
            rejected = True
        else:
            rejected = False
        check(rejected, "empty-input")
    return [
        {
            "kind": "correctness",
            "backend": backend,
            "cases": cases,
            "failures": failures,
            "limitations": limitations,
            "valid": not failures,
        }
    ]
