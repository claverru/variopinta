from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import variopinta as R
from PIL import Image


def image(height: int, width: int) -> np.ndarray:
    values = np.arange(height * width * 3, dtype=np.uint64)
    return ((values * 73 + values // 7 * 19) & 255).astype(np.uint8).reshape(height, width, 3)


def as_array(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()  # type: ignore[union-attr, no-any-return]
    return np.asarray(value)


class PipelineTests(unittest.TestCase):
    def test_transform_catalog_sets_and_binding_support_are_exact(self) -> None:
        from variopinta import _variopinta
        from variopinta.transforms import _TRANSFORM_CATALOG

        native = set(_variopinta.registered_transform_names())
        python = set(_TRANSFORM_CATALOG)
        public = {
            name
            for name in R.__all__
            if isinstance(getattr(R, name), type) and hasattr(getattr(R, name), "_spec")
        }
        self.assertEqual(native, python)
        self.assertEqual(native, public)
        for name, (_, fixture) in _TRANSFORM_CATALOG.items():
            with self.subTest(name=name):
                compiled = R.Compose([fixture(1.0)], seed=137).compile()
                self.assertEqual(compiled.explain()["steps"][0]["name"], name)

    def test_transform_catalog_conformance_matrix(self) -> None:
        from variopinta.transforms import _TRANSFORM_CATALOG

        for height, width in ((1, 7), (5, 1), (7, 11)):
            source = image(height, width * 2)[:, ::2]
            for name, (_, fixture) in _TRANSFORM_CATALOG.items():
                for probability in (0.0, 0.5, 1.0):
                    with self.subTest(
                        name=name, height=height, width=width, probability=probability
                    ):
                        reference = R.Compose([fixture(probability)], seed=137)
                        compiled = reference.compile()
                        for key in (3, 19):
                            expected = as_array(reference(source, key=key))
                            actual = as_array(compiled(source, key=key))
                            np.testing.assert_array_equal(actual, expected)
                            np.testing.assert_array_equal(
                                as_array(compiled(source, key=key)), actual
                            )
                            self.assertTrue(actual.flags.c_contiguous)
                            self.assertFalse(np.shares_memory(actual, source))

    def test_transform_catalog_compiled_pipelines_are_shareable(self) -> None:
        from variopinta.transforms import _TRANSFORM_CATALOG

        source = image(13, 19)
        keys = (3, 7, 19, 29)
        for name, (_, fixture) in _TRANSFORM_CATALOG.items():
            with self.subTest(name=name):
                compiled = R.Compose([fixture(1.0)], seed=137).compile()
                expected = [as_array(compiled(source, key=key)) for key in keys]
                with ThreadPoolExecutor(max_workers=len(keys)) as executor:
                    actual = list(
                        executor.map(lambda key, pipeline=compiled: pipeline(source, key=key), keys)
                    )
                for observed, wanted in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(as_array(observed), wanted)

    def pipeline(self) -> R.Compose:
        return R.Compose(
            [
                R.RandomCrop(13, 17),
                R.Resize(11, 15),
                R.HorizontalFlip(0.5),
                R.ColorJitter(0.2, 0.2, 0.2),
                R.Affine(10.0),
                R.GaussianBlur(5, 1.1),
                R.Normalize(),
            ],
            seed=137,
        )

    def test_reference_and_compiled_match(self) -> None:
        reference = self.pipeline()
        compiled = reference.compile()
        for height, width in ((15, 19), (17, 33), (63, 65), (127, 191)):
            source = image(height, width)
            np.testing.assert_array_equal(reference(source, key=29), compiled(source, key=29))

    def test_explicit_key_is_order_independent(self) -> None:
        pipeline = self.pipeline().compile()
        source = image(31, 47)
        first = pipeline(source, key=7)
        pipeline(source, key=99)
        np.testing.assert_array_equal(first, pipeline(source, key=7))

    def test_explain_reports_safe_optimization(self) -> None:
        reference = self.pipeline()
        compiled = reference.compile()
        self.assertEqual(compiled.explain()["schema_version"], 2)
        self.assertEqual(reference.explain()["fusions"], [])
        self.assertEqual(compiled.explain()["fusions"], [])
        self.assertEqual(
            compiled.explain()["unit_specializations"],
            ["ColorJitter:composed-color-matrix"],
        )
        self.assertEqual(compiled.explain()["optimizations"], ["input-copy-elision"])
        self.assertEqual(compiled.explain()["output_layout"], "HWC")
        self.assertEqual(reference.explain()["sampling"], "native-plan-before-execution")
        self.assertEqual(compiled.explain()["sampling"], "native-plan-before-execution")
        self.assertEqual(compiled.explain()["passes"], 7)
        self.assertEqual(compiled.explain()["pixel_passes"], 9)
        self.assertEqual(compiled.explain()["python_boundary"]["crossings_per_call"], 1)
        self.assertEqual(
            [copy["count"] for copy in compiled.explain()["copies"]], ["0-or-1", "0", "0"]
        )
        self.assertEqual(
            {buffer["name"] for buffer in compiled.explain()["buffers"]},
            {"input", "working-u8", "scratch-u8", "blur-temp", "output-f32"},
        )
        self.assertEqual(compiled.explain()["steps"][0]["category"], "geometry")
        self.assertEqual(compiled.explain()["steps"][-1]["execution"], "terminal")
        self.assertEqual(compiled.explain()["steps"][0]["input_materialization"], "borrowed-input")
        self.assertEqual(compiled.explain()["steps"][0]["kernel_form"], "borrowed-to-owned")
        affine = compiled.explain()["steps"][4]
        self.assertEqual(affine["kernel_form"], "owned-to-owned")
        self.assertEqual(affine["selection_reason"], "benchmark-policy:affine-copy-then-transform")

    def test_explain_distinguishes_conditional_and_unavailable_copy_elision(self) -> None:
        conditional = R.Compose([R.Resize(7, 11, p=0.5)], seed=137).compile().explain()
        unavailable = R.Compose([R.Grayscale()], seed=137).compile().explain()
        blocked = (
            R.Compose([R.CenterCrop(7, 11, p=0.0), R.Resize(5, 9)], seed=137).compile().explain()
        )
        conditional_dtype = R.Compose([R.Normalize(p=0.5)], seed=137).compile().explain()
        self.assertEqual(conditional["copies"][1]["count"], "0-or-1")
        self.assertEqual(conditional["copies"][1]["condition"], "sample-dependent")
        self.assertEqual(unavailable["copies"][1]["count"], "1")
        self.assertEqual(unavailable["optimizations"], [])
        self.assertEqual(unavailable["fusions"], [])
        self.assertEqual(blocked["copies"][1]["count"], "1")
        self.assertEqual(blocked["optimizations"], [])
        self.assertEqual(conditional_dtype["output_dtype"], "uint8-or-float32")
        self.assertEqual(conditional_dtype["buffers"][1]["condition"], "sample-dependent")

    def test_crop_resize_copy_elision_is_not_reported_as_fusion(self) -> None:
        explanation = R.Compose([R.CenterCrop(7, 11), R.Resize(5, 9)], seed=137).compile().explain()
        native_entry = next(
            copy for copy in explanation["copies"] if copy["stage"] == "native-entry"
        )
        self.assertEqual(explanation["fusions"], [])
        self.assertEqual(explanation["unit_specializations"], [])
        self.assertEqual(explanation["optimizations"], ["input-copy-elision"])
        self.assertEqual(native_entry["count"], "0")
        self.assertEqual(explanation["pixel_passes"], 2)

        crop = R.Compose([R.RandomCrop(5, 9)], seed=137).compile().explain()
        crop_entry = next(copy for copy in crop["copies"] if copy["stage"] == "native-entry")
        self.assertEqual(crop_entry["count"], "0")
        self.assertEqual(crop["optimizations"], ["input-copy-elision"])

        for transform in (
            R.VerticalFlip(1.0),
            R.Invert(),
            R.Solarize(128),
            R.Posterize(4),
        ):
            direct = R.Compose([transform], seed=137).compile().explain()
            direct_entry = next(
                copy for copy in direct["copies"] if copy["stage"] == "native-entry"
            )
            self.assertEqual(direct_entry["count"], "0")
            self.assertEqual(direct["optimizations"], ["input-copy-elision"])
            self.assertEqual(direct["fusions"], [])

        conditional_direct = R.Compose([R.Invert(0.5)], seed=137).compile().explain()
        conditional_entry = next(
            copy for copy in conditional_direct["copies"] if copy["stage"] == "native-entry"
        )
        self.assertEqual(conditional_entry["count"], "0-or-1")
        self.assertEqual(conditional_entry["condition"], "sample-dependent")

    def test_crop_entry_ignores_a_statically_skipped_following_resize(self) -> None:
        source = image(13, 17)
        for crop in (R.CenterCrop(7, 11), R.RandomCrop(7, 11)):
            with self.subTest(crop=crop):
                reference = R.Compose([crop, R.Resize(5, 9, p=0.0)], seed=137)
                compiled = reference.compile()
                expected = R.Compose([crop], seed=137)(source, key=11)
                np.testing.assert_array_equal(reference(source, key=11), expected)
                np.testing.assert_array_equal(compiled(source, key=11), expected)
                explanation = compiled.explain()
                native_entry = next(
                    copy for copy in explanation["copies"] if copy["stage"] == "native-entry"
                )
                self.assertEqual(native_entry["count"], "0")
                self.assertEqual(explanation["passes"], 1)
                self.assertEqual(explanation["pixel_passes"], 1)
                self.assertEqual(explanation["steps"][1]["status"], "never")
                self.assertEqual(explanation["steps"][1]["execution"], "skipped")

        controls = (
            ([R.CenterCrop(7, 11)], "0"),
            ([R.CenterCrop(7, 11), R.Invert(p=0.0)], "0"),
            ([R.CenterCrop(7, 11), R.Resize(5, 9)], "0"),
            ([R.CenterCrop(7, 11, p=0.0), R.Resize(5, 9)], "1"),
            ([R.CenterCrop(7, 11, p=0.5), R.Resize(5, 9)], "0-or-1"),
        )
        for transforms, expected_count in controls:
            with self.subTest(transforms=transforms):
                pipeline = R.Compose(transforms, seed=137)
                compiled = pipeline.compile()
                native_entry = next(
                    copy for copy in compiled.explain()["copies"] if copy["stage"] == "native-entry"
                )
                self.assertEqual(native_entry["count"], expected_count)
                for key in range(8):
                    np.testing.assert_array_equal(
                        pipeline(source, key=key), compiled(source, key=key)
                    )

    def test_explain_uses_the_effective_deterministic_route(self) -> None:
        explanation = R.Compose([R.Normalize(p=0.0), R.ToTorch()], seed=137).compile().explain()
        self.assertEqual(explanation["schema_version"], 2)
        self.assertEqual(explanation["passes"], 1)
        self.assertEqual(explanation["pixel_passes"], 1)
        self.assertEqual(explanation["fusions"], [])
        self.assertEqual(explanation["output_dtype"], "uint8")
        self.assertEqual(
            {buffer["name"] for buffer in explanation["buffers"]},
            {"input", "working-u8", "output-u8"},
        )
        normalize, to_torch = explanation["steps"]
        self.assertEqual(normalize["status"], "never")
        self.assertEqual(normalize["execution"], "skipped")
        self.assertEqual(normalize["pixel_passes"], 0)
        self.assertEqual(normalize["allocation"], "none")
        self.assertEqual(normalize["kernel_form"], "skipped")
        self.assertEqual(normalize["selection_reason"], "probability-zero")
        self.assertEqual(to_torch["status"], "always")
        native_entry = next(
            copy for copy in explanation["copies"] if copy["stage"] == "native-entry"
        )
        terminal = next(
            copy for copy in explanation["copies"] if copy["stage"] == "terminal-layout"
        )
        self.assertEqual((native_entry["count"], terminal["count"]), ("1", "1"))

    def test_random_resized_crop_contract_and_compilation(self) -> None:
        source = image(37, 53)[:, ::2]
        reference = R.Compose(
            [
                R.RandomResizedCrop(
                    11,
                    17,
                    scale=(0.2, 0.9),
                    ratio=(0.5, 2.0),
                    p=0.75,
                    antialias=True,
                )
            ],
            seed=137,
        )
        compiled = reference.compile()
        for key in range(20):
            np.testing.assert_array_equal(reference(source, key=key), compiled(source, key=key))

        output = R.Compose([R.RandomResizedCrop(11, 17)], seed=137).compile()(source, key=3)
        self.assertEqual(output.shape, (11, 17, 3))
        self.assertTrue(output.flags.c_contiguous)
        self.assertFalse(np.shares_memory(source, output))

        skipped = R.Compose([R.RandomResizedCrop(11, 17, p=0.0)], seed=137).compile()(source, key=3)
        np.testing.assert_array_equal(skipped, source)
        self.assertTrue(skipped.flags.c_contiguous)
        self.assertFalse(np.shares_memory(source, skipped))

        explanation = compiled.explain()
        native_entry = next(
            copy for copy in explanation["copies"] if copy["stage"] == "native-entry"
        )
        policies = {
            policy["name"]: policy["value"] for policy in explanation["steps"][0]["policies"]
        }
        self.assertEqual(explanation["pixel_passes"], 2)
        self.assertEqual(explanation["fusions"], [])
        self.assertEqual(native_entry["count"], "0-or-1")
        self.assertEqual(policies["sampling-attempts"], "10")
        self.assertEqual(policies["fallback"], "centered-ratio-clamp")

    def test_input_contract_and_ownership(self) -> None:
        source = image(13, 18)[:, ::2]
        snapshot = source.copy()
        output = R.Compose([R.HorizontalFlip(1.0)], seed=137)(source, key=0)
        np.testing.assert_array_equal(source, snapshot)
        np.testing.assert_array_equal(output, source[:, ::-1])
        self.assertTrue(output.flags.c_contiguous)
        self.assertFalse(np.shares_memory(source, output))

    def test_probabilities(self) -> None:
        source = image(7, 11)
        identity = R.Compose([R.HorizontalFlip(0.0)], seed=137).compile()
        flipped = R.Compose([R.HorizontalFlip(1.0)], seed=137).compile()
        np.testing.assert_array_equal(identity(source, key=0), source)
        np.testing.assert_array_equal(flipped(source, key=0), source[:, ::-1])

        invert = R.Compose([R.Invert(0.5)], seed=137)
        for key in range(20):
            np.testing.assert_array_equal(
                invert(source, key=key), invert.compile()(source, key=key)
            )

    def test_public_transform_parameter_branches_match_compiled_execution(self) -> None:
        transforms = (
            R.Resize(7, 9, interpolation=R.Interpolation.NEAREST),
            R.Resize(7, 9, interpolation=R.Interpolation.BILINEAR),
            R.Resize(7, 9, interpolation=R.Interpolation.BILINEAR, antialias=True),
            R.RandomCrop(11, 9, p=0.75),
            R.RandomResizedCrop(
                7,
                9,
                scale=(0.2, 0.9),
                ratio=(0.5, 2.0),
                p=0.75,
                interpolation=R.Interpolation.NEAREST,
            ),
            R.RandomResizedCrop(
                7,
                9,
                scale=(0.2, 0.9),
                ratio=(0.5, 2.0),
                p=0.75,
                antialias=True,
            ),
            R.CenterCrop(11, 9, p=0.75),
            R.PadIfNeeded(
                min_height=23,
                min_width=17,
                position=R.PadPosition.RANDOM,
                fill=(3, 5, 7),
                p=0.75,
            ),
            R.PadIfNeeded(
                pad_height_divisor=8,
                pad_width_divisor=7,
                border_mode=R.BorderMode.REFLECT101,
                p=0.75,
            ),
            R.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.1, 0.7),
                hole_width_range=(2, 9),
                fill=(3, 5, 7),
                p=0.75,
            ),
            R.HorizontalFlip(0.75),
            R.VerticalFlip(0.75),
            R.ColorJitter(0.3, 0.4, 0.5, 0.0, p=0.75),
            R.ColorJitter(0.3, 0.4, 0.5, 0.25, p=0.75),
            R.Affine(
                degrees=(-25.0, 35.0),
                translate=(0.4, 0.3),
                scale=(0.7, 1.5),
                shear=(-17.0, 19.0, -11.0, 13.0),
                interpolation=R.Interpolation.NEAREST,
                fill=(3, 5, 7),
                p=0.75,
            ),
            R.Affine(
                degrees=(-25.0, 35.0),
                translate=(0.4, 0.3),
                scale=(0.7, 1.5),
                shear=(-17.0, 19.0, -11.0, 13.0),
                border_mode=R.BorderMode.REFLECT101,
                p=0.75,
            ),
            R.GaussianBlur(5, 1.1, p=0.75),
            R.GaussianBlur(7, (0.5, 2.5), p=0.75),
            R.Grayscale(0.75),
            R.Invert(0.75),
            R.Solarize(0, 0.75),
            R.Solarize(255, 0.75),
            R.Posterize(1, 0.75),
            R.Posterize(8, 0.75),
            R.Normalize(
                mean=(-0.5, 0.25, 1.5),
                std=(0.5, 1.25, 2.0),
                max_pixel_value=127.5,
                p=0.75,
            ),
        )
        source = image(19, 23)[:, ::2]
        for transform in transforms:
            with self.subTest(transform=transform):
                reference = R.Compose([transform], seed=137)
                compiled = reference.compile()
                for key in range(12):
                    expected = reference(source, key=key)
                    actual = compiled(source, key=key)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertTrue(actual.flags.c_contiguous)
                    self.assertFalse(np.shares_memory(actual, source))

    def test_every_probabilistic_transform_can_skip_without_sampling_or_allocation(self) -> None:
        transforms = (
            R.Resize(100_000, 100_000, p=0.0),
            R.RandomCrop(100_000, 100_000, p=0.0),
            R.RandomResizedCrop(100_000, 100_000, p=0.0),
            R.CenterCrop(100_000, 100_000, p=0.0),
            R.PadIfNeeded(min_height=100_000, min_width=100_000, p=0.0),
            R.CoarseDropout(num_holes_range=(10_000_000, 10_000_000), p=0.0),
            R.HorizontalFlip(0.0),
            R.VerticalFlip(0.0),
            R.ColorJitter(p=0.0),
            R.Affine(p=0.0),
            R.RandomRotation(p=0.0),
            R.GaussianBlur(p=0.0),
            R.GaussianNoise(p=0.0),
            R.Sharpen(p=0.0),
            R.Perspective(p=0.0),
            R.GridDistortion(p=0.0),
            R.Grayscale(0.0),
            R.Invert(0.0),
            R.Solarize(p=0.0),
            R.Posterize(p=0.0),
            R.Normalize(p=0.0),
        )
        source = image(3, 7)[:, ::2]
        for transform in transforms:
            with self.subTest(transform=transform):
                reference = R.Compose([transform], seed=137)
                for pipeline in (reference, reference.compile()):
                    actual = pipeline(source, key=3)
                    np.testing.assert_array_equal(actual, source)
                    self.assertEqual(actual.dtype, np.uint8)
                    self.assertTrue(actual.flags.c_contiguous)
                    self.assertFalse(np.shares_memory(actual, source))

    def test_probability_and_normalize_validation_rejects_ambiguous_values(self) -> None:
        factories = (
            lambda p: R.Resize(3, 5, p=p),
            lambda p: R.RandomCrop(3, 5, p=p),
            lambda p: R.RandomResizedCrop(3, 5, p=p),
            lambda p: R.CenterCrop(3, 5, p=p),
            lambda p: R.PadIfNeeded(min_height=3, min_width=5, p=p),
            lambda p: R.CoarseDropout(p=p),
            lambda p: R.HorizontalFlip(p),
            lambda p: R.VerticalFlip(p),
            lambda p: R.ColorJitter(p=p),
            lambda p: R.Affine(p=p),
            lambda p: R.RandomRotation(p=p),
            lambda p: R.GaussianBlur(p=p),
            lambda p: R.GaussianNoise(p=p),
            lambda p: R.Sharpen(p=p),
            lambda p: R.Perspective(p=p),
            lambda p: R.GridDistortion(p=p),
            lambda p: R.Grayscale(p),
            lambda p: R.Invert(p),
            lambda p: R.Solarize(p=p),
            lambda p: R.Posterize(p=p),
            lambda p: R.Normalize(p=p),
        )
        for factory in factories:
            for probability in (True, -0.1, 1.1, float("nan"), float("inf"), "1"):
                with self.subTest(factory=factory, probability=probability):
                    with self.assertRaises(ValueError):
                        factory(probability)

        for arguments in (
            {"mean": [0.0, 0.0, 0.0]},
            {"mean": (0.0, True, 0.0)},
            {"std": (1.0, 0.0, 1.0)},
            {"std": (1.0, float("nan"), 1.0)},
            {"max_pixel_value": True},
            {"max_pixel_value": "255"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                R.Normalize(**arguments)

    def test_public_float_configuration_is_canonical_float32(self) -> None:
        tiny_probability = R.Invert(p=1e-50)
        self.assertEqual(tiny_probability.p, 0.0)
        pipeline = R.Compose([tiny_probability], seed=137)
        self.assertEqual(pipeline.transforms[0].p, 0.0)
        self.assertEqual(pipeline.explain()["steps"][0]["probability"], 0.0)
        self.assertEqual(pipeline.compile().explain()["steps"][0]["status"], "never")

        for value in (
            np.nextafter(0.0, 1.0),
            np.nextafter(0.5, 0.0),
            0.5,
            np.nextafter(0.5, 1.0),
            np.nextafter(1.0, 0.0),
            np.nextafter(1.0, 2.0),
        ):
            with self.subTest(probability=value):
                transform = R.Invert(p=float(value))
                self.assertEqual(transform.p, float(np.float32(value)))
                self.assertEqual(
                    R.Compose([transform], seed=137).explain()["steps"][0]["probability"],
                    transform.p,
                )

        smallest = float(np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32))
        self.assertEqual(R.GaussianBlur(5, smallest).sigma, (smallest, smallest))
        with self.assertRaisesRegex(ValueError, "positive"):
            R.GaussianBlur(5, smallest / 2.0)
        with self.assertRaisesRegex(ValueError, "float32"):
            R.GaussianBlur(5, float(np.finfo(np.float32).max) * 2.0)

        below_ninety = float(np.nextafter(90.0, 0.0))
        with self.assertRaisesRegex(ValueError, "strictly between"):
            R.Affine(shear=below_ninety)

        normalized = R.Normalize(
            mean=(0.1, 0.2, 0.3),
            std=(0.4, 0.5, 0.6),
            max_pixel_value=np.nextafter(255.0, 256.0),
        )
        self.assertEqual(normalized.mean, tuple(float(np.float32(v)) for v in (0.1, 0.2, 0.3)))
        self.assertEqual(normalized.std, tuple(float(np.float32(v)) for v in (0.4, 0.5, 0.6)))
        self.assertEqual(normalized.max_pixel_value, 255.0)

    def test_configuration_and_output_allocation_limits_fail_cleanly(self) -> None:
        with self.assertRaisesRegex(ValueError, "native backend limit"):
            R.Compose([R.Resize(2**32, 1)], seed=137)
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            R.Compose([R.GaussianBlur(2**63 - 1, 1.0)], seed=137)

        source = image(1, 1)
        huge_output = R.Compose([R.Resize(2_000_000_000, 2_000_000_000)], seed=137)
        for pipeline in (huge_output, huge_output.compile()):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                pipeline(source, key=3)

        huge_dropout = R.Compose(
            [
                R.CoarseDropout(
                    num_holes_range=(2**63 - 1, 2**63 - 1),
                    hole_height_range=(1, 1),
                    hole_width_range=(1, 1),
                    p=1.0,
                )
            ],
            seed=137,
        )
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            huge_dropout(source, key=3)

    def test_affine_q16_boundary_is_exact_and_panic_free(self) -> None:
        factories = (
            lambda degrees,
            interpolation=R.Interpolation.BILINEAR,
            border=R.BorderMode.CONSTANT: R.Affine(
                degrees=degrees,
                interpolation=interpolation,
                border_mode=border,
            ),
            lambda degrees,
            interpolation=R.Interpolation.BILINEAR,
            border=R.BorderMode.CONSTANT: R.RandomRotation(
                degrees=degrees,
                interpolation=interpolation,
                border_mode=border,
            ),
        )
        for height, width in ((1, 32_769), (1, 32_770), (32_769, 1), (32_770, 1)):
            source = image(height, width)
            for factory in factories:
                with self.subTest(shape=source.shape, factory=factory):
                    for interpolation, border in (
                        (R.Interpolation.BILINEAR, R.BorderMode.CONSTANT),
                        (R.Interpolation.NEAREST, R.BorderMode.CONSTANT),
                        (R.Interpolation.BILINEAR, R.BorderMode.REFLECT101),
                    ):
                        transform = factory((0.0, 0.0), interpolation, border)
                        reference = R.Compose([transform], seed=137)
                        compiled = reference.compile()
                        np.testing.assert_array_equal(reference(source, key=3), source)
                        np.testing.assert_array_equal(compiled(source, key=3), source)

                    transform = factory((0.25, 0.25))
                    reference = R.Compose([transform], seed=137)
                    compiled = reference.compile()
                    np.testing.assert_array_equal(compiled(source, key=3), reference(source, key=3))

    def test_invalid_configuration_is_eager(self) -> None:
        with self.assertRaises(ValueError):
            R.Resize(0, 3)
        with self.assertRaises(ValueError):
            R.GaussianBlur(4, 1.1)
        for sigma in (0.0, (0.0, 1.0), (1.2, 0.8), (1.0, float("inf"))):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                R.GaussianBlur(5, sigma)
        with self.assertRaises(ValueError):
            R.Solarize(256)
        with self.assertRaises(ValueError):
            R.Posterize(0)
        with self.assertRaises(TypeError):
            R.Compose([lambda value: value])

    def test_extended_catalog_is_correct_at_arbitrary_sizes(self) -> None:
        for height, width in ((1, 1), (3, 5), (17, 33), (63, 65)):
            source = image(height, width)
            vertical = R.Compose([R.VerticalFlip(1.0)], seed=137).compile()(source, key=1)
            np.testing.assert_array_equal(vertical, source[::-1])

            inverted = R.Compose([R.Invert()], seed=137).compile()(source, key=1)
            np.testing.assert_array_equal(inverted, 255 - source)

            solarized = R.Compose([R.Solarize(128)], seed=137).compile()(source, key=1)
            expected_solarized = np.where(source >= 128, 255 - source, source).astype(np.uint8)
            np.testing.assert_array_equal(solarized, expected_solarized)

            posterized = R.Compose([R.Posterize(4)], seed=137).compile()(source, key=1)
            np.testing.assert_array_equal(posterized, source & 0xF0)

            gray = R.Compose([R.Grayscale()], seed=137).compile()(source, key=1)
            self.assertTrue(np.array_equal(gray[..., 0], gray[..., 1]))
            self.assertTrue(np.array_equal(gray[..., 1], gray[..., 2]))
            expected_gray = (
                (
                    77 * source[..., 0].astype(np.uint16)
                    + 150 * source[..., 1].astype(np.uint16)
                    + 29 * source[..., 2].astype(np.uint16)
                    + 128
                )
                >> 8
            ).astype(np.uint8)
            np.testing.assert_array_equal(gray[..., 0], expected_gray)

            crop_height = max(1, height - 1)
            crop_width = max(1, width - 1)
            centered = R.Compose([R.CenterCrop(crop_height, crop_width)], seed=137).compile()(
                source, key=1
            )
            top = (height - crop_height) // 2
            left = (width - crop_width) // 2
            np.testing.assert_array_equal(
                centered, source[top : top + crop_height, left : left + crop_width]
            )

            resized_crop = R.Compose(
                [R.RandomResizedCrop(5, 7, scale=(0.2, 1.0), ratio=(0.5, 2.0))],
                seed=137,
            )
            resized = resized_crop(source, key=1)
            self.assertEqual(resized.shape, (5, 7, 3))
            self.assertTrue(resized.flags.c_contiguous)
            np.testing.assert_array_equal(resized, resized_crop.compile()(source, key=1))

    def test_extended_catalog_reference_matches_compiled(self) -> None:
        source = image(37, 53)
        reference = R.Compose(
            [
                R.CenterCrop(31, 47),
                R.VerticalFlip(0.5),
                R.Grayscale(0.5),
                R.Invert(0.5),
                R.Solarize(113, 0.5),
                R.Posterize(5, 0.5),
            ],
            seed=137,
        )
        for key in range(20):
            np.testing.assert_array_equal(
                reference(source, key=key), reference.compile()(source, key=key)
            )

    def test_geometry_policies_cross_the_native_boundary(self) -> None:
        source = image(7, 11)
        resize = R.Compose([R.Resize(13, 17, interpolation=R.Interpolation.NEAREST)], seed=137)
        np.testing.assert_array_equal(resize(source, key=3), resize.compile()(source, key=3))
        nearest_policies = {
            policy["name"]: policy["value"]
            for policy in resize.compile().explain()["steps"][0]["policies"]
        }
        self.assertEqual(nearest_policies["antialias"], "ignored")

        downscale_source = image(19, 17)
        fixed = R.Compose([R.Resize(7, 11)], seed=137)
        adaptive = R.Compose([R.Resize(7, 11, antialias=True)], seed=137)
        for transform in (fixed, adaptive):
            np.testing.assert_array_equal(
                transform(downscale_source, key=3),
                transform.compile()(downscale_source, key=3),
            )
        self.assertFalse(np.array_equal(fixed(downscale_source), adaptive(downscale_source)))

        policies = {
            policy["name"]: policy["value"]
            for policy in fixed.compile().explain()["steps"][0]["policies"]
        }
        self.assertEqual(policies["antialias"], "false")
        adaptive_policies = {
            policy["name"]: policy["value"]
            for policy in adaptive.compile().explain()["steps"][0]["policies"]
        }
        self.assertEqual(adaptive_policies["antialias"], "true")

        for border_mode in (R.BorderMode.CONSTANT, R.BorderMode.REFLECT101):
            transform = R.Compose(
                [
                    R.Affine(
                        0.0,
                        interpolation=R.Interpolation.BILINEAR,
                        border_mode=border_mode,
                        fill=(11, 13, 17),
                    )
                ],
                seed=137,
            )
            np.testing.assert_array_equal(transform(source, key=3), source)
            np.testing.assert_array_equal(
                transform(source, key=3), transform.compile()(source, key=3)
            )

    def test_pad_if_needed_positions_fill_and_divisors(self) -> None:
        source = image(2, 3)
        origins = {
            R.PadPosition.CENTER: (1, 2),
            R.PadPosition.TOP_LEFT: (0, 0),
            R.PadPosition.TOP_RIGHT: (0, 5),
            R.PadPosition.BOTTOM_LEFT: (3, 0),
            R.PadPosition.BOTTOM_RIGHT: (3, 5),
        }
        for position, (top, left) in origins.items():
            transform = R.Compose(
                [
                    R.PadIfNeeded(
                        min_height=5,
                        min_width=8,
                        position=position,
                        fill=(11, 13, 17),
                    )
                ],
                seed=137,
            )
            expected = np.empty((5, 8, 3), dtype=np.uint8)
            expected[...] = (11, 13, 17)
            expected[top : top + 2, left : left + 3] = source
            np.testing.assert_array_equal(transform(source, key=3), expected)
            np.testing.assert_array_equal(transform.compile()(source, key=3), expected)

        divisible = R.Compose([R.PadIfNeeded(pad_height_divisor=4, pad_width_divisor=5)], seed=137)
        output = divisible.compile()(image(5, 7), key=3)
        self.assertEqual(output.shape, (8, 10, 3))

        unchanged_source = image(7, 11)[:, ::2]
        unchanged = R.Compose([R.PadIfNeeded(min_height=3, min_width=5)], seed=137)
        output = unchanged.compile()(unchanged_source, key=3)
        np.testing.assert_array_equal(output, unchanged_source)
        self.assertTrue(output.flags.c_contiguous)
        self.assertFalse(np.shares_memory(output, unchanged_source))

    def test_pad_if_needed_reflect_random_and_compilation(self) -> None:
        source = np.repeat(np.arange(6, dtype=np.uint8).reshape(2, 3, 1), 3, axis=2)
        reflect = R.Compose(
            [
                R.PadIfNeeded(
                    min_height=4,
                    min_width=5,
                    position=R.PadPosition.CENTER,
                    border_mode=R.BorderMode.REFLECT101,
                )
            ],
            seed=137,
        )
        expected = np.array(
            [[4, 3, 4, 5, 4], [1, 0, 1, 2, 1], [4, 3, 4, 5, 4], [1, 0, 1, 2, 1]],
            dtype=np.uint8,
        )
        expected = np.repeat(expected[..., None], 3, axis=2)
        np.testing.assert_array_equal(reflect(source, key=3), expected)
        np.testing.assert_array_equal(reflect.compile()(source, key=3), expected)

        random_pad = R.Compose(
            [
                R.PadIfNeeded(
                    min_height=11,
                    min_width=17,
                    position=R.PadPosition.RANDOM,
                    fill=29,
                    p=0.75,
                )
            ],
            seed=137,
        )
        non_contiguous = image(7, 19)[:, ::2]
        for key in range(20):
            actual = random_pad.compile()(non_contiguous, key=key)
            np.testing.assert_array_equal(actual, random_pad(non_contiguous, key=key))
            self.assertTrue(actual.flags.c_contiguous)
            self.assertFalse(np.shares_memory(actual, non_contiguous))

        explanation = random_pad.compile().explain()
        policies = {
            policy["name"]: policy["value"] for policy in explanation["steps"][0]["policies"]
        }
        native_entry = next(
            copy for copy in explanation["copies"] if copy["stage"] == "native-entry"
        )
        self.assertEqual(policies["height"], "minimum-11")
        self.assertEqual(policies["position"], "random")
        self.assertEqual(policies["fill"], "[29,29,29]")
        self.assertEqual(native_entry["count"], "0-or-1")

    def test_coarse_dropout_pixel_and_fraction_ranges(self) -> None:
        source = image(7, 11)
        full = R.Compose(
            [
                R.CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(100, 200),
                    hole_width_range=(1.0, 1.0),
                    fill=(3, 5, 7),
                    p=1.0,
                )
            ],
            seed=137,
        )
        expected = np.empty_like(source)
        expected[...] = (3, 5, 7)
        np.testing.assert_array_equal(full(source, key=3), expected)
        np.testing.assert_array_equal(full.compile()(source, key=3), expected)

        ranged = R.Compose(
            [
                R.CoarseDropout(
                    num_holes_range=(2, 5),
                    hole_height_range=(0.1, 0.6),
                    hole_width_range=(2, 7),
                    fill=29,
                    p=0.75,
                )
            ],
            seed=137,
        )
        non_contiguous = image(17, 65)[:, ::2]
        compiled = ranged.compile()
        for key in range(20):
            expected = ranged(non_contiguous, key=key)
            actual = compiled(non_contiguous, key=key)
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertFalse(np.shares_memory(actual, non_contiguous))

        skipped = R.Compose([R.CoarseDropout(p=0.0)], seed=137).compile()(source, key=3)
        np.testing.assert_array_equal(skipped, source)
        self.assertFalse(np.shares_memory(skipped, source))

        explanation = ranged.compile().explain()
        policies = {
            policy["name"]: policy["value"] for policy in explanation["steps"][0]["policies"]
        }
        self.assertEqual(policies["holes"], "[2,5]")
        self.assertEqual(policies["hole-height"], "fraction-[0.1,0.6]")
        self.assertEqual(policies["hole-width"], "pixels-[2,7]")
        self.assertEqual(policies["fill"], "[29,29,29]")

    def test_geometry_policy_validation_is_eager(self) -> None:
        with self.assertRaises(TypeError):
            R.Resize(3, 5, interpolation="bilinear")
        with self.assertRaises(TypeError):
            R.Resize(3, 5, antialias=1)
        with self.assertRaises(TypeError):
            R.Affine(border_mode="constant")
        with self.assertRaises(ValueError):
            R.Affine(fill=(0, 1, 256))
        invalid_affines = (
            {"degrees": -1.0},
            {"degrees": (10.0, -10.0)},
            {"degrees": (0.0, float("inf"))},
            {"translate": (-0.1, 0.2)},
            {"translate": (0.2, 1.1)},
            {"translate": (0.2,)},
            {"scale": 0.0},
            {"scale": (1.2, 0.8)},
            {"shear": -1.0},
            {"shear": (10.0, -10.0)},
            {"shear": (-10.0, 10.0, 5.0, -5.0)},
            {"shear": 90.0},
        )
        for arguments in invalid_affines:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                R.Affine(**arguments)
        invalid_jitters = (
            {"brightness": -0.1},
            {"brightness": (-0.1, 1.0)},
            {"contrast": (1.2, 0.8)},
            {"saturation": (1.0, float("inf"))},
            {"hue": -0.1},
            {"hue": 0.6},
            {"hue": (-0.6, 0.2)},
            {"hue": (0.2, -0.2)},
        )
        for arguments in invalid_jitters:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                R.ColorJitter(**arguments)
        for scale in ((0.0, 1.0), (0.8, 0.2), (0.2, 1.1)):
            with self.assertRaises(ValueError):
                R.RandomResizedCrop(3, 5, scale=scale)
        with self.assertRaises(ValueError):
            R.RandomResizedCrop(3, 5, ratio=(2.0, 1.0))
        with self.assertRaises(TypeError):
            R.RandomResizedCrop(3, 5, interpolation="bilinear")
        with self.assertRaises(TypeError):
            R.RandomResizedCrop(3, 5, antialias=1)
        invalid_pads = (
            {},
            {"min_height": 3, "min_width": 5, "pad_height_divisor": 2},
            {"min_height": 0, "min_width": 5},
            {"pad_height_divisor": 2, "pad_width_divisor": 0},
        )
        for arguments in invalid_pads:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                R.PadIfNeeded(**arguments)
        with self.assertRaises(TypeError):
            R.PadIfNeeded(min_height=3, min_width=5, position="center")
        with self.assertRaises(TypeError):
            R.PadIfNeeded(min_height=3, min_width=5, border_mode="constant")
        with self.assertRaises(ValueError):
            R.PadIfNeeded(min_height=3, min_width=5, fill=(0, 1, 256))
        invalid_dropouts = (
            {"num_holes_range": (0, 2)},
            {"num_holes_range": (3, 2)},
            {"hole_height_range": (0, 2)},
            {"hole_height_range": (0.0, 0.2)},
            {"hole_width_range": (0.2, 1.1)},
            {"hole_width_range": (0.4, 0.2)},
        )
        for arguments in invalid_dropouts:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                R.CoarseDropout(**arguments)
        with self.assertRaises(TypeError):
            R.CoarseDropout(hole_height_range=(1, 0.2))
        with self.assertRaises(ValueError):
            R.CoarseDropout(fill=(0, 1, 256))

    def test_affine_parameter_surface_is_normalized_and_explained(self) -> None:
        scalar = R.Affine(12.0, scale=1.25, shear=7.0)
        self.assertEqual(scalar.degrees, (-12.0, 12.0))
        self.assertEqual(scalar.scale, (1.25, 1.25))
        self.assertEqual(scalar.shear, (-7.0, 7.0, 0.0, 0.0))

        transform = R.Affine(
            degrees=(-15.0, 25.0),
            translate=(0.4, 0.2),
            scale=(0.8, 1.3),
            shear=(-10.0, 20.0, -5.0, 7.0),
        )
        explanation = R.Compose([transform], seed=137).compile().explain()
        policies = {
            policy["name"]: policy["value"] for policy in explanation["steps"][0]["policies"]
        }
        self.assertEqual(policies["degrees"], "[-15,25]")
        self.assertEqual(policies["translate-fraction"], "[-0.4,0.4]x[-0.2,0.2]")
        self.assertEqual(policies["scale"], "[0.8,1.3]")
        self.assertEqual(policies["shear-degrees"], "[-10,20]x[-5,7]")

    def test_affine_full_surface_matches_reference_at_arbitrary_sizes(self) -> None:
        for interpolation in (R.Interpolation.NEAREST, R.Interpolation.BILINEAR):
            for border_mode in (R.BorderMode.CONSTANT, R.BorderMode.REFLECT101):
                transform = R.Compose(
                    [
                        R.Affine(
                            degrees=(-17.0, 23.0),
                            translate=(0.35, 0.2),
                            scale=(0.75, 1.4),
                            shear=(-13.0, 19.0, -9.0, 11.0),
                            interpolation=interpolation,
                            border_mode=border_mode,
                            fill=(11, 13, 17),
                        )
                    ],
                    seed=137,
                )
                compiled = transform.compile()
                for height, width in ((1, 1), (1, 7), (7, 1), (17, 33), (63, 65)):
                    source = image(height, width)[:, ::-1]
                    expected = transform(source, key=29)
                    actual = compiled(source, key=29)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.shape, source.shape)
                    self.assertEqual(actual.dtype, np.uint8)
                    self.assertTrue(actual.flags.c_contiguous)
                    self.assertFalse(np.shares_memory(source, actual))

    def test_random_rotation_reuses_affine_rasterization(self) -> None:
        scalar = R.RandomRotation(12.0)
        self.assertEqual(scalar.degrees, (-12.0, 12.0))
        source = image(17, 23)
        for interpolation in (R.Interpolation.NEAREST, R.Interpolation.BILINEAR):
            for border_mode in (R.BorderMode.CONSTANT, R.BorderMode.REFLECT101):
                rotation = R.Compose(
                    [
                        R.RandomRotation(
                            (17.0, 17.0),
                            interpolation=interpolation,
                            border_mode=border_mode,
                            fill=(3, 5, 7),
                        )
                    ],
                    seed=137,
                )
                affine = R.Compose(
                    [
                        R.Affine(
                            (17.0, 17.0),
                            interpolation=interpolation,
                            border_mode=border_mode,
                            fill=(3, 5, 7),
                        )
                    ],
                    seed=137,
                )
                np.testing.assert_array_equal(
                    rotation.compile()(source, key=29), affine.compile()(source, key=29)
                )
                explanation = rotation.compile().explain()["steps"][0]
                self.assertEqual(explanation["name"], "RandomRotation")
                self.assertIn({"name": "kernel", "value": "Affine"}, explanation["policies"])

    def test_gaussian_noise_has_stable_uint8_units_and_channel_policy(self) -> None:
        source = np.full((3, 5, 3), [10, 20, 30], dtype=np.uint8)
        shifted = R.Compose([R.GaussianNoise(mean=250.0, std=0.0)], seed=137)
        expected = np.full_like(source, 255)
        np.testing.assert_array_equal(shifted(source, key=3), expected)
        np.testing.assert_array_equal(shifted.compile()(source, key=3), expected)

        independent = R.Compose([R.GaussianNoise(std=10.0)], seed=137).compile()
        shared = R.Compose([R.GaussianNoise(std=10.0, per_channel=False)], seed=137).compile()
        np.testing.assert_array_equal(independent(source, key=7), independent(source, key=7))
        neutral = np.full((3, 5, 3), 128, dtype=np.uint8)
        shared_delta = shared(neutral, key=7).astype(np.int16) - neutral.astype(np.int16)
        np.testing.assert_array_equal(shared_delta[..., 0], shared_delta[..., 1])
        np.testing.assert_array_equal(shared_delta[..., 1], shared_delta[..., 2])

        with self.assertRaises(ValueError):
            R.GaussianNoise(std=-1.0)
        with self.assertRaises(TypeError):
            R.GaussianNoise(per_channel=1)

    def test_sharpen_identity_constant_and_impulse_oracles(self) -> None:
        source = image(7, 11)
        identity = R.Compose([R.Sharpen(alpha=0.0, lightness=5.0)], seed=137)
        np.testing.assert_array_equal(identity.compile()(source, key=3), source)

        constant = np.full((5, 7, 3), 73, dtype=np.uint8)
        sharpen = R.Compose([R.Sharpen(alpha=0.5, lightness=1.0)], seed=137)
        np.testing.assert_array_equal(sharpen.compile()(constant, key=3), constant)

        impulse = np.zeros((3, 3, 3), dtype=np.uint8)
        impulse[1, 1] = 100
        expected = np.zeros_like(impulse)
        expected[1, 1] = 255
        np.testing.assert_array_equal(sharpen.compile()(impulse, key=3), expected)
        with self.assertRaises(ValueError):
            R.Sharpen(alpha=1.1)
        with self.assertRaises(ValueError):
            R.Sharpen(lightness=-0.1)

    def test_perspective_identity_and_bounded_sampling(self) -> None:
        for interpolation in (R.Interpolation.NEAREST, R.Interpolation.BILINEAR):
            identity = R.Compose([R.Perspective(scale=0.0, interpolation=interpolation)], seed=137)
            for height, width in ((1, 1), (1, 7), (7, 1), (17, 23)):
                source = image(height, width)
                np.testing.assert_array_equal(identity(source, key=3), source)
                np.testing.assert_array_equal(identity.compile()(source, key=3), source)

        perspective = R.Compose([R.Perspective(scale=(0.49, 0.49))], seed=137)
        source = image(17, 23)
        np.testing.assert_array_equal(
            perspective.compile()(source, key=29), perspective(source, key=29)
        )
        with self.assertRaises(ValueError):
            R.Perspective(scale=0.5)

    def test_grid_distortion_identity_and_small_axes(self) -> None:
        identity = R.Compose([R.GridDistortion(num_steps=9, distort_limit=0.0)], seed=137)
        for height, width in ((1, 1), (1, 7), (7, 1), (7, 11)):
            source = image(height, width)
            np.testing.assert_array_equal(identity(source, key=3), source)
            np.testing.assert_array_equal(identity.compile()(source, key=3), source)

        distorted = R.Compose([R.GridDistortion(num_steps=4, distort_limit=(-0.8, 0.8))], seed=137)
        source = image(13, 19)
        np.testing.assert_array_equal(
            distorted.compile()(source, key=29), distorted(source, key=29)
        )
        policies = {
            policy["name"]: policy["value"]
            for policy in distorted.compile().explain()["steps"][0]["policies"]
        }
        self.assertEqual(policies["maps"], "positive-monotonic-anchored")
        self.assertEqual(policies["sampler"], "shared-inverse-q8")
        with self.assertRaises(ValueError):
            R.GridDistortion(num_steps=0)
        with self.assertRaises(ValueError):
            R.GridDistortion(distort_limit=1.0)

    def test_color_jitter_ranges_and_hue_are_normalized_and_explained(self) -> None:
        scalar = R.ColorJitter(0.2, 0.3, 0.4, 0.1)
        self.assertEqual(scalar.brightness, tuple(float(np.float32(v)) for v in (0.8, 1.2)))
        self.assertEqual(scalar.contrast, tuple(float(np.float32(v)) for v in (0.7, 1.3)))
        self.assertEqual(scalar.saturation, tuple(float(np.float32(v)) for v in (0.6, 1.4)))
        self.assertEqual(scalar.hue, tuple(float(np.float32(v)) for v in (-0.1, 0.1)))

        transform = R.ColorJitter(
            brightness=(0.7, 1.4),
            contrast=(0.8, 1.2),
            saturation=(0.5, 1.5),
            hue=(-0.25, 0.3),
        )
        explanation = R.Compose([transform], seed=137).compile().explain()
        policies = {
            policy["name"]: policy["value"] for policy in explanation["steps"][0]["policies"]
        }
        self.assertEqual(policies["brightness"], "[0.7,1.4]")
        self.assertEqual(policies["contrast"], "[0.8,1.2]")
        self.assertEqual(policies["saturation"], "[0.5,1.5]")
        self.assertEqual(policies["hue"], "[-0.25,0.3]")
        self.assertEqual(explanation["steps"][0]["pixel_passes"], 5)
        self.assertEqual(explanation["unit_specializations"], [])

    def test_color_jitter_hue_matches_reference_and_rotates_primaries(self) -> None:
        identity_ranges = {
            "brightness": (1.0, 1.0),
            "contrast": (1.0, 1.0),
            "saturation": (1.0, 1.0),
        }
        hue = R.Compose([R.ColorJitter(**identity_ranges, hue=(1.0 / 3.0, 1.0 / 3.0))], seed=137)
        primaries = np.array(
            [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [73, 73, 73]]], dtype=np.uint8
        )
        expected = np.array([[[0, 255, 0], [0, 0, 255], [255, 0, 0], [73, 73, 73]]], dtype=np.uint8)
        np.testing.assert_array_equal(hue(primaries, key=3), expected)
        np.testing.assert_array_equal(hue.compile()(primaries, key=3), expected)

        ranged = R.Compose(
            [
                R.ColorJitter(
                    brightness=(0.7, 1.4),
                    contrast=(0.8, 1.2),
                    saturation=(0.5, 1.5),
                    hue=(-0.25, 0.3),
                )
            ],
            seed=137,
        )
        compiled = ranged.compile()
        for height, width in ((1, 1), (1, 7), (7, 1), (17, 33), (63, 65)):
            source = image(height, width)[:, ::-1]
            expected = ranged(source, key=29)
            actual = compiled(source, key=29)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.shape, source.shape)
            self.assertEqual(actual.dtype, np.uint8)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertFalse(np.shares_memory(source, actual))

    def test_extreme_color_jitter_factors_are_safe_in_reference_and_compiled(self) -> None:
        identity = {
            "brightness": (1.0, 1.0),
            "contrast": (1.0, 1.0),
            "saturation": (1.0, 1.0),
        }
        source = image(3, 34)[:, ::2]
        configurations = (
            {**identity, "brightness": (1_000.0, 1_000.0)},
            {**identity, "contrast": (1_000.0, 1_000.0)},
            {**identity, "saturation": (1_000.0, 1_000.0)},
        )
        for configuration in configurations:
            with self.subTest(configuration=configuration):
                reference = R.Compose([R.ColorJitter(**configuration)], seed=137)
                expected = reference(source, key=29)
                actual = reference.compile()(source, key=29)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.uint8)
                self.assertTrue(actual.flags.c_contiguous)
                self.assertFalse(np.shares_memory(source, actual))

        brightness = R.Compose([R.ColorJitter(**configurations[0])], seed=137)
        positive = np.array([[[255, 2, 3], [0, 0, 0]]], dtype=np.uint8)
        expected = np.where(positive == 0, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(brightness(positive, key=29), expected)
        np.testing.assert_array_equal(brightness.compile()(positive, key=29), expected)
        self.assertEqual(
            brightness.compile().explain()["steps"][0]["fallback"],
            "runtime-avx2-or-numeric-safety-scalar",
        )

    def test_gaussian_blur_sigma_range_matches_reference_at_arbitrary_sizes(self) -> None:
        fixed = R.GaussianBlur(5, 1.1)
        expected_sigma = float(np.float32(1.1))
        self.assertEqual(fixed.sigma, (expected_sigma, expected_sigma))

        ranged = R.Compose([R.GaussianBlur(5, (0.6, 2.0))], seed=137)
        explanation = ranged.compile().explain()
        policies = {
            policy["name"]: policy["value"] for policy in explanation["steps"][0]["policies"]
        }
        self.assertEqual(policies["sigma"], "[0.6,2]")
        self.assertEqual(explanation["steps"][0]["allocation"], "workspace-u16+sampled-kernel")

        compiled = ranged.compile()
        for height, width in ((1, 1), (1, 7), (7, 1), (17, 33), (63, 65)):
            source = image(height, width)[:, ::-1]
            expected = ranged(source, key=29)
            actual = compiled(source, key=29)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.shape, source.shape)
            self.assertEqual(actual.dtype, np.uint8)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertFalse(np.shares_memory(source, actual))

    def test_wide_gaussian_blur_is_normalized_in_reference_and_compiled(self) -> None:
        source = np.full((3, 4, 3), 73, dtype=np.uint8)
        reference = R.Compose([R.GaussianBlur(101, 1_000_000.0)], seed=137)
        for pipeline in (reference, reference.compile()):
            actual = pipeline(source, key=29)
            np.testing.assert_array_equal(actual, source)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertFalse(np.shares_memory(source, actual))

    def test_to_torch_preserves_uint8_and_converts_to_contiguous_chw(self) -> None:
        import torch

        source = image(7, 11)[:, ::2]
        expected = np.moveaxis(np.ascontiguousarray(source), 2, 0)
        reference = R.Compose([R.ToTorch()], seed=137)
        compiled = reference.compile()
        for pipeline in (reference, compiled):
            output = pipeline(source, key=3)
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.dtype, torch.uint8)
            self.assertEqual(tuple(output.shape), (3, 7, 6))
            self.assertTrue(output.is_contiguous())
            self.assertEqual(output.device.type, "cpu")
            np.testing.assert_array_equal(output.numpy(), expected)

    def test_normalize_to_torch_preserves_float32_values(self) -> None:
        import torch

        source = image(7, 11)
        normalized = R.Compose([R.Normalize()], seed=137)(source, key=3)
        expected = np.moveaxis(normalized, 2, 0)
        reference = R.Compose([R.Normalize(), R.ToTorch()], seed=137)
        compiled = reference.compile()
        for pipeline in (reference, compiled):
            output = pipeline(source, key=3)
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(output.is_contiguous())
            np.testing.assert_array_equal(output.numpy(), expected)

        prefixed = R.Compose([R.Invert(), R.Normalize(), R.ToTorch()], seed=137)
        np.testing.assert_array_equal(
            prefixed.compile()(source, key=3).numpy(), prefixed(source, key=3).numpy()
        )

        reference_explanation = reference.explain()
        compiled_explanation = compiled.explain()
        self.assertEqual(reference_explanation["fusions"], [])
        self.assertEqual(compiled_explanation["fusions"], ["Normalize+ToTorch:direct-CHW"])
        self.assertEqual(
            compiled_explanation["steps"][0]["selection_reason"],
            "equivalence-tested-fusion:normalize-to-torch",
        )
        self.assertEqual(compiled_explanation["steps"][1]["kernel_form"], "fused-into-previous")
        self.assertEqual(reference_explanation["pixel_passes"], 2)
        self.assertEqual(compiled_explanation["pixel_passes"], 1)
        self.assertIn(
            "normalized-f32", {buffer["name"] for buffer in reference_explanation["buffers"]}
        )
        self.assertNotIn(
            "normalized-f32", {buffer["name"] for buffer in compiled_explanation["buffers"]}
        )
        terminal_copy = next(
            copy for copy in compiled_explanation["copies"] if copy["stage"] == "terminal-layout"
        )
        self.assertEqual(terminal_copy["count"], "0")
        self.assertEqual(terminal_copy["condition"], "always-fused")

        skipped = R.Compose([R.Normalize(p=0.0), R.ToTorch()], seed=137).compile()(source, key=3)
        self.assertEqual(skipped.dtype, torch.uint8)
        np.testing.assert_array_equal(skipped.numpy(), np.moveaxis(source, 2, 0))

    def test_to_torch_is_terminal_and_reports_the_output_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "ToTorch must be terminal"):
            R.Compose([R.ToTorch(), R.Invert()])
        with self.assertRaisesRegex(
            ValueError, "Normalize must be terminal or immediately precede ToTorch"
        ):
            R.Compose([R.Normalize(), R.Invert(), R.ToTorch()])

        explanation = R.Compose([R.Normalize(p=0.5), R.ToTorch()], seed=137).compile().explain()
        self.assertEqual(explanation["output_layout"], "CHW")
        self.assertEqual(explanation["output_dtype"], "uint8-or-float32")
        self.assertEqual(explanation["output"]["container"], "Torch CPU Tensor")
        self.assertEqual(explanation["input"]["layout"], "HWC")
        self.assertEqual(explanation["steps"][-1]["execution"], "conditional-fused-terminal")
        self.assertEqual(
            explanation["steps"][-1]["kernel_form"],
            "fused-into-previous-or-terminal-layout",
        )
        self.assertEqual(
            explanation["steps"][0]["output_slot"],
            "output-f32-chw-or-output-u8-chw",
        )
        self.assertEqual(
            explanation["steps"][1]["output_slot"],
            "output-f32-chw-or-output-u8-chw",
        )
        self.assertEqual(explanation["fusions"], ["Normalize+ToTorch:direct-CHW"])
        terminal_layout = next(
            copy for copy in explanation["copies"] if copy["stage"] == "terminal-layout"
        )
        self.assertEqual(terminal_layout["count"], "0-or-1")
        self.assertEqual(terminal_layout["condition"], "sample-dependent")
        self.assertEqual(explanation["copies"][-1]["stage"], "torch-adapter")
        self.assertEqual(explanation["copies"][-1]["count"], "0")

    def test_to_torch_has_a_clear_optional_dependency_error(self) -> None:
        pipeline = R.Compose([R.ToTorch()], seed=137)
        real_import = __import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch":
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_torch):
            with self.assertRaisesRegex(ImportError, "ToTorch requires PyTorch"):
                pipeline(image(3, 5), key=1)

    def test_read_image_returns_rgb_numpy(self) -> None:
        source = image(17, 23)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            Image.fromarray(source).save(path, quality=95)
            output = R.read_image(path)
            expected = np.asarray(Image.open(path).convert("RGB"))
        self.assertEqual(output.shape, source.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(output.flags.c_contiguous)
        np.testing.assert_array_equal(output, expected)

    def test_read_image_errors(self) -> None:
        with self.assertRaises(OSError):
            R.read_image("missing.jpg")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jpg"
            path.write_bytes(b"not a jpeg")
            with self.assertRaises(ValueError):
                R.read_image(path)


if __name__ == "__main__":
    unittest.main()
