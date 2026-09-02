from __future__ import annotations

import unittest

import numpy as np
import variopinta as R

from tests._helpers import image


class PipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
