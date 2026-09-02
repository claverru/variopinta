from __future__ import annotations

import unittest

import numpy as np
import variopinta as R

from tests._helpers import image


class GeometryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
