from __future__ import annotations

import unittest

import numpy as np
import variopinta as R

from tests._helpers import image


class GeometryTests(unittest.TestCase):
    def test_longest_max_size_dimensions_pixels_composition_and_explanation(self) -> None:
        cases = (
            ((480, 640), 256, (192, 256)),
            ((640, 480), 256, (256, 192)),
            ((3, 6), 5, (3, 5)),
            ((2, 3), 8, (5, 8)),
            ((1, 1000), 8, (1, 8)),
            ((7, 7), 5, (5, 5)),
            ((7, 11), 11, (7, 11)),
            ((1, 1), 1, (1, 1)),
            ((7, 11), 1, (1, 1)),
        )
        for (height, width), max_size, (output_height, output_width) in cases:
            source = image(height, width)
            transform = R.LongestMaxSize(
                max_size,
                interpolation=R.Interpolation.NEAREST,
            )
            expected = R.Pipeline(
                [R.Resize(output_height, output_width, interpolation=R.Interpolation.NEAREST)]
            )(source)
            reference = R.Pipeline([transform], seed=137)
            compiled = reference.compile()
            with self.subTest(shape=source.shape, max_size=max_size):
                np.testing.assert_array_equal(reference(source, key=3), expected)
                actual = compiled(source, key=3)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.shape, (output_height, output_width, 3))
                self.assertTrue(actual.flags.c_contiguous)
                self.assertFalse(np.shares_memory(actual, source))

        source = image(19, 17)[:, ::2]
        for interpolation, antialias in (
            (R.Interpolation.NEAREST, False),
            (R.Interpolation.BILINEAR, False),
            (R.Interpolation.BILINEAR, True),
        ):
            transform = R.LongestMaxSize(
                7,
                interpolation=interpolation,
                antialias=antialias,
            )
            expected = R.Pipeline(
                [R.Resize(7, 3, interpolation=interpolation, antialias=antialias)]
            )(source)
            actual = R.Pipeline([transform]).compile()(source)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(source, image(19, 17)[:, ::2])

        square = R.Pipeline(
            [
                R.Resize(3, 6, interpolation=R.Interpolation.NEAREST),
                R.LongestMaxSize(5, interpolation=R.Interpolation.NEAREST),
                R.PadIfNeeded(min_height=5, min_width=5, position=R.PadPosition.CENTER, fill=29),
            ]
        ).compile()(image(9, 13))
        self.assertEqual(square.shape, (5, 5, 3))
        self.assertTrue(np.all(square[0] == 29))
        self.assertTrue(np.all(square[-1] == 29))

        compiled = R.Pipeline([R.LongestMaxSize(8)], seed=137).compile()
        for shape, expected_shape in (((2, 3), (5, 8)), ((3, 2), (8, 5)), ((1, 1000), (1, 8))):
            self.assertEqual(compiled(image(*shape), key=3).shape[:2], expected_shape)

        conditional = R.Pipeline([R.LongestMaxSize(8, p=0.5)], seed=137)
        for key in range(20):
            np.testing.assert_array_equal(
                conditional(image(2, 3), key=key),
                conditional.compile()(image(2, 3), key=key),
            )
        skipped_source = image(3, 5)[:, ::2]
        skipped = R.Pipeline([R.LongestMaxSize(8, p=0.0)]).compile()(skipped_source)
        np.testing.assert_array_equal(skipped, skipped_source)
        self.assertTrue(skipped.flags.c_contiguous)
        self.assertFalse(np.shares_memory(skipped, skipped_source))

        with self.assertRaisesRegex(ValueError, "crop larger"):
            R.Pipeline([R.LongestMaxSize(5), R.CenterCrop(4, 5)]).compile()(image(3, 6))

        explanation = (
            R.Pipeline([R.LongestMaxSize(8, interpolation=R.Interpolation.NEAREST)])
            .compile()
            .explain()
        )
        step = explanation["steps"][0]
        policies = {policy["name"]: policy["value"] for policy in step["policies"]}
        self.assertEqual(step["name"], "LongestMaxSize")
        self.assertEqual(step["kernel_form"], "borrowed-to-owned")
        self.assertEqual(policies["max-size"], "8")
        self.assertEqual(policies["rounding"], "nearest-half-up")
        self.assertEqual(policies["upscaling"], "enabled")
        self.assertEqual(policies["antialias"], "ignored")

    def test_longest_max_size_validation_and_native_limits(self) -> None:
        for value in (0, -1, True, False, 1.5, "8", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                R.LongestMaxSize(value)  # type: ignore[arg-type]
        for kwargs in (
            {"interpolation": "nearest"},
            {"antialias": 1},
            {"p": -0.1},
            {"p": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                R.LongestMaxSize(8, **kwargs)  # type: ignore[arg-type]
        if np.dtype(np.intp).itemsize == 8:
            with self.assertRaises(ValueError):
                R.Pipeline([R.LongestMaxSize(2**32)]).compile()
            with self.assertRaises(ValueError):
                R.Pipeline([R.LongestMaxSize(2**32 - 1)]).compile()(image(1, 1))

    def test_random_resized_crop_contract_and_compilation(self) -> None:
        source = image(37, 53)[:, ::2]
        reference = R.Pipeline(
            [
                R.RandomResizedCrop(
                    11,
                    17,
                    area_range=(0.2, 0.9),
                    aspect_ratio_range=(0.5, 2.0),
                    p=0.75,
                    antialias=True,
                )
            ],
            seed=137,
        )
        compiled = reference.compile()
        for key in range(20):
            np.testing.assert_array_equal(reference(source, key=key), compiled(source, key=key))

        output = R.Pipeline([R.RandomResizedCrop(11, 17)], seed=137).compile()(source, key=3)
        self.assertEqual(output.shape, (11, 17, 3))
        self.assertTrue(output.flags.c_contiguous)
        self.assertFalse(np.shares_memory(source, output))

        skipped = R.Pipeline([R.RandomResizedCrop(11, 17, p=0.0)], seed=137).compile()(
            source, key=3
        )
        np.testing.assert_array_equal(skipped, source)
        self.assertTrue(skipped.flags.c_contiguous)
        self.assertFalse(np.shares_memory(source, skipped))

        explanation = compiled.explain()
        native_entry = next(
            copy for copy in explanation["targets"][0]["copies"] if copy["stage"] == "native-entry"
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
                degrees_range=degrees,
                interpolation=interpolation,
                border_mode=border,
            ),
            lambda degrees,
            interpolation=R.Interpolation.BILINEAR,
            border=R.BorderMode.CONSTANT: R.RandomRotation(
                degrees_range=degrees,
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
                        reference = R.Pipeline([transform], seed=137)
                        compiled = reference.compile()
                        np.testing.assert_array_equal(reference(source, key=3), source)
                        np.testing.assert_array_equal(compiled(source, key=3), source)

                    transform = factory((0.25, 0.25))
                    reference = R.Pipeline([transform], seed=137)
                    compiled = reference.compile()
                    np.testing.assert_array_equal(compiled(source, key=3), reference(source, key=3))

    def test_geometry_policies_cross_the_native_boundary(self) -> None:
        source = image(7, 11)
        resize = R.Pipeline([R.Resize(13, 17, interpolation=R.Interpolation.NEAREST)], seed=137)
        np.testing.assert_array_equal(resize(source, key=3), resize.compile()(source, key=3))
        nearest_policies = {
            policy["name"]: policy["value"]
            for policy in resize.compile().explain()["steps"][0]["policies"]
        }
        self.assertEqual(nearest_policies["antialias"], "ignored")

        downscale_source = image(19, 17)
        fixed = R.Pipeline([R.Resize(7, 11)], seed=137)
        adaptive = R.Pipeline([R.Resize(7, 11, antialias=True)], seed=137)
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
            transform = R.Pipeline(
                [
                    R.Affine(
                        degrees_range=(0.0, 0.0),
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
            transform = R.Pipeline(
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

        divisible = R.Pipeline([R.PadIfNeeded(height_divisor=4, width_divisor=5)], seed=137)
        output = divisible.compile()(image(5, 7), key=3)
        self.assertEqual(output.shape, (8, 10, 3))

        unchanged_source = image(7, 11)[:, ::2]
        unchanged = R.Pipeline([R.PadIfNeeded(min_height=3, min_width=5)], seed=137)
        output = unchanged.compile()(unchanged_source, key=3)
        np.testing.assert_array_equal(output, unchanged_source)
        self.assertTrue(output.flags.c_contiguous)
        self.assertFalse(np.shares_memory(output, unchanged_source))

    def test_pad_if_needed_reflect_random_and_compilation(self) -> None:
        source = np.repeat(np.arange(6, dtype=np.uint8).reshape(2, 3, 1), 3, axis=2)
        reflect = R.Pipeline(
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

        random_pad = R.Pipeline(
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
            copy for copy in explanation["targets"][0]["copies"] if copy["stage"] == "native-entry"
        )
        self.assertEqual(policies["height"], "minimum-11")
        self.assertEqual(policies["position"], "random")
        self.assertEqual(policies["fill"], "[29]")
        self.assertEqual(native_entry["count"], "0-or-1")

    def test_coarse_dropout_pixel_and_fraction_ranges(self) -> None:
        pixel_sizes = R.CoarseDropout(
            hole_height_range=(8.0, 16),
            hole_height_unit="pixels",
            hole_width_range=(3, 7.0),
            hole_width_unit="pixels",
        )
        fraction_sizes = R.CoarseDropout(
            hole_height_range=(1, 1),
            hole_height_unit="fraction",
            hole_width_range=(0.25, 1),
            hole_width_unit="fraction",
        )
        self.assertEqual(pixel_sizes.hole_height_range, (8, 16))
        self.assertEqual(pixel_sizes.hole_width_range, (3, 7))
        self.assertEqual(fraction_sizes.hole_height_range, (1.0, 1.0))
        self.assertEqual(
            fraction_sizes.hole_width_range,
            tuple(float(np.float32(value)) for value in (0.25, 1.0)),
        )
        self.assertNotEqual(
            R.CoarseDropout(hole_height_range=(1, 1), hole_height_unit="pixels"),
            R.CoarseDropout(hole_height_range=(1, 1), hole_height_unit="fraction"),
        )

        source = image(7, 11)
        full = R.Pipeline(
            [
                R.CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(100, 200),
                    hole_height_unit="pixels",
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

        ranged = R.Pipeline(
            [
                R.CoarseDropout(
                    num_holes_range=(2, 5),
                    hole_height_range=(0.1, 0.6),
                    hole_width_range=(2, 7),
                    hole_width_unit="pixels",
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

        skipped = R.Pipeline([R.CoarseDropout(p=0.0)], seed=137).compile()(source, key=3)
        np.testing.assert_array_equal(skipped, source)
        self.assertFalse(np.shares_memory(skipped, source))

        explanation = ranged.compile().explain()
        policies = {
            policy["name"]: policy["value"] for policy in explanation["steps"][0]["policies"]
        }
        self.assertEqual(policies["holes"], "[2,5]")
        self.assertEqual(policies["hole-height"], "fraction-[0.1,0.6]")
        self.assertEqual(policies["hole-width"], "pixels-[2,7]")
        self.assertEqual(policies["fill"], "[29]")

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
            {"degrees_range": -1.0},
            {"degrees_range": (10.0, -10.0)},
            {"degrees_range": (0.0, float("inf"))},
            {"translate_max_fraction": (-0.1, 0.2)},
            {"translate_max_fraction": (0.2, 1.1)},
            {"translate_max_fraction": (0.2,)},
            {"scale_range": 0.0},
            {"scale_range": (1.2, 0.8)},
            {"shear_x_range": -1.0},
            {"shear_x_range": (10.0, -10.0)},
            {"shear_y_range": (5.0, -5.0)},
            {"shear_x_range": (90.0, 90.0)},
        )
        for arguments in invalid_affines:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                R.Affine(**arguments)
        invalid_jitters = (
            {"brightness_range": -0.1},
            {"brightness_range": (-0.1, 1.0)},
            {"contrast_range": (1.2, 0.8)},
            {"saturation_range": (1.0, float("inf"))},
            {"hue_range": -0.1},
            {"hue_range": 0.6},
            {"hue_range": (-0.6, 0.2)},
            {"hue_range": (0.2, -0.2)},
        )
        for arguments in invalid_jitters:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                R.ColorJitter(**arguments)
        for scale in ((0.0, 1.0), (0.8, 0.2), (0.2, 1.1)):
            with self.assertRaises(ValueError):
                R.RandomResizedCrop(3, 5, area_range=scale)
        with self.assertRaises(ValueError):
            R.RandomResizedCrop(3, 5, aspect_ratio_range=(2.0, 1.0))
        with self.assertRaises(TypeError):
            R.RandomResizedCrop(3, 5, interpolation="bilinear")
        with self.assertRaises(TypeError):
            R.RandomResizedCrop(3, 5, antialias=1)
        invalid_pads = (
            {},
            {"min_height": 3, "min_width": 5, "height_divisor": 2},
            {"min_height": 0, "min_width": 5},
            {"height_divisor": 2, "width_divisor": 0},
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
        with self.assertRaises(ValueError):
            R.CoarseDropout(hole_height_range=(1, 0.2), hole_height_unit="pixels")
        for arguments in (
            {"hole_height_unit": "percent"},
            {"hole_height_range": (True, 2), "hole_height_unit": "pixels"},
            {"hole_height_range": (8.5, 16), "hole_height_unit": "pixels"},
            {"hole_width_range": (float("inf"), 2), "hole_width_unit": "pixels"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                R.CoarseDropout(**arguments)
        with self.assertRaises(ValueError):
            R.CoarseDropout(fill=(0, 1, 256))

    def test_affine_parameter_surface_is_normalized_and_explained(self) -> None:
        configured = R.Affine(
            degrees_range=(-12.0, 12.0),
            scale_range=(1.25, 1.25),
            shear_x_range=(-7.0, 7.0),
        )
        self.assertEqual(configured.degrees_range, (-12.0, 12.0))
        self.assertEqual(configured.scale_range, (1.25, 1.25))
        self.assertEqual(configured.shear_x_range, (-7.0, 7.0))
        self.assertEqual(configured.shear_y_range, (0.0, 0.0))

        transform = R.Affine(
            degrees_range=(-15.0, 25.0),
            translate_max_fraction=(0.4, 0.2),
            scale_range=(0.8, 1.3),
            shear_x_range=(-10.0, 20.0),
            shear_y_range=(-5.0, 7.0),
        )
        explanation = R.Pipeline([transform], seed=137).compile().explain()
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
                transform = R.Pipeline(
                    [
                        R.Affine(
                            degrees_range=(-17.0, 23.0),
                            translate_max_fraction=(0.35, 0.2),
                            scale_range=(0.75, 1.4),
                            shear_x_range=(-13.0, 19.0),
                            shear_y_range=(-9.0, 11.0),
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
        configured = R.RandomRotation((-12.0, 12.0))
        self.assertEqual(configured.degrees_range, (-12.0, 12.0))
        source = image(17, 23)
        for interpolation in (R.Interpolation.NEAREST, R.Interpolation.BILINEAR):
            for border_mode in (R.BorderMode.CONSTANT, R.BorderMode.REFLECT101):
                rotation = R.Pipeline(
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
                affine = R.Pipeline(
                    [
                        R.Affine(
                            degrees_range=(17.0, 17.0),
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
            identity = R.Pipeline(
                [R.Perspective(distortion_scale_range=(0.0, 0.0), interpolation=interpolation)],
                seed=137,
            )
            for height, width in ((1, 1), (1, 7), (7, 1), (17, 23)):
                source = image(height, width)
                np.testing.assert_array_equal(identity(source, key=3), source)
                np.testing.assert_array_equal(identity.compile()(source, key=3), source)

        perspective = R.Pipeline([R.Perspective(distortion_scale_range=(0.49, 0.49))], seed=137)
        source = image(17, 23)
        np.testing.assert_array_equal(
            perspective.compile()(source, key=29), perspective(source, key=29)
        )
        with self.assertRaises(ValueError):
            R.Perspective(distortion_scale_range=(0.5, 0.5))

    def test_grid_distortion_identity_and_small_axes(self) -> None:
        identity = R.Pipeline(
            [R.GridDistortion(num_steps=9, distortion_range=(0.0, 0.0))], seed=137
        )
        for height, width in ((1, 1), (1, 7), (7, 1), (7, 11)):
            source = image(height, width)
            np.testing.assert_array_equal(identity(source, key=3), source)
            np.testing.assert_array_equal(identity.compile()(source, key=3), source)

        distorted = R.Pipeline(
            [R.GridDistortion(num_steps=4, distortion_range=(-0.8, 0.8))], seed=137
        )
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
            R.GridDistortion(distortion_range=(-1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
