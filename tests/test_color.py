from __future__ import annotations

import platform
import unittest

import numpy as np
import variopinta as R

from tests._helpers import image


class ColorTests(unittest.TestCase):
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
        self.assertEqual(explanation["steps"][0]["pixel_passes"], 2)
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
            (
                "runtime-avx2-or-numeric-safety-scalar"
                if platform.machine() in {"x86_64", "AMD64"}
                else "portable-or-numeric-safety-scalar"
            ),
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


if __name__ == "__main__":
    unittest.main()
