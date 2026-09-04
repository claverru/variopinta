from __future__ import annotations

import platform
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import variopinta as R

from tests._helpers import as_array, image


class CatalogTests(unittest.TestCase):
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
                compiled = R.Pipeline([fixture(1.0)], seed=137).compile()
                self.assertEqual(compiled.explain()["steps"][0]["name"], name)

    def test_fallback_explanations_match_the_native_architecture(self) -> None:
        from variopinta.transforms import _TRANSFORM_CATALOG

        explanations = [
            R.Pipeline([fixture(1.0)], seed=137).compile().explain()
            for _, fixture in _TRANSFORM_CATALOG.values()
        ]
        fallbacks = [
            fallback for explanation in explanations for fallback in explanation["fallbacks"]
        ]
        if platform.machine() in {"x86_64", "AMD64"}:
            self.assertTrue(any("avx2" in fallback for fallback in fallbacks))
        else:
            self.assertFalse(any("avx2" in fallback for fallback in fallbacks))

    def test_transform_catalog_conformance_matrix(self) -> None:
        from variopinta.transforms import _TRANSFORM_CATALOG

        for height, width in ((1, 7), (5, 1), (7, 11)):
            source = image(height, width * 2)[:, ::2]
            for name, (_, fixture) in _TRANSFORM_CATALOG.items():
                for probability in (0.0, 0.5, 1.0):
                    with self.subTest(
                        name=name, height=height, width=width, probability=probability
                    ):
                        reference = R.Pipeline([fixture(probability)], seed=137)
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
                compiled = R.Pipeline([fixture(1.0)], seed=137).compile()
                expected = [as_array(compiled(source, key=key)) for key in keys]
                with ThreadPoolExecutor(max_workers=len(keys)) as executor:
                    actual = list(
                        executor.map(lambda key, pipeline=compiled: pipeline(source, key=key), keys)
                    )
                for observed, wanted in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(as_array(observed), wanted)

    def test_extended_catalog_is_correct_at_arbitrary_sizes(self) -> None:
        for height, width in ((1, 1), (3, 5), (17, 33), (63, 65)):
            source = image(height, width)
            vertical = R.Pipeline([R.VerticalFlip(1.0)], seed=137).compile()(source, key=1)
            np.testing.assert_array_equal(vertical, source[::-1])

            inverted = R.Pipeline([R.Invert()], seed=137).compile()(source, key=1)
            np.testing.assert_array_equal(inverted, 255 - source)

            solarized = R.Pipeline([R.Solarize(128)], seed=137).compile()(source, key=1)
            expected_solarized = np.where(source >= 128, 255 - source, source).astype(np.uint8)
            np.testing.assert_array_equal(solarized, expected_solarized)

            posterized = R.Pipeline([R.Posterize(4)], seed=137).compile()(source, key=1)
            np.testing.assert_array_equal(posterized, source & 0xF0)

            gray = R.Pipeline([R.Grayscale()], seed=137).compile()(source, key=1)
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
            centered = R.Pipeline([R.CenterCrop(crop_height, crop_width)], seed=137).compile()(
                source, key=1
            )
            top = (height - crop_height) // 2
            left = (width - crop_width) // 2
            np.testing.assert_array_equal(
                centered, source[top : top + crop_height, left : left + crop_width]
            )

            resized_crop = R.Pipeline(
                [R.RandomResizedCrop(5, 7, scale=(0.2, 1.0), ratio=(0.5, 2.0))],
                seed=137,
            )
            resized = resized_crop(source, key=1)
            self.assertEqual(resized.shape, (5, 7, 3))
            self.assertTrue(resized.flags.c_contiguous)
            np.testing.assert_array_equal(resized, resized_crop.compile()(source, key=1))

    def test_extended_catalog_reference_matches_compiled(self) -> None:
        source = image(37, 53)
        reference = R.Pipeline(
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


if __name__ == "__main__":
    unittest.main()
