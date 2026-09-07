from __future__ import annotations

import dataclasses
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import variopinta as vp
from variopinta.transforms import _TRANSFORM_CATALOG


def gray(height=7, width=11):
    return (np.arange(height * width).reshape(height, width) * 73 % 256).astype(np.uint8)


class GrayscaleImageTests(unittest.TestCase):
    def test_catalog_rank_ownership_and_reference_equality(self):
        for h, w in ((1, 1), (1, 7), (5, 1), (7, 11), (17, 35)):
            source = gray(h, w * 2)[:, ::2]
            for name, (_, fixture) in _TRANSFORM_CATALOG.items():
                for probability in (0.0, 0.5, 1.0):
                    transform = fixture(probability)
                    if hasattr(transform, "fill"):
                        transform = dataclasses.replace(transform, fill=17)
                    if name == "Normalize":
                        transform = dataclasses.replace(transform, mean=0.5, std=0.5)
                    reference = vp.Pipeline([transform], seed=137)
                    compiled = reference.compile()
                    for key in (3, 19):
                        with self.subTest(name=name, shape=source.shape, p=probability, key=key):
                            actual = compiled(source, key=key)
                            np.testing.assert_array_equal(actual, reference(source, key=key))
                            np.testing.assert_array_equal(
                                actual, compiled(source[..., None], key=key)[..., 0]
                            )
                            self.assertEqual(actual.ndim, 2)
                            self.assertTrue(actual.flags.c_contiguous)
                            self.assertFalse(np.shares_memory(actual, source))

    def test_normalization_broadcast_and_defaults(self):
        source = np.array([[0, 127, 255]], dtype=np.uint8)
        for mean in (0.5, (0.5,), [0.5]):
            for std in (0.5, (0.5,), [0.5]):
                pipeline = vp.Pipeline([vp.Normalize(mean=mean, std=std)])
                for p in (pipeline, pipeline.compile()):
                    result = p(source)
                    self.assertEqual(result.shape, source.shape)
                    self.assertEqual(result.dtype, np.float32)
                    np.testing.assert_allclose(
                        result, source.astype(np.float32) / 127.5 - 1, atol=1e-7
                    )
        for transform in (
            vp.Normalize(),
            vp.Normalize(mean=(0.5,) * 3, std=0.5),
            vp.Normalize(mean=0.5, std=(0.5,) * 3),
        ):
            with self.assertRaisesRegex(ValueError, "Normalize.*requires RGB"):
                vp.Pipeline([transform]).compile()(source)
        for value in (True, False, [], (1, 2), (1, 2, 3, 4), [False], float("inf")):
            for parameter in ("mean", "std"):
                with self.subTest(value=value, parameter=parameter), self.assertRaises(ValueError):
                    vp.Normalize(**{parameter: value})
        for value in (0, -1, 1e-50):
            with self.assertRaises(ValueError):
                vp.Normalize(std=value)

    def test_fill_broadcast_and_inactive_constraints(self):
        source = gray(2, 3)
        transforms = [
            vp.PadIfNeeded(min_height=4, min_width=5, fill=value) for value in (7, (7,), [7])
        ]
        results = [vp.Pipeline([transform]).compile()(source) for transform in transforms]
        for result in results:
            self.assertEqual(int(result[0, 0]), 7)
            np.testing.assert_array_equal(result, results[0])
        for name in (
            "PadIfNeeded",
            "CoarseDropout",
            "Affine",
            "RandomRotation",
            "Perspective",
            "GridDistortion",
        ):
            transform = _TRANSFORM_CATALOG[name][1](1.0)
            transform = dataclasses.replace(transform, fill=(7, 7, 7))
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, name + ".*requires RGB"),
            ):
                vp.Pipeline([transform]).compile()(source)
            inactive = dataclasses.replace(transform, p=0.0)
            np.testing.assert_array_equal(vp.Pipeline([inactive]).compile()(source), source)
        np.testing.assert_array_equal(vp.Pipeline([vp.Normalize(p=0)]).compile()(source), source)
        for value in (True, [True], [], (1, 2), (1, 2, 3, 4), -1, 256):
            with self.assertRaises(ValueError):
                vp.PadIfNeeded(min_height=1, min_width=1, fill=value)

    def test_identity_color_and_noise(self):
        source = gray()
        for transform in (
            vp.Grayscale(),
            vp.ColorJitter(
                brightness_range=(1.0, 1.0),
                contrast_range=(1.0, 1.0),
                saturation_range=(0.19999998807907104, 1.8),
                hue_range=(-0.4, 0.4),
            ),
        ):
            result = vp.Pipeline([transform]).compile()(source, key=3)
            np.testing.assert_array_equal(result, source)
            self.assertFalse(np.shares_memory(result, source))
        a = vp.Pipeline(
            [vp.GaussianNoise(mean_range=(2, 2), std_range=(11, 11), per_channel=True)],
            seed=137,
        ).compile()
        b = vp.Pipeline(
            [vp.GaussianNoise(mean_range=(2, 2), std_range=(11, 11), per_channel=False)],
            seed=137,
        ).compile()
        np.testing.assert_array_equal(a(source, key=7), b(source, key=7))
        rgb = np.repeat(source[..., None], 3, axis=2)
        np.testing.assert_array_equal(b(source, key=7), b(rgb, key=7)[..., 0])

    def test_gray_jitter_execution_reporting(self):
        for carrier in (vp.Array(), vp.Encoded(), vp.Path()):
            target = vp.Image(
                carrier,
                name="image",
                output_specs=vp.ReturnArray(name="value"),
                decode_mode=None if isinstance(carrier, vp.Array) else "gray",
            )
            for brightness, contrast in ((0, 0), (0.2, 0), (0, 0.4)):
                for probability in (0.0, 0.5, 1.0):
                    reference = vp.Pipeline(
                        [
                            vp.ColorJitter(
                                brightness_range=(1.0 - brightness, 1.0 + brightness),
                                contrast_range=(1.0 - contrast, 1.0 + contrast),
                                saturation_range=(0.19999998807907104, 1.8),
                                hue_range=(-0.4, 0.4),
                                p=probability,
                            )
                        ],
                        targets=[target],
                    )
                    for pipeline in (reference, reference.compile()):
                        with self.subTest(
                            input_spec=carrier,
                            brightness_range=brightness,
                            contrast_range=contrast,
                            p=probability,
                            pipeline=type(pipeline).__name__,
                        ):
                            explanation = pipeline.explain()
                            alternatives = explanation["targets"][0]["channel_alternatives"]
                            gray_alternative = alternatives[0]
                            step = gray_alternative["steps"][0]
                            if probability == 0 or brightness == contrast == 0:
                                self.assertEqual(
                                    step["execution"], "skipped" if probability == 0 else "identity"
                                )
                                self.assertEqual(step["pixel_passes"], 0)
                                self.assertEqual(step["fallback"], "none")
                                self.assertEqual(gray_alternative["pixel_passes"], 0)
                                if not isinstance(carrier, vp.Array):
                                    self.assertEqual(explanation["pixel_passes"], 0)
                                    self.assertEqual(explanation["fallbacks"], [])
                            else:
                                self.assertEqual(
                                    step["execution"],
                                    "gray-brightness-contrast-preserve-numeric-barriers",
                                )
                                self.assertGreater(step["pixel_passes"], 0)
                                self.assertEqual(step["fallback"], "portable-scalar")
                            if isinstance(carrier, vp.Array) and probability != 0:
                                self.assertGreater(alternatives[1]["pixel_passes"], 0)

    def test_mixed_targets_share_geometry_and_alternate_calls(self):
        image = vp.Image(name="image", output_specs=vp.ReturnArray(name="value"))
        second = vp.Image(name="second", output_specs=vp.ReturnArray(name="value"))
        mask = vp.Mask(name="mask", output_specs=vp.ReturnArray(name="value"))
        pipeline = vp.Pipeline(
            [vp.RandomCrop(3, 5), vp.HorizontalFlip(p=0.5), vp.VerticalFlip(p=0.5)],
            targets=[image, second, mask],
            seed=137,
        ).compile()
        source = gray()
        rgb = np.repeat(source[..., None], 3, axis=2)
        for first, other in ((source, rgb), (rgb, source[..., None]), (source[..., None], source)):
            result = pipeline(
                image=image.bind(first), second=second.bind(other), mask=mask.bind(source), key=7
            )
            a, b, labels = result.image.value, result.second.value, result.mask.value
            np.testing.assert_array_equal(a if a.ndim == 2 else a[..., 0], labels)
            np.testing.assert_array_equal(b if b.ndim == 2 else b[..., 0], labels)
        jitter = vp.Pipeline(
            [
                vp.ColorJitter(
                    brightness_range=(0.8, 1.2), contrast_range=(0.6, 1.4), hue_range=(-0.3, 0.3)
                ),
                vp.RandomCrop(3, 5),
            ],
            seed=137,
        ).compile()
        for key in (3, 7, 19):
            np.testing.assert_array_equal(
                jitter(source, key=key), jitter(source[..., None], key=key)[..., 0]
            )

    def test_rejection_preserves_next_key_and_plan(self):
        transforms = [vp.RandomCrop(3, 5), vp.Normalize()]
        pipeline = vp.Pipeline(transforms, seed=137).compile()
        control = vp.Pipeline(transforms, seed=137).compile()
        source = np.repeat(gray()[..., None], 3, axis=2)
        explanation = pipeline.explain()
        np.testing.assert_array_equal(pipeline(source), control(source))
        with self.assertRaisesRegex(ValueError, "Normalize"):
            pipeline(source[..., 0])
        np.testing.assert_array_equal(pipeline(source), control(source))
        self.assertEqual(pipeline.explain(), explanation)
        for shape in ((2, 3, 0), (2, 3, 2), (2, 3, 4), (2,), (0, 3), (2, 3, 1, 1)):
            with self.assertRaises(ValueError):
                vp.Pipeline([]).compile()(np.zeros(shape, dtype=np.uint8))
        with self.assertRaises(TypeError):
            vp.Pipeline([]).compile()(gray().astype(np.uint16))

    def test_codecs_modes_siblings_and_writes(self):
        source = gray()
        encoded = vp.encode_image(source, format="png")
        for carrier in (vp.Array(), vp.Encoded(), vp.Path()):
            for mode in (None,) if isinstance(carrier, vp.Array) else (None, "rgb", "gray"):
                array = vp.ReturnArray(name="array")
                encoded_output = vp.Encode("png", name="encoded")
                write = vp.Write("png", name="write")
                target = vp.Image(
                    carrier,
                    name="image",
                    output_specs=[array, encoded_output, write],
                    decode_mode=mode,
                )
                pipeline = vp.Pipeline([vp.Invert()], targets=[target]).compile()
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "input.png"
                    path.write_bytes(encoded)
                    output = Path(directory) / "output.png"
                    data = (
                        source
                        if isinstance(carrier, vp.Array)
                        else encoded
                        if isinstance(carrier, vp.Encoded)
                        else path
                    )
                    result = pipeline(image=target.bind(data, write.bind(output)))[target]
                    expected = 255 - source
                    if not isinstance(carrier, vp.Array) and mode != "gray":
                        expected = np.repeat(expected[..., None], 3, axis=2)
                    np.testing.assert_array_equal(result[array], expected)
                    np.testing.assert_array_equal(
                        vp.decode_image(result[encoded_output], mode="unchanged"), expected
                    )
                    np.testing.assert_array_equal(vp.read_image(output, mode="unchanged"), expected)
        for rank in (2, 3):
            target = vp.Image(name="image", output_specs=vp.Encode("jpeg", name="value"))
            data = source if rank == 2 else source[..., None]
            result = (
                vp.Pipeline([], targets=[target]).compile()(image=target.bind(data)).image.value
            )
            self.assertEqual(vp.decode_image(result, mode="unchanged").shape, source.shape)
        color = np.stack([source, 255 - source, source // 2], axis=2)
        for format in ("jpeg", "png"):
            encoded = vp.encode_image(color, format=format)
            target = vp.Image(
                vp.Encoded(),
                name="image",
                output_specs=vp.ReturnArray(name="value"),
                decode_mode="gray",
            )
            result = (
                vp.Pipeline([], targets=[target]).compile()(image=target.bind(encoded)).image.value
            )
            np.testing.assert_array_equal(result, vp.decode_image(encoded, mode="gray"))
        for mode in ("gray", "rgb"):
            with self.assertRaises(ValueError):
                vp.Image(name="image", decode_mode=mode)
        with self.assertRaises(ValueError):
            vp.Image(vp.Encoded(), name="image", decode_mode="unchanged")
        target = vp.Image(
            vp.Encoded(),
            name="image",
            output_specs=vp.ReturnArray(name="value"),
            decode_mode="gray",
        )
        with self.assertRaisesRegex(ValueError, "Normalize"):
            vp.Pipeline([vp.Normalize()], targets=[target]).compile()
        data16 = vp.encode_image(source.astype(np.uint16) * 256, format="png")
        with self.assertRaisesRegex((TypeError, ValueError), "uint8"):
            vp.Pipeline([], targets=[target]).compile()(image=target.bind(data16))

    def test_tensor_layout_normalization_and_sibling_ownership(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        source = gray()
        for normalized in (False, True):
            transforms = [vp.Normalize(mean=0.5, std=0.5)] if normalized else []
            tensor = vp.ReturnTensor(name="tensor")
            array = vp.ReturnArray(name="array")
            for outputs in (tensor, [tensor, array]):
                target = vp.Image(name="image", output_specs=outputs)
                reference = vp.Pipeline(transforms, targets=[target])
                for pipeline in (reference, reference.compile()):
                    result = pipeline(image=target.bind(source))[target]
                    actual = result[tensor]
                    self.assertEqual(tuple(actual.shape), (1, *source.shape))
                    self.assertTrue(actual.is_contiguous())
                    self.assertEqual(actual.dtype, torch.float32 if normalized else torch.uint8)
                    expected = vp.Pipeline(transforms)(source)
                    np.testing.assert_array_equal(actual.numpy()[0], expected)
                    if isinstance(outputs, list):
                        actual[0, 0, 0] = 12
                        np.testing.assert_array_equal(result[array], expected)

    def test_pickle_and_static_channel_explanation(self):
        pipeline = vp.Pipeline(
            [
                vp.PadIfNeeded(min_height=9, min_width=11, fill=7),
                vp.Normalize(mean=0.5, std=(0.5,)),
            ],
            seed=137,
        ).compile()
        self.assertEqual(pipeline.explain()["schema_version"], 5)
        target = pipeline.explain()["targets"][0]
        self.assertEqual(target["channels"], [1, 3])
        self.assertEqual([a["channels"] for a in target["channel_alternatives"]], [1, 3])
        gray_alternative = target["channel_alternatives"][0]
        fill_policies = {p["name"]: p["value"] for p in gray_alternative["steps"][0]["policies"]}
        self.assertEqual(fill_policies["fill"], "[7]")
        restored = pickle.loads(pickle.dumps(pipeline))
        for source in (gray(), gray()[..., None], np.repeat(gray()[..., None], 3, axis=2)):
            np.testing.assert_array_equal(restored(source, key=19), pipeline(source, key=19))
        target = vp.Image(
            vp.Encoded(),
            name="image",
            output_specs=vp.ReturnArray(name="value"),
            decode_mode="gray",
        )
        restored = pickle.loads(pickle.dumps(vp.Pipeline([], targets=[target]).compile()))
        self.assertEqual(restored.explain()["targets"][0]["carrier"]["mode"], "gray")
        self.assertEqual(restored.explain()["targets"][0]["channels"], [1])
