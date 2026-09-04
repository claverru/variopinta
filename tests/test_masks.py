from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import variopinta as R
from PIL import Image as PillowImage

from tests._helpers import image


def semantic_mask(height: int, width: int) -> np.ndarray:
    return (
        ((np.arange(height * width, dtype=np.uint16) * 37) % 251)
        .astype(np.uint8)
        .reshape(height, width)
    )


class MaskTests(unittest.TestCase):
    def test_generic_image_io_preserves_label_samples(self) -> None:
        labels = np.array([[0, 1, 2, 3, 1], [3, 2, 1, 0, 2]], dtype=np.uint8)
        palette = PillowImage.fromarray(labels, mode="P")
        palette.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255, 19, 23, 29] + [0] * 756)
        palette.info["transparency"] = bytes([0, 63, 127, 255])
        encoded = BytesIO()
        palette.save(encoded, format="PNG", bits=2)
        np.testing.assert_array_equal(R.decode_image(encoded.getvalue(), mode="unchanged"), labels)

        for compression in (0, 3, 9):
            encoded_labels = R.encode_image(labels, format="png", compression=compression)
            np.testing.assert_array_equal(R.decode_image(encoded_labels, mode="unchanged"), labels)
            decoded = PillowImage.open(BytesIO(encoded_labels))
            self.assertEqual((decoded.mode, decoded.size), ("L", (5, 2)))

    def test_mask_inputs_reject_png_transparency_metadata(self) -> None:
        labels = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
        encoded_inputs = []
        for mode in ("L", "P"):
            source = PillowImage.fromarray(labels, mode=mode)
            if mode == "P":
                source.putpalette([0, 0, 0] * 256)
            encoded = BytesIO()
            source.save(encoded, format="PNG", transparency=1)
            encoded_inputs.append(encoded.getvalue())

        result = R.ReturnArray(name="array")
        port = R.Mask(R.Encoded(), outputs=(result,), name="labels")
        pipeline = R.Pipeline([], targets=(port,))
        for encoded in encoded_inputs:
            np.testing.assert_array_equal(R.decode_image(encoded, mode="unchanged"), labels)
            with self.assertRaisesRegex(ValueError, "without alpha"):
                pipeline(labels=port.bind(encoded))

    def test_ports_and_bindings_use_identity_and_hide_payloads(self) -> None:
        source = image(3, 5)
        first = R.Image(outputs=(R.ReturnArray(name="array"),), name="first")
        second = R.Image(outputs=(R.ReturnArray(name="array"),), name="second")
        first_binding = first.bind(source)
        second_binding = second.bind(source)
        self.assertIsNot(first, second)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first_binding, second_binding)
        self.assertFalse(hasattr(first_binding, "source"))
        self.assertNotIn(str(source), repr(first_binding))

        with self.assertRaises(TypeError):
            R.BoundTarget(first, source, ())  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            R.Pipeline([], targets=())
        with self.assertRaises(ValueError):
            R.Pipeline([], targets=(first, first))
        with self.assertRaises(ValueError):
            R.Mask(R.Array(), fill=256)
        with self.assertRaises(ValueError):
            R.Mask(outputs=(R.Encode("jpeg", name="encoded"),))
        with self.assertRaises(TypeError):
            R.Mask(outputs=(R.Write(quality=90, name="written"),))

    def test_binding_arity_identity_order_and_values_fail_before_execution(self) -> None:
        source = image(3, 5)
        labels = semantic_mask(3, 5)
        image_port = R.Image(name="image", outputs=(R.ReturnArray(name="array"),))
        mask_port = R.Mask(name="labels", outputs=(R.ReturnArray(name="array"),))
        foreign = R.Image(name="foreign", outputs=(R.ReturnArray(name="array"),))
        pipeline = R.Pipeline([], targets=(image_port, mask_port))

        for values in (
            (),
            (image_port.bind(source),),
            (image_port.bind(source), mask_port.bind(labels), foreign.bind(source)),
        ):
            with self.assertRaises(TypeError):
                pipeline(*values)
        with self.assertRaises(TypeError):
            pipeline(source, labels)
        with self.assertRaises(ValueError):
            pipeline(image=mask_port.bind(labels), labels=image_port.bind(source))
        with self.assertRaises(ValueError):
            pipeline(image=foreign.bind(source), labels=mask_port.bind(labels))

    def test_repeated_images_and_masks_share_one_sampled_plan(self) -> None:
        source = image(19, 23)
        labels = semantic_mask(19, 23)
        first_image = R.Image(name="first_image", outputs=(R.ReturnArray(name="array"),))
        second_image = R.Image(name="second_image", outputs=(R.ReturnArray(name="array"),))
        first_mask = R.Mask(name="first_mask", outputs=(R.ReturnArray(name="array"),), fill=251)
        second_mask = R.Mask(name="second_mask", outputs=(R.ReturnArray(name="array"),), fill=252)
        pipeline = R.Pipeline(
            [
                R.RandomCrop(17, 19, p=0.75),
                R.Resize(15, 17),
                R.HorizontalFlip(0.5),
                R.ColorJitter(0.3, 0.2, 0.4, 0.1, p=0.75),
                R.Affine(
                    degrees=(-17.0, 23.0),
                    translate=(0.2, 0.1),
                    scale=(0.8, 1.2),
                    shear=(-7.0, 9.0),
                    interpolation=R.Interpolation.NEAREST,
                    fill=7,
                    p=1.0,
                ),
                R.GaussianNoise(std=(0.0, 5.0), p=0.5),
            ],
            seed=137,
            targets=(first_image, second_image, first_mask, second_mask),
        )
        compiled = pipeline.compile()
        bindings = {
            "first_image": first_image.bind(source),
            "second_image": second_image.bind(source),
            "first_mask": first_mask.bind(labels),
            "second_mask": second_mask.bind(labels),
        }
        for key in range(8):
            expected = pipeline(**bindings, key=key)
            actual = compiled(**bindings, key=key)
            expected_values = (
                expected.first_image.array,
                expected.second_image.array,
                expected.first_mask.array,
                expected.second_mask.array,
            )
            actual_values = (
                actual.first_image.array,
                actual.second_image.array,
                actual.first_mask.array,
                actual.second_mask.array,
            )
            for expected_value, actual_value in zip(expected_values, actual_values, strict=True):
                np.testing.assert_array_equal(actual_value, expected_value)
            np.testing.assert_array_equal(actual_values[0], actual_values[1])
            first_border = actual_values[2] == 251
            second_border = actual_values[3] == 252
            np.testing.assert_array_equal(first_border, second_border)
            self.assertTrue(first_border.any())

    def test_every_geometric_transform_uses_nearest_mask_rasterization(self) -> None:
        labels = semantic_mask(17, 19)
        source = np.repeat(labels[..., None], 3, axis=2)
        transforms = (
            R.RandomCrop(13, 15),
            R.CenterCrop(13, 15),
            R.RandomResizedCrop(11, 13, interpolation=R.Interpolation.NEAREST),
            R.Resize(11, 13, interpolation=R.Interpolation.NEAREST),
            R.HorizontalFlip(1.0),
            R.VerticalFlip(1.0),
            R.PadIfNeeded(min_height=21, min_width=23, fill=251),
            R.Affine(
                degrees=(-15.0, 20.0),
                translate=(0.2, 0.1),
                interpolation=R.Interpolation.NEAREST,
                fill=251,
            ),
            R.RandomRotation(
                degrees=(-15.0, 20.0),
                interpolation=R.Interpolation.NEAREST,
                fill=251,
            ),
            R.Perspective(
                scale=0.2,
                interpolation=R.Interpolation.NEAREST,
                fill=251,
            ),
            R.GridDistortion(
                num_steps=4,
                distort_limit=0.4,
                interpolation=R.Interpolation.NEAREST,
                fill=251,
            ),
        )
        for transform in transforms:
            image_port = R.Image(name="image", outputs=(R.ReturnArray(name="array"),))
            mask_port = R.Mask(name="mask", outputs=(R.ReturnArray(name="array"),), fill=251)
            reference = R.Pipeline([transform], seed=137, targets=(image_port, mask_port))
            expected = reference(image=image_port.bind(source), mask=mask_port.bind(labels), key=29)
            actual = reference.compile()(
                image=image_port.bind(source), mask=mask_port.bind(labels), key=29
            )
            expected_image, expected_mask = expected.image.array, expected.mask.array
            actual_image, actual_mask = actual.image.array, actual.mask.array
            with self.subTest(transform=transform):
                np.testing.assert_array_equal(actual_image, expected_image)
                np.testing.assert_array_equal(actual_mask, expected_mask)
                np.testing.assert_array_equal(actual_image[..., 0], actual_mask)
                self.assertTrue(set(np.unique(actual_mask)).issubset(set(labels.ravel()) | {251}))

    def test_image_only_and_terminal_transforms_do_not_change_masks(self) -> None:
        source = image(7, 11)
        labels = semantic_mask(7, 11)
        image_port = R.Image(name="image", outputs=(R.ReturnArray(name="array"),))
        mask_port = R.Mask(name="mask", outputs=(R.ReturnArray(name="array"),))
        pipeline = R.Pipeline(
            [
                R.CoarseDropout(p=1.0),
                R.ColorJitter(p=1.0),
                R.GaussianBlur(p=1.0),
                R.GaussianNoise(p=1.0),
                R.Sharpen(p=1.0),
                R.Grayscale(p=1.0),
                R.Invert(p=1.0),
                R.Solarize(p=1.0),
                R.Posterize(p=1.0),
                R.Normalize(),
            ],
            seed=137,
            targets=(image_port, mask_port),
        )
        for route in (pipeline, pipeline.compile()):
            result = route(image=image_port.bind(source), mask=mask_port.bind(labels), key=3)
            image_output, mask_output = result.image.array, result.mask.array
            self.assertEqual(image_output.dtype, np.float32)
            np.testing.assert_array_equal(mask_output, labels)
            self.assertTrue(mask_output.flags.c_contiguous)
            self.assertTrue(mask_output.flags.owndata)
            self.assertFalse(np.shares_memory(mask_output, labels))

    def test_mixed_carriers_and_outputs_match_array_oracles(self) -> None:
        source = image(13, 17)
        labels = semantic_mask(13, 17)
        encoded_source = R.encode_image(source, format="png")
        encoded_labels = R.encode_image(labels, format="png")
        transforms = [R.RandomCrop(11, 15), R.Resize(7, 9), R.HorizontalFlip(0.5)]
        key = 19

        oracle_image = R.Pipeline(transforms, seed=137)(source, key=key)
        oracle_port = R.Mask(name="mask", outputs=(R.ReturnArray(name="array"),))
        oracle_mask = R.Pipeline(transforms, seed=137, targets=(oracle_port,))(
            mask=oracle_port.bind(labels), key=key
        ).mask.array

        with TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.data"
            labels_path.write_bytes(encoded_labels)
            image_port = R.Image(name="image", outputs=(R.ReturnArray(name="array"),))
            view_port = R.Image(
                R.Encoded(), name="view", outputs=(R.Encode("png", name="encoded"),)
            )
            written = R.Write(name="written")
            mask_port = R.Mask(R.Path(), name="mask", outputs=(written,))
            destination = root / "labels-output.png"
            pipeline = R.Pipeline(
                transforms,
                seed=137,
                targets=(image_port, view_port, mask_port),
            ).compile()
            result = pipeline(
                image=image_port.bind(source),
                view=view_port.bind(encoded_source),
                mask=mask_port.bind(labels_path, written.bind(destination)),
                key=key,
            )
            image_output = result.image.array
            view_bytes = result.view.encoded
            labels_output = result.mask.written
            np.testing.assert_array_equal(image_output, oracle_image)
            np.testing.assert_array_equal(R.decode_image(view_bytes), oracle_image)
            self.assertEqual(labels_output, destination)
            np.testing.assert_array_equal(R.read_image(destination, mode="unchanged"), oracle_mask)

    def test_invalid_canvas_and_encoded_masks_do_not_consume_keys(self) -> None:
        source = image(11, 13)
        labels = semantic_mask(11, 13)

        def make_pipeline() -> tuple[R.Pipeline, R.Image, R.Mask]:
            image_port = R.Image(name="view", outputs=(R.ReturnArray(name="array"),))
            mask_port = R.Mask(name="labels", outputs=(R.ReturnArray(name="array"),))
            pipeline = R.Pipeline(
                [R.HorizontalFlip(0.5)],
                seed=137,
                targets=(image_port, mask_port),
            )
            return pipeline, image_port, mask_port

        pipeline, image_port, mask_port = make_pipeline()
        with self.assertRaisesRegex(ValueError, r'target 1 \("labels"\)'):
            pipeline(view=image_port.bind(source), labels=mask_port.bind(labels[:, :-1]))
        actual = pipeline(view=image_port.bind(source), labels=mask_port.bind(labels))
        fresh, fresh_image, fresh_mask = make_pipeline()
        expected = fresh(view=fresh_image.bind(source), labels=fresh_mask.bind(labels))
        np.testing.assert_array_equal(actual.view.array, expected.view.array)
        np.testing.assert_array_equal(actual.labels.array, expected.labels.array)

        encoded_port = R.Mask(R.Encoded(), name="labels", outputs=(R.ReturnArray(name="array"),))
        encoded_pipeline = R.Pipeline([], targets=(encoded_port,))
        with self.assertRaisesRegex(ValueError, r'target 0 \("labels"\)'):
            encoded_pipeline(labels=encoded_port.bind(R.encode_image(source, format="jpeg")))
        with self.assertRaisesRegex(ValueError, r'target 0 \("labels"\)'):
            encoded_pipeline(labels=encoded_port.bind(R.encode_image(source, format="png")))

    def test_explain_lists_each_static_target_without_runtime_values(self) -> None:
        image_port = R.Image(R.Encoded(), outputs=(R.Encode("jpeg", name="encoded"),), name="view")
        mask_port = R.Mask(R.Path(), outputs=(R.Write(name="written"),), fill=255, name="labels")
        explanation = (
            R.Pipeline(
                [R.HorizontalFlip(1.0), R.ColorJitter()],
                targets=(image_port, mask_port),
            )
            .compile()
            .explain()
        )
        self.assertEqual(explanation["schema_version"], 4)
        self.assertEqual([target["role"] for target in explanation["targets"]], ["image", "mask"])
        self.assertEqual(explanation["targets"][1]["fill"], 255)
        self.assertEqual(explanation["targets"][0]["carrier"]["type"], "encoded")
        self.assertEqual(explanation["targets"][1]["outputs"][0]["type"], "write")
        self.assertEqual(explanation["targets"][1]["carrier"]["formats"], ["png"])
        self.assertEqual(explanation["targets"][1]["outputs"][0]["format"], "png")
        self.assertTrue(explanation["targets"][0]["copies"])
        self.assertTrue(explanation["targets"][1]["buffers"])
        self.assertIn("gil", explanation["targets"][0])
        self.assertNotIn("mask_route", explanation)

    def test_mask_only_explain_reports_the_effective_output_contract(self) -> None:
        port = R.Mask(name="mask", outputs=(R.ReturnTensor(name="tensor"),))
        explanation = R.Pipeline([R.Normalize()], targets=(port,)).compile().explain()
        output = explanation["targets"][0]["outputs"][0]
        self.assertEqual(output["dtype"], "uint8")
        self.assertEqual(output["layout"], "HW")
        self.assertEqual(output["container"], "Torch CPU Tensor")


if __name__ == "__main__":
    unittest.main()
