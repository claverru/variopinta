from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import variopinta as R

from tests._helpers import image


class TorchTests(unittest.TestCase):
    def test_return_tensor_preserves_uint8_and_converts_to_contiguous_chw(self) -> None:
        import torch

        source = image(7, 11)[:, ::2]
        tensor = R.ReturnTensor(name="tensor")
        target = R.Image(name="image", outputs=(tensor,))
        reference = R.Pipeline([], seed=137, targets=(target,))
        for pipeline in (reference, reference.compile()):
            output = pipeline(image=target.bind(source), key=3).image.tensor
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.dtype, torch.uint8)
            self.assertEqual(tuple(output.shape), (3, 7, 6))
            self.assertTrue(output.is_contiguous())
            self.assertEqual(output.device.type, "cpu")
            np.testing.assert_array_equal(output.numpy(), np.moveaxis(source, 2, 0))
        self.assertEqual(
            reference.explain()["targets"][0]["outputs"][0]["terminal_layout"],
            "terminal-HWC-to-CHW-copy",
        )
        self.assertEqual(
            reference.compile().explain()["targets"][0]["outputs"][0]["terminal_layout"],
            "direct-CHW",
        )

    def test_normalize_return_tensor_preserves_float32_values(self) -> None:
        source = image(7, 11)
        expected = np.moveaxis(R.Pipeline([R.Normalize()])(source, key=3), 2, 0)
        tensor = R.ReturnTensor(name="tensor")
        target = R.Image(name="image", outputs=(tensor,))
        reference = R.Pipeline([R.Normalize()], seed=137, targets=(target,))
        for pipeline in (reference, reference.compile()):
            output = pipeline(image=target.bind(source), key=3).image.tensor
            self.assertEqual(str(output.dtype), "torch.float32")
            self.assertTrue(output.is_contiguous())
            np.testing.assert_array_equal(output.numpy(), expected)

        explanation = reference.compile().explain()
        self.assertEqual(explanation["fusions"], ["Normalize+terminal-layout:direct-CHW"])
        terminal = explanation["targets"][0]["outputs"][0]
        self.assertEqual((terminal["layout"], terminal["dtype"]), ("CHW", "float32"))
        self.assertEqual(terminal["terminal_layout"], "direct-CHW")
        self.assertNotIn("buffers", explanation)
        self.assertNotIn("copies", explanation)

        reference_terminal = reference.explain()["targets"][0]["outputs"][0]
        self.assertEqual(reference_terminal["terminal_layout"], "terminal-HWC-to-CHW-copy")
        self.assertTrue(reference_terminal["buffers"])
        self.assertTrue(reference_terminal["copies"])

    def test_compiled_pipeline_ending_in_normalize_reports_direct_chw(self) -> None:
        tensor = R.ReturnTensor(name="tensor")
        target = R.Image(name="image", outputs=(tensor,))
        explanation = R.Pipeline([R.Invert(), R.Normalize()], targets=(target,)).compile().explain()

        self.assertEqual(explanation["fusions"], ["Normalize+terminal-layout:direct-CHW"])
        terminal = explanation["targets"][0]["outputs"][0]
        self.assertEqual(terminal["terminal_layout"], "direct-CHW")
        output_buffer = next(
            buffer
            for buffer in explanation["targets"][0]["buffers"]
            if buffer["name"] == "output-f32"
        )
        self.assertEqual(output_buffer["layout"], "CHW")

    def test_tensor_terminal_layout_copy_is_reported(self) -> None:
        tensor = R.ReturnTensor(name="tensor")
        target = R.Image(name="image", outputs=(tensor,))
        explanation = R.Pipeline([R.Invert()], targets=(target,)).compile().explain()
        layout_copy = next(
            copy
            for copy in explanation["targets"][0]["copies"]
            if copy["stage"] == "terminal-layout"
        )

        self.assertEqual(layout_copy["count"], "1")
        self.assertEqual(layout_copy["condition"], "always")

    def test_tensor_and_array_fan_out_do_not_alias(self) -> None:
        source = image(7, 11)
        array = R.ReturnArray(name="array")
        first = R.ReturnTensor(name="first")
        second = R.ReturnTensor(name="second")
        target = R.Image(name="image", outputs=(array, first, second))
        result = R.Pipeline([], targets=(target,))(image=target.bind(source), key=3).image
        np.testing.assert_array_equal(result.array, source)
        np.testing.assert_array_equal(result.first.numpy(), np.moveaxis(source, 2, 0))
        result.array.fill(0)
        self.assertTrue(np.any(result.first.numpy()))
        result.first.fill_(0)
        self.assertTrue(np.any(result.second.numpy()))

        outputs = R.Pipeline([], targets=(target,)).explain()["targets"][0]["outputs"]
        self.assertEqual(outputs[0]["terminal_layout"], "semantic-HWC-raster")
        self.assertEqual(outputs[1]["terminal_layout"], "from-shared-HWC-raster")
        self.assertEqual(outputs[1]["copies"][0]["count"], "1")
        self.assertEqual(outputs[2]["copies"][0]["count"], "0")

    def test_mask_return_tensor_is_contiguous_hw_uint8(self) -> None:
        source = np.arange(35, dtype=np.uint8).reshape(5, 7)
        tensor = R.ReturnTensor(name="tensor")
        target = R.Mask(name="labels", outputs=(tensor,))
        output = R.Pipeline([R.HorizontalFlip(1.0)], targets=(target,))(
            labels=target.bind(source), key=3
        ).labels.tensor
        self.assertEqual(tuple(output.shape), (5, 7))
        self.assertEqual(str(output.dtype), "torch.uint8")
        self.assertTrue(output.is_contiguous())
        np.testing.assert_array_equal(output.numpy(), source[:, ::-1])

    def test_return_tensor_has_a_clear_optional_dependency_error(self) -> None:
        tensor = R.ReturnTensor(name="tensor")
        target = R.Image(name="image", outputs=(tensor,))
        pipeline = R.Pipeline([], targets=(target,))
        real_import = __import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch":
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_torch):
            with self.assertRaisesRegex(ImportError, "ReturnTensor requires PyTorch"):
                pipeline(image=target.bind(image(3, 5)), key=1)

    def test_missing_torch_fails_before_writing(self) -> None:
        tensor = R.ReturnTensor(name="tensor")
        written = R.Write("png", name="written")
        image_target = R.Image(name="image", outputs=(tensor,))
        mask_target = R.Mask(name="labels", outputs=(written,))
        pipeline = R.Pipeline([], targets=(image_target, mask_target))
        labels = np.arange(15, dtype=np.uint8).reshape(3, 5)
        real_import = __import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch":
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return real_import(name, *args, **kwargs)

        with TemporaryDirectory() as directory:
            destination = Path(directory) / "labels.png"
            with patch("builtins.__import__", side_effect=import_without_torch):
                with self.assertRaisesRegex(ImportError, "ReturnTensor requires PyTorch"):
                    pipeline(
                        image=image_target.bind(image(3, 5)),
                        labels=mask_target.bind(labels, written.bind(destination)),
                    )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
