from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import variopinta as R

from tests._helpers import image


class TorchTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
