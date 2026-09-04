from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import variopinta as R

from tests._helpers import image


class TypedOutputTests(unittest.TestCase):
    def test_ports_and_targets_are_identity_bearing_immutable_values(self) -> None:
        with self.assertRaisesRegex(TypeError, "abstract"):
            R.OutputPort()
        first = R.ReturnArray(name="array")
        second = R.ReturnArray(name="array")
        self.assertIsNot(first, second)
        self.assertNotEqual(first, second)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.name = "changed"

        target = R.Image(name="image", outputs=[first])
        self.assertIsInstance(target.carrier, R.Array)
        self.assertEqual(target.outputs, (first,))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            target.name = "changed"

    def test_targets_reject_custom_output_port_subclasses(self) -> None:
        class CustomOutput(R.OutputPort[object]):
            name = None

        with self.assertRaisesRegex(TypeError, "built-in output ports"):
            R.Image(outputs=(CustomOutput(),))

    def test_names_and_scopes_are_validated(self) -> None:
        for name in ("", "not-valid", "_private", "class", "key"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                R.ReturnArray(name=name)
        with self.assertRaisesRegex(ValueError, "output names"):
            R.Image(outputs=(R.ReturnArray(name="value"), R.ReturnTensor(name="value")))
        with self.assertRaisesRegex(ValueError, "target names"):
            R.Pipeline(
                [],
                targets=(
                    R.Image(name="same", outputs=(R.ReturnArray(name="value"),)),
                    R.Mask(name="same", outputs=(R.ReturnArray(name="value"),)),
                ),
            )
        with self.assertRaisesRegex(ValueError, "target must have a name"):
            R.Pipeline([], targets=(R.Image(outputs=(R.ReturnArray(name="value"),)),))
        with self.assertRaisesRegex(ValueError, "must have a name"):
            R.Pipeline([], targets=(R.Image(name="image"),))

    def test_write_bindings_require_exact_port_identity_and_hide_paths(self) -> None:
        first = R.Write("png", name="first")
        second = R.Write("png", name="second")
        foreign = R.Write("png", name="foreign")
        target = R.Image(name="image", outputs=(first, second))
        source = image(3, 5)
        with TemporaryDirectory() as directory:
            first_binding = first.bind(Path(directory) / "first.png")
            second_binding = second.bind(Path(directory) / "second.png")
            with self.assertRaisesRegex(TypeError, "missing Write"):
                target.bind(source, first_binding)
            with self.assertRaisesRegex(ValueError, "more than once"):
                target.bind(source, first_binding, first_binding)
            with self.assertRaisesRegex(ValueError, "different target"):
                target.bind(source, first_binding, foreign.bind(Path(directory) / "x.png"))
            binding = target.bind(source, second_binding, first_binding)
            self.assertNotIn(str(Path(directory)), repr(binding))
            self.assertNotIn(str(Path(directory)), repr(first_binding))
            with self.assertRaises(TypeError):
                R.WriteBinding(first, Path(directory) / "x.png")
            with self.assertRaises(TypeError):
                R.BoundTarget(target, source, ())

            result = R.Pipeline([], targets=(target,))(
                image=binding,
                key=3,
            )
            self.assertNotIn(str(Path(directory)), repr(result))

    def test_explicit_calls_are_nominal_and_results_never_collapse(self) -> None:
        image_output = R.ReturnArray(name="value")
        mask_output = R.ReturnArray(name="value")
        image_target = R.Image(name="image", outputs=(image_output,))
        mask_target = R.Mask(name="labels", outputs=(mask_output,))
        pipeline = R.Pipeline([], targets=(image_target, mask_target))
        source = image(3, 5)
        labels = np.arange(15, dtype=np.uint8).reshape(3, 5)

        result = pipeline(
            labels=mask_target.bind(labels),
            image=image_target.bind(source),
            key=3,
        )
        self.assertIsInstance(result, R.PipelineResult)
        self.assertIsInstance(result.image, R.TargetResult)
        self.assertIs(result.image.value, result[image_target][image_output])
        self.assertIs(result.labels.value, result[mask_target][mask_output])
        self.assertIn("image", dir(result))
        self.assertIn("value", dir(result.image))
        self.assertNotIn("[[", repr(result))
        with self.assertRaises(AttributeError):
            result.image = None
        with self.assertRaises(TypeError):
            result._names["image"] = None
        with self.assertRaises(TypeError):
            result.image._names["value"] = None
        with self.assertRaises(TypeError):
            pipeline(image_target.bind(source), mask_target.bind(labels))
        with self.assertRaisesRegex(TypeError, "missing"):
            pipeline(image=image_target.bind(source))
        with self.assertRaisesRegex(TypeError, "unexpected"):
            pipeline(
                image=image_target.bind(source),
                labels=mask_target.bind(labels),
                extra=mask_target.bind(labels),
            )
        with self.assertRaisesRegex(ValueError, "different port"):
            pipeline(
                image=R.Image(name="other", outputs=(R.ReturnArray(name="value"),)).bind(source),
                labels=mask_target.bind(labels),
            )

    def test_implicit_numpy_shortcut_remains_direct(self) -> None:
        source = image(3, 5)
        output = R.Pipeline([])(source)
        self.assertIsInstance(output, np.ndarray)
        np.testing.assert_array_equal(output, source)


if __name__ == "__main__":
    unittest.main()
