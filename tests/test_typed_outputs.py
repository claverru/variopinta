from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import variopinta as R

from tests._helpers import image


class TypedOutputTests(unittest.TestCase):
    def test_longest_max_size_supports_typed_outputs_and_encoded_path_inputs(self) -> None:
        source = image(3, 6)
        returned = R.ReturnArray(name="array")
        tensor = R.ReturnTensor(name="tensor")
        encoded = R.Encode("png", name="encoded")
        written = R.Write("png", name="written")
        target = R.Image(
            name="image",
            output_specs=(returned, tensor, encoded, written),
        )
        pipeline = R.Pipeline([R.LongestMaxSize(5)], targets=target).compile()
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "output.png"
            result = pipeline(
                image=target.bind(source, written.bind(output_path)),
                key=3,
            ).image
            self.assertEqual(result.array.shape, (3, 5, 3))
            self.assertEqual(tuple(result.tensor.shape), (3, 3, 5))
            np.testing.assert_array_equal(R.decode_image(result.encoded), result.array)
            np.testing.assert_array_equal(R.read_image(result.written), result.array)

            input_path = Path(directory) / "input.png"
            input_path.write_bytes(R.encode_image(source, format="png"))
            encoded_target = R.Image(
                R.Encoded(), name="encoded_image", output_specs=R.ReturnArray(name="array")
            )
            path_target = R.Image(
                R.Path(), name="path_image", output_specs=R.ReturnArray(name="array")
            )
            carrier_pipeline = R.Pipeline(
                [R.LongestMaxSize(5)], targets=(encoded_target, path_target)
            ).compile()
            carrier_result = carrier_pipeline(
                encoded_image=encoded_target.bind(input_path.read_bytes()),
                path_image=path_target.bind(input_path),
                key=3,
            )
            np.testing.assert_array_equal(
                carrier_result.encoded_image.array,
                carrier_result.path_image.array,
            )
            self.assertEqual(carrier_result.encoded_image.array.shape, (3, 5, 3))

    def test_configuration_arguments_are_keyword_only(self) -> None:
        output = R.ReturnArray(name="array")
        self.assertIsInstance(R.Image(R.Array(), name="image", output_specs=output), R.Image)
        self.assertIsInstance(R.Mask(R.Array(), name="mask", output_specs=output), R.Mask)
        for constructor in (
            lambda: R.Encoded(100),
            lambda: R.Path(100),
            lambda: R.Image(R.Array(), output, name="image"),
            lambda: R.Mask(R.Array(), output, name="mask"),
            lambda: R.Pipeline([], 42),
            lambda: R.Resize(3, 5, 1.0),
            lambda: R.LongestMaxSize(5, R.Interpolation.NEAREST),
            lambda: R.HorizontalFlip(0.5),
            lambda: R.Solarize(128, 0.5),
            lambda: R.Posterize(4, 0.5),
            lambda: R.RandomRotation((-10.0, 10.0), R.Interpolation.NEAREST),
        ):
            with self.subTest(constructor=constructor), self.assertRaises(TypeError):
                constructor()

    def test_ports_and_targets_are_identity_bearing_immutable_values(self) -> None:
        with self.assertRaisesRegex(TypeError, "abstract"):
            R.OutputPort()
        first = R.ReturnArray(name="array")
        second = R.ReturnArray(name="array")
        self.assertIsNot(first, second)
        self.assertNotEqual(first, second)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.name = "changed"

        target = R.Image(name="image", output_specs=[first])
        self.assertIsInstance(target.input_spec, R.Array)
        self.assertEqual(target.output_specs, (first,))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            target.name = "changed"

    def test_targets_reject_custom_output_port_subclasses(self) -> None:
        class CustomOutput(R.OutputPort[object]):
            name = None

        for outputs in (CustomOutput(), (CustomOutput(),)):
            with (
                self.subTest(output_specs=outputs),
                self.assertRaisesRegex(TypeError, "built-in output ports"),
            ):
                R.Image(name="image", output_specs=outputs)

    def test_single_output_ports_are_normalized_to_tuples(self) -> None:
        outputs = (
            R.ReturnArray(name="array"),
            R.ReturnTensor(name="tensor"),
            R.Encode("png", name="encoded"),
            R.Write("png", name="written"),
        )
        for output in outputs:
            with self.subTest(output=output):
                target = R.Image(name="image", output_specs=output)
                self.assertEqual(target.output_specs, (output,))

        mask_output = R.ReturnArray(name="array")
        self.assertEqual(R.Mask(name="mask", output_specs=mask_output).output_specs, (mask_output,))

        with self.assertRaisesRegex(TypeError, "an output port or a sequence"):
            R.Image(name="image", output_specs=object())

    def test_single_targets_preserve_explicit_execution_and_results(self) -> None:
        for target_type, source in (
            (R.Image, image(5, 7)[:, ::-1]),
            (R.Mask, np.arange(35, dtype=np.uint8).reshape(5, 7)[:, ::-1]),
        ):
            output = R.ReturnArray(name="value")
            target = target_type(name="view", output_specs=output)
            for declaration in (target, (target,), [target]):
                reference = R.Pipeline([R.HorizontalFlip(p=1.0)], targets=declaration)
                for pipeline in (reference, reference.compile()):
                    with self.subTest(
                        target=target_type, pipeline=type(pipeline), declaration=type(declaration)
                    ):
                        self.assertEqual(pipeline.targets, (target,))
                        result = pipeline(view=target.bind(source), key=3)
                        self.assertIsInstance(result, R.PipelineResult)
                        self.assertIsInstance(result.view, R.TargetResult)
                        self.assertIs(result.view.value, result[target][output])
                        np.testing.assert_array_equal(result.view.value, source[:, ::-1])
                        self.assertEqual(result.view.value.dtype, np.uint8)
                        self.assertTrue(result.view.value.flags.c_contiguous)
                        self.assertFalse(np.shares_memory(result.view.value, source))
                        with self.assertRaises(TypeError):
                            pipeline(source)
                        with self.assertRaisesRegex(TypeError, "missing"):
                            pipeline()
                        foreign = target_type(name="view", output_specs=output)
                        with self.assertRaisesRegex(ValueError, "different port"):
                            pipeline(view=foreign.bind(source))

    def test_single_targets_require_explicit_names(self) -> None:
        for target_type in (R.Image, R.Mask):
            with self.subTest(target=target_type):
                with self.assertRaises(TypeError):
                    target_type()
                with self.assertRaises(ValueError):
                    target_type(name=None)
                target = target_type(name="view")
                self.assertEqual(target.output_specs[0].name, "array")
                self.assertEqual(R.Pipeline([], targets=target).targets, (target,))

    def test_names_and_scopes_are_validated(self) -> None:
        for name in ("", "not-valid", "_private", "class", "key"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                R.ReturnArray(name=name)
        for output_type, arguments in (
            (R.ReturnArray, ()),
            (R.ReturnTensor, ()),
            (R.Encode, ("png",)),
            (R.Write, ("png",)),
        ):
            with self.subTest(output=output_type), self.assertRaises(TypeError):
                output_type(*arguments)
            with self.subTest(output=output_type), self.assertRaises(ValueError):
                output_type(*arguments, name=None)
        with self.assertRaisesRegex(ValueError, "output names"):
            R.Image(
                name="image",
                output_specs=(R.ReturnArray(name="value"), R.ReturnTensor(name="value")),
            )
        with self.assertRaisesRegex(ValueError, "target names"):
            R.Pipeline(
                [],
                targets=(
                    R.Image(name="same", output_specs=(R.ReturnArray(name="value"),)),
                    R.Mask(name="same", output_specs=(R.ReturnArray(name="value"),)),
                ),
            )
        for output_type in (R.ReturnArray, R.ReturnTensor):
            with self.assertRaises(TypeError):
                output_type()
            with self.assertRaises(ValueError):
                output_type(name=None)

    def test_write_bindings_require_exact_port_identity_and_hide_paths(self) -> None:
        first = R.Write("png", name="first")
        second = R.Write("png", name="second")
        foreign = R.Write("png", name="foreign")
        target = R.Image(name="image", output_specs=(first, second))
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
        image_target = R.Image(name="image", output_specs=(image_output,))
        mask_target = R.Mask(name="labels", output_specs=(mask_output,))
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
        with self.assertRaises(TypeError):
            R.PipelineResult()
        with self.assertRaises(TypeError):
            R.TargetResult()
        with self.assertRaises(TypeError):
            R.CompiledPipeline()
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
                image=R.Image(name="other", output_specs=(R.ReturnArray(name="value"),)).bind(
                    source
                ),
                labels=mask_target.bind(labels),
            )

    def test_implicit_numpy_shortcut_remains_direct(self) -> None:
        source = image(3, 5)
        output = R.Pipeline([])(source)
        self.assertIsInstance(output, np.ndarray)
        np.testing.assert_array_equal(output, source)

    def test_default_outputs_are_named_and_identity_bearing(self) -> None:
        image_target = R.Image(name="image")
        other_target = R.Image(name="other")
        mask_target = R.Mask(name="mask")
        self.assertEqual(image_target.output_specs[0].name, "array")
        self.assertEqual(mask_target.output_specs[0].name, "array")
        self.assertIsNot(image_target.output_specs[0], other_target.output_specs[0])


if __name__ == "__main__":
    unittest.main()
