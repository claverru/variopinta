from __future__ import annotations

import inspect
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import variopinta as R


def image(height: int = 13, width: int = 17) -> np.ndarray:
    values = np.arange(height * width * 3, dtype=np.uint64)
    return ((values * 73 + values // 7 * 19) & 255).astype(np.uint8).reshape(height, width, 3)


class PipelineIoTests(unittest.TestCase):
    def test_configuration_is_normalized_immutable_and_preserved(self) -> None:
        configurations = (
            R.ArrayInput(),
            R.EncodedInput(max_pixels=None, max_encoded_bytes=123),
            R.PathInput(max_pixels=456, max_encoded_bytes=None),
            R.ReturnOutput(),
            R.EncodedOutput(format=".JPG"),
            R.PathOutput(format="PNG"),
        )
        for configuration in configurations:
            with self.subTest(configuration=configuration):
                self.assertFalse(hasattr(configuration, "__dict__"))
                with self.assertRaises((AttributeError, TypeError)):
                    configuration.extra = 1  # type: ignore[attr-defined]

        encoded_output = configurations[4]
        path_output = configurations[5]
        self.assertEqual((encoded_output.format, encoded_output.quality), ("jpeg", 95))
        self.assertEqual((path_output.format, path_output.compression), ("png", 6))

        pipeline = R.Compose(
            [R.Invert()],
            seed=137,
            input=configurations[1],
            output=encoded_output,
        )
        compiled = pipeline.compile()
        self.assertIs(pipeline.input, configurations[1])
        self.assertIs(pipeline.output, encoded_output)
        self.assertIs(compiled.input, configurations[1])
        self.assertIs(compiled.output, encoded_output)
        self.assertIn("destination", inspect.signature(pipeline.__call__).parameters)

    def test_all_nine_routes_match_standalone_oracles(self) -> None:
        source = image()
        encoded = R.encode_image(source, format="png", compression=3)
        transforms = [R.RandomCrop(11, 13), R.Resize(7, 9), R.Invert(0.5)]
        key = 19
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.data"
            source_path.write_bytes(encoded)
            sources = (
                (R.ArrayInput(), source),
                (R.EncodedInput(), encoded),
                (R.PathInput(), source_path),
            )
            expected_pixels = R.Compose(transforms, seed=137)(source, key=key)
            expected_encoded = R.encode_image(expected_pixels, format="png", compression=3)

            for compile_pipeline in (False, True):
                for input_configuration, value in sources:
                    for output_configuration in (
                        R.ReturnOutput(),
                        R.EncodedOutput(format="png", compression=3),
                        R.PathOutput(format="png", compression=3),
                    ):
                        with self.subTest(
                            compiled=compile_pipeline,
                            input=input_configuration,
                            output=output_configuration,
                        ):
                            pipeline = R.Compose(
                                transforms,
                                seed=137,
                                input=input_configuration,
                                output=output_configuration,
                            )
                            if compile_pipeline:
                                pipeline = pipeline.compile()
                            if isinstance(output_configuration, R.PathOutput):
                                destination = root / (
                                    f"{compile_pipeline}-{type(input_configuration).__name__}.png"
                                )
                                result = pipeline(value, destination=destination, key=key)
                                self.assertIsNone(result)
                                self.assertEqual(destination.read_bytes(), expected_encoded)
                            elif isinstance(output_configuration, R.EncodedOutput):
                                self.assertEqual(pipeline(value, key=key), expected_encoded)
                            else:
                                np.testing.assert_array_equal(
                                    pipeline(value, key=key), expected_pixels
                                )

    def test_encoded_carriers_snapshot_and_limits(self) -> None:
        source = image(7, 11)
        encoded = R.encode_image(source, format="png")
        interleaved = bytearray(len(encoded) * 2)
        interleaved[::2] = encoded
        carriers = (encoded, bytearray(encoded), memoryview(encoded), memoryview(interleaved)[::2])
        pipeline = R.Compose([], input=R.EncodedInput(max_encoded_bytes=len(encoded)))
        for carrier in carriers:
            with self.subTest(carrier=type(carrier)):
                np.testing.assert_array_equal(pipeline(carrier, key=3), source)

        with self.assertRaisesRegex(ValueError, "exceeding"):
            R.decode_image(encoded, max_encoded_bytes=len(encoded) - 1)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            R.Compose([], input=R.EncodedInput(max_encoded_bytes=len(encoded) - 1))(encoded)

        released = memoryview(encoded)
        released.release()
        with self.assertRaises(ValueError):
            R.decode_image(released)
        with self.assertRaises(ValueError):
            pipeline(released)

        for invalid in (0, -1, True, 1.5, "10"):
            with self.subTest(limit=invalid):
                with self.assertRaises(ValueError):
                    R.EncodedInput(max_encoded_bytes=invalid)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    R.decode_image(encoded, max_encoded_bytes=invalid)  # type: ignore[arg-type]

    def test_jpeg_sinks_match_standalone_oracles(self) -> None:
        source = image()
        input_bytes = R.encode_image(source, format="png")
        expected_pixels = 255 - source
        expected_bytes = R.encode_image(expected_pixels, format="jpeg", quality=73)

        for compile_pipeline in (False, True):
            pipeline = R.Compose(
                [R.Invert()],
                input=R.EncodedInput(),
                output=R.EncodedOutput(format="jpeg", quality=73),
            )
            if compile_pipeline:
                pipeline = pipeline.compile()
            with self.subTest(compiled=compile_pipeline, sink="encoded"):
                self.assertEqual(pipeline(input_bytes, key=3), expected_bytes)

            with TemporaryDirectory() as directory:
                root = Path(directory)
                source_path = root / "source.png"
                destination = root / "result.jpeg"
                source_path.write_bytes(input_bytes)
                path_pipeline = R.Compose(
                    [R.Invert()],
                    input=R.PathInput(),
                    output=R.PathOutput(format="jpeg", quality=73),
                )
                if compile_pipeline:
                    path_pipeline = path_pipeline.compile()
                with self.subTest(compiled=compile_pipeline, sink="path"):
                    self.assertIsNone(path_pipeline(source_path, destination=destination, key=3))
                    self.assertEqual(destination.read_bytes(), expected_bytes)

    def test_path_limits_destination_validation_and_same_path(self) -> None:
        source = image(7, 11)
        encoded = R.encode_image(source, format="png")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.bin"
            source_path.write_bytes(encoded)
            with self.assertRaisesRegex(ValueError, "exceeding"):
                R.read_image(source_path, max_encoded_bytes=len(encoded) - 1)
            with self.assertRaisesRegex(ValueError, "exceeding"):
                R.Compose([], input=R.PathInput(max_encoded_bytes=len(encoded) - 1))(source_path)

            pipeline = R.Compose(
                [R.Invert()],
                input=R.PathInput(),
                output=R.PathOutput(format="png"),
            )
            with self.assertRaises(TypeError):
                pipeline(source_path)
            with self.assertRaises(ValueError):
                pipeline(root / "missing.png", destination=root / "output.jpg")
            self.assertFalse((root / "output.jpg").exists())

            expected = R.encode_image(255 - source, format="png")
            pipeline(source_path, destination=source_path, key=3)
            self.assertEqual(source_path.read_bytes(), expected)

            destination = root / "overwrite.png"
            destination.write_bytes(b"old contents")
            pipeline(source_path, destination=destination, key=3)
            self.assertEqual(destination.read_bytes(), R.encode_image(source, format="png"))

            missing_destination = root / "missing-parent" / "output.png"
            with self.assertRaises(OSError):
                pipeline(source_path, destination=missing_destination, key=3)
            self.assertFalse(missing_destination.exists())

            for invalid in (b"source.png", "https://example.com/source.png", "*.png"):
                with self.subTest(path=invalid), self.assertRaises(TypeError):
                    R.Compose([], input=R.PathInput())(invalid)  # type: ignore[arg-type]

    def test_closed_source_sink_and_output_contracts(self) -> None:
        source = image(3, 5)
        encoded = R.encode_image(source, format="png")
        with self.assertRaises(TypeError):
            R.Compose([], input=R.EncodedInput())(source)
        with self.assertRaises(TypeError):
            R.Compose([], input=R.PathInput())(encoded)
        with self.assertRaises(TypeError):
            R.Compose([])(encoded)
        with self.assertRaises(TypeError):
            R.Compose([], input=R.EncodedInput())(object())
        with self.assertRaises(TypeError):
            R.Compose([], input=R.EncodedInput())(np.frombuffer(encoded, dtype=np.uint8))
        with self.assertRaises(TypeError):
            R.Compose([], output=R.EncodedOutput(format="png"))(source, destination="ignored.png")
        with self.assertRaises(ValueError):
            R.Compose([R.Normalize(p=0.5)], output=R.EncodedOutput(format="png"))
        with self.assertRaises(ValueError):
            R.Compose([R.ToTorch()], output=R.PathOutput(format="png"))
        allowed = R.Compose([R.Normalize(p=0.0)], output=R.EncodedOutput(format="png"))
        self.assertIsInstance(allowed(source, key=3), bytes)

        sixteen_bit = source.astype(np.uint16) * 257
        encoded_sixteen_bit = R.encode_image(sixteen_bit, format="png")
        with self.assertRaises(TypeError):
            R.Compose([], input=R.EncodedInput())(encoded_sixteen_bit)

    def test_pre_sampling_failures_do_not_advance_implicit_sequence(self) -> None:
        source = image(13, 17)
        encoded = R.encode_image(source, format="png")

        def make_pipeline() -> R.CompiledCompose:
            return R.Compose(
                [R.HorizontalFlip(0.5), R.Invert(0.5)],
                seed=137,
                input=R.EncodedInput(),
            ).compile()

        pipeline = make_pipeline()
        with self.assertRaises(ValueError):
            pipeline(b"not an image")
        np.testing.assert_array_equal(pipeline(encoded), make_pipeline()(encoded))

        path_output = R.Compose(
            [R.HorizontalFlip(0.5)],
            seed=137,
            input=R.PathInput(),
            output=R.PathOutput(format="png"),
        ).compile()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                path_output(root / "missing", destination=root / "conflict.jpg")
            source_path = root / "source.png"
            source_path.write_bytes(encoded)
            destination = root / "actual.png"
            path_output(source_path, destination=destination)
            expected = R.Compose(
                [R.HorizontalFlip(0.5)], seed=137, output=R.EncodedOutput(format="png")
            )(source)
            self.assertEqual(destination.read_bytes(), expected)

    def test_explain_reports_native_source_and_sink(self) -> None:
        pipeline = R.Compose(
            [R.Invert()],
            input=R.EncodedInput(max_pixels=123, max_encoded_bytes=456),
            output=R.EncodedOutput(format="jpeg", quality=90),
        )
        for implementation in (pipeline, pipeline.compile()):
            with self.subTest(implementation=type(implementation).__name__):
                explanation = implementation.explain()
                self.assertEqual(
                    explanation["source"],
                    {
                        "type": "encoded",
                        "mode": "rgb",
                        "formats": ["jpeg", "png"],
                        "max_pixels": 123,
                        "max_encoded_bytes": 456,
                    },
                )
                self.assertEqual(
                    explanation["sink"],
                    {"type": "encoded", "format": "jpeg", "quality": 90},
                )
                self.assertEqual(explanation["python_boundary"]["crossings_per_call"], 1)
                self.assertEqual(explanation["python_boundary"]["augmentation"], "released")
                names = {buffer["name"] for buffer in explanation["buffers"]}
                self.assertIn("encoded-input", names)
                self.assertIn("encoded-output", names)

    def test_owned_routes_are_thread_shareable_with_explicit_keys(self) -> None:
        source = image(31, 47)
        encoded = R.encode_image(source, format="png")
        pipeline = R.Compose(
            [R.RandomCrop(19, 23), R.Resize(11, 13), R.Invert(0.5)],
            seed=137,
            input=R.EncodedInput(),
            output=R.EncodedOutput(format="png"),
        ).compile()
        keys = (3, 7, 19, 29)
        expected = [pipeline(encoded, key=key) for key in keys]
        with ThreadPoolExecutor(max_workers=len(keys)) as executor:
            actual = list(executor.map(lambda key: pipeline(encoded, key=key), keys))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
