from __future__ import annotations

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
    def test_target_configuration_is_normalized_immutable_and_preserved(self) -> None:
        carrier = R.Encoded(max_pixels=None, max_encoded_bytes=123)
        output = R.Encode(format=".JPG", quality=81, name="encoded")
        target = R.Image(carrier, outputs=(output,), name="view")
        pipeline = R.Pipeline([R.Invert()], seed=137, targets=(target,))
        compiled = pipeline.compile()

        self.assertEqual(output.format, "jpeg")
        self.assertIs(target.carrier, carrier)
        self.assertEqual(target.outputs, (output,))
        self.assertIs(pipeline.targets[0], target)
        self.assertIs(compiled.targets[0], target)
        for value in (carrier, output, target):
            self.assertFalse(hasattr(value, "__dict__"))
            with self.assertRaises((AttributeError, TypeError)):
                value.extra = 1  # type: ignore[attr-defined]

    def test_every_image_carrier_and_output_matches_standalone_oracles(self) -> None:
        source = image()
        encoded = R.encode_image(source, format="png", compression=3)
        transforms = [R.RandomCrop(11, 13), R.Resize(7, 9), R.Invert(0.5)]
        key = 19
        expected = R.Pipeline(transforms, seed=137)(source, key=key)
        expected_encoded = R.encode_image(expected, format="png", compression=3)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.data"
            source_path.write_bytes(encoded)
            sources = (
                (R.Array(), source),
                (R.Encoded(), encoded),
                (R.Path(), source_path),
            )
            for compiled in (False, True):
                for carrier, value in sources:
                    for output in (
                        R.ReturnArray(name="value"),
                        R.Encode("png", compression=3, name="value"),
                        R.Write("png", compression=3, name="value"),
                    ):
                        port = R.Image(carrier, outputs=(output,), name="image")
                        pipeline = R.Pipeline(transforms, seed=137, targets=(port,))
                        if compiled:
                            pipeline = pipeline.compile()
                        destination = root / f"{compiled}-{type(carrier).__name__}.data"
                        binding = (
                            port.bind(value, output.bind(destination))
                            if isinstance(output, R.Write)
                            else port.bind(value)
                        )
                        with self.subTest(
                            compiled=compiled,
                            carrier=type(carrier).__name__,
                            output=type(output).__name__,
                        ):
                            result = pipeline(image=binding, key=key).image.value
                            if isinstance(output, R.ReturnArray):
                                np.testing.assert_array_equal(result, expected)
                            elif isinstance(output, R.Encode):
                                self.assertEqual(result, expected_encoded)
                            else:
                                self.assertEqual(result, destination)
                                self.assertEqual(destination.read_bytes(), expected_encoded)

    def test_image_fan_out_matches_independent_terminal_oracles(self) -> None:
        source = image(13, 17)
        transforms = [R.RandomCrop(11, 13), R.Invert(0.5)]
        expected = R.Pipeline(transforms, seed=137)(source, key=19)
        expected_png = R.encode_image(expected, format="png", compression=3)
        expected_jpeg = R.encode_image(expected, format="jpeg", quality=83)

        returned = R.ReturnArray(name="array")
        png = R.Encode("png", compression=3, name="png")
        jpeg = R.Encode("jpeg", quality=83, name="jpeg")
        first_write = R.Write("png", compression=3, name="first")
        second_write = R.Write("png", compression=3, name="second")
        target = R.Image(
            name="image",
            outputs=(returned, png, first_write, jpeg, second_write),
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for compiled in (False, True):
                pipeline = R.Pipeline(transforms, seed=137, targets=(target,))
                if compiled:
                    pipeline = pipeline.compile()
                first_path = root / f"{compiled}-first.png"
                second_path = root / f"{compiled}-second.png"
                result = pipeline(
                    image=target.bind(
                        source,
                        second_write.bind(second_path),
                        first_write.bind(first_path),
                    ),
                    key=19,
                ).image
                np.testing.assert_array_equal(result.array, expected)
                self.assertEqual(result.png, expected_png)
                self.assertEqual(result.jpeg, expected_jpeg)
                self.assertEqual(result.first, first_path)
                self.assertEqual(result.second, second_path)
                self.assertEqual(first_path.read_bytes(), expected_png)
                self.assertEqual(second_path.read_bytes(), expected_png)

    def test_mixed_routes_acquire_before_writing_and_return_in_target_order(self) -> None:
        source = image(9, 13)
        encoded = R.encode_image(source, format="png")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            source_path.write_bytes(encoded)
            array_port = R.Image(name="array", outputs=(R.ReturnArray(name="value"),))
            encoded_port = R.Image(
                R.Encoded(),
                outputs=(R.Encode("jpeg", quality=83, name="value"),),
                name="encoded",
            )
            written = R.Write(name="value")
            path_port = R.Image(R.Path(), outputs=(written,), name="path")
            pipeline = R.Pipeline(
                [R.HorizontalFlip(1.0)],
                targets=(array_port, encoded_port, path_port),
            ).compile()
            destination = source_path
            result = pipeline(
                array=array_port.bind(source),
                encoded=encoded_port.bind(encoded),
                path=path_port.bind(source_path, written.bind(destination)),
                key=3,
            )
            returned = result.array.value
            jpeg = result.encoded.value
            written_path = result.path.value
            np.testing.assert_array_equal(returned, source[:, ::-1])
            self.assertEqual(jpeg, R.encode_image(source[:, ::-1], format="jpeg", quality=83))
            self.assertEqual(written_path, destination)
            np.testing.assert_array_equal(R.read_image(source_path), source[:, ::-1])

    def test_encoded_sources_snapshot_and_enforce_limits(self) -> None:
        source = image(7, 11)
        encoded = R.encode_image(source, format="png")
        interleaved = bytearray(len(encoded) * 2)
        interleaved[::2] = encoded
        port = R.Image(
            R.Encoded(max_encoded_bytes=len(encoded)),
            outputs=(R.ReturnArray(name="array"),),
            name="image",
        )
        pipeline = R.Pipeline([], targets=(port,))
        for carrier in (
            encoded,
            bytearray(encoded),
            memoryview(encoded),
            memoryview(interleaved)[::2],
        ):
            np.testing.assert_array_equal(
                pipeline(image=port.bind(carrier), key=3).image.array, source
            )

        too_small = R.Image(
            R.Encoded(max_encoded_bytes=len(encoded) - 1),
            outputs=(R.ReturnArray(name="array"),),
            name="image",
        )
        with self.assertRaisesRegex(ValueError, "exceeding"):
            R.Pipeline([], targets=(too_small,))(image=too_small.bind(encoded))

    def test_array_targets_preserve_logical_order_for_non_c_layouts(self) -> None:
        source = image(9, 13)
        for value in (np.asfortranarray(source), source[:, ::-1]):
            port = R.Image(name="image", outputs=(R.ReturnArray(name="array"),))
            output = R.Pipeline([], targets=(port,))(image=port.bind(value)).image.array
            np.testing.assert_array_equal(output, value)
            self.assertTrue(output.flags.c_contiguous)
        labels = np.arange(9 * 13, dtype=np.uint8).reshape(9, 13)
        for value in (np.asfortranarray(labels), labels[:, ::-1]):
            port = R.Mask(name="mask", outputs=(R.ReturnArray(name="array"),))
            output = R.Pipeline([], targets=(port,))(mask=port.bind(value)).mask.array
            np.testing.assert_array_equal(output, value)
            self.assertTrue(output.flags.c_contiguous)

    def test_write_inference_validation_duplicates_and_source_aliasing(self) -> None:
        source = image(7, 11)
        encoded = R.encode_image(source, format="png")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            source_path.write_bytes(encoded)
            written = R.Write(name="written")
            port = R.Image(R.Path(), outputs=(written,), name="image")
            pipeline = R.Pipeline([R.Invert()], targets=(port,))
            self.assertEqual(
                pipeline(
                    image=port.bind(source_path, written.bind(source_path)), key=3
                ).image.written,
                source_path,
            )
            np.testing.assert_array_equal(R.read_image(source_path), 255 - source)

            first_write = R.Write("png", name="written")
            second_write = R.Write("png", name="written")
            first = R.Image(outputs=(first_write,), name="first")
            second = R.Image(outputs=(second_write,), name="second")
            duplicate = root / "duplicate.png"
            with self.assertRaisesRegex(ValueError, "duplicate"):
                R.Pipeline([], targets=(first, second))(
                    first=first.bind(source, first_write.bind(duplicate)),
                    second=second.bind(source, second_write.bind(duplicate)),
                )
            self.assertFalse(duplicate.exists())

            png_write = R.Write("png", name="written")
            with self.assertRaises(ValueError):
                png_write.bind(root / "result.jpg")
            with self.assertRaises(TypeError):
                R.Image(R.Array()).bind(source, destination=root / "unused.png")

    def test_pre_sampling_failures_preserve_the_implicit_key_sequence(self) -> None:
        source = image(13, 17)
        encoded = R.encode_image(source, format="png")

        def make_pipeline() -> tuple[R.CompiledPipeline, R.Image]:
            port = R.Image(R.Encoded(), outputs=(R.ReturnArray(name="array"),), name="image")
            return (
                R.Pipeline(
                    [R.HorizontalFlip(0.5), R.Invert(0.5)],
                    seed=137,
                    targets=(port,),
                ).compile(),
                port,
            )

        pipeline, port = make_pipeline()
        with self.assertRaises(ValueError):
            pipeline(image=port.bind(b"not an image"))
        actual = pipeline(image=port.bind(encoded)).image.array
        fresh, fresh_port = make_pipeline()
        expected = fresh(image=fresh_port.bind(encoded)).image.array
        np.testing.assert_array_equal(actual, expected)

        def make_writer() -> tuple[R.CompiledPipeline, R.Image]:
            write = R.Write("png", name="written")
            write_port = R.Image(outputs=(write,), name="image")
            return (
                R.Pipeline(
                    [R.HorizontalFlip(0.5), R.Invert(0.5)],
                    seed=137,
                    targets=(write_port,),
                ).compile(),
                write_port,
            )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer, write_port = make_writer()
            with self.assertRaisesRegex(ValueError, "parent"):
                write = write_port.outputs[0]
                assert isinstance(write, R.Write)
                writer(image=write_port.bind(source, write.bind(root / "missing" / "failed.png")))
            destination = root / "actual.png"
            write = write_port.outputs[0]
            assert isinstance(write, R.Write)
            writer(image=write_port.bind(source, write.bind(destination)))
            fresh_writer, fresh_write_port = make_writer()
            expected_destination = root / "expected.png"
            fresh_write = fresh_write_port.outputs[0]
            assert isinstance(fresh_write, R.Write)
            fresh_writer(
                image=fresh_write_port.bind(source, fresh_write.bind(expected_destination))
            )
            np.testing.assert_array_equal(
                R.read_image(destination), R.read_image(expected_destination)
            )

        def make_cropper() -> tuple[R.CompiledPipeline, R.Image]:
            crop_port = R.Image(outputs=(R.ReturnArray(name="array"),), name="image")
            return (
                R.Pipeline([R.RandomCrop(7, 11)], seed=137, targets=(crop_port,)).compile(),
                crop_port,
            )

        cropper, crop_port = make_cropper()
        with self.assertRaises(ValueError):
            cropper(image=crop_port.bind(image(5, 9)))
        actual = cropper(image=crop_port.bind(image(13, 17))).image.array
        fresh_cropper, fresh_crop_port = make_cropper()
        expected = fresh_cropper(image=fresh_crop_port.bind(image(13, 17))).image.array
        np.testing.assert_array_equal(actual, expected)

    def test_owned_routes_are_thread_shareable_with_explicit_keys(self) -> None:
        source = image(31, 47)
        encoded = R.encode_image(source, format="png")
        port = R.Image(R.Encoded(), outputs=(R.Encode("png", name="encoded"),), name="image")
        pipeline = R.Pipeline(
            [R.RandomCrop(19, 23), R.Resize(11, 13), R.Invert(0.5)],
            seed=137,
            targets=(port,),
        ).compile()
        keys = (3, 7, 19, 29)
        expected = [pipeline(image=port.bind(encoded), key=key).image.encoded for key in keys]
        with ThreadPoolExecutor(max_workers=len(keys)) as executor:
            actual = list(
                executor.map(
                    lambda key: pipeline(image=port.bind(encoded), key=key).image.encoded,
                    keys,
                )
            )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
