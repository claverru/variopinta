from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import variopinta as R
from PIL import Image


def rgb_image(height: int = 13, width: int = 17) -> np.ndarray:
    values = np.arange(height * width * 3, dtype=np.uint64)
    return ((values * 73 + values // 7 * 19) & 255).astype(np.uint8).reshape(height, width, 3)


class ImageIoTests(unittest.TestCase):
    def test_png_round_trip_u8_color_models(self) -> None:
        sources = [
            np.arange(35, dtype=np.uint8).reshape(5, 7),
            np.arange(70, dtype=np.uint8).reshape(5, 7, 2),
            rgb_image(5, 7),
            np.arange(140, dtype=np.uint8).reshape(5, 7, 4),
        ]
        for source in sources:
            with self.subTest(shape=source.shape):
                encoded = R.encode_image(source, format="png")
                output = R.decode_image(encoded, mode="unchanged")
                np.testing.assert_array_equal(output, source)
                self.assertTrue(output.flags.c_contiguous)
                self.assertFalse(np.shares_memory(output, source))

    def test_png_round_trip_u16_color_models(self) -> None:
        for channels in (1, 2, 3, 4):
            shape = (5, 7) if channels == 1 else (5, 7, channels)
            source = (np.arange(np.prod(shape), dtype=np.uint16) * 4099).reshape(shape)
            with self.subTest(shape=shape):
                encoded = R.encode_image(source, format="png", compression=3)
                output = R.decode_image(encoded, mode="unchanged")
                np.testing.assert_array_equal(output, source)
                self.assertEqual(output.dtype, np.uint16)

    def test_png_palette_and_transparency_expand(self) -> None:
        source = Image.new("P", (3, 2))
        source.putdata([0, 1, 2, 2, 1, 0])
        source.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (256 * 3 - 9))
        source.info["transparency"] = bytes([255, 128, 0])
        encoded = BytesIO()
        source.save(encoded, format="PNG", bits=2)
        output = R.decode_image(encoded.getvalue(), mode="unchanged")
        expected = np.asarray(source.convert("RGBA"))
        np.testing.assert_array_equal(output, expected)

    def test_decode_modes(self) -> None:
        source = np.array([[[10, 20, 30, 40], [50, 60, 70, 80]]], dtype=np.uint8)
        encoded = R.encode_image(source, format="png")
        rgb = R.decode_image(encoded, mode="rgb")
        rgba = R.decode_image(encoded, mode="rgba")
        gray = R.decode_image(encoded, mode="gray")
        np.testing.assert_array_equal(rgb, source[..., :3])
        np.testing.assert_array_equal(rgba, source)
        self.assertEqual(gray.shape, (1, 2))

    def test_jpeg_written_by_pillow_matches_rgb_decode(self) -> None:
        source = rgb_image()
        encoded = BytesIO()
        Image.fromarray(source).save(encoded, format="JPEG", quality=95)
        output = R.decode_image(encoded.getvalue())
        expected = np.asarray(Image.open(BytesIO(encoded.getvalue())).convert("RGB"))
        np.testing.assert_array_equal(output, expected)

    def test_cmyk_jpeg_converts_to_rgb(self) -> None:
        source = np.array(
            [[[0, 0, 0, 0], [255, 0, 0, 0], [0, 255, 255, 0], [100, 50, 0, 25]]],
            dtype=np.uint8,
        )
        encoded = BytesIO()
        Image.fromarray(source, "CMYK").save(encoded, format="JPEG", quality=100, subsampling=0)
        output = R.decode_image(encoded.getvalue())
        expected = np.asarray(Image.open(BytesIO(encoded.getvalue())).convert("RGB"))
        np.testing.assert_allclose(output, expected, atol=1)
        unchanged = R.decode_image(encoded.getvalue(), mode="unchanged")
        expected_cmyk = np.asarray(Image.open(BytesIO(encoded.getvalue())))
        np.testing.assert_array_equal(unchanged, expected_cmyk)

    def test_jpeg_encode_is_interoperable(self) -> None:
        source = rgb_image()
        encoded = R.encode_image(source, format="jpeg", quality=91)
        pillow = np.asarray(Image.open(BytesIO(encoded)).convert("RGB"))
        native = R.decode_image(encoded)
        np.testing.assert_array_equal(native, pillow)
        self.assertEqual(native.shape, source.shape)

    def test_read_detects_content_and_write_infers_format(self) -> None:
        source = rgb_image(7, 11)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            png_path = root / "image.png"
            jpeg_path = root / "image.jpeg"
            R.write_image(png_path, source)
            R.write_image(jpeg_path, source, quality=90)
            np.testing.assert_array_equal(R.read_image(png_path), source)
            self.assertEqual(R.read_image(jpeg_path).shape, source.shape)
            disguised = root / "disguised.jpg"
            disguised.write_bytes(png_path.read_bytes())
            np.testing.assert_array_equal(R.read_image(disguised), source)

    def test_write_supports_explicit_format_without_extension(self) -> None:
        source = rgb_image(3, 5)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.data"
            R.write_image(path, source, format="png")
            np.testing.assert_array_equal(R.read_image(path), source)

    def test_non_contiguous_encode_is_normalized_once(self) -> None:
        source = rgb_image()[:, ::2]
        output = R.decode_image(R.encode_image(source, format="png"))
        np.testing.assert_array_equal(output, source)
        self.assertTrue(output.flags.c_contiguous)

    def test_decode_accepts_mutable_buffers(self) -> None:
        source = rgb_image(3, 5)
        encoded = R.encode_image(source, format="png")
        np.testing.assert_array_equal(R.decode_image(bytearray(encoded)), source)
        np.testing.assert_array_equal(R.decode_image(memoryview(encoded)), source)

    def test_limits_and_invalid_options(self) -> None:
        source = rgb_image(5, 7)
        encoded = R.encode_image(source, format="png")
        with self.assertRaises(ValueError):
            R.decode_image(encoded, max_pixels=34)
        np.testing.assert_array_equal(R.decode_image(encoded, max_pixels=None), source)
        with self.assertRaises(ValueError):
            R.read_image("missing.png", max_pixels=0)
        with self.assertRaises(TypeError):
            R.encode_image(source, format="png", quality=90)
        with self.assertRaises(TypeError):
            R.encode_image(source, format="jpeg", compression=3)
        with self.assertRaises(ValueError):
            R.encode_image(source, format="jpeg", quality=0)
        with self.assertRaises(ValueError):
            R.encode_image(source, format="png", compression=10)

    def test_invalid_inputs_are_rejected(self) -> None:
        source = rgb_image(3, 5)
        with self.assertRaises(ValueError):
            R.decode_image(b"not an image")
        with self.assertRaises(ValueError):
            R.decode_image(R.encode_image(source, format="png"), mode="cmyk")
        with self.assertRaises(ValueError):
            R.encode_image(source.astype(np.float32), format="png")
        with self.assertRaises(ValueError):
            R.encode_image(np.zeros((3, 5, 5), dtype=np.uint8), format="png")
        with self.assertRaises(ValueError):
            R.encode_image(np.zeros((3, 5, 4), dtype=np.uint8), format="jpeg")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            with self.assertRaises(ValueError):
                R.write_image(path, source, format="jpeg")
            with self.assertRaises(ValueError):
                R.write_image(Path(directory) / "image.data", source)


if __name__ == "__main__":
    unittest.main()
