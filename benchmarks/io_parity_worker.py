from __future__ import annotations

import binascii
import struct
import warnings
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
import torch
import variopinta as R
from PIL import Image
from torchvision.io import ImageReadMode
from torchvision.io import decode_image as torchvision_decode

MODES = ("unchanged", "gray", "rgb", "rgba")


@dataclass(frozen=True)
class Fixture:
    name: str
    format: str
    encoded: bytes
    expected: np.ndarray
    color: str
    producer: str


def pattern(dtype: np.dtype[Any], channels: int, height: int = 13, width: int = 17) -> np.ndarray:
    shape = (height, width) if channels == 1 else (height, width, channels)
    modulus = 256 if dtype == np.dtype(np.uint8) else 65_536
    values = np.arange(np.prod(shape), dtype=np.uint64)
    return ((values * 4099 + values // 7 * 7919) % modulus).astype(dtype).reshape(shape)


def pillow_bytes(image: np.ndarray, format: str, **options: Any) -> bytes:
    output = BytesIO()
    Image.fromarray(image).save(output, format=format, **options)
    return output.getvalue()


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def reference_png(image: np.ndarray) -> bytes:
    channels = 1 if image.ndim == 2 else image.shape[2]
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    depth = 8 if image.dtype == np.uint8 else 16
    rows = []
    for row in image:
        data = row.tobytes() if depth == 8 else row.astype(">u2", copy=False).tobytes()
        rows.append(b"\x00" + data)
    header = struct.pack(">IIBBBBB", image.shape[1], image.shape[0], depth, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + png_chunk(b"IEND", b"")
    )


def paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distances = (abs(estimate - left), abs(estimate - above), abs(estimate - upper_left))
    return (left, above, upper_left)[distances.index(min(distances))]


def parse_png(encoded: bytes) -> np.ndarray:
    position = 8
    payloads = []
    header = None
    while position < len(encoded):
        length = struct.unpack(">I", encoded[position : position + 4])[0]
        kind = encoded[position + 4 : position + 8]
        payload = encoded[position + 8 : position + 8 + length]
        expected_crc = struct.unpack(">I", encoded[position + 8 + length : position + 12 + length])[
            0
        ]
        if binascii.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError("invalid PNG checksum")
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            payloads.append(payload)
        elif kind == b"IEND":
            break
        position += 12 + length
    if header is None:
        raise ValueError("PNG has no header")
    width, height, depth, color_type, compression, filtering, interlace = header
    if depth not in (8, 16) or compression or filtering or interlace:
        raise ValueError("PNG oracle only supports static 8-bit and 16-bit images")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    sample_bytes = depth // 8
    bytes_per_pixel = channels * sample_bytes
    stride = width * bytes_per_pixel
    packed = zlib.decompress(b"".join(payloads))
    previous = bytearray(stride)
    rows = []
    position = 0
    for _ in range(height):
        filter_kind = packed[position]
        source = packed[position + 1 : position + 1 + stride]
        row = bytearray(stride)
        for index, value in enumerate(source):
            left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            above = previous[index]
            upper_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            predictor = {
                0: 0,
                1: left,
                2: above,
                3: (left + above) // 2,
                4: paeth(left, above, upper_left),
            }.get(filter_kind)
            if predictor is None:
                raise ValueError("unknown PNG filter")
            row[index] = (value + predictor) & 255
        rows.append(row)
        previous = row
        position += stride + 1
    dtype = np.uint8 if depth == 8 else np.dtype(">u2")
    output = np.frombuffer(b"".join(rows), dtype=dtype)
    if depth == 16:
        output = output.astype(np.uint16)
    shape = (height, width) if channels == 1 else (height, width, channels)
    return output.reshape(shape)


def cv_png_bytes(image: np.ndarray) -> bytes:
    channels = 1 if image.ndim == 2 else image.shape[2]
    if channels == 2:
        raise ValueError("OpenCV cannot encode two-channel PNG input")
    native = image
    if channels == 3:
        native = image[..., [2, 1, 0]]
    elif channels == 4:
        native = image[..., [2, 1, 0, 3]]
    valid, encoded = cv2.imencode(".png", native)
    if not valid:
        raise ValueError("OpenCV failed to encode PNG")
    return encoded.tobytes()


def cv_decode(encoded: bytes) -> np.ndarray:
    output = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
    if output is None:
        raise ValueError("OpenCV failed to decode image")
    if output.ndim == 3 and output.shape[2] == 3:
        output = output[..., [2, 1, 0]]
    elif output.ndim == 3 and output.shape[2] == 4:
        output = output[..., [2, 1, 0, 3]]
    return np.ascontiguousarray(output)


def torchvision_array(encoded: bytes, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> np.ndarray:
    source = torch.from_numpy(np.frombuffer(encoded, np.uint8).copy())
    output = torchvision_decode(source, mode=mode).cpu().numpy()
    if output.shape[0] == 1:
        return np.ascontiguousarray(output[0])
    return np.ascontiguousarray(np.moveaxis(output, 0, -1))


def fixtures() -> list[Fixture]:
    output = []
    colors = {1: "gray", 2: "gray_alpha", 3: "rgb", 4: "rgba"}
    for channels in range(1, 5):
        image = pattern(np.dtype(np.uint8), channels)
        encoded = pillow_bytes(image, "PNG")
        output.append(
            Fixture(f"png-u8-c{channels}", "png", encoded, image, colors[channels], "Pillow")
        )
    for channels in range(1, 5):
        image = pattern(np.dtype(np.uint16), channels)
        if channels == 1:
            encoded, producer = pillow_bytes(image, "PNG"), "Pillow"
        elif channels == 2:
            encoded, producer = reference_png(image), "PNG specification oracle"
        else:
            encoded, producer = cv_png_bytes(image), "OpenCV"
        output.append(
            Fixture(f"png-u16-c{channels}", "png", encoded, image, colors[channels], producer)
        )
    palette = Image.new("P", (17, 13))
    palette.putdata((np.arange(17 * 13) % 3).tolist())
    palette.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (256 * 3 - 9))
    palette.info["transparency"] = bytes([255, 128, 0])
    buffer = BytesIO()
    palette.save(buffer, format="PNG", bits=2)
    output.append(
        Fixture(
            "png-palette-trns",
            "png",
            buffer.getvalue(),
            np.asarray(palette.convert("RGBA")),
            "rgba",
            "Pillow",
        )
    )
    jpeg_inputs = (
        ("gray", pattern(np.dtype(np.uint8), 1), "L"),
        ("rgb", pattern(np.dtype(np.uint8), 3), "RGB"),
        ("cmyk", pattern(np.dtype(np.uint8), 4), "CMYK"),
    )
    for color, image, pillow_mode in jpeg_inputs:
        buffer = BytesIO()
        Image.fromarray(image, mode=pillow_mode).save(
            buffer, format="JPEG", quality=93, subsampling=2
        )
        encoded = buffer.getvalue()
        expected = np.asarray(Image.open(BytesIO(encoded))).copy()
        output.append(Fixture(f"jpeg-{color}", "jpeg", encoded, expected, color, "Pillow"))
    progressive = BytesIO()
    Image.fromarray(pattern(np.dtype(np.uint8), 3)).save(
        progressive, format="JPEG", quality=90, progressive=True
    )
    progressive_data = progressive.getvalue()
    output.append(
        Fixture(
            "jpeg-progressive-rgb",
            "jpeg",
            progressive_data,
            np.asarray(Image.open(BytesIO(progressive_data))).copy(),
            "rgb",
            "Pillow",
        )
    )
    return output


def convert_expected(image: np.ndarray, color: str, mode: str) -> np.ndarray:
    if mode == "unchanged":
        return image
    maximum = np.iinfo(image.dtype).max
    if color == "gray":
        gray, alpha = image, np.full(image.shape, maximum, dtype=image.dtype)
        red = green = blue = gray
    elif color == "gray_alpha":
        gray, alpha = image[..., 0], image[..., 1]
        red = green = blue = gray
    elif color in ("rgb", "rgba"):
        red, green, blue = (image[..., index] for index in range(3))
        alpha = image[..., 3] if color == "rgba" else np.full(red.shape, maximum, dtype=image.dtype)
    elif color == "cmyk":
        wide = image.astype(np.uint32)
        key = wide[..., 3]
        channels = [
            ((maximum - wide[..., index]) * (maximum - key) + maximum // 2) // maximum
            for index in range(3)
        ]
        red, green, blue = (channel.astype(image.dtype) for channel in channels)
        alpha = np.full(red.shape, maximum, dtype=image.dtype)
    else:
        raise ValueError(color)
    if mode == "gray":
        wide = (
            299 * red.astype(np.uint32)
            + 587 * green.astype(np.uint32)
            + 114 * blue.astype(np.uint32)
        )
        return ((wide + 500) // 1000).astype(image.dtype)
    rgb = np.stack((red, green, blue), axis=-1)
    return rgb if mode == "rgb" else np.concatenate((rgb, alpha[..., None]), axis=-1)


def pillow_expected(fixture: Fixture, mode: str) -> np.ndarray:
    if mode == "unchanged":
        return fixture.expected
    pillow_mode = {"gray": "L", "rgb": "RGB", "rgba": "RGBA"}[mode]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return np.asarray(Image.open(BytesIO(fixture.encoded)).convert(pillow_mode))


def cv_expected(fixture: Fixture, mode: str) -> np.ndarray:
    image = cv_decode(fixture.encoded)
    if mode == "unchanged":
        return image
    if mode == "gray":
        if image.ndim == 2:
            return image
        conversion = cv2.COLOR_RGB2GRAY if image.shape[2] == 3 else cv2.COLOR_RGBA2GRAY
        return cv2.cvtColor(image, conversion)
    if mode == "rgb":
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image[..., :3]
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    return image


def external_expected(fixture: Fixture, mode: str) -> tuple[np.ndarray, str, int]:
    if fixture.expected.dtype == np.uint8:
        tolerance = 1 if mode == "gray" or fixture.color == "cmyk" else 0
        return pillow_expected(fixture, mode), "Pillow", tolerance
    if fixture.color != "gray_alpha":
        tolerance = 1 if mode == "gray" else 0
        return cv_expected(fixture, mode), "OpenCV", tolerance
    return convert_expected(fixture.expected, fixture.color, mode), "color conversion oracle", 0


def comparison(
    operation: str,
    case: str,
    reference: str,
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: int = 0,
) -> dict[str, Any]:
    shape_match = actual.shape == expected.shape
    dtype_match = actual.dtype == expected.dtype
    max_error = None
    if shape_match:
        difference = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
        max_error = int(difference.max(initial=0))
    contiguous = bool(actual.flags.c_contiguous)
    valid = (
        shape_match
        and dtype_match
        and max_error is not None
        and max_error <= tolerance
        and contiguous
    )
    return {
        "operation": operation,
        "case": case,
        "reference": reference,
        "valid": valid,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "max_abs_error": max_error,
        "tolerance": tolerance,
        "c_contiguous": contiguous,
    }


def decode_and_read_checks(cases: list[Fixture], rows: list[dict[str, Any]]) -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for fixture in cases:
            path = root / f"{fixture.name}.{fixture.format}"
            path.write_bytes(fixture.encoded)
            for mode in MODES:
                expected, reference, tolerance = external_expected(fixture, mode)
                decoded = comparison(
                    "decode",
                    f"{fixture.name}:{mode}",
                    reference,
                    R.decode_image(fixture.encoded, mode=mode),
                    expected,
                    tolerance,
                )
                decoded["fixture_producer"] = fixture.producer
                rows.append(decoded)
                read = comparison(
                    "read",
                    f"{fixture.name}:{mode}",
                    reference,
                    R.read_image(path, mode=mode),
                    expected,
                    tolerance,
                )
                read["fixture_producer"] = fixture.producer
                rows.append(read)


def validate_png_output(
    operation: str,
    case: str,
    encoded: bytes,
    expected: np.ndarray,
    rows: list[dict[str, Any]],
) -> None:
    rows.append(
        comparison(operation, case, "PNG specification oracle", parse_png(encoded), expected)
    )
    channels = 1 if expected.ndim == 2 else expected.shape[2]
    if expected.dtype == np.uint8 or channels == 1:
        pillow = np.asarray(Image.open(BytesIO(encoded)))
        if pillow.dtype != expected.dtype and expected.dtype == np.uint16:
            pillow = pillow.astype(np.uint16)
        rows.append(comparison(operation, case, "Pillow", pillow, expected))
    if channels != 2:
        rows.append(comparison(operation, case, "OpenCV", cv_decode(encoded), expected))
    if expected.dtype == np.uint8 and channels in (1, 3, 4):
        rows.append(
            comparison(operation, case, "Torchvision", torchvision_array(encoded), expected)
        )


def encode_checks(rows: list[dict[str, Any]]) -> None:
    for dtype in (np.dtype(np.uint8), np.dtype(np.uint16)):
        for channels in range(1, 5):
            image = pattern(dtype, channels)
            for compression in range(10):
                case = f"png-{dtype.name}-c{channels}-compression-{compression}"
                encoded = R.encode_image(image, format="png", compression=compression)
                validate_png_output("encode", case, encoded, image, rows)
    for channels in (1, 3):
        image = pattern(np.dtype(np.uint8), channels)
        for quality in range(1, 101):
            case = f"jpeg-u8-c{channels}-quality-{quality}"
            encoded = R.encode_image(image, format="jpeg", quality=quality)
            expected = np.asarray(Image.open(BytesIO(encoded))).copy()
            rows.append(
                comparison(
                    "encode", case, "Pillow", R.decode_image(encoded, mode="unchanged"), expected
                )
            )
            rows.append(comparison("encode", case, "OpenCV", cv_decode(encoded), expected, 1))
            rows.append(
                comparison("encode", case, "Torchvision", torchvision_array(encoded), expected, 1)
            )


def write_checks(rows: list[dict[str, Any]]) -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for dtype in (np.dtype(np.uint8), np.dtype(np.uint16)):
            for channels in range(1, 5):
                image = pattern(dtype, channels)
                for compression in range(10):
                    case = f"png-{dtype.name}-c{channels}-compression-{compression}"
                    path = root / f"{case}.png"
                    R.write_image(path, image, compression=compression)
                    validate_png_output("write", case, path.read_bytes(), image, rows)
        for channels in (1, 3):
            image = pattern(np.dtype(np.uint8), channels)
            for quality in range(1, 101):
                case = f"jpeg-u8-c{channels}-quality-{quality}"
                suffix = ".jpg" if quality != 100 else ".jpeg"
                path = root / f"{case}{suffix}"
                R.write_image(path, image, quality=quality)
                expected = np.asarray(Image.open(path)).copy()
                rows.append(
                    comparison(
                        "write", case, "Pillow", R.read_image(path, mode="unchanged"), expected
                    )
                )
                rows.append(
                    comparison("write", case, "OpenCV", cv_decode(path.read_bytes()), expected, 1)
                )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_operation = {}
    for operation in ("read", "decode", "encode", "write"):
        selected = [row for row in rows if row["operation"] == operation]
        by_operation[operation] = {
            "checks": len(selected),
            "passed": sum(row["valid"] for row in selected),
        }
    references = {}
    for reference in sorted({row["reference"] for row in rows}):
        selected = [row for row in rows if row["reference"] == reference]
        references[reference] = {
            "checks": len(selected),
            "passed": sum(row["valid"] for row in selected),
        }
    return {
        "valid": all(row["valid"] for row in rows),
        "checks": len(rows),
        "passed": sum(row["valid"] for row in rows),
        "by_operation": by_operation,
        "by_reference": references,
    }


def run_planned(items: list[dict[str, Any]], repetition: int) -> list[dict[str, Any]]:
    if len(items) != 1 or items[0]["factory"] != "interoperability":
        raise ValueError("I/O parity expects the interoperability case")
    rows: list[dict[str, Any]] = []
    decode_and_read_checks(fixtures(), rows)
    encode_checks(rows)
    write_checks(rows)
    summary = summarize(rows)
    item = items[0]
    route = item["route"]
    return [
        {
            "case_id": item["case_id"],
            "route_id": route["id"],
            "participant": route["participant"],
            "variant": route["variant"],
            "role": route["role"],
            "size": None,
            "repetition": repetition,
            "case_order": 1,
            "validation": {"summary": summary, "checks": rows},
            "valid": summary["valid"],
        }
    ]
