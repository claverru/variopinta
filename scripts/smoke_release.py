from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata, resources
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import variopinta as V


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--torch", action="store_true")
    return parser.parse_args()


def smoke_base(expected_version: str) -> np.ndarray:
    distribution = metadata.distribution("variopinta")
    assert distribution.version == expected_version
    packaged = {str(path) for path in distribution.files or ()}
    assert any(path.endswith("dist-info/licenses/LICENSE") for path in packaged)
    assert any(path.endswith("dist-info/licenses/THIRD_PARTY_NOTICES") for path in packaged)
    assert any(path.endswith("dist-info/RECORD") for path in packaged)
    assert resources.files(V).joinpath("py.typed").is_file()

    values = np.arange(13 * 19 * 3, dtype=np.uint64)
    contiguous = ((values * 73 + values // 7 * 19) & 255).astype(np.uint8).reshape(13, 19, 3)
    source = contiguous[:, ::2]
    snapshot = source.copy()
    reference = V.Compose([V.Invert(), V.HorizontalFlip(p=1.0), V.Normalize()], seed=137)
    expected = reference(source, key=11)
    output = reference.compile()(source, key=11)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(source, snapshot)
    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    assert not np.shares_memory(source, output)

    encoded_png = V.encode_image(source, format="png")
    decoded_png = V.decode_image(bytearray(encoded_png), mode="unchanged")
    np.testing.assert_array_equal(decoded_png, source)
    assert decoded_png.flags.c_contiguous
    assert not np.shares_memory(source, decoded_png)

    encoded_jpeg = V.encode_image(source, format="jpeg", quality=91)
    decoded_jpeg = V.decode_image(memoryview(encoded_jpeg))
    assert decoded_jpeg.shape == source.shape
    assert decoded_jpeg.dtype == np.uint8
    assert decoded_jpeg.flags.c_contiguous

    with TemporaryDirectory() as directory:
        path = Path(directory) / "round-trip.png"
        V.write_image(path, source)
        np.testing.assert_array_equal(V.read_image(path), source)

    identity = {
        "brightness": (1.0, 1.0),
        "contrast": (1.0, 1.0),
        "saturation": (1.0, 1.0),
    }
    jitter_values = np.arange(3 * 34 * 3, dtype=np.uint64)
    jitter_source = (
        ((jitter_values * 73 + jitter_values // 7 * 19) & 255)
        .astype(np.uint8)
        .reshape(3, 34, 3)[:, ::2]
    )
    for configuration in (
        {**identity, "brightness": (1_000.0, 1_000.0)},
        {**identity, "contrast": (1_000.0, 1_000.0)},
        {**identity, "saturation": (1_000.0, 1_000.0)},
    ):
        jitter = V.Compose([V.ColorJitter(**configuration)], seed=137)
        np.testing.assert_array_equal(
            jitter(jitter_source, key=29), jitter.compile()(jitter_source, key=29)
        )
    positive = np.array([[[255, 2, 3], [0, 0, 0]]], dtype=np.uint8)
    brightness = V.Compose(
        [V.ColorJitter(**{**identity, "brightness": (1_000.0, 1_000.0)})], seed=137
    )
    saturated = np.where(positive == 0, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(brightness(positive, key=29), saturated)
    np.testing.assert_array_equal(brightness.compile()(positive, key=29), saturated)

    blur_source = np.full((3, 4, 3), 73, dtype=np.uint8)
    wide_blur = V.Compose([V.GaussianBlur(101, 1_000_000.0)], seed=137)
    blurred = wide_blur.compile()(blur_source, key=29)
    np.testing.assert_array_equal(blurred, wide_blur(blur_source, key=29))
    np.testing.assert_array_equal(blurred, blur_source)

    return source


def smoke_torch(source: np.ndarray, required: bool) -> None:
    pipeline = V.Compose([V.Normalize(), V.ToTorch()], seed=137).compile()
    if not required:
        try:
            pipeline(source, key=11)
        except ImportError as error:
            assert "ToTorch requires PyTorch" in str(error)
            return
        raise AssertionError("ToTorch unexpectedly succeeded without PyTorch")

    import torch

    expected = V.Compose([V.Normalize()], seed=137)(source, key=11)
    output = pipeline(source, key=11)
    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert output.is_contiguous()
    np.testing.assert_array_equal(output.numpy(), np.moveaxis(expected, 2, 0))


def main() -> None:
    args = parse_args()
    source = smoke_base(args.expected_version)
    smoke_torch(source, args.torch)
    print(
        json.dumps(
            {
                "artifact": metadata.version("variopinta"),
                "numpy": np.__version__,
                "python": ".".join(map(str, sys.version_info[:3])),
                "torch": args.torch,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
