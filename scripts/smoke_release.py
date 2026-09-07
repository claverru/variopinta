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
    reference = V.Pipeline([V.Invert(), V.HorizontalFlip(p=1.0), V.Normalize()], seed=137)
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
        root = Path(directory)
        path = root / "round-trip.png"
        V.write_image(path, source)
        np.testing.assert_array_equal(V.read_image(path), source)

        expected = V.Pipeline([V.Invert()], seed=137)(source, key=11)
        encoded_port = V.Image(
            V.Encoded(), outputs=(V.Encode("png", name="encoded"),), name="image"
        )
        encoded_pipeline = V.Pipeline(
            [V.Invert()],
            seed=137,
            targets=(encoded_port,),
        ).compile()
        encoded_result = encoded_pipeline(
            image=encoded_port.bind(encoded_png), key=11
        ).image.encoded
        np.testing.assert_array_equal(V.decode_image(encoded_result), expected)

        written = V.Write("png", name="written")
        path_port = V.Image(V.Path(), outputs=(written,), name="image")
        path_pipeline = V.Pipeline(
            [V.Invert()],
            seed=137,
            targets=(path_port,),
        ).compile()
        destination = root / "pipeline.png"
        assert (
            path_pipeline(
                image=path_port.bind(path, written.bind(destination)), key=11
            ).image.written
            == destination
        )
        np.testing.assert_array_equal(V.read_image(destination), expected)

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
        jitter = V.Pipeline([V.ColorJitter(**configuration)], seed=137)
        np.testing.assert_array_equal(
            jitter(jitter_source, key=29), jitter.compile()(jitter_source, key=29)
        )
    positive = np.array([[[255, 2, 3], [0, 0, 0]]], dtype=np.uint8)
    brightness = V.Pipeline(
        [V.ColorJitter(**{**identity, "brightness": (1_000.0, 1_000.0)})], seed=137
    )
    saturated = np.where(positive == 0, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(brightness(positive, key=29), saturated)
    np.testing.assert_array_equal(brightness.compile()(positive, key=29), saturated)

    blur_source = np.full((3, 4, 3), 73, dtype=np.uint8)
    wide_blur = V.Pipeline([V.GaussianBlur(101, 1_000_000.0)], seed=137)
    blurred = wide_blur.compile()(blur_source, key=29)
    np.testing.assert_array_equal(blurred, wide_blur(blur_source, key=29))
    np.testing.assert_array_equal(blurred, blur_source)

    return source


def smoke_grayscale() -> None:
    source = np.arange(7 * 22, dtype=np.uint8).reshape(7, 22)[:, ::2]
    reference = V.Pipeline([V.Invert(), V.Normalize(mean=0.5, std=0.5)], seed=137)
    expected = (255 - source).astype(np.float32) / 127.5 - 1
    for pipeline in (reference, reference.compile()):
        for image in (source, source[..., None]):
            output = pipeline(image, key=11)
            assert output.shape == image.shape
            assert output.dtype == np.float32
            assert output.flags.c_contiguous
            assert not np.shares_memory(output, source)
            np.testing.assert_allclose(output.reshape(source.shape), expected, atol=1e-7)

    target = V.Image(
        V.Encoded(), outputs=V.Encode("png", name="encoded"), name="image", decode_mode="gray"
    )
    pipeline = V.Pipeline(
        [V.ColorJitter(brightness=0, contrast=0, saturation=0.8, hue=0.4)], targets=target
    ).compile()
    explanation = pipeline.explain()
    assert explanation["schema_version"] == 5
    assert explanation["pixel_passes"] == 0
    assert explanation["fallbacks"] == []
    encoded = V.encode_image(source, format="png")
    output = pipeline(image=target.bind(encoded), key=11).image.encoded
    np.testing.assert_array_equal(V.decode_image(output, mode="unchanged"), source)


def smoke_torch(source: np.ndarray, required: bool) -> None:
    tensor = V.ReturnTensor(name="tensor")
    target = V.Image(name="image", outputs=(tensor,))
    pipeline = V.Pipeline([V.Normalize()], seed=137, targets=(target,)).compile()
    if not required:
        try:
            pipeline(image=target.bind(source), key=11)
        except ImportError as error:
            assert "ReturnTensor requires PyTorch" in str(error)
            return
        raise AssertionError("ReturnTensor unexpectedly succeeded without PyTorch")

    import torch

    expected = V.Pipeline([V.Normalize()], seed=137)(source, key=11)
    output = pipeline(image=target.bind(source), key=11).image.tensor
    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert output.is_contiguous()
    np.testing.assert_array_equal(output.numpy(), np.moveaxis(expected, 2, 0))

    gray = source[..., 0]
    pipeline = V.Pipeline([V.Normalize(mean=0.5, std=0.5)], seed=137, targets=target).compile()
    output = pipeline(image=target.bind(gray), key=11).image.tensor
    assert output.shape == (1, *gray.shape)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert output.is_contiguous()
    np.testing.assert_allclose(output.numpy()[0], gray.astype(np.float32) / 127.5 - 1, atol=1e-7)


def main() -> None:
    args = parse_args()
    source = smoke_base(args.expected_version)
    smoke_grayscale()
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
