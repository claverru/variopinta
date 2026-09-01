from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
import variopinta as R
from common import RESULTS, control_cpu, make_images, summarize_observations
from PIL import Image


def time_operation(
    function: Callable[[], Any], warmup: int = 10, iterations: int = 100
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        output = function()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    summary = summarize_observations(samples)
    return {
        **summary,
        "operations_per_sec": summary["images_per_sec"],
        "iterations": iterations,
        "samples": len(samples),
        "observations_ms": samples,
        "block_size": 1,
        "warmup_calls": warmup,
        "output_bytes": len(output) if isinstance(output, bytes) else None,
    }


def pillow_decode(data: bytes) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(Image.open(BytesIO(data)).convert("RGB")))


def opencv_decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pillow_read(path: Path) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))


def opencv_read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pillow_encode(image: np.ndarray, format: str) -> bytes:
    output = BytesIO()
    options = {"quality": 95, "subsampling": 2} if format == "jpeg" else {"compress_level": 6}
    Image.fromarray(image).save(output, format=format.upper(), **options)
    return output.getvalue()


def opencv_encode(image: np.ndarray, format: str) -> bytes:
    native = image[..., ::-1]
    extension = f".{format}"
    options = (
        [cv2.IMWRITE_JPEG_QUALITY, 95] if format == "jpeg" else [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )
    valid, output = cv2.imencode(extension, native, options)
    if not valid:
        raise RuntimeError("OpenCV encoding failed")
    return output.tobytes()


def main() -> None:
    cpu = control_cpu()
    image = make_images(512, 1)[0]
    rows = []
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for format in ("jpeg", "png"):
            encoded = pillow_encode(image, format)
            input_path = root / f"input.{format}"
            input_path.write_bytes(encoded)
            functions: dict[tuple[str, str], Callable[[], Any]] = {
                ("decode", "rust"): lambda data=encoded: R.decode_image(data),
                ("decode", "pillow"): lambda data=encoded: pillow_decode(data),
                ("decode", "opencv"): lambda data=encoded: opencv_decode(data),
                ("read", "rust"): lambda path=input_path: R.read_image(path),
                ("read", "pillow"): lambda path=input_path: pillow_read(path),
                ("read", "opencv"): lambda path=input_path: opencv_read(path),
                ("encode", "rust"): lambda fmt=format: R.encode_image(image, format=fmt),
                ("encode", "pillow"): lambda fmt=format: pillow_encode(image, fmt),
                ("encode", "opencv"): lambda fmt=format: opencv_encode(image, fmt),
                ("write", "rust"): lambda fmt=format: R.write_image(root / f"rust.{fmt}", image),
                ("write", "pillow"): lambda fmt=format: Image.fromarray(image).save(
                    root / f"pillow.{fmt}",
                    format=fmt.upper(),
                    **(
                        {"quality": 95, "subsampling": 2}
                        if fmt == "jpeg"
                        else {"compress_level": 6}
                    ),
                ),
                ("write", "opencv"): lambda fmt=format: cv2.imwrite(
                    str(root / f"opencv.{fmt}"),
                    image[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                    if fmt == "jpeg"
                    else [cv2.IMWRITE_PNG_COMPRESSION, 6],
                ),
            }
            for (operation, backend), function in functions.items():
                rows.append(
                    {
                        "format": format,
                        "operation": operation,
                        "backend": backend,
                        **time_operation(function),
                    }
                )
    output = {"cpu": cpu, "rows": rows}
    path = RESULTS / "raw" / "io-performance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    csv_path = RESULTS / "csv" / "io-performance.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        fields = [key for key in rows[0] if key != "observations_ms"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "observations_ms"} for row in rows
        )
    for row in rows:
        print(
            f"{row['format']:4} {row['operation']:6} {row['backend']:6} {row['median_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
