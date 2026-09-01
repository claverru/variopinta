from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from common import RESULTS, SIZES, iterations_for, make_images, output_facts, time_calls
from PIL import Image


def _images(size: int) -> list[Path]:
    directory = RESULTS / "raw" / "images"
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"synthetic-{size}-{index:02d}.jpg" for index in range(8)]
    for path, array in zip(paths, make_images(size), strict=True):
        if not path.exists():
            Image.fromarray(array).save(path, format="JPEG", quality=90, subsampling=2)
    return paths


def _reader(backend: str) -> Callable[[Path], Any]:
    if backend == "torchvision":
        from torchvision.io import ImageReadMode, read_image

        return lambda path: read_image(str(path), mode=ImageReadMode.RGB).contiguous()
    if backend in {"albumentations", "albumentationsx"}:
        import cv2

        def read(path: Path) -> Any:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to read {path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return read

    from variopinta import read_image

    return read_image


def run_io_benchmarks(backend: str, quick: bool) -> list[dict[str, Any]]:
    reader = _reader(backend)
    rows = []
    for size in SIZES:
        paths = _images(size)
        warmup, iterations = iterations_for(size, quick)
        timing, output = time_calls(reader, paths, warmup, iterations)
        facts = output_facts(output)
        expected_shape = [3, size, size] if backend == "torchvision" else [size, size, 3]
        valid = (
            facts["shape"] == expected_shape and facts["dtype"] == "uint8" and facts["c_contiguous"]
        )
        rows.append(
            {
                "kind": "io_jpeg",
                "backend": backend,
                "size": size,
                **timing,
                "validation": facts,
                "valid": valid,
            }
        )
    return rows
