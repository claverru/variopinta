from __future__ import annotations

import hashlib
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
import variopinta as R
from common import make_images, time_calls_adaptive
from PIL import Image


def time_operation(
    function: Callable[[], Any], warmup: int = 10, iterations: int = 100
) -> dict[str, Any]:
    timing, output = time_calls_adaptive(
        lambda _: function(),
        [None],
        budget_ms=100.0,
        warmup_calls=warmup,
        min_samples=min(7, iterations),
        max_calls=iterations,
    )
    return {
        **timing,
        "operations_per_sec": timing["images_per_sec"],
        "output_bytes": len(output) if isinstance(output, bytes) else None,
        "output_sha256": output_sha256(output),
    }


def output_sha256(output: object) -> str | None:
    if isinstance(output, bytes):
        return hashlib.sha256(output).hexdigest()
    if isinstance(output, np.ndarray):
        return hashlib.sha256(output.tobytes()).hexdigest()
    return None


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


def three_call_encoded(data: bytes, pipeline: R.CompiledCompose, format: str) -> bytes:
    decoded = R.decode_image(data)
    transformed = pipeline(decoded, key=11)
    return R.encode_image(transformed, format=format)


def three_call_path(
    source: Path, destination: Path, pipeline: R.CompiledCompose, format: str
) -> None:
    decoded = R.read_image(source)
    transformed = pipeline(decoded, key=11)
    R.write_image(destination, transformed, format=format)


def run_planned(items: list[dict[str, Any]], quick: bool, repetition: int) -> list[dict[str, Any]]:
    image = make_images(512, 1)[0]
    rows = []
    with TemporaryDirectory() as directory:
        root = Path(directory)
        contexts: dict[str, dict[tuple[str, str], Callable[[], Any]]] = {}
        for format_name in {item["factory"].split("|", 1)[0] for item in items}:
            encoded = pillow_encode(image, format_name)
            input_path = root / f"input.{format_name}"
            input_path.write_bytes(encoded)
            array_pipeline = R.Compose([R.Resize(448, 448), R.Invert()], seed=137).compile()
            encoded_input_pipeline = R.Compose(
                [R.Resize(448, 448), R.Invert()], seed=137, input=R.EncodedInput()
            ).compile()
            encoded_output_pipeline = R.Compose(
                [R.Resize(448, 448), R.Invert()],
                seed=137,
                output=R.EncodedOutput(format=format_name),
            ).compile()
            encoded_pipeline = R.Compose(
                [R.Resize(448, 448), R.Invert()],
                seed=137,
                input=R.EncodedInput(),
                output=R.EncodedOutput(format=format_name),
            ).compile()
            path_pipeline = R.Compose(
                [R.Resize(448, 448), R.Invert()],
                seed=137,
                input=R.PathInput(),
                output=R.PathOutput(format=format_name),
            ).compile()
            functions: dict[tuple[str, str], Callable[[], Any]] = {
                ("decode", "variopinta"): lambda data=encoded: R.decode_image(data),
                ("decode", "pillow"): lambda data=encoded: pillow_decode(data),
                ("decode", "opencv"): lambda data=encoded: opencv_decode(data),
                ("read", "variopinta"): lambda path=input_path: R.read_image(path),
                ("read", "pillow"): lambda path=input_path: pillow_read(path),
                ("read", "opencv"): lambda path=input_path: opencv_read(path),
                ("encode", "variopinta"): lambda fmt=format_name: R.encode_image(image, format=fmt),
                ("encode", "pillow"): lambda fmt=format_name: pillow_encode(image, fmt),
                ("encode", "opencv"): lambda fmt=format_name: opencv_encode(image, fmt),
                ("write", "variopinta"): lambda fmt=format_name: R.write_image(
                    root / f"variopinta.{fmt}", image
                ),
                ("write", "pillow"): lambda fmt=format_name: Image.fromarray(image).save(
                    root / f"pillow.{fmt}",
                    format=fmt.upper(),
                    **(
                        {"quality": 95, "subsampling": 2}
                        if fmt == "jpeg"
                        else {"compress_level": 6}
                    ),
                ),
                ("write", "opencv"): lambda fmt=format_name: cv2.imwrite(
                    str(root / f"opencv.{fmt}"),
                    image[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                    if fmt == "jpeg"
                    else [cv2.IMWRITE_PNG_COMPRESSION, 6],
                ),
                ("pipeline-three-call-encoded", "variopinta"): lambda data=encoded,
                pipeline=array_pipeline,
                fmt=format_name: three_call_encoded(data, pipeline, fmt),
                ("pipeline-three-call-path", "variopinta"): lambda source=input_path,
                pipeline=array_pipeline,
                fmt=format_name: three_call_path(source, root / f"three-call.{fmt}", pipeline, fmt),
                ("pipeline-encoded-return", "variopinta"): lambda data=encoded,
                pipeline=encoded_input_pipeline: pipeline(data, key=11),
                (
                    "pipeline-array-encoded",
                    "variopinta",
                ): lambda pipeline=encoded_output_pipeline: pipeline(image, key=11),
                ("pipeline-encoded-encoded", "variopinta"): lambda data=encoded,
                pipeline=encoded_pipeline: pipeline(data, key=11),
                ("pipeline-path-path", "variopinta"): lambda source=input_path,
                pipeline=path_pipeline,
                fmt=format_name: pipeline(source, destination=root / f"native.{fmt}", key=11),
            }
            contexts[format_name] = functions

        for order, item in enumerate(items, start=1):
            format_name, operation = item["factory"].split("|", 1)
            route = item["route"]
            function = contexts[format_name][(operation, route["participant"])]
            policy = dict(item["timing"])
            if quick:
                policy.update(
                    {
                        "budget_ms": min(float(policy["budget_ms"]), 10.0),
                        "warmup_calls": min(int(policy["warmup_calls"]), 2),
                        "min_samples": 3,
                        "max_calls": 64,
                    }
                )
            timing, output = time_calls_adaptive(
                lambda _, selected=function: selected(), [None], **policy
            )
            rows.append(
                {
                    "case_id": item["case_id"],
                    "route_id": route["id"],
                    "participant": route["participant"],
                    "variant": route["variant"],
                    "role": route["role"],
                    "size": 512,
                    "repetition": repetition,
                    "case_order": order,
                    **timing,
                    "output_bytes": len(output) if isinstance(output, bytes) else None,
                    "output_sha256": output_sha256(output),
                    "valid": True,
                }
            )
    return rows
