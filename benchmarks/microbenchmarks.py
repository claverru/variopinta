from __future__ import annotations

from typing import Any

from adapters import Adapter
from common import SIZES, TRANSFORMS, iterations_for, make_images, output_facts, time_calls


def run_microbenchmarks(backend: str, quick: bool) -> list[dict[str, Any]]:
    adapter = Adapter(backend)
    rows = []
    for size in SIZES:
        images = make_images(size)
        native = adapter.native_inputs(images)
        warmup, iterations = iterations_for(size, quick)
        for name in TRANSFORMS:
            transform = adapter.build_micro(name, size)
            timing, output = time_calls(transform, native, warmup, iterations)
            facts = output_facts(output)
            expected_size = (
                max(32, size * 3 // 4) if name in {"Resize", "RandomCrop", "CenterCrop"} else size
            )
            expected_shape = (
                [3, expected_size, expected_size]
                if backend == "torchvision"
                else [expected_size, expected_size, 3]
            )
            valid = facts["shape"] == expected_shape and facts["finite"] and facts["c_contiguous"]
            if name == "Normalize":
                valid = (
                    valid
                    and facts["container"]
                    == ("torch.Tensor" if backend == "torchvision" else "numpy.ndarray")
                    and facts["dtype"] == "float32"
                    and facts["min"] >= -5
                    and facts["max"] <= 5
                )
            else:
                valid = (
                    valid
                    and facts["dtype"] == "uint8"
                    and facts["min"] >= 0
                    and facts["max"] <= 255
                )
            rows.append(
                {
                    "kind": "micro",
                    "backend": backend,
                    "transform": name,
                    "size": size,
                    **timing,
                    "validation": facts,
                    "valid": valid,
                }
            )
        antialiased_resize = adapter.build_antialiased_resize(size)
        if antialiased_resize is not None:
            timing, output = time_calls(antialiased_resize, native, warmup, iterations)
            facts = output_facts(output)
            expected_size = max(32, size * 3 // 4)
            expected_shape = (
                [3, expected_size, expected_size]
                if backend == "torchvision"
                else [expected_size, expected_size, 3]
            )
            valid = (
                facts["shape"] == expected_shape
                and facts["dtype"] == "uint8"
                and facts["min"] >= 0
                and facts["max"] <= 255
                and facts["finite"]
                and facts["c_contiguous"]
            )
            rows.append(
                {
                    "kind": "resize_policy",
                    "policy": "antialias",
                    "backend": backend,
                    "transform": "Resize",
                    "size": size,
                    **timing,
                    "validation": facts,
                    "valid": valid,
                }
            )
    return rows
