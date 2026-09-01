from __future__ import annotations

from typing import Any

from adapters import Adapter
from common import PIPELINES, SIZES, iterations_for, make_images, output_facts, time_calls


def run_pipeline_benchmarks(backend: str, quick: bool) -> list[dict[str, Any]]:
    adapter = Adapter(backend)
    rows = []
    for size in SIZES:
        images = make_images(size)
        native = adapter.native_inputs(images)
        for pipeline in PIPELINES:
            transform = adapter.build_pipeline(size, pipeline)
            warmup, iterations = iterations_for(size, quick)
            timing, output = time_calls(
                transform,
                native,
                warmup,
                iterations,
                block_size=4 if quick else 16,
            )
            facts = output_facts(output)
            expected_shape = [3, 224, 224] if backend == "torchvision" else [224, 224, 3]
            expected_container = "torch.Tensor" if backend == "torchvision" else "numpy.ndarray"
            valid = (
                facts["container"] == expected_container
                and facts["shape"] == expected_shape
                and facts["dtype"] == "float32"
                and facts["finite"]
                and facts["c_contiguous"]
                and facts["min"] >= -5
                and facts["max"] <= 5
            )
            rows.append(
                {
                    "kind": "pipeline_memory",
                    "backend": backend,
                    "pipeline": pipeline,
                    "size": size,
                    **timing,
                    "validation": facts,
                    "valid": valid,
                }
            )
    return rows
