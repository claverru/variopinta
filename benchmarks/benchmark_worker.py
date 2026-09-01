from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from common import control_cpu, metadata, summarize_observations, write_json
from correctness import run_correctness_checks
from io_benchmark import run_io_benchmarks
from microbenchmarks import run_microbenchmarks
from pipeline_benchmark import run_pipeline_benchmarks


def measure_boundary(backend: str, quick: bool) -> dict[str, object]:
    import numpy as np

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    if backend == "torchvision":

        def fn(value: object) -> object:
            return value

        value = image
    elif backend in {"albumentations", "albumentationsx"}:
        import albumentations as A

        compose = A.Compose([])

        def fn(value: object) -> object:
            return compose(image=value)["image"]

        value = image
    else:
        from variopinta._variopinta import boundary

        fn = boundary
        value = image
    calls = 25 if quick else 100
    blocks = 100 if quick else 1000
    for _ in range(10):
        fn(value)
    samples = []
    for _ in range(blocks):
        start = time.perf_counter_ns()
        for _ in range(calls):
            fn(value)
        samples.append((time.perf_counter_ns() - start) / calls / 1_000_000)
    summary = summarize_observations(samples)
    return {
        "kind": "boundary",
        "backend": backend,
        **summary,
        "calls": calls * blocks,
        "iterations": calls * blocks,
        "observations_ms": samples,
        "block_size": calls,
        "warmup_calls": 10,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=("torchvision", "albumentations", "albumentationsx", "rust"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    cpu = control_cpu()
    rows = [measure_boundary(args.backend, args.quick)]
    rows.extend(run_correctness_checks(args.backend))
    rows.extend(run_microbenchmarks(args.backend, args.quick))
    rows.extend(run_pipeline_benchmarks(args.backend, args.quick))
    rows.extend(run_io_benchmarks(args.backend, args.quick))
    write_json(args.output, {"metadata": metadata(args.backend, cpu), "rows": rows})


if __name__ == "__main__":
    main()
