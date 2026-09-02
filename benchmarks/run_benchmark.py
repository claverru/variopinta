from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

from common import (
    PIPELINES,
    RESULTS,
    ROOT,
    TRANSFORMS,
    aggregate_runs,
    evidence_matches_code,
    evidence_provenance,
    write_json,
)
from environments import python_for, require_environments
from plot_results import BACKENDS, LABELS, generate_plots
from run_catalog_audit import render_catalog_audit_summary
from run_catalog_benchmark import render_catalog_benchmark_summary


def evidence_paths(quick: bool) -> dict[str, Path]:
    suffix = "-quick" if quick else ""
    return {
        "runs": RESULTS / "raw" / f"benchmark{suffix}-runs.json",
        "metadata": RESULTS / "raw" / f"metadata{suffix}.json",
        "csv": RESULTS / "csv" / f"benchmark-results{suffix}.csv",
        "plots": RESULTS / "plots" / ("quick" if quick else ""),
        "report": RESULTS / ("benchmark-quick.md" if quick else "benchmark.md"),
    }


def run_workers(backends: list[str], quick: bool, repetitions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata: dict[str, list[dict[str, Any]]] = {backend: [] for backend in backends}
    execution_order = []
    for repetition in range(1, repetitions + 1):
        order = list(backends)
        shift = (repetition - 1) % len(order)
        order = order[shift:] + order[:shift]
        for position, backend in enumerate(order, start=1):
            directory = "quick-runs" if quick else "runs"
            output = RESULTS / "raw" / directory / f"{backend}-{repetition}.json"
            command = [
                str(python_for(backend)),
                str(ROOT / "benchmarks" / "benchmark_worker.py"),
                "--backend",
                backend,
                "--output",
                str(output),
            ]
            if quick:
                command.append("--quick")
            print(f"[{repetition}/{repetitions}] {backend}", flush=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "benchmarks")
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            payload = json.loads(output.read_text())
            for row in payload["rows"]:
                row["repetition"] = repetition
                row["worker"] = f"{backend}-{repetition}"
                row["backend_position"] = position
                rows.append(row)
            metadata[backend].append(payload["metadata"])
            execution_order.append(
                {"repetition": repetition, "position": position, "backend": backend}
            )
    paths = evidence_paths(quick)
    write_json(
        paths["runs"],
        {
            "schema_version": 2,
            "quick": quick,
            "repetitions": repetitions,
            "overrides": {"backends": backends, "quick": quick, "repetitions": repetitions},
            "input_definition": {
                "dtype": "uint8",
                "layout": "HWC RGB",
                "sizes": [224, 512, 1024],
                "images_per_size": 8,
                "seed": 137,
                "materialization": "native contiguous input prepared before timing; output materialized inside timing",
            },
            "execution_order": execution_order,
            "provenance": evidence_provenance(),
            "metadata": metadata,
            "rows": rows,
        },
    )
    write_json(paths["metadata"], metadata)
    return aggregate_runs(rows)


def write_results(rows: list[dict[str, Any]], quick: bool) -> None:
    paths = evidence_paths(quick)
    fields = sorted(
        {key for row in rows for key in row if key not in {"validation", "worker_observations"}}
    ) + ["validation"]
    path = paths["csv"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat.pop("worker_observations", None)
            flat["validation"] = json.dumps(flat.get("validation", {}), sort_keys=True)
            writer.writerow(flat)


def geometric_mean(values: list[float]) -> float:
    return (
        math.exp(sum(math.log(value) for value in values) / len(values)) if values else float("nan")
    )


def render_benchmark_report(
    rows: list[dict[str, Any]],
    quick: bool,
    repetitions: int,
    run_metadata: dict[str, list[dict[str, Any]]],
) -> None:
    reference_run = next(iter(run_metadata.values()))[0]
    package_names = {
        "torchvision": "torchvision",
        "albumentations": "albumentations",
        "albumentationsx": "albumentationsx",
        "rust": "variopinta",
    }
    compared_backends = []
    for backend in BACKENDS:
        if backend not in run_metadata:
            continue
        version = run_metadata[backend][0].get("packages", {}).get(package_names[backend])
        compared_backends.append(f"{LABELS[backend]} {version}" if version else LABELS[backend])
    backend_summary = ", ".join(compared_backends)
    processor = reference_run["processor"]
    platform = reference_run["platform"]
    affinity = reference_run["thread_control"]["cpu_affinity_after"]
    pinned_cpu = affinity[0] if len(affinity) == 1 else affinity
    spreads = [row["run_spread_percent"] for row in rows if "run_spread_percent" in row]
    variation_summary = (
        f"Across timed rows, worker-median spread ranges from {min(spreads):.2f}% to "
        f"{max(spreads):.2f}% (median {statistics.median(spreads):.2f}%)."
        if repetitions > 1 and spreads
        else "This diagnostic has one worker per row, so it cannot estimate run-to-run variation."
    )
    micro = [
        row
        for row in rows
        if row["kind"] == "micro" and row.get("valid", True) and row["size"] == 512
    ]
    pipeline = [
        row
        for row in rows
        if row["kind"] == "pipeline_memory" and row.get("valid", True) and row["size"] == 512
    ]
    antialiased_resize = [
        row
        for row in rows
        if row["kind"] == "resize_policy"
        and row.get("policy") == "antialias"
        and row.get("valid", True)
        and row["size"] == 512
    ]
    complete = not quick and repetitions >= 3 and set(BACKENDS) <= {row["backend"] for row in rows}

    micro_lines = [
        "| Transform | Torchvision | Albumentations | AlbumentationsX | Variopinta | Variopinta vs AX |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for transform in TRANSFORMS:
        values = {
            backend: next(
                (
                    row["median_ms"]
                    for row in micro
                    if row["backend"] == backend and row["transform"] == transform
                ),
                float("nan"),
            )
            for backend in BACKENDS
        }
        micro_lines.append(
            f"| {transform} | {values['torchvision']:.3f} | {values['albumentations']:.3f} | "
            f"{values['albumentationsx']:.3f} | {values['rust']:.3f} | "
            f"{values['albumentationsx'] / values['rust']:.2f}× |"
        )

    pipeline_values = {
        name: {
            backend: next(
                (
                    row["median_ms"]
                    for row in pipeline
                    if row["backend"] == backend and row["pipeline"] == name
                ),
                float("nan"),
            )
            for backend in BACKENDS
        }
        for name in PIPELINES
    }
    io_values = {
        backend: next(
            (
                row["median_ms"]
                for row in rows
                if row["kind"] == "io_jpeg" and row["backend"] == backend and row["size"] == 512
            ),
            float("nan"),
        )
        for backend in BACKENDS
    }
    pipeline_lines = [
        "| Pipeline, 512→224 | Torchvision | Albumentations | AlbumentationsX | Variopinta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in PIPELINES:
        values = pipeline_values[name]
        pipeline_lines.append(
            f"| {name} | {values['torchvision']:.3f} | {values['albumentations']:.3f} | "
            f"{values['albumentationsx']:.3f} | {values['rust']:.3f} |"
        )
    pipeline_table = "\n".join(pipeline_lines)

    antialias_values = {
        backend: next(
            (row["median_ms"] for row in antialiased_resize if row["backend"] == backend),
            float("nan"),
        )
        for backend in ("torchvision", "rust")
    }
    antialias_table = "\n".join(
        [
            "| Resize policy, 512→384 | Torchvision | Albumentations | AlbumentationsX | Variopinta |",
            "|---|---:|---:|---:|---:|",
            f"| bilinear, antialias=True | {antialias_values['torchvision']:.3f} | — | — | {antialias_values['rust']:.3f} |",
        ]
    )

    if complete:
        albu_over_ax = geometric_mean(
            [
                next(
                    row["median_ms"]
                    for row in micro
                    if row["backend"] == "albumentations" and row["transform"] == transform
                )
                / next(
                    row["median_ms"]
                    for row in micro
                    if row["backend"] == "albumentationsx" and row["transform"] == transform
                )
                for transform in TRANSFORMS
            ]
        )
        rust_vs_ax_micro = geometric_mean(
            [
                next(
                    row["median_ms"]
                    for row in micro
                    if row["backend"] == "albumentationsx" and row["transform"] == transform
                )
                / next(
                    row["median_ms"]
                    for row in micro
                    if row["backend"] == "rust" and row["transform"] == transform
                )
                for transform in TRANSFORMS
            ]
        )
        rust_vs_ax_pipeline = (
            pipeline_values["classic"]["albumentationsx"] / pipeline_values["classic"]["rust"]
        )
        rust_vs_albu_pipeline = (
            pipeline_values["classic"]["albumentations"] / pipeline_values["classic"]["rust"]
        )
        rust_vs_cv2_decode = io_values["albumentations"] / io_values["rust"]
        correctness = [row for row in rows if row["kind"] == "correctness"]
        contracts = ", ".join(f"{LABELS[row['backend']]} {row['cases']}" for row in correctness)
        status = (
            "pass"
            if len(correctness) == len(BACKENDS) and all(row.get("valid") for row in correctness)
            else "do not all pass"
        )
        summary = (
            f"Functional contracts {status} ({contracts}). AX is {1.0 / albu_over_ax:.2f}× slower than "
            "Albumentations across the isolated transforms; "
            f"Rust/AX is {rust_vs_ax_micro:.2f}× across optimized microbenchmarks and "
            f"{rust_vs_ax_pipeline:.2f}× in the classic in-memory pipeline. Rust also beats Albumentations by "
            f"{rust_vs_albu_pipeline:.2f}× in that pipeline. Native JPEG loading is "
            f"{rust_vs_cv2_decode:.2f}× faster than OpenCV read plus RGB conversion at 512×512."
        )
    else:
        summary = "This is a partial run, so no comparative conclusion is reported."

    limitations = sorted(
        {
            limitation
            for row in rows
            if row["kind"] == "correctness"
            for limitation in row.get("limitations", [])
        }
    )
    limitation_summary = (
        "Known competitor limitation: " + "; ".join(limitations) + "."
        if limitations
        else "No known competitor-specific contract limitation was observed."
    )

    parity_path = RESULTS / "raw" / "io-parity.json"
    parity_payload = json.loads(parity_path.read_text()) if parity_path.exists() else None
    if parity_payload is not None and evidence_matches_code(parity_payload):
        parity = parity_payload["summary"]
        operations = ", ".join(
            f"{name} {value['passed']}/{value['checks']}"
            for name, value in parity["by_operation"].items()
        )
        parity_summary = (
            f"Native JPEG/PNG I/O passes {parity['passed']}/{parity['checks']} independent "
            f"interoperability checks: {operations} against Pillow, OpenCV, Torchvision, "
            "and a PNG format oracle."
        )
    else:
        parity_summary = "Current codec interoperability evidence is unavailable."

    catalog_path = RESULTS / "raw" / "catalog-audit.json"
    catalog_payload = json.loads(catalog_path.read_text()) if catalog_path.exists() else None
    catalog_audit_section = (
        render_catalog_audit_summary(catalog_payload)
        if catalog_payload is not None and evidence_matches_code(catalog_payload)
        else "## Complete catalog correctness audit\n\nCurrent catalog audit evidence is unavailable."
    )
    catalog_benchmark_path = RESULTS / "raw" / "catalog-benchmark.json"
    catalog_benchmark_payload = (
        json.loads(catalog_benchmark_path.read_text()) if catalog_benchmark_path.exists() else None
    )
    catalog_benchmark_section = (
        render_catalog_benchmark_summary(catalog_benchmark_payload)
        if catalog_benchmark_payload is not None
        and catalog_benchmark_payload.get("schema_version") == 2
        and evidence_matches_code(catalog_benchmark_payload)
        else "## Catalog performance evidence\n\nCurrent full catalog performance evidence is unavailable."
    )

    run_label = "quick smoke" if quick else "full"
    plot_path = "plots/quick/full-pipeline.png" if quick else "plots/full-pipeline.png"
    text = f"""# Benchmark evidence

This report is generated by `benchmarks/run_benchmark.py` from the latest
**{run_label}** run. It compares {backend_summary} on one machine. Results are
not universal.

Each timing is the median of **{repetitions}** independent worker
{"process" if repetitions == 1 else "processes"} per backend.
Raw per-worker observations and the counterbalanced backend execution order are
retained in the benchmark artifact. {variation_summary}

All latency and ratio claims below apply only to the reference hardware:
`{processor}`, `{platform}`, with each worker pinned to logical CPU
`{pinned_cpu}` and library thread counts controlled to one. Their semantic
scope is defined in Method and limits.

## Result

{summary}

{limitation_summary}

{parity_summary}

{chr(10).join(micro_lines)}

The main `Resize` row uses fixed-kernel bilinear for all four backends.

{antialias_table}

{pipeline_table}

| JPEG read + RGB decode (512×512) | Torchvision | Albumentations/OpenCV | AlbumentationsX/OpenCV | Variopinta |
|---|---:|---:|---:|---:|
| ms/image | {io_values["torchvision"]:.3f} | {io_values["albumentations"]:.3f} | {io_values["albumentationsx"]:.3f} | {io_values["rust"]:.3f} |

![In-memory pipeline throughput]({plot_path})

The compiler applies only semantics-preserving optimizations. An
earlier `RandomCrop+Resize` fusion was withdrawn because the resize filter could
read pixels outside the crop boundary; the pinned resizer's crop-box option has
the same mismatch. The current result includes safe direct output, input-copy
elision, buffer reuse, SIMD kernels, HWC normalization, and a direct
`Normalize+ToTorch` CHW terminal fusion; it does not credit Rust for an invalid
geometric fusion.

`RandomRotation` reuses the measured Affine kernel selection. `GaussianNoise`,
`Sharpen`, `Perspective`, and `GridDistortion` currently use portable scalar
kernels. Their catalog rows measure coverage and current cost; they are not
claims of transform-specific SIMD or fusion fast paths.

{catalog_audit_section}

{catalog_benchmark_section}

## Interpretation

The result supports a Python API backed by a compiled Rust pipeline on the
tested CPU. It does not show that Rust is inherently faster: competitors also
use native kernels, and algorithm, layout, copies, and materialization dominate
many rows. The main `Resize` row compares fixed-kernel bilinear policies. The
separate antialias row compares the scale-adaptive bilinear implementations
available in Torchvision and Variopinta; the OpenCV-backed competitors have
no matching bilinear policy in this benchmark.
The pixel-policy pipeline exposes the cost of materializing `CenterCrop` before
`Resize`; any improvement must preserve crop-boundary filtering semantics.

## Run the benchmark

Prepare the isolated environments with `just benchmark-setup`, then run:

```bash
just benchmark
```

Use `just benchmark-quick` for a smoke run. DataLoader is deliberately outside
the current scope. The separate layer experiment is
`just layer-experiments`.

Codec interoperability is checked separately with
`just io-parity`.

Focused JPEG/PNG operation timings use
`just io-performance`.

The complete Rust catalog correctness audit uses:

```bash
just catalog-audit
```

Catalog performance is measured separately with an adaptive per-row budget:

```bash
just catalog-benchmark
```

## Method and limits

- RGB uint8 at 224, 512, and 1024; native contiguous inputs are prepared before
  timing and outputs are materialized inside it.
- `Resize` is measured both as fixed-kernel bilinear across all backends and as
  scale-adaptive antialiased bilinear where a matching policy exists.
- Pipeline samples time blocks of 16 images and report median per-image latency,
  so probabilistic branches contribute to expected work instead of disappearing
  behind the most common individual path.
- Single-thread kernels, controlled library thread counts, fixed CPU affinity,
  and median aggregation across {repetitions} independent worker
  {"process" if repetitions == 1 else "processes"} per backend.
- Per-call observations, warmups, block sizes, worker identity, backend order,
  and worker-median spread are retained so every aggregate can be recalculated.
- Functional equivalence is required, not pixel identity across different valid
  interpolation, rounding, border, clipping, and RNG implementations.
- The correctness rows in the JSON report arbitrary sizes, dtype, layout,
  finiteness, ownership, and invalid-input checks.
- The catalog audit compares one reference/compiled pair per case and size at a
  fixed key, requires exact equality, and does not collect timings.
- The separate catalog benchmark uses adaptive call counts with a fixed target
  budget per row. It records executable-plan copies, buffers, passes, and
  scalar/SIMD fallback coverage. Size scaling can flag but not prove cache
  effects.
- Evidence is image-only, in-memory, on one CPU family. It excludes
  DataLoader, structured targets, GPU, and batch-native execution.
- The borrowed NumPy augmentation path holds the GIL. Native codec and file I/O
  release it before returning an owned NumPy array.

## Canonical artifacts

- `results/raw/benchmark-runs.json`
- `results/raw/io-parity.json`
- `results/raw/io-performance.json`
- `results/raw/catalog-audit.json`
- `results/raw/catalog-benchmark.json`
- [Layer-separation experiments](layers/layer-experiments.md) (diagnostic; not used for
  headline claims)
- `results/layers/raw/all-runs.json`

CSV exports, plots, and this report are derived locally from the raw observations.
"""
    evidence_paths(quick)["report"].write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible augmentation benchmark")
    parser.add_argument("--quick", action="store_true", help="run a short smoke benchmark")
    parser.add_argument(
        "--render-existing", action="store_true", help="regenerate CSV, plots, and report"
    )
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    args = parser.parse_args()

    repetitions = args.repetitions or (1 if args.quick else 3)
    if args.render_existing:
        payload = json.loads(evidence_paths(args.quick)["runs"].read_text())
        if payload.get("schema_version") != 2 or payload.get("quick") is not args.quick:
            raise SystemExit("existing evidence is missing the required raw schema")
        if not evidence_matches_code(payload):
            raise SystemExit(
                "existing evidence does not match the measured code; run just evidence-status"
            )
        rows = aggregate_runs(payload["rows"])
        repetitions = payload["repetitions"]
        run_metadata = payload["metadata"]
        backends = set(run_metadata)
    else:
        require_environments(args.backends)
        rows = run_workers(args.backends, args.quick, repetitions)
        payload = json.loads(evidence_paths(args.quick)["runs"].read_text())
        run_metadata = payload["metadata"]
        backends = set(args.backends)
    write_results(rows, args.quick)
    if backends == set(BACKENDS):
        generate_plots(rows, evidence_paths(args.quick)["plots"])
    render_benchmark_report(rows, args.quick, repetitions, run_metadata)
    print(f"Results: {evidence_paths(args.quick)['csv']}")


if __name__ == "__main__":
    main()
