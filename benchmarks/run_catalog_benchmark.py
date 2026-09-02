from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    ROOT,
    SEED,
    control_cpu,
    evidence_provenance,
    make_images,
    metadata,
    summarize_observations,
    time_calls_adaptive,
    write_json,
)
from run_catalog_audit import (
    as_array,
    catalog_cases,
    validate_catalog_coverage,
    validate_output,
)


def bind(pipeline: Callable[..., Any]) -> Callable[[np.ndarray], Any]:
    return lambda image: pipeline(image, key=SEED)


def timing_policy(quick: bool, budget_ms: float) -> dict[str, int | float]:
    return {
        "budget_ms": budget_ms,
        "warmup_calls": 2 if quick else 3,
        "min_samples": 3 if quick else 7,
        "max_calls": 64 if quick else 512,
        "target_sample_ms": 2.0,
    }


def run_worker(quick: bool, budget_ms: float) -> dict[str, Any]:
    import variopinta as R

    cpu = control_cpu()
    policy = timing_policy(quick, budget_ms)
    rows: list[dict[str, Any]] = []
    for size in (224, 512, 1024):
        images = make_images(size)
        cases = catalog_cases(size)
        validate_catalog_coverage(cases)
        for transform, variant, transforms in cases:
            reference = R.Compose(transforms, seed=SEED)
            compiled = reference.compile()
            reference_output = reference(images[0], key=SEED)
            compiled_output = compiled(images[0], key=SEED)
            exact = bool(np.array_equal(as_array(reference_output), as_array(compiled_output)))
            for mode, pipeline in (("reference", reference), ("compiled", compiled)):
                timing, output = time_calls_adaptive(bind(pipeline), images, **policy)
                facts, output_valid = validate_output(transform, output)
                rows.append(
                    {
                        "transform": transform,
                        "variant": variant,
                        "mode": mode,
                        "size": size,
                        **timing,
                        "reference_exact": exact,
                        "validation": facts,
                        "valid": exact and output_valid,
                        "explanation": pipeline.explain(),
                    }
                )
    return {
        "metadata": metadata("rust-catalog-benchmark", cpu),
        "quick": quick,
        "policy": policy,
        "rows": rows,
    }


def aggregate(payloads: list[dict[str, Any]], quick: bool) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for repetition, payload in enumerate(payloads, start=1):
        for row in payload["rows"]:
            row["repetition"] = repetition
            key = (row["transform"], row["variant"], row["mode"], row["size"])
            grouped.setdefault(key, []).append(row)
    rows = []
    for members in grouped.values():
        row = {
            key: value
            for key, value in members[0].items()
            if key not in {"observations_ms", "repetition"}
        }
        medians = []
        p95s = []
        worker_observations = []
        for member in members:
            observations = member.get("observations_ms")
            if not isinstance(observations, list) or member.get("samples") != len(observations):
                raise ValueError("catalog timing row has incomplete observations")
            summary = summarize_observations(observations)
            medians.append(summary["median_ms"])
            p95s.append(summary["p95_ms"])
            worker_observations.append(
                {
                    "repetition": member["repetition"],
                    "block_size": member["block_size"],
                    "warmup_calls": member["warmup_calls"],
                    "iterations": member["iterations"],
                    "samples": member["samples"],
                    "observations_ms": observations,
                }
            )
        row.update(
            {
                "median_ms": statistics.median(medians),
                "p95_ms": statistics.median(p95s),
                "min_run_ms": min(medians),
                "max_run_ms": max(medians),
                "run_spread_percent": (
                    (max(medians) - min(medians)) / statistics.median(medians) * 100.0
                ),
                "worker_observations": worker_observations,
                "images_per_sec": 1000.0 / statistics.median(medians),
                "repetitions": len(members),
                "reference_exact": all(member["reference_exact"] for member in members),
                "valid": all(member["valid"] for member in members),
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: (row["size"], row["transform"], row["variant"], row["mode"]))
    return {
        "schema_version": 2,
        "metadata": [payload["metadata"] for payload in payloads],
        "quick": quick,
        "policy": payloads[0]["policy"],
        "input_definition": {
            "dtype": "uint8",
            "layout": "HWC RGB",
            "sizes": [224, 512, 1024],
            "images_per_size": 8,
            "seed": SEED,
            "materialization": "native contiguous input prepared before timing; output materialized inside timing",
        },
        "overrides": {
            "budget_ms": payloads[0]["policy"]["budget_ms"],
            "quick": quick,
            "repetitions": len(payloads),
        },
        "repetitions": len(payloads),
        "rows": rows,
    }


def render_catalog_benchmark_summary(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    current = {(row["transform"], row["variant"], row["mode"], row["size"]): row for row in rows}
    cases = sorted({(row["transform"], row["variant"]) for row in rows})
    policy = payload["policy"]
    lines = [
        "## Catalog performance evidence",
        "",
        f"The adaptive benchmark has {sum(row['valid'] for row in rows)}/{len(rows)} valid "
        f"aggregated timing rows from **{payload.get('repetitions', 1)}** independent "
        f"processes. Each row has a {policy['budget_ms']:g} ms target, "
        f"{policy['warmup_calls']} warmup calls, at least {policy['min_samples']} samples, "
        f"and at most {policy['max_calls']} measured calls.",
        "",
        "Median ms/image at 512×512. Buffers and kernel paths come from the executable "
        "compiled plan, not timing attribution.",
        "",
        "| Case | Reference | Compiled | Gain | Pixel passes | Native-entry copies | Buffers | Kernel paths |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    scaling = []
    for transform, variant in cases:
        reference = current[(transform, variant, "reference", 512)]
        compiled = current[(transform, variant, "compiled", 512)]
        explanation = compiled["explanation"]
        entry_copy = next(copy for copy in explanation["copies"] if copy["stage"] == "native-entry")
        buffers = ", ".join(
            buffer["name"]
            for buffer in explanation["buffers"]
            if buffer["name"] != "input" and buffer["condition"] != "not-required"
        )
        paths = ", ".join(explanation["fallbacks"])
        label = transform if variant == "default" else f"{transform} ({variant})"
        lines.append(
            f"| {label} | {reference['median_ms']:.3f} | {compiled['median_ms']:.3f} | "
            f"{reference['median_ms'] / compiled['median_ms']:.2f}× | "
            f"{explanation['pixel_passes']} | {entry_copy['count']} | {buffers or 'none'} | "
            f"{paths or 'none'} |"
        )
        small = current[(transform, variant, "compiled", 224)]["median_ms"]
        large = current[(transform, variant, "compiled", 1024)]["median_ms"]
        scaling.append((large / small / ((1024 / 224) ** 2), label))
    factors = [factor for factor, _ in scaling]
    spreads = [row["run_spread_percent"] for row in rows]
    nonlinear = ", ".join(
        f"{label} {factor:.2f}×" for factor, label in sorted(scaling, reverse=True)[:3]
    )
    lines.extend(
        [
            "",
            f"After normalizing by input pixels, the 1024²/224² latency factor ranges from "
            f"{min(factors):.2f}× to {max(factors):.2f}× (median "
            f"{statistics.median(factors):.2f}×). The largest nonlinear factors are "
            f"{nonlinear}; this flags cache, allocation, or other size-dependent work for "
            "profiling but does not assign causality.",
            "",
            f"Worker-median spread ranges from {min(spreads):.2f}% to {max(spreads):.2f}% "
            f"(median {statistics.median(spreads):.2f}%). Raw per-worker observations are "
            "embedded in the artifact.",
        ]
    )
    return "\n".join(lines)


def run_workers(quick: bool, repetitions: int, budget_ms: float) -> dict[str, Any]:
    payloads = []
    directory = "catalog-benchmark-quick-runs" if quick else "catalog-benchmark-runs"
    for repetition in range(1, repetitions + 1):
        output = ROOT / "results" / "raw" / directory / f"rust-{repetition}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--output",
            str(output),
            "--budget-ms",
            str(budget_ms),
        ]
        if quick:
            command.append("--quick")
        print(f"[catalog benchmark {repetition}/{repetitions}] rust", flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "benchmarks")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        payloads.append(json.loads(output.read_text()))
    return aggregate(payloads, quick)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark representative catalog policies")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--budget-ms", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.repetitions is not None and args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    budget_ms = args.budget_ms if args.budget_ms is not None else (20.0 if args.quick else 100.0)
    if budget_ms <= 0:
        parser.error("--budget-ms must be positive")
    repetitions = args.repetitions if args.repetitions is not None else (1 if args.quick else 3)
    output = args.output or ROOT / "results" / "raw" / (
        "catalog-benchmark-quick.json" if args.quick else "catalog-benchmark.json"
    )
    payload = (
        run_worker(args.quick, budget_ms)
        if args.worker
        else run_workers(args.quick, repetitions, budget_ms)
    )
    payload["provenance"] = evidence_provenance()
    write_json(output, payload)
    invalid = [row for row in payload["rows"] if not row["valid"]]
    if invalid:
        raise SystemExit(f"catalog benchmark failed: {len(invalid)} invalid rows")
    print(f"Catalog benchmark: {len(payload['rows'])} valid rows")
    print(f"Independent processes: {payload.get('repetitions', 1)}")
    print(f"Results: {output}")
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text())
        previous = {
            (row["transform"], row["variant"], row["mode"], row["size"]): row
            for row in baseline["rows"]
        }
        for row in payload["rows"]:
            key = (row["transform"], row["variant"], row["mode"], row["size"])
            if key in previous and row["mode"] == "compiled":
                gain = previous[key]["median_ms"] / row["median_ms"]
                if gain >= 1.10 or gain <= 0.90:
                    print(f"{row['transform']} {row['variant']} {row['size']}: {gain:.2f}x")


if __name__ == "__main__":
    main()
