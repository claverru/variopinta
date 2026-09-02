from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    ROOT,
    TRANSFORMS,
    evidence_matches_code,
    evidence_provenance,
    summarize_observations,
    write_json,
)
from environments import python_for, rebuild_variopinta, require_environments
from run_benchmark import BACKENDS

OUTPUT = ROOT / "results" / "layers"


def execute(
    backends: list[str], repetitions: int, quick: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_environments(backends)
    if "rust" in backends:
        rebuild_variopinta()
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    execution_order = []
    for repetition in range(1, repetitions + 1):
        order = list(backends)
        shift = (repetition - 1) % len(order)
        order = order[shift:] + order[:shift]
        for position, backend in enumerate(order, start=1):
            directory = "quick-runs" if quick else "runs"
            output = OUTPUT / "raw" / directory / f"{backend}-{repetition}.json"
            command = [
                str(python_for(backend)),
                str(ROOT / "benchmarks" / "layer_worker.py"),
                "--backend",
                backend,
                "--repetition",
                str(repetition),
                "--output",
                str(output),
            ]
            if quick:
                command.append("--quick")
            print(f"[layers {repetition}/{repetitions}] {backend}", flush=True)
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "benchmarks")}
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            payload = json.loads(output.read_text())
            for row in payload["rows"]:
                row["worker"] = f"{backend}-{repetition}"
                row["backend_position"] = position
            rows.extend(payload["rows"])
            metadata[backend] = payload["metadata"]
            execution_order.append(
                {"repetition": repetition, "position": position, "backend": backend}
            )
    return rows, {"workers": metadata, "execution_order": execution_order}


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("kind", "backend", "variant", "transform", "case", "size")
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for group, members in grouped.items():
        result = {key: value for key, value in zip(keys, group, strict=True) if value is not None}
        medians = []
        p95s = []
        worker_observations = []
        for member in members:
            observations = member.get("observations_ms")
            if not isinstance(observations, list) or member.get("samples") != len(observations):
                raise ValueError("layer timing row has incomplete observations")
            summary = summarize_observations(observations)
            medians.append(summary["median_ms"])
            p95s.append(summary["p95_ms"])
            worker_observations.append(
                {
                    "repetition": member["repetition"],
                    "worker": member["worker"],
                    "backend_position": member["backend_position"],
                    "block_size": member["block_size"],
                    "warmup_calls": member["warmup_calls"],
                    "iterations": member["iterations"],
                    "samples": member["samples"],
                    "observations_ms": observations,
                }
            )
        result.update(
            {
                "median_ms": statistics.median(medians),
                "min_run_ms": min(medians),
                "max_run_ms": max(medians),
                "p95_ms": statistics.median(p95s),
                "images_per_sec": 1000.0 / statistics.median(medians),
                "repetitions": len(members),
                "run_spread_percent": (
                    (max(medians) - min(medians)) / statistics.median(medians) * 100.0
                ),
                "worker_observations": worker_observations,
                "valid": all(member["valid"] for member in members),
            }
        )
        explanations = [member.get("explanation") for member in members]
        if explanations[0] is not None and all(item == explanations[0] for item in explanations):
            result["explanation"] = explanations[0]
        output.append(result)
    return sorted(output, key=lambda row: tuple(str(row.get(key, "")) for key in keys))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if key != "worker_observations"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            flattened.pop("worker_observations", None)
            if "explanation" in flattened:
                flattened["explanation"] = json.dumps(
                    flattened["explanation"], sort_keys=True, separators=(",", ":")
                )
            writer.writerow(flattened)


def value(rows: list[dict[str, Any]], **filters: Any) -> float:
    return matching_row(rows, **filters)["median_ms"]


def matching_row(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any]:
    return next(
        row for row in rows if all(row.get(key) == expected for key, expected in filters.items())
    )


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(item) for item in values) / len(values))


def pixel_cost_model(rows: list[dict[str, Any]], transform: str) -> tuple[float, float, float]:
    points = [
        (row["size"] ** 2, row["median_ms"] * 1_000_000)
        for row in rows
        if row["kind"] == "layer1_transform"
        and row["backend"] == "rust"
        and row["variant"] == "compiled"
        and row["transform"] == transform
    ]
    mean_x = statistics.mean(point[0] for point in points)
    mean_y = statistics.mean(point[1] for point in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total = sum((y - mean_y) ** 2 for _, y in points)
    r_squared = 1.0 - residual / total if total else 1.0
    return intercept / 1_000, slope, r_squared


def report(
    rows: list[dict[str, Any]], metadata: dict[str, Any], repetitions: int, quick: bool
) -> str:
    transforms = TRANSFORMS
    layer1 = [row for row in rows if row["kind"] == "layer1_transform" and row["size"] == 512]
    process_label = "process" if repetitions == 1 else "processes"
    reference_run = next(iter(metadata.values()))
    processor = reference_run["processor"]
    platform = reference_run["platform"]
    affinity = reference_run["thread_control"]["cpu_affinity_after"]
    pinned_cpu = affinity[0] if len(affinity) == 1 else affinity
    lines = [
        "# Layer-separation experiments",
        "",
        f"{'Quick' if quick else 'Full'} run; {repetitions} independent {process_label} per backend.",
        f"Reference hardware: `{processor}`, `{platform}`; each worker is pinned to logical CPU",
        f"`{pinned_cpu}` with library thread counts controlled to one. All timing and ratio claims apply only",
        "to this environment. Semantic variants and causal limits are disclosed below.",
        "",
        "## 1. Rust backend from Python without global optimization",
        "",
        "Median of medians, ms/image, 512×512 input. Rust uses `staged-fresh`; competitors use isolated public transforms.",
        "",
        "| Transform | Torchvision | Albumentations | AlbumentationsX | Fastest rival | Rust staged | Rust vs fastest | Rust vs AX |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ax_ratios = []
    best_ratios = []
    for transform in transforms:
        values = {
            backend: value(
                layer1,
                backend=backend,
                variant="staged-fresh" if backend == "rust" else "stock",
                transform=transform,
            )
            for backend in BACKENDS
        }
        ax_ratio = values["albumentationsx"] / values["rust"]
        best_backend = min(
            ("torchvision", "albumentations", "albumentationsx"),
            key=lambda backend: values[backend],
        )
        best = values[best_backend]
        best_ratio = best / values["rust"]
        ax_ratios.append(ax_ratio)
        best_ratios.append(best_ratio)
        lines.append(
            f"| {transform} | {values['torchvision']:.3f} | {values['albumentations']:.3f} | "
            f"{values['albumentationsx']:.3f} | {best_backend} {best:.3f} | {values['rust']:.3f} | "
            f"{best_ratio:.2f}× | {ax_ratio:.2f}× |"
        )
    ax_base_gain = geometric_mean(ax_ratios)
    best_base_gain = geometric_mean(best_ratios)
    ax_wins = sum(ratio > 1.0 for ratio in ax_ratios)
    best_wins = sum(ratio > 1.0 for ratio in best_ratios)

    size_summaries = []
    for size in (224, 512, 1024):
        subset = [row for row in rows if row["kind"] == "layer1_transform" and row["size"] == size]
        ratios = []
        for transform in transforms:
            rust = value(subset, backend="rust", variant="staged-fresh", transform=transform)
            best = min(
                value(subset, backend=backend, transform=transform)
                for backend in ("torchvision", "albumentations", "albumentationsx")
            )
            ratios.append(best / rust)
        size_summaries.append((size, geometric_mean(ratios), sum(ratio > 1.0 for ratio in ratios)))
    lines.extend(
        [
            "",
            f"Geometric AX/Rust staged ratio: **{ax_base_gain:.2f}×**; Rust wins {ax_wins}/{len(transforms)} transforms against AX.",
            f"Against the fastest rival in each row: **{best_base_gain:.2f}×**; Rust wins {best_wins}/{len(transforms)}.",
            "Scaling: "
            + ", ".join(
                f"{size}² {ratio:.2f}× and {wins}/{len(transforms)} wins"
                for size, ratio, wins in size_summaries
            )
            + ".",
            "Staged ColorJitter uses its explicit multi-pass oracle; compiled ColorJitter uses a specialized unit kernel.",
            "Direct source-to-destination output is enabled for leading VerticalFlip, Invert, Solarize, and Posterize. A HorizontalFlip candidate regressed at 1024² and was rejected; it retains the established in-place AVX2 path.",
            "Resize uses fixed-kernel bilinear across all backends; the separate general benchmark measures antialiased Torchvision and Variopinta policies.",
            "",
            "### Rust attribution: staged versus compiled unit path",
            "",
            "| Transform (512) | Staged | Compiled | Gain | Fusions | Unit specializations | Pipeline optimizations | Native-entry copies |",
            "|---|---:|---:|---:|---|---|---|---:|",
        ]
    )
    for transform in transforms:
        staged = value(
            layer1,
            backend="rust",
            variant="staged-fresh",
            transform=transform,
        )
        compiled_row = matching_row(
            layer1,
            backend="rust",
            variant="compiled",
            transform=transform,
        )
        explanation = compiled_row["explanation"]
        native_entry = next(
            copy for copy in explanation["copies"] if copy["stage"] == "native-entry"
        )
        lines.append(
            f"| {transform} | {staged:.3f} | {compiled_row['median_ms']:.3f} | "
            f"{staged / compiled_row['median_ms']:.2f}× | "
            f"{', '.join(explanation['fusions']) or 'none'} | "
            f"{', '.join(explanation['unit_specializations']) or 'none'} | "
            f"{', '.join(explanation['optimizations']) or 'none'} | "
            f"{native_entry['count']} |"
        )
    lines.extend(
        [
            "",
            "### Rust compiled size model",
            "",
            "Least-squares model over 224², 512², and 1024² inputs: `time = fixed + pixels × slope`. A negative intercept or a weak fit signals cache, allocation, or other nonlinear effects and must not be interpreted as negative overhead.",
            "",
            "| Transform | Fixed intercept (µs) | Slope (ns/pixel) | R² |",
            "|---|---:|---:|---:|",
        ]
    )
    for transform in (
        "HorizontalFlip",
        "VerticalFlip",
        "Grayscale",
        "Invert",
        "Solarize",
        "Posterize",
    ):
        fixed_us, ns_per_pixel, r_squared = pixel_cost_model(rows, transform)
        lines.append(f"| {transform} | {fixed_us:.3f} | {ns_per_pixel:.3f} | {r_squared:.4f} |")
    lines.extend(
        [
            "",
            "## 2. Memory and pipeline optimization",
            "",
            "| Case (512) | Rust fresh | Rust reuse | Rust compiled | Reuse gain | Compiled-path gain | Total |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    cases = ("crop_resize", "color_jitter", "normalize", "full")
    rust_gains: dict[str, tuple[float, float, float]] = {}
    for case in cases:
        fresh = value(
            rows, kind="layer2_case", backend="rust", variant="staged-fresh", case=case, size=512
        )
        reuse = value(
            rows, kind="layer2_case", backend="rust", variant="staged-reuse", case=case, size=512
        )
        compiled = value(
            rows, kind="layer2_case", backend="rust", variant="compiled", case=case, size=512
        )
        rust_gains[case] = (fresh / reuse, reuse / compiled, fresh / compiled)
        lines.append(
            f"| {case} | {fresh:.3f} | {reuse:.3f} | {compiled:.3f} | "
            f"{fresh / reuse:.2f}× | {reuse / compiled:.2f}× | {fresh / compiled:.2f}× |"
        )

    lines.extend(
        [
            "",
            "### Internal work model",
            "",
            "| Case | staged-fresh | staged-reuse | compiled |",
            "|---|---|---|---|",
            "| crop_resize | input copy + 2 transform outputs | same work with pooled buffers | input copy elided; both transforms still execute |",
            "| color_jitter | unit oracle with intermediate clipping | same work with pooled buffers | specialized unit kernel; no pipeline fusion |",
            "| normalize | input copy + Normalize output | same work with reusable workspace | Normalize reads borrowed input directly |",
            "| full | unit transforms with fresh buffers | same transforms with pooled buffers | copy elision plus unit-kernel specialization |",
            "",
            "Each Rust row stores its native `explain()` payload. Fusions are empty in these measured cases; copies and optimizations come from the executable entry plan. Timings are observations, while the work descriptions above are structural facts.",
            "",
            "### Machine-readable attribution",
            "",
            "| Case | Fusions | Unit specializations | Pipeline optimizations | Native-entry copies | Pixel passes |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for case in cases:
        explanation = matching_row(
            rows,
            kind="layer2_case",
            backend="rust",
            variant="compiled",
            case=case,
            size=512,
        )["explanation"]
        native_entry = next(
            copy for copy in explanation["copies"] if copy["stage"] == "native-entry"
        )
        lines.append(
            f"| {case} | {', '.join(explanation['fusions']) or 'none'} | "
            f"{', '.join(explanation['unit_specializations']) or 'none'} | "
            f"{', '.join(explanation['optimizations']) or 'none'} | "
            f"{native_entry['count']} | {explanation['pixel_passes']} |"
        )
    lines.extend(
        [
            "",
            "### Competitors: standard path versus best official path",
            "",
            "| Case (512) | Backend | Stock | Best official | Official gain | Rust compiled | Final gap |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in ("crop_resize", "full"):
        rust = value(
            rows, kind="layer2_case", backend="rust", variant="compiled", case=case, size=512
        )
        for backend in ("torchvision", "albumentations", "albumentationsx"):
            stock = value(
                rows, kind="layer2_case", backend=backend, variant="stock", case=case, size=512
            )
            best = value(
                rows,
                kind="layer2_case",
                backend=backend,
                variant="best-official",
                case=case,
                size=512,
            )
            lines.append(
                f"| {case} | {backend} | {stock:.3f} | {best:.3f} | {stock / best:.2f}× | "
                f"{rust:.3f} | {best / rust:.2f}× |"
            )

    invalid = [row for row in rows if not row["valid"]]
    strongest = max(rust_gains.items(), key=lambda item: item[1][2])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Baseline AX/Rust staged ratio: **{ax_base_gain:.2f}×**.",
            f"- Baseline ratio against the fastest competitor per transform: **{best_base_gain:.2f}×**; values below 1 favor the competitor.",
            f"- Largest Rust optimization gain: **{strongest[0]}**, {strongest[1][2]:.2f}× fresh→compiled.",
            f"- Invalid rows: **{len(invalid)}**.",
            "- ColorJitter variants with intermediate clipping are treated as semantic variants, not exact equivalents.",
            "",
            "## 3. Architecture guardrails",
            "",
            "- Unit layout and Normalize kernels live under `rust/core/src/kernels` and retain scalar/SIMD oracles.",
            "- `rust/core/src/optimization.rs` owns entry-pattern selection and copy policy.",
            "- Pipeline fusion is a separate category; these measured cases use none. The terminal `Normalize+ToTorch` fusion is covered by the catalog audit instead.",
            "- Reference execution remains the semantic oracle for any future compiled pattern.",
            "",
            "## Causal limits",
            "",
            "- Layer 1 is an operational Python comparison without global fusion; it cannot causally attribute every cycle to the language. Rivals run C/C++/SIMD below their APIs, and kernel quality remains relevant.",
            "- `staged-fresh` crosses Python/Rust once, like compiled, but disables selected global optimizations. It isolates our compiler gain, not a hypothetical identical implementation in another language.",
            "- This report measures no cross-transform fusion. Copy elision and unit-kernel specialization are reported separately.",
            "- Staged ColorJitter clips between operations while compiled clips at the end. Both are valid but not pixel-equivalent, so the row does not establish strict mathematical equivalence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate backend, optimization, and maintenance effects"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    repetitions = args.repetitions or (1 if args.quick else 3)
    suffix = "-quick" if args.quick else ""
    runs_path = OUTPUT / "raw" / f"all-runs{suffix}.json"
    if args.render_existing:
        raw = json.loads(runs_path.read_text())
        if raw.get("schema_version") != 2 or raw.get("quick") is not args.quick:
            raise SystemExit("existing layer evidence is missing the required raw schema")
        if not evidence_matches_code(raw):
            raise SystemExit("existing layer evidence does not match the measured code")
        rows = raw["rows"]
        metadata = raw["metadata"]
        repetitions = raw["repetitions"]
    else:
        rows, metadata = execute(args.backends, repetitions, args.quick)
        write_json(
            runs_path,
            {
                "schema_version": 2,
                "quick": args.quick,
                "repetitions": repetitions,
                "overrides": {
                    "backends": args.backends,
                    "quick": args.quick,
                    "repetitions": repetitions,
                },
                "provenance": evidence_provenance(),
                "metadata": metadata,
                "rows": rows,
            },
        )
    aggregated = aggregate(rows)
    write_json(OUTPUT / "raw" / f"aggregated{suffix}.json", aggregated)
    write_csv(OUTPUT / "csv" / f"layer-results{suffix}.csv", aggregated)
    if set(args.backends) == set(BACKENDS):
        text = report(aggregated, metadata["workers"], repetitions, args.quick)
        report_path = (
            OUTPUT / "layer-experiments-quick.md" if args.quick else OUTPUT / "layer-experiments.md"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text)
        print(text)
    print(f"Results: {OUTPUT}")


if __name__ == "__main__":
    main()
