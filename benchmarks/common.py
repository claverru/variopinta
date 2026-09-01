from __future__ import annotations

import json
import math
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SIZES = (224, 512, 1024)
PIPELINES = ("classic", "extended", "pixel_policy")
TRANSFORMS = (
    "Resize",
    "RandomCrop",
    "CenterCrop",
    "HorizontalFlip",
    "VerticalFlip",
    "ColorJitter",
    "Affine",
    "GaussianBlur",
    "Grayscale",
    "Invert",
    "Solarize",
    "Posterize",
    "Normalize",
)
SEED = 137
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def control_cpu(pin_process: bool = True) -> dict[str, Any]:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    affinity_before = None
    affinity_after = None
    if hasattr(os, "sched_getaffinity"):
        affinity_before = sorted(os.sched_getaffinity(0))
        if pin_process and affinity_before:
            os.sched_setaffinity(0, {affinity_before[0]})
        affinity_after = sorted(os.sched_getaffinity(0))
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        import cv2

        cv2.setNumThreads(0)
        cv_threads = cv2.getNumThreads()
    except ImportError:
        cv_threads = None
    return {
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
        "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cv2_num_threads": cv_threads,
        "cpu_affinity_before": affinity_before,
        "cpu_affinity_after": affinity_after,
    }


def make_images(size: int, count: int = 8) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED + size)
    yy, xx = np.indices((size, size), dtype=np.int64)
    x_grad = (xx * 255 // max(1, size - 1)).astype(np.uint8)
    y_grad = (yy * 255 // max(1, size - 1)).astype(np.uint8)
    checker = (((xx // max(1, size // 16)) + (yy // max(1, size // 16))) & 1).astype(np.uint8) * 255
    waves = np.clip(
        127.5
        + 70 * np.sin(xx * (2 * np.pi / max(3, size // 5)))
        + 45 * np.cos(yy * (2 * np.pi / max(3, size // 7))),
        0,
        255,
    ).astype(np.uint8)
    blocks = np.zeros((size, size, 3), dtype=np.uint8)
    blocks[size // 8 : size * 5 // 8, size // 6 : size * 2 // 3] = (230, 40, 90)
    blocks[size // 3 : size * 7 // 8, size // 2 : size * 9 // 10] = (25, 210, 150)
    patterns = [
        rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        np.stack(
            (x_grad, y_grad, ((x_grad.astype(np.uint16) + y_grad) // 2).astype(np.uint8)), axis=2
        ),
        np.stack((checker, np.roll(checker, max(1, size // 32), axis=1), 255 - checker), axis=2),
        np.full((size, size, 3), (32, 128, 224), dtype=np.uint8),
        np.stack(
            (
                waves,
                np.roll(waves, max(1, size // 11), axis=0),
                np.roll(waves, max(1, size // 13), axis=1),
            ),
            axis=2,
        ),
        blocks,
        np.clip(rng.normal(110, 25, (size, size, 3)), 0, 255).astype(np.uint8),
        rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
    ]
    return [np.ascontiguousarray(patterns[i % len(patterns)]) for i in range(count)]


def iterations_for(size: int, quick: bool) -> tuple[int, int]:
    if quick:
        return 2, {224: 8, 512: 5, 1024: 3}[size]
    return 10, {224: 100, 512: 50, 1024: 20}[size]


def time_calls(
    fn: Callable[[Any], Any],
    inputs: list[Any],
    warmup: int,
    iterations: int,
    *,
    block_size: int = 1,
) -> tuple[dict[str, Any], Any]:
    output = None
    for i in range(warmup * block_size):
        output = fn(inputs[i % len(inputs)])
    samples = []
    for i in range(iterations):
        start = time.perf_counter_ns()
        for offset in range(block_size):
            output = fn(inputs[(i * block_size + offset) % len(inputs)])
        samples.append((time.perf_counter_ns() - start) / 1_000_000 / block_size)
    summary = summarize_observations(samples)
    return {
        **summary,
        "iterations": iterations * block_size,
        "samples": len(samples),
        "observations_ms": samples,
        "block_size": block_size,
        "warmup_calls": warmup * block_size,
    }, output


def time_calls_adaptive(
    fn: Callable[[Any], Any],
    inputs: list[Any],
    *,
    budget_ms: float,
    warmup_calls: int,
    min_samples: int,
    max_calls: int,
    target_sample_ms: float = 2.0,
) -> tuple[dict[str, Any], Any]:
    if budget_ms <= 0 or warmup_calls < 0 or min_samples <= 0 or max_calls < min_samples:
        raise ValueError("invalid adaptive timing policy")
    output = None
    for index in range(warmup_calls):
        output = fn(inputs[index % len(inputs)])

    probe_start = time.perf_counter_ns()
    output = fn(inputs[0])
    probe_ms = max((time.perf_counter_ns() - probe_start) / 1_000_000, 1e-6)
    max_block_size = max(1, max_calls // min_samples)
    block_size = min(max_block_size, max(1, math.ceil(target_sample_ms / probe_ms)))

    samples: list[float] = []
    calls = 0
    timing_start = time.perf_counter_ns()
    while calls + block_size <= max_calls:
        elapsed_ms = (time.perf_counter_ns() - timing_start) / 1_000_000
        if len(samples) >= min_samples and elapsed_ms >= budget_ms:
            break
        sample_start = time.perf_counter_ns()
        for offset in range(block_size):
            output = fn(inputs[(calls + offset) % len(inputs)])
        samples.append((time.perf_counter_ns() - sample_start) / 1_000_000 / block_size)
        calls += block_size

    summary = summarize_observations(samples)
    return {
        **summary,
        "iterations": calls,
        "samples": len(samples),
        "observations_ms": samples,
        "block_size": block_size,
        "budget_ms": budget_ms,
        "warmup_calls": warmup_calls,
    }, output


def summarize_observations(samples: list[float]) -> dict[str, float]:
    if not samples or any(not math.isfinite(sample) or sample <= 0.0 for sample in samples):
        raise ValueError("timing observations must be finite and positive")
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
    median = statistics.median(samples)
    return {
        "median_ms": median,
        "p95_ms": p95,
        "images_per_sec": 1000.0 / median,
    }


def aggregate_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    identity = ("kind", "backend", "transform", "pipeline", "policy", "size")
    for row in rows:
        key = tuple(row.get(field) for field in identity)
        grouped.setdefault(key, []).append(row)

    aggregated = []
    for members in grouped.values():
        result = {
            key: value
            for key, value in members[0].items()
            if key not in {"repetition", "worker", "backend_position", "observations_ms"}
        }
        result["repetitions"] = len(members)
        result["valid"] = all(member.get("valid", True) for member in members)
        if "median_ms" in result:
            worker_observations = []
            medians = []
            p95s = []
            for member in members:
                observations = member.get("observations_ms")
                if not isinstance(observations, list):
                    raise ValueError("timed benchmark row is missing observations_ms")
                expected_samples = member.get("samples")
                if expected_samples is not None and expected_samples != len(observations):
                    raise ValueError("timing sample count does not match observations_ms")
                summary = summarize_observations(observations)
                medians.append(summary["median_ms"])
                p95s.append(summary["p95_ms"])
                worker_observations.append(
                    {
                        "repetition": member["repetition"],
                        "worker": member["worker"],
                        "backend_position": member["backend_position"],
                        "block_size": member.get("block_size", 1),
                        "warmup_calls": member.get("warmup_calls", 0),
                        "iterations": member.get("iterations", len(observations)),
                        "samples": len(observations),
                        "observations_ms": observations,
                    }
                )
            result["median_ms"] = statistics.median(medians)
            result["p95_ms"] = statistics.median(p95s)
            result["min_run_ms"] = min(medians)
            result["max_run_ms"] = max(medians)
            result["run_spread_percent"] = (
                (max(medians) - min(medians)) / result["median_ms"] * 100.0
            )
            result["worker_observations"] = worker_observations
            if "images_per_sec" in result:
                result["images_per_sec"] = 1000.0 / result["median_ms"]
        aggregated.append(result)
    return aggregated


def output_facts(value: Any) -> dict[str, Any]:
    import torch

    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        kind = "torch.Tensor"
    else:
        arr = np.asarray(value)
        kind = "numpy.ndarray"
    finite = bool(np.isfinite(arr).all()) if np.issubdtype(arr.dtype, np.floating) else True
    return {
        "container": kind,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "finite": finite,
        "c_contiguous": bool(arr.flags.c_contiguous),
    }


def metadata(backend: str, cpu: dict[str, Any]) -> dict[str, Any]:
    import importlib.metadata as md

    packages = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "albumentations",
        "albumentationsx",
        "albucore",
        "opencv-python-headless",
        "numkong",
        "variopinta",
    ):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    cpu_model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "backend": backend,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": cpu_model,
        "cpu_count": os.cpu_count(),
        "torch": packages.get("torch"),
        "torchvision": packages.get("torchvision"),
        "packages": packages,
        "thread_control": cpu,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
