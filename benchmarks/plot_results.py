from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKENDS = ("torchvision", "albumentations", "albumentationsx", "rust")
LABELS = {
    "torchvision": "Torchvision",
    "albumentations": "Albumentations",
    "albumentationsx": "AlbumentationsX",
    "rust": "Rust",
}
COLORS = {
    "torchvision": "#3b82f6",
    "albumentations": "#f59e0b",
    "albumentationsx": "#10b981",
    "rust": "#ef4444",
}


def _save(fig: Any, directory: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_plots(rows: list[dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    micro = [r for r in rows if r["kind"] == "micro" and r.get("valid", True)]
    transforms = list(dict.fromkeys(r["transform"] for r in micro))
    sizes = sorted({r["size"] for r in micro})
    fig, axes = plt.subplots(1, len(sizes), figsize=(18, 5), sharey=False)
    for ax, size in zip(axes, sizes, strict=True):
        width = 0.2
        for bi, backend in enumerate(BACKENDS):
            vals = [
                next(
                    r["images_per_sec"]
                    for r in micro
                    if r["size"] == size and r["backend"] == backend and r["transform"] == t
                )
                for t in transforms
            ]
            ax.bar(
                [i + (bi - 1.5) * width for i in range(len(transforms))],
                vals,
                width,
                label=LABELS[backend],
                color=COLORS[backend],
            )
        ax.set_title(f"{size}×{size}")
        ax.set_xticks(range(len(transforms)), transforms, rotation=45, ha="right")
        ax.set_ylabel("images/s")
    axes[0].legend(fontsize=8)
    _save(fig, directory, "throughput-by-transform.png")

    def speedup_plot(reference: str, contender: str, name: str, title: str) -> None:
        fig, axes = plt.subplots(1, len(sizes), figsize=(17, 4), sharey=True)
        for ax, size in zip(axes, sizes, strict=True):
            vals = []
            for t in transforms:
                base = next(
                    r["median_ms"]
                    for r in micro
                    if r["size"] == size and r["backend"] == reference and r["transform"] == t
                )
                other = next(
                    r["median_ms"]
                    for r in micro
                    if r["size"] == size and r["backend"] == contender and r["transform"] == t
                )
                vals.append(base / other)
            ax.bar(transforms, vals, color=COLORS[contender])
            ax.axhline(1.0, color="black", linewidth=1)
            ax.set_title(f"{size}×{size}")
            ax.tick_params(axis="x", rotation=45)
        axes[0].set_ylabel("speedup (×)")
        fig.suptitle(title)
        _save(fig, directory, name)

    speedup_plot("torchvision", "rust", "rust-speedup-vs-torchvision.png", "Rust vs Torchvision")
    speedup_plot(
        "albumentations",
        "albumentationsx",
        "albumentationsx-vs-albumentations.png",
        "AlbumentationsX vs Albumentations",
    )
    speedup_plot(
        "albumentationsx", "rust", "rust-vs-albumentationsx.png", "Rust vs AlbumentationsX"
    )

    pipeline = [r for r in rows if r["kind"] == "pipeline_memory" and r.get("valid", True)]
    pipelines = list(dict.fromkeys(r["pipeline"] for r in pipeline))
    fig, axes = plt.subplots(1, len(pipelines), figsize=(6 * len(pipelines), 5), sharey=True)
    if len(pipelines) == 1:
        axes = [axes]
    for ax, pipeline_name in zip(axes, pipelines, strict=True):
        width = 0.2
        for bi, backend in enumerate(BACKENDS):
            vals = [
                next(
                    r["images_per_sec"]
                    for r in pipeline
                    if r["backend"] == backend
                    and r["size"] == size
                    and r["pipeline"] == pipeline_name
                )
                for size in sizes
            ]
            ax.bar(
                [i + (bi - 1.5) * width for i in range(len(sizes))],
                vals,
                width,
                label=LABELS[backend],
                color=COLORS[backend],
            )
        ax.set_xticks(range(len(sizes)), [f"{s}×{s}" for s in sizes])
        ax.set_title(pipeline_name)
    axes[0].set_ylabel("images/s")
    axes[0].legend()
    _save(fig, directory, "full-pipeline.png")

    io_rows = [r for r in rows if r["kind"] == "io_jpeg" and r.get("valid", True)]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.2
    for bi, backend in enumerate(BACKENDS):
        vals = [
            next(
                r["images_per_sec"]
                for r in io_rows
                if r["backend"] == backend and r["size"] == size
            )
            for size in sizes
        ]
        ax.bar(
            [i + (bi - 1.5) * width for i in range(len(sizes))],
            vals,
            width,
            label=LABELS[backend],
            color=COLORS[backend],
        )
    ax.set_xticks(range(len(sizes)), [f"{size}×{size}" for size in sizes])
    ax.set_ylabel("images/s")
    ax.set_title("JPEG read and RGB decode")
    ax.legend()
    _save(fig, directory, "jpeg-decode.png")

    fig, axes = plt.subplots(1, len(pipelines), figsize=(6 * len(pipelines), 5), sharey=False)
    if len(pipelines) == 1:
        axes = [axes]
    for ax, pipeline_name in zip(axes, pipelines, strict=True):
        for backend in BACKENDS:
            subset = sorted(
                (r for r in pipeline if r["backend"] == backend and r["pipeline"] == pipeline_name),
                key=lambda r: r["size"],
            )
            ax.plot(
                [r["size"] for r in subset],
                [r["median_ms"] for r in subset],
                marker="o",
                label=LABELS[backend],
                color=COLORS[backend],
            )
        ax.set_xlabel("image side (px)")
        ax.set_title(pipeline_name)
    axes[0].set_ylabel("median ms/image")
    axes[0].legend()
    _save(fig, directory, "image-size-effect.png")
