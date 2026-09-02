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
    "rust": "Variopinta",
}
COLORS = {
    "torchvision": "#3b82f6",
    "albumentations": "#f59e0b",
    "albumentationsx": "#10b981",
    "rust": "#ef4444",
}


def generate_plots(rows: list[dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pipeline_rows = [
        row for row in rows if row["kind"] == "pipeline_memory" and row.get("valid", True)
    ]
    pipeline_names = list(dict.fromkeys(row["pipeline"] for row in pipeline_rows))
    sizes = sorted({row["size"] for row in pipeline_rows})
    figure, axes = plt.subplots(
        1, len(pipeline_names), figsize=(6 * len(pipeline_names), 5), sharey=True
    )
    if len(pipeline_names) == 1:
        axes = [axes]
    for axis, pipeline_name in zip(axes, pipeline_names, strict=True):
        width = 0.2
        for backend_index, backend in enumerate(BACKENDS):
            throughput = [
                next(
                    row["images_per_sec"]
                    for row in pipeline_rows
                    if row["backend"] == backend
                    and row["size"] == size
                    and row["pipeline"] == pipeline_name
                )
                for size in sizes
            ]
            axis.bar(
                [index + (backend_index - 1.5) * width for index in range(len(sizes))],
                throughput,
                width,
                label=LABELS[backend],
                color=COLORS[backend],
            )
        axis.set_xticks(range(len(sizes)), [f"{size}×{size}" for size in sizes])
        axis.set_title(pipeline_name)
    axes[0].set_ylabel("images/s")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(directory / "full-pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
