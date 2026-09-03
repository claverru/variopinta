from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from benchmarks.common import summarize_observations
from benchmarks.evidence import RUNS_ROOT, read_shard, status_for
from benchmarks.model import PlannedCase


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["route_id"], row.get("size")), []).append(row)
    output = []
    for _, members in sorted(grouped.items()):
        value = {
            key: members[0].get(key)
            for key in ("case_id", "route_id", "participant", "variant", "role", "size")
        }
        observations = [
            summarize_observations(member["observations_ms"])["median_ms"]
            for member in members
            if isinstance(member.get("observations_ms"), list)
        ]
        if observations:
            value["median_ms"] = statistics.median(observations)
            value["min_run_ms"] = min(observations)
            value["max_run_ms"] = max(observations)
            value["repetitions"] = len(observations)
        value["valid"] = all(member.get("valid") is True for member in members)
        output.append(value)
    return output


def _render_plots(
    summaries: list[dict[str, Any]], case_by_id: dict[str, Any], output: Path
) -> None:
    import matplotlib.pyplot as plt

    for case_id, case in case_by_id.items():
        rows = [row for row in summaries if row["case_id"] == case_id and "median_ms" in row]
        if not rows:
            continue
        sizes = sorted({row["size"] for row in rows if row["size"] is not None})
        size = 512 if 512 in sizes else sizes[0]
        selected = [row for row in rows if row["size"] == size]
        figure, axis = plt.subplots(figsize=(max(6, len(selected) * 1.4), 4))
        labels = [row["route_id"] for row in selected]
        values = [row["median_ms"] for row in selected]
        axis.bar(labels, values)
        axis.set_title(f"{case.label} ({size}×{size})")
        axis.set_ylabel("Median ms / operation")
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        path = output.joinpath(*case_id.split(".")).with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=140)
        plt.close(figure)


def _compatible(shards: list[dict[str, Any]]) -> bool:
    machine = None
    environments: dict[str, dict[str, Any]] = {}
    for shard in shards:
        compatibility = shard["compatibility"]["environments"]
        for name, values in compatibility.items():
            candidate_machine = tuple(
                values.get(key) for key in ("platform", "architecture", "processor", "cpu_count")
            )
            if machine is None:
                machine = candidate_machine
            elif candidate_machine != machine:
                return False
            if name in environments and environments[name] != values:
                return False
            environments[name] = values
    return True


def render(plan: tuple[PlannedCase, ...], output: Path | None = None) -> Path:
    output = output or RUNS_ROOT / "rendered"
    output.mkdir(parents=True, exist_ok=True)
    statuses = {planned.case.id: status_for(planned.case) for planned in plan}
    available = [
        planned for planned in plan if statuses[planned.case.id].state in {"current", "dirty"}
    ]
    summaries = []
    shards = []
    for planned in available:
        shard = read_shard(planned.case)
        shards.append(shard)
        summaries.extend(_summaries(shard["rows"]))

    csv_path = output / "measurements.csv"
    fields = (
        "suite",
        "case_id",
        "label",
        "route_id",
        "participant",
        "variant",
        "role",
        "size",
        "median_ms",
        "min_run_ms",
        "max_run_ms",
        "repetitions",
        "valid",
    )
    case_by_id = {planned.case.id: planned.case for planned in plan}
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            case = case_by_id[row["case_id"]]
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "suite": case.suite,
                    "label": case.label,
                }
            )

    _render_plots(summaries, case_by_id, output / "plots")

    complete = all(status.state == "current" for status in statuses.values()) and _compatible(
        shards
    )
    lines = [
        "# Benchmark evidence",
        "",
        (
            "This view uses a complete, current canonical selection."
            if complete
            else "This is a partial or non-publishable diagnostic view; it makes no headline claims."
        ),
        "",
        "## Evidence status",
        "",
        "| Case | Status |",
        "|---|---|",
    ]
    for planned in plan:
        status = statuses[planned.case.id]
        lines.append(f"| `{planned.case.id}` | {status.state} |")

    for suite in ("transforms", "pipelines", "catalog", "io", "contracts"):
        suite_cases = [planned.case for planned in available if planned.case.suite == suite]
        if not suite_cases:
            continue
        lines.extend(["", f"## {suite.title()}", ""])
        for case in suite_cases:
            rows = [row for row in summaries if row["case_id"] == case.id]
            if not rows or "median_ms" not in rows[0]:
                lines.append(f"- `{case.id}`: validation passed.")
                continue
            sizes = sorted({row["size"] for row in rows if row["size"] is not None})
            size = 512 if 512 in sizes else sizes[0]
            selected = [row for row in rows if row["size"] == size]
            measurements = ", ".join(
                f"{row['route_id']} {row['median_ms']:.3f} ms" for row in selected
            )
            suffix = ""
            if complete and case.comparability in {"exact", "policy"}:
                rust = next(
                    (
                        row
                        for row in selected
                        if row["participant"] == "variopinta" and row["variant"] == "compiled"
                    ),
                    None,
                )
                rivals = [
                    row
                    for row in selected
                    if row["role"] == "public" and row["participant"] != "variopinta"
                ]
                if rust is not None and rivals:
                    fastest = min(rivals, key=lambda row: row["median_ms"])
                    suffix = f"; Variopinta/fastest public rival {rust['median_ms'] / fastest['median_ms']:.2f}×"
            lines.append(f"- `{case.id}` ({size}): {measurements}{suffix}.")

    (output / "benchmark.md").write_text("\n".join(lines) + "\n")
    return output
