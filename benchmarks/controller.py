from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.common import ROOT
from benchmarks.environments import (
    ENVIRONMENTS,
    python_for,
    rebuild_variopinta,
    require_environments,
)
from benchmarks.evidence import (
    CANONICAL_REPETITIONS,
    RUNS_ROOT,
    SCHEMA_VERSION,
    shard_path,
    status_for,
    write_json_atomic,
)
from benchmarks.fingerprints import case_fingerprint, compatibility_signature, source_provenance
from benchmarks.model import PlannedCase


def _run_directory(kind: str) -> Path:
    path = RUNS_ROOT / f"{kind}-{uuid.uuid4().hex[:12]}"
    path.mkdir(parents=True)
    return path


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_path = (str(ROOT), str(ROOT / "benchmarks"))
    environment["PYTHONPATH"] = os.pathsep.join((*python_path, environment.get("PYTHONPATH", "")))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    environment["NO_ALBUMENTATIONS_UPDATE"] = "1"
    return environment


def _item(planned: PlannedCase, route: Any) -> dict[str, Any]:
    return {
        "case_id": planned.case.id,
        "factory": planned.case.factory,
        "route": route.normalized(),
        "sizes": list(planned.sizes),
        "timing": None if planned.case.timing is None else planned.case.timing.normalized(),
    }


def _groups(
    plan: tuple[PlannedCase, ...],
    repetitions: int,
    validate_only: bool,
    current_environment: bool,
) -> dict[tuple[int, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for planned in plan:
        case_repetitions = repetitions if planned.case.timed and not validate_only else 1
        for repetition in range(1, case_repetitions + 1):
            for route in planned.routes:
                if (
                    validate_only
                    and planned.case.executor == "catalog"
                    and route.variant != "compiled"
                ):
                    continue
                environment = (
                    "current"
                    if current_environment or (validate_only and planned.case.executor == "catalog")
                    else route.environment
                )
                groups[(repetition, environment, planned.case.executor)].append(
                    _item(planned, route)
                )
    for (repetition, _, _), items in groups.items():
        items.sort(key=lambda value: (value["case_id"], value["route"]["id"]))
        if items:
            shift = (repetition - 1) % len(items)
            items[:] = items[shift:] + items[:shift]
    return groups


def _prepare_environments(groups: dict[tuple[int, str, str], list[dict[str, Any]]]) -> None:
    required = {environment for _, environment, _ in groups if environment != "current"}
    require_environments(required)
    for environment in sorted(required):
        if ENVIRONMENTS[environment].builds_variopinta:
            rebuild_variopinta(environment)


def execute_plan(
    plan: tuple[PlannedCase, ...],
    *,
    repetitions: int,
    quick: bool,
    validate_only: bool = False,
    current_environment: bool = False,
    kind: str = "run",
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    groups = _groups(plan, repetitions, validate_only, current_environment)
    _prepare_environments(groups)
    run_directory = _run_directory(kind)
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    execution_order = []
    environments = sorted({environment for _, environment, _ in groups})
    worker_index = 0
    for repetition in sorted({key[0] for key in groups}):
        order = list(environments)
        if order:
            shift = (repetition - 1) % len(order)
            order = order[shift:] + order[:shift]
        for environment_position, environment in enumerate(order, start=1):
            executors = sorted(
                executor
                for candidate_repetition, candidate_environment, executor in groups
                if candidate_repetition == repetition and candidate_environment == environment
            )
            for executor in executors:
                worker_index += 1
                items = groups[(repetition, environment, executor)]
                request_path = run_directory / f"request-{worker_index:03d}.json"
                output_path = run_directory / f"worker-{worker_index:03d}.json"
                request = {
                    "schema_version": 1,
                    "executor": executor,
                    "environment": environment,
                    "repetition": repetition,
                    "quick": quick,
                    "validate_only": validate_only,
                    "items": items,
                }
                write_json_atomic(request_path, request)
                executable = (
                    sys.executable if environment == "current" else str(python_for(environment))
                )
                subprocess.run(
                    [
                        executable,
                        str(ROOT / "benchmarks" / "worker.py"),
                        "--request",
                        str(request_path),
                        "--output",
                        str(output_path),
                    ],
                    cwd=ROOT,
                    env=_worker_environment(),
                    check=True,
                )
                payload = json.loads(output_path.read_text())
                if payload.get("schema_version") != 1 or payload.get("executor") != executor:
                    raise RuntimeError(f"invalid response from benchmark worker {worker_index}")
                worker_name = f"{environment}-{executor}-{repetition}"
                for row in payload.get("rows", []):
                    row["worker"] = worker_name
                    row["environment"] = environment
                    row["environment_position"] = environment_position
                    rows.append(row)
                metadata[environment] = payload["metadata"]
                execution_order.append(
                    {
                        "repetition": repetition,
                        "environment_position": environment_position,
                        "environment": environment,
                        "executor": executor,
                        "worker": worker_name,
                        "cases": [item["case_id"] for item in items],
                    }
                )
    payload = {
        "schema_version": 1,
        "quick": quick,
        "validate_only": validate_only,
        "repetitions": repetitions,
        "plan": [planned.normalized() for planned in plan],
        "metadata": metadata,
        "execution_order": execution_order,
        "rows": rows,
        "run_directory": str(run_directory),
    }
    write_json_atomic(run_directory / "run.json", payload)
    return payload


def _expected_identities(
    planned: PlannedCase, repetitions: int, validate_only: bool = False
) -> set[tuple[str, int | None, int]]:
    case_repetitions = repetitions if planned.case.timed and not validate_only else 1
    sizes: tuple[int | None, ...] = planned.sizes if planned.sizes else (None,)
    return {
        (route.id, size, repetition)
        for route in planned.routes
        for size in sizes
        for repetition in range(1, case_repetitions + 1)
    }


def validate_complete(planned: PlannedCase, rows: list[dict[str, Any]], repetitions: int) -> None:
    expected = _expected_identities(planned, repetitions)
    actual = {(row.get("route_id"), row.get("size"), row.get("repetition")) for row in rows}
    if actual != expected or len(rows) != len(expected):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"incomplete evidence for {planned.case.id}: missing={missing}, extra={extra}"
        )
    if not all(row.get("valid") is True for row in rows):
        raise RuntimeError(f"validation failed for {planned.case.id}")
    if planned.case.timed:
        for row in rows:
            observations = row.get("observations_ms")
            if not isinstance(observations, list) or row.get("samples") != len(observations):
                raise RuntimeError(f"incomplete timing observations for {planned.case.id}")


def write_evidence(
    plan: tuple[PlannedCase, ...], payload: dict[str, Any], fingerprints: dict[str, Any]
) -> list[Path]:
    repetitions = int(payload["repetitions"])
    provenance = source_provenance()
    paths = []
    for planned in plan:
        case_rows = [row for row in payload["rows"] if row["case_id"] == planned.case.id]
        validate_complete(planned, case_rows, repetitions)
        relevant_environments = {
            route.environment: payload["metadata"][route.environment]
            for route in planned.case.routes
        }
        shard = {
            "schema_version": SCHEMA_VERSION,
            "case_id": planned.case.id,
            "case": planned.case.normalized(),
            "fingerprint": fingerprints[planned.case.id],
            "provenance": provenance,
            "compatibility": compatibility_signature(relevant_environments),
            "execution_order": [
                entry for entry in payload["execution_order"] if planned.case.id in entry["cases"]
            ],
            "metadata": relevant_environments,
            "rows": case_rows,
        }
        path = shard_path(planned.case)
        write_json_atomic(path, shard)
        paths.append(path)
    return paths


def collect_evidence(plan: tuple[PlannedCase, ...]) -> list[Path]:
    fingerprints = {planned.case.id: case_fingerprint(planned.case) for planned in plan}
    payload = execute_plan(plan, repetitions=CANONICAL_REPETITIONS, quick=False, kind="evidence")
    changed = [
        planned.case.id
        for planned in plan
        if case_fingerprint(planned.case)["digest"] != fingerprints[planned.case.id]["digest"]
    ]
    if changed:
        raise RuntimeError(f"measured sources changed during execution: {', '.join(changed)}")
    return write_evidence(plan, payload, fingerprints)


def stale_plan(plan: tuple[PlannedCase, ...]) -> tuple[PlannedCase, ...]:
    return tuple(
        planned
        for planned in plan
        if status_for(planned.case).state in {"missing", "invalid", "stale"}
    )
