from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.common import ROOT
from benchmarks.fingerprints import (
    case_fingerprint,
    compatibility_signature,
    unclassified_measured_paths,
)
from benchmarks.model import CaseSpec

EVIDENCE_ROOT = ROOT / "benchmarks" / "evidence"
RUNS_ROOT = ROOT / "benchmarks" / ".runs"
SCHEMA_VERSION = 1
CANONICAL_REPETITIONS = 3


@dataclass(frozen=True, slots=True)
class EvidenceStatus:
    case_id: str
    state: str
    detail: str
    path: Path


def shard_path(case: CaseSpec, root: Path = EVIDENCE_ROOT) -> Path:
    parts = case.id.split(".")
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"invalid case identifier: {case.id}")
    return root.joinpath(*parts).with_suffix(".json")


def read_shard(case: CaseSpec, root: Path = EVIDENCE_ROOT) -> dict[str, Any]:
    value = json.loads(shard_path(case, root).read_text())
    if not isinstance(value, dict):
        raise ValueError("evidence shard must contain an object")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _rows_are_complete(case: CaseSpec, rows: list[dict[str, Any]]) -> bool:
    if not all(isinstance(row, dict) for row in rows):
        return False
    repetitions = CANONICAL_REPETITIONS if case.timed else 1
    sizes: tuple[int | None, ...] = case.sizes if case.sizes else (None,)
    expected = {
        (route.id, size, repetition)
        for route in case.routes
        for size in sizes
        for repetition in range(1, repetitions + 1)
    }
    actual = {(row.get("route_id"), row.get("size"), row.get("repetition")) for row in rows}
    if actual != expected or len(rows) != len(expected):
        return False
    for row in rows:
        if row.get("valid") is not True:
            return False
        if case.timed:
            observations = row.get("observations_ms")
            if not isinstance(observations, list) or row.get("samples") != len(observations):
                return False
    return True


def status_for(case: CaseSpec, root: Path = EVIDENCE_ROOT) -> EvidenceStatus:
    path = shard_path(case, root)
    unclassified = unclassified_measured_paths()
    if unclassified:
        names = ", ".join(path.as_posix() for path in unclassified)
        return EvidenceStatus(case.id, "stale", f"unclassified measured paths: {names}", path)
    if not path.is_file():
        return EvidenceStatus(case.id, "missing", "shard does not exist", path)
    try:
        payload = read_shard(case, root)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return EvidenceStatus(case.id, "invalid", str(error), path)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("case_id") != case.id
        or payload.get("case") != case.normalized()
    ):
        return EvidenceStatus(case.id, "invalid", "schema or case definition mismatch", path)
    current = case_fingerprint(case)
    recorded = payload.get("fingerprint", {})
    if not isinstance(recorded, dict) or recorded.get("digest") != current["digest"]:
        return EvidenceStatus(case.id, "stale", "case fingerprint changed", path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not _rows_are_complete(case, rows):
        return EvidenceStatus(case.id, "invalid", "incomplete or invalid observations", path)
    metadata = payload.get("metadata")
    try:
        expected_compatibility = (
            compatibility_signature(metadata) if isinstance(metadata, dict) else None
        )
    except (AttributeError, TypeError):
        expected_compatibility = None
    if expected_compatibility is None or payload.get("compatibility") != expected_compatibility:
        return EvidenceStatus(case.id, "invalid", "invalid compatibility metadata", path)
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict) or provenance.get("source_dirty") is not False:
        return EvidenceStatus(case.id, "dirty", "measured source tree was dirty", path)
    return EvidenceStatus(case.id, "current", current["digest"], path)
