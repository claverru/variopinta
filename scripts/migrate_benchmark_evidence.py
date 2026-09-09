from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from benchmarks.common import ROOT
from benchmarks.controller import validate_complete
from benchmarks.evidence import CANONICAL_REPETITIONS, SCHEMA_VERSION, shard_path, write_json_atomic
from benchmarks.fingerprints import case_fingerprint, compatibility_signature
from benchmarks.model import CaseSpec, PlannedCase
from benchmarks.registry import CASES

PREVIOUS_REVISION = "9c746a82e29b4e0366209a0ea2e3f822aac33542"
CONTROLLER_REMOVALS = (
    (", source_provenance\n", "\n"),
    ("    provenance = source_provenance()\n", ""),
    ('            "provenance": provenance,\n', ""),
)


def migrate_shard(
    case: CaseSpec,
    payload: dict[str, Any],
    previous: dict[str, Any],
    expected: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("case_id") != case.id
        or payload.get("case") != previous["case"]
        or payload.get("fingerprint") != previous["fingerprint"]
    ):
        raise ValueError(f"{case.id}: evidence does not match the previous code and rules")
    definition = case.normalized()
    if {**previous["case"], "tags": definition["tags"]} != definition:
        raise ValueError(f"{case.id}: measured case definition changed")
    if current != expected:
        raise ValueError(f"{case.id}: measured sources or environment changed; rerun this case")
    validate_complete(
        PlannedCase(case, case.routes, case.sizes), payload["rows"], CANONICAL_REPETITIONS
    )
    if payload.get("compatibility") != compatibility_signature(payload["metadata"]):
        raise ValueError(f"{case.id}: invalid compatibility metadata")
    migrated = {**payload, "case": definition, "fingerprint": current}
    migrated.pop("provenance", None)
    return migrated


def prepare_migration(snapshot: Path, root: Path = ROOT) -> list[tuple[Path, dict[str, Any]]]:
    program = """
import json
from benchmarks.fingerprints import case_fingerprint
from benchmarks.registry import CASES
print(json.dumps({case.id: {"case": case.normalized(), "fingerprint": case_fingerprint(case)} for case in CASES}))
"""
    previous = json.loads(
        subprocess.run(
            [sys.executable, "-c", program],
            cwd=snapshot,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if previous.keys() != {case.id for case in CASES}:
        raise ValueError("benchmark case set changed")

    # Only post-measurement Git bookkeeping may differ in the measured controller file.
    controller = snapshot / "benchmarks/controller.py"
    source = controller.read_text()
    for old, new in CONTROLLER_REMOVALS:
        if source.count(old) != 1:
            raise ValueError("unsupported previous benchmark controller")
        source = source.replace(old, new)
    controller.write_text(source)

    updates = []
    for case in CASES:
        path = shard_path(case, root / "benchmarks/evidence")
        updates.append(
            (
                path,
                migrate_shard(
                    case,
                    json.loads(path.read_text()),
                    previous[case.id],
                    case_fingerprint(case, snapshot),
                    case_fingerprint(case, root),
                ),
            )
        )
    return updates


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate pre-scoping benchmark metadata once")
    parser.add_argument("--check", action="store_true", help="verify without writing shards")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="variopinta-evidence-migration-") as directory:
        temporary = Path(directory)
        archive = temporary / "previous.tar"
        snapshot = temporary / "previous"
        snapshot.mkdir()
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", PREVIOUS_REVISION],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(["tar", "-xf", str(archive), "-C", str(snapshot)], check=True)
        updates = prepare_migration(snapshot)
    if not args.check:
        for path, payload in updates:
            write_json_atomic(path, payload)
    print(
        f"{'Verified' if args.check else 'Migrated'} {len(updates)} shards; observations preserved."
    )


if __name__ == "__main__":
    main()
