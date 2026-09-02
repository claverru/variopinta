from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import RESULTS, benchmark_fingerprint

CANONICAL_ARTIFACTS = (
    RESULTS / "raw" / "benchmark-runs.json",
    RESULTS / "raw" / "catalog-audit.json",
    RESULTS / "raw" / "catalog-benchmark.json",
    RESULTS / "raw" / "io-parity.json",
    RESULTS / "raw" / "io-performance.json",
    RESULTS / "layers" / "raw" / "all-runs.json",
)


def artifact_status(path: Path, expected: str) -> tuple[str, str]:
    if not path.exists():
        return "missing", "artifact does not exist"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return "invalid", str(error)
    if not isinstance(payload, dict) or not isinstance(payload.get("provenance", {}), dict):
        return "invalid", "top-level payload or provenance is not an object"
    provenance = payload.get("provenance", {})
    actual = provenance.get("benchmark_fingerprint")
    if actual is None:
        return "legacy", "artifact predates benchmark fingerprints"
    if actual != expected:
        return "stale", f"recorded {actual}"
    if provenance.get("source_dirty") is not False:
        return "dirty", "artifact was generated from a dirty or unknown source tree"
    return "current", actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether canonical evidence matches the code"
    )
    parser.add_argument("paths", type=Path, nargs="*", default=CANONICAL_ARTIFACTS)
    args = parser.parse_args()
    expected = benchmark_fingerprint()
    failed = False
    print(f"Expected fingerprint: {expected}")
    for path in args.paths:
        status, detail = artifact_status(path, expected)
        try:
            label = path.relative_to(RESULTS.parent)
        except ValueError:
            label = path
        print(f"{status:7} {label}: {detail}")
        failed |= status != "current"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
