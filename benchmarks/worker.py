from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common import control_cpu, metadata, write_json


def execute(request: dict[str, Any]) -> dict[str, Any]:
    executor = request["executor"]
    items = request["items"]
    repetition = int(request["repetition"])
    quick = bool(request.get("quick", False))
    validate_only = bool(request.get("validate_only", False))
    cpu = control_cpu()
    if executor == "layers":
        from layer_worker import run_planned

        rows = run_planned(items, quick, repetition)
    elif executor == "catalog":
        from catalog_suite import run_planned

        rows = run_planned(items, quick, repetition, validate_only=validate_only)
    elif executor == "io-performance":
        from io_performance_worker import run_planned

        rows = run_planned(items, quick, repetition)
    elif executor == "io-parity":
        from io_parity_worker import run_planned

        rows = run_planned(items, repetition)
    elif executor == "contracts":
        from contract_suite import run_planned

        rows = run_planned(items, repetition)
    else:
        raise ValueError(f"unknown benchmark executor: {executor}")
    return {
        "schema_version": 1,
        "executor": executor,
        "environment": request["environment"],
        "repetition": repetition,
        "metadata": metadata(request["environment"], cpu),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    request = json.loads(arguments.request.read_text())
    write_json(arguments.output, execute(request))


if __name__ == "__main__":
    main()
