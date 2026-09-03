from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.controller import collect_evidence, execute_plan, stale_plan
from benchmarks.evidence import status_for
from benchmarks.registry import CASES
from benchmarks.selection import Selectors, select_cases, validate_selector_values
from benchmarks.views import render


def _selectors(parser: argparse.ArgumentParser, *, dimensions: bool = True) -> None:
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    if dimensions:
        parser.add_argument("--size", type=int, action="append", default=[])
        parser.add_argument("--participant", action="append", default=[])
        parser.add_argument("--variant", action="append", default=[])


def _values(arguments: argparse.Namespace) -> Selectors:
    return Selectors(
        suites=tuple(arguments.suite),
        cases=tuple(arguments.case),
        tags=tuple(arguments.tag),
        sizes=tuple(getattr(arguments, "size", ())),
        participants=tuple(getattr(arguments, "participant", ())),
        variants=tuple(getattr(arguments, "variant", ())),
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Registry-driven benchmark execution")
    commands = root.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="list the normalized benchmark selection")
    _selectors(listing)

    run = commands.add_parser("run", help="run a local diagnostic selection")
    _selectors(run)
    run.add_argument("--quick", action="store_true")
    run.add_argument("--repetitions", type=int)
    run.add_argument(
        "--current-environment",
        action="store_true",
        help="run workers with the current interpreter instead of managed environments",
    )

    validate = commands.add_parser("validate", help="run correctness without performance timing")
    _selectors(validate, dimensions=False)

    evidence = commands.add_parser("evidence", help="renew complete canonical case shards")
    _selectors(evidence, dimensions=False)
    mode = evidence.add_mutually_exclusive_group()
    mode.add_argument("--stale", action="store_true")
    mode.add_argument("--full", action="store_true")

    status = commands.add_parser("status", help="check canonical evidence")
    _selectors(status, dimensions=False)

    rendering = commands.add_parser("render", help="render derived views from canonical shards")
    _selectors(rendering, dimensions=False)
    rendering.add_argument("--output", type=Path)
    return root


def main() -> None:
    arguments = parser().parse_args()
    selectors = _values(arguments)
    if arguments.command == "evidence" and arguments.full:
        selectors = Selectors()
    try:
        validate_selector_values(CASES, selectors)
        plan = select_cases(CASES, selectors, complete=arguments.command not in {"list", "run"})
        if arguments.command == "list":
            print(json.dumps([planned.normalized() for planned in plan], indent=2, sort_keys=True))
        elif arguments.command == "run":
            repetitions = arguments.repetitions or (1 if arguments.quick else 3)
            payload = execute_plan(
                plan,
                repetitions=repetitions,
                quick=arguments.quick,
                current_environment=arguments.current_environment,
                kind="quick" if arguments.quick else "run",
            )
            print(payload["run_directory"])
        elif arguments.command == "validate":
            supported = tuple(
                planned
                for planned in plan
                if not planned.case.timed or planned.case.suite == "catalog"
            )
            if len(supported) != len(plan):
                unsupported = sorted(
                    planned.case.id for planned in plan if planned not in supported
                )
                raise ValueError(
                    "validation-only execution is not defined for: " + ", ".join(unsupported)
                )
            payload = execute_plan(
                supported,
                repetitions=1,
                quick=True,
                validate_only=True,
                kind="validation",
            )
            invalid = [row for row in payload["rows"] if row.get("valid") is not True]
            if invalid:
                raise RuntimeError(f"validation failed for {len(invalid)} rows")
            print(payload["run_directory"])
        elif arguments.command == "evidence":
            selected = plan if not arguments.stale else stale_plan(plan)
            if not selected:
                print("All selected evidence is current.")
                return
            paths = collect_evidence(selected)
            for path in paths:
                print(path)
        elif arguments.command == "status":
            failed = False
            for planned in plan:
                status = status_for(planned.case)
                print(f"{status.state:7} {status.case_id}: {status.detail}")
                failed |= status.state != "current"
            if failed:
                raise SystemExit(1)
        elif arguments.command == "render":
            print(render(plan, arguments.output))
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
