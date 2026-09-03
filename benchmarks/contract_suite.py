from __future__ import annotations

from typing import Any

from common import control_cpu
from correctness import run_correctness_checks


def run_planned(items: list[dict[str, Any]], repetition: int) -> list[dict[str, Any]]:
    control_cpu()
    rows = []
    for order, item in enumerate(items, start=1):
        route = item["route"]
        backend = "rust" if route["participant"] == "variopinta" else route["participant"]
        checks = run_correctness_checks(backend)
        rows.append(
            {
                "case_id": item["case_id"],
                "route_id": route["id"],
                "participant": route["participant"],
                "variant": route["variant"],
                "role": route["role"],
                "size": None,
                "repetition": repetition,
                "case_order": order,
                "validation": {"checks": checks},
                "valid": all(check.get("valid", True) for check in checks),
            }
        )
    return rows
