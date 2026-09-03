from __future__ import annotations

from dataclasses import dataclass

from benchmarks.model import CaseSpec, PlannedCase


@dataclass(frozen=True, slots=True)
class Selectors:
    suites: tuple[str, ...] = ()
    cases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    sizes: tuple[int, ...] = ()
    participants: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()

    @property
    def has_dimension_filters(self) -> bool:
        return bool(self.sizes or self.participants or self.variants)


def select_cases(
    registry: tuple[CaseSpec, ...], selectors: Selectors, *, complete: bool = False
) -> tuple[PlannedCase, ...]:
    known_ids = {case.id for case in registry}
    unknown = sorted(set(selectors.cases) - known_ids)
    if unknown:
        raise ValueError(f"unknown benchmark cases: {', '.join(unknown)}")

    selected: list[PlannedCase] = []
    for case in registry:
        if selectors.suites and case.suite not in selectors.suites:
            continue
        if selectors.cases and case.id not in selectors.cases:
            continue
        if selectors.tags and not all(tag in case.tags for tag in selectors.tags):
            continue

        routes = case.routes
        sizes = case.sizes
        if not complete:
            if selectors.participants:
                routes = tuple(
                    route for route in routes if route.participant in selectors.participants
                )
            if selectors.variants:
                routes = tuple(route for route in routes if route.variant in selectors.variants)
            if selectors.sizes:
                sizes = tuple(size for size in sizes if size in selectors.sizes)
            if not routes or (case.sizes and not sizes):
                continue
        selected.append(PlannedCase(case, routes, sizes))

    if not selected:
        raise ValueError("selectors matched no benchmark cases")
    return tuple(selected)


def validate_selector_values(registry: tuple[CaseSpec, ...], selectors: Selectors) -> None:
    suites = {case.suite for case in registry}
    participants = {route.participant for case in registry for route in case.routes}
    variants = {route.variant for case in registry for route in case.routes}
    sizes = {size for case in registry for size in case.sizes}
    checks = (
        (selectors.suites, suites, "suites"),
        (selectors.participants, participants, "participants"),
        (selectors.variants, variants, "variants"),
        (selectors.sizes, sizes, "sizes"),
    )
    for requested, known, label in checks:
        unknown = sorted(set(requested) - known)
        if unknown:
            values = ", ".join(str(value) for value in unknown)
            raise ValueError(f"unknown benchmark {label}: {values}")
