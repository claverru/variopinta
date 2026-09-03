from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Suite = Literal["transforms", "pipelines", "catalog", "io", "contracts"]
Role = Literal["public", "attribution", "control"]
Comparability = Literal["exact", "policy", "operational", "control"]


@dataclass(frozen=True, slots=True)
class RouteSpec:
    id: str
    participant: str
    variant: str
    environment: str
    role: Role = "public"

    def normalized(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimingPolicy:
    budget_ms: float
    warmup_calls: int
    min_samples: int
    max_calls: int
    target_sample_ms: float = 2.0
    block_size: int | None = None

    def normalized(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CaseSpec:
    id: str
    suite: Suite
    label: str
    tags: tuple[str, ...]
    routes: tuple[RouteSpec, ...]
    sizes: tuple[int, ...]
    executor: str
    factory: str
    comparability: Comparability
    scopes: tuple[str, ...]
    timing: TimingPolicy | None

    @property
    def timed(self) -> bool:
        return self.timing is not None

    def normalized(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "suite": self.suite,
            "label": self.label,
            "tags": list(self.tags),
            "routes": [route.normalized() for route in self.routes],
            "sizes": list(self.sizes),
            "executor": self.executor,
            "factory": self.factory,
            "comparability": self.comparability,
            "scopes": list(self.scopes),
            "timing": None if self.timing is None else self.timing.normalized(),
        }


@dataclass(frozen=True, slots=True)
class PlannedCase:
    case: CaseSpec
    routes: tuple[RouteSpec, ...]
    sizes: tuple[int, ...]

    def normalized(self) -> dict[str, Any]:
        value = self.case.normalized()
        value["selected_routes"] = [route.id for route in self.routes]
        value["selected_sizes"] = list(self.sizes)
        return value
