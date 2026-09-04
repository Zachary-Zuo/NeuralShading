from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

from ncls.core.identity import sha256_json

from .contracts import DataRequirement, TrainingRouteKind


@dataclass(frozen=True)
class RankPartition:
    rank: int
    world_size: int
    recipe: str = "rank-strided-logical-request"

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("data rank partition is outside the distributed world")
        if not self.recipe:
            raise ValueError("data rank partition recipe is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "recipe": self.recipe,
        }


@dataclass(frozen=True)
class DataRoutePlan:
    name: str
    kind: TrainingRouteKind
    required_fields: tuple[str, ...]
    batch_size: int
    direction_count: int
    seed_offset: int
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name or self.batch_size < 1 or self.direction_count < 1:
            raise ValueError("data route identity and sizes are invalid")
        if self.seed_offset < 0:
            raise ValueError("data route seed offset must be nonnegative")
        required_fields = tuple(str(item) for item in self.required_fields)
        if not required_fields or len(set(required_fields)) != len(required_fields):
            raise ValueError("data route required fields must be unique and nonempty")
        object.__setattr__(self, "required_fields", required_fields)
        object.__setattr__(self, "options", dict(self.options))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "required_fields": list(self.required_fields),
            "batch_size": self.batch_size,
            "direction_count": self.direction_count,
            "seed_offset": self.seed_offset,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class DataExecutionPlan:
    data_key: str
    source_family_id: str
    routes: tuple[DataRoutePlan, ...]
    partition: RankPartition
    num_workers: int
    host_prefetch: int
    ready_batches: int
    reference_batch_steps: int
    reference_inflight: int
    transfer_streams: int
    residency_budget_bytes: int
    format_name: str = "ncls.data-execution-plan"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_name != "ncls.data-execution-plan" or self.format_version != 1:
            raise ValueError("unsupported data execution plan format")
        routes = tuple(self.routes)
        if not self.data_key or not self.source_family_id or not routes:
            raise ValueError("data execution plan identity and routes are required")
        if len({item.name for item in routes}) != len(routes):
            raise ValueError("data execution route names must be unique")
        if self.num_workers < 0 or self.transfer_streams < 0:
            raise ValueError("data worker and transfer stream counts must be nonnegative")
        if min(
            self.host_prefetch,
            self.ready_batches,
            self.reference_batch_steps,
            self.reference_inflight,
            self.residency_budget_bytes,
        ) < 1:
            raise ValueError("data queue, reference and residency budgets must be positive")
        object.__setattr__(self, "routes", routes)

    @classmethod
    def build(
        cls,
        *,
        data_key: str,
        source_family_id: str,
        routes: Sequence[Mapping[str, Any]],
        requirements: Sequence[DataRequirement],
        execution: Mapping[str, Any],
        rank: int = 0,
        world_size: int = 1,
    ) -> "DataExecutionPlan":
        by_kind = {item.route_kind: item for item in requirements}
        route_plans = []
        for route in routes:
            kind = str(route["kind"])
            try:
                requirement = by_kind[cast(TrainingRouteKind, kind)]
            except KeyError as error:
                raise ValueError(
                    f"data route {route.get('name')!r} has no method requirement"
                ) from error
            route_plans.append(
                DataRoutePlan(
                    str(route["name"]),
                    cast(TrainingRouteKind, kind),
                    requirement.fields,
                    int(route["batch_size"]),
                    int(route["direction_count"]),
                    int(route["seed_offset"]),
                    dict(route["options"]),
                )
            )
        missing = set(by_kind) - {item.kind for item in route_plans}
        if missing:
            raise ValueError(f"data execution plan omits required routes {sorted(missing)}")
        residency = execution.get("residency")
        if not isinstance(residency, Mapping) or set(residency) != {"budget_mib"}:
            raise ValueError("data execution residency requires only budget_mib")
        return cls(
            data_key,
            source_family_id,
            tuple(route_plans),
            RankPartition(rank, world_size),
            int(execution["num_workers"]),
            int(execution["host_prefetch"]),
            int(execution["ready_batches"]),
            int(execution["reference_batch_steps"]),
            int(execution["reference_inflight"]),
            int(execution["transfer_streams"]),
            int(residency["budget_mib"]) * 1024 * 1024,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "data_key": self.data_key,
            "source_family_id": self.source_family_id,
            "routes": [item.to_dict() for item in self.routes],
            "partition": self.partition.to_dict(),
            "num_workers": self.num_workers,
            "host_prefetch": self.host_prefetch,
            "ready_batches": self.ready_batches,
            "reference_batch_steps": self.reference_batch_steps,
            "reference_inflight": self.reference_inflight,
            "transfer_streams": self.transfer_streams,
            "residency_budget_bytes": self.residency_budget_bytes,
        }

    @property
    def identity(self) -> str:
        # Checkpoints are shared by all DDP ranks, so their data identity must
        # describe the common partition contract rather than one rank's cursor.
        # The concrete rank remains part of the session identity below.
        value = self.to_dict()
        value["partition"] = {
            "world_size": self.partition.world_size,
            "recipe": self.partition.recipe,
        }
        return sha256_json(value)

    @property
    def session_identity(self) -> str:
        return sha256_json(self.to_dict())
