from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, runtime_checkable


TrainingRouteKind = Literal["asset-tile", "reference-evaluator", "method-sampler"]
_ROUTE_KINDS = {"asset-tile", "reference-evaluator", "method-sampler"}


@dataclass(frozen=True)
class DataRequirement:
    route_kind: TrainingRouteKind
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.route_kind not in _ROUTE_KINDS:
            raise ValueError(f"unsupported training route kind {self.route_kind!r}")
        fields = tuple(str(item) for item in self.fields)
        if not fields or len(set(fields)) != len(fields) or any(not item for item in fields):
            raise ValueError("data requirement fields must be unique and nonempty")
        object.__setattr__(self, "fields", fields)

    def to_dict(self) -> dict[str, Any]:
        return {"route_kind": self.route_kind, "fields": list(self.fields)}


@runtime_checkable
class OnlineBatch(Protocol):
    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def release(self) -> None: ...


@runtime_checkable
class OnlineProducer(Protocol):
    device: Any
    reference_program_identity: str
    reference_execution_plan_identity: str
    native_asset_collection_identity: str
    query_stream_identity: str
    source_contracts: tuple[Mapping[str, Any], ...]
    source_snapshot_ids: tuple[str, ...]

    def next_batch(self, request: Any) -> OnlineBatch: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def end_iteration(self) -> None: ...

    def profile_snapshot(self, *, reset: bool = False) -> Mapping[str, float]: ...

    def native_assets(self) -> Any: ...

    def close(self) -> None: ...


@runtime_checkable
class OnlineDataSession(Protocol):
    device: Any
    reference_program_identity: str
    reference_execution_plan_identity: str
    native_asset_collection_identity: str
    query_stream_identity: str
    source_contracts: tuple[Mapping[str, Any], ...]
    source_snapshot_ids: tuple[str, ...]

    @property
    def consumed_batches(self) -> int: ...

    def next_batch(self, request: Any) -> OnlineBatch: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def end_iteration(self) -> None: ...

    def profile_snapshot(self, *, reset: bool = False) -> Mapping[str, float]: ...

    def drain(self) -> None: ...

    def native_assets(self) -> Any: ...

    def close(self) -> None: ...


class TrainingDataDefinition(Protocol):
    @property
    def key(self) -> str: ...

    def requirements(self) -> tuple[DataRequirement, ...]: ...

    def open_session(self, execution_plan: Any, execution_context: Any) -> OnlineDataSession: ...
