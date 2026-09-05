from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence
import torch
from ncls.core.identity import sha256_json
from ncls.core.source import SourceSnapshot
from ncls.data import DataExecutionPlan, PipelineTrace
from ncls.learning.source_adaptation import NativeAssetCollection
from ncls.learning.batches import TrainingRouteRequest
from ncls.learning.conditioning_resources import AdaptedConditioning


class MethodSourceAdapter(ABC):
    method_key: str
    family_id: str
    source_contract_version: int
    adapter_id: str
    implementation_sha256: str

    def __init__(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> None:
        values = tuple(snapshots)
        if not values:
            raise ValueError("method source adapter requires source snapshots")
        if any(
            snapshot.family_id != self.family_id
            or snapshot.source_contract_version != self.source_contract_version
            for snapshot in values
        ):
            raise ValueError("method source adapter received an incompatible snapshot")
        self.snapshots = values
        self.device = device

    @property
    def identity(self) -> str:
        return sha256_json(
            {
                "adapter_id": self.adapter_id,
                "implementation_sha256": self.implementation_sha256,
                "source_snapshot_ids": [value.snapshot_id for value in self.snapshots],
            }
        )

    @abstractmethod
    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
        options: Mapping[str, Any],
        *,
        execution_source_indices: Sequence[int] | None = None,
    ) -> AdaptedConditioning:
        raise NotImplementedError

    def configure_data_execution(
        self, plan: DataExecutionPlan, trace: PipelineTrace
    ) -> None:
        del plan, trace

    def prefetch_host(
        self,
        candidates: Sequence[int],
        request: TrainingRouteRequest,
    ) -> None:
        del candidates, request

    def execution_source_indices(
        self,
        candidates: Sequence[int],
        request: TrainingRouteRequest,
    ) -> tuple[int, ...]:
        del request
        values = tuple(int(value) for value in candidates)
        if not values:
            raise ValueError("source execution cohort cannot be empty")
        return values

    def close(self) -> None:
        pass

    @abstractmethod
    def native_assets(self) -> NativeAssetCollection:
        raise NotImplementedError
