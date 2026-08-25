from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ncls.learning.data import (
    PARTITION_POLICY_IDS,
    ReferenceCorpusStore,
    ReferenceQueryStore,
    open_reference_store,
)


def _is_descriptor_value(name: str, value: object) -> bool:
    if name == "deployment_candidate":
        return isinstance(value, bool)
    return isinstance(value, str) and bool(value)


@dataclass(frozen=True)
class LearningPipelineDescriptor:
    """候选的可读身份；精确实现身份由结构的 SHA-256 给出（不含研究期分类 `runtime.deployment_candidate`）。"""

    name: str
    stage: str
    data: Mapping[str, str]
    model: Mapping[str, str]
    fitting: Mapping[str, str]
    runtime: Mapping[str, str | bool]
    supported_families: tuple[str, ...]
    scope: str
    schema_name: str = "learning-pipeline"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_name != "learning-pipeline" or self.schema_version != 1:
            raise ValueError("unsupported learning pipeline descriptor")
        if not self.name or not self.stage or not self.scope or not self.supported_families:
            raise ValueError("pipeline name, stage, scope and supported families are required")
        if self.data.get("partition") not in PARTITION_POLICY_IDS:
            raise ValueError("learning pipeline partition policy is unsupported")
        required = {
            "data": (self.data, {"reader", "partition", "source_adapter"}),
            "model": (self.model, {"representation", "architecture", "latent"}),
            "fitting": (self.fitting, {"path", "loss"}),
            "runtime": (self.runtime, {"compiler", "exporter", "deployment_candidate"}),
        }
        for group, (value, fields) in required.items():
            if set(value) != fields or any(
                not _is_descriptor_value(name, item) for name, item in value.items()
            ):
                raise ValueError(f"pipeline {group} fields must be exactly {sorted(fields)}")
        if self.fitting["path"] not in {"gradient", "direct-fit", "hybrid"}:
            raise ValueError("pipeline fitting path is unsupported")
        if len(set(self.supported_families)) != len(self.supported_families) or any(
            not isinstance(family, str) or not family for family in self.supported_families
        ):
            raise ValueError("supported pipeline families must be unique readable names")

    @property
    def partition_policy_id(self) -> str:
        return self.data["partition"]

    @property
    def deployment_candidate(self) -> bool:
        return bool(self.runtime["deployment_candidate"])

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["data"] = dict(self.data)
        value["model"] = dict(self.model)
        value["fitting"] = dict(self.fitting)
        value["runtime"] = dict(self.runtime)
        value["supported_families"] = list(self.supported_families)
        return value

    @property
    def sha256(self) -> str:
        """实现身份；`deployment_candidate` 只是部署预算分类，不改变实现，P1 v1 checkpoint 仍可评测。"""

        value = self.to_dict()
        value["runtime"] = {
            name: item for name, item in value["runtime"].items() if name != "deployment_candidate"
        }
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LearningPipeline(ABC):
    """训练和评测共享的最小生命周期；指标固定由 quality-v1 计算。"""

    descriptor: LearningPipelineDescriptor

    def open_store(self, data_path: str) -> ReferenceQueryStore | ReferenceCorpusStore:
        return open_reference_store(data_path)

    def lifecycle_indices(self, store: ReferenceQueryStore, lifecycle_role: str) -> np.ndarray:
        return store.partition_indices(self.descriptor.partition_policy_id, lifecycle_role)

    def evaluation_indices(self, store: ReferenceQueryStore, evaluation_role: str) -> np.ndarray:
        if evaluation_role in {"adversarial_probe", "dense_slice"}:
            return store.indices_for_query_role(evaluation_role)
        return self.lifecycle_indices(store, evaluation_role)

    def fit_training_state(
        self,
        store: ReferenceQueryStore,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        del store, train_indices
        return {}

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError(f"pipeline {self.descriptor.name} does not accept fitted training state")

    def parameter_costs(self, model: nn.Module) -> Mapping[str, Any]:
        total = sum(parameter.numel() for parameter in model.parameters())
        return {
            "B_asset": 0,
            "B_shared": 4 * total,
            "C_prepare_macs": None,
            "C_eval_macs": None,
            "parameter_count": total,
        }

    def initialize_model_from_checkpoint(
        self,
        model: nn.Module,
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del model, checkpoint
        raise ValueError(f"pipeline {self.descriptor.name} does not accept initialization")

    @abstractmethod
    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def predict_f(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> torch.Tensor:
        """返回线性 RGB `f`，不得预乘 cosine。"""

        raise NotImplementedError

    @abstractmethod
    def training_loss(
        self,
        prediction_f: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError
