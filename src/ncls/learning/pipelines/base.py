from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ncls.learning.data import PARTITION_POLICY_IDS, ReferenceQueryStore


PIPELINE_CONTRACT_FORMAT = "ncls.learning-pipeline"
PIPELINE_CONTRACT_VERSION = 2


@dataclass(frozen=True)
class LearningPipelineDescriptor:
    pipeline_id: str
    candidate_id: str
    research_role: str
    response_reader_id: str
    partition_policy_id: str
    source_adapter_id: str
    feature_transform_id: str
    target_transform_id: str
    representation_id: str
    architecture_id: str
    latent_inference_id: str
    compiler_id: str
    loss_id: str
    metric_suite_id: str
    exporter_id: str
    supported_family_ids: tuple[str, ...]
    scope: str
    format_name: str = PIPELINE_CONTRACT_FORMAT
    format_version: int = PIPELINE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.format_name != PIPELINE_CONTRACT_FORMAT or self.format_version != PIPELINE_CONTRACT_VERSION:
            raise ValueError("unsupported learning pipeline descriptor")
        identifiers = (
            self.pipeline_id,
            self.candidate_id,
            self.response_reader_id,
            self.partition_policy_id,
            self.source_adapter_id,
            self.feature_transform_id,
            self.target_transform_id,
            self.representation_id,
            self.architecture_id,
            self.latent_inference_id,
            self.compiler_id,
            self.loss_id,
            self.metric_suite_id,
            self.exporter_id,
        )
        if any("@" not in value for value in identifiers):
            raise ValueError("all learning pipeline component IDs must be versioned")
        if self.partition_policy_id not in PARTITION_POLICY_IDS:
            raise ValueError("learning pipeline partition policy is unsupported")
        if not self.research_role or not self.scope or not self.supported_family_ids:
            raise ValueError("learning pipeline role, scope and supported families are required")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supported_family_ids"] = list(self.supported_family_ids)
        return value

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LearningPipeline(ABC):
    """公共 runner 使用的最小生命周期；所有候选细节留在注册实现中。"""

    descriptor: LearningPipelineDescriptor
    feature_contract: Mapping[str, Any]

    @abstractmethod
    def open_store(self, dataset_path: str) -> ReferenceQueryStore:
        raise NotImplementedError

    def lifecycle_indices(self, store: ReferenceQueryStore, lifecycle_role: str) -> np.ndarray:
        return store.partition_indices(self.descriptor.partition_policy_id, lifecycle_role)

    def evaluation_indices(self, store: ReferenceQueryStore, evaluation_role: str) -> np.ndarray:
        if evaluation_role == "adversarial_probe":
            return store.dataset.group_indices(query_role=evaluation_role)
        return self.lifecycle_indices(store, evaluation_role)

    def fit_training_state(
        self,
        store: ReferenceQueryStore,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        """只允许从最终 train partition 拟合 transform/codebook 等压缩期状态。"""

        return {}

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError(f"pipeline {self.descriptor.pipeline_id} does not accept fitted training state")

    def parameter_costs(self, model: nn.Module) -> Mapping[str, Any]:
        total = sum(parameter.numel() for parameter in model.parameters())
        return {
            "parameter_count": total,
            "B_asset_fp32": 0,
            "B_shared_fp32": 4 * total,
            "C_prepare_macs": None,
            "C_eval_macs": None,
        }

    def additional_metric_distributions(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> Mapping[str, np.ndarray]:
        return {}

    @abstractmethod
    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def training_loss(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def metric_distributions(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> Mapping[str, np.ndarray]:
        raise NotImplementedError
