from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ncls.learning.data import LayerStackReferenceStore, ReferenceQueryStore
from ncls.learning.evaluation.metrics import directional_relative_l1, response_loss
from ncls.learning.features import FEATURE_CONTRACT, FEATURE_CONTRACT_ID
from ncls.learning.models.legacy_ltc_k2_p1 import ARCHITECTURE_ID, LegacyLtcK2P1Compiler
from ncls.learning.prediction import predict_legacy_ltc_k2_response

from .base import LearningPipeline, LearningPipelineDescriptor


PIPELINE_ID = "legacy-ltc-k2-p1-deployment-regression@1"


class LegacyLtcK2Pipeline(LearningPipeline):
    """保留部署回归基线的注册适配；它不定义目标 evaluator 公共接口。"""

    descriptor = LearningPipelineDescriptor(
        pipeline_id=PIPELINE_ID,
        candidate_id="legacy-ltc-k2-deployment-regression@1",
        research_role="deployment-regression",
        response_reader_id="ncls.reference-query-store@1",
        source_adapter_id="ncls.layer-stack-source-adapter@2",
        feature_transform_id=FEATURE_CONTRACT_ID,
        target_transform_id="ncls.identity-linear-response@1",
        representation_id="legacy-ltc-k2@1",
        architecture_id=ARCHITECTURE_ID,
        latent_inference_id="ncls.none-analytic-state@1",
        compiler_id="legacy-ltc-k2-p1-source-compiler@2",
        loss_id="ncls.legacy-response-log-smape@1",
        metric_suite_id="ncls.basic-directional-response@1",
        exporter_id="ncls.legacy-ltc-k2-method-bundle@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope="source-state-compiler",
    )
    feature_contract = FEATURE_CONTRACT

    def open_store(self, dataset_path: str) -> ReferenceQueryStore:
        return LayerStackReferenceStore(dataset_path)

    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        allowed = {"width"}
        unknown = set(model_parameters) - allowed
        if unknown:
            raise ValueError(f"legacy pipeline received unsupported model parameters: {sorted(unknown)}")
        return LegacyLtcK2P1Compiler(width=int(model_parameters.get("width", 64)))

    def predict(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(store, LayerStackReferenceStore):
            raise TypeError("legacy pipeline requires LayerStackReferenceStore")
        lights = torch.as_tensor(store.lights, dtype=torch.float32, device=device)
        return predict_legacy_ltc_k2_response(model, dict(batch), lights)

    def training_loss(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return response_loss(
            prediction,
            batch["mean"].float(),
            batch["standard_error"].float(),
        )

    def metric_distributions(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> Mapping[str, np.ndarray]:
        target = batch["mean"].float()
        relative = directional_relative_l1(prediction, target)
        replica = torch.sum(
            torch.abs(batch["replica_mean_a"].float() - batch["replica_mean_b"].float()),
            dim=(1, 2),
        ) / torch.clamp(torch.sum(torch.abs(target), dim=(1, 2)), min=1e-8)
        return {
            "relative_l1": relative.detach().cpu().numpy(),
            "replica_relative_l1": replica.detach().cpu().numpy(),
        }
