from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ncls.learning.data import ReferenceQueryStore
from ncls.learning.evaluation.metrics import evaluator_metric_distributions, response_loss
from ncls.learning.models.neural_evaluator import (
    ARCHITECTURE_ID,
    NeuralEvaluatorModelConfig,
    SingleMaterialNeuralEvaluator,
    positive_response,
)

from .base import LearningPipeline, LearningPipelineDescriptor


LINEAR_PIPELINE_ID = "dense-latent-small-mlp-linear-e1@1"
LOG1P_PIPELINE_ID = "dense-latent-small-mlp-log1p-e1@1"
STANDARDIZED_LOG1P_PIPELINE_ID = "dense-latent-small-mlp-standardized-log1p-e1@1"
_FAMILIES = (
    "ncls.layer-stack@1",
    "merl.measured-brdf@1",
    "openpbr.surface@1.1.1",
    "materialx.textured-surface@1",
)
_FEATURE_CONTRACT = {
    "format_name": "ncls.feature-contract",
    "format_version": 1,
    "feature_contract_id": "ncls.local-frame-wo-wi@1",
    "inputs": {
        "prepare": ["material_latent", "wo"],
        "evaluate": ["prepared_view_code", "wi"],
    },
    "direction_space": "source-reference-local-frame",
}


class _DenseE1Pipeline(LearningPipeline):
    target_transform_id: str
    feature_contract = _FEATURE_CONTRACT

    def __init__(self) -> None:
        self._training_state: dict[str, Any] | None = None

    def open_store(self, dataset_path: str) -> ReferenceQueryStore:
        return ReferenceQueryStore(dataset_path)

    def fit_training_state(
        self,
        store: ReferenceQueryStore,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        if not len(train_indices):
            raise ValueError("dense E1 transform fitting requires nonempty train query groups")
        state_indices = np.unique(np.asarray(
            store.dataset.stream["queries/state_index"][train_indices], dtype=np.int64
        ))
        if len(state_indices) != 1:
            raise ValueError("dense E1 capacity runs must select exactly one source material state")
        state_index = int(state_indices[0])
        state_id = str(store.dataset.state_strings("state_id")[state_index])
        family_id = str(store.dataset.state_strings("family_id")[state_index])
        if family_id not in self.descriptor.supported_family_ids:
            raise ValueError(f"dense E1 pipeline does not support family {family_id!r}")

        scale = np.ones(3, dtype=np.float64)
        if self.target_transform_id == "ncls.train-only-channel-log1p@1":
            parts = []
            for start in range(0, len(train_indices), 4096):
                parts.append(np.asarray(
                    store.dataset.stream["responses/mean"][train_indices[start : start + 4096]],
                    dtype=np.float64,
                ).reshape(-1, 3))
            response = np.concatenate(parts, axis=0)
            peak = np.max(np.abs(response), axis=0)
            scale = np.maximum(
                np.quantile(np.maximum(response, 0.0), 0.9, axis=0),
                np.maximum(1e-3 * peak, 1e-6),
            )
        state = {
            "format_name": "ncls.fitted-training-state",
            "format_version": 1,
            "fit_scope": "final-train-query-groups-only",
            "selected_state_id": state_id,
            "selected_family_id": family_id,
            "train_query_group_count": int(len(train_indices)),
            "target_transform_id": self.target_transform_id,
            "target_channel_scale": [float(value) for value in scale],
        }
        return state

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        required = {
            "format_name", "format_version", "fit_scope", "selected_state_id",
            "selected_family_id", "train_query_group_count", "target_transform_id",
            "target_channel_scale",
        }
        if set(state) != required:
            raise ValueError("dense E1 fitted training state fields are unsupported")
        if (
            state["format_name"] != "ncls.fitted-training-state"
            or state["format_version"] != 1
            or state["fit_scope"] != "final-train-query-groups-only"
            or state["target_transform_id"] != self.target_transform_id
        ):
            raise ValueError("dense E1 fitted training state contract is unsupported")
        scale = np.asarray(state["target_channel_scale"], dtype=np.float64)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("dense E1 target channel scale must contain three positive finite values")
        if not str(state["selected_state_id"]) or int(state["train_query_group_count"]) < 1:
            raise ValueError("dense E1 fitted training state is incomplete")
        self._training_state = dict(state)

    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        return SingleMaterialNeuralEvaluator(NeuralEvaluatorModelConfig.from_mapping(model_parameters))

    def _require_batch_state(
        self,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
    ) -> None:
        if self._training_state is None:
            raise RuntimeError("dense E1 target transform has not been loaded")
        state_ids = store.dataset.state_strings("state_id")
        expected = str(self._training_state["selected_state_id"])
        matches = np.flatnonzero(state_ids == expected)
        if len(matches) != 1:
            raise ValueError("fitted dense E1 material state is absent from the dataset")
        if not torch.all(batch["state_index"].long() == int(matches[0])):
            raise ValueError("dense E1 checkpoint cannot evaluate a different material state")

    def _decode(self, raw: torch.Tensor) -> torch.Tensor:
        transformed = positive_response(raw)
        if self.target_transform_id == "ncls.identity-positive-linear-response@1":
            return transformed
        if self._training_state is None:
            raise RuntimeError("dense E1 target transform has not been loaded")
        scale = torch.as_tensor(
            self._training_state["target_channel_scale"],
            dtype=raw.dtype,
            device=raw.device,
        )
        return scale * torch.expm1(torch.clamp(transformed, max=20.0))

    def predict(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> torch.Tensor:
        del device
        self._require_batch_state(batch, store)
        if not isinstance(model, SingleMaterialNeuralEvaluator):
            raise TypeError("dense E1 pipeline requires SingleMaterialNeuralEvaluator")
        return self._decode(model(batch["view"].float(), batch["lights"].float()))

    def training_loss(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        target = batch["mean"].float()
        standard_error = batch["standard_error"].float()
        if self.target_transform_id == "ncls.identity-positive-linear-response@1":
            return response_loss(prediction, target, standard_error)
        if self._training_state is None:
            raise RuntimeError("dense E1 target transform has not been loaded")
        scale = torch.as_tensor(
            self._training_state["target_channel_scale"],
            dtype=prediction.dtype,
            device=prediction.device,
        )
        transformed_prediction = torch.log1p(torch.clamp(prediction, min=0.0) / scale)
        transformed_target = torch.log1p(torch.clamp(target, min=0.0) / scale)
        transformed_se = torch.log1p((torch.clamp(target, min=0.0) + standard_error) / scale) - transformed_target
        confidence = torch.clamp(
            transformed_target / (transformed_target + transformed_se + 1e-5), 0.1, 1.0
        ).detach()
        transformed_mse = torch.sum(
            confidence * torch.square(transformed_prediction - transformed_target)
        ) / torch.sum(confidence)
        return transformed_mse + 0.02 * response_loss(prediction, target, standard_error)

    def metric_distributions(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> Mapping[str, np.ndarray]:
        return evaluator_metric_distributions(
            prediction,
            batch["mean"].float(),
            batch["standard_error"].float(),
            batch["solid_angle_weight"].float(),
            batch["lights"].float(),
        )

    def additional_metric_distributions(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> Mapping[str, np.ndarray]:
        self._require_batch_state(batch, store)
        if not isinstance(model, SingleMaterialNeuralEvaluator):
            raise TypeError("dense E1 pipeline requires SingleMaterialNeuralEvaluator")
        wo = batch["view"].float()
        wi = batch["lights"].float()
        group_count, direction_count, _ = wi.shape
        reverse_raw = model(
            wi.reshape(group_count * direction_count, 3),
            wo[:, None, :].expand(-1, direction_count, -1).reshape(
                group_count * direction_count, 1, 3
            ),
        )
        reverse = self._decode(reverse_raw).reshape(group_count, direction_count, 3)
        forward_raw = model(wo, wi)
        forward = self._decode(forward_raw)
        wi_cosine = torch.abs(wi[..., 2:3])
        wo_cosine = torch.abs(wo[:, None, 2:3])
        valid = (wi_cosine > 0.05) & (wo_cosine > 0.05)
        forward_bsdf = forward / torch.clamp(wi_cosine, min=0.05)
        reverse_bsdf = reverse / torch.clamp(wo_cosine, min=0.05)
        delta = torch.where(valid, torch.abs(forward_bsdf - reverse_bsdf), torch.zeros_like(forward))
        magnitude = torch.where(valid, torch.abs(forward_bsdf), torch.zeros_like(forward))
        reciprocity = torch.sum(delta, dim=(1, 2)) / torch.clamp(
            torch.sum(magnitude, dim=(1, 2)), min=1e-8
        )
        return {"reciprocity_relative_l1": reciprocity.detach().cpu().numpy()}

    def parameter_costs(self, model: nn.Module) -> Mapping[str, Any]:
        if not isinstance(model, SingleMaterialNeuralEvaluator):
            return super().parameter_costs(model)
        costs = model.cost_summary()
        total = int(costs["parameter_count"])
        return {
            **costs,
            "B_asset_fp32": 4 * total,
            "B_shared_fp32": 0,
            "optimized_material_latent_bytes_fp32": 4 * model.material_latent.numel(),
            "optimized_material_network_bytes_fp32": 4 * (total - model.material_latent.numel()),
            "cost_scope": "single-material-capacity; all fitted parameters are asset-specific",
        }


class DenseLinearE1Pipeline(_DenseE1Pipeline):
    target_transform_id = "ncls.identity-positive-linear-response@1"
    descriptor = LearningPipelineDescriptor(
        pipeline_id=LINEAR_PIPELINE_ID,
        candidate_id="ncls.dense-latent-small-mlp@1",
        research_role="e1-single-material-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.identity-source-adapter@1",
        feature_transform_id="ncls.local-frame-wo-wi@1",
        target_transform_id=target_transform_id,
        representation_id="ncls.single-material-neural-evaluator@1",
        architecture_id=ARCHITECTURE_ID,
        latent_inference_id="ncls.optimized-dense-material-latent@1",
        compiler_id="ncls.none-capacity-study@1",
        loss_id="ncls.noise-aware-log-smape@1",
        metric_suite_id="ncls.evaluator-quality-suite@1",
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=_FAMILIES,
        scope="single-material-complete-directional-evaluator",
    )


class DenseLog1pE1Pipeline(_DenseE1Pipeline):
    target_transform_id = "ncls.train-only-channel-log1p@1"
    descriptor = LearningPipelineDescriptor(
        pipeline_id=LOG1P_PIPELINE_ID,
        candidate_id="ncls.dense-latent-small-mlp@1",
        research_role="e1-single-material-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.identity-source-adapter@1",
        feature_transform_id="ncls.local-frame-wo-wi@1",
        target_transform_id=target_transform_id,
        representation_id="ncls.single-material-neural-evaluator@1",
        architecture_id=ARCHITECTURE_ID,
        latent_inference_id="ncls.optimized-dense-material-latent@1",
        compiler_id="ncls.none-capacity-study@1",
        loss_id="ncls.train-only-log1p-noise-aware@1",
        metric_suite_id="ncls.evaluator-quality-suite@1",
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=_FAMILIES,
        scope="single-material-complete-directional-evaluator",
    )


class DenseStandardizedLog1pE1Pipeline(_DenseE1Pipeline):
    target_transform_id = "ncls.train-only-standardized-channel-log1p@1"
    descriptor = LearningPipelineDescriptor(
        pipeline_id=STANDARDIZED_LOG1P_PIPELINE_ID,
        candidate_id="ncls.dense-latent-small-mlp@1",
        research_role="e1-single-material-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.identity-source-adapter@1",
        feature_transform_id="ncls.local-frame-wo-wi@1",
        target_transform_id=target_transform_id,
        representation_id="ncls.single-material-neural-evaluator@1",
        architecture_id=ARCHITECTURE_ID,
        latent_inference_id="ncls.optimized-dense-material-latent@1",
        compiler_id="ncls.none-capacity-study@1",
        loss_id="ncls.standardized-log1p-noise-response-reciprocity@1",
        metric_suite_id="ncls.evaluator-quality-suite@1",
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=_FAMILIES,
        scope="single-material-complete-directional-evaluator",
    )

    def fit_training_state(
        self,
        store: ReferenceQueryStore,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        base = dict(super().fit_training_state(store, train_indices))
        parts = []
        for start in range(0, len(train_indices), 4096):
            parts.append(np.asarray(
                store.dataset.stream["responses/mean"][train_indices[start : start + 4096]],
                dtype=np.float64,
            ).reshape(-1, 3))
        response = np.maximum(np.concatenate(parts, axis=0), 0.0)
        scale = np.maximum(np.quantile(response, 0.5, axis=0), 1e-8)
        transformed = np.log1p(response / scale)
        mean = np.mean(transformed, axis=0)
        standard_deviation = np.maximum(np.std(transformed, axis=0), 1e-6)
        return {
            **base,
            "format_version": 2,
            "target_channel_scale": [float(value) for value in scale],
            "target_channel_mean": [float(value) for value in mean],
            "target_channel_standard_deviation": [
                float(value) for value in standard_deviation
            ],
        }

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        required = {
            "format_name", "format_version", "fit_scope", "selected_state_id",
            "selected_family_id", "train_query_group_count", "target_transform_id",
            "target_channel_scale", "target_channel_mean", "target_channel_standard_deviation",
        }
        if set(state) != required:
            raise ValueError("standardized log1p fitted training state fields are unsupported")
        if (
            state["format_name"] != "ncls.fitted-training-state"
            or state["format_version"] != 2
            or state["fit_scope"] != "final-train-query-groups-only"
            or state["target_transform_id"] != self.target_transform_id
        ):
            raise ValueError("standardized log1p fitted training state contract is unsupported")
        for name, positive in (
            ("target_channel_scale", True),
            ("target_channel_mean", False),
            ("target_channel_standard_deviation", True),
        ):
            values = np.asarray(state[name], dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)) or (
                positive and np.any(values <= 0.0)
            ):
                raise ValueError(f"standardized log1p {name} is invalid")
        if not str(state["selected_state_id"]) or int(state["train_query_group_count"]) < 1:
            raise ValueError("standardized log1p fitted training state is incomplete")
        self._training_state = dict(state)

    def _transform_tensors(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._training_state is None:
            raise RuntimeError("standardized log1p target transform has not been loaded")
        scale = torch.as_tensor(
            self._training_state["target_channel_scale"],
            dtype=reference.dtype,
            device=reference.device,
        )
        mean = torch.as_tensor(
            self._training_state["target_channel_mean"],
            dtype=reference.dtype,
            device=reference.device,
        )
        standard_deviation = torch.as_tensor(
            self._training_state["target_channel_standard_deviation"],
            dtype=reference.dtype,
            device=reference.device,
        )
        return scale, mean, standard_deviation

    def _decode(self, raw: torch.Tensor) -> torch.Tensor:
        scale, mean, standard_deviation = self._transform_tensors(raw)
        log_response = positive_response(raw * standard_deviation + mean)
        return scale * torch.expm1(torch.clamp(log_response, max=20.0))

    def _reciprocity_penalty(
        self,
        model: SingleMaterialNeuralEvaluator,
        batch: Mapping[str, torch.Tensor],
        forward: torch.Tensor,
    ) -> torch.Tensor:
        wo = batch["view"].float()
        wi = batch["lights"].float()
        group_count, direction_count, _ = wi.shape
        reverse = self._decode(model(
            wi.reshape(group_count * direction_count, 3),
            wo[:, None, :].expand(-1, direction_count, -1).reshape(
                group_count * direction_count, 1, 3
            ),
        )).reshape(group_count, direction_count, 3)
        wi_cosine = torch.abs(wi[..., 2:3])
        wo_cosine = torch.abs(wo[:, None, 2:3])
        valid = (wi_cosine > 0.05) & (wo_cosine > 0.05)
        forward_bsdf = forward / torch.clamp(wi_cosine, min=0.05)
        reverse_bsdf = reverse / torch.clamp(wo_cosine, min=0.05)
        scale, _, _ = self._transform_tensors(forward)
        delta = torch.abs(
            torch.log1p(forward_bsdf / scale) - torch.log1p(reverse_bsdf / scale)
        )
        return torch.sum(torch.where(valid, delta, torch.zeros_like(delta))) / torch.clamp(
            torch.sum(valid.expand_as(delta)), min=1
        )

    def predict(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> torch.Tensor:
        prediction = super().predict(model, batch, store, device)
        if model.training:
            if not isinstance(model, SingleMaterialNeuralEvaluator) or not isinstance(batch, dict):
                raise TypeError("standardized log1p training requires mutable tensor batches")
            batch["_reciprocity_penalty"] = self._reciprocity_penalty(model, batch, prediction)
        return prediction

    def training_loss(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        target = batch["mean"].float()
        standard_error = batch["standard_error"].float()
        scale, mean, standard_deviation = self._transform_tensors(prediction)
        transformed_prediction = (
            torch.log1p(torch.clamp(prediction, min=0.0) / scale) - mean
        ) / standard_deviation
        transformed_target = (
            torch.log1p(torch.clamp(target, min=0.0) / scale) - mean
        ) / standard_deviation
        transformed_se = torch.log1p(
            (torch.clamp(target, min=0.0) + standard_error) / scale
        ) - torch.log1p(torch.clamp(target, min=0.0) / scale)
        confidence = torch.clamp(
            torch.abs(transformed_target)
            / (torch.abs(transformed_target) + transformed_se / standard_deviation + 1e-5),
            0.1,
            1.0,
        ).detach()
        transformed_mse = torch.sum(
            confidence * torch.square(transformed_prediction - transformed_target)
        ) / torch.sum(confidence)
        reciprocity = batch.get("_reciprocity_penalty")
        reciprocity_loss = (
            reciprocity if isinstance(reciprocity, torch.Tensor) else prediction.new_zeros(())
        )
        return (
            transformed_mse
            + 0.02 * response_loss(prediction, target, standard_error)
            + 0.02 * reciprocity_loss
        )
