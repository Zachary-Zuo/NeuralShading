from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ncls.learning.data import LayerStackReferenceStore, ReferenceQueryStore
from ncls.learning.evaluation.metrics import evaluator_metric_distributions, response_loss
from ncls.learning.losses import energy_shape_terms
from ncls.learning.models.neural_evaluator import (
    NeuralEvaluatorModelConfig,
    SHARED_ARCHITECTURE_ID,
    SharedMaterialNeuralEvaluator,
    positive_response,
)
from ncls.learning.source_adapters import evaluate_layer_stack_direct_top

from .base import LearningPipeline, LearningPipelineDescriptor


PIPELINE_ID = "dense-latent-shared-small-mlp-energy-shape-e2@1"
ANALYTIC_RESIDUAL_PIPELINE_ID = "analytic-core-shared-neural-residual-energy-shape-e2@1"
_TARGET_TRANSFORM_ID = "ncls.train-only-standardized-channel-log1p@1"
_RESIDUAL_TARGET_TRANSFORM_ID = "ncls.train-only-standardized-asinh-analytic-residual@1"
_FAMILIES = (
    "ncls.layer-stack@1",
    "merl.measured-brdf@1",
    "openpbr.surface@1.1.1",
    "materialx.textured-surface@1",
)
_FEATURE_CONTRACT = {
    "format_name": "ncls.feature-contract",
    "format_version": 1,
    "feature_contract_id": "ncls.local-frame-wo-wi-material-slot@1",
    "inputs": {
        "prepare": ["optimized_material_latent", "wo"],
        "evaluate": ["prepared_view_code", "wi"],
    },
    "direction_space": "source-reference-local-frame",
    "material_addressing": "fitted-target-visible-state-slot",
}


class DenseLatentSharedEvaluatorE2Pipeline(LearningPipeline):
    """E2 target-visible autodecoder 上界；不读取 native payload，也不是 source compiler。"""

    feature_contract = _FEATURE_CONTRACT
    target_transform_id = _TARGET_TRANSFORM_ID
    descriptor = LearningPipelineDescriptor(
        pipeline_id=PIPELINE_ID,
        candidate_id="ncls.dense-latent-small-mlp@1",
        research_role="e2-shared-representation-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.identity-source-adapter@1",
        feature_transform_id="ncls.local-frame-wo-wi-material-slot@1",
        target_transform_id=target_transform_id,
        representation_id="ncls.shared-neural-evaluator-dense-material-latent-table@1",
        architecture_id=SHARED_ARCHITECTURE_ID,
        latent_inference_id="ncls.optimized-target-visible-dense-material-latent-table@1",
        compiler_id="ncls.none-target-visible-capacity-study@1",
        loss_id="ncls.standardized-log1p-energy-shape-reciprocity@1",
        metric_suite_id="ncls.evaluator-quality-suite-by-state@1",
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=_FAMILIES,
        scope="multi-material-target-visible-autodecoder-complete-directional-capacity",
    )

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
            raise ValueError("shared E2 transform fitting requires nonempty train query groups")
        query_states = np.asarray(
            store.dataset.stream["queries/state_index"][train_indices], dtype=np.int64
        )
        state_indices = np.unique(query_states)
        state_ids = store.dataset.state_strings("state_id")
        family_ids = store.dataset.state_strings("family_id")
        selected_families = sorted({str(family_ids[index]) for index in state_indices})
        unsupported = set(selected_families) - set(self.descriptor.supported_family_ids)
        if unsupported:
            raise ValueError(f"shared E2 pipeline does not support families {sorted(unsupported)}")

        response_parts = []
        for start in range(0, len(train_indices), 4096):
            response_parts.append(np.asarray(
                store.dataset.stream["responses/mean"][train_indices[start : start + 4096]],
                dtype=np.float64,
            ).reshape(-1, 3))
        response = np.maximum(np.concatenate(response_parts, axis=0), 0.0)
        scale = np.maximum(np.quantile(response, 0.5, axis=0), 1e-8)
        transformed = np.log1p(response / scale)
        mean = np.mean(transformed, axis=0)
        standard_deviation = np.maximum(np.std(transformed, axis=0), 1e-6)
        counts = [int(np.count_nonzero(query_states == index)) for index in state_indices]
        if min(counts) < 1:
            raise RuntimeError("shared E2 fitted state contains an empty material slot")
        return {
            "format_name": "ncls.fitted-training-state",
            "format_version": 3,
            "fit_scope": "final-train-query-groups-only",
            "latent_scope": "target-visible-selected-states",
            "target_transform_id": self.target_transform_id,
            "state_ids": [str(state_ids[index]) for index in state_indices],
            "family_ids": selected_families,
            "train_query_group_count": int(len(train_indices)),
            "train_query_group_count_by_state": counts,
            "target_channel_scale": [float(value) for value in scale],
            "target_channel_mean": [float(value) for value in mean],
            "target_channel_standard_deviation": [
                float(value) for value in standard_deviation
            ],
        }

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        required = {
            "format_name", "format_version", "fit_scope", "latent_scope",
            "target_transform_id", "state_ids", "family_ids",
            "train_query_group_count", "train_query_group_count_by_state",
            "target_channel_scale", "target_channel_mean",
            "target_channel_standard_deviation",
        }
        if set(state) != required:
            raise ValueError("shared E2 fitted training state fields are unsupported")
        if (
            state["format_name"] != "ncls.fitted-training-state"
            or state["format_version"] != 3
            or state["fit_scope"] != "final-train-query-groups-only"
            or state["latent_scope"] != "target-visible-selected-states"
            or state["target_transform_id"] != self.target_transform_id
        ):
            raise ValueError("shared E2 fitted training state contract is unsupported")
        state_ids = list(map(str, state["state_ids"]))
        counts = np.asarray(state["train_query_group_count_by_state"], dtype=np.int64)
        if not state_ids or len(set(state_ids)) != len(state_ids) or counts.shape != (len(state_ids),):
            raise ValueError("shared E2 material slot table is invalid")
        if np.any(counts < 1) or int(np.sum(counts)) != int(state["train_query_group_count"]):
            raise ValueError("shared E2 per-state train query counts are invalid")
        for name, positive in (
            ("target_channel_scale", True),
            ("target_channel_mean", False),
            ("target_channel_standard_deviation", True),
        ):
            values = np.asarray(state[name], dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)) or (
                positive and np.any(values <= 0.0)
            ):
                raise ValueError(f"shared E2 {name} is invalid")
        self._training_state = dict(state)

    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        if self._training_state is None:
            raise RuntimeError("shared E2 fitted training state has not been loaded")
        config = NeuralEvaluatorModelConfig.from_mapping(model_parameters)
        return SharedMaterialNeuralEvaluator(config, len(self._training_state["state_ids"]))

    def _material_slots(
        self,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
    ) -> torch.Tensor:
        if self._training_state is None:
            raise RuntimeError("shared E2 fitted training state has not been loaded")
        dataset_ids = store.dataset.state_strings("state_id")
        slots_by_state = np.full(len(dataset_ids), -1, dtype=np.int64)
        lookup = {str(value): slot for slot, value in enumerate(self._training_state["state_ids"])}
        for state_index, state_id in enumerate(dataset_ids.tolist()):
            slots_by_state[state_index] = lookup.get(str(state_id), -1)
        mapping = torch.as_tensor(
            slots_by_state, dtype=torch.long, device=batch["state_index"].device
        )
        slots = mapping[batch["state_index"].long()]
        if torch.any(slots < 0):
            raise ValueError("shared E2 checkpoint cannot evaluate a state without a fitted latent")
        return slots

    def _transform_tensors(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._training_state is None:
            raise RuntimeError("shared E2 target transform has not been loaded")
        values = []
        for name in (
            "target_channel_scale",
            "target_channel_mean",
            "target_channel_standard_deviation",
        ):
            values.append(torch.as_tensor(
                self._training_state[name], dtype=reference.dtype, device=reference.device
            ))
        return values[0], values[1], values[2]

    def _decode(self, raw: torch.Tensor) -> torch.Tensor:
        scale, mean, standard_deviation = self._transform_tensors(raw)
        log_response = positive_response(raw * standard_deviation + mean)
        return scale * torch.expm1(torch.clamp(log_response, max=20.0))

    def _raw_prediction(
        self,
        model: SharedMaterialNeuralEvaluator,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
    ) -> torch.Tensor:
        return model(
            batch["view"].float(),
            batch["lights"].float(),
            self._material_slots(batch, store),
        )

    def predict(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> torch.Tensor:
        del device
        if not isinstance(model, SharedMaterialNeuralEvaluator):
            raise TypeError("shared E2 pipeline requires SharedMaterialNeuralEvaluator")
        prediction = self._decode(self._raw_prediction(model, batch, store))
        if model.training and isinstance(batch, dict):
            wo = batch["view"].float()
            wi = batch["lights"].float()
            slots = self._material_slots(batch, store)
            group_count, direction_count, _ = wi.shape
            reverse_raw = model(
                wi.reshape(group_count * direction_count, 3),
                wo[:, None, :].expand(-1, direction_count, -1).reshape(
                    group_count * direction_count, 1, 3
                ),
                slots.repeat_interleave(direction_count),
            )
            reverse = self._decode(reverse_raw).reshape(group_count, direction_count, 3)
            wi_cosine = torch.abs(wi[..., 2:3])
            wo_cosine = torch.abs(wo[:, None, 2:3])
            valid = (wi_cosine > 0.05) & (wo_cosine > 0.05)
            scale, _, _ = self._transform_tensors(prediction)
            delta = torch.abs(
                torch.log1p(prediction / torch.clamp(wi_cosine, min=0.05) / scale)
                - torch.log1p(reverse / torch.clamp(wo_cosine, min=0.05) / scale)
            )
            batch["_reciprocity_penalty"] = torch.sum(
                torch.where(valid, delta, torch.zeros_like(delta))
            ) / torch.clamp(torch.sum(valid.expand_as(delta)), min=1)
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
        transform_loss = torch.sum(
            confidence * torch.square(transformed_prediction - transformed_target)
        ) / torch.sum(confidence)
        reciprocity = batch.get("_reciprocity_penalty")
        reciprocity_loss = (
            reciprocity if isinstance(reciprocity, torch.Tensor) else prediction.new_zeros(())
        )
        base_loss = (
            transform_loss
            + 0.02 * response_loss(prediction, target, standard_error)
            + 0.02 * reciprocity_loss
        )
        energy_loss, shape_loss = energy_shape_terms(prediction, batch)
        return 0.25 * base_loss + 0.5 * energy_loss + 2.0 * shape_loss

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
        del device
        if not isinstance(model, SharedMaterialNeuralEvaluator):
            raise TypeError("shared E2 pipeline requires SharedMaterialNeuralEvaluator")
        wo = batch["view"].float()
        wi = batch["lights"].float()
        slots = self._material_slots(batch, store)
        group_count, direction_count, _ = wi.shape
        forward = self._decode(model(wo, wi, slots))
        reverse = self._decode(model(
            wi.reshape(group_count * direction_count, 3),
            wo[:, None, :].expand(-1, direction_count, -1).reshape(
                group_count * direction_count, 1, 3
            ),
            slots.repeat_interleave(direction_count),
        )).reshape(group_count, direction_count, 3)
        wi_cosine = torch.abs(wi[..., 2:3])
        wo_cosine = torch.abs(wo[:, None, 2:3])
        valid = (wi_cosine > 0.05) & (wo_cosine > 0.05)
        forward_bsdf = forward / torch.clamp(wi_cosine, min=0.05)
        reverse_bsdf = reverse / torch.clamp(wo_cosine, min=0.05)
        delta = torch.where(
            valid, torch.abs(forward_bsdf - reverse_bsdf), torch.zeros_like(forward)
        )
        magnitude = torch.where(valid, torch.abs(forward_bsdf), torch.zeros_like(forward))
        reciprocity = torch.sum(delta, dim=(1, 2)) / torch.clamp(
            torch.sum(magnitude, dim=(1, 2)), min=1e-8
        )
        return {"reciprocity_relative_l1": reciprocity.detach().cpu().numpy()}

    def parameter_costs(self, model: nn.Module) -> Mapping[str, Any]:
        if not isinstance(model, SharedMaterialNeuralEvaluator):
            return super().parameter_costs(model)
        return {
            **model.cost_summary(),
            "cost_scope": "shared decoder plus per-material optimized target-visible dense latent",
        }


class AnalyticResidualSharedEvaluatorE2Pipeline(DenseLatentSharedEvaluatorE2Pipeline):
    """LayerStack direct-top analytic core + 共享 neural residual 的 E2 容量上界。"""

    target_transform_id = _RESIDUAL_TARGET_TRANSFORM_ID
    descriptor = LearningPipelineDescriptor(
        pipeline_id=ANALYTIC_RESIDUAL_PIPELINE_ID,
        candidate_id="ncls.analytic-core-neural-residual@1",
        research_role="e2-shared-representation-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.layer-stack-direct-top-adapter@1",
        feature_transform_id="ncls.local-frame-wo-wi-material-slot@1",
        target_transform_id=target_transform_id,
        representation_id="ncls.analytic-direct-top-shared-neural-residual@1",
        architecture_id=SHARED_ARCHITECTURE_ID,
        latent_inference_id="ncls.optimized-target-visible-dense-material-latent-table@1",
        compiler_id="ncls.none-target-visible-capacity-study@1",
        loss_id="ncls.standardized-asinh-residual-energy-shape-reciprocity@1",
        metric_suite_id="ncls.evaluator-quality-suite-by-state-with-core-ablation@1",
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope="multi-material-target-visible-analytic-residual-autodecoder-capacity",
    )

    def open_store(self, dataset_path: str) -> ReferenceQueryStore:
        return LayerStackReferenceStore(dataset_path)

    def _core(
        self,
        batch: Mapping[str, torch.Tensor],
        view: torch.Tensor,
        lights: torch.Tensor,
        *,
        repeat_count: int = 1,
    ) -> torch.Tensor:
        return evaluate_layer_stack_direct_top(
            batch, view, lights, repeat_count=repeat_count
        )

    def fit_training_state(
        self,
        store: ReferenceQueryStore,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        if not isinstance(store, LayerStackReferenceStore):
            raise TypeError("shared analytic residual requires LayerStackReferenceStore")
        base = dict(super().fit_training_state(store, train_indices))
        residual_parts = []
        for start in range(0, len(train_indices), 256):
            raw = store.batch(train_indices[start : start + 256])
            tensor = {name: torch.as_tensor(value) for name, value in raw.items()}
            core = self._core(tensor, tensor["view"].float(), tensor["lights"].float())
            residual_parts.append(
                raw["mean"].astype(np.float64)
                - core.detach().cpu().numpy().astype(np.float64)
            )
        residual = np.concatenate(residual_parts, axis=0).reshape(-1, 3)
        scale = np.empty(3, dtype=np.float64)
        for channel in range(3):
            absolute_nonzero = np.abs(residual[:, channel])
            absolute_nonzero = absolute_nonzero[absolute_nonzero > 0.0]
            scale[channel] = (
                max(float(np.quantile(absolute_nonzero, 0.5)), 1e-8)
                if len(absolute_nonzero)
                else 1e-8
            )
        transformed = np.arcsinh(residual / scale)
        return {
            **base,
            "target_transform_id": self.target_transform_id,
            "target_channel_scale": [float(value) for value in scale],
            "target_channel_mean": [float(value) for value in np.mean(transformed, axis=0)],
            "target_channel_standard_deviation": [
                float(value) for value in np.maximum(np.std(transformed, axis=0), 1e-6)
            ],
        }

    def _decode(self, raw: torch.Tensor) -> torch.Tensor:
        scale, mean, standard_deviation = self._transform_tensors(raw)
        transformed = torch.clamp(raw * standard_deviation + mean, min=-15.0, max=15.0)
        return scale * torch.sinh(transformed)

    @staticmethod
    def _reciprocity_values(
        forward: torch.Tensor,
        reverse: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> torch.Tensor:
        wi_cosine = torch.abs(wi[..., 2:3])
        wo_cosine = torch.abs(wo[:, None, 2:3])
        valid = (wi_cosine > 0.05) & (wo_cosine > 0.05)
        forward_bsdf = forward / torch.clamp(wi_cosine, min=0.05)
        reverse_bsdf = reverse / torch.clamp(wo_cosine, min=0.05)
        delta = torch.where(
            valid, torch.abs(forward_bsdf - reverse_bsdf), torch.zeros_like(forward)
        )
        magnitude = torch.where(valid, torch.abs(forward_bsdf), torch.zeros_like(forward))
        return torch.sum(delta, dim=(1, 2)) / torch.clamp(
            torch.sum(magnitude, dim=(1, 2)), min=1e-8
        )

    def _reverse_prediction(
        self,
        model: SharedMaterialNeuralEvaluator,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
    ) -> torch.Tensor:
        wo = batch["view"].float()
        wi = batch["lights"].float()
        slots = self._material_slots(batch, store)
        group_count, direction_count, _ = wi.shape
        reverse_view = wi.reshape(group_count * direction_count, 3)
        reverse_light = wo[:, None, :].expand(-1, direction_count, -1).reshape(
            group_count * direction_count, 1, 3
        )
        reverse_core = self._core(
            batch, reverse_view, reverse_light, repeat_count=direction_count
        )
        reverse_residual = self._decode(model(
            reverse_view,
            reverse_light,
            slots.repeat_interleave(direction_count),
        ))
        return torch.clamp(reverse_core + reverse_residual, min=0.0).reshape(
            group_count, direction_count, 3
        )

    def predict(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> torch.Tensor:
        del device
        if not isinstance(model, SharedMaterialNeuralEvaluator):
            raise TypeError("shared analytic residual requires SharedMaterialNeuralEvaluator")
        core = self._core(batch, batch["view"].float(), batch["lights"].float())
        residual = self._decode(self._raw_prediction(model, batch, store))
        prediction = torch.clamp(core + residual, min=0.0)
        if isinstance(batch, dict):
            batch["_analytic_core"] = core
            batch["_predicted_residual"] = residual
            if model.training:
                reverse = self._reverse_prediction(model, batch, store)
                reciprocity = self._reciprocity_values(
                    prediction, reverse, batch["view"].float(), batch["lights"].float()
                )
                batch["_reciprocity_penalty"] = torch.mean(torch.log1p(reciprocity))
        return prediction

    def training_loss(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        core = batch.get("_analytic_core")
        predicted_residual = batch.get("_predicted_residual")
        if not isinstance(core, torch.Tensor) or not isinstance(predicted_residual, torch.Tensor):
            raise RuntimeError("shared analytic residual loss requires prediction state")
        target = batch["mean"].float()
        target_residual = target - core
        scale, mean, standard_deviation = self._transform_tensors(prediction)
        transformed_prediction = (
            torch.asinh(predicted_residual / scale) - mean
        ) / standard_deviation
        transformed_target = (
            torch.asinh(target_residual / scale) - mean
        ) / standard_deviation
        transform_loss = torch.mean(torch.square(transformed_prediction - transformed_target))
        reciprocity = batch.get("_reciprocity_penalty")
        reciprocity_loss = (
            reciprocity if isinstance(reciprocity, torch.Tensor) else prediction.new_zeros(())
        )
        base_loss = (
            transform_loss
            + 0.02 * response_loss(prediction, target, batch["standard_error"].float())
            + 0.02 * reciprocity_loss
        )
        energy_loss, shape_loss = energy_shape_terms(prediction, batch)
        return 0.25 * base_loss + 0.5 * energy_loss + 2.0 * shape_loss

    def additional_metric_distributions(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> Mapping[str, np.ndarray]:
        del device
        if not isinstance(model, SharedMaterialNeuralEvaluator):
            raise TypeError("shared analytic residual requires SharedMaterialNeuralEvaluator")
        core = self._core(batch, batch["view"].float(), batch["lights"].float())
        residual = self._decode(self._raw_prediction(model, batch, store))
        forward = torch.clamp(core + residual, min=0.0)
        reverse = self._reverse_prediction(model, batch, store)
        reciprocity = self._reciprocity_values(
            forward, reverse, batch["view"].float(), batch["lights"].float()
        )
        weights = batch["solid_angle_weight"].float()[..., None]
        target = batch["mean"].float()
        core_error = torch.sum(torch.abs(core - target) * weights, dim=(1, 2)) / torch.clamp(
            torch.sum(torch.abs(target) * weights, dim=(1, 2)), min=1e-8
        )
        return {
            "reciprocity_relative_l1": reciprocity.detach().cpu().numpy(),
            "analytic_core_solid_angle_normalized_l1": core_error.detach().cpu().numpy(),
        }
