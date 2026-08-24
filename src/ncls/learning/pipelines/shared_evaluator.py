from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ncls.learning.data import LayerStackReferenceStore, ReferenceQueryStore
from ncls.learning.evaluation.metrics import evaluator_metric_distributions, response_loss
from ncls.learning.losses import energy_shape_terms
from ncls.learning.models.neural_evaluator import (
    FACTORIZED_LATENT_ARCHITECTURE_ID,
    NeuralEvaluatorModelConfig,
    SHARED_ARCHITECTURE_ID,
    SPARSE_DICTIONARY_ARCHITECTURE_ID,
    TARGET_TENSOR_ENCODER_ARCHITECTURE_ID,
    FactorizedMaterialNeuralEvaluator,
    SharedMaterialNeuralEvaluator,
    SparseDictionaryMaterialNeuralEvaluator,
    TargetTensorEncoderMaterialNeuralEvaluator,
    positive_response,
)
from ncls.learning.source_adapters import evaluate_layer_stack_direct_top

from .base import LearningPipeline, LearningPipelineDescriptor


PIPELINE_ID = "dense-latent-shared-small-mlp-energy-shape-e2@1"
ANALYTIC_RESIDUAL_PIPELINE_ID = "analytic-core-shared-neural-residual-energy-shape-e2@1"
PER_STATE_ANALYTIC_RESIDUAL_PIPELINE_ID = (
    "analytic-core-shared-neural-residual-energy-shape-e2@2"
)
SOURCE_AWARE_ANALYTIC_RESIDUAL_PIPELINE_ID = (
    "analytic-core-shared-neural-residual-energy-shape-e2@3"
)
NOISE_AWARE_ANALYTIC_RESIDUAL_PIPELINE_ID = (
    "analytic-core-shared-neural-residual-energy-shape-e2@4"
)
BOUNDARY_CAPACITY_ANALYTIC_RESIDUAL_PIPELINE_ID = (
    "analytic-core-shared-neural-residual-energy-shape-e2@5"
)
SPARSE_DICTIONARY_ANALYTIC_RESIDUAL_PIPELINE_ID = (
    "sparse-latent-dictionary-analytic-residual-e2@1"
)
FACTORIZED_LATENT_ANALYTIC_RESIDUAL_PIPELINE_ID = (
    "factorized-latent-analytic-residual-e2@1"
)
TARGET_TENSOR_ENCODER_ANALYTIC_RESIDUAL_PIPELINE_ID = (
    "target-tensor-encoder-analytic-residual-e2@1"
)
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
_SPARSE_DICTIONARY_FEATURE_CONTRACT = {
    **_FEATURE_CONTRACT,
    "feature_contract_id": "ncls.local-frame-wo-wi-sparse-material-code@1",
    "inputs": {
        "prepare": ["top_k_dictionary_indices_and_weights", "wo"],
        "evaluate": ["prepared_view_code", "wi"],
    },
    "material_addressing": "fitted-target-visible-top-k-dictionary-code",
}
_FACTORIZED_LATENT_FEATURE_CONTRACT = {
    **_FEATURE_CONTRACT,
    "feature_contract_id": "ncls.local-frame-wo-wi-factorized-material-code@1",
    "inputs": {
        "prepare": ["low_rank_material_coefficients", "wo"],
        "evaluate": ["prepared_view_code", "wi"],
    },
    "material_addressing": "fitted-target-visible-low-rank-material-code",
}
_TARGET_TENSOR_ENCODER_FEATURE_CONTRACT = {
    **_FEATURE_CONTRACT,
    "feature_contract_id": "ncls.train-response-tensor-to-local-frame-evaluator@1",
    "inputs": {
        "latent_inference": [
            "train-query-wo",
            "train-query-wi",
            "train-query-analytic-residual",
            "train-query-solid-angle-weight",
        ],
        "prepare": ["target-encoded-material-latent", "wo"],
        "evaluate": ["prepared_view_code", "wi"],
    },
    "material_addressing": "target-encoded-selected-state-slot",
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
    reciprocity_loss_weight = 0.02

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

    def _decode_for_slots(
        self,
        raw: torch.Tensor,
        material_slots: torch.Tensor,
    ) -> torch.Tensor:
        del material_slots
        return self._decode(raw)

    def _loss_transform_tensors(
        self,
        reference: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del batch
        return self._transform_tensors(reference)

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
        reverse_core, reverse_residual = self._reverse_components(model, batch, store)
        return torch.clamp(reverse_core + reverse_residual, min=0.0)

    def _reverse_components(
        self,
        model: SharedMaterialNeuralEvaluator,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        repeated_slots = slots.repeat_interleave(direction_count)
        reverse_residual = self._decode_for_slots(
            model(reverse_view, reverse_light, repeated_slots), repeated_slots
        )
        return (
            reverse_core.reshape(group_count, direction_count, 3),
            reverse_residual.reshape(group_count, direction_count, 3),
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
        slots = self._material_slots(batch, store)
        residual = self._decode_for_slots(
            self._raw_prediction(model, batch, store), slots
        )
        prediction = torch.clamp(core + residual, min=0.0)
        if isinstance(batch, dict):
            batch["_analytic_core"] = core
            batch["_predicted_residual"] = residual
            batch["_material_slots"] = slots
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
        scale, mean, standard_deviation = self._loss_transform_tensors(prediction, batch)
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
            + self.reciprocity_loss_weight * reciprocity_loss
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
        slots = self._material_slots(batch, store)
        residual = self._decode_for_slots(
            self._raw_prediction(model, batch, store), slots
        )
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


class PerStateAnalyticResidualSharedEvaluatorE2Pipeline(
    AnalyticResidualSharedEvaluatorE2Pipeline
):
    """按 state 的 train query 标定 residual；transform 常量计入每材质成本。"""

    target_transform_id = (
        "ncls.train-only-per-state-standardized-asinh-analytic-residual@1"
    )
    descriptor = LearningPipelineDescriptor(
        pipeline_id=PER_STATE_ANALYTIC_RESIDUAL_PIPELINE_ID,
        candidate_id="ncls.analytic-core-neural-residual@1",
        research_role="e2-shared-representation-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.layer-stack-direct-top-adapter@1",
        feature_transform_id="ncls.local-frame-wo-wi-material-slot@1",
        target_transform_id=target_transform_id,
        representation_id=(
            "ncls.analytic-direct-top-shared-neural-residual-per-state-normalization@1"
        ),
        architecture_id=SHARED_ARCHITECTURE_ID,
        latent_inference_id=(
            "ncls.optimized-target-visible-dense-material-latent-table@1"
        ),
        compiler_id="ncls.none-target-visible-capacity-study@1",
        loss_id=(
            "ncls.per-state-standardized-asinh-residual-energy-shape-reciprocity@1"
        ),
        metric_suite_id="ncls.evaluator-quality-suite-by-state-with-core-ablation@1",
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope=(
            "multi-material-target-visible-per-state-normalized-analytic-residual-"
            "autodecoder-capacity"
        ),
    )
    reciprocity_loss_weight = 0.2

    def fit_training_state(
        self,
        store: ReferenceQueryStore,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        if not isinstance(store, LayerStackReferenceStore):
            raise TypeError("per-state shared analytic residual requires LayerStackReferenceStore")
        base = dict(super().fit_training_state(store, train_indices))
        query_states = np.asarray(
            store.dataset.stream["queries/state_index"][train_indices], dtype=np.int64
        )
        dataset_state_ids = store.dataset.state_strings("state_id")
        dataset_index_by_id = {
            str(state_id): index for index, state_id in enumerate(dataset_state_ids)
        }
        scales: list[list[float]] = []
        means: list[list[float]] = []
        standard_deviations: list[list[float]] = []
        for state_id in base["state_ids"]:
            state_index = dataset_index_by_id[str(state_id)]
            selected = train_indices[query_states == state_index]
            raw = store.batch(selected)
            tensor = {name: torch.as_tensor(value) for name, value in raw.items()}
            core = self._core(tensor, tensor["view"].float(), tensor["lights"].float())
            residual = (
                raw["mean"].astype(np.float64)
                - core.detach().cpu().numpy().astype(np.float64)
            ).reshape(-1, 3)
            state_scale = np.empty(3, dtype=np.float64)
            for channel in range(3):
                absolute_nonzero = np.abs(residual[:, channel])
                absolute_nonzero = absolute_nonzero[absolute_nonzero > 0.0]
                state_scale[channel] = (
                    max(float(np.quantile(absolute_nonzero, 0.5)), 1e-8)
                    if len(absolute_nonzero)
                    else 1e-8
                )
            transformed = np.arcsinh(residual / state_scale)
            scales.append([float(value) for value in state_scale])
            means.append([float(value) for value in np.mean(transformed, axis=0)])
            standard_deviations.append([
                float(value)
                for value in np.maximum(np.std(transformed, axis=0), 1e-6)
            ])
        return {
            **base,
            "format_version": 4,
            "target_channel_scale_by_state": scales,
            "target_channel_mean_by_state": means,
            "target_channel_standard_deviation_by_state": standard_deviations,
            "target_channel_scale": None,
            "target_channel_mean": None,
            "target_channel_standard_deviation": None,
        }

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        required = {
            "format_name", "format_version", "fit_scope", "latent_scope",
            "target_transform_id", "state_ids", "family_ids",
            "train_query_group_count", "train_query_group_count_by_state",
            "target_channel_scale", "target_channel_mean",
            "target_channel_standard_deviation", "target_channel_scale_by_state",
            "target_channel_mean_by_state",
            "target_channel_standard_deviation_by_state",
        }
        if set(state) != required:
            raise ValueError("per-state shared E2 fitted training state fields are unsupported")
        state_ids = list(map(str, state["state_ids"]))
        counts = np.asarray(state["train_query_group_count_by_state"], dtype=np.int64)
        if (
            state["format_name"] != "ncls.fitted-training-state"
            or state["format_version"] != 4
            or state["fit_scope"] != "final-train-query-groups-only"
            or state["latent_scope"] != "target-visible-selected-states"
            or state["target_transform_id"] != self.target_transform_id
            or any(state[name] is not None for name in (
                "target_channel_scale", "target_channel_mean",
                "target_channel_standard_deviation",
            ))
            or not state_ids
            or len(set(state_ids)) != len(state_ids)
            or counts.shape != (len(state_ids),)
            or np.any(counts < 1)
            or int(np.sum(counts)) != int(state["train_query_group_count"])
        ):
            raise ValueError("per-state shared E2 fitted training state contract is unsupported")
        for name, positive in (
            ("target_channel_scale_by_state", True),
            ("target_channel_mean_by_state", False),
            ("target_channel_standard_deviation_by_state", True),
        ):
            values = np.asarray(state[name], dtype=np.float64)
            if (
                values.shape != (len(state_ids), 3)
                or not np.all(np.isfinite(values))
                or (positive and np.any(values <= 0.0))
            ):
                raise ValueError(f"per-state shared E2 {name} is invalid")
        self._training_state = dict(state)

    def _transform_tensors_for_slots(
        self,
        reference: torch.Tensor,
        material_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._training_state is None:
            raise RuntimeError("per-state shared E2 target transform has not been loaded")
        tensors = []
        for name in (
            "target_channel_scale_by_state",
            "target_channel_mean_by_state",
            "target_channel_standard_deviation_by_state",
        ):
            table = torch.as_tensor(
                self._training_state[name], dtype=reference.dtype, device=reference.device
            )
            tensors.append(table[material_slots.long()][:, None, :])
        return tensors[0], tensors[1], tensors[2]

    def _decode_for_slots(
        self,
        raw: torch.Tensor,
        material_slots: torch.Tensor,
    ) -> torch.Tensor:
        scale, mean, standard_deviation = self._transform_tensors_for_slots(
            raw, material_slots
        )
        transformed = torch.clamp(
            raw * standard_deviation + mean, min=-15.0, max=15.0
        )
        return scale * torch.sinh(transformed)

    def _loss_transform_tensors(
        self,
        reference: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = batch.get("_material_slots")
        if not isinstance(slots, torch.Tensor):
            raise RuntimeError("per-state shared E2 loss requires material slots")
        return self._transform_tensors_for_slots(reference, slots)

    def parameter_costs(self, model: nn.Module) -> Mapping[str, Any]:
        costs = dict(super().parameter_costs(model))
        transform_bytes = 9 * 4
        material_count = int(costs.get("material_count", 0))
        costs["B_asset_target_transform_fp32"] = transform_bytes
        costs["B_asset_fp32"] = int(costs["B_asset_fp32"]) + transform_bytes
        costs["B_asset_fp32_total"] = (
            int(costs["B_asset_fp32_total"]) + material_count * transform_bytes
        )
        costs["cost_scope"] = (
            "shared decoder plus per-material optimized dense latent and train-only "
            "residual transform"
        )
        return costs


class SourceAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline(
    PerStateAnalyticResidualSharedEvaluatorE2Pipeline
):
    """保留 source 固有非互易项，并度量 evaluator 对该项的额外偏差。"""

    descriptor = LearningPipelineDescriptor(
        pipeline_id=SOURCE_AWARE_ANALYTIC_RESIDUAL_PIPELINE_ID,
        candidate_id="ncls.analytic-core-neural-residual@1",
        research_role="e2-shared-representation-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.layer-stack-direct-top-adapter@1",
        feature_transform_id="ncls.local-frame-wo-wi-material-slot@1",
        target_transform_id=(
            "ncls.train-only-per-state-standardized-asinh-analytic-residual@1"
        ),
        representation_id=(
            "ncls.analytic-direct-top-shared-neural-residual-per-state-normalization@1"
        ),
        architecture_id=SHARED_ARCHITECTURE_ID,
        latent_inference_id="ncls.optimized-target-visible-dense-material-latent-table@1",
        compiler_id="ncls.none-target-visible-capacity-study@1",
        loss_id=(
            "ncls.per-state-standardized-asinh-residual-energy-shape-reciprocity@1"
        ),
        metric_suite_id=(
            "ncls.evaluator-quality-suite-by-state-with-source-reciprocity@1"
        ),
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope=(
            "multi-material-target-visible-source-aware-per-state-normalized-"
            "analytic-residual-autodecoder-capacity"
        ),
    )

    @staticmethod
    def _source_reciprocity_deviation(
        forward: torch.Tensor,
        reverse: torch.Tensor,
        core_forward: torch.Tensor,
        core_reverse: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
        source_asymmetry_mask: torch.Tensor,
    ) -> torch.Tensor:
        wi_cosine = torch.abs(wi[..., 2:3])
        wo_cosine = torch.abs(wo[:, None, 2:3])
        valid = (wi_cosine > 0.05) & (wo_cosine > 0.05)
        forward_bsdf = forward / torch.clamp(wi_cosine, min=0.05)
        reverse_bsdf = reverse / torch.clamp(wo_cosine, min=0.05)
        predicted_asymmetry = forward_bsdf - reverse_bsdf
        core_asymmetry = (
            core_forward / torch.clamp(wi_cosine, min=0.05)
            - core_reverse / torch.clamp(wo_cosine, min=0.05)
        )
        expected_asymmetry = torch.where(
            source_asymmetry_mask[:, None, None],
            core_asymmetry,
            torch.zeros_like(core_asymmetry),
        )
        delta = torch.where(
            valid,
            torch.abs(predicted_asymmetry - expected_asymmetry),
            torch.zeros_like(predicted_asymmetry),
        )
        magnitude = torch.where(
            valid, torch.abs(forward_bsdf), torch.zeros_like(forward_bsdf)
        )
        return torch.sum(delta, dim=(1, 2)) / torch.clamp(
            torch.sum(magnitude, dim=(1, 2)), min=1e-8
        )

    def additional_metric_distributions(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> Mapping[str, np.ndarray]:
        metrics = dict(super().additional_metric_distributions(
            model, batch, store, device
        ))
        if not isinstance(model, SharedMaterialNeuralEvaluator):
            raise TypeError("source-aware shared residual requires SharedMaterialNeuralEvaluator")
        core_forward = self._core(
            batch, batch["view"].float(), batch["lights"].float()
        )
        slots = self._material_slots(batch, store)
        residual = self._decode_for_slots(
            self._raw_prediction(model, batch, store), slots
        )
        forward = torch.clamp(core_forward + residual, min=0.0)
        core_reverse, reverse_residual = self._reverse_components(model, batch, store)
        reverse = torch.clamp(core_reverse + reverse_residual, min=0.0)
        single_sheen = (
            (batch["interface_counts"].long() == 1)
            & (batch["top_kind"].long() == 3)
        )
        deviation = self._source_reciprocity_deviation(
            forward,
            reverse,
            core_forward,
            core_reverse,
            batch["view"].float(),
            batch["lights"].float(),
            single_sheen,
        )
        metrics["source_reciprocity_deviation_relative_l1"] = (
            deviation.detach().cpu().numpy()
        )
        return metrics


class NoiseAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline(
    SourceAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline
):
    """用 reference SE floor 约束长尾，并以峰支持集区分平台内 argmax 抖动。"""

    descriptor = LearningPipelineDescriptor(
        pipeline_id=NOISE_AWARE_ANALYTIC_RESIDUAL_PIPELINE_ID,
        candidate_id="ncls.analytic-core-neural-residual@1",
        research_role="e2-shared-representation-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.layer-stack-direct-top-adapter@1",
        feature_transform_id="ncls.local-frame-wo-wi-material-slot@1",
        target_transform_id=(
            "ncls.train-only-per-state-standardized-asinh-analytic-residual@1"
        ),
        representation_id=(
            "ncls.analytic-direct-top-shared-neural-residual-per-state-normalization@1"
        ),
        architecture_id=SHARED_ARCHITECTURE_ID,
        latent_inference_id="ncls.optimized-target-visible-dense-material-latent-table@1",
        compiler_id="ncls.none-target-visible-capacity-study@1",
        loss_id=(
            "ncls.per-state-asinh-energy-shape-source-reciprocity-noise-floor@1"
        ),
        metric_suite_id=(
            "ncls.evaluator-quality-by-state-source-reciprocity-peak-support@1"
        ),
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope=(
            "multi-material-target-visible-noise-aware-per-state-normalized-"
            "analytic-residual-autodecoder-capacity"
        ),
    )

    def training_loss(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        base_loss = super().training_loss(prediction, batch)
        target = batch["mean"].float()
        standard_error = batch["standard_error"].float()
        peak = torch.amax(torch.abs(target), dim=(1, 2), keepdim=True)
        noise_floor = standard_error + 0.002 * peak + 1e-6
        error_over_floor = torch.abs(prediction - target) / noise_floor
        return base_loss + 0.02 * torch.mean(torch.log1p(error_over_floor))

    @staticmethod
    def _peak_support_angle(
        prediction: torch.Tensor,
        target: torch.Tensor,
        light_directions: torch.Tensor,
    ) -> torch.Tensor:
        target_magnitude = torch.sum(torch.abs(target), dim=-1)
        prediction_magnitude = torch.sum(torch.abs(prediction), dim=-1)
        prediction_peak = torch.argmax(prediction_magnitude, dim=1)
        rows = torch.arange(len(target), device=target.device)
        prediction_peak_direction = light_directions[rows, prediction_peak]
        target_peak = torch.amax(target_magnitude, dim=1, keepdim=True)
        support = target_magnitude >= 0.95 * target_peak
        angles = torch.rad2deg(torch.acos(torch.clamp(
            torch.sum(
                light_directions * prediction_peak_direction[:, None, :], dim=-1
            ),
            -1.0,
            1.0,
        )))
        return torch.amin(
            torch.where(support, angles, torch.full_like(angles, float("inf"))),
            dim=1,
        )

    def additional_metric_distributions(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> Mapping[str, np.ndarray]:
        metrics = dict(super().additional_metric_distributions(
            model, batch, store, device
        ))
        if not isinstance(model, SharedMaterialNeuralEvaluator):
            raise TypeError("noise-aware shared residual requires SharedMaterialNeuralEvaluator")
        core = self._core(batch, batch["view"].float(), batch["lights"].float())
        slots = self._material_slots(batch, store)
        residual = self._decode_for_slots(
            self._raw_prediction(model, batch, store), slots
        )
        prediction = torch.clamp(core + residual, min=0.0)
        peak_support_angle = self._peak_support_angle(
            prediction, batch["mean"].float(), batch["lights"].float()
        )
        metrics["peak_support_angle_degrees"] = (
            peak_support_angle.detach().cpu().numpy()
        )
        return metrics


class BoundaryCapacityPerStateAnalyticResidualSharedEvaluatorE2Pipeline(
    NoiseAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline
):
    """保留峰支持 metric，但回到已验证的 base loss 以测试 decoder 容量边界。"""

    descriptor = LearningPipelineDescriptor(
        pipeline_id=BOUNDARY_CAPACITY_ANALYTIC_RESIDUAL_PIPELINE_ID,
        candidate_id="ncls.analytic-core-neural-residual@1",
        research_role="e2-shared-representation-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.layer-stack-direct-top-adapter@1",
        feature_transform_id="ncls.local-frame-wo-wi-material-slot@1",
        target_transform_id=(
            "ncls.train-only-per-state-standardized-asinh-analytic-residual@1"
        ),
        representation_id=(
            "ncls.analytic-direct-top-shared-neural-residual-per-state-normalization@1"
        ),
        architecture_id=SHARED_ARCHITECTURE_ID,
        latent_inference_id="ncls.optimized-target-visible-dense-material-latent-table@1",
        compiler_id="ncls.none-target-visible-capacity-study@1",
        loss_id=(
            "ncls.per-state-standardized-asinh-residual-energy-shape-reciprocity@1"
        ),
        metric_suite_id=(
            "ncls.evaluator-quality-by-state-source-reciprocity-peak-support@1"
        ),
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope=(
            "multi-material-target-visible-boundary-capacity-per-state-normalized-"
            "analytic-residual-autodecoder"
        ),
    )

    def training_loss(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return SourceAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline.training_loss(
            self, prediction, batch
        )


class SparseDictionaryAnalyticResidualSharedEvaluatorE2Pipeline(
    BoundaryCapacityPerStateAnalyticResidualSharedEvaluatorE2Pipeline
):
    """E2 top-k latent dictionary：共享 codebook，材质资产只保留局部 ID/权重。"""

    feature_contract = _SPARSE_DICTIONARY_FEATURE_CONTRACT
    descriptor = LearningPipelineDescriptor(
        pipeline_id=SPARSE_DICTIONARY_ANALYTIC_RESIDUAL_PIPELINE_ID,
        candidate_id="ncls.sparse-latent-dictionary-top-k-mixture@1",
        research_role="e2-shared-representation-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.layer-stack-direct-top-adapter@1",
        feature_transform_id=(
            "ncls.local-frame-wo-wi-sparse-material-code@1"
        ),
        target_transform_id=(
            "ncls.train-only-per-state-standardized-asinh-analytic-residual@1"
        ),
        representation_id=(
            "ncls.analytic-direct-top-shared-neural-residual-top-k-latent-dictionary@1"
        ),
        architecture_id=SPARSE_DICTIONARY_ARCHITECTURE_ID,
        latent_inference_id=(
            "ncls.optimized-target-visible-top-k-dictionary-coefficients@1"
        ),
        compiler_id="ncls.none-target-visible-capacity-study@1",
        loss_id=(
            "ncls.per-state-standardized-asinh-residual-energy-shape-reciprocity@1"
        ),
        metric_suite_id=(
            "ncls.evaluator-quality-by-state-source-reciprocity-peak-support@1"
        ),
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope=(
            "multi-material-target-visible-top-k-dictionary-analytic-residual-"
            "capacity"
        ),
    )

    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        if self._training_state is None:
            raise RuntimeError("sparse dictionary fitted training state has not been loaded")
        parameters = dict(model_parameters)
        try:
            dictionary_size = int(parameters.pop("dictionary_size"))
            top_k = int(parameters.pop("top_k"))
        except KeyError as error:
            raise ValueError(
                "sparse dictionary model requires dictionary_size and top_k"
            ) from error
        config = NeuralEvaluatorModelConfig.from_mapping(parameters)
        return SparseDictionaryMaterialNeuralEvaluator(
            config,
            len(self._training_state["state_ids"]),
            dictionary_size=dictionary_size,
            top_k=top_k,
        )

    def parameter_costs(self, model: nn.Module) -> Mapping[str, Any]:
        costs = dict(super().parameter_costs(model))
        costs["cost_scope"] = (
            "shared dictionary/decoder plus per-material uint16 top-k IDs, fp32 "
            "weights and train-only residual transform"
        )
        return costs


class FactorizedLatentAnalyticResidualSharedEvaluatorE2Pipeline(
    BoundaryCapacityPerStateAnalyticResidualSharedEvaluatorE2Pipeline
):
    """E2 factorized latent：材质系数乘共享 basis 后进入同一 decoder。"""

    feature_contract = _FACTORIZED_LATENT_FEATURE_CONTRACT
    descriptor = LearningPipelineDescriptor(
        pipeline_id=FACTORIZED_LATENT_ANALYTIC_RESIDUAL_PIPELINE_ID,
        candidate_id="ncls.plane-tensor-factorization@1",
        research_role="e2-shared-representation-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.layer-stack-direct-top-adapter@1",
        feature_transform_id=(
            "ncls.local-frame-wo-wi-factorized-material-code@1"
        ),
        target_transform_id=(
            "ncls.train-only-per-state-standardized-asinh-analytic-residual@1"
        ),
        representation_id=(
            "ncls.analytic-direct-top-shared-neural-residual-low-rank-material-latent@1"
        ),
        architecture_id=FACTORIZED_LATENT_ARCHITECTURE_ID,
        latent_inference_id=(
            "ncls.optimized-target-visible-low-rank-material-coefficients@1"
        ),
        compiler_id="ncls.none-target-visible-capacity-study@1",
        loss_id=(
            "ncls.per-state-standardized-asinh-residual-energy-shape-reciprocity@1"
        ),
        metric_suite_id=(
            "ncls.evaluator-quality-by-state-source-reciprocity-peak-support@1"
        ),
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope=(
            "multi-material-target-visible-low-rank-latent-analytic-residual-capacity"
        ),
    )

    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        if self._training_state is None:
            raise RuntimeError("factorized latent fitted training state has not been loaded")
        parameters = dict(model_parameters)
        try:
            factor_rank = int(parameters.pop("factor_rank"))
        except KeyError as error:
            raise ValueError("factorized latent model requires factor_rank") from error
        config = NeuralEvaluatorModelConfig.from_mapping(parameters)
        return FactorizedMaterialNeuralEvaluator(
            config,
            len(self._training_state["state_ids"]),
            factor_rank=factor_rank,
        )

    def parameter_costs(self, model: nn.Module) -> Mapping[str, Any]:
        costs = dict(super().parameter_costs(model))
        costs["cost_scope"] = (
            "shared latent basis/decoder plus per-material low-rank coefficients "
            "and train-only residual transform"
        )
        return costs


class TargetTensorEncoderAnalyticResidualSharedEvaluatorE2Pipeline(
    BoundaryCapacityPerStateAnalyticResidualSharedEvaluatorE2Pipeline
):
    """只读取 train query tensor 的 permutation-invariant target encoder 上界。"""

    feature_contract = _TARGET_TENSOR_ENCODER_FEATURE_CONTRACT
    target_encoder_input_id = (
        "ncls.train-only-permutation-invariant-response-residual-points@1"
    )
    descriptor = LearningPipelineDescriptor(
        pipeline_id=TARGET_TENSOR_ENCODER_ANALYTIC_RESIDUAL_PIPELINE_ID,
        candidate_id="ncls.target-tensor-encoder-shared-decoder@1",
        research_role="e2-shared-representation-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.layer-stack-direct-top-adapter@1",
        feature_transform_id=(
            "ncls.train-response-tensor-to-local-frame-evaluator@1"
        ),
        target_transform_id=(
            "ncls.train-only-per-state-standardized-asinh-analytic-residual@1"
        ),
        representation_id=(
            "ncls.analytic-direct-top-target-encoded-shared-neural-residual@1"
        ),
        architecture_id=TARGET_TENSOR_ENCODER_ARCHITECTURE_ID,
        latent_inference_id="ncls.train-only-target-response-tensor-encoder@1",
        compiler_id="ncls.none-target-visible-response-compression@1",
        loss_id=(
            "ncls.per-state-standardized-asinh-residual-energy-shape-reciprocity@1"
        ),
        metric_suite_id=(
            "ncls.evaluator-quality-by-state-source-reciprocity-peak-support@1"
        ),
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope=(
            "multi-material-train-response-tensor-encoded-analytic-residual-capacity"
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._encoder_store: ReferenceQueryStore | None = None
        self._target_encoder_input: np.ndarray | None = None

    def open_store(self, dataset_path: str) -> ReferenceQueryStore:
        store = super().open_store(dataset_path)
        self._encoder_store = store
        return store

    @staticmethod
    def _target_encoder_input_sha256(value: np.ndarray) -> str:
        array = np.asarray(value, dtype="<f4", order="C")
        digest = hashlib.sha256()
        digest.update(
            TargetTensorEncoderAnalyticResidualSharedEvaluatorE2Pipeline
            .target_encoder_input_id.encode("utf-8")
        )
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
        return digest.hexdigest()

    def _build_target_encoder_input(
        self,
        store: ReferenceQueryStore,
        state: Mapping[str, Any],
        train_indices: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(store, LayerStackReferenceStore):
            raise TypeError("target tensor encoder requires LayerStackReferenceStore")
        query_states = np.asarray(
            store.dataset.stream["queries/state_index"][train_indices], dtype=np.int64
        )
        dataset_state_ids = store.dataset.state_strings("state_id")
        dataset_index_by_id = {
            str(state_id): index for index, state_id in enumerate(dataset_state_ids)
        }
        scales = np.asarray(state["target_channel_scale_by_state"], dtype=np.float64)
        means = np.asarray(state["target_channel_mean_by_state"], dtype=np.float64)
        standard_deviations = np.asarray(
            state["target_channel_standard_deviation_by_state"], dtype=np.float64
        )
        expected_counts = np.asarray(
            state["train_query_group_count_by_state"], dtype=np.int64
        )
        material_inputs = []
        for slot, state_id in enumerate(state["state_ids"]):
            state_index = dataset_index_by_id[str(state_id)]
            selected = train_indices[query_states == state_index]
            if len(selected) != int(expected_counts[slot]):
                raise ValueError("target encoder train query count does not match fitted state")
            raw = store.batch(selected)
            tensor = {name: torch.as_tensor(value) for name, value in raw.items()}
            core = self._core(
                tensor, tensor["view"].float(), tensor["lights"].float()
            ).detach().cpu().numpy().astype(np.float64)
            residual = raw["mean"].astype(np.float64) - core
            transformed_residual = (
                np.arcsinh(residual / scales[slot][None, None, :])
                - means[slot][None, None, :]
            ) / standard_deviations[slot][None, None, :]
            direction_count = raw["lights"].shape[1]
            view = np.broadcast_to(
                raw["view"][:, None, :],
                (len(selected), direction_count, 3),
            )
            weight = np.abs(raw["solid_angle_weight"].astype(np.float64))
            weight_scale = max(float(np.median(weight[weight > 0.0])), 1e-12)
            log_weight = np.log1p(weight / weight_scale)[..., None]
            features = np.concatenate(
                (
                    view.astype(np.float64),
                    raw["lights"].astype(np.float64),
                    transformed_residual,
                    log_weight,
                ),
                axis=-1,
            )
            material_inputs.append(features.reshape(-1, features.shape[-1]))
        point_counts = {len(value) for value in material_inputs}
        if len(point_counts) != 1:
            raise ValueError("target encoder requires a fixed train tensor shape per state")
        result = np.asarray(material_inputs, dtype=np.float32)
        if result.ndim != 3 or result.shape[2] != 10 or not np.all(np.isfinite(result)):
            raise ValueError("target encoder input tensor is invalid")
        return result

    def fit_training_state(
        self,
        store: ReferenceQueryStore,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        base = dict(super().fit_training_state(store, train_indices))
        target_encoder_input = self._build_target_encoder_input(
            store, base, train_indices
        )
        return {
            **base,
            "format_version": 5,
            "target_encoder_input_id": self.target_encoder_input_id,
            "target_encoder_input_query_role": "train",
            "target_encoder_input_shape": list(target_encoder_input.shape),
            "target_encoder_input_sha256": self._target_encoder_input_sha256(
                target_encoder_input
            ),
        }

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        extra_fields = {
            "target_encoder_input_id",
            "target_encoder_input_query_role",
            "target_encoder_input_shape",
            "target_encoder_input_sha256",
        }
        if not extra_fields.issubset(state) or state.get("format_version") != 5:
            raise ValueError("target encoder fitted training state fields are unsupported")
        if (
            state["target_encoder_input_id"] != self.target_encoder_input_id
            or state["target_encoder_input_query_role"] != "train"
        ):
            raise ValueError("target encoder input contract is unsupported")
        base = {name: value for name, value in state.items() if name not in extra_fields}
        base["format_version"] = 4
        super().load_training_state(base)
        if self._encoder_store is None:
            raise RuntimeError("target encoder store must be opened before fitted state load")
        all_train_indices = self.lifecycle_indices(self._encoder_store, "train")
        dataset_state_ids = self._encoder_store.dataset.state_strings("state_id")
        selected_state_ids = set(map(str, state["state_ids"]))
        query_states = np.asarray(
            self._encoder_store.dataset.stream["queries/state_index"][all_train_indices],
            dtype=np.int64,
        )
        selected = all_train_indices[np.asarray([
            str(dataset_state_ids[state_index]) in selected_state_ids
            for state_index in query_states
        ])]
        target_encoder_input = self._build_target_encoder_input(
            self._encoder_store, state, selected
        )
        if (
            list(target_encoder_input.shape) != list(state["target_encoder_input_shape"])
            or self._target_encoder_input_sha256(target_encoder_input)
            != state["target_encoder_input_sha256"]
        ):
            raise ValueError("target encoder input hash does not match the train-only H5 tensor")
        self._training_state = dict(state)
        self._target_encoder_input = target_encoder_input

    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        if self._training_state is None or self._target_encoder_input is None:
            raise RuntimeError("target encoder fitted training state has not been loaded")
        parameters = dict(model_parameters)
        try:
            encoder_width = int(parameters.pop("encoder_width"))
            encoder_layer_count = int(parameters.pop("encoder_layer_count"))
        except KeyError as error:
            raise ValueError(
                "target tensor encoder requires encoder_width and encoder_layer_count"
            ) from error
        config = NeuralEvaluatorModelConfig.from_mapping(parameters)
        return TargetTensorEncoderMaterialNeuralEvaluator(
            config,
            torch.from_numpy(self._target_encoder_input),
            encoder_width=encoder_width,
            encoder_layer_count=encoder_layer_count,
        )

    def parameter_costs(self, model: nn.Module) -> Mapping[str, Any]:
        costs = dict(super().parameter_costs(model))
        costs["cost_scope"] = (
            "runtime shared decoder plus baked per-material target-encoded latent and "
            "train-only residual transform; target encoder/input are reported separately "
            "as compression cost"
        )
        return costs
