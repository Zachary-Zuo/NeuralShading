from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ncls.data import SPLIT_NAMES
from ncls.learning.data import LayerStackReferenceStore, ReferenceQueryStore
from ncls.learning.models.neural_evaluator import (
    LAYER_STACK_SOURCE_COMPILER_ARCHITECTURE_ID,
    LayerStackSourceCompilerNeuralEvaluator,
    NeuralEvaluatorModelConfig,
)
from ncls.learning.source_adapters import (
    LAYER_STACK_SOURCE_ADAPTER_ID,
    LAYER_STACK_SOURCE_FEATURE_CONTRACT,
    LAYER_STACK_SOURCE_FEATURE_CONTRACT_ID,
    evaluate_layer_stack_direct_top,
    layer_stack_source_tensors,
    repeat_layer_stack_source_tensors,
)

from .base import LearningPipelineDescriptor
from .shared_evaluator import (
    AnalyticResidualSharedEvaluatorE2Pipeline,
    NoiseAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline,
    SourceAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline,
)


PIPELINE_ID = "layer-stack-source-state-compiler-analytic-residual-e3@1"
TARGET_TRANSFORM_ID = "ncls.source-compiled-per-state-asinh-analytic-residual@1"
_FEATURE_CONTRACT = {
    "format_name": "ncls.feature-contract",
    "format_version": 1,
    "feature_contract_id": (
        "ncls.layer-stack-source-compiled-latent-transform-evaluator@1"
    ),
    "source": LAYER_STACK_SOURCE_FEATURE_CONTRACT,
    "compiler_output": [
        "material_latent",
        "residual_channel_scale",
        "residual_channel_mean",
        "residual_channel_standard_deviation",
    ],
    "prepare": ["compiled_material_state", "wo"],
    "evaluate": ["prepared_view_code", "wi"],
    "runtime_compiler": False,
    "direction_space": "source-reference-local-frame",
}


def _residual_transform_statistics(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flattened = np.asarray(residual, dtype=np.float64).reshape(-1, 3)
    scale = np.empty(3, dtype=np.float64)
    for channel in range(3):
        absolute_nonzero = np.abs(flattened[:, channel])
        absolute_nonzero = absolute_nonzero[absolute_nonzero > 0.0]
        scale[channel] = (
            max(float(np.quantile(absolute_nonzero, 0.5)), 1e-8)
            if len(absolute_nonzero)
            else 1e-8
        )
    transformed = np.arcsinh(flattened / scale)
    mean = np.mean(transformed, axis=0)
    standard_deviation = np.maximum(np.std(transformed, axis=0), 1e-6)
    return scale, mean, standard_deviation


class LayerStackSourceCompilerAnalyticResidualE3Pipeline(
    AnalyticResidualSharedEvaluatorE2Pipeline
):
    """LayerStack 原生 token → latent/transform → shared evaluator 的 E3 路径。"""

    target_transform_id = TARGET_TRANSFORM_ID
    feature_contract = _FEATURE_CONTRACT
    descriptor = LearningPipelineDescriptor(
        pipeline_id=PIPELINE_ID,
        candidate_id="ncls.source-state-compiler-shared-decoder@1",
        research_role="e3-source-compiler-generalization",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.source-state-and-query-role@1",
        source_adapter_id=LAYER_STACK_SOURCE_ADAPTER_ID,
        feature_transform_id=_FEATURE_CONTRACT["feature_contract_id"],
        target_transform_id=TARGET_TRANSFORM_ID,
        representation_id=(
            "ncls.analytic-direct-top-source-compiled-latent-neural-residual@1"
        ),
        architecture_id=LAYER_STACK_SOURCE_COMPILER_ARCHITECTURE_ID,
        latent_inference_id="ncls.pure-feed-forward-source-state-compiler@1",
        compiler_id="ncls.layer-stack-token-gru-latent-transform-compiler@1",
        loss_id=(
            "ncls.source-compiled-asinh-residual-energy-shape-source-reciprocity@1"
        ),
        metric_suite_id=(
            "ncls.evaluator-quality-by-source-split-source-reciprocity-peak-support@1"
        ),
        exporter_id="ncls.neural-evaluator-method-bundle-planned@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope="unseen-source-state-family-topology-feed-forward-compilation",
    )
    transform_supervision_weight = 0.02

    def __init__(self) -> None:
        self._training_state: dict[str, Any] | None = None
        self._transform_targets_by_state_id: dict[str, Mapping[str, Any]] = {}

    def open_store(self, dataset_path: str) -> ReferenceQueryStore:
        return LayerStackReferenceStore(dataset_path)

    def evaluation_indices(
        self,
        store: ReferenceQueryStore,
        evaluation_role: str,
    ) -> np.ndarray:
        if evaluation_role == "adversarial_probe":
            return store.dataset.group_indices(
                source_split="test", query_role="adversarial_probe"
            )
        return super().evaluation_indices(store, evaluation_role)

    def fit_training_state(
        self,
        store: ReferenceQueryStore,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        if not isinstance(store, LayerStackReferenceStore) or not len(train_indices):
            raise ValueError("E3 LayerStack compiler requires nonempty source-train queries")
        query_states = np.asarray(
            store.dataset.stream["queries/state_index"][train_indices], dtype=np.int64
        )
        state_indices = np.unique(query_states)
        state_ids = store.dataset.state_strings("state_id")
        family_ids = store.dataset.state_strings("family_id")
        source_splits = store.dataset.state_splits
        if np.any(source_splits[state_indices] != SPLIT_NAMES.index("train")):
            raise ValueError("E3 source compiler transform fitting crossed the source train split")
        residual_parts: list[np.ndarray] = []
        targets: list[dict[str, Any]] = []
        counts: list[int] = []
        for state_index in state_indices:
            selected = train_indices[query_states == state_index]
            raw = store.batch(selected)
            tensor = {name: torch.as_tensor(value) for name, value in raw.items()}
            core = evaluate_layer_stack_direct_top(
                tensor, tensor["view"].float(), tensor["lights"].float()
            )
            residual = (
                raw["mean"].astype(np.float64)
                - core.detach().cpu().numpy().astype(np.float64)
            )
            residual_parts.append(residual)
            scale, mean, standard_deviation = _residual_transform_statistics(residual)
            targets.append({
                "state_id": str(state_ids[state_index]),
                "scale": [float(value) for value in scale],
                "mean": [float(value) for value in mean],
                "standard_deviation": [float(value) for value in standard_deviation],
            })
            counts.append(int(len(selected)))
        global_scale, global_mean, global_standard_deviation = (
            _residual_transform_statistics(np.concatenate(residual_parts, axis=0))
        )
        return {
            "format_name": "ncls.fitted-training-state",
            "format_version": 1,
            "fit_scope": "source-train-states-and-train-query-groups-only",
            "latent_scope": "feed-forward-unseen-source-state-compilation",
            "target_transform_id": self.target_transform_id,
            "source_feature_contract_id": LAYER_STACK_SOURCE_FEATURE_CONTRACT_ID,
            "state_ids": [str(state_ids[index]) for index in state_indices],
            "family_ids": sorted({str(family_ids[index]) for index in state_indices}),
            "source_train_state_count": int(len(state_indices)),
            "train_query_group_count": int(len(train_indices)),
            "train_query_group_count_by_state": counts,
            "target_channel_scale": [float(value) for value in global_scale],
            "target_channel_mean": [float(value) for value in global_mean],
            "target_channel_standard_deviation": [
                float(value) for value in global_standard_deviation
            ],
            "target_transform_supervision_by_state": targets,
        }

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        required = {
            "format_name",
            "format_version",
            "fit_scope",
            "latent_scope",
            "target_transform_id",
            "source_feature_contract_id",
            "state_ids",
            "family_ids",
            "source_train_state_count",
            "train_query_group_count",
            "train_query_group_count_by_state",
            "target_channel_scale",
            "target_channel_mean",
            "target_channel_standard_deviation",
            "target_transform_supervision_by_state",
        }
        if set(state) != required:
            raise ValueError("E3 source compiler fitted state fields are unsupported")
        state_ids = list(map(str, state["state_ids"]))
        counts = np.asarray(state["train_query_group_count_by_state"], dtype=np.int64)
        targets = list(state["target_transform_supervision_by_state"])
        if (
            state["format_name"] != "ncls.fitted-training-state"
            or state["format_version"] != 1
            or state["fit_scope"]
            != "source-train-states-and-train-query-groups-only"
            or state["latent_scope"]
            != "feed-forward-unseen-source-state-compilation"
            or state["target_transform_id"] != self.target_transform_id
            or state["source_feature_contract_id"]
            != LAYER_STACK_SOURCE_FEATURE_CONTRACT_ID
            or not state_ids
            or len(set(state_ids)) != len(state_ids)
            or int(state["source_train_state_count"]) != len(state_ids)
            or counts.shape != (len(state_ids),)
            or np.any(counts < 1)
            or int(np.sum(counts)) != int(state["train_query_group_count"])
            or len(targets) != len(state_ids)
        ):
            raise ValueError("E3 source compiler fitted state contract is unsupported")
        for name, positive in (
            ("target_channel_scale", True),
            ("target_channel_mean", False),
            ("target_channel_standard_deviation", True),
        ):
            values = np.asarray(state[name], dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)) or (
                positive and np.any(values <= 0.0)
            ):
                raise ValueError(f"E3 source compiler {name} is invalid")
        target_lookup: dict[str, Mapping[str, Any]] = {}
        for target in targets:
            if not isinstance(target, Mapping) or set(target) != {
                "state_id", "scale", "mean", "standard_deviation"
            }:
                raise ValueError("E3 source compiler transform target is malformed")
            target_state_id = str(target["state_id"])
            if target_state_id not in state_ids or target_state_id in target_lookup:
                raise ValueError("E3 source compiler transform target state is invalid")
            for name, positive in (
                ("scale", True), ("mean", False), ("standard_deviation", True)
            ):
                values = np.asarray(target[name], dtype=np.float64)
                if values.shape != (3,) or not np.all(np.isfinite(values)) or (
                    positive and np.any(values <= 0.0)
                ):
                    raise ValueError("E3 source compiler transform target is invalid")
            target_lookup[target_state_id] = target
        if set(target_lookup) != set(state_ids):
            raise ValueError("E3 source compiler transform target coverage is incomplete")
        self._training_state = dict(state)
        self._transform_targets_by_state_id = target_lookup

    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        if self._training_state is None:
            raise RuntimeError("E3 source compiler fitted state has not been loaded")
        parameters = dict(model_parameters)
        try:
            compiler_width = int(parameters.pop("compiler_width"))
            compiler_type_width = int(parameters.pop("compiler_type_width"))
            compiler_layer_count = int(parameters.pop("compiler_layer_count"))
        except KeyError as error:
            raise ValueError("E3 source compiler dimensions are required") from error
        config = NeuralEvaluatorModelConfig.from_mapping(parameters)
        return LayerStackSourceCompilerNeuralEvaluator(
            config,
            compiler_width=compiler_width,
            compiler_type_width=compiler_type_width,
            compiler_layer_count=compiler_layer_count,
            base_transform_scale=torch.as_tensor(
                self._training_state["target_channel_scale"]
            ),
            base_transform_mean=torch.as_tensor(
                self._training_state["target_channel_mean"]
            ),
            base_transform_standard_deviation=torch.as_tensor(
                self._training_state["target_channel_standard_deviation"]
            ),
        )

    @staticmethod
    def _decode_compiled(
        raw: torch.Tensor,
        scale: torch.Tensor,
        mean: torch.Tensor,
        standard_deviation: torch.Tensor,
    ) -> torch.Tensor:
        transformed = torch.clamp(
            raw * standard_deviation + mean, min=-15.0, max=15.0
        )
        return scale * torch.sinh(transformed)

    def _forward_components(
        self,
        model: LayerStackSourceCompilerNeuralEvaluator,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        interface_kinds, continuous, interface_counts = layer_stack_source_tensors(batch)
        core = self._core(batch, batch["view"].float(), batch["lights"].float())
        raw, scale, mean, standard_deviation = model.evaluate_compiled(
            batch["view"].float(),
            batch["lights"].float(),
            interface_kinds,
            continuous,
            interface_counts,
        )
        residual = self._decode_compiled(raw, scale, mean, standard_deviation)
        return core, residual, scale, mean, standard_deviation

    def _reverse_components(
        self,
        model: LayerStackSourceCompilerNeuralEvaluator,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wo = batch["view"].float()
        wi = batch["lights"].float()
        group_count, direction_count, _ = wi.shape
        reverse_view = wi.reshape(group_count * direction_count, 3)
        reverse_light = wo[:, None, :].expand(-1, direction_count, -1).reshape(
            group_count * direction_count, 1, 3
        )
        interface_kinds, continuous, interface_counts = (
            repeat_layer_stack_source_tensors(batch, direction_count)
        )
        reverse_core = self._core(
            batch, reverse_view, reverse_light, repeat_count=direction_count
        )
        raw, scale, mean, standard_deviation = model.evaluate_compiled(
            reverse_view,
            reverse_light,
            interface_kinds,
            continuous,
            interface_counts,
        )
        reverse_residual = self._decode_compiled(
            raw, scale, mean, standard_deviation
        )
        return (
            reverse_core.reshape(group_count, direction_count, 3),
            reverse_residual.reshape(group_count, direction_count, 3),
        )

    def _transform_supervision_loss(
        self,
        store: ReferenceQueryStore,
        batch: Mapping[str, torch.Tensor],
        scale: torch.Tensor,
        mean: torch.Tensor,
        standard_deviation: torch.Tensor,
    ) -> torch.Tensor:
        dataset_state_ids = store.dataset.state_strings("state_id")
        target_rows = []
        for state_index in batch["state_index"].detach().cpu().numpy().tolist():
            state_id = str(dataset_state_ids[int(state_index)])
            try:
                target_rows.append(self._transform_targets_by_state_id[state_id])
            except KeyError as error:
                raise ValueError(
                    "E3 transform supervision attempted to read a non-train source state"
                ) from error
        device, dtype = scale.device, scale.dtype
        target_scale = torch.as_tensor(
            [row["scale"] for row in target_rows], dtype=dtype, device=device
        )[:, None, :]
        target_mean = torch.as_tensor(
            [row["mean"] for row in target_rows], dtype=dtype, device=device
        )[:, None, :]
        target_standard_deviation = torch.as_tensor(
            [row["standard_deviation"] for row in target_rows],
            dtype=dtype,
            device=device,
        )[:, None, :]
        return (
            torch.mean(torch.square(torch.log(scale / target_scale)))
            + torch.mean(torch.square(
                (mean - target_mean) / target_standard_deviation
            ))
            + torch.mean(torch.square(torch.log(
                standard_deviation / target_standard_deviation
            )))
        )

    def predict(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> torch.Tensor:
        del device
        if not isinstance(model, LayerStackSourceCompilerNeuralEvaluator):
            raise TypeError("E3 source compiler received an incompatible model")
        core, residual, scale, mean, standard_deviation = self._forward_components(
            model, batch
        )
        prediction = torch.clamp(core + residual, min=0.0)
        if isinstance(batch, dict):
            batch["_analytic_core"] = core
            batch["_predicted_residual"] = residual
            batch["_compiled_scale"] = scale
            batch["_compiled_mean"] = mean
            batch["_compiled_standard_deviation"] = standard_deviation
            if model.training:
                batch["_compiled_transform_supervision_loss"] = (
                    self._transform_supervision_loss(
                        store, batch, scale, mean, standard_deviation
                    )
                )
                reverse_core, reverse_residual = self._reverse_components(model, batch)
                reverse = torch.clamp(reverse_core + reverse_residual, min=0.0)
                single_sheen = (
                    (batch["interface_counts"].long() == 1)
                    & (batch["top_kind"].long() == 3)
                )
                deviation = (
                    SourceAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline
                    ._source_reciprocity_deviation(
                        prediction,
                        reverse,
                        core,
                        reverse_core,
                        batch["view"].float(),
                        batch["lights"].float(),
                        single_sheen,
                    )
                )
                batch["_reciprocity_penalty"] = torch.mean(torch.log1p(deviation))
        return prediction

    def _loss_transform_tensors(
        self,
        reference: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = tuple(batch.get(name) for name in (
            "_compiled_scale", "_compiled_mean", "_compiled_standard_deviation"
        ))
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise RuntimeError("E3 source compiler loss requires compiled transform state")
        return values  # type: ignore[return-value]

    def training_loss(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        base_loss = super().training_loss(prediction, batch)
        transform_loss = batch.get("_compiled_transform_supervision_loss")
        if not isinstance(transform_loss, torch.Tensor):
            return base_loss
        return base_loss + self.transform_supervision_weight * transform_loss

    def additional_metric_distributions(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: ReferenceQueryStore,
        device: torch.device,
    ) -> Mapping[str, np.ndarray]:
        del store, device
        if not isinstance(model, LayerStackSourceCompilerNeuralEvaluator):
            raise TypeError("E3 source compiler received an incompatible model")
        core, residual, _, _, _ = self._forward_components(model, batch)
        forward = torch.clamp(core + residual, min=0.0)
        reverse_core, reverse_residual = self._reverse_components(model, batch)
        reverse = torch.clamp(reverse_core + reverse_residual, min=0.0)
        single_sheen = (
            (batch["interface_counts"].long() == 1)
            & (batch["top_kind"].long() == 3)
        )
        absolute_reciprocity = self._reciprocity_values(
            forward, reverse, batch["view"].float(), batch["lights"].float()
        )
        source_deviation = (
            SourceAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline
            ._source_reciprocity_deviation(
                forward,
                reverse,
                core,
                reverse_core,
                batch["view"].float(),
                batch["lights"].float(),
                single_sheen,
            )
        )
        weights = batch["solid_angle_weight"].float()[..., None]
        target = batch["mean"].float()
        core_error = torch.sum(torch.abs(core - target) * weights, dim=(1, 2)) / torch.clamp(
            torch.sum(torch.abs(target) * weights, dim=(1, 2)), min=1e-8
        )
        peak_support = (
            NoiseAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline
            ._peak_support_angle(forward, target, batch["lights"].float())
        )
        return {
            "reciprocity_relative_l1": absolute_reciprocity.detach().cpu().numpy(),
            "source_reciprocity_deviation_relative_l1": (
                source_deviation.detach().cpu().numpy()
            ),
            "analytic_core_solid_angle_normalized_l1": (
                core_error.detach().cpu().numpy()
            ),
            "peak_support_angle_degrees": peak_support.detach().cpu().numpy(),
        }

    def parameter_costs(self, model: nn.Module) -> Mapping[str, Any]:
        if not isinstance(model, LayerStackSourceCompilerNeuralEvaluator):
            return super().parameter_costs(model)
        return {
            **model.cost_summary(),
            "cost_scope": (
                "runtime baked compiled latent/transform plus shared evaluator; LayerStack "
                "token compiler weights/input/MAC are reported separately as offline compile cost"
            ),
        }
