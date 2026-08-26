from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ncls.learning.data import UnifiedScatteringTrainingStore
from ncls.learning.models import (
    NvidiaNeuralAppearanceLtcAdaptationModel,
    NvidiaNeuralAppearanceModel,
)
from ncls.learning.slang import NVIDIA_NEURAL_APPEARANCE_LAYOUT

from .base import LearningPipeline, LearningPipelineDescriptor
from .registry import register_pipeline
from .sampler_objective import sampler_cross_entropy


class NvidiaNeuralAppearancePipeline(LearningPipeline):
    """原规模 learned-frame evaluator 与 GGX9 sampler 的联合复现。"""

    descriptor = LearningPipelineDescriptor(
        name="nvidia-frame-two-lobe-paper-v1",
        stage="P1",
        data={
            "reader": "unified-scattering-entry-v1",
            "partition": "target-visible-v1",
            "source_adapter": "layer-stack-uniform-latent-v1",
        },
        model={
            "representation": "nvidia-learned-frame-two-lobe-v1",
            "architecture": "nvidia-evaluator-3x64-sampler-3x32-v1",
            "latent": "direct-fit-z8-v1",
        },
        fitting={
            "path": "gradient",
            "loss": "nvidia-log1p-response-l1+joint-detached-kl+mollification-entry-v1",
        },
        runtime={
            "compiler": "nvidia-neural-appearance-slang-v1",
            "exporter": "nvidia-neural-appearance-packed-v1",
            "deployment_candidate": False,
        },
        supported_families=("layer-stack",),
        scope="03 NVIDIA 原规模 learned-frame evaluator 与 GGX9 sampler 复现",
    )

    def __init__(self) -> None:
        self._fitted_state: dict[str, Any] | None = None

    def open_store(self, data_path: str) -> UnifiedScatteringTrainingStore:
        return UnifiedScatteringTrainingStore(Path(data_path))

    def fit_training_state(
        self,
        store: UnifiedScatteringTrainingStore,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        del train_indices
        return {
            "contract": "ncls.nvidia-neural-appearance-fit@1",
            "data_entry_id": store.data_id,
            "state_ids": list(map(str, store.state_strings("state_id").tolist())),
            "train_only": True,
        }

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        state_ids = state.get("state_ids")
        if (
            state.get("contract") != "ncls.nvidia-neural-appearance-fit@1"
            or state.get("data_entry_id") != UnifiedScatteringTrainingStore.ENTRY_ID
            or not isinstance(state_ids, list)
            or not state_ids
            or any(not isinstance(value, str) or not value for value in state_ids)
            or len(set(state_ids)) != len(state_ids)
            or state.get("train_only") is not True
        ):
            raise ValueError("invalid NVIDIA baseline fitted training state")
        self._fitted_state = dict(state)

    def _require_state(self) -> dict[str, Any]:
        if self._fitted_state is None:
            raise ValueError("NVIDIA baseline fitted state has not been loaded")
        return self._fitted_state

    def create_model(self, model_parameters: Mapping[str, Any]) -> torch.nn.Module:
        if set(model_parameters) != {"state_count", "sampler"}:
            raise ValueError(
                "NVIDIA baseline model parameters must be state_count and sampler"
            )
        if model_parameters["sampler"] != "nvidia-diffuse-ggx9":
            raise ValueError("NVIDIA reproduction requires its native GGX9 sampler")
        state_count = int(model_parameters["state_count"])
        if state_count != len(self._require_state()["state_ids"]):
            raise ValueError("NVIDIA baseline state count disagrees with fitted state")
        return NvidiaNeuralAppearanceModel(state_count=state_count)

    def predict_f(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: Any,
        device: torch.device,
    ) -> torch.Tensor:
        del store, device
        if not isinstance(
            model,
            (NvidiaNeuralAppearanceModel, NvidiaNeuralAppearanceLtcAdaptationModel),
        ):
            raise TypeError("NVIDIA pipeline requires NvidiaNeuralAppearanceModel")
        return model(
            batch["state_index"].long(),
            batch["wo"].float(),
            batch["wi"].float(),
        )

    def training_loss(
        self,
        prediction_f: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        minimum_cosine = float(
            NVIDIA_NEURAL_APPEARANCE_LAYOUT["response_adapter"]["minimum_cosine"]
        )
        cosine = torch.clamp(batch["wi"][..., 2:3], min=minimum_cosine)
        predicted_response = prediction_f * cosine
        target_response = torch.clamp(batch["mean"], min=0.0)
        return torch.mean(
            torch.abs(torch.log1p(predicted_response) - torch.log1p(target_response))
        )

    def auxiliary_training_batch(
        self,
        store: UnifiedScatteringTrainingStore,
        train_indices: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        *,
        step: int,
        total_steps: int,
    ) -> Mapping[str, np.ndarray]:
        return store.training_batch(
            train_indices,
            batch_size,
            rng,
            step=step,
            total_steps=total_steps,
        )

    def training_objective(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        auxiliary_batch: Mapping[str, torch.Tensor] | None,
        store: UnifiedScatteringTrainingStore,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Mapping[str, torch.Tensor | float],
    ]:
        del store, device
        if not isinstance(model, NvidiaNeuralAppearanceModel):
            raise TypeError("NVIDIA pipeline requires NvidiaNeuralAppearanceModel")
        if auxiliary_batch is None:
            raise ValueError("NVIDIA joint training requires an independent sampler batch")
        prediction = model(
            batch["state_index"].long(), batch["wo"].float(), batch["wi"].float()
        )
        evaluator_loss = self.training_loss(prediction, batch)
        sampler_target = model(
            auxiliary_batch["state_index"].long(),
            auxiliary_batch["wo"].float(),
            auxiliary_batch["wi"].float(),
        ).detach()
        proposal_pdf, _ = model.sampler_pdf_with_head(
            auxiliary_batch["state_index"].long(),
            auxiliary_batch["wo"].float(),
            auxiliary_batch["wi"].float(),
            "nvidia-diffuse-ggx9",
        )
        sampler_loss, relative_kl = sampler_cross_entropy(
            sampler_target,
            auxiliary_batch["wi"],
            auxiliary_batch["solid_angle_weight"],
            proposal_pdf,
        )
        total = evaluator_loss + sampler_loss
        return prediction, total, {
            "evaluator_log1p_l1": evaluator_loss.detach(),
            "sampler_cross_entropy": sampler_loss.detach(),
            "sampler_relative_kl": relative_kl.detach(),
        }

    def parameter_costs(self, model: torch.nn.Module | None) -> Mapping[str, Any]:
        layout = NVIDIA_NEURAL_APPEARANCE_LAYOUT
        evaluator_parameters = 9859
        frame_parameters = 96
        sampler_parameters = 2793
        shared_parameters = evaluator_parameters + frame_parameters + sampler_parameters
        total_parameters = shared_parameters
        if model is not None:
            if not isinstance(model, NvidiaNeuralAppearanceModel):
                raise TypeError("NVIDIA cost accounting requires its native model")
            total_parameters = sum(parameter.numel() for parameter in model.parameters())
        return {
            "B_asset": int(layout["compiled_material"]["total_bytes"]),
            "B_shared": 2 * shared_parameters,
            "B_evaluate_weights": 2 * evaluator_parameters,
            "C_prepare_macs": int(
                layout["evaluator"]["frame_macs"] + layout["sampler"]["prepare_macs"]
            ),
            "C_eval_macs": int(layout["evaluator"]["evaluate_macs"]),
            "state_bytes_per_pixel": int(layout["state"]["stride_bytes"]),
            "latent_bytes": 16,
            "runtime_class": "diagnostic",
            "parameter_count": total_parameters,
        }

    def gradient_evidence(self, model: torch.nn.Module) -> Mapping[str, Any]:
        if not isinstance(model, NvidiaNeuralAppearanceModel):
            raise TypeError("NVIDIA pipeline requires NvidiaNeuralAppearanceModel")

        def summarize(names: tuple[str, ...]) -> dict[str, Any]:
            tensors = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name == names[0] or name.startswith(names[1:])
            ]
            if not tensors or any(value is None for value in tensors):
                raise RuntimeError("NVIDIA joint objective left a parameter group without gradients")
            gradients = [value for value in tensors if value is not None]
            if any(not bool(torch.isfinite(value).all()) for value in gradients):
                raise RuntimeError("NVIDIA joint objective produced non-finite group gradients")
            maximum = max(float(value.detach().abs().max()) for value in gradients)
            if maximum == 0.0:
                raise RuntimeError("NVIDIA joint objective produced an all-zero parameter group")
            return {
                "all_present": True,
                "all_finite": True,
                "tensor_count": len(gradients),
                "maximum_absolute_value": maximum,
            }

        return {
            "latent": summarize(("latent", "__never__")),
            "evaluator": summarize(("__never__", "frame_", "evaluate_")),
            "sampler": summarize(("__never__", "sampler_")),
        }


def register_nvidia_neural_appearance_pipeline() -> None:
    register_pipeline(NvidiaNeuralAppearancePipeline)
