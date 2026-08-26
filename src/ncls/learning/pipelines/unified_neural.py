from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ncls.learning.data import UnifiedScatteringTrainingStore
from ncls.learning.models import UnifiedNeuralModel
from ncls.learning.source_adapters import fit_direct_top_state
from ncls.learning.slang import UNIFIED_LAYOUT

from .appearance_loss import p1_appearance_loss
from .base import LearningPipeline, LearningPipelineDescriptor
from .p1_evaluator import fit_p1_scales
from .registry import register_pipeline


@dataclass(frozen=True)
class UnifiedEvaluatorSpec:
    name: str
    evaluator: str
    runtime_class: str


UNIFIED_EVALUATORS = (
    UnifiedEvaluatorSpec(
        "core-frame-neural-v1",
        "core-frame-neural-v1",
        "realtime",
    ),
)


class UnifiedNeuralPipeline(LearningPipeline):
    def __init__(self, spec: UnifiedEvaluatorSpec) -> None:
        self.spec = spec
        self.descriptor = LearningPipelineDescriptor(
            name=spec.name,
            stage="P1",
            data={
                "reader": "unified-scattering-entry-v1",
                "partition": "target-visible-v1",
                "source_adapter": "layer-stack-direct-top-v1",
            },
            model={
                "representation": spec.evaluator,
                "architecture": "core-frame-positive-residual-v1",
                "latent": "direct-fit-z16-v1",
            },
            fitting={"path": "gradient", "loss": "p1-appearance-v3+mollification-v1"},
            runtime={
                "compiler": "unified-slang-core-v1",
                "exporter": "packed-compiled-material-v1",
                "deployment_candidate": spec.runtime_class == "realtime",
            },
            supported_families=("layer-stack",),
            scope="03 core-frame positive-residual neural evaluator 候选",
        )
        self._fitted_state: dict[str, Any] | None = None

    def open_store(self, data_path: str) -> UnifiedScatteringTrainingStore:
        return UnifiedScatteringTrainingStore(Path(data_path))

    def fit_training_state(
        self,
        store: UnifiedScatteringTrainingStore,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        direct_top = fit_direct_top_state(store)
        scales = fit_p1_scales(
            store,
            train_indices,
            direct_top if self.spec.evaluator == "core-frame-neural-v1" else None,
        )
        return {
            "contract": "ncls.unified-neural-fit@1",
            "data_entry_id": store.data_id,
            "state_ids": scales["state_ids"],
            "target_scale": scales["target_scale"],
            "response_scale": scales["output_scale"],
            "initial_output_ratio": float(scales["initial_output_ratio"]),
            "direct_top": direct_top,
            "train_only": True,
        }

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        response_scale = np.asarray(state.get("response_scale"), dtype=np.float64)
        target_scale = np.asarray(state.get("target_scale"), dtype=np.float64)
        state_ids = state.get("state_ids")
        direct_top = state.get("direct_top")
        if (
            state.get("contract") != "ncls.unified-neural-fit@1"
            or state.get("data_entry_id") != UnifiedScatteringTrainingStore.ENTRY_ID
            or response_scale.ndim != 2
            or response_scale.shape[1:] != (3,)
            or target_scale.shape != response_scale.shape
            or np.any(response_scale <= 0.0)
            or np.any(target_scale <= 0.0)
            or not isinstance(state_ids, list)
            or len(state_ids) != len(response_scale)
            or not isinstance(direct_top, Mapping)
            or direct_top.get("state_ids") != state_ids
            or state.get("train_only") is not True
        ):
            raise ValueError("invalid unified fitted training state")
        self._fitted_state = dict(state)

    def _require_state(self) -> dict[str, Any]:
        if self._fitted_state is None:
            raise ValueError("unified pipeline fitted state has not been loaded")
        return self._fitted_state

    def create_model(self, model_parameters: Mapping[str, Any]) -> torch.nn.Module:
        if set(model_parameters) != {"state_count", "sampler"}:
            raise ValueError("unified evaluator model parameters must be state_count and sampler")
        if model_parameters["sampler"] not in {"nvidia-diffuse-ggx9", "ltc-k2"}:
            raise ValueError("unified evaluator sampler is unsupported")
        state = self._require_state()
        state_count = int(model_parameters["state_count"])
        if state_count != len(state["state_ids"]):
            raise ValueError("unified model state count disagrees with fitted state")
        return UnifiedNeuralModel(
            state_count=state_count,
            response_scale=state["response_scale"],
            top_rows=state["direct_top"]["rows"],
            evaluator=self.spec.evaluator,
            runtime_class=self.spec.runtime_class,
            initial_output_ratio=float(state["initial_output_ratio"]),
        )

    def predict_f(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: Any,
        device: torch.device,
    ) -> torch.Tensor:
        del store, device
        if not isinstance(model, UnifiedNeuralModel):
            raise TypeError("unified pipeline requires UnifiedNeuralModel")
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
        return p1_appearance_loss(
            prediction_f,
            batch,
            self._require_state()["target_scale"],
        )

    def core_f(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: Any,
        device: torch.device,
    ) -> torch.Tensor:
        """exact-core候选的top-interface覆盖探针；direct baseline不声明解析core。"""

        del store, device
        if self.spec.evaluator != "core-frame-neural-v1":
            raise ValueError("direct unified evaluator has no analytic core probe")
        if not isinstance(model, UnifiedNeuralModel):
            raise TypeError("unified core probe requires UnifiedNeuralModel")
        return model.top_interface_f(
            batch["state_index"].long(), batch["wo"].float(), batch["wi"].float()
        )

    def parameter_costs(self, model: torch.nn.Module | None) -> Mapping[str, Any]:
        if model is None:
            if self.spec.runtime_class == "diagnostic":
                evaluator_weights = 17 * 64 + 64 + 2 * (64 * 64 + 64) + 64 * 3 + 3
                total_without_latent = 23 * 64 + 64 + 64 * 64 + 64 + 64 * 27 + 27 + evaluator_weights
            else:
                evaluator_weights = 17 * 32 + 32 + 32 * 32 + 32 + 32 * 3 + 3
                total_without_latent = 23 * 64 + 64 + 64 * 64 + 64 + 64 * 27 + 27 + evaluator_weights
            section = UNIFIED_LAYOUT["paper" if self.spec.runtime_class == "diagnostic" else "realtime"]
            return {
                "B_asset": 128,
                "B_shared": int(2 * total_without_latent),
                "B_evaluate_weights": int(2 * evaluator_weights),
                "C_prepare_macs": int(UNIFIED_LAYOUT["realtime"]["prepare_macs"]),
                "C_eval_macs": int(section["evaluate_macs"]),
                "state_bytes_per_pixel": 64,
                "latent_bytes": 32,
                "runtime_class": self.spec.runtime_class,
                "parameter_count": total_without_latent,
            }
        if not isinstance(model, UnifiedNeuralModel):
            raise TypeError("unified cost accounting requires UnifiedNeuralModel")
        total = sum(parameter.numel() for parameter in model.parameters())
        latent = model.latent.numel()
        evaluator_weights = (
            model.evaluate_w0.numel() + model.evaluate_b0.numel()
            + model.evaluate_w1.numel() + model.evaluate_b1.numel()
            + model.evaluate_out_w.numel() + model.evaluate_out_b.numel()
            + (0 if model.evaluate_w2 is None else model.evaluate_w2.numel())
            + (0 if model.evaluate_b2 is None else model.evaluate_b2.numel())
        )
        section = UNIFIED_LAYOUT["paper" if self.spec.runtime_class == "diagnostic" else "realtime"]
        return {
            "B_asset": 128,
            "B_shared": int(2 * (total - latent)),
            "B_evaluate_weights": int(2 * evaluator_weights),
            "C_prepare_macs": int(UNIFIED_LAYOUT["realtime"]["prepare_macs"]),
            "C_eval_macs": int(section["evaluate_macs"]),
            "state_bytes_per_pixel": 64,
            "latent_bytes": 32,
            "runtime_class": self.spec.runtime_class,
            "parameter_count": total,
        }


def register_unified_neural_pipelines() -> None:
    for spec in UNIFIED_EVALUATORS:
        register_pipeline(lambda spec=spec: UnifiedNeuralPipeline(spec))
