from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch

from ncls.learning.models import ConditionedSharedEvaluator, PerStateTeacher
from ncls.learning.source_adapters import direct_top_bsdf, fit_direct_top_state

from .appearance_loss import p1_appearance_loss
from .base import LearningPipeline, LearningPipelineDescriptor
from .registry import register_pipeline


CAPACITY_DEFAULTS: dict[str, dict[str, int]] = {
    "S": {
        "width": 128,
        "latent_dim": 32,
        "prepare_blocks": 2,
        "evaluate_blocks": 3,
        "fourier_bands": 4,
    },
    "M": {
        "width": 256,
        "latent_dim": 64,
        "prepare_blocks": 3,
        "evaluate_blocks": 6,
        "fourier_bands": 5,
    },
    "L": {
        "width": 512,
        "latent_dim": 128,
        "prepare_blocks": 3,
        "evaluate_blocks": 8,
        "fourier_bands": 6,
    },
}


def _fit_scales(
    store: Any,
    train_indices: np.ndarray,
    direct_top: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target_values: list[list[np.ndarray]] = [[] for _ in range(store.state_count)]
    output_values: list[list[np.ndarray]] = [[] for _ in range(store.state_count)]
    scale_fields = (
        ("state_index", "wo", "wi", "mean")
        if direct_top is not None
        else ("state_index", "wi", "mean")
    )
    for raw in store.iter_batches(
        train_indices,
        batch_size=8,
        fields=scale_fields,
    ):
        targets = np.asarray(raw["mean"], dtype=np.float32)
        cosine = np.abs(np.asarray(raw["wi"], dtype=np.float32)[..., 2:3])
        output = targets / np.maximum(cosine, 1e-4)
        if direct_top is not None:
            with torch.no_grad():
                core = direct_top_bsdf(
                    direct_top,
                    torch.as_tensor(raw["state_index"], dtype=torch.long),
                    torch.as_tensor(raw["wo"], dtype=torch.float32),
                    torch.as_tensor(raw["wi"], dtype=torch.float32),
                ).cpu().numpy()
            output = output - core
        for row, state_index in enumerate(np.asarray(raw["state_index"], dtype=np.int64)):
            index = int(state_index)
            target_values[index].append(targets[row].reshape(-1, 3))
            valid = cosine[row, :, 0] >= 1e-4
            output_values[index].append(output[row, valid].reshape(-1, 3))
    target_scales = []
    output_scales = []
    normalized_output_values = []
    for state_index, (target_parts, output_parts) in enumerate(
        zip(target_values, output_values, strict=True)
    ):
        if not target_parts or not output_parts:
            raise ValueError(f"P1 fitted scale has no train queries for state {state_index}")
        target = np.concatenate(target_parts, axis=0)
        output = np.concatenate(output_parts, axis=0)
        target_scale = np.maximum(np.quantile(np.abs(target), 0.99, axis=0), 1e-6)
        output_scale = np.maximum(np.quantile(np.abs(output), 0.99, axis=0), 1e-6)
        target_scales.append(target_scale.tolist())
        output_scales.append(output_scale.tolist())
        normalized_output_values.append(np.abs(output) / output_scale[None, :])
    initial_output_ratio = float(np.clip(
        np.quantile(np.concatenate(normalized_output_values, axis=0), 0.5),
        1e-4,
        0.25,
    ))
    return {
        "contract": "ncls.p1-output-scale@3",
        "state_ids": list(map(str, store.state_strings("state_id").tolist())),
        "target_scale": target_scales,
        "output_scale": output_scales,
        "initial_output_ratio": initial_output_ratio,
    }


@dataclass(frozen=True)
class P1PipelineSpec:
    family: str
    capacity: str

    @property
    def name(self) -> str:
        if self.family == "m1":
            return f"film-evaluator-{self.capacity.lower()}-v1"
        if self.family == "m2":
            return f"analytic-residual-{self.capacity.lower()}-v1"
        if self.family == "teacher":
            return "per-state-teacher-l-v1"
        raise ValueError("unsupported P1 pipeline family")


class P1EvaluatorPipeline(LearningPipeline):
    # 可选成员，供 evaluation/p1_audit.py 探测：M2 暴露解析 core（E_core/E_ref），并声明 signed 残差（死区诊断）。
    core_f: Callable[[torch.nn.Module, Mapping[str, torch.Tensor], Any, torch.device], torch.Tensor] | None = None

    def __init__(self, spec: P1PipelineSpec) -> None:
        self.spec = spec
        if spec.family == "m2":
            self.core_f = self._direct_top_core
        representation = {
            "m1": "conditioned-shared-evaluator-v1",
            "m2": "analytic-core-neural-residual-v1",
            "teacher": "per-state-diagnostic-teacher-v1",
        }[spec.family]
        architecture = {
            "m1": "film-prepare-evaluate-calibrated-softplus-v2",
            "m2": "film-prepare-evaluate-residual-v1",
            "teacher": "independent-state-mlp-calibrated-softplus-v2",
        }[spec.family]
        latent = "none" if spec.family == "teacher" else "autodecoder-v1"
        self.descriptor = LearningPipelineDescriptor(
            name=spec.name,
            stage="P1",
            data={
                "reader": "reference-corpus-v2",
                "partition": "target-visible-v1",
                "source_adapter": (
                    "layer-stack-direct-top-v1" if spec.family == "m2" else "none"
                ),
            },
            model={
                "representation": representation,
                "architecture": architecture,
                "latent": latent,
            },
            fitting={"path": "gradient", "loss": "p1-appearance-v3"},
            runtime={"compiler": "none", "exporter": "deferred", "deployment_candidate": False},
            supported_families=("layer-stack",),
            scope=f"P1 {spec.family.upper()} {spec.capacity} appearance validation",
        )
        self._fitted_state: dict[str, Any] | None = None

    @property
    def has_signed_residual(self) -> bool:
        return self.spec.family == "m2"

    def fit_training_state(
        self,
        store: Any,
        train_indices: np.ndarray,
    ) -> Mapping[str, Any]:
        direct_top = fit_direct_top_state(store) if self.spec.family == "m2" else None
        result = _fit_scales(store, train_indices, direct_top)
        if self.spec.family == "m2":
            result["direct_top"] = direct_top
        return result

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        if state.get("contract") != "ncls.p1-output-scale@3":
            raise ValueError("P1 fitted training state contract is unsupported")
        target = np.asarray(state.get("target_scale"), dtype=np.float64)
        output = np.asarray(state.get("output_scale"), dtype=np.float64)
        state_ids = state.get("state_ids")
        if (
            target.ndim != 2
            or target.shape[1:] != (3,)
            or output.shape != target.shape
            or not isinstance(state_ids, list)
            or len(state_ids) != len(target)
            or np.any(target <= 0.0)
            or np.any(output <= 0.0)
            or not isinstance(state.get("initial_output_ratio"), (int, float))
            or not 0.0 < float(state["initial_output_ratio"]) <= 0.25
        ):
            raise ValueError("P1 fitted training scales are invalid")
        if self.spec.family == "m2" and not isinstance(state.get("direct_top"), dict):
            raise ValueError("M2 fitted state is missing the analytic core")
        self._fitted_state = dict(state)

    def _require_state(self) -> dict[str, Any]:
        if self._fitted_state is None:
            raise ValueError("P1 pipeline fitted training state has not been loaded")
        return self._fitted_state

    def create_model(self, model_parameters: Mapping[str, Any]) -> torch.nn.Module:
        state = self._require_state()
        expected = CAPACITY_DEFAULTS[self.spec.capacity]
        state_count = int(model_parameters.get("state_count", 0))
        if state_count != len(state["state_ids"]):
            raise ValueError("P1 model state_count disagrees with fitted state")
        if self.spec.family == "teacher":
            allowed = {"state_count", "width", "block_count", "fourier_bands"}
            if set(model_parameters) != allowed:
                raise ValueError("teacher model parameters are incomplete or unknown")
            return PerStateTeacher(
                state_count=state_count,
                output_scale=state["output_scale"],
                width=int(model_parameters["width"]),
                block_count=int(model_parameters["block_count"]),
                fourier_bands=int(model_parameters["fourier_bands"]),
                initial_output_ratio=float(state["initial_output_ratio"]),
            )
        allowed = {
            "state_count", "width", "latent_dim", "prepare_blocks",
            "evaluate_blocks", "fourier_bands",
        }
        if set(model_parameters) != allowed:
            raise ValueError("conditioned evaluator model parameters are incomplete or unknown")
        resolved = {name: int(model_parameters[name]) for name in allowed if name != "state_count"}
        if resolved != expected:
            raise ValueError(
                f"{self.spec.name} capacity parameters are frozen at {expected}"
            )
        return ConditionedSharedEvaluator(
            state_count=state_count,
            output_scale=state["output_scale"],
            output_mode="residual" if self.spec.family == "m2" else "direct",
            initial_output_ratio=float(state["initial_output_ratio"]),
            **resolved,
        )

    def _direct_top_core(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: Any,
        device: torch.device,
    ) -> torch.Tensor:
        del model, store, device
        return direct_top_bsdf(
            self._require_state()["direct_top"],
            batch["state_index"].long(),
            batch["wo"].float(),
            batch["wi"].float(),
        )

    def predict_f(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: Any,
        device: torch.device,
    ) -> torch.Tensor:
        prediction = model(batch["state_index"].long(), batch["wo"].float(), batch["wi"].float())
        if self.spec.family != "m2":
            return prediction
        core = self._direct_top_core(model, batch, store, device)
        return torch.clamp(core + prediction, min=0.0)

    def training_loss(
        self,
        prediction_f: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return p1_appearance_loss(prediction_f, batch, self._require_state()["target_scale"])

    def parameter_costs(self, model: torch.nn.Module) -> Mapping[str, Any]:
        total = sum(parameter.numel() for parameter in model.parameters())
        if isinstance(model, PerStateTeacher):
            network = model.networks[0]
            width = network.input.out_features
            evaluate_macs = (
                network.input.in_features * width
                + len(network.blocks) * 2 * width * width
                + 3 * width
            )
            return {
                "B_asset": 4 * (total + 3 * model.state_count),
                "B_shared": 0,
                "C_prepare_macs": 0,
                "C_eval_macs": int(evaluate_macs),
                "parameter_count": total,
            }
        latent = model.latent.weight.numel()
        width = model.width
        latent_dim = model.latent.embedding_dim
        block_count = len(model.prepare_layers) + len(model.evaluate_layers)
        condition_output = width + 2 * width * block_count
        prepare_macs = (
            latent_dim * width
            + width * condition_output
            + model.prepare_input.in_features * width
            + len(model.prepare_layers) * 2 * width * width
        )
        evaluate_macs = (
            model.evaluate_input.in_features * width
            + len(model.evaluate_layers) * 2 * width * width
            + 3 * width
        )
        analytic_core_state_bytes = 56 if self.spec.family == "m2" else 0
        return {
            "B_asset": (
                4 * (latent + 3 * model.state_count)
                + analytic_core_state_bytes * model.state_count
            ),
            "B_shared": 4 * (total - latent),
            "C_prepare_macs": int(prepare_macs),
            "C_eval_macs": int(evaluate_macs),
            "analytic_core_state_bytes": analytic_core_state_bytes,
            "C_eval_excludes_analytic_core": self.spec.family == "m2",
            "parameter_count": total,
        }


def register_p1_pipelines() -> None:
    for family in ("m1", "m2"):
        for capacity in ("S", "M", "L"):
            spec = P1PipelineSpec(family, capacity)
            register_pipeline(lambda spec=spec: P1EvaluatorPipeline(spec))
    teacher = P1PipelineSpec("teacher", "L")
    register_pipeline(lambda: P1EvaluatorPipeline(teacher))
