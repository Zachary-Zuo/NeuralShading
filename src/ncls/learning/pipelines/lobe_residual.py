"""`lobe-residual` 候选（docs/research/p1_v2_plan.md §1.1）的注册占位。

模型、前向与 loss 等 P2.5 的 SlangPy 接入（`ncls.learning.slang.session`）；本文件只登记
descriptor 与 §1.2 的解析成本，供部署预算单元门与配置解析使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .base import LearningPipeline, LearningPipelineDescriptor
from .registry import register_pipeline


LATENT_DIM = 16
WO_FEATURES = 7
PREPARE_WIDTH = 64
HIDDEN_DIM = 8
LOBE_PARAMS = 9
PSI_FEATURES = 6
CORRECTION_WIDTH = 32
CORRECTIONS = ("none", "log32")
TOP_INTERFACE_IR_BYTES = 64
TOP_EVAL_FLOPS = 150  # §1.2：精确顶层 microfacet 反射
LOBE_EVAL_MACS = 60


@dataclass(frozen=True)
class LobeResidualSpec:
    lobe_count: int
    correction: str

    def __post_init__(self) -> None:
        if self.lobe_count not in {2, 3} or self.correction not in CORRECTIONS:
            raise ValueError("lobe-residual spec only exposes K in {2, 3} and correction in {none, log32}")

    @property
    def name(self) -> str:
        suffix = "" if self.correction == "none" else f"-{self.correction}"
        return f"lobe-residual-k{self.lobe_count}{suffix}-v1"

    @property
    def deployment_candidate(self) -> bool:
        return self.lobe_count == 2 and self.correction == "none"


def lobe_residual_costs(spec: LobeResidualSpec) -> dict[str, Any]:
    """§1.2 的静态成本；state 与权重按部署实际字节计。"""

    raw = HIDDEN_DIM + LOBE_PARAMS * spec.lobe_count
    prepare_macs = (LATENT_DIM + WO_FEATURES) * PREPARE_WIDTH + PREPARE_WIDTH ** 2 + PREPARE_WIDTH * raw
    prepare_parameters = prepare_macs + 2 * PREPARE_WIDTH + raw
    corrected = spec.correction == "log32"
    correction_macs = ((HIDDEN_DIM + PSI_FEATURES) * CORRECTION_WIDTH + CORRECTION_WIDTH * 3) if corrected else 0
    correction_parameters = (correction_macs + CORRECTION_WIDTH + 3) if corrected else 0
    lobe_words = (LOBE_PARAMS * spec.lobe_count + 1) // 2
    state_bytes = 4 + 4 * lobe_words + (2 * HIDDEN_DIM if corrected else 0) + 8
    return {
        "B_asset": 4 * LATENT_DIM + TOP_INTERFACE_IR_BYTES,
        "B_shared": 4 * (prepare_parameters + correction_parameters),
        "B_evaluate_weights": 2 * correction_parameters,
        "C_prepare_macs": prepare_macs,
        "C_eval_macs": TOP_EVAL_FLOPS + LOBE_EVAL_MACS * spec.lobe_count + correction_macs,
        "state_bytes_per_pixel": state_bytes,
        "analytic_core_state_bytes": 0,
        "C_eval_excludes_analytic_core": False,
        "parameter_count": prepare_parameters + correction_parameters,
    }


class LobeResidualPipeline(LearningPipeline):
    def __init__(self, spec: LobeResidualSpec) -> None:
        self.spec = spec
        self.descriptor = LearningPipelineDescriptor(
            name=spec.name,
            stage="P1",
            data={
                "reader": "reference-corpus-v2",
                "partition": "target-visible-v1",
                "source_adapter": "layer-stack-direct-top-v1",
            },
            model={
                "representation": "analytic-core-lobe-residual-v1",
                "architecture": "lobe-residual-prepare-mlp-v1",
                "latent": "autodecoder-v1",
            },
            fitting={"path": "gradient", "loss": "p1-appearance-v3"},
            runtime={
                "compiler": "slangpy-lobe-residual-core-v1",
                "exporter": "method-bundle-v1",
                "deployment_candidate": spec.deployment_candidate,
            },
            supported_families=("layer-stack",),
            scope=f"P1 M2B lobe-residual K={spec.lobe_count} correction={spec.correction}",
        )

    def _pending(self, member: str) -> NotImplementedError:
        return NotImplementedError(
            f"{self.descriptor.name}.{member} 等 P2.5 SlangPy 接入（ncls.learning.slang.session）"
        )

    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        del model_parameters
        raise self._pending("create_model")

    def predict_f(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        store: Any,
        device: torch.device,
    ) -> torch.Tensor:
        del model, batch, store, device
        raise self._pending("predict_f")

    def training_loss(
        self,
        prediction_f: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        del prediction_f, batch
        raise self._pending("training_loss")

    def parameter_costs(self, model: nn.Module | None = None) -> Mapping[str, Any]:
        del model
        return lobe_residual_costs(self.spec)


def register_lobe_residual_pipelines() -> None:
    for spec in (
        LobeResidualSpec(2, "none"),
        LobeResidualSpec(2, "log32"),
        LobeResidualSpec(3, "log32"),
    ):
        register_pipeline(lambda spec=spec: LobeResidualPipeline(spec))
