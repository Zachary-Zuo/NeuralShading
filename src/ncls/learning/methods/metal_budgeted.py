from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ncls.core.identity import sha256_json
from ncls.core.scattering import (
    BackendCapability,
    InstancePayload,
    MaterialPayload,
    RuntimePayload,
)
from ncls.core.source import SourceSnapshot
from ncls.learning.appearance_metrics import (
    AppearanceMetricCalibration,
    appearance_error_metrics,
)
from ncls.learning.batches import (
    EvaluatorBatch,
    MethodSamplerBatch,
    OnlineTrainingBatch,
)
from ncls.learning.method import (
    ComponentContract,
    MethodDefinition,
    MethodDescriptor,
    MethodReadinessPolicy,
    SourceAdaptationContract,
    TensorField,
    TrainingInitializationRequest,
)
from ncls.learning.methods.contracts import MethodPlugin
from ncls.learning.metal_runtime import fake_quantize_fp16_ste
from ncls.learning.models.metal_budgeted import (
    METAL_BUDGETED_REQUIRED_CONTEXT,
    MetalBudgetedModel,
)
from ncls.learning.models.metal_budgeted_profile import (
    METAL_BUDGETED_HYBRID_PROFILE,
    METAL_BUDGETED_LAYOUT_PATH,
    load_metal_budgeted_layout,
)
from ncls.learning.objectives import sampler_forward_kl_score
from ncls.learning.source_adapters import MetalBudgetedMdlSourceAdapter
from ncls.paths import PROJECT_ROOT


_PHASES = ("joint-response-fit", "deployment-qat-refine")
_GROUPS = (
    "asset_encoder",
    "asset_variant",
    "typed_compiler",
    "semantic_prepare",
    "directional_evaluator",
    "proposal_sampler",
)
_PREPARE_TENSORS = (
    "source_index",
    "wo",
    "uv",
    "uv_dx",
    "uv_dy",
    "paired_uv",
    "paired_uv_dx",
    "paired_uv_dy",
    "mip_level",
    "metal_mip_fraction",
    "metal_texture_patches",
    "metal_paired_texture_patches",
    "metal_texture_slot_mask",
    "metal_texture_role_class",
    "metal_graph_index",
    "metal_schema_index",
    "metal_recipe_index",
    "metal_identity_index",
    "metal_finish_index",
    "metal_asset_index",
    "metal_typed_semantic_id",
    "metal_typed_type_id",
    "metal_typed_responsibility_id",
    "metal_typed_discrete",
    "metal_typed_continuous",
    "metal_typed_presence",
    "metal_canonical_optical",
    "metal_access_state",
    "metal_frame_state",
    "metal_distribution_id",
)


def metal_budgeted_parameter_groups(
    model: MetalBudgetedModel,
) -> Mapping[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {name: [] for name in _GROUPS}
    for name, _ in model.named_parameters():
        if name.startswith("typed_compiler."):
            group = "typed_compiler"
        elif name == "asset.variant_scale_bias.weight":
            group = "asset_variant"
        elif name.startswith("asset."):
            group = "asset_encoder"
        elif name.startswith("prepared_model.proposal_adapter."):
            group = "proposal_sampler"
        elif name.startswith("prepared_model."):
            group = "semantic_prepare"
        elif name.startswith("evaluator."):
            group = "directional_evaluator"
        else:
            raise ValueError(f"unclassified Metal budgeted parameter: {name}")
        groups[group].append(name)
    if any(not names for names in groups.values()):
        raise ValueError("Metal budgeted parameter group cannot be empty")
    return {name: tuple(values) for name, values in groups.items()}


def metal_budgeted_runtime_parameter_names(
    model: MetalBudgetedModel,
) -> frozenset[str]:
    names = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("typed_compiler.")
        or name.startswith("prepared_model.")
        or name.startswith("evaluator.")
        or name == "asset.variant_scale_bias.weight"
    }
    if not names or any(name.startswith("asset.detail_encoder.") for name in names):
        raise ValueError("Metal budgeted runtime parameter classification drifted")
    return frozenset(names)


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        PROJECT_ROOT / "src/ncls/learning/models/metal_budgeted.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_budgeted_asset.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_budgeted_compiler.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_budgeted_evaluator.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_budgeted_sampler.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_budgeted_profile.py",
        PROJECT_ROOT / "src/ncls/learning/appearance_metrics.py",
        PROJECT_ROOT / "src/ncls/learning/source_adapters.py",
        METAL_BUDGETED_LAYOUT_PATH,
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little") + relative)
        digest.update(len(payload).to_bytes(8, "little") + payload)
    return digest.hexdigest()


def _state_schema() -> tuple[TensorField, ...]:
    with torch.device("meta"):
        model = MetalBudgetedModel.from_context(METAL_BUDGETED_REQUIRED_CONTEXT)
    return tuple(
        TensorField(
            name,
            str(value.dtype).removeprefix("torch."),
            tuple(int(item) for item in value.shape),
        )
        for name, value in model.state_dict().items()
    )


def _parameter_accounting() -> Mapping[str, int]:
    with torch.device("meta"):
        model = MetalBudgetedModel.from_context(METAL_BUDGETED_REQUIRED_CONTEXT)
    total = sum(parameter.numel() for parameter in model.parameters())
    runtime = sum(parameter.numel() for parameter in model.prepared_model.parameters())
    runtime += sum(parameter.numel() for parameter in model.evaluator.parameters())
    return {
        "P_trainable": total,
        "P_runtime_prepare_evaluate": runtime,
        "B_runtime_fp16_weights": 2 * runtime,
        "P_typed_compiler": sum(
            parameter.numel() for parameter in model.typed_compiler.parameters()
        ),
        "P_asset_encoder_and_variant": sum(
            parameter.numel() for parameter in model.asset.parameters()
        ),
    }


_PARAMETER_ACCOUNTING = _parameter_accounting()


def _component(
    component_id: str,
    groups: tuple[str, ...],
    batches: tuple[str, ...],
    outputs: tuple[str, ...],
) -> ComponentContract:
    return ComponentContract(
        component_id,
        True,
        groups,
        _PHASES,
        batches,
        outputs,
        ("checkpoint:model_state",),
        (),
    )


_COMPONENTS = (
    _component(
        "responsibility-aware-typed-compiler",
        ("typed_compiler",),
        ("reference-evaluator",),
        ("compiler_responsibility_groups_trace",),
    ),
    _component(
        "two-read-detail-context-asset",
        ("asset_encoder", "asset_variant"),
        ("reference-evaluator",),
        ("asset_detail_trace", "asset_context_trace"),
    ),
    _component(
        "runtime-semantic-24x32x32x24-prepare",
        ("semantic_prepare",),
        ("reference-evaluator",),
        ("semantic_runtime_trace", "appearance/semantic_runtime"),
    ),
    _component(
        "stable-two-frame-half-difference-28d",
        ("directional_evaluator",),
        ("reference-evaluator",),
        ("direction_half_trace", "direction_two_frame_trace"),
    ),
    _component(
        "same-shape-direct-or-dual-lobe-hybrid-evaluator",
        ("directional_evaluator", "typed_compiler"),
        ("reference-evaluator",),
        ("positive_rgb_trace", "rgb_gate_trace", "analytic_lobes_trace"),
    ),
    _component(
        "three-component-matched-proposal",
        ("proposal_sampler",),
        ("reference-evaluator", "method-sampler"),
        ("proposal_loss", "sample_pdf_identity"),
    ),
)


class _ProposalExecution(nn.Module):
    def __init__(
        self, model: MetalBudgetedModel, definition: "MetalBudgetedMethodDefinition"
    ) -> None:
        super().__init__()
        self.model = model
        self.definition = definition

    def forward(
        self, batches: Mapping[str, OnlineTrainingBatch], qat: bool
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        return self.definition._proposal_objective_impl(self.model, batches, qat=qat)


class _AppearanceExecution(nn.Module):
    def __init__(
        self, model: MetalBudgetedModel, definition: "MetalBudgetedMethodDefinition"
    ) -> None:
        super().__init__()
        self.model = model
        self.definition = definition

    def forward(
        self,
        batch: EvaluatorBatch,
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        return self.definition._appearance_objective_impl(
            self.model, batch, phase, qat=True
        )


class MetalBudgetedMethodDefinition(MethodDefinition):
    _layout = load_metal_budgeted_layout()
    descriptor = MethodDescriptor(
        "metal-budgeted-neural-material",
        1,
        "NVIDIA-class Metal budgeted neural material",
        _implementation_sha256(),
        (
            SourceAdaptationContract(
                "mdl.program@1", 1, ("/arguments",), "runtime-patch"
            ),
        ),
        {
            "reference-evaluator": (
                "wi",
                "target_f",
                "paired_target_f",
                *_PREPARE_TENSORS,
            ),
            "method-sampler": (
                "sample_u",
                *tuple(
                    name
                    for name in _PREPARE_TENSORS
                    if not name.startswith("paired_")
                    and name != "metal_paired_texture_patches"
                ),
            ),
        },
        _state_schema(),
        "ncls.metal-budgeted-method@1",
        int(
            BackendCapability.PREPARE
            | BackendCapability.EVALUATE
            | BackendCapability.SAMPLE
            | BackendCapability.PDF
            | BackendCapability.ANISOTROPIC_FRAME
            | BackendCapability.REVERSE_PDF
        ),
        {
            "maximum_prepare_steps": int(
                _layout["bounded_execution"]["maximum_prepare_steps"]
            ),
            "maximum_evaluate_steps": int(
                _layout["bounded_execution"]["maximum_evaluate_steps"]
            ),
            "maximum_state_bytes": int(
                _layout["bounded_execution"]["maximum_prepared_state_bytes"]
            ),
            "maximum_reads": int(
                _layout["bounded_execution"]["maximum_texture_reads"]
            ),
        },
        {
            "runtime_class": "nvidia-class-budgeted-neural-material",
            "profile_id": METAL_BUDGETED_HYBRID_PROFILE.profile_id,
            "direct_control_profile_id": "metal_budgeted_direct_control_v1",
            "C_prepare_macs": METAL_BUDGETED_HYBRID_PROFILE.prepare_dense_macs,
            "C_eval_macs": METAL_BUDGETED_HYBRID_PROFILE.evaluate_dense_macs,
            "B_prepared": METAL_BUDGETED_HYBRID_PROFILE.prepared_state_bytes,
            "maximum_texture_reads": METAL_BUDGETED_HYBRID_PROFILE.maximum_texture_reads,
            "proposal_components": 3,
            **_PARAMETER_ACCOUNTING,
            "observed_quality_gate": False,
            "observed_latency_gate": False,
        },
        metal_budgeted_parameter_groups(
            MetalBudgetedModel.from_context(METAL_BUDGETED_REQUIRED_CONTEXT)
        ),
        _COMPONENTS,
        readiness_policies={
            "diagnostic-evaluator": MethodReadinessPolicy(
                (
                    "asset_encoder",
                    "asset_variant",
                    "typed_compiler",
                    "semantic_prepare",
                    "directional_evaluator",
                ),
                (*_PHASES, "complete"),
            )
        },
    )

    def create_trainable(self, context: Mapping[str, Any]) -> nn.Module:
        return MetalBudgetedModel.from_context(context)

    @staticmethod
    def _calibration_recipe(phase: Mapping[str, Any]) -> Mapping[str, Any]:
        recipes = phase.get("recipes")
        value = recipes.get("appearance_calibration") if isinstance(recipes, Mapping) else None
        required = {
            "schema",
            "route",
            "sample_count",
            "seed",
            "scale_percentile",
            "peak_percentile",
            "scale_clamp",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != required
            or value.get("schema") != "train-only-reference-rgb-percentiles@1"
            or value.get("route") != "evaluator"
            or int(value.get("sample_count", 0)) != 16384
            or int(value.get("seed", -1)) != 2026090401
            or float(value.get("scale_percentile", -1.0)) != 0.5
            or float(value.get("peak_percentile", -1.0)) != 0.95
            or tuple(float(item) for item in value.get("scale_clamp", ()))
            != (2.0**-12, 2.0**8)
        ):
            raise ValueError(
                "Metal budgeted phase requires the frozen train-only RGB calibration recipe"
            )
        return dict(value)

    @staticmethod
    def _calibration(model: MetalBudgetedModel) -> AppearanceMetricCalibration:
        if int(model.appearance_calibrated.item()) != 1:
            raise RuntimeError("Metal budgeted appearance calibration is not initialized")
        return AppearanceMetricCalibration(
            model.appearance_scale_rgb,
            model.appearance_peak_rgb,
            float(model.appearance_energy_epsilon.item()),
        )

    def initialization_requests(
        self, config: Mapping[str, Any]
    ) -> tuple[TrainingInitializationRequest, ...]:
        phases = config.get("phases")
        if not isinstance(phases, list) or not phases:
            raise ValueError("Metal budgeted initialization requires training phases")
        recipe = self._calibration_recipe(phases[0])
        return (
            TrainingInitializationRequest(
                "appearance-calibration",
                str(phases[0]["name"]),
                str(recipe["route"]),
                int(recipe["sample_count"]),
                int(recipe["seed"]),
                ("target_f",),
            ),
        )

    def initialize_training_state(
        self,
        model: nn.Module,
        values: Mapping[str, Mapping[str, torch.Tensor]],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(model, MetalBudgetedModel):
            raise TypeError("Metal budgeted method requires MetalBudgetedModel")
        if set(values) != {"appearance-calibration"}:
            raise ValueError("Metal budgeted initialization payload is incomplete")
        tensors = values["appearance-calibration"]
        if set(tensors) != {"target_f"}:
            raise ValueError("Metal budgeted calibration requires only target_f")
        target = tensors["target_f"].to(dtype=torch.float32).reshape(-1, 3)
        if target.shape != (16384, 3):
            raise ValueError("Metal budgeted calibration query count drifted")
        scale_values: list[torch.Tensor] = []
        peak_values: list[torch.Tensor] = []
        for channel in range(3):
            values_channel = target[:, channel]
            finite = values_channel[torch.isfinite(values_channel)]
            positive = finite[finite > 0.0]
            nonnegative = finite[finite >= 0.0]
            if positive.numel() == 0 or nonnegative.numel() == 0:
                raise ValueError("Metal budgeted calibration channel has no finite support")
            scale_values.append(torch.quantile(positive, 0.5))
            peak_values.append(torch.quantile(nonnegative, 0.95))
        scale = torch.clamp(
            torch.stack(scale_values), min=2.0**-12, max=2.0**8
        )
        peak = torch.stack(peak_values)
        epsilon = max(
            64.0 * torch.finfo(torch.float32).eps * float(scale.max()),
            1.0e-6,
        )
        identity_payload = {
            "schema": "ncls.train-only-rgb-calibration-payload@1",
            "metadata": dict(metadata),
            "scale_rgb": [float(value) for value in scale],
            "peak_rgb": [float(value) for value in peak],
            "energy_epsilon": epsilon,
        }
        identity = sha256_json(identity_payload)
        model.set_appearance_calibration(scale, peak, epsilon, identity)
        return {
            "appearance_calibration_identity": identity,
            "appearance_calibration_sample_count": 16384,
        }

    @staticmethod
    def _proposal_weight(phase: Mapping[str, Any]) -> float:
        recipes = phase.get("recipes")
        schedule = recipes.get("proposal_weight") if isinstance(recipes, Mapping) else None
        if not isinstance(schedule, Mapping) or schedule.get("schema") != "linear-nonzero-ramp@1":
            raise ValueError("Metal budgeted proposal schedule is invalid")
        start = float(schedule.get("start", 0.0))
        end = float(schedule.get("end", 0.0))
        steps = int(schedule.get("ramp_steps", 0))
        if start <= 0.0 or end < start or steps < 1:
            raise ValueError("Metal budgeted proposal schedule is invalid")
        progress = min(1.0, max(0, int(phase.get("phase_step", 0))) / float(steps))
        return start + (end - start) * progress

    @staticmethod
    def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.where(valid, value, 0.0)) / torch.clamp(
            valid.to(value.dtype).sum(), min=1.0
        )

    def _appearance_objective_impl(
        self,
        model: MetalBudgetedModel,
        batch: EvaluatorBatch,
        phase: Mapping[str, Any],
        *,
        qat: bool,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        values = batch.tensors
        program = model.compile_program_state(values)
        asset = model.sample_asset(values, program, qat=qat)
        prepared = model.prepare_from_components(program, asset, values["wo"])
        evaluated = model.evaluate_prepared(prepared, values["wo"], values["wi"])
        paired_prepared = model.prepare_paired(values, program, qat=qat)
        paired = model.evaluate_prepared(
            paired_prepared, values["wo"], values["wi"]
        )
        calibration = self._calibration(model)
        metrics = dict(
            appearance_error_metrics(
                evaluated.f,
                values["target_f"],
                values["wi"],
                calibration,
                paired_prediction=paired.f,
                paired_target=values["paired_target_f"],
            )
        )
        scale, _ = calibration.tensors_like(evaluated.f)
        semantic_runtime = torch.mean(
            torch.abs(
                torch.log1p(evaluated.analytic_f / scale)
                - torch.log1p(values["target_f"] / scale)
            )
        )
        core = (
            metrics["appearance/log_rgb"]
            + 0.20 * metrics["appearance/linear_rgb"]
            + 0.25 * metrics["appearance/chroma"]
            + 0.35 * metrics["appearance/peak_rgb"]
            + 0.50 * metrics["appearance/spatial_gradient"]
        )
        direct_auxiliary = evaluated.f.new_zeros(())
        if model.profile.evaluator_mode == "direct":
            direct_auxiliary = torch.mean(
                torch.abs(
                    torch.log1p(evaluated.direct_core_auxiliary / scale)
                    - torch.log1p(evaluated.analytic_f.detach() / scale)
                )
            )
        loss = core + 0.10 * semantic_runtime + 0.05 * direct_auxiliary
        metrics["appearance/core"] = core.detach()
        metrics["appearance/semantic_runtime"] = semantic_runtime.detach()
        metrics["appearance/direct_core_auxiliary"] = direct_auxiliary.detach()
        execution_trace = {**prepared.trace, **evaluated.trace}
        for name, value in execution_trace.items():
            metrics[f"trace/{name}"] = value.detach()
        metrics.update(
            {
                "compiler_responsibility_groups_trace": execution_trace[
                    "compiler_responsibility_groups"
                ].detach(),
                "asset_detail_trace": execution_trace["asset_detail"].detach(),
                "asset_context_trace": execution_trace["asset_context"].detach(),
                "semantic_runtime_trace": execution_trace[
                    "semantic_runtime"
                ].detach(),
                "direction_half_trace": execution_trace["direction_half"].detach(),
                "direction_two_frame_trace": execution_trace[
                    "direction_two_frame"
                ].detach(),
                "positive_rgb_trace": execution_trace["positive_rgb"].detach(),
                "rgb_gate_trace": execution_trace["rgb_gate"].detach(),
                "analytic_lobes_trace": execution_trace[
                    "analytic_lobes"
                ].detach(),
            }
        )
        return loss, metrics

    def _appearance_objective(
        self,
        model: MetalBudgetedModel,
        batch: EvaluatorBatch,
        phase: Mapping[str, Any],
        *,
        qat: bool,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        if not qat:
            return self._appearance_objective_impl(
                model, batch, phase, qat=False
            )
        execution = _AppearanceExecution(model, self)
        runtime_names = metal_budgeted_runtime_parameter_names(model)
        state: dict[str, torch.Tensor] = {}
        quantization_error: list[torch.Tensor] = []
        for name, value in execution.named_parameters():
            model_name = name.removeprefix("model.")
            if model_name in runtime_names:
                state[name] = fake_quantize_fp16_ste(value)
                quantization_error.append(
                    torch.mean(
                        torch.abs(
                            value.detach()
                            - value.detach().to(torch.float16).to(value.dtype)
                        )
                    )
                )
            else:
                state[name] = value
        for name, value in execution.named_buffers():
            state[name] = value
        loss, raw_metrics = torch.func.functional_call(
            execution, state, (batch, phase), strict=True
        )
        metrics = dict(raw_metrics)
        metrics["qat/runtime_weight_mae"] = torch.stack(
            quantization_error
        ).mean().detach()
        return loss, metrics

    @staticmethod
    def _proposal_target(target_f: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        luminance = torch.sum(
            target_f * target_f.new_tensor((0.2126, 0.7152, 0.0722)), dim=-1
        )
        return (
            torch.clamp(luminance, min=0.0)
            * torch.clamp(wi[..., 2], min=0.0)
        ).detach()

    def _proposal_objective_impl(
        self,
        model: MetalBudgetedModel,
        batches: Mapping[str, OnlineTrainingBatch],
        *,
        qat: bool,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        if set(batches) != {"evaluator", "sampler"}:
            raise ValueError("Metal budgeted proposal requires evaluator and sampler routes")
        evaluator_batch = batches["evaluator"]
        sampler_batch = batches["sampler"]
        if not isinstance(evaluator_batch, EvaluatorBatch) or not isinstance(
            sampler_batch, MethodSamplerBatch
        ):
            raise ValueError("Metal budgeted proposal received wrong typed batches")

        sampler_values = sampler_batch.tensors
        sampler_prepared = model.prepare(sampler_values, qat=qat)
        sampled = model.sample_prepared(
            sampler_prepared,
            sampler_values["wo"],
            sampler_values["sample_u"],
        )
        score, valid_fraction = sampler_forward_kl_score(
            sampled.f,
            sampled.wi,
            sampled.forward_pdf,
            sampled.valid,
        )

        evaluator_values = evaluator_batch.tensors
        evaluator_prepared = model.prepare(evaluator_values, qat=qat)
        density = model.pdf_prepared(
            evaluator_prepared,
            evaluator_values["wo"],
            evaluator_values["wi"],
        )
        target = self._proposal_target(
            evaluator_values["target_f"], evaluator_values["wi"]
        )
        target = target / torch.clamp(target.mean(), min=1e-4)
        density_loss = self._masked_mean(
            -target * torch.log(torch.clamp(density.forward, min=1e-12)),
            density.valid,
        )
        independent = model.pdf_prepared(
            sampler_prepared, sampler_values["wo"], sampled.wi
        )
        identity = self._masked_mean(
            torch.abs(sampled.forward_pdf - independent.forward), sampled.valid
        )
        loss = 0.5 * score + 0.5 * density_loss + identity
        return loss, {
            "proposal/forward_kl": score.detach(),
            "proposal/density_nll": density_loss.detach(),
            "proposal/valid_fraction": valid_fraction.detach(),
            "proposal/sample_pdf_identity": identity.detach(),
            "proposal/fallback_weight": sampler_prepared.proposal_state[:, -1, 0]
            .mean()
            .detach(),
        }

    def _proposal_objective(
        self,
        model: MetalBudgetedModel,
        batches: Mapping[str, OnlineTrainingBatch],
        *,
        qat: bool,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        execution = _ProposalExecution(model, self)
        proposal_names = set(self.descriptor.parameter_groups["proposal_sampler"])
        runtime_names = metal_budgeted_runtime_parameter_names(model)
        state: dict[str, torch.Tensor] = {}
        for name, value in execution.named_parameters():
            model_name = name.removeprefix("model.")
            current = (
                fake_quantize_fp16_ste(value)
                if qat and model_name in runtime_names
                else value
            )
            state[name] = (
                current if model_name in proposal_names else current.detach()
            )
        for name, value in execution.named_buffers():
            state[name] = value
        return torch.func.functional_call(
            execution, state, (batches, qat), strict=True
        )

    def training_objective(
        self,
        model: nn.Module,
        batches: Mapping[str, OnlineTrainingBatch],
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        if not isinstance(model, MetalBudgetedModel):
            raise TypeError("Metal budgeted method requires MetalBudgetedModel")
        phase_name = str(phase.get("name"))
        if phase_name not in _PHASES or set(batches) != {"evaluator", "sampler"}:
            raise ValueError("Metal budgeted phases require evaluator and sampler routes")
        evaluator = batches["evaluator"]
        if not isinstance(evaluator, EvaluatorBatch):
            raise ValueError("Metal budgeted evaluator route has the wrong batch type")
        qat = phase_name == "deployment-qat-refine"
        appearance, metrics = self._appearance_objective(
            model, evaluator, phase, qat=qat
        )
        proposal, proposal_metrics = self._proposal_objective(
            model, batches, qat=qat
        )
        proposal_weight = self._proposal_weight(phase)
        total = appearance + proposal_weight * proposal
        result: dict[str, torch.Tensor | float] = {
            **metrics,
            **proposal_metrics,
            "proposal_loss": proposal.detach(),
            "sample_pdf_identity": proposal_metrics[
                "proposal/sample_pdf_identity"
            ],
            "loss/optimization_total": total.detach(),
            "loss/appearance": appearance.detach(),
            "loss/proposal": proposal.detach(),
            "loss/proposal_weight": proposal_weight,
        }
        return total, result

    def export_training_state(self, model: nn.Module) -> Mapping[str, torch.Tensor]:
        if not isinstance(model, MetalBudgetedModel):
            raise TypeError("Metal budgeted method requires MetalBudgetedModel")
        result = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        if set(result) != {field.name for field in self.descriptor.tensor_state_schema}:
            raise ValueError("Metal budgeted checkpoint tensor state is incomplete")
        return result

    def restore_training_state(
        self, model: nn.Module, state: Mapping[str, torch.Tensor]
    ) -> None:
        if not isinstance(model, MetalBudgetedModel):
            raise TypeError("Metal budgeted method requires MetalBudgetedModel")
        current = model.state_dict()
        if set(state) != set(current):
            raise ValueError("Metal budgeted checkpoint tensor state is incomplete")
        restored = {}
        for name, target in current.items():
            value = state[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Metal budgeted checkpoint tensor {name!r} drifted")
            restored[name] = value.to(target.device)
        model.load_state_dict(restored, strict=True)

    def compile_program(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        del checkpoint
        raise RuntimeError("Metal budgeted Slang deployment is produced after pilot selection")

    def compile_asset(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> MaterialPayload:
        del snapshot, checkpoint
        raise RuntimeError("Metal budgeted asset cook is produced after pilot selection")

    def compile_instance(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> InstancePayload:
        del snapshot, checkpoint
        raise RuntimeError("Metal budgeted instance pack is produced after pilot selection")

    def validate_training_config(self, config: Mapping[str, Any]) -> None:
        if config.get("source_adaptation_id") != MetalBudgetedMdlSourceAdapter.adapter_id:
            raise ValueError("Metal budgeted source adapter identity is required")
        context = config.get("model_context")
        if not isinstance(context, Mapping):
            raise ValueError("Metal budgeted model_context is required")
        MetalBudgetedModel.from_context(context)
        expected_correspondence = (
            "metal-budgeted-semantic-hybrid@1"
            if context["profile_id"] == "metal_budgeted_hybrid_v1"
            else "metal-budgeted-semantic-direct-control@1"
        )
        if config.get("correspondence_id") != expected_correspondence:
            raise ValueError("Metal budgeted correspondence/profile identity drifted")
        source = config.get("source")
        if not isinstance(source, Mapping) or source.get("family_id") != "mdl.program@1":
            raise ValueError("Metal budgeted training requires native MDL source semantics")
        phases = config.get("phases")
        if not isinstance(phases, list) or tuple(
            str(item.get("name")) for item in phases
        ) != _PHASES:
            raise ValueError("Metal budgeted training requires the frozen two phases")
        for phase in phases:
            name = str(phase["name"])
            if tuple(phase.get("parameter_groups", ())) != _GROUPS:
                raise ValueError("Metal budgeted phase parameter groups drifted")
            routes = {str(item["name"]): item for item in phase.get("routes", ())}
            if {key: value.get("kind") for key, value in routes.items()} != {
                "evaluator": "reference-evaluator",
                "sampler": "method-sampler",
            }:
                raise ValueError("Metal budgeted phases require evaluator/sampler routes")
            if any(int(route.get("direction_count", 0)) != 1 for route in routes.values()):
                raise ValueError("Metal budgeted online routes require one direction")
            evaluator_options = routes["evaluator"].get("options", {})
            sampler_options = routes["sampler"].get("options", {})
            if (
                evaluator_options.get("direction_proposal")
                != "balanced-four-mode-probe@1"
                or not bool(evaluator_options.get("paired_uv", False))
                or evaluator_options.get("paired_uv_recipe")
                != "one-native-texel-axis-balanced@1"
                or evaluator_options.get("footprint_recipe")
                != "balanced-zero-one-four-texel@1"
                or tuple(evaluator_options.get("spatial_anchor", ()))
                != (0.371, 0.619)
                or int(evaluator_options.get("validation_seed", -1))
                != 2026090402
                or int(evaluator_options.get("source_patch_size", 0)) < 8
                or sampler_options.get("direction_proposal")
                != "uniform-hemisphere-conditioning@1"
                or int(sampler_options.get("validation_seed", -1))
                != 2026090402
            ):
                raise ValueError("Metal budgeted query recipe drifted")
            recipes = phase.get("recipes")
            if not isinstance(recipes, Mapping) or recipes.get("profile_id") != context["profile_id"]:
                raise ValueError("Metal budgeted phase profile identity drifted")
            cook_mode = recipes.get("asset_cook_mode")
            if cook_mode not in {
                "encoder-only@1",
                "bounded-refinement@1",
                "direct-control@1",
            }:
                raise ValueError("Metal budgeted asset cook mode is unsupported")
            if (
                recipes.get("compiler_role") != "pure-typed-compiler@1"
                or recipes.get("optimized_program_state_control") != "report-only@1"
            ):
                raise ValueError("Metal budgeted compiler/control roles drifted")
            calibration_recipe = self._calibration_recipe(phase)
            if int(calibration_recipe["seed"]) != int(config.get("seed", -1)):
                raise ValueError("Metal budgeted calibration seed must equal the train seed")
            self._proposal_weight(phase)
            expected_precision = (
                {"autocast": "bfloat16", "gradient_scaler": False}
                if name == "joint-response-fit"
                else {"autocast": "fp32", "gradient_scaler": False}
            )
            if phase.get("precision") != expected_precision:
                raise ValueError("Metal budgeted phase precision drifted")
            if name == "deployment-qat-refine" and recipes.get("runtime_quantization") != (
                "fp16-weights-state-rgba8-snorm-asset@1"
            ):
                raise ValueError("Metal budgeted QAT phase requires deployed precision")
            if phase.get("transition") is not None:
                raise ValueError("Metal budgeted phases cannot hide asset transitions")

    def configure_phase(self, model: nn.Module, phase: Mapping[str, Any]) -> None:
        if not isinstance(model, MetalBudgetedModel):
            raise TypeError("Metal budgeted method requires MetalBudgetedModel")
        super().configure_phase(model, phase)


METHOD_DEFINITION = MetalBudgetedMethodDefinition()


def _create_source_adapter(
    snapshots: Sequence[SourceSnapshot], device: torch.device
) -> MetalBudgetedMdlSourceAdapter:
    return MetalBudgetedMdlSourceAdapter(tuple(snapshots), device)


METHOD_PLUGIN = MethodPlugin.adapt_definition(
    "metal", METHOD_DEFINITION, source_adapter_factory=_create_source_adapter
)


__all__ = [
    "METHOD_DEFINITION",
    "METHOD_PLUGIN",
    "MetalBudgetedMethodDefinition",
    "metal_budgeted_parameter_groups",
    "metal_budgeted_runtime_parameter_names",
]
