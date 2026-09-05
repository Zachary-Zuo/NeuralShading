from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ncls.core.identity import sha256_json
from ncls.bundle import RGBA8_SNORM_DDS_DTYPE, encode_rgba8_snorm_dds
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
    Method,
    MethodDescriptor,
    SourceAdaptationContract,
    TensorField,
    TrainingInitializationRequest,
)
from ncls.learning.method import Method
from ncls.learning.methods.metal.spatial_runtime import SPATIAL_COMPILED_WORD_COUNT
from ncls.learning.methods.metal.runtime import (
    evaluate_metal_budgeted_cooked_asset,
    metal_budgeted_runtime_parameter_names,
    pack_metal_budgeted_compiled_material,
    pack_metal_budgeted_program,
    quantize_metal_budgeted_program_state,
    quantize_metal_budgeted_runtime_model,
)
from ncls.learning.methods.metal.model import (
    METAL_BUDGETED_REQUIRED_CONTEXT,
    MetalBudgetedModel,
)
from ncls.learning.methods.metal.profile import (
    METAL_BUDGETED_DIRECT_PROFILE_ID,
    METAL_BUDGETED_CENTER_DETAIL_PROFILE_ID,
    METAL_BUDGETED_DUAL_LOCAL_PROFILE_ID,
    METAL_BUDGETED_HYBRID_PROFILE,
    METAL_BUDGETED_HYBRID_PROFILE_ID,
    METAL_BUDGETED_ROLE_DETAIL_PROFILE_ID,
    METAL_BUDGETED_LAYOUT_PATH,
    load_metal_budgeted_layout,
    METAL_SPATIAL_PROFILE,
    METAL_SPATIAL_PROFILE_ID,
    METAL_SPATIAL_SUMMARY_PROFILE_ID,
)
from ncls.learning.objectives import sampler_forward_kl_score
from ncls.learning.methods.metal.data import MetalBudgetedMdlSourceAdapter
from ncls.paths import PROJECT_ROOT


def fake_quantize_fp16_ste(value: torch.Tensor) -> torch.Tensor:
    """Round a floating tensor to deployed FP16 while retaining master gradients."""

    if not value.is_floating_point():
        raise TypeError("Metal runtime fake quantization requires a floating tensor")
    rounded = value.to(torch.float16).to(value.dtype)
    return value + (rounded - value).detach()


_PHASES = ("joint-response-fit", "deployment-qat-refine")
_GROUPS = (
    "asset_encoder",
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
    "filter_random",
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


_PARITY_VIEW = (0.17364818, -0.33682409, 0.92541658)
_PARITY_LIGHTS = (
    (0.0, 0.0, 1.0),
    (0.34202015, 0.16317591, 0.92541658),
    (-0.49240388, 0.41317591, 0.76604444),
    (0.71984631, -0.60402277, 0.34202015),
)


def _module_closure(entry: Path) -> dict[str, bytes]:
    include_pattern = re.compile(rb'^\s*#include\s+"([^"]+)"', re.MULTILINE)
    shader_root = PROJECT_ROOT / "shaders"
    pending = [entry.resolve()]
    result: dict[str, bytes] = {}
    while pending:
        path = pending.pop()
        try:
            relative = path.relative_to(shader_root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"Metal budgeted shader dependency escapes shader root: {path}"
            ) from error
        if relative in result:
            continue
        payload = path.read_bytes()
        result[relative] = payload
        for match in include_pattern.finditer(payload):
            dependency = (path.parent / match.group(1).decode("utf-8")).resolve()
            if dependency.is_file():
                pending.append(dependency)
    return result


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        PROJECT_ROOT / "src/ncls/learning/methods/metal/model.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/asset.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/asset_read.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/spatial_encoder.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/spatial_asset.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/spatial_bundle.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/spatial_cook.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/spatial_runtime.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/spatial_schedule.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/native_uv.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/native_assets.py",
        PROJECT_ROOT / "src/ncls/learning/conditioning_resources.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/compiler.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/evaluator.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/sampler.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/profile.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/asset_cook.py",
        PROJECT_ROOT / "src/ncls/learning/methods/metal/runtime.py",
        PROJECT_ROOT / "src/ncls/learning/appearance_metrics.py",
        PROJECT_ROOT / "src/ncls/learning/source_adapters.py",
        PROJECT_ROOT / "src/ncls/bundle/typed_texture.py",
        PROJECT_ROOT / "shaders/ncls/backends/metal_budgeted/metal_budgeted.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_budgeted/metal_budgeted_common.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_budgeted/metal_budgeted_asset.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_budgeted/metal_spatial_asset.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_budgeted/metal_budgeted_evaluator.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_budgeted/metal_budgeted_sampler.slang",
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
        "P_asset_encoder": sum(
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
        "native-uv-group-spatial-asset",
        ("asset_encoder",),
        ("reference-evaluator",),
        ("asset_detail_trace", "asset_context_trace"),
    ),
    _component(
        "runtime-semantic-137x32x32x24-prepare",
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
        self, model: MetalBudgetedModel, definition: "MetalBudgetedMethod"
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
        self, model: MetalBudgetedModel, definition: "MetalBudgetedMethod"
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


class MetalBudgetedMethod(Method):
    key = "metal"

    def create_source_adapter(self, snapshots, device):
        return _create_source_adapter(snapshots, device)

    _layout = load_metal_budgeted_layout()
    descriptor = MethodDescriptor(
        "metal-budgeted-neural-material",
        2,
        "Metal 原始语义与多 UV 空间 neural material",
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
        "ncls.metal-spatial-method@1",
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
            "runtime_class": "native-uv-spatial-neural-material",
            "profile_id": METAL_SPATIAL_PROFILE_ID,
            "planned_summary_control_profile_id": METAL_SPATIAL_SUMMARY_PROFILE_ID,
            "C_prepare_macs": METAL_SPATIAL_PROFILE.runtime_prepare_dense_macs,
            "C_eval_macs": METAL_SPATIAL_PROFILE.evaluate_dense_macs,
            "B_prepared": METAL_SPATIAL_PROFILE.prepared_state_bytes,
            "maximum_texture_reads": METAL_SPATIAL_PROFILE.maximum_texture_reads,
            "proposal_components": 3,
            **_PARAMETER_ACCOUNTING,
            "observed_quality_gate": False,
            "observed_latency_gate": False,
        },
        metal_budgeted_parameter_groups(
            MetalBudgetedModel.from_context(METAL_BUDGETED_REQUIRED_CONTEXT)
        ),
        _COMPONENTS,
        training_resource_requirements={"reference-evaluator": ("metal_spatial",), "method-sampler": ("metal_spatial",)},
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
            or int(value.get("sample_count", 0)) < 1
            or int(value.get("seed", -1)) < 0
            or float(value.get("scale_percentile", -1.0)) != 0.5
            or float(value.get("peak_percentile", -1.0)) != 0.95
            or tuple(float(item) for item in value.get("scale_clamp", ()))
            != (2.0**-12, 2.0**8)
        ):
            raise ValueError(
                "Metal phase requires a valid train-only RGB calibration recipe"
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
        requests = metadata.get("requests", ())
        requested = next((int(item["sample_count"]) for item in requests
                          if item["name"] == "appearance-calibration"), target.shape[0])
        if target.shape != (requested, 3) or requested < 1:
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
            "appearance_calibration_sample_count": requested,
        }

    @staticmethod
    def _proposal_weight(phase: Mapping[str, Any]) -> float:
        recipes = phase.get("recipes")
        schedule = recipes.get("proposal_weight") if isinstance(recipes, Mapping) else None
        if isinstance(schedule, Mapping) and schedule == {"schema": "disabled@1"}:
            return 0.0
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
        resource_args = {}
        if model.profile.is_spatial:
            resource_args = {"resources": batch.conditioning.resources,
                             "binding": batch.conditioning.bindings["metal_spatial"],
                             "encoded": model.asset.encode_resources(batch.conditioning.resources)}
        asset = model.sample_asset(values, program, qat=qat, **resource_args)
        prepared = model.prepare_from_components(program, asset, values["wo"], qat=qat)
        evaluated = model.evaluate_prepared(prepared, values["wo"], values["wi"])
        paired_prepared = model.prepare_paired(values, program, qat=qat, **resource_args)
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
        sampler_prepared = model.prepare(sampler_values, qat=qat, resources=sampler_batch.conditioning.resources,
                                         binding=sampler_batch.conditioning.bindings.get("metal_spatial"))
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
        evaluator_prepared = model.prepare(evaluator_values, qat=qat, resources=evaluator_batch.conditioning.resources,
                                           binding=evaluator_batch.conditioning.bindings.get("metal_spatial"))
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
        proposal_weight = self._proposal_weight(phase)
        expected_routes = {"evaluator", "sampler"} if proposal_weight > 0 else {"evaluator"}
        if phase_name not in _PHASES or set(batches) != expected_routes:
            raise ValueError("Metal phase routes disagree with its evaluator/proposal objectives")
        evaluator = batches["evaluator"]
        if not isinstance(evaluator, EvaluatorBatch):
            raise ValueError("Metal budgeted evaluator route has the wrong batch type")
        qat = phase_name == "deployment-qat-refine" or phase.get("recipes", {}).get("runtime_quantization") == "fp16-weights-state-rgba8-snorm-asset@1"
        appearance, metrics = self._appearance_objective(
            model, evaluator, phase, qat=qat
        )
        if proposal_weight > 0:
            proposal, proposal_metrics = self._proposal_objective(model, batches, qat=qat)
        else:
            proposal = appearance.new_zeros(())
            proposal_metrics = {"proposal/sample_pdf_identity": proposal}
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
        fields = {field.name: field for field in self.descriptor.tensor_state_schema}
        symbols: dict[str, int] = {}
        for name, tensor in result.items():
            field = fields[name]
            if field.dtype != str(tensor.dtype).removeprefix("torch."):
                raise ValueError(
                    f"Metal budgeted checkpoint tensor {name!r} dtype drifted"
                )
            if len(field.shape) != tensor.ndim:
                raise ValueError(
                    f"Metal budgeted checkpoint tensor {name!r} rank drifted"
                )
            for expected, actual in zip(field.shape, tensor.shape, strict=True):
                if isinstance(expected, int) and expected != actual:
                    raise ValueError(
                        f"Metal budgeted checkpoint tensor {name!r} shape drifted"
                    )
                if isinstance(expected, str):
                    previous = symbols.setdefault(expected, int(actual))
                    if previous != actual:
                        raise ValueError(
                            "Metal budgeted checkpoint symbolic dimension "
                            f"{expected!r} drifted"
                        )
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

    def _deployment(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.descriptor.adaptation_contract(snapshot)
        cache_key = (snapshot.snapshot_id, id(checkpoint))
        cached = getattr(self, "_deployment_cache", None)
        if isinstance(cached, tuple) and cached[0] == cache_key:
            return cached[1]
        state = checkpoint.get("model_state")
        training_config = checkpoint.get("training_config")
        if (
            not isinstance(state, Mapping)
            or not isinstance(training_config, Mapping)
        ):
            raise ValueError(
                "Metal deployment requires checkpoint model state and model context"
            )
        context = training_config.get("model_context")
        if not isinstance(context, Mapping):
            raise ValueError("Metal budgeted deployment checkpoint has no model_context")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.create_trainable(context).to(device)
        if not isinstance(model, MetalBudgetedModel):
            raise TypeError("Metal budgeted deployment requires MetalBudgetedModel")
        self.restore_training_state(model, state)
        quantize_metal_budgeted_runtime_model(model)
        adapter = MetalBudgetedMdlSourceAdapter((snapshot,), device)
        try:
            tensors = adapter.compiler_tensors_for_source(0, device=device)
            from ncls.learning.methods.metal.spatial_cook import compile_spatial_asset
            asset_index, slots, groups = adapter.spatial_contract_for_source(0)
            asset = compile_spatial_asset(model, adapter.native_assets(), asset_index, slots, groups)
            with torch.no_grad():
                program_state = model.compile_program_state(tensors)
        finally:
            adapter.close()
        result = {
            "model": model,
            "tensors": tensors,
            "asset": asset,
            "program_state": program_state,
        }
        self._deployment_cache = (cache_key, result)
        return result

    def compile_program(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        state = checkpoint.get("model_state")
        training_config = checkpoint.get("training_config")
        if not isinstance(state, Mapping) or not isinstance(training_config, Mapping):
            raise ValueError(
                "Metal budgeted runtime compilation requires checkpoint model_state"
            )
        context = training_config.get("model_context")
        if not isinstance(context, Mapping):
            raise ValueError("Metal budgeted runtime compilation requires model_context")
        model = self.create_trainable(context)
        if not isinstance(model, MetalBudgetedModel):
            raise TypeError("Metal budgeted runtime requires MetalBudgetedModel")
        self.restore_training_state(model, state)
        quantize_metal_budgeted_runtime_model(model)
        packed = pack_metal_budgeted_program(model)
        module = "ncls/backends/metal_budgeted/metal_budgeted.slang"
        closure = _module_closure(PROJECT_ROOT / "shaders" / module)
        capabilities = int(
            BackendCapability.PREPARE
            | BackendCapability.EVALUATE
            | BackendCapability.SAMPLE
            | BackendCapability.PDF
            | BackendCapability.ANISOTROPIC_FRAME
            | BackendCapability.REVERSE_PDF
        )
        return RuntimePayload(
            module,
            closure,
            {"shared-weights": packed.payload},
            {
                "shared-weights": {
                    "kind": "structured-buffer",
                    "dtype": "packed-float16x2-uint32@1",
                    "shape": [len(packed.payload) // 4],
                    "stride": 4,
                    "alignment": 16,
                    "usage": "gNclsRuntimeWeights",
                }
            },
            capabilities,
            packed.defines,
        )

    def compile_asset(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> MaterialPayload:
        asset = self._deployment(snapshot, checkpoint)["asset"]
        from ncls.learning.methods.metal.spatial_runtime import spatial_material_payload
        return spatial_material_payload(snapshot.snapshot_id, asset)

    def compile_instance(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> InstancePayload:
        deployment = self._deployment(snapshot, checkpoint)
        compiled = pack_metal_budgeted_compiled_material(
            deployment["program_state"], deployment["asset"]
        )
        return InstancePayload(
            {"compiled_material_index": 0},
            {"compiled-material": compiled},
            {
                "compiled-material": {
                    "kind": "structured-buffer",
                    "dtype": "ncls-metal-spatial-compiled-material@1",
                    "shape": [SPATIAL_COMPILED_WORD_COUNT],
                    "stride": 4,
                    "alignment": 16,
                    "usage": "gNclsCompiledMaterials",
                }
            },
        )

    def package_validation(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        deployment = self._deployment(snapshot, checkpoint)
        with torch.no_grad():
            expected = evaluate_metal_budgeted_cooked_asset(
                deployment["model"],
                deployment["asset"],
                deployment["tensors"],
                uv=(0.0, 0.0),
                mip_level=0.0,
                filter_random=0.0,
                wo=_PARITY_VIEW,
                wi=_PARITY_LIGHTS,
            )
        packed = pack_metal_budgeted_program(deployment["model"])
        asset = deployment["asset"]
        from ncls.learning.methods.metal.runtime import prepare_metal_budgeted_cooked_asset

        model = deployment["model"]
        device = next(model.parameters()).device
        prepared = prepare_metal_budgeted_cooked_asset(
            model, asset, deployment["tensors"], uv=(0.0, 0.0), mip_level=0.0,
            filter_random=0.0, wo=_PARITY_VIEW,
        )
        random_values = []
        for seed in range(1, len(_PARITY_LIGHTS) + 1):
            first = (seed * 1664525 + 1013904223) & 0xffffffff
            second = (first * 1664525 + 1013904223) & 0xffffffff
            random_values.append([(first >> 8) * 2**-24, (second >> 8) * 2**-24])
        with torch.no_grad():
            samples = [model.sample_prepared(prepared,
                torch.tensor([_PARITY_VIEW], device=device),
                torch.tensor([value], device=device)) for value in random_values]
        if not all(bool(sample.valid.all()) for sample in samples):
            raise ValueError("deployment sampler witness contains invalid samples")
        return {
            "status": "gpu-parity-required",
            "scope": "prepare-evaluate-sample-pdf",
            "runtime_cost": {
                "prepare_dense_macs": deployment["model"].profile.runtime_prepare_dense_macs,
                "evaluate_dense_macs": deployment["model"].profile.evaluate_dense_macs,
                "prepared_state_bytes": model.profile.prepared_state_bytes,
                "asset_reads": asset.texture_reads,
            },
            "sampling": {
                "oracle": "metal-budgeted-fp16-snorm8-python@1",
                "sample_u": random_values,
                "expected_wi": [sample.wi[0, 0].cpu().tolist() for sample in samples],
                "expected_pdf": [[sample.forward_pdf[0].item(), sample.reverse_pdf[0].item()] for sample in samples],
                "expected_weight": [sample.weight[0, 0].cpu().tolist() for sample in samples],
                "relative_tolerance": 2e-3, "absolute_tolerance": 2e-5,
            },
            "parity": {
                "oracle": "metal-budgeted-fp16-snorm8-python@1",
                "uv": [0.0, 0.0],
                "mip_level": 0.0,
                "view": list(_PARITY_VIEW),
                "lights": [list(value) for value in _PARITY_LIGHTS],
                "expected_f": expected.tolist(),
                "relative_tolerance": 3e-2,
                "absolute_tolerance": 5e-4,
            },
            "storage": {
                "B_shared": len(packed.payload),
                "B_asset": asset.latent_bytes,
                "B_instance": 4 * SPATIAL_COMPILED_WORD_COUNT,
            },
        }

    def validate_training_config(self, config: Mapping[str, Any]) -> None:
        for phase in config["phases"]:
            self._calibration_recipe(phase)
            proposal_weight = self._proposal_weight(phase)
            routes = {route["name"]: route for route in phase["routes"]}
            if set(routes) != ({"evaluator", "sampler"} if proposal_weight > 0 else {"evaluator"}):
                raise ValueError("Metal route 必须与 evaluator/proposal objective 一致")
            if not routes["evaluator"]["options"].get("paired_uv", False):
                raise ValueError("当前 Metal 空间差分 objective 需要 paired_uv")

    def configure_phase(self, model: nn.Module, phase: Mapping[str, Any]) -> None:
        if not isinstance(model, MetalBudgetedModel):
            raise TypeError("Metal budgeted method requires MetalBudgetedModel")
        super().configure_phase(model, phase)


def _create_source_adapter(
    snapshots: Sequence[SourceSnapshot], device: torch.device
) -> MetalBudgetedMdlSourceAdapter:
    return MetalBudgetedMdlSourceAdapter(tuple(snapshots), device)


METHOD = MetalBudgetedMethod()


__all__ = [
    "METHOD",

    "MetalBudgetedMethod",
    "metal_budgeted_parameter_groups",
    "metal_budgeted_runtime_parameter_names",
]
